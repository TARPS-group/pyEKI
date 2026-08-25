"""Base classes for structured linear operators.

An operator represents a matrix implicitly, by how it acts on vectors, so
that known structure is exploited instead of storing or factorizing a dense
array. This module defines the operator interface and the machinery for
writing new operators; concrete operators live in :mod:`.leaves` and
:mod:`.composite`. The normative specification of everything promised here
is the "Linear operator contract" page of the documentation.

Class hierarchy
---------------
Each level is a mathematical claim about the map, and adds the operations
that the claim makes well defined.

``LinOp``
    Any linear map, possibly rectangular.
    Provides ``matvec``, ``rmatvec``, ``matmat``, ``rmatmat``, ``T``,
    ``to_dense``.
``SquareLinOp``
    A square map.
    Adds ``solve``, ``solve_mat``, ``logdet``, ``diag``.
``PSDLinOp``
    A symmetric positive semi-definite map.
    Adds ``factor``, ``whiten``, ``whiten_mat``.

Array shapes
------------
Every method takes leading batch axes; the operand's core shape is trailing.

==============  =====================  =====================================
method          core operand           signature
==============  =====================  =====================================
``matvec``      vector ``(n_in,)``     ``(..., n_in) -> (..., n_out)``
``rmatvec``     vector ``(n_out,)``    ``(..., n_out) -> (..., n_in)``
``matmat``      matrix ``(n_in, k)``   ``(..., n_in, k) -> (..., n_out, k)``
``rmatmat``     matrix ``(n_out, k)``  ``(..., n_out, k) -> (..., n_in, k)``
``solve``       vector ``(n,)``        ``(..., n) -> (..., n)``
``solve_mat``   matrix ``(n, k)``      ``(..., n, k) -> (..., n, k)``
``whiten``      vector ``(n,)``        ``(..., n) -> (..., n)``
``whiten_mat``  matrix ``(n, k)``      ``(..., n, k) -> (..., n, k)``
==============  =====================  =====================================

The ``k`` in the matrix methods is part of the core shape, never a batch
axis. Use the vector method for a batch of vectors and the matrix method
for a single (possibly batched) matrix operand; neither infers which you
meant from the number of dimensions.

Defining a new operator
-----------------------
Subclass the appropriate level, decorate with :func:`linop`, and implement
the hooks: ``shape``, ``_matvec``, ``_rmatvec`` (``PSDLinOp`` provides it)
and ``_to_dense`` are required, and ``_solve``, ``_logdet``, ``_diag``,
``_factor`` and ``_whiten`` are optional. The public methods are defined
once, here: they check the capability gate, validate the operand, and
dispatch to the hook, which receives the operand unchanged — batch axes
included. Use :func:`dense_matvec` and :func:`tri_solve` for the array work
so the shape convention is honoured, mark non-array fields with
:func:`static_field`, and validate every new operator with
:func:`pyeki.linalg.testing.check_operator`.

Unsupported operations
----------------------
Not every operator can do everything cheaply. Calling an optional operation
the operator does not implement raises :class:`UnsupportedOpError` rather
than falling back to dense linear algebra. Query support with
``op.supports(name)`` or ``op.capabilities()``, and use :func:`densify`
when a dense fallback is genuinely wanted.

Notes
-----
Design rationale for the interface lives in the "Linear operator contract"
documentation page rather than here; this module implements it.
"""
from __future__ import annotations

import abc
import dataclasses
import typing
from contextlib import contextmanager
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "LinOp",
    "SquareLinOp",
    "PSDLinOp",
    "UnsupportedOpError",
    "densify",
    "linop",
    "static_field",
    "dense_matvec",
    "tri_solve",
    "set_debug_checks",
    "debug_checks",
    "value_check",
]


class UnsupportedOpError(NotImplementedError):
    """Raised when an operator has no cheap implementation of an operation.

    The message names the operator type, the operation, and the operations
    the operator does support. For an explicit dense fallback, use
    ``densify(op)``.

    Parameters
    ----------
    name
        The operation that was requested.
    operator_type
        Name of the operator's class.
    capabilities
        Names of the optional operations the operator does support.

    Notes
    -----
    The constructor takes only strings and a tuple of strings, so the
    exception pickles and rebuilds from ``args`` — it must survive the
    worker-process boundary that parallel forward-model evaluation crosses.
    """

    def __init__(
        self, name: str, operator_type: str, capabilities: tuple[str, ...] = ()
    ) -> None:
        capabilities = tuple(capabilities)
        super().__init__(name, operator_type, capabilities)
        self.name = name
        self.operator_type = operator_type
        self.capabilities = capabilities

    def __str__(self) -> str:
        have = ", ".join(sorted(self.capabilities)) or "none"
        return (
            f"{self.operator_type} has no cheap `{self.name}`. It supports: {have}. "
            f"Use densify(op) for an explicit dense fallback."
        )


# ---------------------------------------------------------------------------
# debug switch for value-level checks
# ---------------------------------------------------------------------------

_debug_checks_enabled = False


def set_debug_checks(enabled: bool) -> bool:
    """Enable or disable value-level validation; return the previous setting.

    When enabled, constructors and ``from_matrix``-style classmethods assert
    value preconditions — positivity of diagonal entries, positive
    definiteness of factorized matrices — on concrete inputs. The checks are
    always skipped on tracers, so enabling them does not affect ``jit``-ed
    code. When disabled (the default), violated preconditions produce ``nan``
    or ``inf`` downstream rather than an exception.

    Parameters
    ----------
    enabled
        The new process-global setting.

    Returns
    -------
    bool
        The previous setting, so callers can restore it.
    """
    global _debug_checks_enabled
    previous = _debug_checks_enabled
    _debug_checks_enabled = bool(enabled)
    return previous


@contextmanager
def debug_checks(enabled: bool = True):
    """Context manager form of :func:`set_debug_checks`.

    Sets the flag on entry and restores the previous setting on exit.
    """
    previous = set_debug_checks(enabled)
    try:
        yield
    finally:
        set_debug_checks(previous)


def value_check(x, predicate, message: str) -> None:
    """Assert a value-level precondition when debug checks are enabled.

    The helper operator authors use inside ``__post_init__`` and
    ``from_matrix``-style classmethods for preconditions that are values
    rather than shapes — positivity, finiteness, definiteness.

    Skipped when debug checks are off, when ``x`` is not array-like (a
    placeholder used during pytree unflattening), or when ``x`` is a tracer.
    """
    if not _debug_checks_enabled:
        return
    if getattr(x, "ndim", None) is None or isinstance(x, jax.core.Tracer):
        return
    if not bool(predicate(x)):
        raise ValueError(message)


# ---------------------------------------------------------------------------
# dataclass / pytree plumbing
# ---------------------------------------------------------------------------


def static_field(**kwargs):
    """Declare a dataclass field as pytree *metadata* rather than a child.

    Static metadata must be hashable and cheap to compare — ints, bools,
    strings, tuples of those — never arrays.
    """
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["static"] = True
    return field(metadata=metadata, **kwargs)


def _is_data_annotation(ann) -> bool:
    """Return True if this annotation is allowed to be pytree data."""
    if ann is Array:
        return True
    if isinstance(ann, type) and issubclass(ann, LinOp):
        return True
    if typing.get_origin(ann) is tuple:
        args = [a for a in typing.get_args(ann) if a is not Ellipsis]
        return bool(args) and all(_is_data_annotation(a) for a in args)
    return False


def linop(cls: type) -> type:
    """Class decorator: make ``cls`` a frozen dataclass and a JAX pytree.

    Fields are classified by an allowlist: a field is a pytree *child*
    (data) if and only if its annotation is ``jax.Array``, a :class:`LinOp`
    subtype, or a tuple of those. Every other field must be declared with
    :func:`static_field`.

    Raises
    ------
    TypeError
        If a field is neither allowlisted data nor marked static, or if a
        field annotation cannot be resolved at class definition time.

    Notes
    -----
    - Operators compare by identity (``eq=False``): dataclass equality would
      compare arrays elementwise and raise on the ambiguous truth value.
      They hash by identity, which makes them usable as dictionary keys but
      never as ``static_argnums`` — every call would retrace silently.
    - No ``__repr__`` is generated (``repr=False``), so
      :meth:`LinOp.__repr__` applies; a generated one would print whole
      arrays into tracebacks and test identifiers.
    - Annotations are resolved with ``typing.get_type_hints``, so they must
      name types importable in the defining module at class definition time:
      no ``TYPE_CHECKING``-only names and no self- or forward-references.
    """
    cls = dataclass(frozen=True, eq=False, repr=False)(cls)
    try:
        hints = typing.get_type_hints(cls)
    except NameError as e:
        raise TypeError(
            f"{cls.__name__}: field annotations must resolve at class definition "
            f"time ({e}). Avoid TYPE_CHECKING-only names and forward references."
        ) from e

    data_fields, meta_fields = [], []
    for f in dataclasses.fields(cls):
        if f.metadata.get("static", False):
            meta_fields.append(f.name)
        elif _is_data_annotation(hints.get(f.name)):
            data_fields.append(f.name)
        else:
            raise TypeError(
                f"{cls.__name__}.{f.name}: only `Array`, `LinOp` subtypes, and "
                f"tuples of those may be pytree data. Mark this field with "
                f"static_field(), or store it as an array."
            )

    jax.tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )
    return cls


# ---------------------------------------------------------------------------
# array helpers honouring the batch contract
# ---------------------------------------------------------------------------


def dense_matvec(M: Array, x: Array) -> Array:
    """Apply an unbatched dense matrix to ``x``, contracting its trailing axis.

    ``(m, n) x (..., n) -> (..., m)``. Use this rather than ``M @ x`` when
    implementing ``_matvec``: for operands with two or more dimensions the
    ``@`` operator contracts the second-to-last axis, which silently returns
    a wrong answer when ``M`` is square.
    """
    return jnp.einsum("ij,...j->...i", M, x)


def tri_solve(L: Array, x: Array, *, lower: bool, trans: int = 0) -> Array:
    """Solve a triangular system, contracting the trailing axis of ``x``.

    ``L`` is an unbatched square triangular matrix; ``x`` may carry any
    number of leading batch axes. ``trans=1`` solves against ``L.T``.
    """
    flat = x.reshape(-1, x.shape[-1]).swapaxes(-1, -2)  # (n, m)
    out = jax.scipy.linalg.solve_triangular(L, flat, lower=lower, trans=trans)
    return out.swapaxes(-1, -2).reshape(x.shape)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _check_vec(op: LinOp, method: str, x, size: int) -> Array:
    """Validate a vector operand: rank at least 1, trailing axis ``size``."""
    x = jnp.asarray(x)
    if x.ndim < 1 or x.shape[-1] != size:
        raise ValueError(
            f"{op!r}.{method}: expected core shape (..., {size}), "
            f"got operand shape {x.shape}"
        )
    return x


def _check_mat(op: LinOp, method: str, X, size: int) -> Array:
    """Validate a matrix operand: rank at least 2, axis ``-2`` of ``size``."""
    X = jnp.asarray(X)
    if X.ndim < 2 or X.shape[-2] != size:
        raise ValueError(
            f"{op!r}.{method}: expected core shape (..., {size}, k), "
            f"got operand shape {X.shape}"
        )
    return X


def _check_rank_floor(cls_name: str, field_name: str, value, core_ndim: int) -> None:
    """Reject an array field whose rank is below its core rank.

    Extra leading axes are accepted — such an operator is a vmapped family,
    reconstructed at ``vmap`` exit boundaries. Fields without ``ndim``
    (placeholders used by JAX during unflattening) pass untouched.
    """
    ndim = getattr(value, "ndim", None)
    if ndim is not None and ndim < core_ndim:
        raise ValueError(
            f"{cls_name}.{field_name}: expected an array of rank at least "
            f"{core_ndim}, got rank {ndim}"
        )


def _as_scalar(op: LinOp, c) -> Array:
    """Convert a scaling factor to a 0-d array, rejecting non-scalars."""
    arr = jnp.asarray(c)
    if arr.ndim != 0:
        raise TypeError(
            f"only a true scalar can scale {op!r}, got an array of shape "
            f"{arr.shape}. For per-coordinate scaling of a PSD operator, "
            f"use diag_congruence()."
        )
    return arr


def _scale(op: LinOp, c: Array) -> LinOp:
    """Build the scaled composite matching ``op``'s hierarchy level."""
    from .composite import PSDScaled, Scaled, SquareScaled

    if isinstance(op, Scaled):  # fold nested scalings into a single wrapper
        return _scale(op.op, op.c * c)
    if isinstance(op, PSDLinOp):
        return PSDScaled(op, c)
    if isinstance(op, SquareLinOp):
        return SquareScaled(op, c)
    return Scaled(op, c)


# ---------------------------------------------------------------------------
# level 1 -- LinOp
# ---------------------------------------------------------------------------


class LinOp(abc.ABC):
    """A linear map, possibly rectangular.

    The base of the operator hierarchy. Concrete subclasses implement the
    hooks ``shape``, ``_matvec``, ``_rmatvec`` and ``_to_dense``; the public
    methods defined here validate operands and dispatch to the hooks. See
    the module docstring for the shape convention and for how to define a
    new operator.
    """

    #: NumPy consults this on binary operations: ``None`` makes NumPy defer
    #: to this class's reflected dunders, so ``np_array * op`` raises the
    #: guided error instead of silently building an object array.
    __array_ufunc__ = None

    # -- required hooks ------------------------------------------------------
    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        """Shape as ``(n_out, n_in)``, a concrete tuple of Python ints.

        A property computed from static information, never a stored field,
        so it stays concrete under ``jit`` and is usable in shape
        expressions.
        """

    @abc.abstractmethod
    def _matvec(self, x: Array) -> Array:
        """Hook: apply the operator to a validated operand, batch included."""

    @abc.abstractmethod
    def _rmatvec(self, x: Array) -> Array:
        """Hook: apply the transpose to a validated operand, batch included."""

    @abc.abstractmethod
    def _to_dense(self) -> Array:
        """Hook: the dense ``(n_out, n_in)`` array.

        Built from stored arrays and static metadata by a code path that
        never routes through ``matvec``; the conformance suite enforces
        this mechanically.
        """

    # -- derived hooks (override when a faster direct form exists) ------------
    def _matmat(self, X: Array) -> Array:
        return self._matvec(X.swapaxes(-1, -2)).swapaxes(-1, -2)

    def _rmatmat(self, X: Array) -> Array:
        return self._rmatvec(X.swapaxes(-1, -2)).swapaxes(-1, -2)

    # -- public surface --------------------------------------------------------
    def matvec(self, x) -> Array:
        """Apply the operator to ``x``, contracting its trailing axis."""
        return self._matvec(_check_vec(self, "matvec", x, self.shape[1]))

    def rmatvec(self, x) -> Array:
        """Apply the transpose to ``x``, contracting its trailing axis."""
        return self._rmatvec(_check_vec(self, "rmatvec", x, self.shape[0]))

    def matmat(self, X) -> Array:
        """Apply the operator to a matrix ``X`` of core shape ``(n_in, k)``.

        Equal to ``matvec`` applied over the columns of ``X``.
        """
        return self._matmat(_check_mat(self, "matmat", X, self.shape[1]))

    def rmatmat(self, X) -> Array:
        """Apply the transpose to a matrix ``X`` of core shape ``(n_out, k)``."""
        return self._rmatmat(_check_mat(self, "rmatmat", X, self.shape[0]))

    def to_dense(self) -> Array:
        """Return the operator as a dense ``(n_out, n_in)`` array.

        The reference and debugging path; it has no size guard. Use
        :func:`densify` for a guarded fallback that returns an operator.
        """
        return self._to_dense()

    @property
    def T(self) -> LinOp:  # noqa: N802 - mirrors the NumPy attribute
        """The transpose, as an operator.

        The default wraps the operator in a :class:`~.composite.Transposed`
        view, a plain ``LinOp``. Operators whose transpose supports more
        override this to return a structured result.
        """
        from .composite import Transposed

        return Transposed(self)

    # -- capability introspection ----------------------------------------------
    def supports(self, name: str) -> bool:
        """Return True exactly when calling ``name`` on this operator succeeds.

        ``name`` must be one of the twelve operation names; anything else
        raises ``ValueError``, so a typo cannot silently steer callers onto
        a fallback branch. Names below the operator's level return False.

        Notes
        -----
        This is an instance method because support can depend on an
        operator's contents: a block-diagonal operator can solve only if all
        of its blocks can.
        """
        if name not in _KNOWN_OPS:
            known = ", ".join(sorted(_KNOWN_OPS))
            raise ValueError(f"unknown operation {name!r}; known operations: {known}")
        if name in _ALWAYS_OPS:
            return True
        hook = _PRIMITIVE_HOOKS.get(name)
        if hook is not None:
            return getattr(type(self), hook, None) is not None
        impl = getattr(type(self), "_" + name, None)
        if impl is not None and impl is not _DERIVED_DEFAULTS[name]:
            return True
        return self.supports(_DERIVED_DEPS[name])

    def capabilities(self) -> frozenset[str]:
        """Return the supported operations among those not always available."""
        return frozenset(n for n in _OPTIONAL_OPS if self.supports(n))

    def _require(self, name: str) -> None:
        """Raise :class:`UnsupportedOpError` unless ``name`` is supported."""
        if not self.supports(name):
            raise UnsupportedOpError(
                name, type(self).__name__, tuple(sorted(self.capabilities()))
            )

    def _has_shape_info(self) -> bool:
        """Return True when every stored array, recursively, reports a shape.

        False while JAX unflattens with placeholder leaves; cross-field
        structural checks in constructors must be skipped then.
        """
        for f in dataclasses.fields(self):
            if f.metadata.get("static", False):
                continue
            value = getattr(self, f.name)
            items = value if isinstance(value, tuple) else (value,)
            for item in items:
                if isinstance(item, LinOp):
                    if not item._has_shape_info():
                        return False
                elif getattr(item, "ndim", None) is None:
                    return False
        return True

    # -- operator arithmetic ------------------------------------------------------
    def __matmul__(self, other):
        """Compose with another operator; arrays get a guided error."""
        if isinstance(other, LinOp):
            from .composite import product

            return product(self, other)
        raise TypeError(
            f"{self!r} @ <array> is not supported: `@` would contract axis -2, "
            f"which is silently wrong for leading-batch vectors. Use matvec(x) "
            f"for a batch of vectors or matmat(X) for a matrix operand."
        )

    def __rmatmul__(self, other):
        raise TypeError(
            f"<array> @ {self!r} is not supported. Use rmatvec(x) or rmatmat(X), "
            f"or transpose with `op.T`."
        )

    def __mul__(self, c):
        """Scale by a scalar, returning a level-preserving scaled operator."""
        if isinstance(c, LinOp):
            raise TypeError(
                "op1 * op2 is not defined; use op1 @ op2 for composition."
            )
        return _scale(self, _as_scalar(self, c))

    __rmul__ = __mul__

    def __truediv__(self, c):
        """Scale by the reciprocal of a scalar."""
        if isinstance(c, LinOp):
            raise TypeError("op1 / op2 is not defined.")
        return _scale(self, 1.0 / _as_scalar(self, c))

    def __repr__(self) -> str:
        """Return the type name and shape, as ``Dense(4, 6)``.

        Deliberately omits field values, which are usually arrays too large
        to be worth printing.
        """
        return f"{type(self).__name__}{self.shape}"


# ---------------------------------------------------------------------------
# level 2 -- SquareLinOp
# ---------------------------------------------------------------------------


class SquareLinOp(LinOp):
    """A square linear map, for which an inverse and determinant are defined."""

    @property
    def n(self) -> int:
        """Side length of the operator."""
        return self.shape[0]

    # -- derived hook -----------------------------------------------------------
    def _solve_mat(self, B: Array) -> Array:
        return self._solve(B.swapaxes(-1, -2)).swapaxes(-1, -2)

    # -- public surface -----------------------------------------------------------
    def solve(self, b) -> Array:
        """Solve ``A x = b`` exactly, contracting the trailing axis of ``b``.

        Requires the operator to be nonsingular.
        """
        self._require("solve")
        return self._solve(_check_vec(self, "solve", b, self.n))

    def solve_mat(self, B) -> Array:
        """Solve ``A X = B`` for a matrix ``B`` of core shape ``(n, k)``."""
        self._require("solve_mat")
        return self._solve_mat(_check_mat(self, "solve_mat", B, self.n))

    def logdet(self) -> Array:
        """Return ``log |det A|`` as a real scalar array.

        Always a 0-d JAX array, never a Python float: converting would fail
        on a tracer under ``jit``. The absolute value matters only for
        non-PSD operators and matches ``slogdet``'s magnitude convention.
        """
        self._require("logdet")
        return self._logdet()

    def diag(self) -> Array:
        """Return the diagonal as a vector of length ``n``."""
        self._require("diag")
        return self._diag()


# ---------------------------------------------------------------------------
# level 3 -- PSDLinOp
# ---------------------------------------------------------------------------


class PSDLinOp(SquareLinOp):
    """A symmetric positive semi-definite linear map.

    Self-adjoint, so ``_rmatvec`` is provided (it delegates to ``_matvec``)
    and ``T`` returns the operator itself.
    """

    def _rmatvec(self, x: Array) -> Array:
        # A delegating method, not a class-body alias: an alias of the
        # abstract _matvec would keep every subclass abstract and would
        # never see subclass overrides.
        return self._matvec(x)

    @property
    def T(self) -> PSDLinOp:  # noqa: N802 - mirrors the NumPy attribute
        """The transpose: the operator itself, since it is self-adjoint."""
        return self

    # -- derived hook -----------------------------------------------------------
    def _whiten_mat(self, X: Array) -> Array:
        return self._whiten(X.swapaxes(-1, -2)).swapaxes(-1, -2)

    # -- public surface -----------------------------------------------------------
    def factor(self) -> LinOp:
        """Return an operator ``L`` of shape ``(n, k)`` with ``L @ L.T == self``.

        The sampling interface: ``L.matvec(eps)`` for standard normal ``eps``
        of length ``k`` has covariance equal to this operator, and
        ``L.rmatvec`` applies ``L.T``. No triangularity, squareness, or
        orientation is promised: ``k > n`` means the operator is a sum of
        simpler pieces, and ``k < n`` means it is singular.
        """
        self._require("factor")
        return self._factor()

    def whiten(self, x) -> Array:
        """Apply a fixed matrix ``W`` satisfying ``W A W.T == I`` to ``x``.

        Transforms ``x`` so that data with this operator as its covariance
        becomes uncorrelated with unit variance. Requires the operator to be
        nonsingular. ``W`` is fixed per instance but otherwise unspecified —
        in particular it need not invert the ``L`` that :meth:`factor`
        returns, so whitened quantities agree with sampled ones in
        distribution, not elementwise.
        """
        self._require("whiten")
        return self._whiten(_check_vec(self, "whiten", x, self.n))

    def whiten_mat(self, X) -> Array:
        """Apply the whitener to a matrix ``X`` of core shape ``(n, k)``."""
        self._require("whiten_mat")
        return self._whiten_mat(_check_mat(self, "whiten_mat", X, self.n))


# ---------------------------------------------------------------------------
# capability tables
# ---------------------------------------------------------------------------

#: Operations available on every operator, at every level.
_ALWAYS_OPS = frozenset({"matvec", "rmatvec", "matmat", "rmatmat", "to_dense"})

#: Operations that are not unconditionally available; `capabilities()` reports
#: the supported subset of these.
_OPTIONAL_OPS = ("solve", "solve_mat", "logdet", "diag", "factor", "whiten", "whiten_mat")

_KNOWN_OPS = _ALWAYS_OPS | frozenset(_OPTIONAL_OPS)

#: Optional operations implemented directly: supported iff the class defines
#: the hook.
_PRIMITIVE_HOOKS = {
    "solve": "_solve",
    "logdet": "_logdet",
    "diag": "_diag",
    "factor": "_factor",
    "whiten": "_whiten",
}

#: Derived operations: supported iff the dependency is, or the class
#: overrides the derived hook with a direct implementation.
_DERIVED_DEPS = {"solve_mat": "solve", "whiten_mat": "whiten"}
_DERIVED_DEFAULTS = {
    "solve_mat": SquareLinOp._solve_mat,
    "whiten_mat": PSDLinOp._whiten_mat,
}


# ---------------------------------------------------------------------------
# explicit densification
# ---------------------------------------------------------------------------


def densify(op: LinOp, *, max_n: int = 4096) -> LinOp:
    """Return ``op`` as a dense operator at the same hierarchy level.

    The explicit fallback for operations an operator does not support
    cheaply. The returned operator provides everything its level defines:

    - a :class:`PSDLinOp` densifies to :class:`~.leaves.DensePSD` (backed by
      a Cholesky factor),
    - any other :class:`SquareLinOp` to :class:`~.leaves.DenseSquare`
      (backed by an LU factorization),
    - anything else to :class:`~.leaves.Dense`.

    Parameters
    ----------
    op
        Operator to materialize.
    max_n
        Largest side length allowed. Raises above this, before allocating,
        so an unintended O(n^3) cost surfaces immediately. Raise it
        deliberately if the cost is wanted.

    Raises
    ------
    ValueError
        If either side of ``op.shape`` exceeds ``max_n``.
    """
    from .leaves import Dense, DensePSD, DenseSquare

    n_out, n_in = op.shape
    if max(n_out, n_in) > max_n:
        raise ValueError(
            f"densify({op!r}) exceeds max_n={max_n}. Raise max_n deliberately "
            f"if you really want an O(n^3) fallback."
        )
    A = op.to_dense()
    if isinstance(op, PSDLinOp):
        return DensePSD.from_matrix(A)
    if isinstance(op, SquareLinOp):
        return DenseSquare.from_matrix(A)
    return Dense(A)
