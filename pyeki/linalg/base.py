"""Base classes for structured linear operators.

An operator represents a matrix implicitly, by how it acts on vectors, so that
known structure can be exploited instead of storing or factorizing a dense
array. Concrete operators live in :mod:`.leaves` and :mod:`.composite`; this
module defines the interface they share and the helpers for writing new ones.

Class hierarchy
---------------
Each level adds the operations that become well defined at that level.

``LinOp``
    Any linear map, possibly rectangular.
    Provides ``matvec``, ``matmat``, ``to_dense``.
``SquareLinOp``
    A square map.
    Adds ``solve``, ``solve_mat``, ``logdet``, ``diag``.
``PSDOperator``
    A symmetric positive semi-definite map.
    Adds ``factor``, ``cholesky``, ``whiten``.

Array shapes
------------
Every method takes leading batch axes; the operand's core shape is trailing.

==============  ====================  ====================================
method          core operand          signature
==============  ====================  ====================================
``matvec``      vector ``(n_in,)``    ``(..., n_in) -> (..., n_out)``
``matmat``      matrix ``(n_in, k)``  ``(..., n_in, k) -> (..., n_out, k)``
``solve``       vector ``(n,)``       ``(..., n) -> (..., n)``
``solve_mat``   matrix ``(n, k)``     ``(..., n, k) -> (..., n, k)``
==============  ====================  ====================================

The ``k`` in ``matmat`` is part of the core shape, not a batch axis. Use
``matvec`` for a batch of vectors and ``matmat`` for a single matrix operand;
neither infers which you meant from the number of dimensions.

Unsupported operations
----------------------
Not every operator can do everything cheaply. An operator raises
:class:`UnsupportedOpError` rather than falling back to dense linear algebra. Query
support with ``op.supports(name)`` or ``op.capabilities()``, and use
:func:`densify` when a dense fallback is genuinely wanted.

Defining a new operator
-----------------------
Subclass the appropriate level, decorate with :func:`operator`, and implement
``shape``, ``matvec``, ``to_dense``, plus whichever optional methods the
structure supports. Use :func:`dense_matvec` and :func:`tri_solve` for the
array work so the shape convention above is honoured. Mark non-array fields
with :func:`static_field`. Validate against
:func:`pyeki.linalg.testing.check_operator`.

Notes
-----
Design decisions behind the above, recorded because they are not obvious and
are easy to undo by accident:

- The hierarchy has three levels rather than two because ``factor()`` returns
  a square root *as an operator*, and square roots are generally rectangular
  and not self-adjoint. Keeping them in ``LinOp`` means they cannot expose a
  ``solve`` that has no meaning.
- Implementations must contract the *trailing* axis. Writing ``self.M @ x``
  instead contracts the second-to-last axis once ``x`` has two or more
  dimensions, which returns a wrong answer without raising whenever the
  operator is square.
- ``to_dense`` is built from stored arrays rather than by applying the
  operator to an identity matrix. The test suite compares ``matvec`` against
  ``to_dense``, so an implementation via ``matvec`` would make that check
  vacuous.
- Unsupported operations raise instead of densifying so that an accidental
  O(n^3) cost is visible rather than silent.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "LinOp",
    "SquareLinOp",
    "PSDOperator",
    "UnsupportedOpError",
    "densify",
    "operator",
    "static_field",
    "dense_matvec",
    "tri_solve",
]


class UnsupportedOpError(NotImplementedError):
    """Raised when an operator has no cheap implementation of an operation.

    The message names the operator type and lists the operations it does
    support. For an explicit dense fallback, use ``densify(op).solve(b)``.
    """

    def __init__(self, name: str, op: LinOp) -> None:
        cls = type(op).__name__
        have = ", ".join(sorted(op.capabilities())) or "none"
        super().__init__(
            f"{cls} has no cheap `{name}`. It supports: {have}. "
            f"Use densify(op) for an explicit dense fallback."
        )
        self.name, self.operator_type = name, cls


# ---------------------------------------------------------------------------
# dataclass / pytree plumbing
# ---------------------------------------------------------------------------

_SCALARISH = {"int", "float", "bool", "str", "NoneType", "None"}
_CONTAINERS = {"tuple", "list", "dict", "set", "frozenset"}


def _is_likely_static(ann: Any) -> bool:
    """Return True if this field annotation should have been marked static.

    Containers are judged by their parameters, so ``tuple[int, ...]`` is
    static-looking while ``tuple[LinOp, ...]`` is not, since the latter holds
    arrays.

    Notes
    -----
    Annotations arrive as strings in any module using
    ``from __future__ import annotations``, so a type-only check would never
    fire.
    """
    if isinstance(ann, type):
        return ann.__name__ in _SCALARISH or ann.__name__ in _CONTAINERS
    if not isinstance(ann, str):
        return False
    head, _, rest = ann.strip().partition("[")
    head = head.strip().split(".")[-1]
    if head in _SCALARISH:
        return True
    if head in _CONTAINERS:
        parts = [
            p.strip().split(".")[-1]
            for p in rest.rstrip("]").split(",")
            if p.strip() not in ("", "...")
        ]
        return all(p in _SCALARISH for p in parts) if parts else True
    return False


def static_field(**kwargs: Any):
    """Declare a dataclass field as pytree *metadata* rather than a child."""
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["static"] = True
    return field(metadata=metadata, **kwargs)


def operator(cls: type) -> type:
    """Class decorator: make ``cls`` a frozen dataclass and a JAX pytree.

    Array-valued fields become pytree children, so operators can be passed
    through ``jit``, ``vmap`` and ``grad``. Fields marked with
    :func:`static_field` become static metadata instead.

    Raises
    ------
    TypeError
        If a field annotated as a scalar or plain container is not marked
        static. Such a field would become a pytree child and arrive as a
        tracer under ``jit``, typically failing later inside a shape
        expression.

    Notes
    -----
    Operators compare by identity (``eq=False``). Dataclass equality would
    compare arrays elementwise and raise on the ambiguous truth value, and the
    generated ``__hash__`` would fail on array fields. Pass operators as
    ordinary traced arguments, never via ``static_argnums``.

    No ``__repr__`` is generated either (``repr=False``), so that
    :meth:`LinOp.__repr__` is the one that applies. A generated ``__repr__``
    is defined on the decorated class itself, which shadows the base class's,
    and prints every field — dumping whole arrays into tracebacks, pytest
    identifiers and debugger output.
    """
    cls = dataclass(frozen=True, eq=False, repr=False)(cls)

    data_fields, meta_fields = [], []
    for f in dataclasses.fields(cls):
        if f.metadata.get("static", False):
            meta_fields.append(f.name)
            continue
        if _is_likely_static(f.type):
            name = f.type if isinstance(f.type, str) else f.type.__name__
            raise TypeError(
                f"{cls.__name__}.{f.name}: `{name}` field is not marked static, "
                f"so it would become a pytree child and arrive as a tracer under "
                f"jit. Use static_field(), or store it as an array."
            )
        data_fields.append(f.name)

    jax.tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )
    return cls


# ---------------------------------------------------------------------------
# array helpers honouring the batch contract
# ---------------------------------------------------------------------------

def dense_matvec(M: Array, x: Array) -> Array:
    """Apply a dense matrix to ``x``, contracting its trailing axis.

    Use this rather than ``M @ x`` when implementing ``matvec``: for arrays
    with two or more dimensions the ``@`` operator contracts the
    second-to-last axis, which silently returns a wrong answer when ``M`` is
    square.
    """
    return jnp.einsum("ij,...j->...i", M, x)


def tri_solve(L: Array, x: Array, *, lower: bool, trans: int = 0) -> Array:
    """Solve a triangular system, contracting the trailing axis of ``x``.

    Accepts any number of leading batch axes.
    """
    flat = x.reshape(-1, x.shape[-1]).T                     # (n, m)
    out = jax.scipy.linalg.solve_triangular(L, flat, lower=lower, trans=trans)
    return out.T.reshape(x.shape)


# ---------------------------------------------------------------------------
# level 1 -- LinOp
# ---------------------------------------------------------------------------

_OPTIONAL = ("solve", "solve_mat", "logdet", "diag", "factor", "cholesky", "whiten")


class LinOp:
    """A linear map, possibly rectangular.

    The base of the operator hierarchy. See the module docstring for the shape
    convention and for how to define a new operator.
    """

    #: Names of operations this subclass explicitly does not support, even
    #: though a base class provides them. Rarely needed.
    _WITHDRAWN: ClassVar[frozenset[str]] = frozenset()

    # -- required -----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Shape as ``(n_out, n_in)``.

        Notes
        -----
        Defined as a property rather than a stored field so that it stays a
        concrete Python tuple under ``jit`` and can be used in shape
        expressions.
        """
        raise NotImplementedError(f"{type(self).__name__} must define `shape`")

    def matvec(self, x: Array) -> Array:
        """Apply the operator to ``x``, contracting its trailing axis."""
        raise NotImplementedError(f"{type(self).__name__} must define `matvec`")

    def to_dense(self) -> Array:
        """Return the operator as a dense array.

        Implementations build this from their stored arrays rather than by
        applying the operator to an identity matrix.
        """
        raise NotImplementedError(f"{type(self).__name__} must define `to_dense`")

    # -- derived ------------------------------------------------------------
    def matmat(self, X: Array) -> Array:
        """Apply the operator to a matrix ``X`` of core shape ``(n_in, k)``."""
        return self.matvec(X.swapaxes(-1, -2)).swapaxes(-1, -2)

    # -- capability introspection -------------------------------------------
    def supports(self, name: str) -> bool:
        """Return True if ``name`` has a cheap implementation on this instance.

        Notes
        -----
        This is an instance method because support can depend on an operator's
        contents: a ``BlockDiag`` can only ``solve`` if all of its blocks can.
        """
        if name in self._WITHDRAWN:
            return False
        impl = getattr(type(self), name, None)
        return impl is not None and impl is not _BASE_IMPL.get(name)

    def capabilities(self) -> frozenset[str]:
        """Return the names of all optional operations this instance supports."""
        return frozenset(n for n in _OPTIONAL if self.supports(n))

    def _unsupported(self, name: str):
        raise UnsupportedOpError(name, self)

    def __repr__(self) -> str:
        """Return the type name and shape, as ``Dense(4, 6)``.

        Deliberately omits field values, which are usually arrays too large to
        be worth printing. :func:`operator` suppresses the dataclass-generated
        ``__repr__`` so that this one is not shadowed.
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

    def solve(self, b: Array) -> Array:
        """Solve ``A x = b``, contracting the trailing axis of ``b``."""
        self._unsupported("solve")

    def solve_mat(self, B: Array) -> Array:
        """Solve ``A X = B`` for a matrix ``B`` of core shape ``(n, k)``."""
        return self.solve(B.swapaxes(-1, -2)).swapaxes(-1, -2)

    def logdet(self) -> Array:
        """Return the log-determinant as a real scalar array.

        Notes
        -----
        Always an array, never a Python ``float``: converting would fail on a
        tracer under ``jit``, and also on any complex intermediate produced by
        a spectrally diagonalized operator.
        """
        self._unsupported("logdet")

    def diag(self) -> Array:
        """Return the diagonal as a vector of length ``n``."""
        self._unsupported("diag")


# ---------------------------------------------------------------------------
# level 3 -- PSDOperator
# ---------------------------------------------------------------------------

class PSDOperator(SquareLinOp):
    """A symmetric positive semi-definite linear map.

    Notes
    -----
    There are no ``rmatvec``/``rmatmat`` methods. A PSD operator is
    self-adjoint, so they would duplicate ``matvec`` and ``matmat``.
    """

    def factor(self) -> LinOp:
        """Return an operator ``L`` of shape ``(n, k)`` with ``L @ L.T == self``.

        Use this to draw samples: ``L.matvec(eps)`` for standard normal
        ``eps`` of length ``k`` has covariance equal to this operator.

        The factor need not be square, in either direction. A
        low-rank-plus-diagonal operator gives ``k > n``; a reduced-rank
        operator gives ``k < n`` and is singular, so it will have no ``solve``.
        """
        self._unsupported("factor")

    def cholesky(self) -> LinOp:
        """Return the square triangular ``L`` with ``L @ L.T == self``.

        Available only when the factor is square. To whiten a vector, prefer
        :meth:`whiten`, which does not require the factor to be triangular.
        """
        self._unsupported("cholesky")

    def whiten(self, x: Array) -> Array:
        """Return ``L^-1 x`` for a factor ``L`` satisfying ``L @ L.T == self``.

        Transforms ``x`` so that data with this operator as its covariance
        becomes uncorrelated with unit variance.
        """
        return self.cholesky().solve(x)


# Snapshot of the base implementations, so `supports` can tell "inherited the
# raising default" from "overridden with something real".
_BASE_IMPL: dict[str, Callable] = {}
for _lvl in (LinOp, SquareLinOp, PSDOperator):
    for _name in _OPTIONAL:
        if _name in _lvl.__dict__:
            _BASE_IMPL[_name] = _lvl.__dict__[_name]
del _lvl, _name


# ---------------------------------------------------------------------------
# explicit densification
# ---------------------------------------------------------------------------

def densify(op: LinOp, *, max_n: int = 4096):
    """Return ``op`` as a dense operator, subject to a size limit.

    The explicit fallback for operations an operator does not support cheaply.
    Returns a :class:`~.leaves.DensePSD` if ``op`` is PSD, otherwise a
    :class:`~.leaves.Dense`.

    Parameters
    ----------
    op
        Operator to materialize.
    max_n
        Largest side length allowed. Raises above this rather than allocating,
        so an unintended O(n^3) cost surfaces immediately. Raise it
        deliberately if the cost is wanted.

    Raises
    ------
    ValueError
        If either side of ``op.shape`` exceeds ``max_n``.
    """
    from .leaves import Dense, DensePSD

    n_out, n_in = op.shape
    if max(n_out, n_in) > max_n:
        raise ValueError(
            f"densify({type(op).__name__}{op.shape}) exceeds max_n={max_n}. "
            f"Raise max_n deliberately if you really want an O(n^3) fallback."
        )
    A = op.to_dense()
    return DensePSD.from_matrix(A) if isinstance(op, PSDOperator) else Dense(A)
