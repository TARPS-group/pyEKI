"""Operators defined directly by their own arrays.

===========================  ===============================================
class                        represents
===========================  ===============================================
:class:`Identity`            the identity matrix
:class:`PSDDiagonal`         a diagonal matrix with positive entries
:class:`Dense`               an explicit array, possibly rectangular
:class:`DenseSquare`         a dense square matrix, stored with its LU
:class:`Triangular`          a square triangular matrix
:class:`DensePSD`            a dense PSD matrix, stored as its Cholesky
:class:`PSDLowRank`          ``F F.T`` for a stored factor ``F``
===========================  ===============================================

Elementary operators at the PSD level whose unrestricted mathematical
namesake is not PSD carry the ``PSD`` prefix; the generic names stay
reserved for unrestricted classes, should one ever be needed.

See :mod:`pyeki.linalg.base` for the shape convention shared by all
operators, and :mod:`pyeki.linalg.composite` for operators built out of
these.

Notes
-----
Anything computed from a matrix — a Cholesky or LU factorization — is
computed by the constructor, once, and stored in a field. Pytree
reconstruction rebuilds operators from their stored fields alone, bypassing
the constructor, so the fields must already hold everything the operator
needs — and a factorization cached lazily inside a traced function is
written to a temporary copy and discarded.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .base import (
    LinOp,
    PSDLinOp,
    SquareLinOp,
    _broadcast_batch,
    _check_core_rank,
    _check_finite,
    _check_real,
    _check_triangular,
    _construct_unchecked,
    dense_matvec,
    linop,
    static_field,
    tri_solve,
    value_check,
)

__all__ = [
    "Identity",
    "PSDDiagonal",
    "Dense",
    "DenseSquare",
    "Triangular",
    "DensePSD",
    "PSDLowRank",
]


def _check_size(cls_name: str, size) -> None:
    """Validate a static side-length field."""
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError(f"{cls_name}.size must be an int, got {type(size).__name__}")
    if size < 1:
        raise ValueError(f"{cls_name}.size must be positive, got {size}")


def _check_square_field(cls_name: str, field_name: str, value) -> None:
    """Structural check for a stored square-matrix field: rank exactly 2
    and equal axes."""
    _check_core_rank(cls_name, field_name, value, 2)
    shape = getattr(value, "shape", None)
    if shape is not None and shape[-1] != shape[-2]:
        raise ValueError(
            f"{cls_name}.{field_name}: expected a square matrix, got core shape "
            f"({shape[-2]}, {shape[-1]})"
        )


@linop
class Identity(PSDLinOp):
    """The identity matrix.

    Parameters
    ----------
    size
        Side length, a positive int.
    """

    size: int = static_field()

    def __post_init__(self) -> None:
        _check_size("Identity", self.size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    def _matvec(self, x: Array) -> Array:
        return x

    def _solve(self, b: Array) -> Array:
        return b

    def _logdet(self) -> Array:
        return jnp.asarray(0.0)

    def _diag(self) -> Array:
        return jnp.ones(self.size)

    def _factor(self) -> LinOp:
        return self

    def _whiten(self, x: Array) -> Array:
        return x

    def _to_dense(self) -> Array:
        return jnp.eye(self.size)


@linop
class PSDDiagonal(PSDLinOp):
    """A diagonal matrix with strictly positive entries.

    Parameters
    ----------
    diagonal
        The diagonal entries, strictly positive. Their number sets the
        size. (Named ``diagonal`` rather than ``diag``, which would shadow
        the inherited :meth:`~pyeki.linalg.SquareLinOp.diag` method.)

    Notes
    -----
    The positivity precondition is not a claim about diagonal matrices in
    general: it is what makes the class's PSD level and its advertised
    capabilities (``solve``, ``whiten``, ``factor``, ``logdet``) true, and
    the name says so. A signed diagonal would be a separate
    :class:`~.base.SquareLinOp`-level class, to be added when a consumer
    needs it.
    """

    diagonal: Array

    def __post_init__(self) -> None:
        _check_core_rank("PSDDiagonal", "diagonal", self.diagonal, 1)
        _check_real("PSDDiagonal", "diagonal", self.diagonal)
        _check_finite("PSDDiagonal", "diagonal", self.diagonal)
        value_check(
            self.diagonal,
            lambda d: bool(jnp.all(d > 0)),
            "PSDDiagonal entries must be strictly positive",
        )

    @property
    def shape(self) -> tuple[int, int]:
        n = self.diagonal.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.diagonal.shape[:-1])

    def _matvec(self, x: Array) -> Array:
        return self.diagonal * x

    def _solve(self, b: Array) -> Array:
        return b / self.diagonal

    def _logdet(self) -> Array:
        return jnp.sum(jnp.log(self.diagonal), axis=-1)

    def _diag(self) -> Array:
        return self.diagonal

    def _factor(self) -> LinOp:
        return PSDDiagonal(jnp.sqrt(self.diagonal))

    def _whiten(self, x: Array) -> Array:
        return x / jnp.sqrt(self.diagonal)

    def _to_dense(self) -> Array:
        return jnp.diag(self.diagonal)


@linop
class Dense(LinOp):
    """An explicit dense array, with no structure assumed.

    May be rectangular, and is not assumed symmetric or definite, so it
    provides application and transposition only.

    Parameters
    ----------
    A
        The array, of shape ``(n_out, n_in)``.
    """

    A: Array

    def __post_init__(self) -> None:
        _check_core_rank("Dense", "A", self.A, 2)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.A.shape[-2], self.A.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.A.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.A, x)

    def _rmatvec(self, x: Array) -> Array:
        return dense_matvec(self.A.swapaxes(-1, -2), x)

    def _matmat(self, X: Array) -> Array:
        return self.A @ X

    def _rmatmat(self, X: Array) -> Array:
        return self.A.swapaxes(-1, -2) @ X

    def _to_dense(self) -> Array:
        return self.A


def _lu_is_nonsingular(lu: Array) -> bool:
    """Whether an LU factor is finite and numerically nonsingular.

    Rounding rarely leaves an exactly zero pivot, even for an exactly
    singular matrix, so a pivot counts as zero when it is below ``n`` units
    of roundoff relative to the largest entry of the factor.
    """
    n = lu.shape[-1]
    pivots = jnp.abs(jnp.diagonal(lu, axis1=-2, axis2=-1))
    tol = n * jnp.finfo(lu.dtype).eps * jnp.max(jnp.abs(lu))
    return bool(jnp.all(jnp.isfinite(lu)) & jnp.all(pivots > tol))


def _check_given_lu(A: Array, lu: Array, piv: Array) -> None:
    """Check the shapes and pivot dtype of an LU passed to :class:`DenseSquare`."""
    n = A.shape[-1]
    _check_core_rank("DenseSquare", "lu", lu, 2)
    _check_core_rank("DenseSquare", "piv", piv, 1)
    if lu.shape[-2:] != (n, n):
        raise ValueError(
            f"DenseSquare.lu: expected core shape ({n}, {n}) to match A, got "
            f"{lu.shape[-2:]}"
        )
    if piv.shape[-1] != n:
        raise ValueError(
            f"DenseSquare.piv: expected length {n} to match A, got {piv.shape[-1]}"
        )
    if not jnp.issubdtype(piv.dtype, jnp.integer):
        raise TypeError(
            f"DenseSquare.piv must be an integer array, got dtype {piv.dtype}"
        )


def _check_lu_factorizes(
    A: Array, lu: Array, piv: Array, lu_of_transpose: bool
) -> None:
    """Debug check that an LU passed to :class:`DenseSquare` factorizes ``A``.

    Solves with it for one right-hand side and requires a backward-stable
    residual, which any factorization of a different matrix misses.
    """
    n = A.shape[-1]

    def factorizes(M):
        x = jnp.linspace(1.0, 2.0, n)
        b = dense_matvec(M, x)
        y = jax.scipy.linalg.lu_solve((lu, piv), b, trans=int(lu_of_transpose))
        residual = jnp.linalg.norm(dense_matvec(M, y) - b)
        scale = jnp.linalg.norm(M) * jnp.linalg.norm(y) + jnp.linalg.norm(b)
        return bool(residual <= 1e-8 * scale)

    value_check(
        A,
        factorizes,
        "DenseSquare: lu and piv do not factorize A"
        + (".T" if lu_of_transpose else "")
        + ". To factorize a new matrix, build DenseSquare(A).",
    )


@linop
class DenseSquare(SquareLinOp):
    """A dense square matrix with no symmetry assumed, stored with its LU.

    What :func:`~pyeki.linalg.densify` returns for a square non-PSD operator.
    ``DenseSquare(A)`` computes the LU factorization once, at construction.

    Parameters
    ----------
    A
        The matrix, of shape ``(n, n)``. It must be nonsingular; a singular
        one yields ``inf`` or ``nan`` from ``solve`` and ``logdet``, without
        raising unless debug checks are enabled.
    lu, piv
        Keyword-only: an LU factorization of ``A`` already computed, as
        returned by ``jax.scipy.linalg.lu_factor`` — ``lu`` of shape
        ``(n, n)``, ``piv`` an integer array of shape ``(n,)``. Pass both or
        neither; when omitted, they are computed from ``A``.
    lu_of_transpose
        Keyword-only, and only with ``lu`` and ``piv``: whether they
        factorize ``A.T`` rather than ``A``. Set by ``T``, which reuses the
        factorization instead of recomputing it.

    Raises
    ------
    TypeError
        If only one of ``lu`` and ``piv`` is given, if ``piv`` is not an
        integer array, or if ``lu_of_transpose`` is not a ``bool``.
    ValueError
        If ``A`` is not a square matrix of rank exactly 2, if ``lu`` or
        ``piv`` does not match its size, if ``lu_of_transpose`` is set
        without ``lu`` and ``piv``, or — in debug mode — if the
        factorization is singular or non-finite, or a given one does not
        factorize ``A``.

    Notes
    -----
    ``piv`` is an integer array and is pytree data (it must batch under
    ``vmap``), so differentiating with respect to a pytree containing a
    ``DenseSquare`` requires ``jax.grad(..., allow_int=True)``.

    :func:`dataclasses.replace` passes the stored ``lu`` and ``piv`` back
    to the constructor, so replacing ``A`` alone pairs the new matrix with
    the old factorization. Build a new ``DenseSquare(A)`` instead; debug
    mode rejects the mismatch.
    """

    A: Array
    lu: Array
    piv: Array
    lu_of_transpose: bool = static_field(default=False)

    def __init__(
        self, A, *, lu=None, piv=None, lu_of_transpose: bool = False
    ) -> None:
        A = jnp.asarray(A)
        _check_square_field("DenseSquare", "A", A)
        if not isinstance(lu_of_transpose, bool):
            raise TypeError(
                f"DenseSquare.lu_of_transpose must be a bool, got "
                f"{type(lu_of_transpose).__name__}"
            )
        if (lu is None) != (piv is None):
            raise TypeError("DenseSquare: pass lu and piv together, or neither")
        given = lu is not None
        if not given:
            if lu_of_transpose:
                raise ValueError(
                    "DenseSquare: lu_of_transpose describes a given "
                    "factorization, so it needs lu and piv"
                )
            lu, piv = jax.scipy.linalg.lu_factor(A)
        else:
            lu, piv = jnp.asarray(lu), jnp.asarray(piv)
            _check_given_lu(A, lu, piv)
        value_check(
            lu, _lu_is_nonsingular, "DenseSquare: matrix is singular or non-finite"
        )
        if given:
            _check_lu_factorizes(A, lu, piv, lu_of_transpose)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "lu", lu)
        object.__setattr__(self, "piv", piv)
        object.__setattr__(self, "lu_of_transpose", lu_of_transpose)

    @property
    def shape(self) -> tuple[int, int]:
        n = self.A.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch(
            "DenseSquare",
            self.A.shape[:-2],
            self.lu.shape[:-2],
            self.piv.shape[:-1],
        )

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.A, x)

    def _rmatvec(self, x: Array) -> Array:
        return dense_matvec(self.A.swapaxes(-1, -2), x)

    def _matmat(self, X: Array) -> Array:
        return self.A @ X

    def _rmatmat(self, X: Array) -> Array:
        return self.A.swapaxes(-1, -2) @ X

    def _solve(self, b: Array) -> Array:
        trans = 1 if self.lu_of_transpose else 0
        flat = b.reshape(-1, b.shape[-1]).swapaxes(-1, -2)  # (n, m)
        out = jax.scipy.linalg.lu_solve((self.lu, self.piv), flat, trans=trans)
        return out.swapaxes(-1, -2).reshape(b.shape)

    def _logdet(self) -> Array:
        diag_u = jnp.diagonal(self.lu, axis1=-2, axis2=-1)
        return jnp.sum(jnp.log(jnp.abs(diag_u)), axis=-1)

    def _diag(self) -> Array:
        return jnp.diagonal(self.A, axis1=-2, axis2=-1)

    @property
    def T(self) -> DenseSquare:  # noqa: N802 - mirrors the NumPy attribute
        """The transpose, backed by the same LU factorization.

        Built through the constructor-bypassing path so that it also works
        as a view on a vmapped family, whose batched leaves the strict
        constructor would reject.
        """
        return _construct_unchecked(
            DenseSquare,
            A=self.A.swapaxes(-1, -2),
            lu=self.lu,
            piv=self.piv,
            lu_of_transpose=not self.lu_of_transpose,
        )

    def _to_dense(self) -> Array:
        return self.A


@linop
class Triangular(SquareLinOp):
    """A square triangular matrix.

    The natural return type of ``DensePSD.factor()``. Not itself PSD, so
    it provides ``solve``, ``logdet`` and ``diag`` but no factorization.

    Parameters
    ----------
    L
        Square array, actually triangular in the direction ``lower`` says;
        entries in the other triangle must be zero.
    lower
        Whether ``L`` is lower triangular.
    """

    L: Array
    lower: bool = static_field(default=True)

    def __post_init__(self) -> None:
        _check_square_field("Triangular", "L", self.L)
        if not isinstance(self.lower, bool):
            raise TypeError(
                f"Triangular.lower must be a bool, got {type(self.lower).__name__}"
            )
        _check_finite("Triangular", "L", self.L)
        _check_triangular("Triangular", "L", self.L, lower=self.lower)

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, x)

    def _rmatvec(self, x: Array) -> Array:
        return dense_matvec(self.L.swapaxes(-1, -2), x)

    def _matmat(self, X: Array) -> Array:
        return self.L @ X

    def _rmatmat(self, X: Array) -> Array:
        return self.L.swapaxes(-1, -2) @ X

    def _solve(self, b: Array) -> Array:
        return tri_solve(self.L, b, lower=self.lower)

    def _logdet(self) -> Array:
        d = jnp.diagonal(self.L, axis1=-2, axis2=-1)
        return jnp.sum(jnp.log(jnp.abs(d)), axis=-1)

    def _diag(self) -> Array:
        return jnp.diagonal(self.L, axis1=-2, axis2=-1)

    @property
    def T(self) -> Triangular:  # noqa: N802 - mirrors the NumPy attribute
        """The transpose, which is triangular in the opposite direction.

        Built through the constructor-bypassing path so that it also works
        as a view on a vmapped family, whose batched leaves the strict
        constructor would reject.
        """
        return _construct_unchecked(
            Triangular, L=self.L.swapaxes(-1, -2), lower=not self.lower
        )

    def _to_dense(self) -> Array:
        return self.L


@linop
class DensePSD(PSDLinOp):
    """A dense positive-definite matrix, stored as its Cholesky factor.

    Build from the matrix, ``DensePSD(A)``, which computes the Cholesky
    factor once, at construction; or from a factor already computed,
    ``DensePSD(L=L)``. Exactly one of the two must be given.

    Parameters
    ----------
    A
        The matrix, of shape ``(n, n)``: symmetric positive definite.
    L
        Keyword-only: the lower Cholesky factor of the matrix, lower
        triangular with a strictly positive diagonal and ``L @ L.T`` equal
        to the matrix. It is stored as given.

    Raises
    ------
    TypeError
        If both or neither of ``A`` and ``L`` are given.
    ValueError
        If the given array is not a square matrix of rank exactly 2, or — in
        debug mode — if ``A`` is not symmetric positive definite or ``L`` is
        not a Cholesky factor.

    Notes
    -----
    Without debug checks, an invalid argument gives an operator for a
    different matrix, silently. The Cholesky factorizes the symmetric part
    ``(A + A.T) / 2``, so a non-symmetric matrix is replaced by it, and an
    indefinite one gives a ``nan`` factor. A matrix passed as ``L`` is
    multiplied out whole by ``matvec`` but read only in its lower triangle
    by ``solve``, ``whiten`` and ``logdet``.
    """

    L: Array

    def __init__(self, A=None, *, L=None) -> None:
        if (A is None) == (L is None):
            raise TypeError(
                "DensePSD takes exactly one of the matrix, positionally, or its "
                "Cholesky factor, as L="
            )
        if L is None:
            A = jnp.asarray(A)
            _check_square_field("DensePSD", "A", A)
            value_check(
                A,
                lambda M: bool(jnp.allclose(M, M.swapaxes(-1, -2))),
                "DensePSD: matrix must be symmetric",
            )
            L = jnp.linalg.cholesky(A)
            _check_finite(
                "DensePSD",
                "L",
                L,
                hint="A matrix that is not positive definite has a nan "
                "Cholesky factor.",
            )
        else:
            L = jnp.asarray(L)
            _check_square_field("DensePSD", "L", L)
            _check_finite("DensePSD", "L", L)
            _check_triangular(
                "DensePSD",
                "L",
                L,
                lower=True,
                hint="To build from the matrix itself, pass it positionally: "
                "DensePSD(A).",
            )
            value_check(
                L,
                lambda mat: bool(jnp.all(jnp.diagonal(mat) > 0)),
                "DensePSD.L must have a strictly positive diagonal, as a "
                "Cholesky factor does",
            )
        object.__setattr__(self, "L", L)

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        # A x = L (L^T x); never re-forms A.
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def _matmat(self, X: Array) -> Array:
        return self.L @ (self.L.swapaxes(-1, -2) @ X)

    def _solve(self, b: Array) -> Array:
        y = tri_solve(self.L, b, lower=True)
        return tri_solve(self.L, y, lower=True, trans=1)

    def _logdet(self) -> Array:
        d = jnp.diagonal(self.L, axis1=-2, axis2=-1)
        return 2.0 * jnp.sum(jnp.log(d), axis=-1)

    def _diag(self) -> Array:
        return jnp.sum(self.L * self.L, axis=-1)

    def _factor(self) -> LinOp:
        return Triangular(self.L, lower=True)

    def _whiten(self, x: Array) -> Array:
        return tri_solve(self.L, x, lower=True)

    def _to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)


@linop
class PSDLowRank(PSDLinOp):
    """``F @ F.T`` for a stored factor ``F``, which may be thin, square or wide.

    Singular whenever ``F`` is thin, and typed accordingly: it provides
    ``diag`` and ``factor`` and nothing else, so ``solve``, ``whiten`` and
    ``logdet`` raise :class:`~.base.UnsupportedOpError` at every width.

    Parameters
    ----------
    F
        The factor, of shape ``(n, k)``, both sizes at least 1. No relation
        between ``n`` and ``k`` is required. The operator it represents has
        side ``n`` and rank at most ``k``.

    Notes
    -----
    Nothing is computed at construction, because the stored field *is* the
    factorization: ``factor()`` wraps ``F`` as a :class:`Dense`, and
    ``matvec`` applies ``F (F.T x)`` rather than ever forming ``F F.T``.

    Withholding ``solve`` and ``whiten`` is forced when ``k < n``: a
    statically thin factor makes the operator singular by construction. It
    extends to ``k >= n``, and to ``logdet``, because a capability asserts
    a *cheap* implementation, and none of the three is cheap at any width:
    each needs the ``(n, n)`` Gram matrix ``F F.T`` formed and factorized,
    which is what :func:`~pyeki.linalg.densify` already does.
    Rank is a second reason: capabilities that varied with the stored width
    would advertise a ``solve`` that is ``nan`` for every rank-deficient
    wide factor, which no shape can rule out.

    Densifying is therefore the route to those operations, and it is only
    valid on an instance known to be full rank. Densifying a thin-factor
    instance returns ``nan`` without raising, except under
    :func:`~pyeki.linalg.debug_checks`.

    Construction validates the rank and sizes of ``F`` always, and its
    finiteness when debug checks are enabled — the same check
    :class:`DensePSD` applies to its own factor, since a non-finite factor
    makes every operation ``nan`` with no exception.
    """

    F: Array

    def __post_init__(self) -> None:
        _check_core_rank("PSDLowRank", "F", self.F, 2)
        _check_finite("PSDLowRank", "F", self.F)

    @property
    def shape(self) -> tuple[int, int]:
        n = self.F.shape[-2]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.F.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        # A x = F (F^T x); never forms A.
        return dense_matvec(self.F, dense_matvec(self.F.swapaxes(-1, -2), x))

    def _diag(self) -> Array:
        return jnp.sum(self.F * self.F, axis=-1)

    def _factor(self) -> LinOp:
        return Dense(self.F)

    def _to_dense(self) -> Array:
        return self.F @ self.F.swapaxes(-1, -2)
