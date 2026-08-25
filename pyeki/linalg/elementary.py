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
===========================  ===============================================

Elementary operators at the PSD level whose unrestricted mathematical
namesake is not PSD carry the ``PSD`` prefix; the generic names stay
reserved for unrestricted classes, should one ever be needed.

See :mod:`pyeki.linalg.base` for the shape convention shared by all
operators, and :mod:`pyeki.linalg.composite` for operators built out of
these.

Notes
-----
Anything computed from a matrix — a Cholesky or LU factorization — is done
in a ``from_matrix`` classmethod, once, at construction time. The dataclass
constructor itself only stores: JAX rebuilds operators through it on every
``jit``/``vmap`` boundary, so a computing constructor would silently
recompute there, and a factorization cached lazily inside a traced function
is written to a temporary copy and discarded.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .base import (
    LinOp,
    PSDLinOp,
    SquareLinOp,
    _check_core_rank,
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
]


def _check_size(cls_name: str, size) -> None:
    """Validate a static side-length field."""
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError(f"{cls_name}.size must be an int, got {type(size).__name__}")
    if size < 1:
        raise ValueError(f"{cls_name}.size must be positive, got {size}")


def _strict_square_matrix(cls_name: str, A) -> Array:
    """Validate a matrix passed to a ``from_matrix`` classmethod."""
    A = jnp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError(
            f"{cls_name}.from_matrix: expected a square 2-D matrix with positive "
            f"size, got shape {A.shape}"
        )
    return A


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
        the inherited :meth:`~.base.SquareLinOp.diag` method.)

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
        value_check(
            self.diagonal,
            lambda d: bool(jnp.all(d > 0)),
            "PSDDiagonal entries must be strictly positive",
        )

    @property
    def shape(self) -> tuple[int, int]:
        n = self.diagonal.shape[-1]
        return (n, n)

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


@linop
class DenseSquare(SquareLinOp):
    """A dense square matrix with no symmetry assumed, stored with its LU.

    What :func:`~.base.densify` returns for a square non-PSD operator.
    Construct with :meth:`from_matrix` rather than directly; the LU
    factorization runs once, there.

    Parameters
    ----------
    A
        The matrix, of shape ``(n, n)``.
    lu, piv
        Its LU factorization, as returned by ``jax.scipy.linalg.lu_factor``.
    lu_of_transpose
        Static flag: whether ``lu``/``piv`` factorize ``A.T`` rather than
        ``A``. Set by :attr:`T`, which reuses the factorization instead of
        recomputing it.
    """

    A: Array
    lu: Array
    piv: Array
    lu_of_transpose: bool = static_field(default=False)

    def __post_init__(self) -> None:
        _check_square_field("DenseSquare", "A", self.A)
        _check_core_rank("DenseSquare", "lu", self.lu, 2)
        _check_core_rank("DenseSquare", "piv", self.piv, 1)

    @classmethod
    def from_matrix(cls, A) -> DenseSquare:
        """Build from a dense square matrix, factorizing once, here.

        The matrix must be nonsingular; a singular one yields ``inf`` or
        ``nan`` from ``solve`` and ``logdet`` (or an error in debug mode).
        """
        A = _strict_square_matrix("DenseSquare", A)
        lu, piv = jax.scipy.linalg.lu_factor(A)
        value_check(
            lu,
            lambda f: bool(
                jnp.all(jnp.isfinite(f)) & jnp.all(jnp.diagonal(f) != 0)
            ),
            "DenseSquare.from_matrix: matrix is singular or non-finite",
        )
        return cls(A, lu, piv)

    @property
    def shape(self) -> tuple[int, int]:
        n = self.A.shape[-1]
        return (n, n)

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
        """The transpose, backed by the same LU factorization."""
        return DenseSquare(
            self.A.swapaxes(-1, -2), self.lu, self.piv, not self.lu_of_transpose
        )

    def _to_dense(self) -> Array:
        return self.A


@linop
class Triangular(SquareLinOp):
    """A square triangular matrix.

    The natural return type of :meth:`DensePSD.factor`. Not itself PSD, so
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
        value_check(
            self.L,
            lambda mat: bool(
                jnp.allclose(mat, jnp.tril(mat) if self.lower else jnp.triu(mat))
            ),
            "Triangular.L has nonzero entries outside its declared triangle",
        )

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

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
        """The transpose, which is triangular in the opposite direction."""
        return Triangular(self.L.swapaxes(-1, -2), not self.lower)

    def _to_dense(self) -> Array:
        return self.L


@linop
class DensePSD(PSDLinOp):
    """A dense positive-definite matrix, stored as its Cholesky factor.

    Construct with :meth:`from_matrix` rather than directly; the
    factorization runs once, there.

    Parameters
    ----------
    L
        Lower Cholesky factor, satisfying ``L @ L.T`` equals the matrix.
    """

    L: Array

    def __post_init__(self) -> None:
        _check_square_field("DensePSD", "L", self.L)
        value_check(
            self.L,
            lambda mat: bool(jnp.all(jnp.isfinite(mat))),
            "DensePSD.L must be finite; a nan factor means the matrix was not "
            "positive definite",
        )

    @classmethod
    def from_matrix(cls, A) -> DensePSD:
        """Build from a dense positive-definite matrix, factorizing once, here.

        The matrix must be positive definite; the Cholesky of anything else
        is ``nan`` without an exception (or an error in debug mode).
        """
        A = _strict_square_matrix("DensePSD", A)
        return cls(jnp.linalg.cholesky(A))

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

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
