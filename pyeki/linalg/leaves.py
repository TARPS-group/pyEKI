"""Operators defined directly by their own arrays.

============================  ==============================================
class                         represents
============================  ==============================================
:class:`Identity`             the identity matrix
:class:`ScaledIdentity`       a positive multiple of the identity
:class:`Diagonal`             a diagonal matrix with positive entries
:class:`Dense`                an explicit array, possibly rectangular
:class:`Triangular`           a square triangular matrix
:class:`DensePSD`             a dense PSD matrix, stored as its Cholesky
============================  ==============================================

See :mod:`pyeki.linalg.base` for the shape convention shared by
all operators, and :mod:`pyeki.linalg.composite` for operators
built out of these.

Notes
-----
Any factorization an operator needs is performed when it is constructed, not
on first use. A factor cached lazily inside a traced function is written to a
temporary copy of the operator and discarded, which would make the operator
re-factorize on every call.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .base import (
    LinOp,
    PSDOperator,
    SquareLinOp,
    dense_matvec,
    operator,
    static_field,
    tri_solve,
)

__all__ = ["Identity", "ScaledIdentity", "Diagonal", "Dense", "Triangular", "DensePSD"]


@operator
class Identity(PSDOperator):
    """The identity matrix.

    Parameters
    ----------
    size
        Side length.
    """

    size: int = static_field()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def matvec(self, x: Array) -> Array:
        return x

    def solve(self, b: Array) -> Array:
        return b

    def factor(self) -> LinOp:
        return self

    def cholesky(self) -> LinOp:
        return self

    def whiten(self, x: Array) -> Array:
        return x

    def diag(self) -> Array:
        return jnp.ones(self.size)

    def logdet(self) -> Array:
        return jnp.asarray(0.0)

    def to_dense(self) -> Array:
        return jnp.eye(self.size)


@operator
class ScaledIdentity(PSDOperator):
    """A positive multiple of the identity.

    Parameters
    ----------
    c
        Scalar array, strictly positive. Held as an array so it can be
        differentiated with respect to.
    size
        Side length.
    """

    c: Array
    size: int = static_field()

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def matvec(self, x: Array) -> Array:
        return self.c * x

    def solve(self, b: Array) -> Array:
        return b / self.c

    def factor(self) -> LinOp:
        return ScaledIdentity(jnp.sqrt(self.c), self.size)

    def cholesky(self) -> LinOp:
        return self.factor()

    def whiten(self, x: Array) -> Array:
        return x / jnp.sqrt(self.c)

    def diag(self) -> Array:
        return jnp.full((self.size,), self.c)

    def logdet(self) -> Array:
        return self.size * jnp.log(self.c)

    def to_dense(self) -> Array:
        return self.c * jnp.eye(self.size)


@operator
class Diagonal(PSDOperator):
    """A diagonal matrix.

    Parameters
    ----------
    d
        Diagonal entries, strictly positive. Its length sets the size.
    """

    d: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.d.shape[-1]
        return (n, n)

    def matvec(self, x: Array) -> Array:
        return self.d * x

    def solve(self, b: Array) -> Array:
        return b / self.d

    def factor(self) -> LinOp:
        return Diagonal(jnp.sqrt(self.d))

    def cholesky(self) -> LinOp:
        return self.factor()

    def whiten(self, x: Array) -> Array:
        return x / jnp.sqrt(self.d)

    def diag(self) -> Array:
        return self.d

    def logdet(self) -> Array:
        return jnp.sum(jnp.log(self.d))

    def to_dense(self) -> Array:
        return jnp.diag(self.d)


@operator
class Dense(LinOp):
    """An explicit dense array, with no structure assumed.

    May be rectangular, and is not assumed symmetric or definite, so it
    provides only ``matvec``, ``matmat`` and ``to_dense``.

    Parameters
    ----------
    A
        The array, of shape ``(n_out, n_in)``.
    """

    A: Array

    @property
    def shape(self) -> tuple[int, int]:
        return (self.A.shape[-2], self.A.shape[-1])

    def matvec(self, x: Array) -> Array:
        return dense_matvec(self.A, x)

    def to_dense(self) -> Array:
        return self.A


@operator
class Triangular(SquareLinOp):
    """A square triangular matrix.

    Returned by :meth:`~.base.PSDOperator.cholesky`. Not itself PSD, so it
    provides ``solve``, ``logdet`` and ``diag`` but no factorization.

    Parameters
    ----------
    L
        Square array whose relevant triangle holds the matrix.
    lower
        Whether ``L`` is lower triangular.
    """

    L: Array
    lower: bool = static_field(default=True)

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    def matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, x)

    def solve(self, b: Array) -> Array:
        return tri_solve(self.L, b, lower=self.lower)

    def diag(self) -> Array:
        return jnp.diagonal(self.L)

    def logdet(self) -> Array:
        return jnp.sum(jnp.log(jnp.abs(jnp.diagonal(self.L))))

    def to_dense(self) -> Array:
        return self.L

    @property
    def T(self) -> Triangular:  # noqa: N802 - mirrors the NumPy attribute
        """The transpose, which is triangular in the opposite direction."""
        return Triangular(self.L.swapaxes(-1, -2), not self.lower)


@operator
class DensePSD(PSDOperator):
    """A dense positive-definite matrix, stored as its Cholesky factor.

    Construct with :meth:`from_matrix` rather than directly; the
    factorization runs once, there.

    Parameters
    ----------
    L
        Lower Cholesky factor, satisfying ``L @ L.T`` equals the matrix.
    """

    L: Array

    @classmethod
    def from_matrix(cls, A: Array) -> DensePSD:
        """Build from a dense positive-definite matrix.

        Runs the Cholesky factorization once, here, and stores the factor.
        """
        return cls(jnp.linalg.cholesky(jnp.asarray(A)))

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    def matvec(self, x: Array) -> Array:
        # A x = L (L^T x); never re-forms A.
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def solve(self, b: Array) -> Array:
        y = tri_solve(self.L, b, lower=True)
        return tri_solve(self.L, y, lower=True, trans=1)

    def factor(self) -> LinOp:
        return Triangular(self.L, lower=True)

    def cholesky(self) -> LinOp:
        return Triangular(self.L, lower=True)

    def whiten(self, x: Array) -> Array:
        return tri_solve(self.L, x, lower=True)

    def diag(self) -> Array:
        return jnp.sum(self.L * self.L, axis=-1)

    def logdet(self) -> Array:
        return 2.0 * jnp.sum(jnp.log(jnp.diagonal(self.L)))

    def to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)
