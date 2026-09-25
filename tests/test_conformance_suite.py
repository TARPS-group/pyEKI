"""Regression tests for the operator conformance suite itself.

Each test builds a deliberately broken operator that returns wrong answers,
or fails only under a transformation, without raising in ordinary use, and
requires :func:`pyeki.linalg.testing.check_operator` to reject it. Every
operator here passed an earlier version of the suite; the test names the bug
class the suite now catches.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.linalg import (
    DensePSD,
    DenseSquare,
    LinOp,
    PSDLinOp,
    SquareLinOp,
    Triangular,
    dense_matvec,
    linop,
    static_field,
)
from pyeki.linalg.base import _construct_unchecked
from pyeki.linalg.testing import check_operator

RNG = np.random.default_rng(0)


def _square(n: int) -> Array:
    """A well-conditioned, non-symmetric square matrix."""
    return jnp.asarray(RNG.normal(size=(n, n)) + n * np.eye(n))


def _psd(n: int) -> Array:
    M = RNG.normal(size=(n, n))
    return jnp.asarray(M @ M.T + n * np.eye(n))


@linop
class _DenseBase(SquareLinOp):
    """A correct dense square operator, for the broken ones to override."""

    A: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.A.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.A.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.A, x)

    def _rmatvec(self, x: Array) -> Array:
        return dense_matvec(self.A.swapaxes(-1, -2), x)

    def _to_dense(self) -> Array:
        return self.A


# ---------------------------------------------------------------------------
# application at batch and column sizes equal to the operator size
# ---------------------------------------------------------------------------


@linop
class _OrientationSniffing(_DenseBase):
    """Treats a leading axis of length ``n`` as the contracted one."""

    def _matvec(self, x: Array) -> Array:
        if x.ndim >= 2 and x.shape[0] == self.A.shape[-1]:
            return self.A @ x
        return dense_matvec(self.A, x)


def test_batch_size_equal_to_n():
    """``matvec`` that contracts the wrong axis when the batch size is ``n``."""
    with pytest.raises(AssertionError):
        check_operator(_OrientationSniffing(_square(5)))


@linop
class _SqueezingMatmat(LinOp):
    """Squeezes the ``matmat`` result, dropping a ``k = 1`` column axis."""

    A: Array

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
        return jnp.squeeze(self.A @ X)

    def _to_dense(self) -> Array:
        return self.A


def test_single_column_matmat():
    """``matmat`` that loses the column axis when ``k = 1``."""
    with pytest.raises(AssertionError):
        check_operator(_SqueezingMatmat(jnp.asarray(RNG.normal(size=(3, 4)))))


# ---------------------------------------------------------------------------
# factor and transpose, checked as operators in their own right
# ---------------------------------------------------------------------------


@linop
class _MislabelledFactor(DensePSD):
    """Returns its lower Cholesky factor labelled as upper triangular."""

    def _factor(self) -> LinOp:
        return Triangular(self.L, lower=False)


def test_factor_solve():
    """A factor whose dense form is right but whose ``solve`` is wrong."""
    with pytest.raises(AssertionError):
        check_operator(_MislabelledFactor(jnp.linalg.cholesky(_psd(4))))


@linop
class _FlagSettingTranspose(DenseSquare):
    """Sets ``lu_of_transpose`` on transposition instead of toggling it."""

    def __init__(self, A, **kwargs) -> None:
        super().__init__(A, **kwargs)

    @property
    def T(self) -> DenseSquare:  # noqa: N802
        return _construct_unchecked(
            type(self),
            A=self.A.swapaxes(-1, -2),
            lu=self.lu,
            piv=self.piv,
            lu_of_transpose=True,
        )


def test_double_transpose_solve():
    """``T.T`` whose dense form is right but whose ``solve`` is wrong."""
    with pytest.raises(AssertionError):
        check_operator(_FlagSettingTranspose(_square(4)))


# ---------------------------------------------------------------------------
# pytree behaviour of every operation, not only matvec
# ---------------------------------------------------------------------------


@linop
class _PrivateCache(_DenseBase):
    """Stores its inverse in an attribute that is not a pytree field."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "_inv", jnp.linalg.inv(self.A))

    def _solve(self, b: Array) -> Array:
        return dense_matvec(self._inv, b)


def test_factorization_outside_the_pytree():
    """``solve`` reads state that pytree reconstruction does not carry."""
    with pytest.raises(AssertionError):
        check_operator(_PrivateCache(_square(4)))


@linop
class _NumPySolve(_DenseBase):
    """Solves with NumPy, which cannot run on traced values."""

    def _solve(self, b: Array) -> Array:
        n = self.shape[0]
        flat = np.asarray(b).reshape(-1, n).T
        out = np.linalg.solve(np.asarray(self.A), flat).T
        return jnp.asarray(out.reshape(b.shape))


def test_numpy_solve_under_jit():
    """``solve`` that is correct eagerly and fails under ``jit``."""
    with pytest.raises(AssertionError):
        check_operator(_NumPySolve(_square(4)))


@linop
class _StaticInverse(_DenseBase):
    """Stores its inverse as static metadata, a nested tuple of floats."""

    inv: tuple = static_field()

    def __init__(self, A) -> None:
        A = jnp.asarray(A)
        object.__setattr__(self, "A", A)
        inv = np.linalg.inv(np.asarray(A))
        object.__setattr__(self, "inv", tuple(map(tuple, inv.tolist())))

    def _solve(self, b: Array) -> Array:
        return dense_matvec(jnp.asarray(self.inv), b)


def test_array_data_in_static_metadata():
    """A family built from two different instances exposes stale metadata."""
    with pytest.raises(AssertionError):
        check_operator(_StaticInverse(_square(4)), other=_StaticInverse(_square(4)))


# ---------------------------------------------------------------------------
# gradient values
# ---------------------------------------------------------------------------


@linop
class _StopGradient(_DenseBase):
    """Blocks the gradient of ``matvec`` with respect to its matrix."""

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(jax.lax.stop_gradient(self.A), x)


def test_gradient_values():
    """``matvec`` whose gradient disagrees with that of the dense form."""
    with pytest.raises(AssertionError):
        check_operator(_StopGradient(_square(4)))


# ---------------------------------------------------------------------------
# output types and precision
# ---------------------------------------------------------------------------


@linop
class _NumPyDiag(_DenseBase):
    """Returns its diagonal as a NumPy array."""

    def _diag(self):
        return np.diag(np.asarray(self.A))


def test_numpy_output():
    """An operation that returns a NumPy array instead of a JAX array."""
    with pytest.raises(AssertionError):
        check_operator(_NumPyDiag(_square(4)))


@linop
class _Float32Logdet(_DenseBase):
    """Computes ``logdet`` in single precision."""

    def _logdet(self) -> Array:
        return jnp.linalg.slogdet(self.A.astype(jnp.float32))[1]


def test_single_precision_output():
    """An operation computed in float32 on float64 data."""
    with pytest.raises(AssertionError):
        check_operator(_Float32Logdet(_square(4)))


# ---------------------------------------------------------------------------
# the PSD claim itself
# ---------------------------------------------------------------------------


@linop
class _IndefinitePSD(PSDLinOp):
    """A symmetric but indefinite matrix, typed as PSD."""

    S: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.S.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.S.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.S, x)

    def _diag(self) -> Array:
        return jnp.diagonal(self.S, axis1=-2, axis2=-1)

    def _to_dense(self) -> Array:
        return self.S


def test_indefinite_psd():
    """A PSD-level type whose matrix has a negative eigenvalue."""
    with pytest.raises(AssertionError):
        check_operator(_IndefinitePSD(jnp.diag(jnp.asarray([2.0, -1.0, 3.0]))))
