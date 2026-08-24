"""Conformance and hazard tests for the structured-operator layer."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.linalg import (
    BlockDiag,
    Dense,
    DensePSD,
    Diagonal,
    HStack,
    Identity,
    LinOp,
    Product,
    PSDOperator,
    ScaledIdentity,
    Triangular,
    UnsupportedOpError,
    densify,
    operator,
    static_field,
)
from pyeki.linalg.testing import check_operator

RNG = np.random.default_rng(0)


def _psd(n: int) -> np.ndarray:
    M = RNG.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _instances() -> list[LinOp]:
    d = jnp.asarray(RNG.uniform(0.5, 3.0, 6))
    A5, A3 = jnp.asarray(_psd(5)), jnp.asarray(_psd(3))
    tri = jnp.linalg.cholesky(A5)
    return [
        Identity(6),
        ScaledIdentity(jnp.asarray(2.5), 6),
        Diagonal(d),
        Dense(jnp.asarray(RNG.normal(size=(4, 6)))),
        Triangular(tri, lower=True),
        DensePSD.from_matrix(A5),
        Product((Diagonal(d), Dense(jnp.asarray(RNG.normal(size=(6, 4)))))),
        HStack((Dense(jnp.asarray(RNG.normal(size=(5, 2)))), DensePSD.from_matrix(A5))),
        BlockDiag((Diagonal(d), DensePSD.from_matrix(A3))),
        BlockDiag((Identity(2), ScaledIdentity(jnp.asarray(4.0), 3))),
    ]


@pytest.mark.parametrize("op", _instances(), ids=lambda o: type(o).__name__)
def test_conformance(op):
    check_operator(op)


# ---------------------------------------------------------------------------
# the specific traps adversarial review surfaced
# ---------------------------------------------------------------------------

def test_matvec_contracts_trailing_axis_when_square():
    """`M @ x` contracts the wrong axis for ndim>=2 -- silent when the shapes align.

    The batch size is chosen equal to n on purpose: that is the case where the
    naive `A @ x` is shape-valid and therefore returns a wrong answer with no
    error. With batch != n it merely raises, which is the benign failure.
    """
    n = 4
    A = _psd(n)
    op = DensePSD.from_matrix(jnp.asarray(A))
    x = jnp.asarray(RNG.normal(size=(n, n)))          # n batched vectors of length n
    want = np.einsum("ij,bj->bi", A, np.asarray(x))
    np.testing.assert_allclose(np.asarray(op.matvec(x)), want, rtol=1e-9)

    naive = A @ np.asarray(x)                          # contracts the wrong axis
    assert naive.shape == want.shape                   # ... shape-valid, so silent
    assert not np.allclose(naive, want)                # ... and wrong


def test_hstack_is_not_a_block_column():
    """[A B] splits the input and sums; it does not apply each block to all of x."""
    A = jnp.asarray(RNG.normal(size=(4, 2)))
    B = jnp.asarray(RNG.normal(size=(4, 3)))
    op = HStack((Dense(A), Dense(B)))
    assert op.shape == (4, 5)
    x = jnp.asarray(RNG.normal(size=5))
    want = np.asarray(A) @ np.asarray(x[:2]) + np.asarray(B) @ np.asarray(x[2:])
    np.testing.assert_allclose(np.asarray(op.matvec(x)), want, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(op.to_dense()), np.hstack([np.asarray(A), np.asarray(B)]), rtol=1e-9
    )


def test_factor_may_be_wider_than_the_operator():
    """No k >= n or k <= n constraint: [L U] is (n, n+r)."""
    A = _psd(4)
    U = RNG.normal(size=(4, 2))
    L = HStack((DensePSD.from_matrix(jnp.asarray(A)).factor(), Dense(jnp.asarray(U))))
    assert L.shape == (4, 6)
    Ld = np.asarray(L.to_dense())
    np.testing.assert_allclose(Ld @ Ld.T, A + U @ U.T, rtol=1e-8, atol=1e-8)


def test_blockdiag_capabilities_are_conditional_on_children():
    """A ClassVar cannot express this; `supports` must consult the children."""

    @operator
    class NoSolve(PSDOperator):
        size: int = static_field()

        @property
        def shape(self):
            return (self.size, self.size)

        def matvec(self, x):
            return x

        def to_dense(self):
            return jnp.eye(self.size)

    good = BlockDiag((Identity(2), Identity(3)))
    mixed = BlockDiag((Identity(2), NoSolve(3)))
    assert good.supports("solve")
    assert not mixed.supports("solve")
    with pytest.raises(UnsupportedOpError):
        mixed.solve(jnp.ones(5))


def test_rectangular_operators_have_no_solve_at_the_type_level():
    """`solve` lives on SquareLinOp, so a LinOp does not merely refuse it -- it
    does not have it. That is the point of the three-level split: a factor
    cannot advertise a meaningless inverse."""
    rect = Dense(jnp.asarray(RNG.normal(size=(4, 6))))
    assert not hasattr(rect, "solve")
    assert not rect.supports("solve")
    prod = Product((DensePSD.from_matrix(jnp.asarray(_psd(4))),))
    assert not hasattr(prod, "solve")


def test_unsupported_raises_rather_than_densifying():
    """A square operator lacking a cheap solve raises, and says what to do."""

    @operator
    class BareSquare(PSDOperator):
        size: int = static_field()

        @property
        def shape(self):
            return (self.size, self.size)

        def matvec(self, x):
            return x

        def to_dense(self):
            return jnp.eye(self.size)

    op = BareSquare(4)
    assert not op.supports("solve")
    with pytest.raises(UnsupportedOpError) as exc:
        op.solve(jnp.ones(4))
    assert "densify" in str(exc.value)
    # the explicit escape hatch does work
    np.testing.assert_allclose(np.asarray(densify(op).solve(jnp.ones(4))), np.ones(4))


def test_densify_guard_is_static_and_raises_before_allocating():
    op = Identity(10_000)
    with pytest.raises(ValueError, match="max_n"):
        densify(op, max_n=4096)
    small = densify(Diagonal(jnp.asarray([1.0, 2.0, 3.0])))
    assert isinstance(small, PSDOperator)


def test_undeclared_scalar_field_is_rejected_at_class_definition():
    """An unmarked int field would arrive as a tracer under jit."""
    with pytest.raises(TypeError, match="not marked static"):

        @operator
        class Bad(PSDOperator):
            size: int          # missing static_field()


def test_operators_use_identity_equality_and_are_not_hashable_as_static():
    a = Diagonal(jnp.asarray([1.0, 2.0]))
    assert a == a and not (a == Diagonal(jnp.asarray([1.0, 2.0])))


def test_logdet_is_a_real_jax_scalar_not_a_python_float():
    op = DensePSD.from_matrix(jnp.asarray(_psd(4)))
    ld = op.logdet()
    assert isinstance(ld, jnp.ndarray) and not jnp.iscomplexobj(ld)
    jax.jit(lambda o: o.logdet())(op)      # would fail if float() were called


def test_x64_is_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_dense_psd_factorizes_once_at_construction():
    """The Cholesky is stored, not recomputed per call (lazy caches do not survive)."""
    op = DensePSD.from_matrix(jnp.asarray(_psd(4)))
    assert op.L.shape == (4, 4)
    leaves = jax.tree_util.tree_leaves(op)
    assert len(leaves) == 1 and leaves[0].shape == (4, 4)
