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
    Kron,
    KronGeneral,
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
    d3 = jnp.asarray(RNG.uniform(0.5, 3.0, 3))
    A5, A3, A4 = jnp.asarray(_psd(5)), jnp.asarray(_psd(3)), jnp.asarray(_psd(4))
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
        Kron(Diagonal(d3), DensePSD.from_matrix(A4)),
        Kron(DensePSD.from_matrix(A3), Diagonal(d3)),
        Kron(Identity(2), ScaledIdentity(jnp.asarray(3.0), 3)),
        KronGeneral(
            Dense(jnp.asarray(RNG.normal(size=(3, 2)))),
            Dense(jnp.asarray(RNG.normal(size=(4, 5)))),
        ),
        BlockDiag((Kron(Diagonal(d3), Diagonal(d3)), Identity(2))),
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


# ---------------------------------------------------------------------------
# Kronecker products: orientation, rectangularity, and the log-determinant
# ---------------------------------------------------------------------------

def test_kron_matvec_matches_np_kron_at_every_batch_rank():
    """The primary check: matvec against np.kron, batch rank 0, 1 and 2."""
    n_A, n_B = 3, 4
    A, B = _psd(n_A), _psd(n_B)
    op = Kron(DensePSD.from_matrix(jnp.asarray(A)), DensePSD.from_matrix(jnp.asarray(B)))
    assert op.shape == (n_A * n_B, n_A * n_B)
    K = np.kron(A, B)
    for batch in [(), (3,), (2, 3)]:
        x = jnp.asarray(RNG.normal(size=(*batch, n_A * n_B)))
        got = op.matvec(x)
        assert got.shape == (*batch, n_A * n_B)
        np.testing.assert_allclose(
            np.asarray(got),
            np.einsum("ij,...j->...i", K, np.asarray(x)),
            rtol=1e-9,
            atol=1e-9,
        )


def test_kron_orientation_is_not_transposed():
    """A (x) B, not B (x) A -- an error that is silent rather than loud.

    The two factors are the same size on purpose. That is the case where the
    reversed product is shape-valid, so nothing raises; and because a
    Kronecker product of PSD factors is PSD in either order, the result is
    still a legitimate covariance, just the wrong one.
    """
    n = 3
    P, Q = _psd(n), _psd(n)
    op = Kron(DensePSD.from_matrix(jnp.asarray(P)), DensePSD.from_matrix(jnp.asarray(Q)))
    x = jnp.asarray(RNG.normal(size=n * n))

    want = np.kron(P, Q) @ np.asarray(x)
    np.testing.assert_allclose(np.asarray(op.matvec(x)), want, rtol=1e-9, atol=1e-9)

    reverse = np.kron(Q, P) @ np.asarray(x)
    assert np.linalg.eigvalsh(np.kron(Q, P)).min() > 0   # ... still PD, so silent
    assert not np.allclose(reverse, want)                # ... and wrong


def test_kron_to_dense_is_pinned_independently_of_matvec():
    """Both code paths need pinning to np.kron, not just to each other.

    ``check_operator`` compares ``matvec`` against ``to_dense``, so a pair that
    reversed the factors *consistently* would agree with each other and pass.
    That check constrains consistency; only a comparison against an external
    reference constrains orientation.
    """
    A, B = _psd(3), _psd(4)
    op = Kron(DensePSD.from_matrix(jnp.asarray(A)), DensePSD.from_matrix(jnp.asarray(B)))
    np.testing.assert_allclose(
        np.asarray(op.to_dense()), np.kron(A, B), rtol=1e-9, atol=1e-9
    )


def test_kron_general_matvec_matches_np_kron_when_rectangular():
    """The square implementation does not generalize to this case.

    The operand's core shape is ``(k_A, k_B)`` and the result's is
    ``(n_A, n_B)``. An implementation that reshapes both alike -- which is all
    the square case needs -- cannot express a rectangular product at all.
    """
    n_A, k_A, n_B, k_B = 3, 2, 4, 5
    A = RNG.normal(size=(n_A, k_A))
    B = RNG.normal(size=(n_B, k_B))
    op = KronGeneral(Dense(jnp.asarray(A)), Dense(jnp.asarray(B)))
    assert op.shape == (n_A * n_B, k_A * k_B)
    assert (k_A, k_B) != (n_A, n_B)             # the two reshapes really do differ

    K = np.kron(A, B)
    for batch in [(), (3,), (2, 3)]:
        x = jnp.asarray(RNG.normal(size=(*batch, k_A * k_B)))
        got = op.matvec(x)
        assert got.shape == (*batch, n_A * n_B)
        np.testing.assert_allclose(
            np.asarray(got),
            np.einsum("ij,...j->...i", K, np.asarray(x)),
            rtol=1e-9,
            atol=1e-9,
        )


def test_kron_logdet_scales_each_factor_by_the_other_factors_size():
    """log det(A (x) B) = n_B log det A + n_A log det B.

    The sizes must differ or the test is vacuous: with n_A == n_B the swapped
    pairing gives the same number, so the error survives.
    """
    n_A, n_B = 3, 5
    A, B = _psd(n_A), _psd(n_B)
    op = Kron(DensePSD.from_matrix(jnp.asarray(A)), DensePSD.from_matrix(jnp.asarray(B)))

    want = np.linalg.slogdet(np.kron(A, B))[1]
    np.testing.assert_allclose(np.asarray(op.logdet()), want, rtol=1e-9)

    swapped = n_A * np.linalg.slogdet(A)[1] + n_B * np.linalg.slogdet(B)[1]
    assert not np.isclose(swapped, want)


def test_kron_diag_follows_the_same_slow_fast_ordering():
    """diag(A (x) B) is the outer product of the diagonals, first factor slow."""
    A, B = _psd(3), _psd(4)
    op = Kron(DensePSD.from_matrix(jnp.asarray(A)), DensePSD.from_matrix(jnp.asarray(B)))
    np.testing.assert_allclose(
        np.asarray(op.diag()), np.diag(np.kron(A, B)), rtol=1e-9, atol=1e-9
    )


def test_kron_cholesky_is_triangular_but_does_not_advertise_a_solve():
    """The Kronecker product of two lower triangles is the Cholesky factor.

    It is genuinely triangular, but ``KronGeneral`` is a plain ``LinOp`` -- its
    factors may be rectangular in the ``factor()`` case -- so it exposes no
    ``solve``. That is why ``whiten`` is implemented on the factors instead of
    inherited from the base, which solves against ``cholesky()``.
    """
    n_A, n_B = 3, 4
    A, B = _psd(n_A), _psd(n_B)
    op = Kron(DensePSD.from_matrix(jnp.asarray(A)), DensePSD.from_matrix(jnp.asarray(B)))

    L = np.asarray(op.cholesky().to_dense())
    np.testing.assert_allclose(L, np.linalg.cholesky(np.kron(A, B)), rtol=1e-8, atol=1e-8)
    assert np.abs(np.triu(L, 1)).max() == 0.0
    assert not hasattr(op.cholesky(), "solve")

    x = jnp.asarray(RNG.normal(size=n_A * n_B))
    np.testing.assert_allclose(
        np.asarray(op.whiten(x)), np.linalg.solve(L, np.asarray(x)), rtol=1e-8, atol=1e-8
    )


def test_kron_factor_is_the_kron_of_the_factors_factors():
    """factor(A (x) B) == factor(A) (x) factor(B), which may be rectangular."""
    A = _psd(3)
    op = Kron(DensePSD.from_matrix(jnp.asarray(A)), Identity(4))
    L = op.factor()
    assert isinstance(L, KronGeneral)
    Ld = np.asarray(L.to_dense())
    np.testing.assert_allclose(Ld @ Ld.T, np.kron(A, np.eye(4)), rtol=1e-8, atol=1e-8)


def test_kron_capabilities_are_conditional_on_its_factors():
    """As for BlockDiag: a Kron solves only if both factors do."""

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

    assert Kron(Identity(2), Identity(3)).supports("solve")
    mixed = Kron(Identity(2), NoSolve(3))
    assert not mixed.supports("solve")
    with pytest.raises(UnsupportedOpError):
        mixed.solve(jnp.ones(6))


def test_kron_rejects_non_psd_and_non_square_factors():
    """Both guards fire in the constructor, before any array work."""

    @operator
    class Oblong(PSDOperator):
        @property
        def shape(self):
            return (3, 4)

        def matvec(self, x):
            return jnp.zeros(3)

        def to_dense(self):
            return jnp.zeros((3, 4))

    with pytest.raises(TypeError, match="PSDOperator"):
        Kron(Dense(jnp.asarray(RNG.normal(size=(3, 3)))), Identity(2))
    with pytest.raises(ValueError, match="square"):
        Kron(Oblong(), Identity(2))


def test_kron_solve_and_whiten_never_form_the_full_matrix():
    """Both are applied factor by factor, so they work at a size dense linear
    algebra could not reach: 500 x 500 factors are a 250000 x 250000 matrix."""
    n = 500
    op = Kron(Diagonal(jnp.full((n,), 2.0)), Diagonal(jnp.full((n,), 8.0)))
    assert op.shape == (n * n, n * n)
    b = jnp.ones(n * n)
    np.testing.assert_allclose(np.asarray(op.solve(b)), np.full(n * n, 1 / 16), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(op.whiten(b)), np.full(n * n, 0.25), rtol=1e-9)
