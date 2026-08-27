"""Targeted regression and exactness tests for the structured-operator layer.

Conformance lives in ``test_conformance.py``; this file holds one test per
class of bug that produces wrong numbers or silent misbehaviour without
raising, plus exactness checks against closed forms. Do not delete these as
redundant with conformance — they document why the contract's rules exist.
"""
from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.linalg import (
    BlockDiag,
    Dense,
    DensePSD,
    DenseSquare,
    HStack,
    Identity,
    Product,
    PSDBlockDiag,
    PSDDiagCongruence,
    PSDDiagonal,
    PSDLinOp,
    PSDLowRank,
    PSDScaled,
    Scaled,
    SquareLinOp,
    SquareScaled,
    Transposed,
    Triangular,
    UnsupportedOpError,
    block_diag,
    debug_checks,
    dense_matvec,
    densify,
    diag_congruence,
    hstack,
    linop,
    product,
    static_field,
)

RNG = np.random.default_rng(0)


def _psd(n: int) -> np.ndarray:
    M = RNG.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


# ---------------------------------------------------------------------------
# axis contraction and stacking semantics
# ---------------------------------------------------------------------------


def test_matvec_contracts_trailing_axis_when_square():
    """`M @ x` contracts the wrong axis for ndim>=2 -- silent when shapes align.

    The batch size equals n on purpose: that is the case where the naive
    `A @ x` is shape-valid and returns a wrong answer with no error.
    """
    n = 4
    A = _psd(n)
    op = DensePSD.from_matrix(jnp.asarray(A))
    x = jnp.asarray(RNG.normal(size=(n, n)))  # n batched vectors of length n
    want = np.einsum("ij,bj->bi", A, np.asarray(x))
    np.testing.assert_allclose(np.asarray(op.matvec(x)), want, rtol=1e-9)

    naive = A @ np.asarray(x)  # contracts the wrong axis
    assert naive.shape == want.shape  # ... shape-valid, so silent
    assert not np.allclose(naive, want)  # ... and wrong


def test_hstack_is_not_a_block_column():
    """[A B] splits the input and sums; it does not apply each block to all of x."""
    A = jnp.asarray(RNG.normal(size=(4, 2)))
    B = jnp.asarray(RNG.normal(size=(4, 3)))
    op = hstack(Dense(A), Dense(B))
    assert op.shape == (4, 5)
    x = jnp.asarray(RNG.normal(size=5))
    want = np.asarray(A) @ np.asarray(x[:2]) + np.asarray(B) @ np.asarray(x[2:])
    np.testing.assert_allclose(np.asarray(op.matvec(x)), want, rtol=1e-9)


def test_hstack_transpose_is_the_block_column():
    """hstack(...).T concatenates the blocks' transposed applications."""
    A = jnp.asarray(RNG.normal(size=(4, 2)))
    B = jnp.asarray(RNG.normal(size=(4, 3)))
    col = hstack(Dense(A), Dense(B)).T
    assert col.shape == (5, 4)
    y = jnp.asarray(RNG.normal(size=4))
    want = np.concatenate(
        [np.asarray(A).T @ np.asarray(y), np.asarray(B).T @ np.asarray(y)]
    )
    np.testing.assert_allclose(np.asarray(col.matvec(y)), want, rtol=1e-9)


def test_factor_may_be_wider_than_the_operator():
    """No k >= n or k <= n constraint: [L U] is (n, n+r)."""
    A = _psd(4)
    U = RNG.normal(size=(4, 2))
    L = hstack(DensePSD.from_matrix(jnp.asarray(A)).factor(), Dense(jnp.asarray(U)))
    assert L.shape == (4, 6)
    Ld = np.asarray(L.to_dense())
    np.testing.assert_allclose(Ld @ Ld.T, A + U @ U.T, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# hierarchy and capabilities
# ---------------------------------------------------------------------------


def test_rectangular_operators_have_no_solve_at_the_type_level():
    """A LinOp does not merely refuse `solve` -- it does not have it, and
    calling it is an AttributeError while supports() answers False."""
    rect = Dense(jnp.asarray(RNG.normal(size=(4, 6))))
    assert not hasattr(rect, "solve")
    assert not rect.supports("solve")
    with pytest.raises(AttributeError):
        rect.solve(jnp.ones(4))


def test_blockdiag_capabilities_are_conditional_on_children():
    """Composite support is per instance: it intersects over the blocks,
    and the public-method gate enforces the same answer."""

    @linop
    class NoSolve(PSDLinOp):
        size: int = static_field()

        @property
        def shape(self):
            return (self.size, self.size)

        @property
        def batch_shape(self):
            return ()

        def _matvec(self, x):
            return x

        def _to_dense(self):
            return jnp.eye(self.size)

    good = block_diag(Identity(2), Identity(3))
    mixed = block_diag(Identity(2), NoSolve(3))
    assert good.supports("solve")
    assert not mixed.supports("solve")
    with pytest.raises(UnsupportedOpError):
        mixed.solve(jnp.ones(5))


def test_derived_operations_report_supported():
    """supports() must not deny working derived operations (the dense-branch
    trap: reporting False would steer callers onto an O(n^3) fallback)."""
    d = PSDDiagonal(jnp.asarray([1.0, 2.0, 3.0]))
    assert d.supports("solve_mat") and d.supports("whiten_mat")
    assert d.supports("matmat") and d.supports("to_dense")
    assert "solve_mat" in d.capabilities()


def test_unknown_capability_name_raises():
    """A typo must not silently answer False."""
    with pytest.raises(ValueError, match="choleksy"):
        Identity(3).supports("choleksy")


def test_unsupported_raises_and_densify_always_provides():
    """An unsupported operation raises with guidance, and the advertised
    fallback works at every level -- including square non-PSD, which
    densifies to the LU-backed DenseSquare."""

    @linop
    class BareSquare(SquareLinOp):
        A: Array

        @property
        def shape(self):
            return (self.A.shape[-1], self.A.shape[-1])

        @property
        def batch_shape(self):
            return tuple(self.A.shape[:-2])

        def _matvec(self, x):
            return jnp.einsum("ij,...j->...i", self.A, x)

        def _rmatvec(self, x):
            return jnp.einsum("ij,...i->...j", self.A, x)

        def _to_dense(self):
            return self.A

    A = RNG.normal(size=(4, 4)) + 4 * np.eye(4)
    op = BareSquare(jnp.asarray(A))
    assert not op.supports("solve")
    with pytest.raises(UnsupportedOpError, match="densify"):
        op.solve(jnp.ones(4))

    dense = densify(op)
    assert isinstance(dense, DenseSquare)
    np.testing.assert_allclose(
        np.asarray(dense.solve(jnp.ones(4))), np.linalg.solve(A, np.ones(4)), rtol=1e-9
    )


def test_densify_guard_is_static_and_raises_before_allocating():
    with pytest.raises(ValueError, match="max_n"):
        densify(Identity(10_000), max_n=4096)
    assert isinstance(densify(PSDDiagonal(jnp.asarray([1.0, 2.0]))), DensePSD)


def test_unsupported_error_pickles_and_rebuilds_from_args():
    """The error crosses worker-process boundaries by pickling."""
    e = UnsupportedOpError("solve", "Foo", ("diag", "logdet"))
    assert str(pickle.loads(pickle.dumps(e))) == str(e)
    assert str(type(e)(*e.args)) == str(e)
    assert "densify" in str(e)


# ---------------------------------------------------------------------------
# PSDLowRank
# ---------------------------------------------------------------------------


def _low_rank_widths() -> list[tuple[int, int]]:
    """(n, k) at every relation: thin, square, wide."""
    return [(5, 2), (4, 4), (3, 6)]


@pytest.mark.parametrize("n,k", _low_rank_widths())
def test_psd_low_rank_to_dense_is_the_outer_product(n, k):
    """to_dense assembles F F.T from the stored array, matching a
    hand-written dense product -- never routed through matvec, which would
    make every dense comparison in the suite compare matvec with itself."""
    F = RNG.normal(size=(n, k))
    op = PSDLowRank(jnp.asarray(F))
    assert op.shape == (n, n)
    np.testing.assert_allclose(np.asarray(op.to_dense()), F @ F.T, rtol=1e-12)


@pytest.mark.parametrize("n,k", _low_rank_widths())
def test_psd_low_rank_factor_round_trips(n, k):
    """factor() hands back the stored factor itself: a (n, k) Dense whose
    L L.T is the operator, with no factorization computed anywhere."""
    F = RNG.normal(size=(n, k))
    op = PSDLowRank(jnp.asarray(F))
    L = op.factor()
    assert type(L) is Dense and L.shape == (n, k)
    np.testing.assert_allclose(np.asarray(L.to_dense()), F, rtol=1e-12)
    Ld = np.asarray(L.to_dense())
    np.testing.assert_allclose(Ld @ Ld.T, np.asarray(op.to_dense()), rtol=1e-12)


@pytest.mark.parametrize("n,k", _low_rank_widths())
def test_psd_low_rank_diag_is_the_rowwise_sum_of_squares(n, k):
    """diag reduces over the factor's trailing axis; reducing over the
    wrong one is shape-valid when k == n and silently wrong."""
    F = RNG.normal(size=(n, k))
    op = PSDLowRank(jnp.asarray(F))
    np.testing.assert_allclose(np.asarray(op.diag()), np.diag(F @ F.T), rtol=1e-12)


@pytest.mark.parametrize("n,k", _low_rank_widths())
def test_psd_low_rank_withholds_solve_whiten_and_logdet_at_every_width(n, k):
    """Capabilities are exactly {"diag", "factor"} regardless of width.

    The k >= n instances are generically nonsingular, so `solve`, `whiten`
    and `logdet` all exist mathematically and are still withheld: support
    is a property of the type, not of the stored shape. This test pins that
    decision so the hooks are not "helpfully" added later.
    """
    op = PSDLowRank(jnp.asarray(RNG.normal(size=(n, k))))
    assert op.capabilities() == frozenset({"diag", "factor"})
    if k >= n:  # ... and the withheld operations are not vacuous here
        assert np.linalg.matrix_rank(np.asarray(op.to_dense())) == n
    for name in ("solve", "solve_mat", "logdet", "whiten", "whiten_mat"):
        assert not op.supports(name)
    with pytest.raises(UnsupportedOpError, match=r"no cheap `solve`"):
        op.solve(jnp.ones(n))
    with pytest.raises(UnsupportedOpError, match=r"no cheap `solve_mat`"):
        op.solve_mat(jnp.ones((n, 2)))
    with pytest.raises(UnsupportedOpError, match=r"no cheap `logdet`"):
        op.logdet()
    with pytest.raises(UnsupportedOpError, match=r"no cheap `whiten`"):
        op.whiten(jnp.ones(n))
    with pytest.raises(UnsupportedOpError, match=r"no cheap `whiten_mat`"):
        op.whiten_mat(jnp.ones((n, 2)))


def test_psd_low_rank_validation_is_not_covered_by_conformance():
    """The tier-2 checks are load-bearing precisely because check_operator
    misses them: an otherwise identical operator with no __post_init__
    passes the whole suite, yet accepts a rank-3 factor (yielding a
    *directly constructed* operator that reports a non-empty batch_shape,
    which the contract forbids) and k = 0 (an empty core axis).
    """
    from pyeki.linalg.testing import check_operator

    @linop
    class Unvalidated(PSDLinOp):  # PSDLowRank minus its __post_init__
        F: Array

        @property
        def shape(self):
            n = self.F.shape[-2]
            return (n, n)

        @property
        def batch_shape(self):
            return tuple(self.F.shape[:-2])

        def _matvec(self, x):
            return dense_matvec(self.F, dense_matvec(self.F.swapaxes(-1, -2), x))

        def _diag(self):
            return jnp.sum(self.F * self.F, axis=-1)

        def _factor(self):
            return Dense(self.F)

        def _to_dense(self):
            return self.F @ self.F.swapaxes(-1, -2)

    check_operator(Unvalidated(jnp.asarray(RNG.normal(size=(5, 2)))))  # passes
    assert Unvalidated(jnp.ones((2, 5, 3))).batch_shape == (2,)  # yet accepts
    assert Unvalidated(jnp.ones((5, 0))).shape == (5, 5)  # ... and this

    # PSDLowRank refuses both at construction.
    with pytest.raises(ValueError, match="vmap"):
        PSDLowRank(jnp.ones((2, 5, 3)))
    with pytest.raises(ValueError, match="positive"):
        PSDLowRank(jnp.ones((5, 0)))


# ---------------------------------------------------------------------------
# operand and constructor validation
# ---------------------------------------------------------------------------


def test_wrong_operand_shapes_raise_instead_of_broadcasting():
    """Structured operators used to fail silently here: Identity returned
    the wrong shape and PSDDiagonal broadcast a length-1 operand."""
    with pytest.raises(ValueError, match="matvec"):
        Identity(6).matvec(jnp.ones(3))
    with pytest.raises(ValueError, match="matvec"):
        PSDDiagonal(jnp.ones(6)).matvec(jnp.ones(1))
    with pytest.raises(ValueError, match="matmat"):
        PSDDiagonal(jnp.ones(6)).matmat(jnp.ones(6))  # rank 1 is not a matrix


def test_constructors_reject_batched_arrays():
    """Every constructor demands exact core rank -- the (J, n)-where-(n,)
    slip constructs nothing. Batched families exist only through pytree
    reconstruction, which bypasses the constructor."""
    with pytest.raises(ValueError, match="vmap"):
        PSDDiagonal(jnp.ones((100, 6)))  # an ensemble where a diagonal was meant
    with pytest.raises(ValueError, match="vmap"):
        Dense(jnp.ones((4, 3, 3)))
    with pytest.raises(ValueError, match="vmap"):
        PSDLowRank(jnp.ones((2, 5, 3)))  # a stack of factors, not one factor
    with pytest.raises(ValueError, match="rank"):
        PSDDiagonal(jnp.asarray(2.0))  # below core rank
    As = jnp.asarray(np.stack([_psd(3) for _ in range(4)]))
    with pytest.raises(ValueError, match="from_matrix"):
        DensePSD.from_matrix(As)


def test_constructors_reject_empty_operators():
    """Shapes are strictly positive: an empty core axis is a construction
    error everywhere, not just in from_matrix classmethods."""
    with pytest.raises(ValueError, match="positive"):
        PSDDiagonal(jnp.zeros((0,)))
    with pytest.raises(ValueError, match="positive"):
        Dense(jnp.zeros((0, 3)))
    with pytest.raises(ValueError, match="positive"):
        Dense(jnp.zeros((3, 0)))
    with pytest.raises(ValueError, match="positive"):
        Triangular(jnp.zeros((0, 0)))
    with pytest.raises(ValueError, match="positive"):
        PSDLowRank(jnp.zeros((5, 0)))  # k = 0: an operator of rank zero
    with pytest.raises(ValueError, match="positive"):
        PSDLowRank(jnp.zeros((0, 5)))
    with pytest.raises(ValueError, match="positive"):
        Identity(0)


def test_unflatten_bypasses_the_constructor():
    """tree_unflatten rebuilds instances without running __init__, so
    batched leaves (vmap exit) and placeholder leaves (JAX internals)
    coexist with strict constructors."""
    op = PSDDiagonal(jnp.ones(3))
    leaves, treedef = jax.tree_util.tree_flatten(op)
    fam = jax.tree_util.tree_unflatten(treedef, [jnp.ones((5, 3))])
    assert isinstance(fam, PSDDiagonal)
    jax.tree_util.tree_unflatten(treedef, [object()])  # sentinels pass too

    # The family is legible and inert: it names itself and refuses
    # direct application with a pointer to vmap.
    assert fam.batch_shape == (5,)
    assert repr(fam) == "vmapped(PSDDiagonal(3, 3), batch=(5,))"
    with pytest.raises(ValueError, match="vmap"):
        fam.matvec(jnp.ones(3))
    with pytest.raises(ValueError, match="vmap"):
        fam.logdet()


def test_inconsistently_stacked_leaves_are_diagnosed_at_batch_shape():
    """A hand-assembled pytree whose leaves disagree on the batch fails at
    the batch_shape property, not by downstream shape wreckage."""
    op = DenseSquare.from_matrix(jnp.asarray(np.eye(3) * 2.0))
    leaves, treedef = jax.tree_util.tree_flatten(op)
    bad = jax.tree_util.tree_unflatten(
        treedef, [jnp.stack([leaf] * (3 if i else 2)) for i, leaf in enumerate(leaves)]
    )
    with pytest.raises(ValueError, match="batch"):
        bad.batch_shape  # noqa: B018 - the property itself raises
    # ... but repr never raises: a raising repr would mask the failure.
    assert "unprintable" in repr(bad)


def test_vmap_over_scalars_builds_a_scaled_family():
    """The vmap exit boundary reconstructs a Scaled family whose base
    leaves are broadcast -- unconstructible when unflatten ran __init__."""
    op = DensePSD.from_matrix(jnp.asarray(_psd(3)))
    cs = jnp.asarray([0.5, 1.0, 2.0])
    xs = jnp.asarray(RNG.normal(size=(3, 3)))
    got = jax.vmap(lambda c, x: (op * c).solve(x))(cs, xs)
    want = np.stack(
        [np.linalg.solve(float(cs[i]) * np.asarray(op.to_dense()), np.asarray(xs[i]))
         for i in range(3)]
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-8)


def test_field_allowlist_rejects_undeclared_non_array_fields():
    """A misclassified field would arrive as a tracer far from the
    declaration; the allowlist rejects it at class definition."""
    with pytest.raises(TypeError, match="static_field"):

        @linop
        class Bad(PSDLinOp):
            size: int  # missing static_field()

    with pytest.raises(TypeError, match="static_field"):

        @linop
        class Bad2(PSDLinOp):
            flag: bool | None  # optional fields are not representable data

    with pytest.raises(TypeError, match="static_field"):

        @linop
        class Bad3(PSDLinOp):
            A: np.ndarray  # store JAX arrays, not NumPy


# ---------------------------------------------------------------------------
# JAX integration
# ---------------------------------------------------------------------------


def test_vmap_family_construct_and_return_roundtrips():
    """A batched family built inside vmap is reconstructed, outside any
    trace, through the storing constructor at the vmap exit boundary."""
    As = jnp.asarray(np.stack([_psd(3) for _ in range(4)]))
    family = jax.vmap(DensePSD.from_matrix)(As)
    xs = jnp.asarray(RNG.normal(size=(4, 3)))
    got = jax.vmap(lambda C, x: C.solve(x))(family, xs)
    want = np.stack(
        [np.linalg.solve(np.asarray(As[i]), np.asarray(xs[i])) for i in range(4)]
    )
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-8)


def test_custom_vjp_sentinel_unflatten_survives_composites():
    """jax.custom_vjp unflattens arguments with bare object() leaves; a
    composite whose validation read a child's shape would crash there."""
    op = block_diag(Identity(2), DensePSD.from_matrix(jnp.asarray(_psd(3))))

    @jax.custom_vjp
    def f(o, x):
        return jnp.sum(o.matvec(x))

    f.defvjp(
        lambda o, x: (f(o, x), o),
        lambda o, g: (None, g * o.matvec(jnp.ones(5))),
    )
    value, grad = jax.value_and_grad(f, argnums=1)(op, jnp.ones(5))
    assert np.isfinite(float(value)) and grad.shape == (5,)


def test_operators_use_identity_equality_and_are_never_static():
    a = PSDDiagonal(jnp.asarray([1.0, 2.0]))
    assert a == a and not (a == PSDDiagonal(jnp.asarray([1.0, 2.0])))
    assert {a: "usable as a dict key"}[a]


def test_logdet_is_a_real_jax_scalar_not_a_python_float():
    op = DensePSD.from_matrix(jnp.asarray(_psd(4)))
    ld = op.logdet()
    assert isinstance(ld, jnp.ndarray) and not jnp.iscomplexobj(ld)
    jax.jit(lambda o: o.logdet())(op)  # would fail if float() were called


def test_x64_is_enabled():
    assert jnp.zeros(1).dtype == jnp.float64


def test_dense_psd_factorizes_once_at_construction():
    """The Cholesky is stored, not recomputed per call (lazy caches do not
    survive tracing)."""
    op = DensePSD.from_matrix(jnp.asarray(_psd(4)))
    leaves = jax.tree_util.tree_leaves(op)
    assert len(leaves) == 1 and leaves[0].shape == (4, 4)


def test_repr_is_shape_based_and_does_not_dump_arrays():
    assert repr(Dense(jnp.zeros((200, 300)))) == "Dense(200, 300)"
    assert repr(Identity(4)) == "Identity(4, 4)"
    assert repr(block_diag(Identity(2), Identity(3))) == "PSDBlockDiag(5, 5)"


# ---------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------


def test_numpy_left_operands_defer_to_the_guided_errors():
    """Without __array_ufunc__ = None, np_array * op silently builds an
    object array of per-element scaled operators."""
    op = PSDDiagonal(jnp.ones(3))
    with pytest.raises(TypeError, match="scalar"):
        np.ones(3) * op
    with pytest.raises(TypeError, match="rmatvec"):
        np.ones(3) @ op
    with pytest.raises(TypeError, match="matvec"):
        op @ np.ones(3)


def test_matmul_composes_operators_via_the_product_factory():
    d = PSDDiagonal(jnp.asarray([1.0, 2.0, 3.0]))
    A = Dense(jnp.asarray(RNG.normal(size=(3, 5))))
    comp = d @ A
    assert isinstance(comp, Product)
    np.testing.assert_allclose(
        np.asarray(comp.to_dense()),
        np.asarray(d.to_dense()) @ np.asarray(A.to_dense()),
        rtol=1e-9,
    )
    with pytest.raises(TypeError, match="@"):
        d * A  # multiplication of operators has a name, and it is @


def test_tempering_scales_with_a_traced_increment():
    """The Scaled consumer: per-step noise Sigma / dbeta with dbeta chosen
    inside a jit-ed step. Whitening the tempered operator multiplies by
    sqrt(dbeta), exactly."""
    R = DensePSD.from_matrix(jnp.asarray(_psd(3)))
    r = jnp.asarray(RNG.normal(size=3))

    @jax.jit
    def whiten_tempered(dbeta):
        return (R / dbeta).whiten(r)

    dbeta = jnp.asarray(0.25)
    np.testing.assert_allclose(
        np.asarray(whiten_tempered(dbeta)),
        np.sqrt(0.25) * np.asarray(R.whiten(r)),
        rtol=1e-12,
    )
    grad = jax.grad(lambda db: jnp.sum((R / db).whiten(r)))(dbeta)
    assert np.isfinite(float(grad))


def test_nested_scalings_fold_into_one_wrapper():
    op = DensePSD.from_matrix(jnp.asarray(_psd(3)))
    q = 2.0 * (op * 3.0) / 4.0
    assert isinstance(q, PSDScaled) and not isinstance(q.op, Scaled)
    np.testing.assert_allclose(
        np.asarray(q.to_dense()), 1.5 * np.asarray(op.to_dense()), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# transposition
# ---------------------------------------------------------------------------


def test_default_transpose_is_an_unwrapping_view():
    rect = Dense(jnp.asarray(RNG.normal(size=(3, 5))))
    assert isinstance(rect.T, Transposed) and rect.T.T is rect


def test_structured_transposes_keep_their_capabilities():
    """Triangular.T and DenseSquare.T stay solvable; the default view would
    not be."""
    L = jnp.linalg.cholesky(jnp.asarray(_psd(4)))
    upper = Triangular(L, lower=True).T
    assert isinstance(upper, Triangular) and upper.supports("solve")
    b = jnp.asarray(RNG.normal(size=4))
    np.testing.assert_allclose(
        np.asarray(upper.solve(b)), np.linalg.solve(np.asarray(L).T, np.asarray(b)),
        rtol=1e-9,
    )

    sq = DenseSquare.from_matrix(jnp.asarray(RNG.normal(size=(4, 4)) + 4 * np.eye(4)))
    np.testing.assert_allclose(
        np.asarray(sq.T.solve(b)),
        np.linalg.solve(np.asarray(sq.to_dense()).T, np.asarray(b)),
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# value preconditions and debug mode
# ---------------------------------------------------------------------------


def test_debug_checks_catch_value_violations_eagerly():
    with debug_checks():
        with pytest.raises(ValueError, match="positive"):
            PSDDiagonal(jnp.asarray([1.0, -2.0]))
        with pytest.raises(ValueError, match="positive definite|finite"):
            DensePSD.from_matrix(jnp.asarray(-np.eye(3)))
        with pytest.raises(ValueError, match="singular"):
            DenseSquare.from_matrix(jnp.eye(3) * 0.0 + jnp.diag(jnp.zeros(3)))
        with pytest.raises(ValueError, match="symmetric"):
            DensePSD.from_matrix(jnp.asarray([[4.0, 0.5], [1.5, 3.0]]))
        with pytest.raises(ValueError, match="nonzero"):
            0.0 * DenseSquare.from_matrix(jnp.eye(3) * 2.0)
        # ... and tracers are exempt, so jit-ed code is unaffected.
        jax.jit(lambda d: PSDDiagonal(d).logdet())(jnp.asarray([1.0, 2.0]))


def test_value_violations_are_silent_nan_outside_debug_mode():
    assert bool(jnp.isnan(PSDDiagonal(jnp.asarray([1.0, -2.0])).logdet()))


# ---------------------------------------------------------------------------
# factories and composite anatomy
# ---------------------------------------------------------------------------


def test_factories_choose_the_most_capable_class():
    psd_blocks = block_diag(Identity(2), PSDDiagonal(jnp.ones(3)))
    assert isinstance(psd_blocks, PSDBlockDiag)
    mixed = block_diag(Identity(2), Dense(jnp.ones((3, 3))))
    assert isinstance(mixed, BlockDiag) and not isinstance(mixed, PSDLinOp)


def test_factories_unwrap_single_operands_and_reject_empty():
    op = Identity(3)
    assert block_diag(op) is op and product(op) is op and hstack(op) is op
    for factory in (block_diag, product, hstack):
        with pytest.raises(ValueError, match="at least one"):
            factory()


def test_block_anatomy_is_exposed_for_localization():
    """Consumers align sub-vectors with blocks through blocks/block_shapes."""
    a, b = PSDDiagonal(jnp.ones(2)), DensePSD.from_matrix(jnp.asarray(_psd(3)))
    op = block_diag(a, b)
    assert op.blocks == (a, b)
    assert op.block_shapes == ((2, 2), (3, 3))


def test_diag_congruence_is_taper_reciprocal_inflation():
    """Inflating noise by 1/taper is the congruence with s = 1/sqrt(taper),
    and its whitening stays as cheap as the original's."""
    R = DensePSD.from_matrix(jnp.asarray(_psd(4)))
    taper = jnp.asarray(RNG.uniform(0.2, 1.0, 4))
    inflated = diag_congruence(R, 1.0 / jnp.sqrt(taper))
    assert isinstance(inflated, PSDDiagCongruence)
    want = np.asarray(R.to_dense()) / np.sqrt(
        np.outer(np.asarray(taper), np.asarray(taper))
    )
    np.testing.assert_allclose(np.asarray(inflated.to_dense()), want, rtol=1e-12)
    # exact logdet: 2 sum log s + logdet R
    np.testing.assert_allclose(
        float(inflated.logdet()),
        float(-jnp.sum(jnp.log(taper)) + R.logdet()),
        rtol=1e-12,
    )


def test_diag_congruence_size_mismatch_raises():
    with pytest.raises(ValueError, match="scale"):
        diag_congruence(Identity(3), jnp.ones(4))


# ---------------------------------------------------------------------------
# error paths and harness self-tests
# ---------------------------------------------------------------------------


def test_structural_error_paths_raise_loudly():
    """Every documented construction error actually fires."""
    A22, A23 = jnp.ones((2, 2)), jnp.ones((2, 3))
    with pytest.raises(ValueError, match="shape mismatch"):
        Product((Dense(A23), Dense(A23)))
    with pytest.raises(ValueError, match="row count"):
        HStack((Dense(A22), Dense(jnp.ones((3, 2)))))
    with pytest.raises(TypeError, match="tuple"):
        Product([Dense(A22)])  # list, not tuple
    with pytest.raises(ValueError, match="at least one"):
        Product(())
    with pytest.raises(TypeError, match="PSDLinOp"):
        PSDBlockDiag((Dense(A22),))
    with pytest.raises(TypeError, match="LinOp"):
        Transposed(3)
    with pytest.raises(TypeError, match="LinOp"):
        Scaled("x", jnp.asarray(1.0))
    with pytest.raises(TypeError, match="SquareLinOp"):
        SquareScaled(Dense(A23), jnp.asarray(1.0))
    with pytest.raises(TypeError, match="PSDLinOp"):
        PSDScaled(DenseSquare.from_matrix(jnp.eye(2) * 2), jnp.asarray(1.0))
    with pytest.raises(TypeError, match="PSDLinOp"):
        PSDDiagCongruence(Dense(A22), jnp.ones(2))
    with pytest.raises(ValueError, match="scale length"):
        PSDDiagCongruence(Identity(3), jnp.ones(4))
    with pytest.raises(TypeError, match="operators"):
        block_diag(Identity(2), "not an operator")
    with pytest.raises(TypeError, match="PSDLinOp"):
        diag_congruence(Dense(A22), jnp.ones(2))
    with pytest.raises(TypeError, match="int"):
        Identity("3")
    with pytest.raises(ValueError, match="positive"):
        Identity(-1)
    with pytest.raises(ValueError, match="square"):
        Triangular(jnp.ones((2, 3)))
    with pytest.raises(TypeError, match="bool"):
        Triangular(jnp.eye(2), lower=1)
    with pytest.raises(TypeError):
        Identity(2) / Identity(2)


def test_densify_rectangular_returns_dense():
    rect = (2.0 * Dense(jnp.asarray(RNG.normal(size=(3, 5))))).T
    dense = densify(rect)
    assert type(dense) is Dense and dense.shape == (5, 3)
    np.testing.assert_allclose(
        np.asarray(dense.to_dense()), np.asarray(rect.to_dense()), rtol=1e-12
    )


def test_unresolvable_annotation_is_a_guided_type_error():
    with pytest.raises(TypeError, match="resolve"):

        @linop
        class Bad(PSDLinOp):  # noqa: F811 - deliberately discarded
            x: NoSuchType  # noqa: F821 - the point of the test


def test_check_operator_rejects_broken_operators():
    """The harness is itself load-bearing: it must fail on operators that
    violate the contract, not only pass on ones that satisfy it."""
    from pyeki.linalg.testing import check_operator

    @linop
    class RoutedToDense(PSDLinOp):
        diagonal: Array

        @property
        def shape(self):
            n = self.diagonal.shape[-1]
            return (n, n)

        @property
        def batch_shape(self):
            return tuple(self.diagonal.shape[:-1])

        def _matvec(self, x):
            return self.diagonal * x

        def _to_dense(self):
            return self.matmat(jnp.eye(self.shape[0]))  # forbidden routing

    with pytest.raises(AssertionError, match="to_dense"):
        check_operator(RoutedToDense(jnp.asarray([1.0, 2.0, 3.0])))

    @linop
    class SlightlyWrong(PSDLinOp):
        diagonal: Array

        @property
        def shape(self):
            n = self.diagonal.shape[-1]
            return (n, n)

        @property
        def batch_shape(self):
            return tuple(self.diagonal.shape[:-1])

        def _matvec(self, x):
            return self.diagonal * x + 1e-4  # off by a constant

        def _to_dense(self):
            return jnp.diag(self.diagonal)

    with pytest.raises(AssertionError, match="matvec"):
        check_operator(SlightlyWrong(jnp.asarray([1.0, 2.0, 3.0])))
