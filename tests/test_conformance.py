"""Conformance tests: every extensible type, through its contract's suite.

Two layers ship a conformance harness, because two layers are open to
extension. One instance per operator class — plus variants whose
capabilities, structure depth, block count, size, or sign differ — runs
through :func:`pyeki.linalg.testing.check_operator`, each paired with a
second instance of the same structure and different values to make up the
family it is applied as under ``vmap``; and every shipped EKI policy runs
through the check for its axis in :mod:`pyeki.eki.testing`, which is the
harness a user's own schedule, update rule or inflation is meant to be run
through.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.eki import (
    AdaptiveESSSchedule,
    AdaptiveMisfitSchedule,
    AdditiveInflation,
    DiscrepancyStop,
    FixedSchedule,
    MultiplicativeInflation,
    PathwiseUpdate,
    TransformUpdate,
)
from pyeki.eki.testing import (
    check_inflation,
    check_schedule,
    check_stopping_rule,
    check_update,
)
from pyeki.linalg import (
    BlockDiag,
    Dense,
    DensePSD,
    DenseSquare,
    Identity,
    LinOp,
    PSDDiagonal,
    PSDLowRank,
    Transposed,
    Triangular,
    block_diag,
    diag_congruence,
    hstack,
    product,
)
from pyeki.linalg.testing import check_operator


def _instances(rng: np.random.Generator) -> list[LinOp]:
    """The conformance instances, with their values drawn from ``rng``.

    Every call builds the same list of types and shapes, so two calls with
    different generators pair each instance with a second, different one of
    identical pytree structure.
    """

    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape))

    def uniform(low, high, n):
        return jnp.asarray(rng.uniform(low, high, n))

    def psd(n: int) -> jnp.ndarray:
        M = rng.normal(size=(n, n))
        return jnp.asarray(M @ M.T + n * np.eye(n))

    def square(n: int) -> jnp.ndarray:
        return jnp.asarray(rng.normal(size=(n, n)) + n * np.eye(n))

    d = uniform(0.5, 3.0, 6)
    well_conditioned = square(5)
    lu_t = jax.scipy.linalg.lu_factor(well_conditioned.T)
    return [
        Identity(6),
        2.5 * Identity(6),  # the scaled identity, via arithmetic
        PSDDiagonal(d),
        Dense(normal(4, 6)),
        Dense(normal(4, 4)),  # square, so a wrong contraction is silent
        DenseSquare(well_conditioned),
        Triangular(jnp.linalg.cholesky(psd(5)), lower=True),
        Triangular(jnp.linalg.cholesky(psd(4)).T, lower=False),
        DensePSD(psd(5)),
        DensePSD(L=jnp.linalg.cholesky(psd(4))),
        # negative pivots, so a logdet that drops the absolute value fails;
        # the triangular factor of a PSD matrix never has one
        Triangular(
            jnp.asarray(
                np.tril(rng.normal(size=(4, 4)), -1) + np.diag([2.0, -3.0, 1.5, -1.0])
            ),
            lower=True,
        ),
        Triangular(
            jnp.asarray(
                np.triu(rng.normal(size=(3, 3)), 1) + np.diag([-2.0, 1.0, 1.5])
            ),
            lower=False,
        ),
        DenseSquare(well_conditioned.at[0].multiply(-1.0)),
        DenseSquare(well_conditioned, lu=lu_t[0], piv=lu_t[1], lu_of_transpose=True),
        # a low-rank PSD operator at each width: thin (singular), square,
        # and wide (generically nonsingular, yet still no solve/whiten)
        PSDLowRank(normal(5, 2)),
        PSDLowRank(normal(4, 4)),
        PSDLowRank(normal(3, 6)),
        # size one, where a batch axis of length n and k = n + 1 = 2 meet
        Identity(1),
        PSDDiagonal(uniform(0.5, 3.0, 1)),
        Dense(normal(1, 3)),
        Dense(normal(3, 1)),
        DenseSquare(-jnp.asarray([[rng.uniform(0.5, 2.0)]])),
        # composites
        product(PSDDiagonal(d), Dense(normal(6, 4))),
        hstack(Dense(normal(5, 2)), DensePSD(psd(5))),
        BlockDiag((Dense(normal(2, 3)), Dense(normal(3, 3)))),
        block_diag(PSDDiagonal(d), DensePSD(psd(3))),
        block_diag(Identity(2), 4.0 * Identity(3)),
        # composites over a block that disclaims solve/whiten/logdet: the
        # capability intersection must survive, and the block-diagonal
        # factor is rectangular because one block's factor is
        block_diag(PSDDiagonal(d), PSDLowRank(normal(5, 2))),
        diag_congruence(PSDLowRank(normal(4, 2)), uniform(0.5, 2, 4)),
        2.5 * PSDLowRank(normal(4, 2)),
        diag_congruence(DensePSD(psd(4)), uniform(0.5, 2, 4)),
        # three or more blocks or factors, so split-point accumulation and
        # the order of composition are exercised
        hstack(Dense(normal(4, 2)), Dense(normal(4, 3)), Dense(normal(4, 1))),
        BlockDiag((Dense(normal(2, 3)), Dense(normal(1, 2)), Dense(normal(3, 1)))),
        block_diag(PSDDiagonal(uniform(0.5, 3.0, 2)), DensePSD(psd(3)), Identity(1)),
        product(Dense(normal(3, 4)), DensePSD(psd(4)), Dense(normal(4, 2))),
        # a square product, and nesting
        product(PSDDiagonal(d), DensePSD(psd(6))),
        block_diag(
            diag_congruence(DensePSD(psd(3)), uniform(0.5, 2, 3)), Identity(2)
        ),
        2.0 * block_diag(PSDDiagonal(uniform(0.5, 3.0, 2)), DensePSD(psd(3))),
        diag_congruence(
            diag_congruence(DensePSD(psd(3)), uniform(0.5, 2, 3)), uniform(0.5, 2, 3)
        ),
        block_diag(2.0 * DensePSD(psd(3)), 3.0 * PSDDiagonal(uniform(0.5, 3.0, 2))),
        # direct view construction, over a symmetric and a non-symmetric
        # operator; only the second shows a matvec/rmatvec swap
        Transposed(DensePSD(psd(4))),
        Transposed(DenseSquare(square(4))),
        # arithmetic-built composites, both signs
        2.0 * DensePSD(psd(4)),
        3.0 * DenseSquare(well_conditioned),
        -2.0 * DenseSquare(well_conditioned),
        2.0 * DenseSquare(square(4)).T,
        1.5 * Dense(normal(3, 5)),
        Dense(normal(3, 5)).T,
    ]


_PAIRS = list(
    zip(
        _instances(np.random.default_rng(0)),
        _instances(np.random.default_rng(1)),
        strict=True,
    )
)


@pytest.mark.parametrize(
    ("op", "other"), _PAIRS, ids=[type(op).__name__ for op, _ in _PAIRS]
)
def test_conformance(op, other):
    check_operator(op, other=other)


def test_densepsd_factor_solves():
    """The factor of a ``DensePSD`` built from its matrix solves exactly
    against the Cholesky factor, in both orientations."""
    rng = np.random.default_rng(2)
    M = rng.normal(size=(4, 4))
    A = M @ M.T + 4 * np.eye(4)
    C = np.linalg.cholesky(A)
    L = DensePSD(jnp.asarray(A)).factor()
    b = jnp.asarray(rng.normal(size=(3, 4)))
    np.testing.assert_allclose(L.to_dense(), C, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        L.solve(b), np.linalg.solve(C, np.asarray(b).T).T, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        L.T.solve(b), np.linalg.solve(C.T, np.asarray(b).T).T, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# every shipped EKI policy, through the harness the layer ships for user ones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "schedule",
    [
        FixedSchedule.uniform(4),
        FixedSchedule.constant(1.0, 1),
        FixedSchedule((0.25, 0.5, 0.25)),
        AdaptiveESSSchedule(),
        AdaptiveESSSchedule(beta_target=None, ess_fraction=0.3, n_bisect=12),
        AdaptiveMisfitSchedule(),
        AdaptiveMisfitSchedule(beta_target=None, divergence_budget=3.0),
    ],
    ids=repr,
)
def test_schedule_conformance(schedule):
    check_schedule(schedule)


@pytest.mark.parametrize("update", [TransformUpdate(), PathwiseUpdate()], ids=repr)
def test_update_conformance(update):
    check_update(update)


@pytest.mark.parametrize(
    "inflation",
    [
        MultiplicativeInflation(1.02),
        MultiplicativeInflation(2.0),
        AdditiveInflation(DensePSD(jnp.eye(3) * 0.05)),
        AdditiveInflation(PSDDiagonal(jnp.full((3,), 0.02))),
    ],
    ids=repr,
)
def test_inflation_conformance(inflation):
    check_inflation(inflation)


@pytest.mark.parametrize(
    "stop", [DiscrepancyStop(), DiscrepancyStop(tau=2.0)], ids=repr
)
def test_stopping_rule_conformance(stop):
    check_stopping_rule(stop)
