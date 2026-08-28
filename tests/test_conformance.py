"""Conformance tests: every extensible type, through its contract's suite.

Two layers ship a conformance harness, because two layers are open to
extension. One instance per operator class — plus variants whose
capabilities, structure depth, block count, or sign differ — runs through
:func:`pyeki.linalg.testing.check_operator`; and every shipped EKI policy runs
through the check for its axis in :mod:`pyeki.eki.testing`, which is the
harness a user's own schedule, update rule or inflation is meant to be run
through.
"""
from __future__ import annotations

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

RNG = np.random.default_rng(0)


def _psd(n: int) -> jnp.ndarray:
    M = RNG.normal(size=(n, n))
    return jnp.asarray(M @ M.T + n * np.eye(n))


def _instances() -> list[LinOp]:
    d = jnp.asarray(RNG.uniform(0.5, 3.0, 6))
    well_conditioned = jnp.asarray(RNG.normal(size=(5, 5)) + 5 * np.eye(5))
    return [
        Identity(6),
        2.5 * Identity(6),  # the scaled identity, via arithmetic
        PSDDiagonal(d),
        Dense(jnp.asarray(RNG.normal(size=(4, 6)))),
        DenseSquare.from_matrix(well_conditioned),
        Triangular(jnp.linalg.cholesky(_psd(5)), lower=True),
        Triangular(jnp.linalg.cholesky(_psd(4)).T, lower=False),
        DensePSD.from_matrix(_psd(5)),
        # a low-rank PSD operator at each width: thin (singular), square,
        # and wide (generically nonsingular, yet still no solve/whiten)
        PSDLowRank(jnp.asarray(RNG.normal(size=(5, 2)))),
        PSDLowRank(jnp.asarray(RNG.normal(size=(4, 4)))),
        PSDLowRank(jnp.asarray(RNG.normal(size=(3, 6)))),
        # composites
        product(PSDDiagonal(d), Dense(jnp.asarray(RNG.normal(size=(6, 4))))),
        hstack(
            Dense(jnp.asarray(RNG.normal(size=(5, 2)))), DensePSD.from_matrix(_psd(5))
        ),
        BlockDiag(
            (
                Dense(jnp.asarray(RNG.normal(size=(2, 3)))),
                Dense(jnp.asarray(RNG.normal(size=(3, 3)))),
            )
        ),
        block_diag(PSDDiagonal(d), DensePSD.from_matrix(_psd(3))),
        block_diag(Identity(2), 4.0 * Identity(3)),
        # composites over a block that disclaims solve/whiten/logdet: the
        # capability intersection must survive, and the block-diagonal
        # factor is rectangular because one block's factor is
        block_diag(PSDDiagonal(d), PSDLowRank(jnp.asarray(RNG.normal(size=(5, 2))))),
        diag_congruence(
            PSDLowRank(jnp.asarray(RNG.normal(size=(4, 2)))),
            jnp.asarray(RNG.uniform(0.5, 2, 4)),
        ),
        2.5 * PSDLowRank(jnp.asarray(RNG.normal(size=(4, 2)))),
        diag_congruence(
            DensePSD.from_matrix(_psd(4)), jnp.asarray(RNG.uniform(0.5, 2, 4))
        ),
        # three or more blocks, so split-point accumulation is exercised
        hstack(
            Dense(jnp.asarray(RNG.normal(size=(4, 2)))),
            Dense(jnp.asarray(RNG.normal(size=(4, 3)))),
            Dense(jnp.asarray(RNG.normal(size=(4, 1)))),
        ),
        BlockDiag(
            (
                Dense(jnp.asarray(RNG.normal(size=(2, 3)))),
                Dense(jnp.asarray(RNG.normal(size=(1, 2)))),
                Dense(jnp.asarray(RNG.normal(size=(3, 1)))),
            )
        ),
        block_diag(
            PSDDiagonal(jnp.asarray(RNG.uniform(0.5, 3.0, 2))),
            DensePSD.from_matrix(_psd(3)),
            Identity(1),
        ),
        # a square product, and nesting
        product(PSDDiagonal(d), DensePSD.from_matrix(_psd(6))),
        block_diag(
            diag_congruence(
                DensePSD.from_matrix(_psd(3)), jnp.asarray(RNG.uniform(0.5, 2, 3))
            ),
            Identity(2),
        ),
        Transposed(DensePSD.from_matrix(_psd(4))),  # direct view construction
        # arithmetic-built composites, both signs
        2.0 * DensePSD.from_matrix(_psd(4)),
        3.0 * DenseSquare.from_matrix(well_conditioned),
        -2.0 * DenseSquare.from_matrix(well_conditioned),
        1.5 * Dense(jnp.asarray(RNG.normal(size=(3, 5)))),
        Dense(jnp.asarray(RNG.normal(size=(3, 5)))).T,
    ]


@pytest.mark.parametrize("op", _instances(), ids=lambda o: type(o).__name__)
def test_conformance(op):
    check_operator(op)


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
        AdditiveInflation(DensePSD.from_matrix(jnp.eye(3) * 0.05)),
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
