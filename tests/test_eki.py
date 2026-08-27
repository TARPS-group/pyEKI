"""Conformance and regression tests for the Ensemble Kalman Inversion layer.

The file has two sections. The first works through the twenty-six numbered
conformance obligations of the "Ensemble Kalman Inversion contract"; the
second holds one targeted regression test per class of silent failure that
contract names, under the same do-not-delete rule as the two layers below —
each documents why a rule of the contract exists, and deleting it as redundant
loses that.

Two rules govern the reference throughout:

- **The dense reference is hand-written here.** Plain dense linear algebra
  over means, anomalies and materialized operators, never routed through
  ``pyeki.eki`` or ``pyeki.gauss``, so every comparison is between two
  genuinely independent paths.
- **Exactness tests compare against closed forms**, at a tolerance of a few
  machine epsilons times the natural scale of the quantity, never a tolerance
  chosen to make the test pass.
"""
from __future__ import annotations

import logging
import warnings

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
    EKIError,
    EKIResult,
    EKIState,
    Evaluation,
    FixedSchedule,
    HistoryRecord,
    INTERRUPTED,
    MultiplicativeInflation,
    PathwiseUpdate,
    SCHEDULE_EXHAUSTED,
    STOPPING_RULE,
    TransformUpdate,
    advance,
    apply,
    effective_sample_size,
    evaluate,
    iterate,
    misfits,
    repair_failed_members,
    run,
)
from pyeki.gauss import EnsembleJoint, Gaussian
from pyeki.linalg import (
    DensePSD,
    PSDDiagonal,
    PSDLowRank,
    UnsupportedOpError,
    block_diag,
    debug_checks,
)

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# fixtures: an affine problem, and its dense posterior
# ---------------------------------------------------------------------------


def _psd(n: int, seed: int = 3) -> np.ndarray:
    """A well-conditioned dense PSD matrix, as a NumPy array."""
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _exact_moment_ensemble(J: int, mu: np.ndarray, F: np.ndarray) -> np.ndarray:
    """An ensemble whose empirical moments are exactly ``mu`` and ``F @ F.T``.

    The QR-of-ones construction: take the complete QR of the all-ones vector
    in R^J, let E be its next k columns — orthonormal and each orthogonal to
    the ones vector — and set the members to ``mu + sqrt(J - 1) E F^T``. The
    rows then have mean ``mu`` and empirical covariance ``F F^T`` with the
    package's J - 1 divisor. Only ``J >= k + 1`` binds.
    """
    k = F.shape[1]
    assert J >= k + 1, "the construction needs J >= k + 1"
    Q, _ = np.linalg.qr(np.ones((J, 1)), mode="complete")
    E = Q[:, 1 : k + 1]
    return mu + np.sqrt(J - 1) * E @ F.T


class _AffineProblem:
    """An affine forward model, its prior, and the dense posterior it implies."""

    def __init__(self, P: int = 3, N: int = 5, J: int = 12, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.P, self.N, self.J = P, N, J
        self.G = rng.normal(size=(N, P))
        self.m0 = rng.normal(size=P)
        self.C0 = _psd(P, seed=seed + 1)
        self.R = _psd(N, seed=seed + 2)
        self.y = rng.normal(size=N)
        self.factor = np.linalg.cholesky(self.C0)
        self.members = _exact_moment_ensemble(J, self.m0, self.factor)
        self.noise_cov = DensePSD.from_matrix(jnp.asarray(self.R))
        self.calls: list[np.ndarray] = []

    def forward(self, u):
        """The model, recording every input it is given."""
        u = jnp.asarray(u)
        self.calls.append(np.asarray(u))
        return u @ jnp.asarray(self.G).T

    def state(self, seed: int = 0) -> EKIState:
        return EKIState(jnp.asarray(self.members), 0.0, 0, jax.random.key(seed))

    def posterior(self, level: float = 1.0):
        """The exact Gaussian posterior at tempering level ``level``."""
        prior_precision = np.linalg.inv(self.C0)
        data_precision = self.G.T @ np.linalg.solve(self.R, self.G)
        cov = np.linalg.inv(prior_precision + level * data_precision)
        mean = cov @ (
            prior_precision @ self.m0
            + level * self.G.T @ np.linalg.solve(self.R, self.y)
        )
        return mean, cov


def _moments(members) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and covariance, divisor J - 1, of a row-wise ensemble."""
    members = np.asarray(members)
    J = members.shape[0]
    A = members - members.mean(axis=0)
    return members.mean(axis=0), A.T @ A / (J - 1)


#: Ladders whose increments sum to 1 *exactly* in binary floating point, which
#: is the clause of the exactness claim that ``FixedSchedule.uniform`` misses.
_EXACT_LADDERS = [
    (1.0,),
    (0.5, 0.5),
    (0.25, 0.25, 0.25, 0.25),
    (0.125,) * 8,
    (0.5, 0.25, 0.25),
    (0.75, 0.125, 0.0625, 0.0625),
    (0.0625, 0.4375, 0.25, 0.25),
]


# ===========================================================================
# Section 1 -- the twenty-six conformance obligations
# ===========================================================================


@pytest.mark.parametrize("increments", _EXACT_LADDERS)
def test_1_the_ladder_telescopes_to_one_shot_conditioning(increments):
    """A ladder summing to 1 reproduces the exact posterior, to floating point.

    The layer's central correctness property. Per-step precisions add: with an
    affine forward model, conditioning with ``R / increment`` at each rung
    contributes ``increment * G^T R^-1 G`` to the posterior precision, so a
    ladder summing to 1 composes to one-shot conditioning at beta = 1.

    Every clause of the claim is load-bearing and is supplied here: an affine
    model, a Gaussian prior, an ensemble whose empirical moments equal the
    prior's exactly, the square-root update, no inflation, no failed members,
    and increments summing *exactly* to 1.
    """
    problem = _AffineProblem()
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=FixedSchedule(increments),
    )
    assert result.n_steps == len(increments)
    got_mean, got_cov = _moments(result.ensemble)
    want_mean, want_cov = problem.posterior()

    scale = max(np.abs(want_mean).max(), np.abs(want_cov).max())
    tolerance = 1e3 * EPS * scale
    assert np.abs(got_mean - want_mean).max() < tolerance
    assert np.abs(got_cov - want_cov).max() < tolerance


def test_2_the_level_mis_scaling_is_caught_by_that_tolerance():
    """``R / beta`` instead of ``R / increment`` fails test 1 by a wide margin.

    The layer's signature silent failure: it raises nothing and produces a
    plausible-looking posterior, wrong by an amount that *grows with ladder
    length*. On a uniform T-rung ladder the level form accumulates
    ``sum_t t/T = (T + 1)/2`` times the data precision instead of one.

    The mis-scaling is written out locally, against ``EnsembleJoint``
    directly, so this test measures the bug rather than the implementation.
    Two assertions: the mis-scaled ladder lands on the exact posterior at
    level ``(T + 1)/2`` — the documented, problem-independent form — and its
    disagreement with the correct answer is orders of magnitude above test 1's
    tolerance, so that test 1 is known to be tight enough to catch it.
    """
    problem = _AffineProblem()
    y = jnp.asarray(problem.y)
    for n_steps in (2, 5, 10):
        members = jnp.asarray(problem.members)
        beta = 0.0
        for _ in range(n_steps):
            beta += 1.0 / n_steps
            predictions = problem.forward(members)
            members = EnsembleJoint(
                u_samples=members, v_samples=predictions
            ).transform_update(y, problem.noise_cov / beta)  # the bug: level

        got_mean, got_cov = _moments(members)
        want_mean, want_cov = problem.posterior(level=(n_steps + 1) / 2)
        scale = max(np.abs(want_mean).max(), np.abs(want_cov).max())
        assert np.abs(got_mean - want_mean).max() < 1e3 * EPS * scale
        assert np.abs(got_cov - want_cov).max() < 1e3 * EPS * scale

        correct_mean, correct_cov = problem.posterior()
        correct_scale = max(np.abs(correct_mean).max(), np.abs(correct_cov).max())
        disagreement = max(
            np.abs(got_mean - correct_mean).max(),
            np.abs(got_cov - correct_cov).max(),
        )
        assert disagreement > 1e9 * EPS * correct_scale
