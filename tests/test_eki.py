"""Conformance and regression tests for the Ensemble Kalman Inversion layer.

The file has two sections. The first works through the thirty numbered
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

import dataclasses
import logging
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.eki import (
    INTERRUPTED,
    SCHEDULE_EXHAUSTED,
    STOPPING_RULE,
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
    MultiplicativeInflation,
    PathwiseUpdate,
    TransformUpdate,
    advance,
    assimilate,
    effective_sample_size,
    evaluate,
    iterate,
    misfits,
    repair_failed_members,
    run,
)
from pyeki.eki.driver import _check_predictions
from pyeki.gauss import EmpiricalJoint, Gaussian
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

    def __init__(
        self,
        P: int = 3,
        N: int = 5,
        J: int = 12,
        seed: int = 7,
        prior_rank: int | None = None,
        noise_scale: float = 1.0,
    ):
        rng = np.random.default_rng(seed)
        self.P, self.N, self.J = P, N, J
        self.G = rng.normal(size=(N, P))
        self.m0 = rng.normal(size=P)
        self.C0 = _psd(P, seed=seed + 1)
        self.R = noise_scale * _psd(N, seed=seed + 2)
        self.y = rng.normal(size=N)
        self.factor = np.linalg.cholesky(self.C0)
        if prior_rank is not None:
            self.factor = self.factor[:, :prior_rank]
            self.C0 = self.factor @ self.factor.T
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
# Section 1 -- the thirty conformance obligations
# ===========================================================================


@pytest.mark.parametrize("increments", _EXACT_LADDERS)
def test_1_the_ladder_telescopes_to_one_shot_conditioning(increments):
    """A ladder summing to 1 reproduces the exact posterior, to floating point.

    The layer's central correctness property. Per-step precisions add: with an
    affine forward model, conditioning with ``R / increment`` at each step
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
    assert result.n_evaluations == len(increments)
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
    length*. On a uniform T-step ladder the level form accumulates
    ``sum_t t/T = (T + 1)/2`` times the data precision instead of one.

    The mis-scaling is written out locally, against ``EmpiricalJoint``
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
            members = EmpiricalJoint(
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


def test_3_the_stochastic_update_composes_to_the_same_posterior():
    """``PathwiseUpdate`` reproduces a dense reference, and telescopes in mean.

    Two halves. Elementwise, for a fixed key, a ladder reproduces a
    hand-written dense perturbed-observation reference applied step by step.
    And its posterior moments match the one-shot posterior *in expectation*,
    tested as a mean over many keys with a tolerance derived from the
    ``K R K^T / J`` scale rather than tuned.
    """
    problem = _AffineProblem(J=16)
    y = jnp.asarray(problem.y)
    increments = (0.25, 0.25, 0.5)

    # -- elementwise, against a dense reference --------------------------------
    state = problem.state(seed=5)
    members = np.asarray(problem.members)
    key = state.key
    for increment in increments:
        key_next, _, key_update = jax.random.split(key, 3)
        predictions = members @ problem.G.T
        members = _dense_pathwise_step(
            members, predictions, problem, y=problem.y,
            increment=increment, key=key_update,
        )
        key = key_next
    result = run(
        state,
        problem.forward,
        y,
        problem.noise_cov,
        schedule=FixedSchedule(increments),
        update=PathwiseUpdate(),
    )
    scale = np.abs(members).max()
    assert np.abs(np.asarray(result.ensemble) - members).max() < 1e4 * EPS * scale

    # -- in expectation, over many keys ---------------------------------------
    want_mean, want_cov = problem.posterior()
    gain = want_cov @ problem.G.T @ np.linalg.inv(problem.R)
    noise_scale = np.sqrt(np.diag(gain @ problem.R @ gain.T).max() / problem.J)

    n_keys = 400
    means = np.zeros((n_keys, problem.P))
    for index in range(n_keys):
        state = problem.state(seed=1000 + index)
        drawn = run(
            state,
            problem.forward,
            y,
            problem.noise_cov,
            schedule=FixedSchedule(increments),
            update=PathwiseUpdate(),
        )
        means[index] = np.asarray(drawn.ensemble).mean(axis=0)
    # Three standard errors of the Monte Carlo mean, from the KRK'/J scale.
    assert np.abs(means.mean(axis=0) - want_mean).max() < 3 * noise_scale / np.sqrt(
        n_keys
    )


def _dense_pathwise_step(members, predictions, problem, *, y, increment, key):
    """One perturbed-observation step, in plain dense linear algebra."""
    J = members.shape[0]
    Au = members - members.mean(axis=0)
    Av = predictions - predictions.mean(axis=0)
    R = problem.R / increment
    gain = (Au.T @ Av / (J - 1)) @ np.linalg.inv(Av.T @ Av / (J - 1) + R)
    W = _recovered_whitener(problem.noise_cov / increment, problem.N)
    eps = np.asarray(jax.random.normal(key, (J, problem.N)))
    perturbed = y - predictions - eps @ np.linalg.inv(W).T
    return members + perturbed @ gain.T


def _recovered_whitener(noise_cov, n: int) -> np.ndarray:
    """The operator's own whitener, recovered by whitening the identity."""
    return np.asarray(noise_cov.whiten(jnp.eye(n))).T


def test_4_fixed_schedule_takes_its_increments_and_completes_under_its_own_bound():
    """Exactly its increments, exactly T evaluations, and no off-by-one bound.

    The executable form of the exhaustion-before-bound ordering: a schedule
    that exhausts at step T must complete under ``max_steps == T``, which is
    the value a caller naturally passes. Checking the bound first would turn
    every such run into an ``EKIError`` on its final re-entry.
    """
    problem = _AffineProblem()
    increments = (0.1, 0.4, 0.2, 0.3)
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=FixedSchedule(increments),
        max_steps=len(increments),
    )
    assert result.status == SCHEDULE_EXHAUSTED
    assert result.n_evaluations == len(increments)
    assert len(problem.calls) == len(increments)
    assert np.allclose(np.asarray(result.stacked.increment), increments)
    assert float(result.beta) == pytest.approx(sum(increments), abs=8 * EPS)


def test_4_a_schedule_returning_none_ends_the_run_with_a_terminal_record():
    """``next_increment`` may end the ladder on evidence only the evaluation has."""

    class _GiveUpAfterTwo:
        n_steps = None
        beta_target = None

        def next_increment(self, evaluation):
            return None if evaluation.step >= 2 else 0.25

    problem = _AffineProblem(J=10)
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=_GiveUpAfterTwo(),
    )
    assert result.status == SCHEDULE_EXHAUSTED
    assert result.n_evaluations == 3
    assert float(result.stacked.increment[-1]) == 0.0
    assert float(result.stacked.beta_next[-1]) == float(result.stacked.beta[-1])
    # The literal float(J), not effective_sample_size(misfits, 0.0): J = 12 is
    # one of the sizes where exp(log J) == J exactly, so this fixture uses a
    # size where the two differ and the pin has content.
    assert problem.J == 10
    assert float(np.exp(np.log(problem.J))) != float(problem.J)
    assert float(result.stacked.ess[-1]) == float(problem.J)


@pytest.mark.parametrize(
    "schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule]
)
def test_4_both_adaptive_schedules_reach_their_budget_without_exceeding_it(
    schedule_type,
):
    """Arrive at ``beta_target`` within ``budget_tol``, never above it."""
    problem = _AffineProblem()
    schedule = schedule_type(beta_target=1.0)
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=schedule,
    )
    beta = float(result.beta)
    assert beta <= 1.0 + 8 * EPS
    assert beta >= 1.0 - 1e-12
    assert np.all(np.asarray(result.stacked.increment) > 0.0)
    # The exhaustion check is true on arrival, so a second run is a no-op.
    again = run(
        result.state,
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=schedule,
    )
    assert again.n_evaluations == 0
    assert again.n_updates == 0


@pytest.mark.parametrize(
    "schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule]
)
def test_4_the_clamp_precedence_holds_in_all_three_regimes(schedule_type):
    """Floor beats criterion, ceiling binds, and the budget cap beats the floor."""
    evaluation = _evaluation_with_misfits(np.array([1.0, 1.0001, 0.9999, 1.00005]))

    # The floor binds: a widely spread ensemble drives the criterion far below
    # the floor, and the floor must win. The ceiling is slack here, so this
    # regime is the floor's alone -- inverting the maximum into a minimum
    # returns the criterion instead and fails.
    spread_out = _evaluation_with_misfits(np.array([1.0, 400.0, 0.5, 900.0]))
    tiny = schedule_type(beta_target=None, min_increment=0.5, max_increment=2.0)
    unclamped_is_smaller = float(
        schedule_type(
            beta_target=None, min_increment=1e-12, max_increment=2.0
        ).next_increment(spread_out)
    )
    assert unclamped_is_smaller < 0.5, (
        "the fixture must put the criterion below the floor"
    )
    assert float(tiny.next_increment(spread_out)) == pytest.approx(0.5)

    # The ceiling binds: a nearly-degenerate ensemble wants a huge step.
    capped = schedule_type(beta_target=None, min_increment=1e-6, max_increment=0.3)
    assert float(capped.next_increment(evaluation)) == pytest.approx(0.3)

    # The budget cap beats the floor: only 0.01 of budget is left.
    near_budget = _evaluation_with_misfits(
        np.array([1.0, 1.0001, 0.9999, 1.00005]), beta=0.99
    )
    budgeted = schedule_type(
        beta_target=1.0, min_increment=0.5, max_increment=2.0
    )
    assert float(budgeted.next_increment(near_budget)) == pytest.approx(0.01)


@pytest.mark.parametrize(
    "schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule]
)
def test_4_a_degenerate_ensemble_takes_the_largest_allowed_step(schedule_type):
    """Identical misfits mean no increment changes the target's shape."""
    evaluation = _evaluation_with_misfits(np.full(5, 2.5))
    unbounded = schedule_type(beta_target=None, max_increment=0.4)
    assert float(unbounded.next_increment(evaluation)) == pytest.approx(0.4)

    part_way = _evaluation_with_misfits(np.full(5, 2.5), beta=0.75)
    budgeted = schedule_type(beta_target=1.0, max_increment=0.4)
    assert float(budgeted.next_increment(part_way)) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule]
)
def test_4_an_unbounded_ladder_runs_without_raising(schedule_type):
    """The executable form of the conditional budget term in the clamp.

    Writing the three-term clamp unconditionally is a ``TypeError`` on every
    unbounded run, which is a shipped configuration.
    """
    problem = _AffineProblem()
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=schedule_type(beta_target=None),
        stop=DiscrepancyStop(tau=50.0),
        max_steps=20,
    )
    assert result.status == STOPPING_RULE


def test_4_step_counts_are_exact_under_a_capped_criterion():
    """A budget of 1 under a ceiling of 0.3 takes exactly four steps.

    One test pins the ``>=`` in the exhaustion check, ``budget_tol``,
    cap-beats-floor, and the absence of a trailing dribble step. The
    criterion is ``inf`` because every misfit is identical, so the ceiling is
    the only thing choosing the increment.
    """

    class _ConstantModel:
        """Every member predicts the same thing, so every misfit is equal."""

        def __init__(self, n_members, v_dim):
            self.shape = (n_members, v_dim)
            self.calls = 0

        def __call__(self, u):
            self.calls += 1
            return jnp.ones(self.shape)

    problem = _AffineProblem()
    model = _ConstantModel(problem.J, problem.N)
    for schedule_type in (AdaptiveESSSchedule, AdaptiveMisfitSchedule):
        model.calls = 0
        result = run(
            problem.state(),
            model,
            jnp.asarray(problem.y),
            problem.noise_cov,
            schedule=schedule_type(beta_target=1.0, max_increment=0.3),
        )
        assert result.n_evaluations == 4, schedule_type
        assert model.calls == 4
        got = np.asarray(result.stacked.increment)
        assert np.allclose(got, [0.3, 0.3, 0.3, 0.1], atol=1e-12)


def test_4_ess_bisection_returns_the_safe_end_and_matches_a_hand_written_one():
    """The returned increment meets the ESS target, and reproduces a reference.

    Asserted at a **small** ``n_bisect``: at the default of 50 the two
    bracket ends differ by 2**-50 and no float64 tolerance can tell them
    apart, so the obligation would not distinguish returning ``lo`` from
    returning ``hi``. Both clamps are slack here.
    """
    misfit_values = np.array([0.5, 2.0, 4.5, 9.0, 1.25, 3.0])
    evaluation = _evaluation_with_misfits(misfit_values)
    n_bisect = 8
    schedule = AdaptiveESSSchedule(
        beta_target=None, min_increment=1e-9, max_increment=4.0, n_bisect=n_bisect
    )
    got = float(schedule.next_increment(evaluation))
    target = 0.5 * misfit_values.size

    assert _reference_ess(misfit_values, got) >= target
    assert 1e-9 < got < 4.0

    # A hand-written bisection at the same count, pinning both the step
    # count and the log-space computation.
    lo, hi = 0.0, 4.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if _reference_ess(misfit_values, mid) >= target:
            lo = mid
        else:
            hi = mid
    assert got == pytest.approx(lo, abs=1e-14)


def test_4_the_misfit_schedule_returns_the_larger_of_its_two_bounds():
    """One test per regime, decided by the misfits' coefficient of variation."""
    theta = 2.0

    # cv > 1/sqrt(theta): the mean bound is the larger one.
    spread_out = np.array([0.05, 4.0, 0.1, 8.0, 0.2])
    mean, variance = spread_out.mean(), spread_out.var(ddof=1)
    assert np.sqrt(variance) / mean > 1 / np.sqrt(theta)
    schedule = AdaptiveMisfitSchedule(
        beta_target=None, divergence_budget=theta, max_increment=1e6
    )
    got = float(schedule.next_increment(_evaluation_with_misfits(spread_out)))
    assert got == pytest.approx(theta / mean, rel=1e-12)
    assert theta / mean > np.sqrt(theta / variance)

    # cv < 1/sqrt(theta): the variance bound takes over.
    clustered = np.array([10.0, 10.4, 9.7, 10.1, 9.9])
    mean, variance = clustered.mean(), clustered.var(ddof=1)
    assert np.sqrt(variance) / mean < 1 / np.sqrt(theta)
    got = float(schedule.next_increment(_evaluation_with_misfits(clustered)))
    assert got == pytest.approx(np.sqrt(theta / variance), rel=1e-12)
    assert np.sqrt(theta / variance) > theta / mean


def test_4_the_misfit_schedule_guards_both_of_its_divisions():
    """``inf`` at a vanishing denominator, ``nan`` at a ``nan`` one."""
    schedule = AdaptiveMisfitSchedule(beta_target=None, max_increment=0.7)

    # Zero misfit spread, and zero mean misfit: both yield the largest step.
    assert float(
        schedule.next_increment(_evaluation_with_misfits(np.full(4, 3.0)))
    ) == pytest.approx(0.7)
    assert float(
        schedule.next_increment(_evaluation_with_misfits(np.zeros(4)))
    ) == pytest.approx(0.7)

    # A nan misfit must not silently select that same largest step.
    poisoned = _evaluation_with_misfits(np.array([1.0, np.nan, 2.0, 3.0]))
    assert not np.isfinite(float(schedule.next_increment(poisoned)))
    assert np.isnan(float(schedule.next_increment(poisoned)))


def test_4_the_entry_time_budget_check_raises_before_any_evaluation():
    """The bound is checked against the schedule's own floor-bound worst case."""
    problem = _AffineProblem()
    scaled_down = AdaptiveESSSchedule(beta_target=0.01, min_increment=1e-3)
    # ceil(0.01 / 1e-3) == 10, so max_steps=10 is exactly enough and 9 is not.
    with pytest.raises(ValueError, match="cannot accommodate"):
        run(
            problem.state(),
            problem.forward,
            jnp.asarray(problem.y),
            problem.noise_cov,
            schedule=scaled_down,
            max_steps=9,
        )
    assert problem.calls == []
    run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=scaled_down,
        max_steps=10,
    )

    # The shipped defaults satisfy the relation with no slack: 1 / 1e-3 == 1000.
    run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )
    with pytest.raises(ValueError, match="cannot accommodate"):
        run(
            problem.state(),
            problem.forward,
            jnp.asarray(problem.y),
            problem.noise_cov,
            schedule=AdaptiveESSSchedule(beta_target=2.0),
        )


def _reference_ess(misfit_values: np.ndarray, increment: float) -> float:
    """ESS of ``exp(-increment * misfits)``, in plain NumPy from the definition."""
    shifted = -increment * misfit_values
    weights = np.exp(shifted - shifted.max())
    return float(weights.sum() ** 2 / (weights**2).sum())


def _evaluation_with_misfits(
    misfit_values: np.ndarray, *, beta: float = 0.0, step: int = 0, v_dim: int = 4
) -> Evaluation:
    """An ``Evaluation`` whose misfits are exactly the values given.

    Row ``j`` of the whitened residuals is placed on the first coordinate at
    ``sqrt(2 * phi_j)``, so ``0.5 * ||b_j||**2`` is exactly ``phi_j``.
    """
    n_members = misfit_values.size
    residuals = np.zeros((n_members, v_dim))
    residuals[:, 0] = np.sqrt(2.0 * misfit_values)
    members = np.asarray(
        _exact_moment_ensemble(n_members, np.zeros(2), np.eye(2))
    )
    return Evaluation(
        step=step,
        beta=beta,
        ensemble=jnp.asarray(members),
        predictions=jnp.zeros((n_members, v_dim)),
        whitened_residuals=jnp.asarray(residuals),
        rms_parameter_spread=jnp.asarray(1.0),
        n_valid=n_members,
    )


def test_5_the_effective_sample_size_matches_its_definition():
    """``J`` at zero, monotone, and equal to ``J / (1 + cv^2)`` of the weights."""
    misfit_values = np.array([0.25, 1.0, 3.5, 7.0, 0.75, 2.0, 12.0])
    J = misfit_values.size

    at_zero = float(effective_sample_size(jnp.asarray(misfit_values), 0.0))
    assert at_zero == pytest.approx(J, abs=8 * EPS * J)

    grid = np.linspace(0.0, 3.0, 40)
    values = np.array(
        [float(effective_sample_size(jnp.asarray(misfit_values), d)) for d in grid]
    )
    assert np.all(np.diff(values) <= 8 * EPS * J)
    assert values[-1] < values[0]

    for increment in (0.1, 0.7, 2.5):
        weights = np.exp(-increment * misfit_values)
        got = float(effective_sample_size(jnp.asarray(misfit_values), increment))
        assert got == pytest.approx(_reference_ess(misfit_values, increment), rel=1e-12)
        cv_squared = weights.var(ddof=0) / weights.mean() ** 2
        assert got == pytest.approx(J / (1.0 + cv_squared), rel=1e-12)
        assert 1.0 - 1e-12 <= got <= J + 1e-9


def test_5_the_effective_sample_size_survives_misfits_the_naive_form_cannot():
    """Misfits of order 1e4 return a finite value where naive weights are ``nan``.

    A targeted regression test for the non-log-space form: naive
    exponentiation underflows every weight to zero and the ratio is a ``0/0``
    that would silently poison the bisection.
    """
    misfit_values = np.array([1.0e4, 1.2e4, 1.5e4, 1.1e4])
    got = float(effective_sample_size(jnp.asarray(misfit_values), 1.0))
    assert np.isfinite(got)
    assert 1.0 <= got <= misfit_values.size

    with np.errstate(invalid="ignore", under="ignore"):
        naive_weights = np.exp(-1.0 * misfit_values)
        naive = naive_weights.sum() ** 2 / (naive_weights**2).sum()
    assert np.isnan(naive)


def test_6_the_two_adaptive_criteria_are_distinct_and_measurably_so():
    """The misfit criterion drives the ESS to its floor and takes far longer steps.

    Pinned in both directions, so that neither schedule can silently drift
    into implementing the other's criterion.
    """
    rng = np.random.default_rng(11)
    misfit_values = 50.0 + 8.0 * rng.normal(size=40)
    evaluation = _evaluation_with_misfits(misfit_values, v_dim=80)

    unbounded = dict(beta_target=None, min_increment=1e-9, max_increment=1e6)
    misfit_step = float(AdaptiveMisfitSchedule(**unbounded).next_increment(evaluation))
    ess_step = float(AdaptiveESSSchedule(**unbounded).next_increment(evaluation))

    at_misfit_step = float(
        effective_sample_size(jnp.asarray(misfit_values), misfit_step)
    )
    assert at_misfit_step < 4.0, "the misfit criterion should sit near the ESS floor"
    assert misfit_step >= 5.0 * ess_step


def test_7_the_discrepancy_stop_fires_on_its_threshold_and_ends_the_run():
    """Fires exactly when ``2 Phi(vbar) <= tau^2 N``, before the increment."""
    v_dim = 4
    for tau in (0.5, 1.0, 2.0):
        rule = DiscrepancyStop(tau=tau)
        for centre_misfit in (0.1, 0.9, 2.0, 4.5, 9.0):
            evaluation = _evaluation_with_misfits(
                np.full(4, centre_misfit), v_dim=v_dim
            )
            # Every member has the same residual, so the centre misfit is it.
            assert float(evaluation.centre_misfit) == pytest.approx(centre_misfit)
            assert rule(evaluation) is (2.0 * centre_misfit <= tau**2 * v_dim)


def test_7_a_fired_stop_ends_the_run_with_a_zero_increment_terminal_record():
    """The state is left unchanged and the last record's increment is exactly 0."""
    problem = _AffineProblem()
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=FixedSchedule.constant(0.5, 50),
        stop=DiscrepancyStop(tau=1e6),
    )
    assert result.status == STOPPING_RULE
    assert result.n_evaluations == 1
    assert float(result.stacked.increment[0]) == 0.0
    assert len(problem.calls) == 1
    # Ended at step 0, with an empty *update* history and the state untouched.
    assert float(result.beta) == 0.0
    assert np.array_equal(np.asarray(result.ensemble), problem.members)
    assert result.last_evaluation is not None
    assert np.array_equal(
        np.asarray(result.last_evaluation.ensemble), np.asarray(result.ensemble)
    )


def test_8_max_steps_raises_with_a_payload_that_makes_the_run_resumable():
    """Every raise path carries ``state`` and ``history``, and resuming is exact."""
    problem = _AffineProblem()
    schedule = FixedSchedule.constant(0.25, 40)
    with pytest.raises(EKIError, match="max_steps") as caught:
        run(
            problem.state(),
            problem.forward,
            jnp.asarray(problem.y),
            problem.noise_cov,
            schedule=schedule,
            max_steps=6,
        )
    failure = caught.value
    assert "FixedSchedule" in str(failure)
    assert "no stopping rule" in str(failure)
    assert isinstance(failure.state, EKIState)
    assert len(failure.history) == 6
    assert failure.state.step == 6

    resumed = run(
        failure.state,
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=schedule,
        max_steps=40,
    )
    uninterrupted = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=schedule,
        max_steps=40,
    )
    assert np.array_equal(
        np.asarray(resumed.ensemble), np.asarray(uninterrupted.ensemble)
    )
    assert resumed.n_evaluations == 34


def test_8_a_budgeted_schedule_with_a_positive_floor_never_reaches_the_bound():
    problem = _AffineProblem()
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
        max_steps=1000,
    )
    assert result.status == SCHEDULE_EXHAUSTED


def test_9_the_repair_moves_only_the_failed_members_and_damps_the_moments():
    """The three exact identities, pinned as equalities rather than tolerances."""
    rng = np.random.default_rng(19)
    J, P, N = 9, 3, 4
    ensemble = jnp.asarray(rng.normal(size=(J, P)))
    predictions = jnp.asarray(rng.normal(size=(J, N)))
    valid = jnp.asarray([True, True, False, True, False, True, True, True, False])
    n_valid = int(np.asarray(valid).sum())

    repaired, repaired_predictions = repair_failed_members(
        ensemble=ensemble, predictions=predictions, valid=valid
    )
    mask = np.asarray(valid)
    u_hat = np.asarray(ensemble)[mask].mean(axis=0)
    v_hat = np.asarray(predictions)[mask].mean(axis=0)

    # Valid members are bit-identical; failed ones sit exactly at the centre.
    assert np.array_equal(np.asarray(repaired)[mask], np.asarray(ensemble)[mask])
    assert np.array_equal(
        np.asarray(repaired_predictions)[mask], np.asarray(predictions)[mask]
    )
    assert np.abs(np.asarray(repaired)[~mask] - u_hat).max() < 8 * EPS
    assert np.abs(np.asarray(repaired_predictions)[~mask] - v_hat).max() < 8 * EPS

    # The all-J mean is the valid-member mean.
    got_mean = np.asarray(repaired).mean(axis=0)
    assert np.abs(got_mean - u_hat).max() < 16 * EPS * max(1.0, np.abs(u_hat).max())

    # The all-J covariance is the valid-member one, damped by exactly this factor.
    damping = (n_valid - 1) / (J - 1)
    valid_u = np.asarray(ensemble)[mask] - u_hat
    valid_v = np.asarray(predictions)[mask] - v_hat
    for got, want in (
        (
            _moments(repaired)[1],
            damping * (valid_u.T @ valid_u) / (n_valid - 1),
        ),
        (
            (np.asarray(repaired) - got_mean).T
            @ (np.asarray(repaired_predictions) - v_hat)
            / (J - 1),
            damping * (valid_u.T @ valid_v) / (n_valid - 1),
        ),
    ):
        scale = max(1.0, np.abs(want).max())
        assert np.abs(got - want).max() < 64 * EPS * scale


def test_9_a_no_failure_step_skips_the_repair_entirely():
    """Adding failure handling changes nothing about a run in which nothing fails.

    The repair formula is mathematically the identity when every member is
    valid but *not* bit-exactly so, so the driver branches in Python on the
    synchronized valid count and skips it. Asserted by object identity, which
    is the only thing that distinguishes "skipped" from "applied and happened
    to agree".
    """
    problem = _AffineProblem()
    state = problem.state()
    evaluation = evaluate(
        state, problem.forward, jnp.asarray(problem.y), problem.noise_cov
    )
    assert evaluation.ensemble is state.ensemble
    assert evaluation.n_valid == problem.J

    # And the helper itself is bit-exact on an all-valid mask.
    predictions = jnp.asarray(problem.members @ problem.G.T)
    repaired, repaired_predictions = repair_failed_members(
        ensemble=state.ensemble,
        predictions=predictions,
        valid=jnp.ones(problem.J, dtype=bool),
    )
    assert np.array_equal(np.asarray(repaired), np.asarray(state.ensemble))
    assert np.array_equal(np.asarray(repaired_predictions), np.asarray(predictions))


def test_9_the_failure_modes_raise_where_the_contract_says_they_do():
    problem = _AffineProblem()
    state = problem.state()
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    def fail(indices):
        def forward(u):
            v = jnp.asarray(u) @ jnp.asarray(problem.G).T
            return v.at[jnp.asarray(indices), 0].set(jnp.nan)

        return forward

    with pytest.raises(EKIError, match="on_failure='raise'"):
        evaluate(state, fail([2]), y, noise, on_failure="raise")
    with pytest.raises(EKIError, match=r"\[2\]"):
        evaluate(state, fail([2]), y, noise, on_failure="raise")

    all_but_one = list(range(1, problem.J))
    for mode in ("repair", "raise"):
        with pytest.raises(EKIError, match="At least 2 are required"):
            evaluate(state, fail(all_but_one), y, noise, on_failure=mode)

    with pytest.raises(ValueError, match="on_failure"):
        evaluate(state, problem.forward, y, noise, on_failure="Raise")
    with pytest.raises(ValueError, match="on_failure"):
        run(state, problem.forward, y, noise, schedule=FixedSchedule.uniform(2),
            on_failure="skip")

    # A repaired run produces no nan statistics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(
            state, fail([1, 4]), y, noise, schedule=FixedSchedule.uniform(3)
        )
    stacked = result.stacked
    for name in ("misfit_mean", "misfit_min", "misfit_max", "centre_misfit", "ess"):
        assert np.all(np.isfinite(np.asarray(getattr(stacked, name)))), name
    assert int(stacked.n_valid[0]) == problem.J - 2
    assert result.min_n_valid == problem.J - 2


def test_10_multiplicative_inflation_scales_the_anomalies_not_the_covariance():
    """The mean is preserved and the covariance is scaled by the factor *squared*."""
    rng = np.random.default_rng(23)
    ensemble = jnp.asarray(rng.normal(size=(10, 4)))
    before_mean, before_cov = _moments(ensemble)
    for factor in (1.02, 1.2, 2.0):
        inflated = MultiplicativeInflation(factor)(
            jax.random.key(0), ensemble=ensemble, step=0, beta=jnp.asarray(0.0)
        )
        after_mean, after_cov = _moments(inflated)
        assert np.abs(after_mean - before_mean).max() < 16 * EPS * max(
            1.0, np.abs(before_mean).max()
        )
        want = factor**2 * before_cov
        assert np.abs(after_cov - want).max() < 64 * EPS * np.abs(want).max()


def test_10_multiplicative_inflation_stays_in_the_span_and_additive_leaves_it():
    """The executable form of the subspace property, in both directions."""
    # P = 5 with rank-3 initial anomalies, so the span is a proper subspace of
    # R^P and "leaving it" is a statement with content.
    problem = _AffineProblem(P=5, N=6, J=4, seed=17, prior_rank=3)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    basis = _span_basis(problem.members)
    assert basis.shape[1] < problem.P

    plain = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.uniform(4),
    )
    assert _leaves_span(plain.ensemble, problem.members, basis) < 1e-9

    multiplicative = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.uniform(4),
        inflation=MultiplicativeInflation(1.05),
    )
    assert _leaves_span(multiplicative.ensemble, problem.members, basis) < 1e-9

    additive = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.uniform(4),
        inflation=AdditiveInflation(
            DensePSD.from_matrix(jnp.eye(problem.P) * 0.05)
        ),
    )
    assert _leaves_span(additive.ensemble, problem.members, basis) > 1e-3


def test_10_additive_inflation_matches_its_pinned_elementwise_definition():
    """Delegation to ``Gaussian.sample``, centred, and mean-preserving."""
    rng = np.random.default_rng(29)
    J, P = 8, 3
    ensemble = jnp.asarray(rng.normal(size=(J, P)))
    cov = DensePSD.from_matrix(jnp.asarray(_psd(P, seed=31) * 0.01))
    key = jax.random.key(4)

    got = AdditiveInflation(cov)(key, ensemble=ensemble, step=0, beta=jnp.asarray(0.0))
    pert = Gaussian(jnp.zeros(P), cov).sample(key, J)
    want = ensemble + (pert - pert.mean(axis=0))
    assert np.array_equal(np.asarray(got), np.asarray(want))

    before, after = _moments(ensemble)[0], _moments(got)[0]
    assert np.abs(after - before).max() < 16 * EPS * max(1.0, np.abs(before).max())


def _span_basis(members) -> np.ndarray:
    """An orthonormal basis for the span of an ensemble's anomalies."""
    anomalies = np.asarray(members) - np.asarray(members).mean(axis=0)
    basis, *_ = np.linalg.svd(anomalies.T, full_matrices=False)
    return basis[:, : int(np.linalg.matrix_rank(anomalies))]


def _leaves_span(members, initial, basis: np.ndarray) -> float:
    """How far the members sit outside the initial ensemble's affine subspace."""
    moved = np.asarray(members) - np.asarray(initial).mean(axis=0)
    return float(np.abs(moved - moved @ basis @ basis.T).max())


@pytest.mark.parametrize("batch", [(), (3,), (2, 3)])
def test_11_misfits_matches_a_dense_quadratic_form_at_every_batch_rank(batch):
    """``0.5 (y - v)^T R^-1 (y - v)``, batched per the operator layer's contract."""
    rng = np.random.default_rng(41)
    N = 5
    R = _psd(N, seed=43)
    y = rng.normal(size=N)
    predictions = rng.normal(size=(*batch, N))
    got = misfits(
        jnp.asarray(y), jnp.asarray(predictions), DensePSD.from_matrix(jnp.asarray(R))
    )
    residual = y - predictions
    want = 0.5 * np.einsum(
        "...i,ij,...j->...", residual, np.linalg.inv(R), residual
    )
    assert np.asarray(got).shape == batch
    assert np.abs(np.asarray(got) - want).max() < 1e-9 * max(1.0, np.abs(want).max())


def test_11_misfits_is_whitener_invariant_and_carries_the_half():
    """Two operators for the same R agree, and the factor of 1/2 is pinned."""
    diagonal = np.array([2.0, 0.5, 4.0])
    y = jnp.asarray([1.0, 2.0, 3.0])
    predictions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    as_diagonal = PSDDiagonal(jnp.asarray(diagonal))
    as_dense = DensePSD.from_matrix(jnp.diag(jnp.asarray(diagonal)))
    got = misfits(y, predictions, as_diagonal)
    assert np.abs(
        np.asarray(got) - np.asarray(misfits(y, predictions, as_dense))
    ).max() < 1e-12
    # Closed form: 0.5 * (1/2 + 4/0.5 + 9/4) for the first row, 0 for the second.
    assert float(got[0]) == pytest.approx(0.5 * (0.5 + 8.0 + 2.25))
    assert float(got[1]) == 0.0


def test_11_the_centre_misfit_differs_from_the_mean_by_exactly_the_spread_term():
    """``mean(Phi_j) - Phi(vbar) == (J-1)/(2J) tr(W Chat_vv W^T)``, exactly.

    A divisor the derivation gets wrong easily, and which no tolerance-based
    test would catch. The identity is also what makes ``centre_misfit`` and
    ``misfit_mean`` two different fields rather than one.
    """
    problem = _AffineProblem(J=10)
    state = problem.state()
    evaluation = evaluate(
        state, problem.forward, jnp.asarray(problem.y), problem.noise_cov
    )
    J = problem.J
    whitened = np.asarray(evaluation.whitened_residuals)
    anomalies = whitened - whitened.mean(axis=0)
    trace = float(np.sum(anomalies**2) / (J - 1))
    gap = float(np.mean(np.asarray(evaluation.misfits))) - float(
        evaluation.centre_misfit
    )
    want = (J - 1) / (2 * J) * trace
    assert gap == pytest.approx(want, rel=1e-11)

    # Both are recovered from the stored whitened residuals alone.
    assert np.abs(
        np.asarray(evaluation.misfits) - 0.5 * np.sum(whitened**2, axis=1)
    ).max() < 1e-12
    assert float(evaluation.centre_misfit) == pytest.approx(
        0.5 * float(np.sum(whitened.mean(axis=0) ** 2)), rel=1e-12
    )

    # And so are the whitened prediction anomalies, as -W A_v.
    W = _recovered_whitener(problem.noise_cov, problem.N)
    predictions = np.asarray(evaluation.predictions)
    want_anomalies = -(predictions - predictions.mean(axis=0)) @ W.T
    assert np.abs(anomalies - want_anomalies).max() < 1e-8

    # The free function and the property are the same quantity.
    assert np.abs(
        np.asarray(evaluation.misfits)
        - np.asarray(
            misfits(jnp.asarray(problem.y), evaluation.predictions, problem.noise_cov)
        )
    ).max() < 1e-10


def test_12_runs_are_reproducible_resumable_and_agree_with_iterate():
    """Bit-identical repeats, an exact tail after a resumption, and one driver."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    schedule = FixedSchedule.uniform(8)
    common = dict(schedule=schedule, update=PathwiseUpdate())

    first = run(problem.state(), problem.forward, y, noise, **common)
    second = run(problem.state(), problem.forward, y, noise, **common)
    assert np.array_equal(np.asarray(first.ensemble), np.asarray(second.ensemble))

    # Stop after four steps and resume: the tail is bit-exact.
    partial_state = problem.state()
    taken = 0
    for yielded in iterate(problem.state(), problem.forward, y, noise, **common):
        partial_state = yielded[0]
        taken += 1
        if taken == 4:
            break
    assert taken == 4 and partial_state.step == 4
    resumed = run(partial_state, problem.forward, y, noise, **common)
    assert np.array_equal(np.asarray(resumed.ensemble), np.asarray(first.ensemble))
    for got, want in zip(resumed.history, first.history[4:], strict=True):
        for name in ("beta", "increment", "misfit_mean", "ess"):
            assert float(getattr(got, name)) == float(getattr(want, name))

    # iterate and run agree, and iterate returns the status.
    generator = iterate(problem.state(), problem.forward, y, noise, **common)
    records, last_state, last_evaluation = [], None, None
    while True:
        try:
            last_state, record, last_evaluation = next(generator)
        except StopIteration as finished:
            status = finished.value
            break
        records.append(record)
    assert status == first.status
    assert len(records) == first.n_evaluations
    assert np.array_equal(
        np.asarray(last_state.ensemble), np.asarray(first.ensemble)
    )
    assert np.array_equal(
        np.asarray(last_evaluation.predictions),
        np.asarray(first.last_evaluation.predictions),
    )


def test_13_the_optimization_form_approaches_the_restricted_least_squares_fit():
    """A monotone misfit, and a centre approaching the subspace-restricted fit.

    The correct target is the least-squares solution restricted to the initial
    ensemble's affine subspace, not the unrestricted minimizer: every iterate
    lies in that subspace, however many steps are run. The fixture makes the
    two differ, with ``P = 5`` against rank-3 initial anomalies, so that a
    test comparing against the unrestricted minimizer would fail.
    """
    problem = _AffineProblem(P=5, N=6, J=4, seed=13, prior_rank=3, noise_scale=1e-2)
    basis = _span_basis(problem.members)
    origin = np.asarray(problem.members).mean(axis=0)

    # An observation the model can nearly reproduce, so the discrepancy
    # principle has something to fire on: the truth lies in the ensemble's own
    # subspace, perturbed by one draw of the observation noise.
    truth = origin + basis @ np.array([1.5, -2.0, 1.0])
    perturbation = np.linalg.cholesky(problem.R) @ np.random.default_rng(3).normal(
        size=problem.N
    )
    problem.y = problem.G @ truth + perturbation
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    design = problem.G @ basis
    precision = np.linalg.inv(problem.R)
    coefficients = np.linalg.solve(
        design.T @ precision @ design,
        design.T @ precision @ (problem.y - problem.G @ origin),
    )
    restricted = origin + basis @ coefficients
    unrestricted = np.linalg.lstsq(problem.G, problem.y, rcond=None)[0]
    assert np.abs(restricted - unrestricted).max() > 1e-2, (
        "the fixture must distinguish the two targets"
    )

    result = run(
        problem.state(),
        problem.forward,
        y,
        noise,
        schedule=FixedSchedule.constant(1.0, 200),
        stop=DiscrepancyStop(tau=1.0),
        max_steps=200,
    )
    assert result.stop_fired, "the run must terminate on the discrepancy principle"

    centre_misfits = np.asarray(result.stacked.centre_misfit)
    assert np.all(np.diff(centre_misfits) <= 1e-9 * max(1.0, centre_misfits[0]))

    got = np.asarray(result.ensemble).mean(axis=0)
    assert np.abs(got - restricted).max() < 1e-2 * max(1.0, np.abs(restricted).max())
    assert np.abs(got - unrestricted).max() > 1e-2


def test_14_a_run_compiles_a_bounded_number_of_times_whatever_its_length():
    """The executable form of the traced-increment requirement.

    A static field on an object crossing a ``jit`` boundary is what would
    force a retrace per step, so nothing whole — no ``EKIState``, no
    ``Evaluation`` — is passed into a jitted function; the arrays are.
    Asserted against the compilation caches of the layer's own jitted
    functions rather than inspected by eye: a thirty-step run must add no
    compilations that a three-step run of the same problem has not already
    paid for.

    An increment baked in as a Python constant is deliberately *not* what
    this guards: a Python float passed as a ``jit`` argument does not
    retrace, so the test that would have caught it cannot fail.
    """
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    configuration = dict(
        update=PathwiseUpdate(), inflation=MultiplicativeInflation(1.01)
    )

    before = _compilation_count()
    run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.constant(0.05, 3), **configuration,
    )
    after_short = _compilation_count()
    run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.constant(0.05, 30), **configuration,
    )
    after_long = _compilation_count()

    assert after_long == after_short, (
        f"compilations grew with the number of steps: {after_short} -> "
        f"{after_long} over ten times as many steps"
    )
    assert after_short - before <= 12


def _jitted_functions():
    """Every ``jax.jit``-wrapped function the layer defines, across its modules."""
    from pyeki.eki import driver, helpers, policies, values

    found = []
    for module in (helpers, values, policies, driver):
        for value in vars(module).values():
            if hasattr(value, "_cache_size") and not any(
                value is seen for seen in found
            ):
                found.append(value)
    return found


def _compilation_count() -> int:
    """How many distinct traces the layer's jitted functions currently hold."""
    return sum(fn._cache_size() for fn in _jitted_functions())


def test_14_the_pytree_classes_round_trip_and_families_are_inert():
    """Flatten and unflatten preserve type and behaviour, with sentinel leaves."""
    problem = _AffineProblem()
    state = problem.state()
    evaluation = evaluate(
        state, problem.forward, jnp.asarray(problem.y), problem.noise_cov
    )
    record = advance(
        state, problem.forward, jnp.asarray(problem.y), problem.noise_cov,
        increment=0.25,
    )[1]

    for obj in (state, evaluation, record):
        leaves, treedef = jax.tree.flatten(obj)
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert type(rebuilt) is type(obj)
        assert rebuilt.batch_shape == ()
        sentinel = jax.tree.unflatten(treedef, [object()] * len(leaves))
        assert type(sentinel) is type(obj)
        assert "unprintable" in repr(sentinel) or repr(sentinel).startswith(
            type(obj).__name__
        )

    # A family reports its batch shape, takes the vmapped repr, and refuses.
    family = jax.tree.map(lambda x: jnp.stack([x, x]), evaluation)
    assert family.batch_shape == (2,)
    assert repr(family).startswith("vmapped(")
    with pytest.raises(ValueError, match="vmapped family"):
        _ = family.misfits
    assert family.n_members == evaluation.n_members


def test_15_the_history_stacks_including_its_two_integer_fields():
    """A targeted regression test for the treedef trap.

    Declaring ``step`` or ``n_valid`` as static metadata makes every record a
    different pytree type, and the underlying ``jax.tree.map`` then raises.
    """
    problem = _AffineProblem()
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=FixedSchedule.uniform(5),
    )
    stacked = result.stacked
    assert isinstance(stacked, HistoryRecord)
    assert stacked.batch_shape == (5,)
    for name in ("step", "n_valid", "beta", "increment", "ess"):
        assert np.asarray(getattr(stacked, name)).shape == (5,)
    assert list(np.asarray(stacked.step)) == [0, 1, 2, 3, 4]

    empty = EKIResult(
        state=problem.state(),
        history=(),
        status=SCHEDULE_EXHAUSTED,
        last_evaluation=None,
    ).stacked
    assert empty.batch_shape == (0,)
    for name in ("step", "n_valid", "beta", "increment", "ess"):
        assert np.asarray(getattr(empty, name)).shape == (0,)


def test_16_the_two_phases_compose_and_one_evaluation_serves_two_increments():
    """``advance`` is ``assimilate`` of ``evaluate``, and a rejected trial is free."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    state = problem.state()

    composed_state, composed_record = advance(
        state, problem.forward, y, noise, increment=0.3
    )
    calls_after_advance = len(problem.calls)
    evaluation = evaluate(state, problem.forward, y, noise)
    applied_state, applied_record = assimilate(
        state, evaluation, increment=0.3, y=y, noise_cov=noise
    )
    assert np.array_equal(
        np.asarray(composed_state.ensemble), np.asarray(applied_state.ensemble)
    )
    assert float(composed_record.ess) == float(applied_record.ess)
    assert calls_after_advance == 1

    # One evaluation, two trial increments, no further model calls.
    before = len(problem.calls)
    small, _ = assimilate(state, evaluation, increment=0.1, y=y, noise_cov=noise)
    large, _ = assimilate(state, evaluation, increment=0.9, y=y, noise_cov=noise)
    assert len(problem.calls) == before
    assert float(small.beta) == pytest.approx(0.1)
    assert float(large.beta) == pytest.approx(0.9)
    assert not np.array_equal(np.asarray(small.ensemble), np.asarray(large.ensemble))
    assert np.array_equal(np.asarray(state.ensemble), problem.members)
    assert evaluation.step == 0 and float(evaluation.beta) == 0.0

    # A mismatched evaluation is refused, and the increment is checked first.
    moved, _ = advance(state, problem.forward, y, noise, increment=0.2)
    with pytest.raises(ValueError, match="different state"):
        assimilate(moved, evaluation, increment=0.1, y=y, noise_cov=noise)
    before = len(problem.calls)
    for bad in (0.0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match="strictly positive"):
            assimilate(state, evaluation, increment=bad, y=y, noise_cov=noise)
    assert len(problem.calls) == before, (
        "assimilate must validate its increment before doing any array work"
    )


def test_16_iterate_yields_what_a_hand_written_two_phase_loop_yields():
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    increments = (0.2, 0.3, 0.5)

    by_hand = problem.state()
    for increment in increments:
        by_hand, _ = advance(by_hand, problem.forward, y, noise, increment=increment)

    driven = None
    for yielded in iterate(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule(increments),
    ):
        driven = yielded[0]
    assert np.array_equal(
        np.asarray(driven.ensemble), np.asarray(by_hand.ensemble)
    )


@pytest.mark.parametrize("P,N,J", [(1, 1, 2), (3, 1, 2), (1, 4, 5), (2, 2, 2)])
def test_17_the_degenerate_shapes_all_work(P, N, J):
    problem = _AffineProblem(P=P, N=N, J=max(J, P + 1), seed=53)
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=FixedSchedule.uniform(3),
    )
    assert np.all(np.isfinite(np.asarray(result.ensemble)))


def test_17_a_collapsed_ensemble_neither_moves_nor_produces_nan():
    problem = _AffineProblem()
    collapsed = jnp.tile(jnp.asarray(problem.members[:1]), (problem.J, 1))
    state = EKIState(collapsed, 0.0, 0, jax.random.key(0))
    result = run(
        state,
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=FixedSchedule.uniform(3),
    )
    assert np.all(np.isfinite(np.asarray(result.ensemble)))
    # The gain vanishes with the prediction anomalies, so the only movement is
    # the round-off of re-forming the mean of J identical rows.
    scale = float(np.abs(np.asarray(collapsed)).max())
    assert np.abs(
        np.asarray(result.ensemble) - np.asarray(collapsed)
    ).max() < 8 * EPS * scale


def test_17_a_wholly_failing_model_and_a_nan_update_both_raise():
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    with pytest.raises(EKIError, match="At least 2 are required"):
        run(
            problem.state(),
            lambda u: jnp.full((problem.J, problem.N), jnp.nan),
            y,
            noise,
            schedule=FixedSchedule.uniform(3),
        )

    def poison(key, *, ensemble, **_):
        return jnp.full_like(ensemble, jnp.nan)

    with pytest.raises(EKIError, match="non-finite ensemble at step 0"):
        run(
            problem.state(), problem.forward, y, noise,
            schedule=FixedSchedule.uniform(3), update=poison,
        )


def test_18_every_tier_two_and_tier_three_rule_raises_as_specified():
    """The validation table, rule by rule."""
    key = jax.random.key(0)
    ensemble = jnp.zeros((4, 3))

    # -- EKIState ------------------------------------------------------------
    with pytest.raises(ValueError, match="rank 2"):
        EKIState(jnp.zeros((4,)), 0.0, 0, key)
    with pytest.raises(ValueError, match="at least 2 members"):
        EKIState(jnp.zeros((1, 3)), 0.0, 0, key)
    with pytest.raises(ValueError, match="core sizes must be positive"):
        EKIState(jnp.zeros((4, 0)), 0.0, 0, key)
    with pytest.raises(ValueError, match="must not be negative"):
        EKIState(ensemble, 0.0, -1, key)
    with pytest.raises(TypeError, match="Python int"):
        EKIState(ensemble, 0.0, 1.0, key)
    with pytest.raises(ValueError, match="scalar"):
        EKIState(ensemble, jnp.zeros((2,)), 0, key)
    with pytest.raises(TypeError, match="typed PRNG key"):
        EKIState(ensemble, 0.0, 0, jnp.zeros(()))
    with pytest.raises(TypeError, match="typed PRNG key"):
        EKIState(ensemble, 0.0, 0, jax.random.PRNGKey(0))
    with debug_checks():
        with pytest.raises(ValueError, match="must not be negative"):
            EKIState(ensemble, -0.5, 0, key)
        with pytest.raises(ValueError, match="finite"):
            EKIState(jnp.full((4, 3), jnp.nan), 0.0, 0, key)

    # -- policies ------------------------------------------------------------
    with pytest.raises(ValueError, match="must not be empty"):
        FixedSchedule(())
    with pytest.raises(ValueError, match="strictly positive"):
        FixedSchedule((0.5, 0.0, 0.5))
    with pytest.raises(ValueError, match="strictly positive"):
        FixedSchedule((0.5, jnp.inf))
    with pytest.raises(TypeError, match="tuple"):
        FixedSchedule([0.5, 0.5])
    with pytest.raises(ValueError, match="ess_fraction"):
        AdaptiveESSSchedule(ess_fraction=1.0)
    with pytest.raises(ValueError, match="ess_fraction"):
        AdaptiveESSSchedule(ess_fraction=0.0)
    with pytest.raises(ValueError, match="n_bisect"):
        AdaptiveESSSchedule(n_bisect=0)
    with pytest.raises(ValueError, match="max_increment"):
        AdaptiveESSSchedule(max_increment=float("inf"))
    with pytest.raises(ValueError, match="min_increment"):
        AdaptiveESSSchedule(min_increment=2.0, max_increment=1.0)
    with pytest.raises(ValueError, match="beta_target"):
        AdaptiveESSSchedule(beta_target=0.0)
    with pytest.raises(ValueError, match="divergence_budget"):
        AdaptiveMisfitSchedule(divergence_budget=0.0)
    with pytest.raises(ValueError, match="divergence_budget"):
        AdaptiveMisfitSchedule(divergence_budget=float("inf"))
    with pytest.raises(ValueError, match="tau"):
        DiscrepancyStop(tau=0.0)
    with pytest.raises(TypeError, match="PSDLinOp"):
        AdditiveInflation(jnp.eye(3))
    with debug_checks():
        with pytest.raises(ValueError, match="anomaly_factor"):
            MultiplicativeInflation(-1.0)

    # -- call-time problem and policy outputs ---------------------------------
    problem = _AffineProblem()
    state, noise = problem.state(), problem.noise_cov
    y = jnp.asarray(problem.y)
    ladder = FixedSchedule.uniform(2)
    with pytest.raises(ValueError, match="shape"):
        run(state, problem.forward, jnp.zeros((problem.N + 1,)), noise, schedule=ladder)
    with pytest.raises(ValueError, match="must be finite"):
        run(state, problem.forward, jnp.full((problem.N,), jnp.nan), noise,
            schedule=ladder)
    with pytest.raises(TypeError, match="PSDLinOp"):
        run(state, problem.forward, y, jnp.eye(problem.N), schedule=ladder)
    with pytest.raises(ValueError, match="max_steps"):
        run(state, problem.forward, y, noise, schedule=ladder, max_steps=0)
    with pytest.raises(ValueError, match="forward model returned shape"):
        run(state, lambda u: jnp.zeros((problem.J, problem.N + 1)), y, noise,
            schedule=ladder)
    with pytest.raises(ValueError, match="real floating dtype"):
        run(state, lambda u: jnp.zeros((problem.J, problem.N), dtype=jnp.int32),
            y, noise, schedule=ladder)
    with pytest.raises(ValueError, match="inflation returned shape"):
        run(state, problem.forward, y, noise, schedule=ladder,
            inflation=lambda key, *, ensemble, step, beta, **_: ensemble[:-1])
    with pytest.raises(ValueError, match="update returned shape"):
        run(state, problem.forward, y, noise, schedule=ladder,
            update=lambda key, *, ensemble, **_: ensemble[:, :-1])
    with pytest.raises(ValueError, match="update returned dtype"):
        run(state, problem.forward, y, noise, schedule=ladder,
            update=lambda key, *, ensemble, **_: ensemble.astype(jnp.float32))
    with pytest.raises(ValueError, match="cov has side"):
        run(state, problem.forward, y, noise, schedule=ladder,
            inflation=AdditiveInflation(DensePSD.from_matrix(jnp.eye(problem.P + 1))))

    class _NonPositive:
        n_steps, beta_target = 4, None

        def next_increment(self, evaluation):
            return -0.5

    with pytest.raises(ValueError, match="strictly positive"):
        run(state, problem.forward, y, noise, schedule=_NonPositive())

    # -- unsupported operations propagate unmodified, before any evaluation ---
    before = len(problem.calls)
    with pytest.raises(UnsupportedOpError):
        run(state, problem.forward, y, PSDLowRank(jnp.eye(problem.N)),
            schedule=ladder)
    assert len(problem.calls) == before, (
        "a covariance without whiten must cost no forward-model evaluations"
    )

    # -- repair_failed_members ------------------------------------------------
    with pytest.raises(ValueError, match="at least 2 valid"):
        repair_failed_members(
            ensemble=jnp.zeros((4, 3)),
            predictions=jnp.zeros((4, 2)),
            valid=jnp.asarray([True, False, False, False]),
        )
    with pytest.raises(ValueError, match="boolean"):
        repair_failed_members(
            ensemble=jnp.zeros((4, 3)),
            predictions=jnp.zeros((4, 2)),
            valid=jnp.ones(4),
        )

    # -- EKIResult ------------------------------------------------------------
    with pytest.raises(ValueError, match="status"):
        EKIResult(state=state, history=(), status="stopping-rule",
                  last_evaluation=None)


def test_18_reprs_are_types_and_static_sizes_with_no_array_data():
    problem = _AffineProblem()
    state = problem.state()
    evaluation = evaluate(
        state, problem.forward, jnp.asarray(problem.y), problem.noise_cov
    )
    _, record = advance(
        state, problem.forward, jnp.asarray(problem.y), problem.noise_cov,
        increment=0.5,
    )
    result = run(
        problem.state(), problem.forward, jnp.asarray(problem.y), problem.noise_cov,
        schedule=FixedSchedule.uniform(2),
    )
    assert repr(state) == f"EKIState(n_members={problem.J}, u_dim={problem.P}, step=0)"
    assert repr(evaluation) == f"Evaluation(step=0, n_members={problem.J})"
    assert repr(record) == "HistoryRecord(step=0)"
    assert repr(result) == (
        "EKIResult(status='schedule_exhausted', n_evaluations=2, beta=1)"
    )
    assert repr(TransformUpdate()) == "TransformUpdate()"
    assert repr(DiscrepancyStop(tau=2.0)) == "DiscrepancyStop(tau=2.0)"
    assert repr(MultiplicativeInflation(1.02)) == (
        "MultiplicativeInflation(anomaly_factor=1.02)"
    )
    # FixedSchedule summarizes rather than enumerating its increments.
    long_ladder = repr(FixedSchedule.constant(1.0, 200))
    assert long_ladder == "FixedSchedule(n_steps=200, total=200.0)"
    assert "1.0, 1.0" not in long_ladder


def test_18_the_pinned_prior_draw_and_a_short_run_are_snapshotted():
    """A JAX-side PRNG change is detected rather than absorbed."""
    prior = Gaussian(jnp.zeros(2), PSDDiagonal(jnp.asarray([1.0, 4.0])))
    state = EKIState.from_prior(jax.random.key(0), prior, 3)

    key_sample, key_state = jax.random.split(jax.random.key(0))
    factor = prior.cov.factor()
    want = prior.mean + factor.matvec(jax.random.normal(key_sample, (3, factor.shape[1])))
    assert np.array_equal(np.asarray(state.ensemble), np.asarray(want))
    assert np.array_equal(
        np.asarray(jax.random.key_data(state.key)),
        np.asarray(jax.random.key_data(key_state)),
    )
    assert float(state.beta) == 0.0 and state.step == 0

    problem = _AffineProblem(P=2, N=2, J=4, seed=61)
    result = run(
        problem.state(seed=2), problem.forward, jnp.asarray(problem.y),
        problem.noise_cov, schedule=FixedSchedule.uniform(3),
        update=PathwiseUpdate(),
    )
    snapshot = np.array(
        [
            [-4.822437772347259, -0.5647789498818425],
            [-0.7782512445291673, -0.0207102206543081],
            [-1.273350642921045, 1.501021381875314],
            [-0.9664786494421309, -0.2782707344895974],
        ]
    )
    assert np.abs(np.asarray(result.ensemble) - snapshot).max() < 1e-12


def test_19_the_result_reports_the_run_on_four_fixtures():
    """``stop_fired`` and ``budget_complete`` are each true exactly on their status."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    # 1. A completed budget with no stopping rule.
    completed = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.uniform(4),
    )
    assert completed.status == SCHEDULE_EXHAUSTED
    assert completed.budget_complete and not completed.stop_fired

    # 2. An optimization run whose ladder ran out before its stop fired.
    unfitted = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.constant(1.0, 5),
        stop=DiscrepancyStop(tau=1e-6), max_steps=5,
    )
    assert unfitted.budget_complete and not unfitted.stop_fired

    # 3. A stop firing on a budgeted ladder, on a problem it has *not* fit.
    #    The threshold is met by the ensemble centre while the ladder is only
    #    part-way, which is the trap the two booleans exist to expose.
    early = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.uniform(10),
        stop=_StopAtStepThree(),
    )
    assert early.stop_fired and not early.budget_complete
    assert float(early.beta) < 1.0

    # 4. A user-built INTERRUPTED result.
    interrupted = EKIResult(
        state=problem.state(), history=(), status=INTERRUPTED, last_evaluation=None
    )
    assert not interrupted.stop_fired and not interrupted.budget_complete

    for result in (completed, unfitted, early, interrupted):
        assert result.status in (SCHEDULE_EXHAUSTED, STOPPING_RULE, INTERRUPTED)


class _StopAtStepThree:
    """A stopping rule that fires on position, never on the misfits.

    Deliberately independent of the data, so that an implementation deriving
    ``stop_fired`` or ``budget_complete`` from the last record's misfit fails
    rather than passing.
    """

    def __call__(self, evaluation) -> bool:
        return evaluation.step >= 3


def test_19_last_evaluation_is_the_final_forward_call_and_off_by_one_where_it_should_be():
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    exhausted = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.uniform(4),
    )
    assert exhausted.n_evaluations == len(problem.calls) == 4
    # Equality against the model's own last recorded input, not merely
    # inequality against result.ensemble, which a pre-repair pair would also
    # satisfy.
    assert np.array_equal(
        np.asarray(exhausted.last_evaluation.ensemble), problem.calls[-1]
    )
    assert np.array_equal(
        np.asarray(exhausted.last_evaluation.predictions),
        np.asarray(problem.forward(problem.calls[-1])),
    )
    # The returned ensemble is one update past the final forward evaluation.
    assert not np.array_equal(
        np.asarray(exhausted.ensemble),
        np.asarray(exhausted.last_evaluation.ensemble),
    )

    # On a stopping-rule termination the state is unchanged, so they agree.
    stopped = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule.uniform(10), stop=_StopAtStepThree(),
    )
    assert np.array_equal(
        np.asarray(stopped.ensemble), np.asarray(stopped.last_evaluation.ensemble)
    )

    # A run that made no evaluation has none.
    finished = run(
        stopped.state, problem.forward, y, noise, schedule=FixedSchedule.uniform(3)
    )
    assert finished.n_evaluations == 0 and finished.last_evaluation is None


def test_20_inflation_and_the_update_see_the_true_ladder_and_are_applied_in_place():
    """Placement is asserted against an instrumented model, not assumed."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    seen: dict[str, list] = {"inflate": [], "update": []}
    shift = 0.125

    def recording_inflation(key, *, ensemble, step, beta, **_):
        seen["inflate"].append((step, float(beta)))
        return ensemble + shift

    def recording_update(key, *, ensemble, predictions, y, noise_cov, increment,
                         step, beta, **_):
        seen["update"].append((step, float(beta), float(increment)))
        return TransformUpdate()(
            key, ensemble=ensemble, predictions=predictions, y=y,
            noise_cov=noise_cov, increment=increment, step=step, beta=beta,
        )

    increments = (0.2, 0.3, 0.5)
    result = run(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule(increments),
        inflation=recording_inflation, update=recording_update,
    )
    levels = [0.0, 0.2, 0.5]
    assert seen["inflate"] == [(t, levels[t]) for t in range(3)]
    assert seen["update"] == [
        (t, levels[t], increments[t]) for t in range(3)
    ]

    # The model's *first* recorded input is the initial ensemble plus the
    # shift, bit-exactly: inflation runs before the evaluation, and so the
    # ensemble the caller supplied is never itself evaluated.
    assert np.array_equal(problem.calls[0], np.asarray(problem.members + shift))
    # And the returned ensemble is an update output, never an inflation one.
    assert not np.array_equal(
        np.asarray(result.ensemble), np.asarray(result.last_evaluation.ensemble)
    )

    # A rule that varies with beta still gives an exactly resumable run.
    def beta_dependent(key, *, ensemble, predictions, y, noise_cov, increment,
                       step, beta, **_):
        scaled = TransformUpdate()(
            key, ensemble=ensemble, predictions=predictions, y=y,
            noise_cov=noise_cov, increment=increment, step=step, beta=beta,
        )
        return scaled * (1.0 + 0.01 * beta)

    whole = run(problem.state(), problem.forward, y, noise,
                schedule=FixedSchedule(increments), update=beta_dependent)
    part = run(problem.state(), problem.forward, y, noise,
               schedule=FixedSchedule(increments[:1]), update=beta_dependent)
    rest = run(part.state, problem.forward, y, noise,
               schedule=FixedSchedule(increments), update=beta_dependent)
    assert np.array_equal(np.asarray(rest.ensemble), np.asarray(whole.ensemble))


def test_21_every_record_field_agrees_with_the_evaluation_it_came_from():
    """The only guard against a plausible-scalar bug in any of eleven fields."""
    problem = _AffineProblem(J=9)
    increments = (0.15, 0.35, 0.2, 0.3)
    result = run(
        problem.state(), problem.forward, jnp.asarray(problem.y),
        problem.noise_cov, schedule=FixedSchedule(increments),
    )
    W = _recovered_whitener(problem.noise_cov, problem.N)
    level = 0.0
    for index, (record, members) in enumerate(
        zip(result.history, problem.calls, strict=True)
    ):
        predictions = np.asarray(members) @ problem.G.T
        residuals = (problem.y - predictions) @ W.T
        phi = 0.5 * np.sum(residuals**2, axis=1)
        anomalies = members - members.mean(axis=0)
        spread = np.linalg.norm(anomalies) / np.sqrt(
            (problem.J - 1) * problem.P
        )
        weights = np.exp(-increments[index] * (phi - phi.min()))

        assert int(record.step) == index
        assert int(record.n_valid) == problem.J
        assert float(record.beta) == pytest.approx(level, abs=8 * EPS)
        assert float(record.increment) == pytest.approx(increments[index])
        assert float(record.beta_next) == pytest.approx(
            level + increments[index], abs=8 * EPS
        )
        assert float(record.misfit_mean) == pytest.approx(phi.mean(), rel=1e-11)
        assert float(record.misfit_min) == pytest.approx(phi.min(), rel=1e-11)
        assert float(record.misfit_max) == pytest.approx(phi.max(), rel=1e-11)
        assert float(record.centre_misfit) == pytest.approx(
            0.5 * np.sum(residuals.mean(axis=0) ** 2), rel=1e-10
        )
        assert float(record.spread) == pytest.approx(spread, rel=1e-11)
        assert float(record.ess) == pytest.approx(
            weights.sum() ** 2 / (weights**2).sum(), rel=1e-11
        )
        level += increments[index]

    assert float(result.beta) == pytest.approx(sum(increments), abs=8 * EPS)


def test_22_the_parameter_spread_is_exact_against_a_closed_form():
    """A divisor of J and a missing 1/sqrt(P) are separately distinguishable."""
    J, P, N = 5, 3, 2

    # Every coordinate has empirical variance exactly c**2, so the field is c.
    for c in (1.0, 0.25, 7.0):
        members = _exact_moment_ensemble(J, np.zeros(P), c * np.eye(P))
        got = _spread_of(members, N)
        assert got == pytest.approx(c, rel=1e-12)

    # Distinct per-coordinate variances: the root mean square of the three.
    a, b, d = 1.0, 2.0, 4.0
    members = _exact_moment_ensemble(J, np.zeros(P), np.diag([a, b, d]))
    got = _spread_of(members, N)
    assert got == pytest.approx(np.sqrt((a**2 + b**2 + d**2) / 3), rel=1e-12)
    # The two errors this catches, at J = 5 and P = 3.
    assert got != pytest.approx(np.sqrt((a**2 + b**2 + d**2) / 3 * (J - 1) / J))
    assert got != pytest.approx(np.sqrt(a**2 + b**2 + d**2))


def _spread_of(members: np.ndarray, v_dim: int) -> float:
    """The ``rms_parameter_spread`` the layer reports for these members."""
    J = members.shape[0]
    state = EKIState(jnp.asarray(members), 0.0, 0, jax.random.key(0))
    evaluation = evaluate(
        state,
        lambda u: jnp.zeros((J, v_dim)),
        jnp.zeros(v_dim),
        PSDDiagonal(jnp.ones(v_dim)),
    )
    return float(evaluation.rms_parameter_spread)


def test_23_a_finished_ladder_is_a_no_op_and_restart_gives_the_full_one():
    """The trap the contract devotes a warning to, and which nothing tested."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    for schedule in (FixedSchedule.uniform(4), AdaptiveESSSchedule(beta_target=1.0)):
        finished = run(problem.state(), problem.forward, y, noise, schedule=schedule)
        assert finished.n_evaluations > 0
        calls_before = len(problem.calls)

        again = run(finished.state, problem.forward, y, noise, schedule=schedule)
        assert again.status == SCHEDULE_EXHAUSTED
        assert again.n_evaluations == 0
        assert again.n_updates == 0
        assert again.history == ()
        assert again.last_evaluation is None
        assert np.array_equal(
            np.asarray(again.ensemble), np.asarray(finished.ensemble)
        )
        assert len(problem.calls) == calls_before, "zero forward calls"

        restarted = run(
            finished.state.restart(), problem.forward, y, noise, schedule=schedule
        )
        assert restarted.n_evaluations == finished.n_evaluations
        assert float(restarted.state.beta) == pytest.approx(float(finished.beta))


def test_23_a_no_op_run_says_so_at_warning_level(caplog):
    problem = _AffineProblem()
    finished = run(
        problem.state(), problem.forward, jnp.asarray(problem.y), problem.noise_cov,
        schedule=FixedSchedule.uniform(2),
    )
    with caplog.at_level(logging.WARNING, logger="pyeki.eki"):
        run(finished.state, problem.forward, jnp.asarray(problem.y),
            problem.noise_cov, schedule=FixedSchedule.uniform(2))
    assert "no forward evaluations" in caplog.text
    assert "restart" in caplog.text


def test_24_the_driver_hands_the_update_both_repaired_blocks():
    """Repairing one block and not the other is a silent wrong answer.

    Obligation 9 exercises the helper; this exercises the driver, which is
    where the two blocks could diverge.
    """
    problem = _AffineProblem(J=6)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    shift = 0.0625
    received: dict[str, np.ndarray] = {}

    def failing(u):
        problem.calls.append(np.asarray(u))
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[2, 0].set(jnp.nan)

    def recording_update(key, *, ensemble, predictions, **_):
        received["ensemble"] = np.asarray(ensemble)
        received["predictions"] = np.asarray(predictions)
        return jnp.asarray(ensemble)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(
            problem.state(), failing, y, noise,
            schedule=FixedSchedule.uniform(1),
            inflation=lambda key, *, ensemble, step, beta, **_: ensemble + shift,
            update=recording_update,
        )

    inflated = jnp.asarray(problem.members + shift)
    raw_predictions = jnp.asarray(inflated) @ jnp.asarray(problem.G).T
    raw_predictions = raw_predictions.at[2, 0].set(jnp.nan)
    valid = jnp.all(jnp.isfinite(raw_predictions), axis=-1)
    want_ensemble, want_predictions = repair_failed_members(
        ensemble=inflated, predictions=raw_predictions, valid=valid
    )
    assert np.array_equal(received["ensemble"], np.asarray(want_ensemble))
    assert np.array_equal(received["predictions"], np.asarray(want_predictions))


@pytest.mark.parametrize(
    "schedule",
    [
        FixedSchedule.uniform(3),
        FixedSchedule.constant(0.3, 3),
        AdaptiveESSSchedule(beta_target=0.6, min_increment=0.2, max_increment=0.2),
        AdaptiveMisfitSchedule(beta_target=0.6, min_increment=0.2, max_increment=0.2),
    ],
)
@pytest.mark.parametrize("update", [TransformUpdate(), PathwiseUpdate()])
@pytest.mark.parametrize("inflation_kind", ["none", "multiplicative", "additive"])
@pytest.mark.parametrize("with_stop", [False, True])
def test_25_the_three_axes_compose(schedule, update, inflation_kind, with_stop):
    """The executable form of the orthogonality claim, over the whole matrix.

    Asserts only that each run terminates, reports a permitted status, stacks,
    stays finite, and matches an instrumented forward-call count. It also
    catches a driver that inspects one axis to decide another — an ``ess``
    computed only for the ESS schedule, say.
    """
    problem = _AffineProblem(P=2, N=3, J=5, seed=71)
    inflation = {
        "none": None,
        "multiplicative": MultiplicativeInflation(1.01),
        "additive": AdditiveInflation(
            DensePSD.from_matrix(jnp.eye(problem.P) * 0.01)
        ),
    }[inflation_kind]
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=schedule,
        update=update,
        inflation=inflation,
        stop=DiscrepancyStop(tau=1e-8) if with_stop else None,
        max_steps=20,
    )
    assert result.status in (SCHEDULE_EXHAUSTED, STOPPING_RULE)
    assert result.n_evaluations == len(problem.calls)
    assert np.all(np.isfinite(np.asarray(result.ensemble)))
    stacked = result.stacked
    assert stacked.batch_shape == (result.n_evaluations,)
    assert np.all(np.isfinite(np.asarray(stacked.ess)))
    assert np.all(np.asarray(stacked.ess) >= 1.0 - 1e-9)


# ---------------------------------------------------------------------------
# 26. every runnable block of the contract page
# ---------------------------------------------------------------------------


def test_26_the_two_form_example_runs():
    """The contract's opening example, both forms, verbatim in structure."""
    problem = _AffineProblem()
    key = jax.random.key(0)
    prior = Gaussian(
        jnp.asarray(problem.m0), DensePSD.from_matrix(jnp.asarray(problem.C0))
    )
    forward, y, noise_cov = problem.forward, jnp.asarray(problem.y), problem.noise_cov

    state = EKIState.from_prior(key, prior, n_members=64)

    sampled = run(state, forward, y, noise_cov, schedule=AdaptiveESSSchedule())
    ensemble = sampled.ensemble
    centre = sampled.mean
    assert ensemble.shape == (64, problem.P)
    assert centre.shape == (problem.P,)

    fit = run(
        state, forward, y, noise_cov,
        schedule=FixedSchedule.constant(1.0, n_steps=200),
        stop=DiscrepancyStop(tau=1.0),
        max_steps=200,
    )
    assert isinstance(fit.stop_fired, bool)  # False means the ladder ran out first


def test_26_the_pinned_prior_draw_and_restart_blocks_run():
    problem = _AffineProblem()
    key = jax.random.key(9)
    prior = Gaussian(
        jnp.asarray(problem.m0), DensePSD.from_matrix(jnp.asarray(problem.C0))
    )

    key_sample, key_state = jax.random.split(key)
    pinned = EKIState(prior.sample(key_sample, 8), 0.0, 0, key_state)
    assert np.array_equal(
        np.asarray(pinned.ensemble),
        np.asarray(EKIState.from_prior(key, prior, 8).ensemble),
    )

    state = run(
        pinned, problem.forward, jnp.asarray(problem.y), problem.noise_cov,
        schedule=FixedSchedule.uniform(10),
    ).state
    phase2 = state.restart()  # step = 0, beta = 0.0, same ensemble and key
    assert phase2.step == 0 and float(phase2.beta) == 0.0
    assert np.array_equal(np.asarray(phase2.ensemble), np.asarray(state.ensemble))


def test_26_the_backtracking_loop_runs_and_costs_what_the_contract_says():
    """One forward evaluation per step plus one per rejection."""
    problem = _AffineProblem()
    forward, y, noise_cov = problem.forward, jnp.asarray(problem.y), problem.noise_cov
    accepted = 0

    def done(_evaluation):
        return accepted >= 3

    s, delta = problem.state(), 1.0
    current = evaluate(s, forward, y, noise_cov)
    rejections = 0
    while not done(current):
        trial, record = assimilate(
            s, current, increment=delta, y=y, noise_cov=noise_cov
        )
        probe = evaluate(trial, forward, y, noise_cov)
        if probe.centre_misfit < current.centre_misfit:
            s, current, delta = trial, probe, delta * 1.5
            accepted += 1
        else:
            delta = delta / 2
            rejections += 1
        assert rejections < 20, "the loop must make progress"

    assert len(problem.calls) == 1 + accepted + rejections


def test_26_the_additive_inflation_definition_and_the_stacked_one_liner_run():
    problem = _AffineProblem()
    P, J = problem.P, problem.J
    cov = DensePSD.from_matrix(jnp.eye(P) * 0.01)
    key = jax.random.key(3)
    ensemble = jnp.asarray(problem.members)

    pert = Gaussian(jnp.zeros(P), cov).sample(key, J)
    written_out = ensemble + (pert - pert.mean(axis=0))
    assert np.array_equal(
        np.asarray(written_out),
        np.asarray(
            AdditiveInflation(cov)(key, ensemble=ensemble, step=0, beta=jnp.asarray(0.0))
        ),
    )

    result = run(
        problem.state(), problem.forward, jnp.asarray(problem.y), problem.noise_cov,
        schedule=FixedSchedule.uniform(4),
    )
    # plt.plot(result.stacked.step, result.stacked.misfit_mean)
    xs, ys = result.stacked.step, result.stacked.misfit_mean
    assert xs.shape == ys.shape == (4,)

    fit = Gaussian.from_samples(result.ensemble)
    assert fit.cov.diag().shape == (P,)
    assert fit.sample(jax.random.key(1), 1000).shape == (1000, P)


def test_26_the_eki_error_checkpoint_and_interrupted_result_patterns_run():
    problem = _AffineProblem()
    forward, y, noise_cov = problem.forward, jnp.asarray(problem.y), problem.noise_cov
    sched = FixedSchedule.constant(0.25, 40)
    checkpointed, diagnosed = [], []

    try:
        run(problem.state(), forward, y, noise_cov, schedule=sched, max_steps=3)
    except EKIError as exc:
        checkpointed.append(exc.state)      # resume from here after investigating
        diagnosed.append(exc.history)
    assert len(checkpointed) == 1 and len(diagnosed[0]) == 3

    records = []
    state, evaluation = problem.state(), None
    for yielded in iterate(problem.state(), forward, y, noise_cov, schedule=sched):
        state, record, evaluation = yielded
        records.append(record)
        if len(records) >= 2:
            break
    result = EKIResult(
        state=state, history=tuple(records),
        status=INTERRUPTED, last_evaluation=evaluation,
    )
    assert result.status == INTERRUPTED
    assert not result.stop_fired and not result.budget_complete
    assert result.n_evaluations == 2


def test_26_the_tikhonov_augmentation_needs_no_new_code_and_double_counts():
    """The augmentation runs, and its documented hazard is asserted exactly.

    Appending the parameters to the predictions and the prior mean to the data
    adds ``0.5 ||C0^-1/2 (u - m0)||^2`` to the tempered misfit. Started from a
    *prior* ensemble and run to beta = 1, the prior therefore enters twice, and
    the result is over-concentrated by exactly one extra copy of the prior
    precision. Pinned as an equality so that a later change cannot quietly
    "fix" it.
    """
    problem = _AffineProblem()
    forward, y = problem.forward, jnp.asarray(problem.y)
    noise_cov = problem.noise_cov
    prior = Gaussian(
        jnp.asarray(problem.m0), DensePSD.from_matrix(jnp.asarray(problem.C0))
    )

    forward_aug = lambda u: jnp.concatenate([forward(u), u], axis=-1)  # noqa: E731
    y_aug = jnp.concatenate([y, prior.mean])
    noise_aug = block_diag(noise_cov, prior.cov)

    result = run(
        problem.state(), forward_aug, y_aug, noise_aug,
        schedule=FixedSchedule((0.5, 0.25, 0.25)),
    )
    _, got_cov = _moments(result.ensemble)

    prior_precision = np.linalg.inv(problem.C0)
    data_precision = problem.G.T @ np.linalg.solve(problem.R, problem.G)
    # Once through the initial ensemble, once through the appended block.
    doubled = np.linalg.inv(2 * prior_precision + data_precision)
    _, honest = problem.posterior()
    scale = np.abs(doubled).max()
    assert np.abs(got_cov - doubled).max() < 1e3 * EPS * scale
    assert np.abs(got_cov - honest).max() > 1e6 * EPS * scale
    # Over-concentrated, in the direction the warning names.
    assert np.trace(got_cov) < np.trace(honest)


def test_26_the_external_executable_wrapper_of_the_guide_runs(tmp_path):
    """The one runnable recipe outside the contract page that earns a test.

    The wrapper obligation -- catch your own failures and return a non-finite
    row -- is the single thing the layer needs from a forward model beyond its
    shape, and nothing else here exercises it against a real process. The
    solver exits non-zero on a negative decay rate, which the prior puts mass
    on, so the failure path is reached without being contrived.
    """
    solver = tmp_path / "solver.py"
    solver.write_text(
        "import sys, numpy as np\n"
        "u = np.loadtxt(sys.argv[1])\n"
        "if u[1] < 0.0:\n"
        "    sys.exit('solver diverged: negative decay rate')\n"
        "np.savetxt(sys.argv[2], u[0] * np.exp(-u[1] * np.array([0.5, 1.0, 2.0])))\n"
    )
    v_dim = 3

    def forward(ensemble):
        members = np.asarray(ensemble)
        predictions = np.full((members.shape[0], v_dim), np.nan)
        for j, member in enumerate(members):
            member_in, member_out = tmp_path / f"in_{j}.txt", tmp_path / f"out_{j}.txt"
            member_out.unlink(missing_ok=True)
            np.savetxt(member_in, member)
            try:
                subprocess.run(
                    [sys.executable, str(solver), str(member_in), str(member_out)],
                    check=True, capture_output=True, timeout=60,
                )
                row = np.loadtxt(member_out)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    OSError, ValueError):
                continue
            if row.shape == (v_dim,):
                predictions[j] = row
        return predictions

    times = jnp.array([0.5, 1.0, 2.0])
    truth = jnp.array([2.0, 0.7])
    y = truth[0] * jnp.exp(-truth[1] * times) + jnp.array([0.02, -0.01, 0.015])
    noise = PSDDiagonal(jnp.full(v_dim, 0.01))
    prior = Gaussian(
        mean=jnp.array([1.0, 1.0]), cov=PSDDiagonal(jnp.array([1.0, 0.5]))
    )
    state = EKIState.from_prior(jax.random.key(0), prior, n_members=32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(state, forward, y, noise, schedule=AdaptiveESSSchedule())

    # The failure path was genuinely taken, and reported: the prior puts mass
    # on negative decay rates, on which the solver exits non-zero.
    assert result.min_n_valid < state.n_members
    assert [w for w in caught if "evaluations failed" in str(w.message)]
    # The numbers the guide prints, so its output cannot rot unnoticed.
    assert result.min_n_valid == 29
    assert np.allclose(np.asarray(result.mean), [2.0798, 0.7406], atol=5e-5)
    # A run driven entirely by subprocesses, returning a numpy array read
    # back off disk, still lands in the run's dtype.
    assert result.last_evaluation.predictions.dtype == jnp.float64
    assert np.abs(np.asarray(result.mean) - np.asarray(truth)).max() < 0.2


def test_27_the_forward_model_receives_what_the_contract_promises():
    """Concrete, two-dimensional, in the run's dtype, and post-inflation.

    The tracer assertion is the one that fails silently and late: a concrete
    argument is what makes a subprocess model legal, and a driver rewritten
    over ``lax.scan`` would break it without breaking any other test here.
    """
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    state = problem.state()
    seen = []

    def recording(u):
        assert isinstance(u, jax.Array)
        assert not isinstance(u, jax.core.Tracer)
        assert u.shape == (problem.J, problem.P)
        assert u.dtype == state.ensemble.dtype
        # np.asarray on the argument is a read-only view, which the guide's
        # wrapper depends on being safe to hold and unsafe to write.
        assert not np.asarray(u).flags.writeable
        seen.append(np.asarray(u))
        return u @ jnp.asarray(problem.G).T

    run(state, recording, y, noise, schedule=FixedSchedule.uniform(2))
    assert len(seen) == 2
    # The first call is on the state's own members, untouched.
    assert np.array_equal(seen[0], np.asarray(state.ensemble))

    # With an inflation, it is the inflated members -- not state.ensemble.
    seen.clear()
    factor = 3.0
    run(
        state, recording, y, noise, schedule=FixedSchedule.uniform(1),
        inflation=MultiplicativeInflation(factor),
    )
    members = np.asarray(state.ensemble)
    centre = members.mean(axis=0, keepdims=True)
    expected = centre + factor * (members - centre)
    assert np.abs(seen[0] - expected).max() < 64 * EPS * np.abs(expected).max()
    assert np.abs(seen[0] - members).max() > 0.0

    # Writing into that view raises, rather than silently corrupting the
    # ensemble: the failure a wrapper hits is loud.
    with pytest.raises(ValueError, match="read-only"):
        np.asarray(state.ensemble)[0, 0] = 1.0


def test_28_the_accepted_containers_are_a_promise_not_a_tolerance():
    """A jax array, a numpy array and a nested list give bit-identical runs."""
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    G = np.asarray(problem.G)

    # The same numbers in three containers: computed once, so this tests the
    # container and not whether numpy's BLAS and XLA agree bit for bit.
    def as_jax(u):
        return jnp.asarray(np.asarray(u) @ G.T)

    def as_numpy(u):
        return np.asarray(u) @ G.T

    def as_list(u):
        return (np.asarray(u) @ G.T).tolist()

    runs = [
        run(problem.state(), f, y, noise, schedule=FixedSchedule.uniform(3))
        for f in (as_jax, as_numpy, as_list)
    ]
    reference = np.asarray(runs[0].ensemble)
    for other in runs[1:]:
        # Bit-identical: the claim is that the container cannot matter.
        assert np.array_equal(np.asarray(other.ensemble), reference)
        assert other.last_evaluation.predictions.dtype == jnp.float64


def test_29_a_narrow_forward_model_is_promoted_and_warned_about_once():
    """Promotion widens only, warns once per run, and never demotes the state.

    "Exactly once" is what this pins. A per-step warning left to the
    caller's filter displays once under the default filter too, and would
    pass any test asserting only that a warning was seen.
    """
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    state = problem.state()

    def coarse(u):
        return (jnp.asarray(u) @ jnp.asarray(problem.G).T).astype(jnp.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(state, coarse, y, noise, schedule=FixedSchedule.uniform(4))
    promotions = [w for w in caught if "promoted" in str(w.message)]
    assert len(promotions) == 1
    assert "float32" in str(promotions[0].message)
    assert result.n_evaluations == 4
    # Promoted on receipt: nothing downstream sees the narrow dtype.
    assert result.last_evaluation.predictions.dtype == jnp.float64
    assert result.ensemble.dtype == jnp.float64

    # A float64 model on a float64 run is silent.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(state, problem.forward, y, noise, schedule=FixedSchedule.uniform(4))
    assert not [w for w in caught if "promoted" in str(w.message)]

    # Once per *run*, not once if the first step happened to promote. A model
    # that degrades only in some parameter regimes is the realistic case, and
    # a flag keyed off `state.step` would warn here not at all -- and never at
    # all on a resumed run, since `step` is cumulative.
    class _DegradesLater:
        def __init__(self):
            self.calls = 0

        def __call__(self, u):
            self.calls += 1
            v = jnp.asarray(u) @ jnp.asarray(problem.G).T
            return v if self.calls == 1 else v.astype(jnp.float32)

    late = _DegradesLater()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(state, late, y, noise, schedule=FixedSchedule.uniform(4))
    promotions = [w for w in caught if "promoted" in str(w.message)]
    assert late.calls == 4
    assert len(promotions) == 1, "the warning is once per run, not once at step 0"
    assert "float32" in str(promotions[0].message)

    # And once on a run resumed from a nonzero step.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run(result.state.restart(), coarse, y, noise,
            schedule=FixedSchedule.uniform(2))
    assert len([w for w in caught if "promoted" in str(w.message)]) == 1

    # The evaluate phase has no run to be once per, so it warns per call.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = evaluate(state, coarse, y, noise)
        evaluate(state, coarse, y, noise)
    assert len([w for w in caught if "promoted" in str(w.message)]) == 2
    assert first.predictions.dtype == jnp.float64

    # A non-floating return is still a ValueError, never a conversion.
    with pytest.raises(ValueError, match="real floating dtype"):
        run(state, lambda u: np.asarray(np.asarray(u) @ problem.G.T, np.int64),
            y, noise, schedule=FixedSchedule.uniform(2))


def test_29_promotion_only_ever_widens():
    """A model wider than the run is left exactly as it is."""
    problem = _AffineProblem(J=8)
    state = problem.state()
    working = state.ensemble.dtype
    narrow = jnp.zeros((problem.J, problem.N), jnp.float32)
    wide = jnp.zeros((problem.J, problem.N), working)

    got, arrived = _check_predictions(narrow, problem.J, problem.N, working)
    assert got.dtype == working and arrived == jnp.float32
    got, arrived = _check_predictions(wide, problem.J, problem.N, working)
    assert got.dtype == working and arrived is None
    # Against a float32 run, a float64 model is not demoted and not warned on.
    got, arrived = _check_predictions(wide, problem.J, problem.N, jnp.float32)
    assert got.dtype == working and arrived is None


def test_30_the_evaluation_and_update_counts_hold_on_every_exit():
    """One evaluation per step, plus one when stopping needed a look.

    Asserted as equalities in both directions on all four termination paths.
    "At most one apart" would pass an implementation that spent a needless
    evaluation on the declarative paths, which is the regression this guards:
    the exhaustion check reads only ``step`` and ``beta``, so it must exit
    before evaluating.
    """
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    class _Counted:
        def __init__(self):
            self.calls = 0

        def __call__(self, u):
            self.calls += 1
            return jnp.asarray(u) @ jnp.asarray(problem.G).T

    class _StopAt:
        def __init__(self, k):
            self.k = k

        def __call__(self, evaluation):
            return int(evaluation.step) >= self.k

    class _NoneAt:
        n_steps, beta_target = None, None

        def __init__(self, k):
            self.k = k

        def next_increment(self, evaluation):
            return None if int(evaluation.step) >= self.k else 0.1

    cases = [
        ("fixed ladder", dict(schedule=FixedSchedule.uniform(5)), 0),
        ("budgeted adaptive", dict(schedule=AdaptiveESSSchedule()), 0),
        ("stopping rule", dict(schedule=FixedSchedule.constant(0.1, n_steps=50),
                               stop=_StopAt(3)), 1),
        ("increment None", dict(schedule=_NoneAt(3)), 1),
    ]
    for label, kwargs, terminal in cases:
        state = problem.state()
        model = _Counted()
        result = run(state, model, y, noise, **kwargs)
        assert result.n_evaluations == model.calls, label
        assert result.n_updates == int(result.state.step) - int(state.step), label
        # The exact relationship, in both directions.
        assert result.n_evaluations - result.n_updates == terminal, label
        # A terminal record is the zero-increment one, at most one, always last.
        zeros = [i for i, r in enumerate(result.history)
                 if float(r.increment) == 0.0]
        assert zeros == ([len(result.history) - 1] if terminal else []), label

    # max_steps is an exact cap on forward calls, hence on member evaluations.
    class _Unbounded:
        n_steps, beta_target = None, None

        def next_increment(self, evaluation):
            return 0.1

    for bound in (1, 3, 7):
        model = _Counted()
        with pytest.raises(EKIError, match="max_steps"):
            run(problem.state(), model, y, noise,
                schedule=_Unbounded(), max_steps=bound)
        # Exactly `bound` ensemble evaluations, so exactly J * bound member
        # evaluations: the hard budget a caller with an expensive model needs.
        assert model.calls == bound


# ===========================================================================
# Section 2 -- one targeted regression test per silent-failure class
#
# The list is derived from the contract's prose rather than curated: every
# place it says a mistake raises nothing, warns nobody, or looks normal earns
# an entry, and the two must be kept in step. Several classes are already
# pinned by a conformance test above and are named here rather than
# duplicated: the R/beta mis-scaling (test 2), the non-log-space ESS
# (test 5), a repair applied when nothing failed (test 9), a HistoryRecord
# field declared static (test 15), the safety bound checked before ladder
# exhaustion (test 4), the min/max inversion and the two clamp-precedence
# inversions (test 4), the bisection returning `hi` (test 4), a record field
# disagreeing with its evaluation (test 21), two forward evaluations per step
# (tests 4, 16 and 19), chaining a fresh ladder onto a finished state
# (test 23), DiscrepancyStop on a budgeted ladder (test 19), and the Tikhonov
# augmentation at beta = 1 (test 26).
# ===========================================================================


def test_regression_inflation_scales_the_anomalies_not_the_covariance():
    """``anomaly_factor`` is the field's name because the two conventions differ.

    A caller passing an intended *variance* inflation of 1.2 gets 1.44. The
    error is invisible at the small values normally used, where r and
    sqrt(gamma) barely differ, and severe at large ones — which is why the
    name pins the convention and this test pins the name.
    """
    rng = np.random.default_rng(83)
    ensemble = jnp.asarray(rng.normal(size=(12, 3)))
    _, before = _moments(ensemble)
    inflated = MultiplicativeInflation(1.2)(
        jax.random.key(0), ensemble=ensemble, step=0, beta=jnp.asarray(0.0)
    )
    _, after = _moments(inflated)
    assert np.abs(after - 1.44 * before).max() < 64 * EPS * np.abs(before).max()
    assert np.abs(after - 1.2 * before).max() > 1e-3 * np.abs(before).max()


def test_regression_the_repair_does_not_rescale_the_surviving_members():
    """The moment-exact variant would inflate silently, by a factor that is not small.

    At J = 100 with 10% of members failing, the rescaling is sqrt(99/89) ≈
    1.055 — a 5.5% re-injection of spread per step, larger than the
    multiplicative inflation factors practitioners actually use, applied by
    default under the name "repair".
    """
    rng = np.random.default_rng(89)
    J, P, N = 100, 4, 3
    ensemble = jnp.asarray(rng.normal(size=(J, P)))
    predictions = jnp.asarray(rng.normal(size=(J, N)))
    mask = np.ones(J, dtype=bool)
    mask[:10] = False
    valid = jnp.asarray(mask)

    repaired, _ = repair_failed_members(
        ensemble=ensemble, predictions=predictions, valid=valid
    )
    assert np.array_equal(np.asarray(repaired)[mask], np.asarray(ensemble)[mask])

    n_valid = int(mask.sum())
    rescaling = np.sqrt((J - 1) / (n_valid - 1))
    assert rescaling == pytest.approx(np.sqrt(99 / 89), rel=1e-12)
    u_hat = np.asarray(ensemble)[mask].mean(axis=0)
    moment_exact = u_hat + (np.asarray(ensemble)[mask] - u_hat) * rescaling
    assert np.abs(np.asarray(repaired)[mask] - moment_exact).max() > 1e-3


def test_regression_misfits_are_computed_after_the_repair():
    """A failed member would otherwise poison every statistic and criterion.

    After repair its prediction is the valid centre, so it contributes the
    centre's misfit — which sits *below* the valid members' mean misfit, a
    downward bias the contract names rather than hides.
    """
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[1, 0].set(jnp.nan)

    evaluation = evaluate(problem.state(), failing, y, noise)
    assert np.all(np.isfinite(np.asarray(evaluation.misfits)))
    assert evaluation.n_valid == problem.J - 1

    mask = np.ones(problem.J, dtype=bool)
    mask[1] = False
    valid_predictions = (problem.members @ problem.G.T)[mask]
    v_hat = valid_predictions.mean(axis=0)
    W = _recovered_whitener(noise, problem.N)
    want_repaired = 0.5 * float(np.sum((W @ (problem.y - v_hat)) ** 2))
    assert float(evaluation.misfits[1]) == pytest.approx(want_repaired, rel=1e-10)

    valid_misfits = 0.5 * np.sum(((problem.y - valid_predictions) @ W.T) ** 2, axis=1)
    assert want_repaired < valid_misfits.mean()


def test_regression_a_schedule_that_counts_its_own_calls_is_caught():
    """Purity is what makes a run resumable, and the harness is what catches it."""
    from pyeki.eki.testing import check_schedule, synthetic_evaluation

    class _CountsItsCalls:
        n_steps, beta_target = None, 1.0

        def __init__(self):
            self.seen = 0

        def next_increment(self, evaluation):
            self.seen += 1
            return 0.1 * self.seen

    with pytest.raises(AssertionError, match="not pure"):
        check_schedule(_CountsItsCalls(), synthetic_evaluation())


def test_regression_the_key_split_is_three_way_in_a_pinned_order():
    """No numeric test can catch this in the default configuration.

    The default consumes no randomness at all, so the guard is a
    ``jax.random.key_data`` snapshot of a multi-step ``PathwiseUpdate`` run,
    together with the property the fixed arity exists for: turning inflation
    on must not shift the update's stream.
    """
    problem = _AffineProblem(P=2, N=2, J=4, seed=61)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    common = dict(schedule=FixedSchedule.uniform(3), update=PathwiseUpdate())

    without = run(problem.state(seed=2), problem.forward, y, noise, **common)
    identity = lambda key, *, ensemble, step, beta, **_: ensemble  # noqa: E731
    with_inflation = run(
        problem.state(seed=2), problem.forward, y, noise,
        inflation=identity, **common,
    )
    assert np.array_equal(
        np.asarray(without.ensemble), np.asarray(with_inflation.ensemble)
    ), "turning inflation on shifted the update's random stream"

    # The order and the arity, snapshotted through the state's own key.
    assert [int(x) for x in np.asarray(jax.random.key_data(without.state.key))] == [
        916975276,
        3797780651,
    ]
    key = problem.state(seed=2).key
    for _ in range(3):
        key, _, _ = jax.random.split(key, 3)
    assert np.array_equal(
        np.asarray(jax.random.key_data(without.state.key)),
        np.asarray(jax.random.key_data(key)),
    )


def test_regression_a_fill_value_model_stalls_an_adaptive_ladder_silently():
    """Finite nonsense is invisible to the layer, and the ladder crawls.

    A solver returning a sentinel such as -9999 produces an enormous misfit
    for one member, which an adaptive schedule reads as genuine ensemble
    disagreement and answers by shrinking the increment — here all the way to
    the floor, so a one-step ladder becomes a fifty-step one. Nothing raises
    and nothing warns; ``n_valid`` reports every member valid. Documented as
    a behaviour rather than fixed, because the layer cannot see it.

    Scaled down to a budget of 0.05 against the same floor, so the stall is
    exercised at 5% of the cost the shipped defaults would incur.
    """
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    schedule = AdaptiveMisfitSchedule(beta_target=0.05, min_increment=1e-3)

    clean = run(
        problem.state(), problem.forward, y, noise, schedule=schedule, max_steps=50
    )
    assert clean.n_evaluations == 1

    def with_fill_value(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[0].set(-9999.0)

    stalled = run(
        problem.state(), with_fill_value, y, noise, schedule=schedule, max_steps=50
    )
    assert stalled.min_n_valid == problem.J, "every member looks valid"
    assert stalled.n_evaluations == 50
    assert np.all(np.asarray(stalled.stacked.increment) == pytest.approx(1e-3))


def test_regression_a_systematically_failing_member_is_visible_only_in_n_valid():
    """The run completes and looks normal; three things say otherwise."""
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    def always_fails_member_three(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[3, 0].set(jnp.nan)

    with pytest.warns(UserWarning, match="forward-model evaluations failed"):
        result = run(
            problem.state(), always_fails_member_three, y, noise,
            schedule=FixedSchedule.uniform(4),
        )
    assert result.status == SCHEDULE_EXHAUSTED
    assert np.all(np.isfinite(np.asarray(result.ensemble)))
    assert result.min_n_valid == problem.J - 1
    assert list(np.asarray(result.stacked.n_valid)) == [problem.J - 1] * 4


def test_regression_a_failing_step_logs_at_warning_level(caplog):
    problem = _AffineProblem(J=8)

    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[3, 0].set(jnp.nan)

    with caplog.at_level(logging.WARNING, logger="pyeki.eki"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(
            problem.state(), failing, jnp.asarray(problem.y), problem.noise_cov,
            schedule=FixedSchedule.uniform(2),
        )
    assert "were finite" in caplog.text


def test_regression_a_float32_update_cannot_quietly_demote_a_run():
    """Every downstream test would still pass at its own tolerance."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov

    def demoting(
        key, *, ensemble, predictions, y, noise_cov, increment, step, beta, **_
    ):
        return TransformUpdate()(
            key, ensemble=ensemble, predictions=predictions, y=y,
            noise_cov=noise_cov, increment=increment, step=step, beta=beta,
        ).astype(jnp.float32)

    with pytest.raises(ValueError, match="float32"):
        run(problem.state(), problem.forward, y, noise,
            schedule=FixedSchedule.uniform(2), update=demoting)

    # The asymmetry with a float32 forward model is deliberate: an update's
    # return becomes the state and demotes every later step, while a
    # prediction is consumed within one step and is promoted on receipt
    # (test 29). Either way the run stays float64.
    def coarse(u):
        return (jnp.asarray(u) @ jnp.asarray(problem.G).T).astype(jnp.float32)

    with pytest.warns(UserWarning, match="promoted"):
        result = run(
            problem.state(), coarse, y, noise, schedule=FixedSchedule.uniform(2)
        )
    assert result.ensemble.dtype == jnp.float64


def test_20_the_reported_spread_is_of_the_ensemble_that_was_evaluated():
    """``spread`` describes the post-inflation, post-repair members.

    A regression test for a mutation that survives every other test: reporting
    the spread of ``state.ensemble`` instead. Obligation 21 recomputes the
    field, but on a run with no inflation and no failures, where the two
    ensembles coincide and the mutation is invisible. Both departures are
    exercised here, separately.
    """
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    before = float(
        evaluate(problem.state(), problem.forward, y, noise).rms_parameter_spread
    )

    # Inflation: the spread must be the inflated one, exactly r times larger.
    factor = 3.0
    inflated = evaluate(
        problem.state(), problem.forward, y, noise,
        inflation=MultiplicativeInflation(factor),
    )
    assert float(inflated.rms_parameter_spread) == pytest.approx(
        factor * before, rel=1e-11
    )

    # Repair: the spread must be the repaired one, which is strictly smaller.
    def failing(u):
        v = jnp.asarray(u) @ jnp.asarray(problem.G).T
        return v.at[2, 0].set(jnp.nan)

    repaired = evaluate(problem.state(), failing, y, noise)
    members, _ = repair_failed_members(
        ensemble=jnp.asarray(problem.members),
        predictions=jnp.asarray(problem.members @ problem.G.T).at[2, 0].set(jnp.nan),
        valid=jnp.asarray([True, True, False, True, True, True, True, True]),
    )
    anomalies = np.asarray(members) - np.asarray(members).mean(axis=0)
    want = np.linalg.norm(anomalies) / np.sqrt((problem.J - 1) * problem.P)
    assert float(repaired.rms_parameter_spread) == pytest.approx(want, rel=1e-11)
    assert float(repaired.rms_parameter_spread) < before


def test_8_every_eki_error_path_carries_the_history_accumulated_so_far():
    """All four raise paths, not just the bound.

    Deleting the re-raise bookkeeping on either the evaluate or the apply path
    survives every other test, because only the ``max_steps`` payload was ever
    asserted.
    """
    problem = _AffineProblem(J=8)
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    ladder = FixedSchedule.constant(0.2, 30)

    def fails_at(step, indices):
        state = {"n": 0}

        def forward(u):
            v = jnp.asarray(u) @ jnp.asarray(problem.G).T
            hit = state["n"] == step
            state["n"] += 1
            return v.at[jnp.asarray(indices), 0].set(jnp.nan) if hit else v

        return forward

    # (a) fewer than two valid members, on the fourth step
    with pytest.raises(EKIError, match="At least 2 are required") as caught:
        run(problem.state(), fails_at(3, list(range(1, problem.J))), y, noise,
            schedule=ladder)
    assert len(caught.value.history) == 3
    assert caught.value.state.step == 3

    # (b) on_failure="raise", on the second step
    with pytest.raises(EKIError, match="on_failure='raise'") as caught:
        run(problem.state(), fails_at(1, [2]), y, noise, schedule=ladder,
            on_failure="raise")
    assert len(caught.value.history) == 1
    assert caught.value.state.step == 1

    # (c) a non-finite update, on the third step
    poison = {"n": 0}

    def sometimes_poisons(key, *, ensemble, predictions, y, noise_cov, increment,
                          step, beta, **_):
        poison["n"] += 1
        if poison["n"] == 3:
            return jnp.full_like(ensemble, jnp.nan)
        return TransformUpdate()(
            key, ensemble=ensemble, predictions=predictions, y=y,
            noise_cov=noise_cov, increment=increment, step=step, beta=beta,
        )

    with pytest.raises(EKIError, match="non-finite ensemble") as caught:
        run(problem.state(), problem.forward, y, noise, schedule=ladder,
            update=sometimes_poisons)
    assert len(caught.value.history) == 2
    assert caught.value.state.step == 2

    # Every payload is a tuple of records, never the driver's live list.
    assert isinstance(caught.value.history, tuple)
    resumed = run(caught.value.state, problem.forward, y, noise,
                  schedule=ladder, max_steps=30)
    assert resumed.n_evaluations == 28


def test_4_budget_tol_absorbs_a_ladder_that_lands_just_below_its_budget():
    """A regression test for `budget_tol`, which nothing else engages.

    Ten increments of 0.1 accumulate to 0.9999999999999999, strictly below
    1.0, so an exhaustion check written as `beta >= beta_target` would demand
    an eleventh step. The four-step `(0.3, 0.3, 0.3, 0.1)` ladder does not
    catch this: those increments happen to sum to exactly 1.0.
    """
    problem = _AffineProblem()
    accumulated = 0.0
    for _ in range(10):
        accumulated += 0.1
    assert accumulated < 1.0, "the fixture depends on this being inexact"

    schedule = AdaptiveMisfitSchedule(
        beta_target=1.0, min_increment=0.1, max_increment=0.1
    )
    result = run(
        problem.state(), problem.forward, jnp.asarray(problem.y),
        problem.noise_cov, schedule=schedule, max_steps=12,
    )
    assert result.n_evaluations == 10, "a bare `>=` would take an eleventh step"
    assert float(result.beta) == pytest.approx(accumulated, abs=8 * EPS)
    assert float(result.beta) < 1.0


def test_regression_the_anomalies_are_formed_stably():
    """Identical members give exactly zero spread, not round-off.

    ``jnp.mean`` of J bit-identical rows does not in general return the value
    it was given, so a naive `x - x.mean()` gives a collapsed ensemble
    spurious anomalies of about eps*|xbar| — which the gain then amplifies
    into a finite, nan-free, wrong update once the members are large.
    """
    from pyeki.eki.helpers import _anomalies

    for magnitude in (1.0, 6e23):
        collapsed = jnp.full((7, 3), magnitude)
        assert np.array_equal(np.asarray(_anomalies(collapsed)), np.zeros((7, 3)))
        naive = np.asarray(collapsed) - np.asarray(collapsed).mean(axis=0)
        if magnitude > 1.0:
            assert np.abs(naive).max() > 0.0, "the naive form is not exactly zero"

    # And the batched form subtracts per-member means, not per-batch ones.
    batched = jnp.asarray(np.random.default_rng(5).normal(size=(3, 3, 4)))
    got = np.asarray(_anomalies(batched))
    want = np.asarray(batched) - np.asarray(batched).mean(axis=-2, keepdims=True)
    assert np.abs(got - want).max() < 64 * EPS


def test_regression_a_vmapped_inflation_refuses_rather_than_broadcasting():
    """A family whose batch size equals P otherwise inflates per coordinate.

    Shape-correct, exception-free, and wrong: each parameter coordinate gets
    a different factor. Every other family-aware object in the layer refuses.
    """
    family = jax.vmap(MultiplicativeInflation)(jnp.asarray([1.0, 2.0, 3.0]))
    assert family.batch_shape == (3,)
    with pytest.raises(ValueError, match="vmapped family"):
        family(
            jax.random.key(0),
            ensemble=jnp.arange(18.0).reshape(6, 3),
            step=0,
            beta=jnp.asarray(0.0),
        )

    additive = jax.vmap(lambda d: AdditiveInflation(PSDDiagonal(d)))(
        jnp.ones((2, 3))
    )
    assert additive.batch_shape == (2,)
    with pytest.raises(ValueError, match="vmapped family"):
        additive(
            jax.random.key(0),
            ensemble=jnp.zeros((6, 3)),
            step=0,
            beta=jnp.asarray(0.0),
        )


def test_regression_a_nan_misfit_does_not_become_the_floor_step():
    """Both adaptive schedules must propagate a nan, not absorb it.

    Every comparison against a nan is False, so a bisection on the effective
    sample size would leave its bracket at zero and the floor would turn a
    poisoned ensemble into an ordinary-looking smallest step. The misfit
    schedule's guarded division already sends nan on; this pins the same for
    the ESS schedule, and pins that the driver then raises.
    """
    poisoned = _evaluation_with_misfits(np.array([1.0, np.nan, 2.0, 3.0]))
    for schedule in (AdaptiveESSSchedule(), AdaptiveMisfitSchedule()):
        got = float(schedule.next_increment(poisoned))
        assert np.isnan(got), f"{schedule!r} returned {got} on a nan misfit"
        assert got != pytest.approx(schedule.min_increment)

    # Through the driver a poisoned ensemble is caught earlier still, by the
    # validity check, so the schedule guard is the second line of defence.
    problem = _AffineProblem()
    with pytest.raises(EKIError, match="At least 2 are required"):
        run(problem.state(), lambda u: jnp.full((problem.J, problem.N), jnp.inf),
            jnp.asarray(problem.y), problem.noise_cov,
            schedule=AdaptiveESSSchedule())


def test_16_the_provenance_check_catches_a_stale_evaluation_not_a_foreign_run():
    """What the check establishes, and what it deliberately does not.

    It compares the evaluation's *position* — its step and its level — which
    catches re-applying an evaluation the state has moved past, the mistake
    the two-phase split makes easy. It does not establish that the two came
    from the same run, and cannot: the update reads its members from the
    evaluation and only its key from the state, so an evaluation from an
    unrelated problem at the same position is accepted. Both halves are
    asserted, so that the docstring's scope is the tested one.
    """
    first = _AffineProblem(seed=7)
    second = _AffineProblem(seed=77)
    y, noise = jnp.asarray(first.y), first.noise_cov
    assert (first.P, first.N, first.J) == (second.P, second.N, second.J)

    state = first.state()
    evaluation = evaluate(state, first.forward, y, noise)

    # Caught: the state has moved on, so the positions disagree.
    moved, _ = assimilate(state, evaluation, increment=0.25, y=y, noise_cov=noise)
    with pytest.raises(ValueError, match="different state"):
        assimilate(moved, evaluation, increment=0.25, y=y, noise_cov=noise)

    # Not caught: a foreign evaluation at the same position is accepted, and
    # the result is the foreign problem's answer rather than this one's.
    foreign = evaluate(
        second.state(), second.forward, jnp.asarray(second.y), second.noise_cov
    )
    assert foreign.step == state.step and float(foreign.beta) == float(state.beta)
    mixed, _ = assimilate(state, foreign, increment=0.25, y=y, noise_cov=noise)
    expected, _ = assimilate(
        second.state(), foreign, increment=0.25, y=y, noise_cov=noise
    )
    assert np.array_equal(np.asarray(mixed.ensemble), np.asarray(expected.ensemble))
    assert not np.array_equal(np.asarray(mixed.ensemble), np.asarray(moved.ensemble))


def test_14_n_valid_is_data_so_a_jitted_policy_does_not_retrace_per_step():
    """A static ``n_valid`` would give each step its own treedef.

    ``next_increment`` is the layer's documented extension point, and jitting
    one is the obvious thing to do with it; a static field on the object it
    receives would retrace it once per distinct valid-member count.
    """
    from pyeki.eki.testing import synthetic_evaluation

    counts = {"traces": 0}

    @jax.jit
    def criterion(evaluation):
        counts["traces"] += 1
        return jnp.mean(evaluation.misfits) + evaluation.n_valid

    for n_valid in (6, 5, 4, 3, 2):
        evaluation = dataclasses.replace(
            synthetic_evaluation(n_members=6), n_valid=jnp.asarray(n_valid)
        )
        criterion(evaluation)
    assert counts["traces"] == 1, "n_valid must not enter the treedef"

    # It is still an integer, still validated, and still stacks in a history.
    evaluation = synthetic_evaluation(n_members=6)
    assert evaluation.n_valid.shape == ()
    assert int(evaluation.n_valid) == 6
    with debug_checks():
        with pytest.raises(ValueError, match="n_valid"):
            dataclasses.replace(evaluation, n_valid=jnp.asarray(1))


def test_4_the_entry_budget_check_measures_the_remaining_budget():
    """A resumed run gets the bound the caller asked for.

    Bounding a resumption by the *whole* budget rejects calls that cannot
    reach the bound, and silently shrinks the allowance. The worst case is
    over what is left.
    """
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    schedule = AdaptiveESSSchedule(beta_target=1.0, min_increment=1e-3)

    # 0.01 of budget remains, so at most 10 further steps are possible.
    part_way = EKIState(
        jnp.asarray(problem.members), 0.99, 0, jax.random.key(0)
    )
    resumed = run(part_way, problem.forward, y, noise, schedule=schedule,
                  max_steps=10)
    assert resumed.budget_complete
    with pytest.raises(ValueError, match="cannot accommodate"):
        run(part_way, problem.forward, y, noise, schedule=schedule, max_steps=9)

    # A fresh run is still held to the whole budget: the check is a worst-case
    # guarantee, not a prediction of how many steps this problem will take.
    with pytest.raises(ValueError, match="cannot accommodate"):
        run(problem.state(), problem.forward, y, noise, schedule=schedule,
            max_steps=999)

    # And the quotient is robust to its own round-off: 1e-9 / 1e-12 is 1000
    # steps, not the 1001 a naive ceil of the float division reports.
    from pyeki.eki.driver import _steps_needed

    assert _steps_needed(1e-9, 1e-12) == 1000
    assert _steps_needed(1.0, 1e-3) == 1000
    assert _steps_needed(0.55, 0.1) == 6
    assert _steps_needed(-0.5, 1e-3) == 0
