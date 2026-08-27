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


def test_3_the_stochastic_update_composes_to_the_same_posterior():
    """``PathwiseUpdate`` reproduces a dense reference, and telescopes in mean.

    Two halves. Elementwise, for a fixed key, a ladder reproduces a
    hand-written dense perturbed-observation reference applied rung by rung.
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
    """One perturbed-observation rung, in plain dense linear algebra."""
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
    assert result.n_steps == len(increments)
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

    problem = _AffineProblem()
    result = run(
        problem.state(),
        problem.forward,
        jnp.asarray(problem.y),
        problem.noise_cov,
        schedule=_GiveUpAfterTwo(),
    )
    assert result.status == SCHEDULE_EXHAUSTED
    assert result.n_steps == 3
    assert float(result.stacked.increment[-1]) == 0.0
    assert float(result.stacked.beta_next[-1]) == float(result.stacked.beta[-1])
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
    assert again.n_steps == 0


@pytest.mark.parametrize(
    "schedule_type", [AdaptiveESSSchedule, AdaptiveMisfitSchedule]
)
def test_4_the_clamp_precedence_holds_in_all_three_regimes(schedule_type):
    """Floor beats criterion, ceiling binds, and the budget cap beats the floor."""
    evaluation = _evaluation_with_misfits(np.array([1.0, 1.0001, 0.9999, 1.00005]))

    # The floor binds: a huge criterion value would otherwise be tiny, and a
    # tiny one is lifted to the floor.
    floored = schedule_type(beta_target=None, min_increment=0.5, max_increment=2.0)
    assert float(floored.next_increment(evaluation)) >= 0.5

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


def test_4_rung_counts_are_exact_under_a_capped_criterion():
    """A budget of 1 under a ceiling of 0.3 takes exactly four rungs.

    One test pins the ``>=`` in the exhaustion check, ``budget_tol``,
    cap-beats-floor, and the absence of a trailing dribble rung. The
    criterion is ``inf`` because every misfit is identical, so the ceiling is
    the only thing choosing the increment.
    """

    class _ConstantModel:
        """Every member predicts the same thing, so every misfit is equal."""

        def __init__(self, n_members, n_obs):
            self.shape = (n_members, n_obs)
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
        assert result.n_steps == 4, schedule_type
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

    # A hand-written bisection at the same count, pinning both the iteration
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
    assert got == pytest.approx(theta / mean)
    assert theta / mean > np.sqrt(theta / variance)

    # cv < 1/sqrt(theta): the variance bound takes over.
    clustered = np.array([10.0, 10.4, 9.7, 10.1, 9.9])
    mean, variance = clustered.mean(), clustered.var(ddof=1)
    assert np.sqrt(variance) / mean < 1 / np.sqrt(theta)
    got = float(schedule.next_increment(_evaluation_with_misfits(clustered)))
    assert got == pytest.approx(np.sqrt(theta / variance))
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
    misfit_values: np.ndarray, *, beta: float = 0.0, step: int = 0, n_obs: int = 4
) -> Evaluation:
    """An ``Evaluation`` whose misfits are exactly the values given.

    Row ``j`` of the whitened residuals is placed on the first coordinate at
    ``sqrt(2 * phi_j)``, so ``0.5 * ||b_j||**2`` is exactly ``phi_j``.
    """
    n_members = misfit_values.size
    residuals = np.zeros((n_members, n_obs))
    residuals[:, 0] = np.sqrt(2.0 * misfit_values)
    members = np.asarray(
        _exact_moment_ensemble(n_members, np.zeros(2), np.eye(2))
    )
    return Evaluation(
        step=step,
        beta=beta,
        ensemble=jnp.asarray(members),
        predictions=jnp.zeros((n_members, n_obs)),
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
    evaluation = _evaluation_with_misfits(misfit_values, n_obs=80)

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
    n_obs = 4
    for tau in (0.5, 1.0, 2.0):
        rule = DiscrepancyStop(tau=tau)
        for centre_misfit in (0.1, 0.9, 2.0, 4.5, 9.0):
            evaluation = _evaluation_with_misfits(
                np.full(4, centre_misfit), n_obs=n_obs
            )
            # Every member has the same residual, so the centre misfit is it.
            assert float(evaluation.centre_misfit) == pytest.approx(centre_misfit)
            assert rule(evaluation) is (2.0 * centre_misfit <= tau**2 * n_obs)


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
    assert result.n_steps == 1
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
    assert resumed.n_steps == 34


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


def test_10_additive_inflation_from_cov_factorizes_once():
    """``from_cov`` holds a covariance whose ``factor`` is free, and agrees in law."""
    P, J = 4, 500
    cov = DensePSD.from_matrix(jnp.asarray(_psd(P, seed=37)))
    precomputed = AdditiveInflation.from_cov(cov)
    assert isinstance(precomputed.cov, PSDLowRank)
    assert np.abs(
        np.asarray(precomputed.cov.to_dense()) - np.asarray(cov.to_dense())
    ).max() < 1e-9

    ensemble = jnp.zeros((J, P))
    drawn = precomputed(
        jax.random.key(0), ensemble=ensemble, step=0, beta=jnp.asarray(0.0)
    )
    _, empirical = _moments(drawn)
    want = np.asarray(cov.to_dense())
    # Monte Carlo, at J = 500: a few relative standard errors of the estimate.
    assert np.abs(empirical - want).max() < 0.5 * np.abs(want).max()


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

    # Stop after four rungs and resume: the tail is bit-exact.
    partial_state = problem.state()
    taken = 0
    for partial_state, _, _ in iterate(
        problem.state(), problem.forward, y, noise, **common
    ):
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
    assert len(records) == first.n_steps
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
    functions rather than inspected by eye: a thirty-rung run must add no
    compilations that a three-rung run of the same problem has not already
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
        f"{after_long} over ten times as many rungs"
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
    """``advance`` is ``apply`` of ``evaluate``, and a rejected trial is free."""
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    state = problem.state()

    composed_state, composed_record = advance(
        state, problem.forward, y, noise, increment=0.3
    )
    calls_after_advance = len(problem.calls)
    evaluation = evaluate(state, problem.forward, y, noise)
    applied_state, applied_record = apply(
        state, evaluation, increment=0.3, y=y, noise_cov=noise
    )
    assert np.array_equal(
        np.asarray(composed_state.ensemble), np.asarray(applied_state.ensemble)
    )
    assert float(composed_record.ess) == float(applied_record.ess)
    assert calls_after_advance == 1

    # One evaluation, two trial increments, no further model calls.
    before = len(problem.calls)
    small, _ = apply(state, evaluation, increment=0.1, y=y, noise_cov=noise)
    large, _ = apply(state, evaluation, increment=0.9, y=y, noise_cov=noise)
    assert len(problem.calls) == before
    assert float(small.beta) == pytest.approx(0.1)
    assert float(large.beta) == pytest.approx(0.9)
    assert not np.array_equal(np.asarray(small.ensemble), np.asarray(large.ensemble))
    assert np.array_equal(np.asarray(state.ensemble), problem.members)
    assert evaluation.step == 0 and float(evaluation.beta) == 0.0

    # A mismatched evaluation is refused, and the increment is checked first.
    moved, _ = advance(state, problem.forward, y, noise, increment=0.2)
    with pytest.raises(ValueError, match="different state"):
        apply(moved, evaluation, increment=0.1, y=y, noise_cov=noise)
    before = len(problem.calls)
    for bad in (0.0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match="strictly positive"):
            apply(state, evaluation, increment=bad, y=y, noise_cov=noise)
    assert len(problem.calls) == before, (
        "apply must validate its increment before doing any array work"
    )


def test_16_iterate_yields_what_a_hand_written_two_phase_loop_yields():
    problem = _AffineProblem()
    y, noise = jnp.asarray(problem.y), problem.noise_cov
    increments = (0.2, 0.3, 0.5)

    by_hand = problem.state()
    for increment in increments:
        by_hand, _ = advance(by_hand, problem.forward, y, noise, increment=increment)

    driven = None
    for driven, _, _ in iterate(
        problem.state(), problem.forward, y, noise,
        schedule=FixedSchedule(increments),
    ):
        pass
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
