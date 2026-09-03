"""The tutorial series: its runnable blocks, its figures, and its prose numbers.

Conformance obligation 26 of the EKI contract requires every runnable block in
the documentation to be executed by a test with its printed output pinned.
Figures make that harder rather than optional, so there are three kinds of test
here:

1. **The blocks**, in the order each page runs them, with every value the page
   prints asserted to the digits it shows.
2. **The figures**, through :func:`figures.build` into a temporary directory,
   and then through the data dictionary each figure function returns. Those are
   the values the figure *plotted*, so a figure drawing the wrong array fails
   here rather than merely looking wrong. Pixels are deliberately not compared;
   ``docs/figures.py`` records why.
3. **The prose numbers that are not in any block** — the counts and ratios the
   pages state in a sentence. Each one is derived here from the same data the
   figure plotted, so the two cannot drift apart.

Two claims the pages rest on get their own tests: that an evaluation and the
record from the same step carry the same tempering level, which is what makes
an ensemble pairable with the distribution it belongs to; and that the grid
reference the pages compare against is converged, which is the only reason a
nonlinear problem has a reference at all.

The identity behind the ``centre_misfit`` gap is *not* re-tested here — it is
``tests/test_eki.py::test_11_the_centre_misfit_differs_from_the_mean_by_exactly_the_spread_term``.
This file pins only the numbers tutorial 2 prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import prints_as

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki import toy
from pyeki.eki import (
    AdaptiveESSSchedule,
    DiscrepancyStop,
    EKIState,
    FixedSchedule,
    effective_sample_size,
    iterate,
    misfits,
    run,
)
from pyeki.eki.testing import check_forward_model
from pyeki.gauss import Gaussian, GaussianJoint
from pyeki.linalg import PSDDiagonal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
import figures  # noqa: E402  -- docs/figures.py, on the path just above

#: The pages this file covers, for the reference scan below.
TUTORIAL_DIR = Path(__file__).resolve().parents[1] / "docs" / "tutorials"


# ===========================================================================
# 1. the runnable blocks of each page
# ===========================================================================


def test_1_tutorial_1_blocks_run():
    """Every runnable block of "Your first inversion", in order."""
    problem = toy.exponential_decay()

    assert (problem.u_dim, problem.v_dim) == (2, 12)
    prints_as(problem.u_true, [2.0, 1.5])
    prints_as(problem.y[:4], [1.3705, 0.929, 0.6856, 0.45])

    # The hand-written forward model, and its agreement with the problem's.
    times = problem.times
    prints_as([float(times[0]), float(times[-1])], [0.25, 3.0], decimals=2)

    def forward(ensemble):
        amplitude = ensemble[:, 0:1]
        rate = ensemble[:, 1:2]
        return amplitude * jnp.exp(-rate * times)

    ensemble = jnp.array([[2.0, 1.5], [1.0, 0.5]])
    assert forward(ensemble).shape == (2, 12)
    assert jnp.array_equal(forward(ensemble), problem.forward(ensemble))

    # The vmap wrapper, which the page offers as the alternative.
    def one_member(member):
        return member[0] * jnp.exp(-member[1] * times)

    vmapped = jax.vmap(one_member)
    assert jnp.array_equal(vmapped(ensemble), problem.forward(ensemble))

    assert check_forward_model(forward, u_dim=2, v_dim=12) is None

    # The prior and the noise the page writes out are the problem's own.
    prior = Gaussian(
        mean=jnp.array([1.0, 1.0]), cov=PSDDiagonal(jnp.array([1.0, 1.0]))
    )
    noise_cov = PSDDiagonal(jnp.full(12, 0.02**2))
    assert jnp.array_equal(prior.mean, problem.prior.mean)
    assert jnp.array_equal(prior.cov.to_dense(), problem.prior.cov.to_dense())
    assert jnp.array_equal(noise_cov.to_dense(), problem.noise_cov.to_dense())

    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    assert state.ensemble.shape == (64, 2)

    one_step = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=FixedSchedule.constant(1.0, n_steps=1),
    )
    prints_as(one_step.mean, [2.0129, 1.6505])
    prints_as(one_step.ensemble.std(axis=0, ddof=1), [0.0781, 0.7632])

    # The page's claim that doing the fit by hand gives the same numbers.
    joint = GaussianJoint.from_samples(
        u_samples=state.ensemble, v_samples=problem.forward(state.ensemble)
    )
    conditioned = joint.condition(problem.y, problem.noise_cov)
    prints_as(conditioned.mean, [2.0129, 1.6505])
    prints_as(conditioned.cov.diag() ** 0.5, [0.0781, 0.7632])
    # "to floating point": the page does not claim bit-identity, and the two
    # are not bit-identical -- they reach the moments by different routes.
    assert float(jnp.abs(conditioned.mean - one_step.mean).max()) < 1e-14
    assert (
        float(
            jnp.abs(
                jnp.sqrt(conditioned.cov.diag())
                - one_step.ensemble.std(axis=0, ddof=1)
            ).max()
        )
        < 1e-14
    )

    result = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )
    assert result.status == "schedule_exhausted"
    assert result.n_evaluations == 6
    assert float(result.beta) == 1.0
    assert result.n_evaluations * 64 == 384
    assert result.ensemble.shape == (64, 2)
    prints_as(result.mean, [1.9802, 1.4741])
    prints_as(result.ensemble.std(axis=0, ddof=1), [0.0396, 0.0363])

    # The prose's three comparisons against the true parameters.
    error = np.abs(np.asarray(result.mean) - np.asarray(problem.u_true))
    prints_as(error, [0.0198, 0.0259])
    assert error[0] < 0.02 and error[1] < 0.026
    rate_narrowing = float(
        one_step.ensemble.std(axis=0, ddof=1)[1]
        / result.ensemble.std(axis=0, ddof=1)[1]
    )
    assert 20.5 < rate_narrowing < 21.5, rate_narrowing


def test_1_tutorial_2_blocks_run():
    """Every runnable block of "Reading a run", in order."""
    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    result = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )

    assert result.status == "schedule_exhausted"
    assert result.budget_complete is True
    assert result.stop_fired is False
    assert float(result.beta) == 1.0
    assert result.min_n_valid == 64
    assert result.n_evaluations == 6
    assert result.n_completed_steps == 6

    history = result.stacked
    prints_as(history.beta, [0.0, 0.001, 0.0028, 0.0115, 0.0677, 0.5514])
    prints_as(history.increment, [0.001, 0.0018, 0.0087, 0.0562, 0.4837, 0.4486])

    # The eleven fields the page's table groups.
    for field in (
        "step",
        "n_valid",
        "beta",
        "increment",
        "beta_next",
        "misfit_mean",
        "misfit_min",
        "misfit_max",
        "centre_misfit",
        "spread",
        "ess",
    ):
        assert np.asarray(getattr(history, field)).shape == (6,)

    prints_as(history.misfit_min[-1], 4.5927)
    prints_as(history.misfit_mean[-1], 6.7961)
    prints_as(history.misfit_max[-1], 43.2523)
    assert float(history.misfit_max[-1] / history.misfit_mean[-1]) > 6.0

    evaluation = result.last_evaluation
    prints_as(evaluation.misfits[:4], [5.0189, 29.4126, 4.9678, 4.7952])
    prints_as(evaluation.centre_misfit, 4.6065)
    prints_as(evaluation.misfits.mean(), 6.7961)
    prints_as(evaluation.misfits.mean() - evaluation.centre_misfit, 2.1896)

    fitted = Gaussian.from_samples(result.ensemble)
    prints_as(fitted.mean, [1.9802, 1.4741])
    prints_as(fitted.cov.diag() ** 0.5, [0.0396, 0.0363])
    correlation = float(np.corrcoef(np.asarray(result.ensemble).T)[0, 1])
    prints_as(correlation, 0.83, decimals=2)

    phi = misfits(problem.y, problem.forward(result.ensemble), problem.noise_cov)
    prints_as(phi.mean(), 5.7186)
    prints_as(effective_sample_size(phi, 0.1), 61.7289)
    prints_as(effective_sample_size(phi, 1.0), 57.1571)

    # The note on the first step's effective sample size and the increment
    # floor: the schedule wanted a shorter step than 1e-3 and could not take
    # one, which is why 24.6 sits below the floor of 32 without being a bug.
    first = next(
        iter(
            iterate(
                state,
                problem.forward,
                problem.y,
                problem.noise_cov,
                schedule=AdaptiveESSSchedule(),
            )
        )
    )
    _, record, first_evaluation = first
    assert float(record.increment) == pytest.approx(1e-3)
    prints_as(record.ess, 24.5662)
    prints_as(effective_sample_size(first_evaluation.misfits, 1e-4), 53.32, 2)


def test_1_tutorial_3_blocks_run():
    """Every runnable block of "Sampling or optimizing", in order."""
    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)

    sampled = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )
    assert float(sampled.beta) == 1.0
    assert sampled.budget_complete is True
    prints_as(sampled.mean, [1.9802, 1.4741])
    prints_as(sampled.ensemble.std(axis=0, ddof=1), [0.0396, 0.0363])
    assert AdaptiveESSSchedule().beta_target == 1.0

    fit = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=FixedSchedule.constant(1.0, n_steps=200),
        stop=DiscrepancyStop(tau=1.0),
    )
    assert fit.status == "stopping_rule"
    assert float(fit.beta) == 3.0
    assert fit.n_evaluations == 4
    assert fit.n_completed_steps == 3
    prints_as(fit.mean, [1.9819, 1.4758])
    prints_as(fit.stacked.centre_misfit, [9654.0285, 577.3526, 6.5196, 4.5978])
    # The page reads the threshold off those values: 6.52 above, 4.60 below.
    threshold = 1.0**2 * problem.v_dim / 2
    assert threshold == 6.0
    assert float(fit.stacked.centre_misfit[-2]) > threshold
    assert float(fit.stacked.centre_misfit[-1]) <= threshold

    trap = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
        stop=DiscrepancyStop(tau=1.0),
    )
    prints_as(trap.beta, 0.0677)
    assert trap.stop_fired is True
    assert trap.budget_complete is False
    prints_as(trap.ensemble.std(axis=0, ddof=1), [0.1504, 0.1431])


def test_2_tutorial_3s_comparison_table():
    """The four rows of tutorial 3's table, and the sentences reading them.

    The table is the page's evidence for both of its claims -- that the
    optimization form gives an excellent point estimate and a spread that is
    not an uncertainty, and that the stopped run's plausible-looking spread is
    an accident of where it stopped. Every cell is asserted, since a row
    moving would invert one of those readings without changing the prose.
    """
    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    reference_mean, reference_sd = figures._tempered_moments(1.0)
    prints_as(reference_mean, [1.9769, 1.4719])
    prints_as(reference_sd, [0.0366, 0.0317])

    rows = {
        "sampling": dict(schedule=AdaptiveESSSchedule()),
        "tau2": dict(
            schedule=FixedSchedule.constant(1.0, n_steps=200),
            stop=DiscrepancyStop(tau=2.0),
        ),
        "tau1": dict(
            schedule=FixedSchedule.constant(1.0, n_steps=200),
            stop=DiscrepancyStop(tau=1.0),
        ),
        "beta30": dict(schedule=FixedSchedule.constant(1.0, n_steps=30)),
    }
    reached, calls, mean_error, sd_ratio = {}, {}, {}, {}
    for name, kwargs in rows.items():
        result = run(
            state, problem.forward, problem.y, problem.noise_cov, **kwargs
        )
        reached[name] = float(result.beta)
        calls[name] = result.n_evaluations
        mean_error[name] = float(
            np.abs(np.asarray(result.mean) - reference_mean).max()
        )
        sd_ratio[name] = np.asarray(
            result.ensemble.std(axis=0, ddof=1)
        ) / reference_sd

    assert reached == {"sampling": 1.0, "tau2": 2.0, "tau1": 3.0, "beta30": 30.0}
    assert calls == {"sampling": 6, "tau2": 3, "tau1": 4, "beta30": 30}
    prints_as(mean_error["sampling"], 0.0033, 4)
    prints_as(mean_error["tau2"], 0.0528, 4)
    prints_as(mean_error["tau1"], 0.0049, 4)
    prints_as(mean_error["beta30"], 0.0005, 4)
    prints_as(sd_ratio["sampling"], [1.08, 1.14], 2)
    prints_as(sd_ratio["tau2"], [1.4, 2.94], 2)
    prints_as(sd_ratio["tau1"], [0.87, 0.98], 2)
    prints_as(sd_ratio["beta30"], [0.19, 0.19], 2)

    # "the smallest error in the mean of the four rows, six times smaller than
    # the sampling form's".
    assert mean_error["beta30"] == min(mean_error.values())
    assert 5.5 < mean_error["sampling"] / mean_error["beta30"] < 7.5
    # "a spread five times too small", and "within 13% of the target's".
    assert 4.5 < 1.0 / sd_ratio["beta30"].max() < 5.5
    assert np.abs(sd_ratio["tau1"] - 1.0).max() < 0.13
    # "nearly three times too large in the rate" one step earlier: the point
    # is that the spread is set by where the run stopped, not by the target.
    assert sd_ratio["tau2"][1] > 2.8
    # "the trap's spread is about four times the target's".
    trap = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
        stop=DiscrepancyStop(tau=1.0),
    )
    trap_ratio = np.asarray(trap.ensemble.std(axis=0, ddof=1)) / reference_sd
    prints_as(trap_ratio, [4.11, 4.51], 2)


# ===========================================================================
# 2. the figures
# ===========================================================================


def test_3_every_figure_builds(tmp_path):
    """Every figure is written, and is a plausible PNG rather than an empty file."""
    plotted = figures.build(tmp_path)
    assert set(plotted) == set(figures.FIGURES)
    for name in figures.FIGURES:
        path = tmp_path / f"{name}.png"
        assert path.exists(), name
        assert path.stat().st_size > 20_000, (name, path.stat().st_size)
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_3_every_figure_a_page_references_is_generated():
    """No page references a figure nobody builds, and no figure is an orphan.

    A missing figure would already fail the documentation build, since it runs
    with warnings as errors. An *orphan* would not: a figure generated on every
    build and referenced by nothing is cost with no reader.
    """
    referenced = set()
    for page in sorted(TUTORIAL_DIR.glob("*.md")):
        for line in page.read_text().splitlines():
            marker = "../_generated/figures/"
            if marker in line:
                referenced.add(line.split(marker, 1)[1].removesuffix(".png").strip())
    assert referenced == set(figures.FIGURES)


def test_4_the_one_step_figure_plots_what_tutorial_1_says():
    """Tutorial 1's four-panel figure, and the sentences that read it."""
    data = figures.one_step()[1]
    assert data["n_members"] == 64
    prints_as(data["conditioned_mean"], [2.0129, 1.6505])
    prints_as(data["conditioned_sd"], [0.0781, 0.7632])
    prints_as(data["one_step_mean"], [2.0129, 1.6505])
    prints_as(data["one_step_sd"], [0.0781, 0.7632])

    # The caption's two counts.
    assert data["prior_predictive_above_panel"] == 3
    assert data["prior_negative_rates"] == 9

    # "at t = 1 the members' predictions have a standard deviation of 0.578,
    # against an observation error of 0.02".
    prints_as(data["one_step_predictive_sd"][figures.JOINT_INDEX], 0.5777)
    # "its standard deviation is 0.76, which is most of the prior's own 1.0".
    assert 0.7 < data["one_step_sd"][1] < 0.8


def test_4_the_answer_figure_plots_what_tutorial_1_says():
    """Tutorial 1's closing figure, and the predictive spread its caption gives."""
    data = figures.answer()[1]
    assert data["n_members"] == 64
    assert data["status"] == "schedule_exhausted"
    assert data["n_evaluations"] == 6
    prints_as(data["mean"], [1.9802, 1.4741])
    prints_as(data["sd"], [0.0396, 0.0363])
    # "a standard deviation of 0.0099 at t = 1 against an observation error
    # of 0.02" -- the fan is inside the error bars.
    prints_as(data["predictive_sd"][figures.JOINT_INDEX], 0.0099)
    assert data["predictive_sd"][figures.JOINT_INDEX] < 0.02


def test_4_the_trajectories_figure_plots_what_tutorial_2_says():
    """Tutorial 2's three panels, and every ratio the page reads off them."""
    data = figures.trajectories()[1]
    assert data["n_members"] == 64
    assert data["noise_floor"] == 6.0

    adaptive_ess = data["adaptive_ess"]
    prints_as(adaptive_ess, [24.5662, 32.0, 32.0, 32.0, 32.0, 57.4061])
    # "sits on 32 ... the last step is the exception, at 57.4, because by then
    # only 0.4486 of budget remained".
    assert np.allclose(adaptive_ess[1:-1], 32.0)
    prints_as(1.0 - data["adaptive_beta"][-1], 0.4486)
    prints_as(data["adaptive_misfit_mean"][-1], 6.7961)

    coarse_ess = data["three_equal_steps_ess"]
    prints_as(coarse_ess[0], 1.0002)
    assert coarse_ess[0] < 1.001

    # "its spread falls smoothly, by a factor between 1.3 and 2.7 per step"
    # against "a factor of 5.2 in one step".
    adaptive_ratios = data["adaptive_spread"][:-1] / data["adaptive_spread"][1:]
    coarse_ratios = (
        data["three_equal_steps_spread"][:-1]
        / data["three_equal_steps_spread"][1:]
    )
    assert 1.3 < adaptive_ratios.min() and adaptive_ratios.max() < 2.8
    prints_as(coarse_ratios.max(), 5.19, 2)

    # "35% larger in the amplitude and 47% larger in the rate".
    inflation = data["three_equal_steps_sd"] / data["adaptive_sd"]
    prints_as(inflation, [1.35, 1.47], 2)


def test_4_the_two_forms_figure_plots_what_tutorial_3_says():
    """Tutorial 3's figure, and the caption's stopping level."""
    data = figures.two_forms()[1]
    assert data["n_members"] == 64
    assert data["stopped_status"] == "stopping_rule"
    assert data["stopped_beta"] == 3.0
    assert data["stopped_evaluations"] == 4
    assert data["unstopped_beta"] == 30.0
    prints_as(data["sampled_sd"], [0.0396, 0.0363])
    prints_as(data["stopped_sd"], [0.0319, 0.031])
    prints_as(data["unstopped_sd"], [0.007, 0.0061])
    prints_as(data["reference_mean"], [1.9769, 1.4719])
    prints_as(data["reference_sd"], [0.0366, 0.0317])
    # The right panel's message: the spread keeps falling past the stop.
    assert data["unstopped_sd"].max() < data["stopped_sd"].min()


# ===========================================================================
# 3. the two claims the pages rest on
# ===========================================================================


def test_5_an_evaluation_and_its_record_carry_the_same_level():
    """``Evaluation.beta`` is bit-identical to the ``beta`` of its own record.

    Every figure that overlays an ensemble on the distribution it belongs to
    depends on this, and getting it wrong is silent: pairing each cloud with
    ``beta_next`` instead puts all of them one contour set out of step, which
    reads as the method tracking badly rather than as a plotting bug. Tutorial
    4's figure is built on it, and ``docs/figures.py`` pairs on
    ``evaluation.beta`` for this reason.

    Bit-identity is asserted rather than a tolerance because the two fields
    are the same value carried two ways, not two computations of it.
    """
    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=32)
    levels = []
    for _, record, evaluation in iterate(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
    ):
        assert jnp.array_equal(evaluation.beta, record.beta)
        assert int(evaluation.step) == int(record.step)
        # The distinct field, so that the assertion above is not vacuous.
        if float(record.increment) > 0.0:
            assert not jnp.array_equal(evaluation.beta, record.beta_next)
        levels.append(float(record.beta))
    assert levels[0] == 0.0
    assert levels == sorted(levels)


@pytest.mark.parametrize("n", [150, 200, 300, 400])
@pytest.mark.parametrize("refinements", [2, 3])
def test_6_the_grid_reference_is_converged_at_beta_one(n, refinements):
    """Tutorial 3's reference, across three grid resolutions.

    The nonlinear problem has no closed form, so the target's moments are
    quadrature. Tutorial 3 states them to four decimals and says they are
    converged; this is that claim, and it is why ``_tempered_moments`` refines
    its box rather than using one grid.
    """
    mean, sd = figures._tempered_moments(1.0, n=n, refinements=refinements)
    prints_as(mean, [1.9769, 1.4719])
    prints_as(sd, [0.0366, 0.0317])


@pytest.mark.parametrize("n", [200, 400])
def test_6_the_grid_reference_recovers_the_prior_at_beta_zero(n):
    """At ``beta = 0`` the target *is* the prior, which is known exactly.

    This is the one level where the quadrature can be checked against a
    closed form rather than against itself, so it is the strongest available
    evidence that the estimator is right and not merely self-consistent.
    """
    problem = toy.exponential_decay()
    mean, sd = figures._tempered_moments(0.0, n=n)
    prints_as(mean, np.asarray(problem.prior.mean), decimals=4)
    prints_as(sd, np.sqrt(np.asarray(problem.prior.cov.diag())), decimals=4)


def test_6_regression_one_unrefined_grid_is_not_enough():
    """Why the box is refined, stated as the failure it avoids.

    A single grid wide enough for the prior spans eight units in each
    parameter, and the target's standard deviation is 0.037 — about five grid
    points across its whole width at ``n = 200``. It reports a mean wrong in
    the fourth decimal, which is the precision tutorial 3 prints. And a box
    sized for the target truncates the prior: at ``beta = 0`` it puts the
    prior's mean at ``[1.4839, 1.3053]`` rather than ``[1, 1]``.

    Neither failure raises, and each is invisible in a contour plot.
    """
    unrefined = figures._tempered_moments(1.0, n=200, refinements=1)[0]
    prints_as(unrefined, [1.9772, 1.4719])
    assert not np.array_equal(np.round(unrefined, 4), [1.9769, 1.4719])

    narrow = figures._grid((0.5, 3.5, 0.2, 3.0), 160)[2]
    truncated, _ = figures._moments(narrow, figures._log_terms(narrow)[0])
    prints_as(truncated, [1.4839, 1.3053])


# ===========================================================================
# 4. regressions
# ===========================================================================


def test_7_regression_the_figure_cache_notices_a_changed_source(tmp_path):
    """``is_current`` is false for a missing figure and for a stale one.

    The build skips regeneration when every output postdates every source, so
    a cache that answered ``True`` too readily would let a figure rot through a
    documentation build that reported success.
    """
    assert figures.is_current(tmp_path) is False  # nothing written yet
    figures.build(tmp_path)
    assert figures.is_current(tmp_path) is True

    # A source file newer than the outputs must invalidate them.
    for path in tmp_path.glob("*.png"):
        stale = figures._newest_source_time() - 60.0
        import os

        os.utime(path, (stale, stale))
    assert figures.is_current(tmp_path) is False


def test_7_regression_a_toy_problem_is_not_passed_to_run():
    """The three-argument call the tutorials teach is the real signature.

    Every tutorial writes ``run(state, problem.forward, problem.y,
    problem.noise_cov, ...)``. A reader who tries the container instead should
    meet an error rather than a wrong answer, and the pages say so.
    """
    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=8)
    assert not callable(problem)
    with pytest.raises(TypeError):
        run(
            state,
            problem,
            problem.y,
            problem.noise_cov,
            schedule=FixedSchedule.constant(1.0, n_steps=1),
        )
