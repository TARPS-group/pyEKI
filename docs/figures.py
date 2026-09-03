"""The documentation's figures, and the Sphinx extension that generates them.

Every figure in the tutorials is produced by a function here and written into
``docs/_generated/figures`` when the documentation is built. Nothing is
committed: a figure cannot disagree with the code that made it, because it is
remade from that code on every build whose inputs have changed.

Contents
--------

============================== ===========================================
name                           page
============================== ===========================================
``01-one-step``                :doc:`tutorials/01-first-inversion`
``01-answer``                  :doc:`tutorials/01-first-inversion`
``02-trajectories``            :doc:`tutorials/02-reading-a-run`
``03-two-forms``               :doc:`tutorials/03-sampling-or-optimizing`
============================== ===========================================

Each figure function takes no arguments, is deterministic, and returns the
figure together with a dictionary of the numbers it plotted.
:func:`build` writes the figures; ``python docs/figures.py <dir>`` does the
same from the command line, for looking at one while writing a page.

Notes
-----
**How a figure is kept from rotting.** Two mechanisms, because the obvious one
is not enough on its own.

*The build regenerates it.* ``setup`` registers a ``builder-inited`` handler,
so a documentation build with warnings as errors fails on a figure whose code
raises. Regeneration is skipped when every output is newer than both this
module and every source file of the package, so an unchanged build is not
slowed by it; set ``PYEKI_DOCS_FIGURES=force`` to override.

*A test pins the numbers.* ``tests/test_tutorials.py`` calls each function and
asserts the values in the returned dictionary, which is why the dictionary
exists. That catches the case a build cannot: a figure that draws the wrong
array, or a change in the library that moves the answer. The values are the
ones *plotted*, not values recomputed by the test, so drawing the wrong array
fails the test rather than merely looking wrong. Conformance obligation 26 of
:doc:`eki-contract` is the rule this satisfies.

Pixels are deliberately not compared. Two machines with different fonts
rasterize the same figure differently, so a pixel comparison would be either
flaky or vacuous, and it would not catch a wrong number in a legible plot.

**A grid, not a closed form.** :func:`pyeki.toy.exponential_decay` has no
closed-form posterior, but it has two parameters, so its tempered densities
can be evaluated on a grid up to a constant, which is all a contour plot
needs. Grid quadrature is used for contours everywhere here, and for reference
moments only at :math:`\\beta = 1`, where it is converged: the mean and
standard deviation agree to five digits across two boxes and three
resolutions. A box that holds the posterior comfortably still truncates the
prior, so grid moments at small :math:`\\beta` are not reported.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pyeki  # noqa: E402, F401  -- enables float64 before any array exists
from pyeki import toy  # noqa: E402
from pyeki.eki import (  # noqa: E402
    AdaptiveESSSchedule,
    DiscrepancyStop,
    EKIState,
    FixedSchedule,
    iterate,
    misfits,
    run,
)
from pyeki.gauss import GaussianJoint  # noqa: E402

#: Where a build writes the figures, relative to this file.
OUTPUT_DIR = Path(__file__).parent / "_generated" / "figures"

#: The problem the tutorials work. Deterministic in its arguments.
PROBLEM = toy.exponential_decay()

#: The observation index whose joint with the decay rate tutorial 1 draws.
JOINT_INDEX = 3

# Colours. Deliberately few, and the same meaning on every figure: the prior
# and things derived from it in grey, the ensemble and its answer in the
# theme's green, the observation in rust, the truth in black.
C_PRIOR = "#8593a0"
C_ENSEMBLE = "#1a6d4f"
C_ALT = "#8c4a9e"
C_DATA = "#b5480a"
C_TRUTH = "#101010"
C_CONTOUR = "#5a6675"
C_COLLAPSED = "#2c4a7c"

#: Contour levels for a density scaled to a maximum of one. For a Gaussian
#: these are the one-, two- and three-standard-deviation ellipses.
LEVELS = (float(np.exp(-4.5)), float(np.exp(-2.0)), float(np.exp(-0.5)))


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------


def _style() -> None:
    """Apply the shared look. Called once per figure, before any axes exist."""
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def _log_terms(grid):
    """The prior log-density and the misfit of every point of ``grid``.

    Both terms are independent of the tempering level, so one call serves
    every level: the tempered log-density at ``beta`` is
    ``log_prior - beta * phi``, up to an additive constant.
    """
    log_prior = PROBLEM.prior.log_density(grid)
    phi = misfits(PROBLEM.y, PROBLEM.forward(grid), PROBLEM.noise_cov)
    return np.asarray(log_prior), np.asarray(phi)


def _grid(box, n):
    """A ``(n, n)`` grid over ``box = (amp_lo, amp_hi, rate_lo, rate_hi)``."""
    amp = jnp.linspace(box[0], box[1], n)
    rate = jnp.linspace(box[2], box[3], n)
    A, R = jnp.meshgrid(amp, rate, indexing="ij")
    return np.asarray(amp), np.asarray(rate), jnp.stack([A.ravel(), R.ravel()], -1)


def _moments(grid, log_pi):
    """Quadrature moments of an unnormalized log-density on a grid.

    Trustworthy only where the grid holds essentially all of the mass — see
    the module's notes. Used to size panels, and to report a reference at
    ``beta = 1``.
    """
    weights = np.exp(log_pi - log_pi.max())
    weights /= weights.sum()
    points = np.asarray(grid)
    mean = (weights[:, None] * points).sum(axis=0)
    sd = np.sqrt((weights[:, None] * (points - mean) ** 2).sum(axis=0))
    return mean, sd


#: A box wide enough for the prior, used to find where a level's mass sits.
_WIDE_BOX = (-3.0, 5.0, -3.0, 5.0)


def _tempered_moments(beta, n=200):
    """Mean and standard deviation of the tempered density at ``beta``."""
    _, _, grid = _grid(_WIDE_BOX, n)
    log_prior, phi = _log_terms(grid)
    return _moments(grid, log_prior - beta * phi)


def _tempered_contours(beta, box, n=160):
    """The tempered density at ``beta`` on ``box``, scaled to a maximum of 1."""
    amp, rate, grid = _grid(box, n)
    log_prior, phi = _log_terms(grid)
    log_pi = log_prior - beta * phi
    density = np.exp(log_pi - log_pi.max()).reshape(n, n)
    return amp, rate, density


def _box_around(mean, sd, half_widths=3.6):
    """A plotting box of ``mean +/- half_widths * sd``, in the grid's order."""
    lo = mean - half_widths * sd
    hi = mean + half_widths * sd
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


def _draw_contours(ax, beta, box, n=160):
    """Contours of the tempered density at ``beta``, on the axes' own box."""
    amp, rate, density = _tempered_contours(beta, box, n)
    ax.contour(
        amp,
        rate,
        density.T,
        levels=LEVELS,
        colors=C_CONTOUR,
        linewidths=0.8,
        alpha=0.9,
    )
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])


def _cloud(ax, ensemble, color=C_ENSEMBLE, label=None, size=5.0, alpha=0.55):
    """Scatter an ensemble in the (amplitude, rate) plane."""
    e = np.asarray(ensemble)
    ax.scatter(
        e[:, 0], e[:, 1], s=size, c=color, alpha=alpha, linewidths=0, label=label
    )


def _mark_truth(ax):
    """The parameters the observation was generated from."""
    ax.scatter(
        [float(PROBLEM.u_true[0])],
        [float(PROBLEM.u_true[1])],
        marker="+",
        s=70,
        c=C_TRUTH,
        linewidths=1.4,
        zorder=5,
        label="true parameters",
    )


def _prior_state(n_members, seed=0):
    """An initial state drawn from the problem's prior."""
    return EKIState.from_prior(jax.random.key(seed), PROBLEM.prior, n_members)


def _ladder(n_members, schedule, seed=0):
    """Run a ladder, returning ``(levels, clouds)`` including the final one.

    Each cloud is paired with the level of the evaluation it came from, read
    off ``Evaluation.beta`` so the pairing cannot drift. The last pair is the
    terminal ensemble at the level the run reached, which no evaluation
    covers — the final record's ``beta`` is the level *entering* the last
    step.
    """
    state = _prior_state(n_members, seed)
    levels, clouds = [], []
    for _, _, evaluation in iterate(
        state, PROBLEM.forward, PROBLEM.y, PROBLEM.noise_cov, schedule=schedule
    ):
        levels.append(float(evaluation.beta))
        clouds.append(np.asarray(evaluation.ensemble))
    result = run(
        state, PROBLEM.forward, PROBLEM.y, PROBLEM.noise_cov, schedule=schedule
    )
    levels.append(float(result.beta))
    clouds.append(np.asarray(result.ensemble))
    return levels, clouds


def _sd(ensemble):
    """Per-coordinate standard deviation, with the package's divisor."""
    return np.asarray(ensemble).std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# tutorial 1
# ---------------------------------------------------------------------------


def one_step():
    """One approximate conditioning step, in four panels.

    The prior ensemble; its predictions against the observation; the fitted
    Gaussian over one parameter and one prediction, with the conditioning
    line; and where a single step leaves the ensemble.
    """
    _style()
    n_members = 64
    state = _prior_state(n_members)
    ensemble = np.asarray(state.ensemble)
    predictions = np.asarray(PROBLEM.forward(state.ensemble))
    times = np.asarray(PROBLEM.times)
    y = np.asarray(PROBLEM.y)
    noise_sd = float(np.sqrt(np.asarray(PROBLEM.noise_cov.diag())[0]))

    joint = GaussianJoint.from_samples(
        u_samples=state.ensemble, v_samples=PROBLEM.forward(state.ensemble)
    )
    conditioned = joint.condition(PROBLEM.y, PROBLEM.noise_cov)
    updated = run(
        state,
        PROBLEM.forward,
        PROBLEM.y,
        PROBLEM.noise_cov,
        schedule=FixedSchedule.constant(1.0, n_steps=1),
    )

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))

    # (a) the prior ensemble in the parameter plane.
    ax = axes[0, 0]
    prior_mean = np.asarray(PROBLEM.prior.mean)
    prior_sd = np.sqrt(np.asarray(PROBLEM.prior.cov.diag()))
    box = _box_around(prior_mean, prior_sd, 3.2)
    _draw_contours(ax, 0.0, box)
    _cloud(ax, ensemble, C_PRIOR, size=7.0, alpha=0.8)
    _mark_truth(ax)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("decay rate")
    ax.set_title(f"(a) {n_members} members drawn from the prior")

    # (b) every member through the model, once.
    ax = axes[0, 1]
    for row in predictions:
        ax.plot(times, row, color=C_PRIOR, alpha=0.28, linewidth=0.7)
    ax.errorbar(
        times,
        y,
        yerr=noise_sd,
        fmt="o",
        ms=3.2,
        color=C_DATA,
        ecolor=C_DATA,
        elinewidth=1.0,
        capsize=2,
        zorder=5,
        label="observations",
    )
    ax.set_ylim(-1.0, 2.6)
    ax.set_xlabel("time")
    ax.set_ylabel("prediction")
    ax.set_title("(b) their predictions")
    ax.legend(loc="upper right")

    # (c) the fitted Gaussian over one parameter and one prediction.
    ax = axes[1, 0]
    index = JOINT_INDEX
    pair = np.column_stack([ensemble[:, 1], predictions[:, index]])
    pair_mean = pair.mean(axis=0)
    pair_cov = np.cov(pair, rowvar=False, ddof=1)
    ax.scatter(
        pair[:, 0], pair[:, 1], s=7.0, c=C_PRIOR, alpha=0.8, linewidths=0,
        label="members",
    )
    for scale in (1.0, 2.0):
        ax.plot(*_ellipse(pair_mean, pair_cov, scale), color=C_CONTOUR, lw=0.9)
    ax.axhline(y[index], color=C_DATA, lw=1.1, label="observed value")
    rate_mean = float(conditioned.mean[1])
    rate_sd = float(np.sqrt(conditioned.cov.diag()[1]))
    ax.axvspan(
        rate_mean - rate_sd,
        rate_mean + rate_sd,
        color=C_ENSEMBLE,
        alpha=0.16,
        linewidth=0,
        label="the rates it implies",
    )
    ax.axvline(rate_mean, color=C_ENSEMBLE, lw=1.1)
    ax.set_xlim(-1.6, 4.0)
    ax.set_ylim(-0.7, 2.4)
    ax.set_xlabel("decay rate")
    ax.set_ylabel(f"prediction at t = {times[index]:.2f}")
    ax.set_title("(c) the Gaussian that gets conditioned")
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="none")

    # (d) the same picture as (b), after that one step: on (b)'s own axes, so
    # the narrowing and what is left of it are both legible.
    ax = axes[1, 1]
    updated_predictions = np.asarray(PROBLEM.forward(updated.ensemble))
    for row in updated_predictions:
        ax.plot(times, row, color=C_ENSEMBLE, alpha=0.28, linewidth=0.7)
    ax.errorbar(
        times, y, yerr=noise_sd, fmt="o", ms=3.2, color=C_DATA, ecolor=C_DATA,
        elinewidth=1.0, capsize=2, zorder=5, label="observations",
    )
    ax.set_ylim(-1.0, 2.6)
    ax.set_xlabel("time")
    ax.set_ylabel("prediction")
    ax.set_title("(d) their predictions after one step")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig, {
        "n_members": n_members,
        "prior_predictive_above_panel": int((predictions[:, index] > 2.4).sum()),
        "prior_negative_rates": int((ensemble[:, 1] <= 0.0).sum()),
        "conditioned_mean": np.asarray(conditioned.mean),
        "conditioned_sd": np.sqrt(np.asarray(conditioned.cov.diag())),
        "one_step_mean": np.asarray(updated.mean),
        "one_step_sd": _sd(updated.ensemble),
        "prior_predictive_sd": predictions.std(axis=0, ddof=1),
        "one_step_predictive_sd": updated_predictions.std(axis=0, ddof=1),
    }


def _ellipse(mean, cov, n_sd, n_points=200):
    """Points on the ``n_sd`` contour of a two-dimensional Gaussian."""
    values, vectors = np.linalg.eigh(cov)
    angle = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.column_stack([np.cos(angle), np.sin(angle)])
    points = mean + n_sd * (circle * np.sqrt(values)) @ vectors.T
    return points[:, 0], points[:, 1]


def answer():
    """What a finished run gives you: a predictive fit, and a parameter cloud."""
    _style()
    n_members = 64
    state = _prior_state(n_members)
    result = run(
        state,
        PROBLEM.forward,
        PROBLEM.y,
        PROBLEM.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )
    times = np.asarray(PROBLEM.times)
    y = np.asarray(PROBLEM.y)
    noise_sd = float(np.sqrt(np.asarray(PROBLEM.noise_cov.diag())[0]))
    predictions = np.asarray(PROBLEM.forward(result.ensemble))

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    ax = axes[0]
    for row in predictions:
        ax.plot(times, row, color=C_ENSEMBLE, alpha=0.22, linewidth=0.7)
    ax.errorbar(
        times, y, yerr=noise_sd, fmt="o", ms=3.2, color=C_DATA, ecolor=C_DATA,
        elinewidth=1.0, capsize=2, zorder=5, label="observations",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("prediction")
    ax.set_title("what the answer predicts")
    ax.legend(loc="upper right")

    ax = axes[1]
    _cloud(ax, result.ensemble, C_ENSEMBLE, label="the answer", size=9.0, alpha=0.7)
    _mark_truth(ax)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("decay rate")
    ax.set_title("where the parameters ended up")
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig, {
        "n_members": n_members,
        "status": result.status,
        "n_evaluations": result.n_evaluations,
        "mean": np.asarray(result.mean),
        "sd": _sd(result.ensemble),
        "predictive_sd": np.asarray(predictions).std(axis=0, ddof=1),
    }


# ---------------------------------------------------------------------------
# tutorial 2
# ---------------------------------------------------------------------------


def trajectories():
    """Misfit, effective sample size and spread, for two ladders."""
    _style()
    n_members = 64
    state = _prior_state(n_members)
    runs = {}
    for label, schedule in (
        ("adaptive", AdaptiveESSSchedule()),
        ("three equal steps", FixedSchedule.uniform(3)),
    ):
        runs[label] = run(
            state, PROBLEM.forward, PROBLEM.y, PROBLEM.noise_cov, schedule=schedule
        )

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    styles = {
        "adaptive": dict(color=C_ENSEMBLE, marker="o"),
        "three equal steps": dict(color=C_ALT, marker="s"),
    }
    panels = [
        ("misfit_mean", "mean misfit", True),
        ("ess", "effective sample size", False),
        ("spread", "ensemble spread", True),
    ]
    for ax, (field, title, log) in zip(axes, panels, strict=True):
        for label, result in runs.items():
            history = result.stacked
            ax.plot(
                np.asarray(history.step),
                np.asarray(getattr(history, field)),
                label=label,
                markersize=3.4,
                **styles[label],
            )
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(title)
    axes[0].axhline(
        PROBLEM.v_dim / 2, color=C_DATA, lw=1.0, ls="--", label="noise alone"
    )
    axes[1].axhline(
        n_members / 2,
        color=C_DATA,
        lw=1.0,
        ls="--",
        label="the adaptive schedule's floor",
    )
    for ax in axes:
        ax.legend(loc="best")

    fig.tight_layout()
    return fig, {
        "n_members": n_members,
        "noise_floor": PROBLEM.v_dim / 2,
        **{
            f"{label.replace(' ', '_')}_{field}": np.asarray(
                getattr(result.stacked, field)
            )
            for label, result in runs.items()
            for field in ("beta", "ess", "misfit_mean", "spread")
        },
        **{
            f"{label.replace(' ', '_')}_sd": _sd(result.ensemble)
            for label, result in runs.items()
        },
    }


# ---------------------------------------------------------------------------
# tutorial 3
# ---------------------------------------------------------------------------


def two_forms():
    """The two destinations: a posterior ensemble, and a collapsing fit."""
    _style()
    n_members = 64
    state = _prior_state(n_members)
    sampled = run(
        state,
        PROBLEM.forward,
        PROBLEM.y,
        PROBLEM.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )
    stopped = run(
        state,
        PROBLEM.forward,
        PROBLEM.y,
        PROBLEM.noise_cov,
        schedule=FixedSchedule.constant(1.0, n_steps=200),
        stop=DiscrepancyStop(tau=1.0),
    )
    unstopped = run(
        state,
        PROBLEM.forward,
        PROBLEM.y,
        PROBLEM.noise_cov,
        schedule=FixedSchedule.constant(1.0, n_steps=30),
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    ax = axes[0]
    post_mean, post_sd = _tempered_moments(1.0)
    box = _box_around(post_mean, post_sd, 4.5)
    _draw_contours(ax, 1.0, box)
    _cloud(ax, sampled.ensemble, C_ENSEMBLE, label="sampling form", size=9.0)
    _cloud(
        ax, stopped.ensemble, C_ALT, label=r"optimization form, stopped", size=9.0
    )
    _cloud(
        ax,
        unstopped.ensemble,
        C_COLLAPSED,
        label=r"the same, run to $\beta = 30$",
        size=5.0,
        alpha=0.8,
    )
    _mark_truth(ax)
    ax.set_xlabel("amplitude")
    ax.set_ylabel("decay rate")
    ax.set_title("against the posterior's contours")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="none")

    ax = axes[1]
    history = unstopped.stacked
    ax.plot(
        np.asarray(history.beta),
        np.asarray(history.spread),
        color=C_ALT,
        marker="s",
        markersize=3.0,
        label="optimization form",
    )
    ax.axhline(
        float(np.mean(_sd(sampled.ensemble))),
        color=C_ENSEMBLE,
        lw=1.1,
        ls="--",
        label="sampling form, at $\\beta = 1$",
    )
    ax.axvline(
        float(stopped.beta),
        color=C_DATA,
        lw=1.0,
        ls=":",
        label=f"stopping rule fires, $\\beta$ = {float(stopped.beta):.0f}",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("ensemble spread")
    ax.set_title("the fit keeps collapsing")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig, {
        "n_members": n_members,
        "sampled_mean": np.asarray(sampled.mean),
        "sampled_sd": _sd(sampled.ensemble),
        "stopped_status": stopped.status,
        "stopped_beta": float(stopped.beta),
        "stopped_evaluations": stopped.n_evaluations,
        "stopped_mean": np.asarray(stopped.mean),
        "stopped_sd": _sd(stopped.ensemble),
        "unstopped_beta": float(unstopped.beta),
        "unstopped_sd": _sd(unstopped.ensemble),
        "unstopped_misfit": np.asarray(unstopped.stacked.misfit_mean),
        "reference_mean": post_mean,
        "reference_sd": post_sd,
    }


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

#: Every figure, by the file name it is written under.
FIGURES: dict[str, Callable[[], tuple[plt.Figure, dict]]] = {
    "01-one-step": one_step,
    "01-answer": answer,
    "02-trajectories": trajectories,
    "03-two-forms": two_forms,
}


def build(output_dir=OUTPUT_DIR, *, only=None) -> dict[str, dict]:
    """Write the figures as PNG files, and return what each one plotted.

    Parameters
    ----------
    output_dir
        The directory to write into. Created if it does not exist.
    only
        A figure name, or an iterable of them. Defaults to all of them.

    Returns
    -------
    dict
        The data dictionary of each figure built, by name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if only is None:
        names = list(FIGURES)
    elif isinstance(only, str):
        names = [only]
    else:
        names = list(only)

    plotted = {}
    for name in names:
        figure, data = FIGURES[name]()
        figure.savefig(output_dir / f"{name}.png")
        plt.close(figure)
        plotted[name] = data
    return plotted


def _newest_source_time() -> float:
    """The most recent modification time of this module and the package."""
    package = Path(pyeki.__file__).parent
    sources = [Path(__file__), *package.rglob("*.py")]
    return max(path.stat().st_mtime for path in sources)


def is_current(output_dir=OUTPUT_DIR) -> bool:
    """Whether every figure exists and postdates every source file."""
    output_dir = Path(output_dir)
    paths = [output_dir / f"{name}.png" for name in FIGURES]
    if not all(path.exists() for path in paths):
        return False
    return min(path.stat().st_mtime for path in paths) > _newest_source_time()


def setup(app):
    """Register the build-time hook. Called by Sphinx; see the module notes."""
    import os

    def generate(_app):
        force = os.environ.get("PYEKI_DOCS_FIGURES") == "force"
        if not force and is_current():
            return
        build()

    app.connect("builder-inited", generate)
    return {"version": pyeki.__version__, "parallel_read_safe": True}


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    for name, data in build(target).items():
        print(f"{name}: {target / (name + '.png')}")
        for key, value in data.items():
            print(f"    {key} = {value}")
