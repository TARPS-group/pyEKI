"""The three axes of a run, as protocols and their shipped implementations.

A variant of Ensemble Kalman Inversion is a choice on three independent axes,
and the driver is the same in every case.

=================== =========================================================
axis                question it answers
=================== =========================================================
:class:`Schedule`,  what is the increment :math:`\\Delta\\beta_t`, and when
:class:`StoppingRule` does the run end?
:class:`EnsembleUpdate` given the increment, what is the new ensemble?
:class:`Inflation`  what happens to the ensemble before each forward
                    evaluation?
=================== =========================================================

Each axis is a **protocol**, not a base class: an implementation is anything
with the right call signature — a shipped class, a user's frozen dataclass, or
a plain function where the protocol has a single method. Nothing subclasses
anything.

===================================== ======================================
shipped                               is
===================================== ======================================
:class:`TransformUpdate`              the deterministic square-root
                                      transform; **the default**
:class:`PathwiseUpdate`               the perturbed-observation update
:class:`FixedSchedule`                a ladder given in advance
:class:`AdaptiveESSSchedule`          adaptive tempering on the effective
                                      sample size
:class:`AdaptiveMisfitSchedule`       adaptive tempering on the noise level
:class:`DiscrepancyStop`              Morozov's discrepancy principle
:class:`MultiplicativeInflation`      scales the ensemble's anomalies
:class:`AdditiveInflation`            adds centred draws from a covariance
===================================== ======================================

Conventions shared by everything in the module:

- **Everything after the key is keyword-only.** Two of an update's arguments
  are arrays whose shapes coincide whenever :math:`P = N`, so a positional
  protocol would let ``ensemble`` and ``predictions`` be transposed with no
  error at all. Implementations **should** also accept and ignore ``**_``,
  which is the layer's forwards-compatibility seam.
- **Policies are pure and stateless.** A policy must be a pure function of
  its arguments and its own frozen fields, and must not carry step
  state — which is what makes a run resumable from an
  :class:`~pyeki.eki.EKIState` alone, and why a schedule receives the step
  index instead of counting calls.
- **Policies consume their key whole.** No policy splits, stores or advances
  a key; splitting is the driver's.

Notes
-----
The behaviour of this module is specified by the "Ensemble Kalman Inversion
contract" page of the documentation, which is normative, and the sources the
shipped policies reproduce are listed there.

This module implements no covariance arithmetic of its own. The two update
rules are two lines over :mod:`pyeki.gauss`, and the entire numerical content
of an update — the whitened-SVD kernel, the bounded gain multiplier, the
identity-completed square-root transform, the graceful degradation at zero
prediction anomalies — belongs to that layer.
"""
from __future__ import annotations

import dataclasses
import math
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from jax import Array, lax

from ..gauss import EmpiricalJoint, Gaussian
from ..linalg import PSDLinOp, static_field, value_check
from ..linalg.base import _broadcast_batch, _pytree_dataclass
from .helpers import (
    _anomalies,
    _check_field_rank,
    _check_not_vmap_family,
    _ess_from_misfits,
)
from .values import Evaluation

__all__ = [
    "AdaptiveESSSchedule",
    "AdaptiveMisfitSchedule",
    "AdditiveInflation",
    "DiscrepancyStop",
    "EnsembleUpdate",
    "FixedSchedule",
    "Inflation",
    "MultiplicativeInflation",
    "PathwiseUpdate",
    "Schedule",
    "StoppingRule",
    "TransformUpdate",
]


# ---------------------------------------------------------------------------
# the protocols
# ---------------------------------------------------------------------------


class EnsembleUpdate(Protocol):
    """One step of the ladder: the move that an increment produces.

    An implementation maps an ensemble, its predictions, the observation, the
    **base** noise covariance and a 0-d increment to a new ensemble. The two
    shipped rules do so by conditioning with ``noise_cov / increment``.

    Requirements on any implementation:

    - It receives both the increment and the absolute level, and the two mean
      different things: ``increment`` is how far this step moves,
      ``beta`` is where the step starts, and ``step`` is which step
      it is. The shipped rules use only the increment.
    - It consumes the key whole and is a deterministic function of its
      arguments including the key. A deterministic rule ignores the key.
    - It is ``jit``- and ``vmap``-safe with static shapes, and holds any
      arrays it needs as pytree data so that it can be passed through a trace
      boundary.

    Notes
    -----
    A rule that varies with ``beta`` or ``step`` — an annealed threshold, a
    decaying damping — breaks the telescoping identity the layer's exactness
    claim rests on, and the layer states that consequence rather than
    preventing it. Withholding the arguments would only push callers into
    keeping a counter inside the rule, which violates purity and silently
    breaks resumption.

    The base noise covariance and the increment are passed separately, rather
    than as the pre-scaled per-step operator, so that a rule needing the
    increment as a step size in its own right — a Langevin-type sampler — has
    it.
    """

    def __call__(
        self,
        key,
        *,
        ensemble: Array,
        predictions: Array,
        y: Array,
        noise_cov: PSDLinOp,
        increment: Array,
        step: int,
        beta: Array,
        **_,
    ) -> Array:
        """Return the new ``(J, P)`` ensemble."""
        ...


class Schedule(Protocol):
    """One method and two declarative attributes: how far each step moves.

    ``n_steps`` is the ladder's length in steps, or ``None`` if it is not
    step-bounded; ``beta_target`` is the temperature budget, or ``None`` for
    an unbounded ladder. Both are read, never called, and must be constant
    for the life of the object — they are static metadata on a frozen policy,
    not state. A schedule with both ``None`` is legal and unbounded, and must
    be ended by a stopping rule.

    ``next_increment(evaluation)`` returns the next increment — scalar,
    finite and strictly positive — or ``None`` to declare the ladder finished
    on evidence only the evaluation carries.

    Notes
    -----
    **The driver decides exhaustion, not the schedule.** A ladder is finished
    when ``step >= n_steps`` or ``beta >= beta_target - budget_tol``, a check
    that is misfit-independent and runs before any forward evaluation, so a
    run whose ladder is already finished pays nothing.

    The two exits mean different things and both are needed. The attributes
    say "I can tell from the bookkeeping that there is nothing left to do",
    and cost no evaluation; ``None`` says "having seen this ensemble, there
    is nothing useful left to do", and costs the evaluation that produced it.
    Neither shipped schedule returns ``None``.

    ``next_increment`` must be pure: a schedule that counted its own calls
    could not be resumed from a checkpoint, and ``evaluation.step`` is passed
    precisely so that it need not.
    """

    n_steps: int | None
    beta_target: float | None

    def next_increment(self, evaluation: Evaluation):
        """Return the next increment, or ``None`` if the ladder is finished."""
        ...


class StoppingRule(Protocol):
    """Whether to stop, given the current evaluation.

    Returns a Python ``bool``, is pure, and — like a schedule — must not hold
    step state. It is consulted before the increment is chosen, so a run
    that already fits the data takes no further update.

    Notes
    -----
    A misfit-based stopping rule necessarily costs one forward evaluation
    whose update is discarded: the misfits that trigger the stop are the
    misfits of an ensemble that is then returned unchanged. That evaluation
    is visible in the history as the terminal record, and it is inherent
    rather than an implementation artifact.
    """

    def __call__(self, evaluation: Evaluation) -> bool:
        """Return whether the run should end now."""
        ...


class Inflation(Protocol):
    """A shape-preserving transformation applied before each forward evaluation.

    Pure and ``jit``-safe. It runs at the top of every step, on the ensemble
    that is about to be evaluated and updated; ``step`` and ``beta`` are
    supplied because a decaying inflation schedule is common practice and the
    alternative is a rule that mutates a counter and cannot be resumed.

    Notes
    -----
    Placing inflation before the evaluation rather than after the analysis
    means the ensemble a run *returns* is never an inflated one, and that
    predictions always match the members they update. The two placements
    coincide in the interior of a run and differ at its two ends: the cost of
    this choice is that the **initial** ensemble is inflated before it is
    ever evaluated. A caller who needs the pristine initial ensemble
    evaluated drives the first step with ``inflation=None``.

    Applying inflation between the forward evaluation and the update would
    invalidate the predictions, and is forbidden.

    Inflation breaks the telescoping identity by design: a run with inflation
    on is not a tempering ladder for the target family, it is a deliberately
    widened variant.
    """

    def __call__(
        self, key, *, ensemble: Array, step: int, beta: Array, **_
    ) -> Array:
        """Return the inflated ``(J, P)`` ensemble."""
        ...


# ---------------------------------------------------------------------------
# update rules
# ---------------------------------------------------------------------------


@_pytree_dataclass
class TransformUpdate:
    """The deterministic square-root update; ignores the key. **The default.**

    Delegates to
    :meth:`EmpiricalJoint.transform_update <pyeki.gauss.EmpiricalJoint.transform_update>`
    with the tempered operator ``noise_cov / increment``. Holds no field.

    Notes
    -----
    Its output has the posterior moments of the fitted joint Gaussian
    exactly, per realization, and adds no Monte Carlo noise, at the cost of a
    deterministic member-to-member coupling. Telescoping therefore holds
    exactly under it, which is why it is the default: the layer's exactness
    claim is a property of the default configuration rather than of one a
    caller has to know to ask for. It is also deterministic, so a user's
    first two runs agree and a difference between them is a bug rather than a
    seed.
    """

    def __call__(
        self,
        key,
        *,
        ensemble,
        predictions,
        y,
        noise_cov,
        increment,
        step,
        beta,
        **_,
    ) -> Array:
        """Return the new ``(J, P)`` ensemble; ``key``, ``step`` and ``beta`` unused."""
        return _transform_update(ensemble, predictions, y, noise_cov, increment)

    def __repr__(self) -> str:
        """As ``TransformUpdate()``."""
        return "TransformUpdate()"


@_pytree_dataclass
class PathwiseUpdate:
    """The stochastic perturbed-observation update; consumes the key.

    Delegates to
    :meth:`EmpiricalJoint.pathwise_update <pyeki.gauss.EmpiricalJoint.pathwise_update>`
    with the tempered operator ``noise_cov / increment``. Holds no field.

    Notes
    -----
    Its output has the posterior moments of the fitted joint Gaussian in
    expectation, with per-realization spread of order :math:`KRK^\\top/J`, so
    the telescoping identity holds in expectation rather than per
    realization. It is the form for which the pathwise-sampling reading of a
    run is available, and the form most of the ensemble-Kalman-inversion
    literature is written in.
    """

    def __call__(
        self,
        key,
        *,
        ensemble,
        predictions,
        y,
        noise_cov,
        increment,
        step,
        beta,
        **_,
    ) -> Array:
        """Return the new ``(J, P)`` ensemble; ``step`` and ``beta`` unused."""
        return _pathwise_update(key, ensemble, predictions, y, noise_cov, increment)

    def __repr__(self) -> str:
        """As ``PathwiseUpdate()``."""
        return "PathwiseUpdate()"


@jax.jit
def _transform_update(ensemble, predictions, y, noise_cov, increment) -> Array:
    return EmpiricalJoint(
        u_samples=ensemble, v_samples=predictions
    ).transform_update(y, noise_cov / increment)


@jax.jit
def _pathwise_update(key, ensemble, predictions, y, noise_cov, increment) -> Array:
    return EmpiricalJoint(
        u_samples=ensemble, v_samples=predictions
    ).pathwise_update(key, y, noise_cov / increment)


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------


@_pytree_dataclass
class FixedSchedule:
    """A ladder given in advance: the increments, in order.

    Parameters
    ----------
    increments
        A non-empty tuple of increments, each strictly positive and finite.
        Static metadata, stored as a tuple of Python floats.

    Raises
    ------
    ValueError
        If the tuple is empty, or any increment is not strictly positive and
        finite.
    TypeError
        If ``increments`` is not a tuple of real numbers.

    Notes
    -----
    ``n_steps`` is the tuple's length and ``beta_target`` is ``None``:
    a fixed ladder is step-bounded, not budgeted, whatever its increments sum
    to. ``next_increment`` returns ``increments[evaluation.step]`` and
    ignores everything else.

    Because it indexes the state's *cumulative* step, a fixed schedule
    resumes a partially-completed ladder correctly and treats a finished
    state as finished — see :meth:`EKIState.restart
    <pyeki.eki.EKIState.restart>` before chaining one run onto another.

    Its ``repr`` summarizes rather than enumerates, since a 200-step
    optimization ladder would otherwise print 200 floats into every traceback
    and test id.
    """

    increments: tuple[float, ...] = static_field()

    def __post_init__(self) -> None:
        if not isinstance(self.increments, tuple):
            raise TypeError(
                f"FixedSchedule.increments: must be a tuple, got "
                f"{type(self.increments).__name__}. It is static metadata, so it "
                f"must be hashable."
            )
        if not self.increments:
            raise ValueError("FixedSchedule.increments: must not be empty")
        values = []
        for index, increment in enumerate(self.increments):
            try:
                value = float(increment)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"FixedSchedule.increments[{index}]: must be a real number, "
                    f"got {increment!r}"
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"FixedSchedule.increments[{index}]: every increment must be "
                    f"finite and strictly positive, got {increment!r}"
                )
            values.append(value)
        object.__setattr__(self, "increments", tuple(values))

    @classmethod
    def uniform(cls, n_steps: int) -> FixedSchedule:
        """``T`` equal steps of ``1 / T``: the sampling form's ladder.

        Reaches :math:`\\beta = 1` to round-off, not exactly:
        :math:`T \\cdot (1/T)` is not exactly 1 in floating point.

        Notes
        -----
        The layer's exactness claim therefore holds to round-off under this
        constructor rather than exactly. No correction is applied to the last
        step; a caller who needs the sum to be exact passes increments
        that are exact in binary, such as powers of two.
        """
        n_steps = _positive_int("FixedSchedule.uniform", "n_steps", n_steps)
        return cls((1.0 / n_steps,) * n_steps)

    @classmethod
    def constant(cls, increment: float, n_steps: int) -> FixedSchedule:
        """``T`` equal steps of ``increment``.

        The optimization form's ladder when ``increment * n_steps > 1``, and
        a single Kalman update at ``constant(1.0, 1)``.
        """
        n_steps = _positive_int("FixedSchedule.constant", "n_steps", n_steps)
        return cls((float(increment),) * n_steps)

    @property
    def n_steps(self) -> int:
        """The ladder's length in steps."""
        return len(self.increments)

    @property
    def beta_target(self) -> None:
        """``None``: a fixed ladder is step-bounded rather than budgeted."""
        return None

    def next_increment(self, evaluation: Evaluation) -> float:
        """The increment for ``evaluation.step``."""
        step = evaluation.step
        if not 0 <= step < len(self.increments):
            raise IndexError(
                f"{self!r}.next_increment: step {step} is outside a ladder of "
                f"{len(self.increments)} steps. The driver checks exhaustion "
                f"before evaluating, so this is reachable only by calling the "
                f"schedule directly."
            )
        return self.increments[step]

    def __repr__(self) -> str:
        """As ``FixedSchedule(n_steps=200, total=200.0)``; never raises."""
        try:
            total = float(f"{math.fsum(self.increments):.12g}")
            return (
                f"FixedSchedule(n_steps={len(self.increments)}, total={total!r})"
            )
        except Exception:
            return "<FixedSchedule (unprintable)>"


@_pytree_dataclass
class AdaptiveESSSchedule:
    """Adaptive tempering targeting an effective sample size.

    Measures the move by the effective sample size of the importance weights
    that would carry the ensemble from one target to the next, and finds the
    increment by bisection.

    Parameters
    ----------
    beta_target
        The temperature budget, or ``None`` for an unbounded ladder. Default
        ``1.0``; must be strictly positive when given.
    min_increment
        A floor guaranteeing progress. Default ``1e-3``; must be positive and
        no greater than ``max_increment``.
    max_increment
        A ceiling. Default ``1.0``; must be finite.
    ess_fraction
        The effective sample size sought, as a fraction of :math:`J`. Default
        ``0.5``; must lie in :math:`(0,\\ 1 - 10^{-6}]`.
    n_bisect
        How many bisection steps to run. Default ``50``; must be at
        least 1. Each step halves the bracket, so the returned
        increment sits within :math:`2^{-n}` of the criterion — about
        :math:`10^{-9}` at 30, and float64 round-off at the default.

    Raises
    ------
    ValueError
        If any field is outside its domain.

    Notes
    -----
    For an increment :math:`\\delta` the weights and their effective sample
    size are :math:`w_j = e^{-\\delta\\Phi_j}` and
    :math:`\\mathrm{ESS} = (\\sum_j w_j)^2 / \\sum_j w_j^2`, and the criterion
    is the largest :math:`\\delta` at which
    :math:`\\mathrm{ESS} \\ge \\texttt{ess\\_fraction}\\cdot J`, clamped by
    the floor, the ceiling and the remaining budget in that order.

    :math:`\\mathrm{ESS}` is monotone non-increasing in :math:`\\delta` and
    its derivative *vanishes* at :math:`\\delta = 0`, so the function is flat
    at the left endpoint and a derivative-based root find started there
    stalls. Bisection is the method for that reason, not for simplicity. The
    bracket is :math:`[0, \\min(\\delta_{\\max}, \\beta_{\\text{target}} -
    \\beta)]` and the **safe** end is returned, so the guarantee is one-sided:
    the returned increment meets the target before the floor is applied.

    The construction is the standard ESS-based adaptive tempering of the
    sequential Monte Carlo literature, used here purely as a step-size
    heuristic: pyEKI computes no importance weights, does no resampling, and
    makes no importance-sampling correctness claim.

    ``ess_fraction`` is bounded away from 1 because
    :math:`\\mathrm{ESS}(0)` evaluates to ``exp(log J)`` rather than exactly
    :math:`J`, so a fraction within round-off of 1 would make
    :math:`\\delta = 0` an invalid lower bracket. Its default of ``0.5`` is
    pyEKI's choice rather than a canonical value; the tempering literature
    uses targets between about a third and a half, and a smaller target takes
    longer steps and fewer of them.
    """

    beta_target: float | None = static_field(default=1.0)
    min_increment: float = static_field(default=1e-3)
    max_increment: float = static_field(default=1.0)
    ess_fraction: float = static_field(default=0.5)
    n_bisect: int = static_field(default=50)

    def __post_init__(self) -> None:
        _check_adaptive_fields(self)
        if not 0.0 < self.ess_fraction <= 1.0 - 1e-6:
            raise ValueError(
                f"AdaptiveESSSchedule.ess_fraction: must lie in (0, 1 - 1e-6], "
                f"got {self.ess_fraction}. It is bounded away from 1 because "
                f"ESS(0) evaluates to exp(log J) rather than exactly J."
            )
        if type(self.n_bisect) is not int or self.n_bisect < 1:
            raise ValueError(
                f"AdaptiveESSSchedule.n_bisect: must be a Python int of at least "
                f"1, got {self.n_bisect!r}"
            )

    @property
    def n_steps(self) -> None:
        """``None``: an adaptive ladder is budgeted rather than step-bounded."""
        return None

    def next_increment(self, evaluation: Evaluation) -> Array:
        """The bisected increment, clamped by floor, ceiling and budget."""
        delta_hi = _bracket_top(self, evaluation.beta)
        target = self.ess_fraction * evaluation.n_members
        unclamped = _bisect_ess(
            evaluation.misfits, delta_hi, target, self.n_bisect
        )
        return _clamp(self, unclamped, evaluation.beta)

    def __repr__(self) -> str:
        """The static fields, as ``AdaptiveESSSchedule(beta_target=1.0, ...)``."""
        return _policy_repr(self)


@_pytree_dataclass
class AdaptiveMisfitSchedule:
    """Adaptive tempering measured against the noise level, in closed form.

    Where :class:`AdaptiveESSSchedule` asks whether the ensemble can still
    describe the next target, this asks how much data the noise at this
    step can explain, and solves for the increment directly instead of
    bisecting.

    Parameters
    ----------
    beta_target
        The temperature budget, or ``None`` for an unbounded ladder. Default
        ``1.0``; must be strictly positive when given.
    min_increment
        A floor guaranteeing progress. Default ``1e-3``.
    max_increment
        A ceiling. Default ``1.0``; must be finite.
    divergence_budget
        The ceiling :math:`\\theta` on the Jeffreys divergence between
        consecutive tempered measures. Default ``None``, meaning
        :math:`N/2`, at which the schedule has no tuning parameter at all —
        the observation dimension supplies it. Must be finite and strictly
        positive when given.

    Raises
    ------
    ValueError
        If any field is outside its domain.

    Notes
    -----
    Write :math:`\\overline{\\Phi}` and :math:`\\sigma^2_\\Phi` for the
    ensemble mean and variance of the base misfits, the variance with the
    package's :math:`J-1` divisor. The criterion is

    .. math::

        \\delta^\\star = \\max\\left\\{
            \\frac{\\theta}{\\overline{\\Phi}},\\ \\
            \\sqrt{\\frac{\\theta}{\\sigma^2_\\Phi}} \\right\\},

    clamped by the floor, the ceiling and the remaining budget.

    **The ``max`` is deliberate and is not a ``min``.** The Jeffreys
    divergence between consecutive tempered measures is approximated by the
    *smaller* of two expressions valid in different regimes, and a ``min``
    bounded by :math:`\\theta` is satisfied exactly when :math:`\\delta` is
    below the *larger* of the two thresholds. Writing ``min`` here would
    impose both approximations at once, including whichever is invalid in the
    current regime, and would stall the ladder on the invalid one.

    Which bound binds is decided by the misfits' coefficient of variation:
    the mean bound is the larger exactly when
    :math:`\\sigma_\\Phi/\\overline{\\Phi} > 1/\\sqrt{\\theta}`, which at the
    default is the coefficient of variation of a :math:`\\chi^2_N` variate.
    For any appreciable :math:`N` that is the common case; the variance bound
    takes over for an unusually tightly clustered ensemble.

    Both bounds are **guarded divisions**, yielding :math:`+\\infty` rather
    than ``nan`` when their denominator vanishes — a collapsed ensemble gives
    zero misfit variance, and an ensemble that already fits the data exactly
    gives zero mean misfit. A ``nan`` misfit yields ``nan``, not
    :math:`+\\infty`, so a poisoned ensemble raises at the driver's increment
    validation rather than silently taking the largest allowed step.

    Compared with :class:`AdaptiveESSSchedule`, this takes far longer steps:
    under the increment it chooses, the effective sample size is at or near
    its floor. The two are not variants of one idea and they do not agree.
    Prefer the ESS schedule when the posterior ensemble is the deliverable,
    and this one when the fit is, or when model evaluations are scarce enough
    that a tuning-free schedule taking few large steps is worth its coarser
    approximation.
    """

    beta_target: float | None = static_field(default=1.0)
    min_increment: float = static_field(default=1e-3)
    max_increment: float = static_field(default=1.0)
    divergence_budget: float | None = static_field(default=None)

    def __post_init__(self) -> None:
        _check_adaptive_fields(self)
        if self.divergence_budget is not None:
            budget = self.divergence_budget
            if not isinstance(budget, (int, float)) or isinstance(budget, bool):
                raise ValueError(
                    f"AdaptiveMisfitSchedule.divergence_budget: must be None or a "
                    f"real number, got {budget!r}"
                )
            if not math.isfinite(budget) or budget <= 0.0:
                raise ValueError(
                    f"AdaptiveMisfitSchedule.divergence_budget: must be finite and "
                    f"strictly positive, got {budget}"
                )

    @property
    def n_steps(self) -> None:
        """``None``: an adaptive ladder is budgeted rather than step-bounded."""
        return None

    def next_increment(self, evaluation: Evaluation) -> Array:
        """The closed-form increment, clamped by floor, ceiling and budget."""
        theta = (
            0.5 * evaluation.v_dim
            if self.divergence_budget is None
            else float(self.divergence_budget)
        )
        return _clamp(
            self, _misfit_criterion(evaluation.misfits, theta), evaluation.beta
        )

    def __repr__(self) -> str:
        """The static fields, as ``AdaptiveMisfitSchedule(beta_target=1.0, ...)``."""
        return _policy_repr(self)


# ---------------------------------------------------------------------------
# stopping rules
# ---------------------------------------------------------------------------


@_pytree_dataclass
class DiscrepancyStop:
    """Morozov's discrepancy principle: stop once the centre fits to the noise.

    Fires as soon as

    .. math::

        2\\,\\Phi(\\bar v) \\;\\le\\; \\tau^2 N ,

    where :math:`\\bar v` is the **mean prediction** and :math:`\\Phi(\\bar
    v)` is the evaluation's ``centre_misfit``.

    Parameters
    ----------
    tau
        The tolerance, default ``1.0``; must be strictly positive.

    Raises
    ------
    ValueError
        If ``tau`` is not strictly positive.

    Notes
    -----
    The scaling is the natural one. At the true parameter the whitened
    residual is the whitened noise, so :math:`2\\Phi` is a :math:`\\chi^2_N`
    variate with mean :math:`N` and standard deviation :math:`\\sqrt{2N}`:
    :math:`\\tau = 1` stops when the centre's residual reaches the size the
    noise alone explains in expectation, and
    :math:`\\tau^2 = 1 + k\\sqrt{2/N}` puts the threshold :math:`k` standard
    deviations above that, values up to about :math:`\\tau = 2` being the
    common conservative choice. Fitting *below* the noise level is
    over-fitting, which is what the rule exists to prevent.

    The residual is measured at the mean **prediction**, not at the
    prediction of the mean parameter. These differ at second order in the
    ensemble spread, and only the former is available: the latter would cost
    an extra forward-model evaluation per step.

    Pairing this with a *budgeted* schedule is constructible and usually a
    mistake: a stopping rule fires on the misfits regardless of how much
    budget remains, so it can end a sampling run at an arbitrary intermediate
    level, whose ensemble is neither a posterior nor a fit. The result's
    ``stop_fired`` and ``budget_complete`` are what make that visible.
    """

    tau: float = static_field(default=1.0)

    def __post_init__(self) -> None:
        if not isinstance(self.tau, (int, float)) or isinstance(self.tau, bool):
            raise ValueError(
                f"DiscrepancyStop.tau: must be a real number, got {self.tau!r}"
            )
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError(
                f"DiscrepancyStop.tau: must be finite and strictly positive, got "
                f"{self.tau}"
            )

    def __call__(self, evaluation: Evaluation) -> bool:
        """Whether :math:`2\\Phi(\\bar v) \\le \\tau^2 N`."""
        threshold = self.tau**2 * evaluation.v_dim
        return bool(2.0 * evaluation.centre_misfit <= threshold)

    def __repr__(self) -> str:
        """As ``DiscrepancyStop(tau=1.0)``."""
        return _policy_repr(self)


# ---------------------------------------------------------------------------
# inflation
# ---------------------------------------------------------------------------


@_pytree_dataclass
class MultiplicativeInflation:
    """Scale the ensemble's anomalies about their mean.

    :math:`u_j \\mapsto \\bar u + r(u_j - \\bar u)`, which multiplies the
    empirical covariance by :math:`r^2`. The mean is preserved, and so is the
    affine subspace the initial ensemble spans.

    Parameters
    ----------
    anomaly_factor
        The factor :math:`r` applied to the anomalies. A scalar, held as a
        0-d array so a traced value can flow through it. Must be positive.

    Raises
    ------
    ValueError
        If ``anomaly_factor`` is not a scalar. In debug mode, also if it is
        not positive or not finite.

    Notes
    -----
    The field is named ``anomaly_factor`` rather than ``factor`` because the
    literature is split between the anomaly convention and the covariance
    convention :math:`C \\mapsto \\gamma C`, uses the same symbols for both,
    and the two are related by :math:`\\gamma = r^2`. Naming the field for
    what it multiplies is the only way to keep a caller from passing an
    intended variance inflation of 1.2 and silently getting 1.44 — an error
    that is invisible at the small values normally used and severe at large
    ones.
    """

    anomaly_factor: Array

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "anomaly_factor", jnp.asarray(self.anomaly_factor)
        )
        _check_field_rank(
            "MultiplicativeInflation", "anomaly_factor", self.anomaly_factor, 0
        )
        value_check(
            self.anomaly_factor,
            lambda r: bool(jnp.all(jnp.isfinite(r)) and jnp.all(r > 0)),
            "MultiplicativeInflation.anomaly_factor: must be finite and positive.",
        )

    def __call__(self, key, *, ensemble, step, beta, **_) -> Array:
        """Return the inflated ensemble; ``key``, ``step`` and ``beta`` unused."""
        _check_not_vmap_family(self, "__call__")
        return _multiplicative_inflate(ensemble, self.anomaly_factor)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return _broadcast_batch(
            "MultiplicativeInflation", tuple(self.anomaly_factor.shape)
        )

    def __repr__(self) -> str:
        """As ``MultiplicativeInflation(anomaly_factor=1.02)``; never raises."""
        try:
            return f"MultiplicativeInflation(anomaly_factor={self.anomaly_factor})"
        except Exception:
            return "<MultiplicativeInflation (unprintable leaves)>"


@_pytree_dataclass
class AdditiveInflation:
    """Add centred draws from a covariance to the ensemble.

    With :math:`P` the parameter dimension and :math:`J` the ensemble size,

    .. code-block:: python

        pert = Gaussian(jnp.zeros(P), cov).sample(key, J)
        ensemble + (pert - pert.mean(axis=0))

    The perturbations are centred, so the ensemble mean is preserved exactly
    and the empirical covariance is inflated by ``cov`` in expectation under
    the :math:`J-1` divisor. It is the only shipped mechanism that moves the
    ensemble out of the affine subspace its initial members span.

    Parameters
    ----------
    cov
        A :class:`~pyeki.linalg.PSDLinOp` of side :math:`P` supporting
        ``factor``. A scale is folded into the operator by the caller —
        ``AdditiveInflation(0.01 * prior.cov)`` — rather than carried as a
        second field.

    Raises
    ------
    TypeError
        If ``cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
    ValueError
        If ``cov`` is a vmapped family.
    UnsupportedOpError
        If ``cov`` does not support ``factor``, raised when the inflation is
        applied.

    Notes
    -----
    Nothing is precomputed here, and nothing needs to be. Every call reaches
    ``cov.factor()`` through :meth:`~pyeki.gauss.Gaussian.sample`, and the
    operator layer factorizes at construction, so ``factor()`` returns a
    stored factor rather than computing one — for every covariance this layer
    can be given.
    """

    cov: PSDLinOp

    def __post_init__(self) -> None:
        if not isinstance(self.cov, PSDLinOp):
            raise TypeError(
                f"AdditiveInflation.cov: must be a pyeki.linalg.PSDLinOp, got "
                f"{type(self.cov).__name__}"
            )
        if self.cov.batch_shape != ():
            raise ValueError(
                f"AdditiveInflation.cov: {self.cov!r} is a vmapped family; build "
                f"a family of inflations with jax.vmap over the constructor."
            )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return self.cov.batch_shape

    def __call__(self, key, *, ensemble, step, beta, **_) -> Array:
        """Return the inflated ensemble; ``step`` and ``beta`` unused."""
        _check_not_vmap_family(self, "__call__")
        ensemble = jnp.asarray(ensemble)
        n_members, u_dim = ensemble.shape[-2], ensemble.shape[-1]
        if self.cov.shape[0] != u_dim:
            raise ValueError(
                f"{self!r}: cov has side {self.cov.shape[0]}, but the ensemble has "
                f"parameter dimension {u_dim}"
            )
        return _additive_inflate(self.cov, key, ensemble, n_members, u_dim)

    def __repr__(self) -> str:
        """As ``AdditiveInflation(cov=DensePSD(12, 12))``; never raises."""
        try:
            return f"AdditiveInflation(cov={self.cov!r})"
        except Exception:
            return "<AdditiveInflation (unprintable leaves)>"


@jax.jit
def _multiplicative_inflate(ensemble: Array, anomaly_factor: Array) -> Array:
    return jnp.mean(ensemble, axis=-2) + anomaly_factor * _anomalies(ensemble)


@partial(jax.jit, static_argnums=(3, 4))
def _additive_inflate(cov, key, ensemble, n_members: int, u_dim: int) -> Array:
    pert = Gaussian(jnp.zeros(u_dim), cov).sample(key, n_members)
    return ensemble + (pert - pert.mean(axis=0))


# ---------------------------------------------------------------------------
# private: the clamp, the two criteria, and the shared field checks
# ---------------------------------------------------------------------------


def _bracket_top(schedule, beta: Array) -> Array:
    """The largest increment the ceiling and the budget allow."""
    ceiling = jnp.asarray(float(schedule.max_increment))
    if schedule.beta_target is None:
        return ceiling
    return jnp.minimum(ceiling, schedule.beta_target - beta)


def _clamp(schedule, unclamped: Array, beta: Array) -> Array:
    """Floor, then ceiling, then budget, in that order.

    The floor beats the criterion, so a step is always taken even where the
    criterion would demand an arbitrarily small one. The **budget cap beats
    the floor**, so the ladder cannot overshoot ``beta_target``. Both
    inversions are silent bugs: the first lets an adaptive ladder stall short
    of its budget for an unbounded number of steps, the second lets a
    sampling run condition on more data than it has.

    The budget term is present only when ``beta_target is not None``; writing
    the three-term form unconditionally is a ``TypeError`` on every unbounded
    run, which is a shipped configuration.
    """
    delta = jnp.maximum(unclamped, schedule.min_increment)
    delta = jnp.minimum(delta, schedule.max_increment)
    if schedule.beta_target is not None:
        delta = jnp.minimum(delta, schedule.beta_target - beta)
    return delta


@partial(jax.jit, static_argnums=(3,))
def _bisect_ess(misfits: Array, delta_hi: Array, target, n_bisect: int) -> Array:
    """Bisect ``[0, delta_hi]`` for the largest increment meeting the target.

    The cap-binds case is folded in **branchlessly**, by initialising ``lo``
    to ``delta_hi`` when the top of the bracket already meets the target and
    letting the loop run unchanged. That is not an optimization: without it a
    degenerate ensemble reaches the largest allowed step only to
    :math:`2^{-\\texttt{n\\_bisect}}`, and a budgeted ladder never consumes
    its budget exactly.

    A ``nan`` effective sample size propagates rather than being absorbed.
    Every comparison against ``nan`` is ``False``, so the bracket would
    otherwise never leave zero and the floor would turn a poisoned ensemble
    into an ordinary-looking smallest step. Sending ``nan`` on instead makes
    it reach the driver's increment validation, which raises — the same
    failure mode :func:`_guarded_ratio` gives the other adaptive schedule.
    """
    ess_hi = _ess_from_misfits(misfits, delta_hi)
    lo = jnp.where(ess_hi >= target, delta_hi, jnp.zeros_like(delta_hi))

    def body(_, bracket):
        low, high = bracket
        mid = 0.5 * (low + high)
        ok = _ess_from_misfits(misfits, mid) >= target
        return jnp.where(ok, mid, low), jnp.where(ok, high, mid)

    low, _ = lax.fori_loop(0, n_bisect, body, (lo, delta_hi))
    return jnp.where(jnp.isnan(ess_hi), jnp.nan, low)


@jax.jit
def _misfit_criterion(misfits: Array, theta) -> Array:
    """The larger of the mean and variance bounds, both guarded."""
    mean = jnp.mean(misfits)
    variance = jnp.var(misfits, ddof=1)
    return jnp.maximum(
        _guarded_ratio(theta, mean), jnp.sqrt(_guarded_ratio(theta, variance))
    )


def _guarded_ratio(theta, denominator: Array) -> Array:
    """``theta / denominator``: ``inf`` at zero, ``nan`` at ``nan`` or below zero.

    The inner ``where`` is what keeps ``theta / 0`` from being formed at all;
    the single-``where`` form still forms it and yields a ``nan`` derivative.
    The outer fallback distinguishes a vanishing denominator, where
    :math:`+\\infty` is the right answer and the clamps then deliver the
    largest allowed step, from a ``nan`` one, which must **not** silently
    select that step and instead propagates to the driver's increment
    validation.
    """
    positive = denominator > 0
    safe = jnp.where(positive, denominator, 1.0)
    return jnp.where(
        positive,
        theta / safe,
        jnp.where(denominator == 0, jnp.inf, jnp.nan),
    )


def _check_adaptive_fields(schedule) -> None:
    """Validate the three fields both adaptive schedules share."""
    name = type(schedule).__name__
    if schedule.beta_target is not None:
        if not isinstance(schedule.beta_target, (int, float)) or isinstance(
            schedule.beta_target, bool
        ):
            raise ValueError(
                f"{name}.beta_target: must be None or a real number, got "
                f"{schedule.beta_target!r}"
            )
        if not math.isfinite(schedule.beta_target) or schedule.beta_target <= 0.0:
            raise ValueError(
                f"{name}.beta_target: must be finite and strictly positive when "
                f"given, got {schedule.beta_target}"
            )
    for field_name in ("min_increment", "max_increment"):
        value = getattr(schedule, field_name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{name}.{field_name}: must be a real number, got {value!r}"
            )
        object.__setattr__(schedule, field_name, float(value))
    if schedule.beta_target is not None:
        object.__setattr__(schedule, "beta_target", float(schedule.beta_target))
    if not math.isfinite(schedule.max_increment):
        raise ValueError(
            f"{name}.max_increment: must be finite, got {schedule.max_increment}. "
            f"An unbounded ladder with an infinite ceiling has no upper clamp at "
            f"all, and a degenerate ensemble then reaches the driver's increment "
            f"validation instead."
        )
    if schedule.min_increment <= 0.0:
        raise ValueError(
            f"{name}.min_increment: must be strictly positive, got "
            f"{schedule.min_increment}"
        )
    if schedule.min_increment > schedule.max_increment:
        raise ValueError(
            f"{name}: min_increment {schedule.min_increment} exceeds "
            f"max_increment {schedule.max_increment}"
        )


def _positive_int(where: str, name: str, value) -> int:
    """Require a positive Python ``int`` that determines a static length."""
    if type(value) is not int:
        raise TypeError(
            f"{where}: {name} must be a Python int, got {type(value).__name__}. "
            f"It determines a static ladder length, so it can never be traced."
        )
    if value < 1:
        raise ValueError(f"{where}: {name} must be at least 1, got {value}")
    return value


def _policy_repr(policy) -> str:
    """Type name and static fields, in declaration order; never raises."""
    try:
        shown = ", ".join(
            f"{f.name}={getattr(policy, f.name)!r}"
            for f in dataclasses.fields(policy)
        )
        return f"{type(policy).__name__}({shown})"
    except Exception:
        return f"<{type(policy).__name__} (unprintable)>"
