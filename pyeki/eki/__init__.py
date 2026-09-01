"""Ensemble Kalman Inversion: the ladder, the policies that shape it, and the run.

The algorithmic top of the package. It turns the Gaussian conditioning of
:mod:`pyeki.gauss` into a *run* — an initial ensemble, a ladder of tempered
targets, an ensemble update per step, and a record of what happened.

For a prior :math:`\\pi_0` and the misfit
:math:`\\Phi(v) = \\tfrac12\\lVert W(y - v)\\rVert^2`, the family of targets is

.. math::

    \\pi_\\beta(u) \\;\\propto\\; \\pi_0(u)\\, e^{-\\beta\\Phi(G(u))},
    \\qquad \\beta \\ge 0 ,

the prior at :math:`\\beta = 0`, the Bayesian posterior at :math:`\\beta = 1`,
and increasingly concentrated on the minimizers of :math:`\\Phi \\circ G` as
:math:`\\beta \\to \\infty`. Moving one increment up that ladder is an
identity, not an approximation: it is conditioning on the same observation
with the noise covariance divided by that increment. That is why this layer
needs nothing from :mod:`pyeki.gauss` beyond its two update methods.

Two well-known modes of EKI fall out of the same driver, and neither is
privileged — they are two schedules, not two drivers and not a flag::

    from pyeki.eki import (
        AdaptiveESSSchedule, DiscrepancyStop, EKIState, FixedSchedule, run,
    )

    state = EKIState.from_prior(key, prior, n_members=64)

    # Sampling form: adaptive ladder, budget 1, deterministic update.
    sampled = run(state, forward, y, noise_cov, schedule=AdaptiveESSSchedule())

    # Optimization form: unit steps, no budget, stop on the discrepancy principle.
    fit = run(state, forward, y, noise_cov,
              schedule=FixedSchedule.constant(1.0, n_steps=200),
              stop=DiscrepancyStop(tau=1.0))

===================================== ======================================
object                                is
===================================== ======================================
:class:`EKIState`                     the loop-carried state
:class:`Evaluation`                   what one forward evaluation produced
:class:`HistoryRecord`                one row of the run's history
:class:`EKIResult`                    the final state, the history, and why
                                      the run ended
:class:`EnsembleUpdate`,              the three axes, as protocols
:class:`Schedule`,
:class:`StoppingRule`,
:class:`Inflation`
:func:`run`, :func:`iterate`          the driver, as a function and as a
                                      generator
:func:`evaluate`, :func:`assimilate`, one step, as its two phases and
:func:`advance`                       their composition
:func:`misfits`,                      the array-level pieces schedules and
:func:`effective_sample_size`,        custom drivers need
:func:`repair_failed_members`
:class:`EKIError`                     raised when a run cannot continue,
                                      carrying the state and the history
===================================== ======================================

The shipped policies are :class:`TransformUpdate` and :class:`PathwiseUpdate`;
:class:`FixedSchedule`, :class:`AdaptiveESSSchedule` and
:class:`AdaptiveMisfitSchedule`; :class:`DiscrepancyStop`; and
:class:`MultiplicativeInflation` and :class:`AdditiveInflation`. This is the
one place where pyEKI is deliberately open to extension at the algorithm
level: :mod:`pyeki.linalg` is extended by writing an operator,
:mod:`pyeki.gauss` is closed, and :mod:`pyeki.eki` is extended by writing a
schedule, an update rule, or an inflation. :mod:`pyeki.eki.testing` holds the
conformance checks for one.

Conventions shared by everything in the layer:

- **Ensembles are stored row-wise**, a ``(J, dim)`` array, and vectors passed
  to the layer are exactly core-shaped, as in :mod:`pyeki.gauss`. The batched
  exception is :func:`misfits`.
- **The tempering variable is a level, and steps take increments.** A state
  carries :math:`\\beta`; a step takes :math:`\\Delta\\beta` and conditions
  with per-step noise :math:`R/\\Delta\\beta` — never :math:`R/\\beta`.
- **The misfit carries the factor** :math:`\\tfrac12` and is measured against
  the base noise covariance, never a tempered one.
- **PRNG keys are typed keys**, the output of :func:`jax.random.key`.

Notes
-----
The behaviour of this layer is specified by the "Ensemble Kalman Inversion
contract" page of the documentation, which is normative; the user guide's
"Running an inversion" page explains when to reach for each piece.

**What the layer promises.** Exactness is claimed for the affine-Gaussian
case only: with an affine forward model, a Gaussian prior, an ensemble whose
empirical moments equal the prior's, :class:`TransformUpdate`, no inflation,
no failed members, and a ladder whose increments sum exactly to 1, a run
reproduces the exact posterior mean and covariance to floating point. Every
clause there is load-bearing. For a nonlinear forward model the output is an
approximation with no consistency guarantee. Three departures from the ladder
are opt-in and each is named where it happens: inflation, the stochastic
update, and repairing failed members. The user guide's "Running an inversion"
page works through what each one costs.

Every iterate of a run lies in the affine subspace spanned by the initial
ensemble, whose dimension is at most :math:`J - 1`, however many steps are run
and however the schedule is chosen. :math:`J` therefore bounds what a run can
*represent*, not merely how accurately it estimates moments, and
:class:`AdditiveInflation` is the only shipped mechanism that leaves that
subspace.
"""
from .driver import advance, assimilate, evaluate, iterate, run
from .helpers import effective_sample_size, misfits, repair_failed_members
from .policies import (
    AdaptiveESSSchedule,
    AdaptiveMisfitSchedule,
    AdditiveInflation,
    DiscrepancyStop,
    EnsembleUpdate,
    FixedSchedule,
    Inflation,
    MultiplicativeInflation,
    PathwiseUpdate,
    Schedule,
    StoppingRule,
    TransformUpdate,
)
from .values import (
    INTERRUPTED,
    SCHEDULE_EXHAUSTED,
    STOPPING_RULE,
    EKIError,
    EKIResult,
    EKIState,
    Evaluation,
    HistoryRecord,
)

__all__ = [
    # value classes
    "EKIState",
    "Evaluation",
    "HistoryRecord",
    "EKIResult",
    # protocols
    "EnsembleUpdate",
    "Schedule",
    "StoppingRule",
    "Inflation",
    # update rules
    "TransformUpdate",
    "PathwiseUpdate",
    # schedules
    "FixedSchedule",
    "AdaptiveESSSchedule",
    "AdaptiveMisfitSchedule",
    # stopping rules
    "DiscrepancyStop",
    # inflation
    "MultiplicativeInflation",
    "AdditiveInflation",
    # the driver, and one step
    "run",
    "iterate",
    "evaluate",
    "assimilate",
    "advance",
    # helpers
    "misfits",
    "effective_sample_size",
    "repair_failed_members",
    # status constants, and the exception
    "SCHEDULE_EXHAUSTED",
    "STOPPING_RULE",
    "INTERRUPTED",
    "EKIError",
]
