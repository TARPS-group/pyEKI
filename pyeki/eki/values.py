"""The value classes of a run, and the error a run raises.

============================ ==================================================
object                       holds
============================ ==================================================
:class:`EKIState`            the loop-carried state: ensemble, level, step
                             index, key
:class:`Evaluation`          everything one forward evaluation produced: the
                             members, their predictions, and the whitened
                             residuals
:class:`HistoryRecord`       one row of the run's history, scalars only
:class:`EKIResult`           the final state, the history, and why the run
                             ended
:class:`EKIError`            raised when a run cannot continue, carrying the
                             state and the history so the work is not lost
============================ ==================================================

Conventions shared by everything in the module:

- **The first three are unbatched frozen pytrees**, exactly like operators
  and the classes of :mod:`pyeki.gauss`: they compare by identity, they are
  never valid ``static_argnums``, and a pytree reconstruction with batched
  leaves produces a *vmapped family*, which reports its ``batch_shape`` and
  refuses every method and array-computing property. :class:`EKIResult` is
  the exception — it is a report, never an argument to traced code, so it is
  a plain frozen dataclass.
- **Ensembles are stored row-wise**, a ``(J, dim)`` array, one member per
  row, as in :mod:`pyeki.gauss`.
- **A run's status is one of three strings**, exported as the constants
  :data:`SCHEDULE_EXHAUSTED`, :data:`STOPPING_RULE` and :data:`INTERRUPTED`
  so that a comparison cannot be misspelled.

Notes
-----
The behaviour of this module is specified by the "Ensemble Kalman Inversion
contract" page of the documentation, which is normative.

Every field of :class:`HistoryRecord` is a 0-d array, ``step`` and
``n_valid`` included. Static fields live in a pytree's treedef, so two
records with different ``step`` values would have different treedefs and
:func:`jax.tree.map` across a history would raise instead of stacking. The
history is the one collection in the package meant to be stacked, so its
element type must be homogeneous as a pytree. :attr:`Evaluation.step` stays a
static ``int`` for the opposite reason: :class:`~pyeki.eki.FixedSchedule`
indexes a Python tuple with it, which a traced value cannot do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array

from ..linalg import static_field, value_check
from ..linalg.base import _broadcast_batch, _pytree_dataclass
from .helpers import (
    _check_field_rank,
    _check_finite,
    _check_not_vmap_family,
    _misfits_from_residuals,
)

__all__ = [
    "EKIError",
    "EKIResult",
    "EKIState",
    "Evaluation",
    "HistoryRecord",
    "INTERRUPTED",
    "SCHEDULE_EXHAUSTED",
    "STOPPING_RULE",
]

#: The ladder finished, by the schedule's attributes or by ``next_increment``
#: returning ``None``.
SCHEDULE_EXHAUSTED = "schedule_exhausted"

#: The stopping rule fired; the last record is terminal.
STOPPING_RULE = "stopping_rule"

#: The run was ended by its caller, not by a policy. Never produced by
#: :func:`~pyeki.eki.run`.
INTERRUPTED = "interrupted"

Status = Literal["schedule_exhausted", "stopping_rule", "interrupted"]

#: How a run treats members whose predictions came back non-finite.
OnFailure = Literal["repair", "raise"]

_STATUSES = (SCHEDULE_EXHAUSTED, STOPPING_RULE, INTERRUPTED)


class EKIError(RuntimeError):
    """A run cannot continue.

    Raised on four conditions: ``max_steps`` exceeded, fewer than two valid
    members, any invalid member under ``on_failure="raise"``, and a
    non-finite updated ensemble. A run is long and expensive enough that a
    caller wants to catch its failures specifically, to checkpoint and
    investigate, without catching every :class:`RuntimeError` in the process.

    Attributes
    ----------
    state : EKIState
        The last good state, populated on every raise path.
    history : tuple of HistoryRecord
        The records accumulated up to the failure.

    Notes
    -----
    The two attributes are what make a caught error recoverable::

        try:
            result = run(state, forward, y, noise_cov, schedule=sched)
        except EKIError as exc:
            checkpoint(exc.state)          # resume from here
            diagnose(exc.history)

    Resuming from ``exc.state`` continues the run exactly, so nothing beyond
    the two attributes is needed to make the recovery exact.
    """

    def __init__(self, message: str, *, state=None, history=()) -> None:
        super().__init__(message)
        self.state = state
        self.history = tuple(history)


# ---------------------------------------------------------------------------
# the loop-carried state
# ---------------------------------------------------------------------------


@_pytree_dataclass
class EKIState:
    """Everything a run needs in order to continue.

    A state is the unit of checkpointing and of resumption: ``run`` on a
    state returned by a previous run continues it, and the tail of the run is
    bit-identical to an uninterrupted one.

    Parameters
    ----------
    ensemble
        The members, a ``(J, P)`` array with :math:`J \\ge 2`, one member per
        row.
    beta
        The accumulated tempering level, a scalar. A Python float or 0-d
        array is accepted and stored as a 0-d float array. Must not be
        negative.
    step
        The cumulative step index, a non-negative Python ``int``. It counts
        across resumptions, and position-dependent schedules read it.
    key
        A typed JAX PRNG key of shape ``()`` — the output of
        :func:`jax.random.key`. The only source of randomness in a run.

    Raises
    ------
    ValueError
        If ``ensemble`` is not rank 2, if :math:`J < 2`, if ``beta`` is not a
        scalar, or if ``step`` is negative. In debug mode, also if
        ``ensemble`` is not finite or ``beta`` is negative or not finite.
    TypeError
        If ``step`` is not a Python ``int``, or if ``key`` is not a typed
        PRNG key.

    Notes
    -----
    ``key`` is checked on **shape and dtype**, since a typed key and a 0-d
    float array agree on shape and only the ``prng_key`` dtype separates
    them. A raw ``uint32`` key array of shape ``(2,)`` is rejected because it
    would make the state's ``batch_shape`` ambiguous.

    **``step`` is cumulative across runs.** Resuming a partially-completed
    ladder is the case that is designed for: a ten-step
    :class:`~pyeki.eki.FixedSchedule` interrupted after four steps resumes
    at step four. The same property makes *chaining* a second, different
    ladder onto a finished state a silent no-op — the fresh schedule finds
    ``step >= n_steps`` already true and the run returns immediately with an
    empty history and the ensemble unchanged. A new ladder needs a new counter,
    which is what :meth:`restart` is for.
    """

    ensemble: Array
    beta: Array
    step: int = static_field()
    key: Array

    def __post_init__(self) -> None:
        _check_field_rank("EKIState", "ensemble", self.ensemble, 2)
        if self.ensemble.shape[0] < 2:
            raise ValueError(
                f"EKIState.ensemble: at least 2 members are required, got "
                f"{self.ensemble.shape[0]}. A single member has no anomalies."
            )
        object.__setattr__(self, "beta", _as_level("EKIState", "beta", self.beta))
        if type(self.step) is not int:
            raise TypeError(
                f"EKIState.step: must be a Python int, got "
                f"{type(self.step).__name__}. It is static metadata that a "
                f"schedule indexes a tuple with, so it can never be traced."
            )
        if self.step < 0:
            raise ValueError(f"EKIState.step: must not be negative, got {self.step}")
        _check_typed_key("EKIState", self.key)
        _check_finite("EKIState", "ensemble", self.ensemble)

    @classmethod
    def from_prior(cls, key, prior, n_members: int) -> EKIState:
        """Draw the initial ensemble from a prior: the usual way to start.

        The draw is pinned as

        .. code-block:: python

            key_sample, key_state = jax.random.split(key)
            EKIState(prior.sample(key_sample, n_members), 0.0, 0, key_state)

        so the initial ensemble is exactly :meth:`~pyeki.gauss.Gaussian.sample`'s
        pinned draw, and the state's own stream is independent of it.

        Parameters
        ----------
        key
            A typed JAX PRNG key, consumed whole and split once.
        prior
            A :class:`~pyeki.gauss.Gaussian` whose covariance supports
            ``factor``.
        n_members
            The ensemble size :math:`J`, a Python ``int`` at least 2.

        Returns
        -------
        EKIState
            At ``beta = 0.0`` and ``step = 0``.

        Raises
        ------
        UnsupportedOpError
            If ``prior.cov`` does not support ``factor``. Propagated
            unmodified from the operator layer.

        Notes
        -----
        Nothing in the layer requires that the initial ensemble came from the
        prior. A warm start — an ensemble from a previous run, a Latin
        hypercube, a hand-built design — is direct construction with
        ``beta=0.0`` and ``step=0``, and the tempered family's :math:`\\pi_0`
        is whatever that ensemble represents.
        """
        key_sample, key_state = jax.random.split(key)
        return cls(prior.sample(key_sample, n_members), 0.0, 0, key_state)

    def restart(self) -> EKIState:
        """A copy at ``step = 0`` and ``beta = 0.0``, ready for a new ladder.

        Same ensemble and same key; only the two counters are reset. Use it
        to chain a second ladder onto a finished run, which otherwise returns
        immediately with nothing raised — a step-bounded schedule exhausts on
        ``step`` and a budgeted one on ``beta``, which is why this resets
        both.

        Raises
        ------
        ValueError
            If this is a vmapped family.
        """
        _check_not_vmap_family(self, "restart")
        return EKIState(self.ensemble, 0.0, 0, self.key)

    @property
    def n_members(self) -> int:
        """The ensemble size :math:`J`."""
        return int(self.ensemble.shape[-2])

    @property
    def u_dim(self) -> int:
        """The parameter dimension :math:`P`."""
        return int(self.ensemble.shape[-1])

    @property
    def mean(self) -> Array:
        """The ensemble mean, a ``(P,)`` array."""
        _check_not_vmap_family(self, "mean")
        return jnp.mean(self.ensemble, axis=-2)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return _broadcast_batch(
            "EKIState",
            tuple(self.ensemble.shape[:-2]),
            tuple(self.beta.shape),
            tuple(self.key.shape),
        )

    def __repr__(self) -> str:
        """As ``EKIState(n_members=64, u_dim=12, step=3)``; never raises."""
        try:
            base = (
                f"EKIState(n_members={self.n_members}, u_dim={self.u_dim}, "
                f"step={self.step})"
            )
            batch = self.batch_shape
        except Exception:
            return "<EKIState (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base


# ---------------------------------------------------------------------------
# what one forward evaluation produced
# ---------------------------------------------------------------------------


@_pytree_dataclass
class Evaluation:
    """Everything one forward evaluation produced.

    What a :class:`~pyeki.eki.Schedule`'s ``next_increment`` and a
    :class:`~pyeki.eki.StoppingRule` see, what :func:`~pyeki.eki.assimilate`
    consumes, and what a run reports as its ``last_evaluation``. Returned by
    :func:`~pyeki.eki.evaluate`.

    Parameters
    ----------
    step
        The index of this step, a non-negative Python ``int``.
    beta
        The level *entering* the step, a 0-d array.
    ensemble
        The ``(J, P)`` members that were evaluated — **after** inflation and
        **after** repair, so not in general the members the caller supplied.
    predictions
        Their ``(J, N)`` predictions, after repair.
    whitened_residuals
        A ``(J, N)`` array whose row :math:`j` is :math:`W(y - v_j)`, after
        repair.
    rms_parameter_spread
        A 0-d array, :math:`\\lVert A_u\\rVert_F / \\sqrt{(J-1)P}`: the
        root-mean-square per-coordinate ensemble standard deviation.
    n_valid
        How many members' predictions were finite, a 0-d integer array of at
        least 2. Data rather than static metadata: nothing indexes a Python
        object with it, and a static field would give two evaluations with
        different counts different treedefs, retracing any ``jit``-ed policy
        once per step.

    Raises
    ------
    ValueError
        If any field has the wrong rank, if the arrays disagree on :math:`J`
        or :math:`N`, or if :math:`J < 2`. In debug mode, also if ``n_valid``
        is outside :math:`[2, J]`.
    TypeError
        If ``step`` is not a Python ``int``, or ``n_valid`` not an integer.

    Notes
    -----
    **The whitened residuals are carried rather than just the misfit
    vector.** Writing :math:`b_j = W(y - v_j)` for the rows,

    .. math::

        \\Phi_j = \\tfrac12\\lVert b_j\\rVert^2, \\qquad
        \\bar b = W(y - \\bar v), \\qquad
        b_j - \\bar b = -W(v_j - \\bar v) ,

    so the misfits, the misfit of the mean prediction, and the whitened
    prediction anomalies are all recoverable from this one array — the last
    being, up to a sign and a :math:`\\sqrt{J-1}`, the whitened factor the
    conditioning kernel of :mod:`pyeki.gauss` is built on.
    It costs nothing: the driver must whiten the residuals to compute the
    misfits at all. :math:`N` is likewise recoverable from the trailing axis,
    so a criterion may be calibrated to the observation dimension without
    storing it.

    The recovered anomaly matrix is a diagnostic and never a substitute for
    the update's own: :mod:`pyeki.gauss` centres before it whitens — the
    factor it whitens was centred when it was built — precisely because
    centring already-whitened predictions cancels a common
    :math:`W\\bar v` and loses accuracy as the ensemble collapses.

    ``rms_parameter_spread`` is scale-dependent, and the name says so: it
    averages per-coordinate standard deviations across parameters that may
    carry unrelated units, so for an unscaled parameter vector it is
    dominated by the largest-magnitude coordinate.
    """

    step: int = static_field()
    beta: Array
    ensemble: Array
    predictions: Array
    whitened_residuals: Array
    rms_parameter_spread: Array
    n_valid: Array

    def __post_init__(self) -> None:
        if type(self.step) is not int:
            raise TypeError(
                f"Evaluation.step: must be a Python int, got "
                f"{type(self.step).__name__}. A schedule indexes a tuple with "
                f"it, so it can never be traced."
            )
        if self.step < 0:
            raise ValueError(f"Evaluation.step: must not be negative, got {self.step}")
        object.__setattr__(self, "beta", _as_level("Evaluation", "beta", self.beta))
        _check_field_rank("Evaluation", "ensemble", self.ensemble, 2)
        _check_field_rank("Evaluation", "predictions", self.predictions, 2)
        _check_field_rank(
            "Evaluation", "whitened_residuals", self.whitened_residuals, 2
        )
        _check_field_rank(
            "Evaluation", "rms_parameter_spread", self.rms_parameter_spread, 0
        )
        n_members = self.ensemble.shape[0]
        if n_members < 2:
            raise ValueError(
                f"Evaluation.ensemble: at least 2 members are required, got "
                f"{n_members}. A single member has no anomalies."
            )
        if self.predictions.shape[0] != n_members:
            raise ValueError(
                f"Evaluation: ensemble and predictions must have the same number "
                f"of members, got shapes {self.ensemble.shape} and "
                f"{self.predictions.shape}"
            )
        if self.whitened_residuals.shape != self.predictions.shape:
            raise ValueError(
                f"Evaluation: whitened_residuals must have the shape of "
                f"predictions, got {self.whitened_residuals.shape} and "
                f"{self.predictions.shape}"
            )
        object.__setattr__(self, "n_valid", jnp.asarray(self.n_valid))
        _check_field_rank("Evaluation", "n_valid", self.n_valid, 0)
        if not jnp.issubdtype(self.n_valid.dtype, jnp.integer):
            raise TypeError(
                f"Evaluation.n_valid: must be an integer, got dtype "
                f"{self.n_valid.dtype}"
            )
        value_check(
            self.n_valid,
            lambda n: bool(2 <= n <= n_members),
            f"Evaluation.n_valid: must lie in [2, {n_members}].",
        )

    @property
    def misfits(self) -> Array:
        """The per-member misfits, a ``(J,)`` array.

        :math:`\\Phi_j = \\tfrac12\\lVert b_j\\rVert^2`, the same quantity
        :func:`~pyeki.eki.misfits` computes from ``y`` and the predictions.
        Computed on access, not cached.
        """
        _check_not_vmap_family(self, "misfits")
        return _misfits_from_residuals(self.whitened_residuals)

    @property
    def centre_misfit(self) -> Array:
        """The misfit of the mean prediction, a 0-d array.

        :math:`\\Phi(\\bar v) = \\tfrac12\\lVert \\bar b\\rVert^2`. This is
        **not** the mean of :attr:`misfits`: the two differ by half the
        whitened prediction spread,

        .. math::

            \\overline{\\Phi_j} = \\Phi(\\bar v)
            + \\tfrac{J-1}{2J}\\operatorname{tr}
              \\bigl(W \\widehat{C}_{vv} W^\\top\\bigr) ,

        coinciding only as the ensemble collapses. A discrepancy principle
        asks about the centre; a tempering criterion asks about the
        individual members.
        """
        _check_not_vmap_family(self, "centre_misfit")
        return _misfits_from_residuals(jnp.mean(self.whitened_residuals, axis=-2))

    @property
    def n_members(self) -> int:
        """The ensemble size :math:`J`."""
        return int(self.ensemble.shape[-2])

    @property
    def u_dim(self) -> int:
        """The parameter dimension :math:`P`."""
        return int(self.ensemble.shape[-1])

    @property
    def v_dim(self) -> int:
        """The observation dimension :math:`N`."""
        return int(self.predictions.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return _broadcast_batch(
            "Evaluation",
            tuple(self.beta.shape),
            tuple(self.ensemble.shape[:-2]),
            tuple(self.predictions.shape[:-2]),
            tuple(self.whitened_residuals.shape[:-2]),
            tuple(self.rms_parameter_spread.shape),
            tuple(self.n_valid.shape),
        )

    def __repr__(self) -> str:
        """As ``Evaluation(step=3, n_members=64)``; never raises."""
        try:
            base = f"Evaluation(step={self.step}, n_members={self.n_members})"
            batch = self.batch_shape
        except Exception:
            return "<Evaluation (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base


# ---------------------------------------------------------------------------
# one row of the history
# ---------------------------------------------------------------------------


@_pytree_dataclass
class HistoryRecord:
    """One row of a run's history: eleven 0-d arrays.

    Built by the driver from an :class:`Evaluation` and the increment that
    was chosen, so the evaluation is the single source of truth for
    everything both describe.

    Parameters
    ----------
    step
        The index of the step, a 0-d integer array.
    n_valid
        How many members' predictions were finite, a 0-d integer array.
    beta
        The level entering the step.
    increment
        The increment taken, or exactly ``0.0`` on a terminal record.
    beta_next
        ``beta + increment``, stored rather than derived so that plotting a
        ladder needs no arithmetic.
    misfit_mean, misfit_min, misfit_max
        Summaries of the step's per-member misfits.
    centre_misfit
        The misfit of the mean prediction.
    spread
        The evaluation's ``rms_parameter_spread``.
    ess
        The effective sample size at the increment actually taken, computed
        for **every** schedule.

    Raises
    ------
    ValueError
        If any field is not a 0-d array.

    Notes
    -----
    The per-member misfit vector is deliberately absent, as is anything else
    of size :math:`J` or larger. A caller who wants per-member or
    per-observation quantities uses :func:`~pyeki.eki.iterate` and keeps them
    from ``evaluation.whitened_residuals``.
    """

    step: Array
    n_valid: Array
    beta: Array
    increment: Array
    beta_next: Array
    misfit_mean: Array
    misfit_min: Array
    misfit_max: Array
    centre_misfit: Array
    spread: Array
    ess: Array

    def __post_init__(self) -> None:
        for name in _RECORD_FIELDS:
            _check_field_rank("HistoryRecord", name, getattr(self, name), 0)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return _broadcast_batch(
            "HistoryRecord",
            *[tuple(getattr(self, name).shape) for name in _RECORD_FIELDS],
        )

    def __repr__(self) -> str:
        """As ``HistoryRecord(step=3)``; never raises."""
        try:
            batch = self.batch_shape
            base = (
                "HistoryRecord" if batch != () else f"HistoryRecord(step={self.step})"
            )
        except Exception:
            return "<HistoryRecord (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base


#: The declaration order of :class:`HistoryRecord`'s fields.
_RECORD_FIELDS = (
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
)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class EKIResult:
    """What a run produced, and why it ended.

    All four fields are keyword-only.

    Parameters
    ----------
    state
        The final :class:`EKIState`. Keyword-only.
    history
        A tuple of :class:`HistoryRecord`, one per ensemble evaluation.
        Keyword-only.
    status
        Why the run ended: :data:`SCHEDULE_EXHAUSTED`, :data:`STOPPING_RULE`
        or :data:`INTERRUPTED`. Keyword-only.
    last_evaluation
        The :class:`Evaluation` of the final forward evaluation, or ``None``
        if the run made none. Keyword-only.

    Raises
    ------
    ValueError
        If ``status`` is not one of the three permitted strings.
    TypeError
        If any field has the wrong type.

    Notes
    -----
    The fields are keyword-only because the type is user-constructible and
    ``state`` and ``history`` are as swappable as any other same-arity pair.

    **The returned ensemble has never been evaluated**, which is why
    ``last_evaluation`` exists. On a :data:`SCHEDULE_EXHAUSTED` run the last
    update produces ``state.ensemble`` and the loop then ends, so
    ``result.last_evaluation.ensemble`` is *not* ``result.ensemble`` there:
    the last record's misfits describe the ensemble before that update. On a
    :data:`STOPPING_RULE` termination the state is left unchanged, so the two
    *are* the same array — the off-by-one is a property of the exit path, not
    an invariant.

    Moments beyond the mean are one line through the layer below::

        fit = Gaussian.from_samples(result.ensemble)
        fit.cov.diag()                       # (P,) per-coordinate variances
        fit.sample(key, 1000)                # draws from the fitted moments

    That line is not a further conditioning step, and its result is not a
    posterior whatever the run's configuration — it is the fit to the
    terminal ensemble.
    """

    state: EKIState = field(kw_only=True)
    history: tuple[HistoryRecord, ...] = field(kw_only=True)
    status: Status = field(kw_only=True)
    last_evaluation: Evaluation | None = field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.state, EKIState):
            raise TypeError(
                f"EKIResult.state: must be an EKIState, got "
                f"{type(self.state).__name__}"
            )
        if not isinstance(self.history, tuple) or not all(
            isinstance(record, HistoryRecord) for record in self.history
        ):
            raise TypeError(
                "EKIResult.history: must be a tuple of HistoryRecord"
            )
        if self.status not in _STATUSES:
            raise ValueError(
                f"EKIResult.status: must be one of {_STATUSES}, got "
                f"{self.status!r}. Compare against the exported constants "
                f"SCHEDULE_EXHAUSTED, STOPPING_RULE and INTERRUPTED rather than "
                f"against a literal."
            )
        if self.last_evaluation is not None and not isinstance(
            self.last_evaluation, Evaluation
        ):
            raise TypeError(
                f"EKIResult.last_evaluation: must be an Evaluation or None, got "
                f"{type(self.last_evaluation).__name__}"
            )

    @property
    def ensemble(self) -> Array:
        """The final ensemble, a ``(J, P)`` array."""
        return self.state.ensemble

    @property
    def beta(self) -> Array:
        """The level the run reached, a 0-d array."""
        return self.state.beta

    @property
    def mean(self) -> Array:
        """The final ensemble's mean, a ``(P,)`` array."""
        return self.state.mean

    @property
    def n_evaluations(self) -> int:
        """How many times this run called the forward model: one per record.

        The run's cost in *member* evaluations is
        :math:`J\\,n_{\\text{evaluations}}`, which is the caller's own
        multiplication — this counts calls, and each one is handed the whole
        ensemble.
        """
        return len(self.history)

    @property
    def n_completed_steps(self) -> int:
        """How many steps this run completed: how many times it moved the
        ensemble.

        Equal to :attr:`n_evaluations`, or one less when the run ended on a
        terminal evaluation — a stopping rule that fired, or a schedule whose
        ``next_increment`` returned ``None``. Both decisions need an
        evaluation to reach, so the evaluation is spent and no update
        follows; a terminal record is the one with an ``increment`` of
        exactly zero, and there is at most one, always last.

        Not derivable from :attr:`status`, which does not distinguish a
        schedule exhausted declaratively from one exhausted by returning
        ``None``, nor from ``state.step``, which is cumulative across a
        chain of runs.
        """
        if not self.history:
            return 0
        ended_on_evaluation = float(self.history[-1].increment) == 0.0
        return len(self.history) - int(ended_on_evaluation)

    @property
    def min_n_valid(self) -> int | None:
        """The smallest ``n_valid`` over the history, or ``None`` if empty.

        Reported so that a run with recurring failures is visible without the
        caller having to stack the history and look.
        """
        if not self.history:
            return None
        return min(int(record.n_valid) for record in self.history)

    @property
    def stacked(self) -> HistoryRecord:
        """The history as one record whose fields are ``(T,)``-shaped.

        A family in the sense of the layer's batch machinery: inert to
        methods, legible in its ``repr``, and exactly what a stacked history
        should be::

            plt.plot(result.stacked.step, result.stacked.misfit_mean)

        A property rather than a documented one-liner because the one-liner,
        ``jax.tree.map(lambda *xs: jnp.stack(xs), *result.history)``, raises
        on an empty history. This returns ``(0,)``-shaped fields there
        instead, which is the answer that needs no branch.
        """
        if not self.history:
            return jax.tree.map(
                lambda x: jnp.zeros((0, *x.shape), x.dtype), _zero_record()
            )
        return jax.tree.map(lambda *xs: jnp.stack(xs), *self.history)

    @property
    def stop_fired(self) -> bool:
        """Whether the run ended because its stopping rule fired.

        ``status == STOPPING_RULE``. With
        :class:`~pyeki.eki.DiscrepancyStop` this is the optimization form's
        answer to *did it fit?*; with any other rule it reports only that
        that rule fired.
        """
        return self.status == STOPPING_RULE

    @property
    def budget_complete(self) -> bool:
        """Whether the run ended because its ladder finished.

        The sampling form's one-word answer to *did the ladder finish?* The
        pair with :attr:`stop_fired` makes the layer's one real trap legible
        from the result alone: a stopping rule on a budgeted ladder reports
        ``stop_fired=True, budget_complete=False``, having ended a sampling
        run at an arbitrary intermediate level.
        """
        return self.status == SCHEDULE_EXHAUSTED

    def __repr__(self) -> str:
        """As ``EKIResult(status='schedule_exhausted', n_evaluations=17, beta=1)``."""
        try:
            return (
                f"EKIResult(status={self.status!r}, "
                f"n_evaluations={self.n_evaluations}, "
                f"beta={float(self.state.beta):g})"
            )
        except Exception:
            return "<EKIResult (unprintable)>"


# ---------------------------------------------------------------------------
# private
# ---------------------------------------------------------------------------


def _zero_record() -> HistoryRecord:
    """A valid record of zeros, the prototype an empty ``stacked`` maps over."""
    zero_int = jnp.zeros((), dtype=jnp.result_type(int))
    zero = jnp.zeros((), dtype=jnp.result_type(float))
    return HistoryRecord(
        step=zero_int,
        n_valid=zero_int,
        beta=zero,
        increment=zero,
        beta_next=zero,
        misfit_mean=zero,
        misfit_min=zero,
        misfit_max=zero,
        centre_misfit=zero,
        spread=zero,
        ess=zero,
    )


def _as_level(cls_name: str, field_name: str, value) -> Array:
    """Validate a tempering level and return it as a 0-d float array."""
    if isinstance(value, bool):
        raise TypeError(
            f"{cls_name}.{field_name}: expected a scalar level, got the bool "
            f"{value!r}, which would silently become {float(value)}"
        )
    level = jnp.asarray(value)
    if level.ndim != 0:
        raise ValueError(
            f"{cls_name}.{field_name}: expected a scalar, got shape {level.shape}"
        )
    if not jnp.issubdtype(level.dtype, jnp.floating):
        level = level.astype(jnp.result_type(float))
    value_check(
        level,
        lambda x: bool(x >= 0.0),
        f"{cls_name}.{field_name}: must not be negative.",
    )
    _check_finite(cls_name, field_name, level)
    return level


def _check_typed_key(cls_name: str, key) -> None:
    """Require a typed PRNG key of shape ``()``, by shape *and* dtype."""
    dtype = getattr(key, "dtype", None)
    shape = getattr(key, "shape", None)
    if dtype is None or not jnp.issubdtype(dtype, jax.dtypes.prng_key):
        raise TypeError(
            f"{cls_name}.key: must be a typed PRNG key from jax.random.key, got "
            f"{type(key).__name__} of dtype {dtype}. A raw uint32 key array from "
            f"jax.random.PRNGKey has shape (2,), which would make the object's "
            f"batch_shape ambiguous; convert it with jax.random.wrap_key_data."
        )
    if shape != ():
        raise ValueError(
            f"{cls_name}.key: must be a single key of shape (), got shape {shape}"
        )
