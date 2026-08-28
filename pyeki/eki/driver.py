"""One rung, as its two phases, and the two loops over them.

=================== =========================================================
function            does
=================== =========================================================
:func:`evaluate`    inflate, call the forward model, repair, summarize
:func:`apply`       validate the increment, update, check finiteness, advance
:func:`advance`     :func:`evaluate` then :func:`apply`, at a given increment
:func:`iterate`     the driver as a generator, yielding after every rung
:func:`run`         the driver as a function, returning an
                    :class:`~pyeki.eki.EKIResult`
=================== =========================================================

The two phases are public because the forward evaluation is the resource a
run is organized around, and separating them is what lets a caller spend it
deliberately: one :class:`~pyeki.eki.Evaluation` serves any number of trial
increments, so a backtracking or damped loop costs one forward evaluation per
rung plus one per rejection rather than two per trial.

:func:`run` and :func:`iterate` are two wrappers around one private driver.
Neither is implemented over the other: all three end-paths look alike from
outside the yield stream, so a consumer of it cannot tell a stopping rule from
an exhausted ladder, and only the driver itself knows why it stopped.

Conventions shared by everything in the module:

- **The loop is ordinary Python, not** :func:`jax.lax.scan`. The forward
  model is an arbitrary callable — possibly a subprocess, a job-scheduler
  submission, or a non-traceable legacy code — so it can never be traced.
  Every *array* computation is nonetheless ``jit``-safe with static shapes,
  and the number of compilations is bounded independently of the number of
  steps.
- **The forward model is any callable** ``(J, P) -> (J, N)``, never traced
  and never inspected. Failure is signalled by non-finite predictions: a
  wrapper around a model that may crash, time out or lose a worker must
  catch that itself and return a non-finite row.
- **Progress is reported through the standard library's** :mod:`logging`, on
  the logger named ``pyeki.eki``: one record per step at ``INFO`` and one at
  ``WARNING`` when any member fails. No handler is installed and no
  configuration is read, so a caller who does nothing sees nothing.

Notes
-----
The behaviour of this module is specified by the "Ensemble Kalman Inversion
contract" page of the documentation, which is normative.

One rung synchronizes with the device a small fixed number of times — the
validity count, the increment, and the updated ensemble's finiteness — and
those reads cannot be coalesced, since the increment decides whether to
dispatch the update and the finiteness check reads a value the update
produces. It is a deliberate cost: :math:`O(1)` scalars and one reduction
against :math:`J` forward-model evaluations, and it is what allows
termination, validation and adaptive increments to be ordinary Python.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import warnings

import jax
import jax.numpy as jnp
from jax import Array

from ..linalg import PSDLinOp
from .helpers import (
    _anomalies,
    effective_sample_size,
    repair_failed_members,
)
from .policies import TransformUpdate
from .values import (
    SCHEDULE_EXHAUSTED,
    STOPPING_RULE,
    EKIError,
    EKIResult,
    EKIState,
    Evaluation,
    HistoryRecord,
    OnFailure,
)

__all__ = ["advance", "apply", "evaluate", "iterate", "run"]

#: The layer's logger. No handler is installed; a caller who wants progress
#: reports adds one.
logger = logging.getLogger("pyeki.eki")

#: Relative slack on a budget's exhaustion check. Relative rather than
#: absolute so that a small ``beta_target`` is not swallowed whole: an
#: absolute floor would make any budget at or below it exhaust at
#: :math:`\beta = 0`, returning an untouched ensemble and an empty history
#: with nothing raised.
_BUDGET_TOL_RELATIVE = 1e-12

_ON_FAILURE = ("repair", "raise")


# ---------------------------------------------------------------------------
# the two phases
# ---------------------------------------------------------------------------


def evaluate(
    state: EKIState,
    forward,
    y,
    noise_cov,
    *,
    inflation=None,
    on_failure: OnFailure = "repair",
) -> Evaluation:
    """Run the forward model once and summarize what it produced.

    Takes **no increment** and moves nothing: ``state`` is untouched, and the
    level and the step index are carried into the result unchanged. In order:
    split the key three ways, inflate, evaluate the forward model, repair any
    failed members, and summarize.

    Parameters
    ----------
    state
        The state to evaluate at.
    forward
        The forward model, a callable mapping the ``(J, P)`` ensemble to a
        ``(J, N)`` array of predictions. Never traced.
    y
        The observation, a ``(N,)`` finite array.
    noise_cov
        The **base** observation-noise covariance, a
        :class:`~pyeki.linalg.PSDLinOp` of side ``N`` supporting ``whiten``,
        with ``batch_shape == ()``.
    inflation
        An :class:`~pyeki.eki.Inflation`, or ``None`` for none — in which
        case the ensemble passes through bit-exactly. Keyword-only.
    on_failure
        ``"repair"`` (the default) to replace failed members by the valid
        centre, or ``"raise"`` to raise :class:`~pyeki.eki.EKIError` on any
        failure. Keyword-only.

    Returns
    -------
    Evaluation
        Carrying the members that were evaluated — after inflation and after
        repair — their predictions, and the whitened residuals.

    Raises
    ------
    EKIError
        If fewer than two members are valid, or if any member is invalid
        under ``on_failure="raise"``. Carries ``state`` and an empty
        ``history``.
    ValueError
        If the problem's shapes are wrong, if the forward model's output is
        not ``(J, N)`` of a real floating dtype, if an inflation's output is
        not ``(J, P)``, or if ``on_failure`` is not one of the two permitted
        strings.
    TypeError
        If ``state`` is not an :class:`~pyeki.eki.EKIState`, or ``noise_cov``
        not a :class:`~pyeki.linalg.PSDLinOp`.
    UnsupportedOpError
        If ``noise_cov`` does not support ``whiten``.

    Notes
    -----
    The key is split into exactly three, ``(key_next, key_inflate,
    key_update)``, whatever the policies are. Fixed arity means turning
    inflation on or off does not shift the update's random stream. Both
    phases perform this split independently from the same ``state.key``,
    rather than :func:`evaluate` handing the update's key on through the
    evaluation: the split is deterministic, so they agree, and it keeps the
    update's key off an object that every schedule and stopping rule
    receives.

    A member is *invalid* when its prediction row contains any non-finite
    entry. When every member is valid the ensemble and predictions pass
    through **bit-exactly untouched** — the repair is skipped in Python on
    the synchronized valid count, because the repair formula is
    mathematically the identity there but not bit-exactly so.
    """
    _check_state(state, "evaluate")
    y, n_obs = _check_problem("evaluate", y, noise_cov)
    _check_on_failure(on_failure)
    return _evaluate(state, forward, y, noise_cov, inflation, on_failure, n_obs)


def _evaluate(state, forward, y, noise_cov, inflation, on_failure, n_obs):
    """Steps 1-5 of a rung, with the problem already validated."""
    _, key_inflate, _ = _split_key(state.key)

    members = state.ensemble
    if inflation is not None:
        members = jnp.asarray(
            inflation(
                key_inflate, ensemble=members, step=state.step, beta=state.beta
            )
        )
        if members.shape != state.ensemble.shape:
            raise ValueError(
                f"evaluate: the inflation returned shape {members.shape}, "
                f"expected {state.ensemble.shape}. An Inflation is shape "
                f"preserving."
            )

    predictions = _check_predictions(forward(members), state.n_members, n_obs)
    members, predictions, n_valid = _handle_failures(
        state, members, predictions, on_failure=on_failure
    )
    whitened_residuals, spread = _summarize(y, members, predictions, noise_cov)
    return Evaluation(
        step=state.step,
        beta=state.beta,
        ensemble=members,
        predictions=predictions,
        whitened_residuals=whitened_residuals,
        rms_parameter_spread=spread,
        n_valid=n_valid,
    )


def apply(
    state: EKIState,
    evaluation: Evaluation,
    *,
    increment,
    y,
    noise_cov,
    update=TransformUpdate(),  # noqa: B008 - frozen and field-less
):
    """Move ``state`` forward by ``increment``, using an evaluation of it.

    Chooses nothing: no schedule, no stopping rule, no ``max_steps``. Those
    are the driver's, and everything they decide is passed in. In order:
    validate the increment and the evaluation's provenance, split the key,
    call the update rule, check the result is finite, and advance.

    Parameters
    ----------
    state
        The state to move.
    evaluation
        An :class:`~pyeki.eki.Evaluation` obtained from ``state``.
    increment
        The tempering increment, a scalar that must be finite and **strictly
        positive**. Keyword-only.
    y
        The observation, a ``(N,)`` finite array. Keyword-only.
    noise_cov
        The **base** noise covariance; the update conditions with
        ``noise_cov / increment``. Keyword-only.
    update
        An :class:`~pyeki.eki.EnsembleUpdate`, defaulting to
        :class:`~pyeki.eki.TransformUpdate`. Keyword-only.

    Returns
    -------
    tuple
        The new :class:`~pyeki.eki.EKIState` and the step's
        :class:`~pyeki.eki.HistoryRecord`.

    Raises
    ------
    ValueError
        If ``increment`` is not a finite, strictly positive scalar, if
        ``evaluation`` does not belong to ``state``, or if the update's
        output is not ``(J, P)`` of the incoming dtype.
    TypeError
        If ``state`` is not an :class:`~pyeki.eki.EKIState`, ``evaluation``
        not an :class:`~pyeki.eki.Evaluation`, or ``noise_cov`` not a
        :class:`~pyeki.linalg.PSDLinOp`.
    EKIError
        If the updated ensemble contains a non-finite entry. Carries
        ``state`` and an empty ``history``.

    Notes
    -----
    ``y`` and ``noise_cov`` are passed again rather than carried on the
    evaluation, which holds no problem data — it is a record of an
    evaluation, not a bound problem. That is also what makes this the entry
    point for anything varying the *data* between rungs, which the driver
    deliberately fixes for a whole run.

    **The evaluation must belong to the state**, checked on ``step`` and
    ``beta``. Without the check, pairing an evaluation with a different state
    would take the key split from one and the members from the other and
    return finite, plausible nonsense.

    A zero increment is rejected even though it raises nothing downstream:
    :math:`R/0` whitens to zero, so the gain vanishes and the ensemble is
    returned unchanged while :math:`\\beta` never advances, and an adaptive
    ladder would then spin until ``max_steps``.

    The increment is validated **before** the key split and before any array
    work, which is what makes the two-phase split worth having: a single
    fused step would have spent :math:`J` model evaluations before rejecting
    an argument the caller got wrong.
    """
    _check_state(state, "apply")
    y, _ = _check_problem("apply", y, noise_cov)
    dbeta = _check_increment("apply", increment)
    _check_provenance(state, evaluation)
    return _apply(state, evaluation, dbeta, y, noise_cov, update)


def _apply(state, evaluation, dbeta, y, noise_cov, update):
    """Steps 6-9 of a rung, with the increment already validated."""
    key_next, _, key_update = _split_key(state.key)
    updated = jnp.asarray(
        update(
            key_update,
            ensemble=evaluation.ensemble,
            predictions=evaluation.predictions,
            y=y,
            noise_cov=noise_cov,
            increment=dbeta,
            step=state.step,
            beta=state.beta,
        )
    )
    if updated.shape != state.ensemble.shape:
        raise ValueError(
            f"apply: the update returned shape {updated.shape}, expected "
            f"{state.ensemble.shape}"
        )
    if updated.dtype != state.ensemble.dtype:
        raise ValueError(
            f"apply: the update returned dtype {updated.dtype}, expected "
            f"{state.ensemble.dtype}. A float32 update quietly demotes a run's "
            f"precision, and every downstream check still passes at its own "
            f"tolerance."
        )
    if not bool(jnp.all(jnp.isfinite(updated))):
        raise EKIError(
            f"apply: the update returned a non-finite ensemble at step "
            f"{state.step}, beta {float(state.beta):g}. Silent nan propagation "
            f"through a long run is the worst outcome available to this layer, "
            f"so it is raised here rather than carried forward.",
            state=state,
        )
    new_state = EKIState(updated, state.beta + dbeta, state.step + 1, key_next)
    return new_state, _record(evaluation, dbeta)


def advance(
    state: EKIState,
    forward,
    y,
    noise_cov,
    *,
    increment,
    update=TransformUpdate(),  # noqa: B008 - frozen and field-less
    inflation=None,
    on_failure: OnFailure = "repair",
):
    """One rung at a known increment: :func:`evaluate` then :func:`apply`.

    Exactly ``apply(state, evaluate(state, forward, y, noise_cov,
    inflation=..., on_failure=...), increment=increment, y=y,
    noise_cov=noise_cov, update=...)``, provided because one rung at a known
    increment is the common case.

    Parameters
    ----------
    state
        The state to move.
    forward, y, noise_cov
        The problem, as :func:`evaluate` takes them.
    increment
        The tempering increment, a finite, strictly positive scalar.
        Keyword-only.
    update
        An :class:`~pyeki.eki.EnsembleUpdate`. Keyword-only.
    inflation
        An :class:`~pyeki.eki.Inflation`, or ``None`` for none. Keyword-only.
    on_failure
        ``"repair"`` or ``"raise"``. Keyword-only.

    Returns
    -------
    tuple
        The new :class:`~pyeki.eki.EKIState` and the step's
        :class:`~pyeki.eki.HistoryRecord`.

    Raises
    ------
    EKIError, ValueError, TypeError
        As :func:`evaluate` and :func:`apply` do.

    Notes
    -----
    Named ``advance`` rather than ``step`` because ``step`` is an index on
    three classes and a keyword argument, and ``step = step(...)`` is a
    shadowing mistake waiting to happen.
    """
    evaluation = evaluate(
        state, forward, y, noise_cov, inflation=inflation, on_failure=on_failure
    )
    return apply(
        state,
        evaluation,
        increment=increment,
        y=y,
        noise_cov=noise_cov,
        update=update,
    )


# ---------------------------------------------------------------------------
# the two loops
# ---------------------------------------------------------------------------


def iterate(
    state: EKIState,
    forward,
    y,
    noise_cov,
    *,
    schedule,
    update=TransformUpdate(),  # noqa: B008 - frozen and field-less
    inflation=None,
    stop=None,
    on_failure: OnFailure = "repair",
    max_steps: int = 1000,
):
    """The driver as a generator: yields after every rung.

    Yields ``(EKIState, HistoryRecord, Evaluation)`` after each iteration,
    including the terminal evaluation-only iteration, and **returns** the
    terminating status as its :class:`StopIteration` value.

    This is the extension point for anything that needs to *observe* or
    *interrupt* a run: per-step checkpointing, custom logging, a wall-clock
    budget, stopping on parameter stagnation, an early ``break``. Exceptions
    propagate; abandoning the generator is safe. Anything that needs to
    *revisit* a rung — backtracking, damping, trial increments — uses
    :func:`evaluate` and :func:`apply` directly instead.

    Parameters
    ----------
    state
        The state to start or resume from.
    forward, y, noise_cov
        The problem, as :func:`evaluate` takes them. Bound once for the whole
        run.
    schedule
        A :class:`~pyeki.eki.Schedule`. Keyword-only.
    update
        An :class:`~pyeki.eki.EnsembleUpdate`, defaulting to
        :class:`~pyeki.eki.TransformUpdate`. Keyword-only.
    inflation
        An :class:`~pyeki.eki.Inflation`, or ``None`` for none — the default.
        Keyword-only.
    stop
        A :class:`~pyeki.eki.StoppingRule`, or ``None``. Keyword-only.
    on_failure
        ``"repair"`` or ``"raise"``. Keyword-only.
    max_steps
        A safety bound on the iterations of **this call**, a positive
        ``int``, default ``1000``. Keyword-only.

    Yields
    ------
    tuple
        ``(state, record, evaluation)`` after each iteration.

    Raises
    ------
    EKIError
        If ``max_steps`` is reached, or on any of the failure conditions
        :func:`evaluate` and :func:`apply` raise. Every raise path carries
        ``state`` and ``history``.
    ValueError
        On any invalid argument, including a ``max_steps`` too small to
        accommodate the schedule's own floor-bound worst case. Raised on the
        first iteration rather than at the call, since a generator's body
        does not run until it is first advanced.
    TypeError
        If ``state`` is not an :class:`~pyeki.eki.EKIState`, or ``noise_cov``
        not a :class:`~pyeki.linalg.PSDLinOp`.

    Notes
    -----
    The evaluation is yielded because every recipe this layer recommends
    needs it and the record cannot carry it: the record holds scalars only,
    so per-member misfits, the whitened residual matrix and the posterior
    predictive are reachable only here. A caller who wants none of them
    ignores the third element.

    A caller who ends the loop themselves has everything an
    :class:`~pyeki.eki.EKIResult` needs — the last yielded state, the records
    they accumulated, :data:`~pyeki.eki.INTERRUPTED`, and the last yielded
    evaluation.

    **``max_steps`` bounds the iterations of this call, not
    ``state.step``.** The distinction is invisible on a fresh run and
    decisive on a resumed one: bounding the cumulative index would make the
    catch-checkpoint-resume recovery a guaranteed no-op for the ``max_steps``
    raise itself, and would silently shrink a resumed run's allowance.
    """
    return (
        yield from _drive(
            state,
            forward,
            y,
            noise_cov,
            schedule=schedule,
            update=update,
            inflation=inflation,
            stop=stop,
            on_failure=on_failure,
            max_steps=max_steps,
            where="iterate",
        )
    )


def run(
    state: EKIState,
    forward,
    y,
    noise_cov,
    *,
    schedule,
    update=TransformUpdate(),  # noqa: B008 - frozen and field-less
    inflation=None,
    stop=None,
    on_failure: OnFailure = "repair",
    max_steps: int = 1000,
) -> EKIResult:
    """The driver as a function: run the ladder and report what happened.

    Parameters
    ----------
    state
        The state to start or resume from.
    forward, y, noise_cov
        The problem, as :func:`evaluate` takes them. Bound once for the whole
        run.
    schedule
        A :class:`~pyeki.eki.Schedule`. Keyword-only.
    update
        An :class:`~pyeki.eki.EnsembleUpdate`. Keyword-only.
    inflation
        An :class:`~pyeki.eki.Inflation`, or ``None`` for none — the default.
        Keyword-only.
    stop
        A :class:`~pyeki.eki.StoppingRule`, or ``None``. Keyword-only.
    on_failure
        ``"repair"`` or ``"raise"``. Keyword-only.
    max_steps
        A safety bound on the iterations of this call, a positive ``int``,
        default ``1000``. Keyword-only.

    Returns
    -------
    EKIResult
        With ``status`` either :data:`~pyeki.eki.SCHEDULE_EXHAUSTED` or
        :data:`~pyeki.eki.STOPPING_RULE`;
        :data:`~pyeki.eki.INTERRUPTED` is never produced here.

    Raises
    ------
    EKIError
        As :func:`iterate` does, carrying ``state`` and ``history`` so that a
        caught failure is a resumable checkpoint rather than discarded work.
    ValueError
        On any invalid argument, including a ``max_steps`` too small to
        accommodate the schedule's own floor-bound worst case.
    TypeError
        If ``state`` is not an :class:`~pyeki.eki.EKIState`, or ``noise_cov``
        not a :class:`~pyeki.linalg.PSDLinOp`.

    Warns
    -----
    UserWarning
        Once per run in which any member ever failed. Under
        ``on_failure="repair"`` a run can otherwise complete, return a
        normal-looking result, and have been conditioning on a covariance
        damped at every rung.

    Notes
    -----
    Running on a state returned by a previous run **continues** it: same
    schedule, same policies, and the tail of the run is bit-identical to an
    uninterrupted one. This is the sole mechanism for checkpointing, and it
    is why policies may not hold iteration state.

    Chaining a *new* ladder onto a finished state is a different thing and is
    a silent no-op — use :meth:`EKIState.restart <pyeki.eki.EKIState.restart>`.

    There is no ``"max_steps"`` status, because exceeding ``max_steps``
    raises. The bound is a safety net against a schedule that can never be
    exhausted and a run with no stopping rule; a genuinely step-limited run
    is a :class:`~pyeki.eki.FixedSchedule` with that many rungs, or a
    ``break`` in an :func:`iterate` loop.
    """
    driver = _drive(
        state,
        forward,
        y,
        noise_cov,
        schedule=schedule,
        update=update,
        inflation=inflation,
        stop=stop,
        on_failure=on_failure,
        max_steps=max_steps,
    )
    records: list[HistoryRecord] = []
    last_evaluation = None
    final_state = state
    while True:
        try:
            final_state, record, last_evaluation = next(driver)
        except StopIteration as finished:
            status = finished.value
            break
        records.append(record)
    result = EKIResult(
        state=final_state,
        history=tuple(records),
        status=status,
        last_evaluation=last_evaluation,
    )
    worst = result.min_n_valid
    if worst is not None and worst < state.n_members:
        warnings.warn(
            f"pyeki.eki.run: some forward-model evaluations failed; the worst "
            f"step had {worst} of {state.n_members} members valid. "
            f"Each such step conditioned on a covariance damped by "
            f"(n_valid - 1) / (J - 1). Inspect result.stacked.n_valid.",
            stacklevel=2,
        )
    return result


# ---------------------------------------------------------------------------
# private: the one driver both loops wrap
# ---------------------------------------------------------------------------


def _drive(
    state,
    forward,
    y,
    noise_cov,
    *,
    schedule,
    update,
    inflation,
    stop,
    on_failure,
    max_steps,
    where="run",
):
    """The loop that knows why it stopped, as a generator returning the status."""
    _check_state(state, where)
    y, n_obs = _check_problem(where, y, noise_cov)
    _check_on_failure(on_failure)
    _check_max_steps(where, max_steps)
    _check_schedule(where, schedule)
    _check_budget_against_bound(schedule, max_steps)

    records: list[HistoryRecord] = []
    completed = 0
    while True:
        if _ladder_finished(schedule, state.step, state.beta):
            status = SCHEDULE_EXHAUSTED
            break
        if completed >= max_steps:
            raise EKIError(
                f"{where}: max_steps={max_steps} reached at step {state.step}, beta "
                f"{float(state.beta):g}, with schedule {schedule!r} and "
                f"{'a' if stop is not None else 'no'} stopping rule. An "
                f"unbounded schedule with no stopping rule is the usual cause.",
                state=state,
                history=records,
            )
        try:
            evaluation = _evaluate(
                state, forward, y, noise_cov, inflation, on_failure, n_obs
            )
        except EKIError as failure:
            failure.history = tuple(records)
            raise
        if evaluation.n_valid < state.n_members:
            logger.warning(
                "step %d: %d of %d members' predictions were finite; the rest "
                "were repaired to the valid centre",
                evaluation.step,
                evaluation.n_valid,
                state.n_members,
            )

        if stop is not None and bool(stop(evaluation)):
            record = _terminal_record(evaluation)
            records.append(record)
            _log_step(record)
            yield state, record, evaluation
            status = STOPPING_RULE
            break

        increment = schedule.next_increment(evaluation)
        if increment is None:
            record = _terminal_record(evaluation)
            records.append(record)
            _log_step(record)
            yield state, record, evaluation
            status = SCHEDULE_EXHAUSTED
            break

        try:
            state, record = _apply(
                state,
                evaluation,
                _check_increment(where, increment),
                y,
                noise_cov,
                update,
            )
        except EKIError as failure:
            failure.history = tuple(records)
            raise
        records.append(record)
        completed += 1
        _log_step(record)
        yield state, record, evaluation

    if not records:
        logger.warning(
            "the run performed no forward evaluations: its ladder was already "
            "finished on entry, at step %d and beta %g. Chaining a new ladder "
            "onto a finished state needs EKIState.restart().",
            state.step,
            float(state.beta),
        )
    return status


def _check_schedule(where: str, schedule) -> None:
    """Require the two declarative attributes the exhaustion check reads."""
    missing = [
        name for name in ("n_steps", "beta_target") if not hasattr(schedule, name)
    ]
    if missing or not callable(getattr(schedule, "next_increment", None)):
        raise ValueError(
            f"{where}: a Schedule must provide next_increment and the attributes "
            f"n_steps and beta_target; {schedule!r} is missing "
            f"{', '.join(missing) or 'next_increment'}. Either attribute may be "
            f"None, but the driver reads both to decide exhaustion."
        )


def _ladder_finished(schedule, step: int, beta) -> bool:
    """The driver's exhaustion check, read from the schedule's attributes."""
    n_steps = schedule.n_steps
    if n_steps is not None and step >= n_steps:
        return True
    beta_target = schedule.beta_target
    if beta_target is None:
        return False
    return bool(beta >= beta_target - _BUDGET_TOL_RELATIVE * beta_target)


# ---------------------------------------------------------------------------
# private: the array work of one rung
# ---------------------------------------------------------------------------


def _split_key(key):
    """The pinned three-way split, ``(key_next, key_inflate, key_update)``.

    The arity is fixed whatever the policies are, so that turning inflation
    on does not shift the update's random stream. Both phases perform it
    independently from the same key; the split is deterministic, so they
    agree.
    """
    key_next, key_inflate, key_update = jax.random.split(key, 3)
    return key_next, key_inflate, key_update


@jax.jit
def _valid_mask(predictions: Array) -> Array:
    """``True`` where a member's whole prediction row is finite."""
    return jnp.all(jnp.isfinite(predictions), axis=-1)


@jax.jit
def _summarize(y: Array, members: Array, predictions: Array, noise_cov):
    """The whitened residuals and the root-mean-square parameter spread."""
    whitened_residuals = noise_cov.whiten(y - predictions)
    anomalies = _anomalies(members)
    n_members, u_dim = members.shape[-2], members.shape[-1]
    spread = jnp.linalg.norm(anomalies) / jnp.sqrt((n_members - 1) * u_dim)
    return whitened_residuals, spread


def _handle_failures(state, members, predictions, *, on_failure):
    """Apply the ``on_failure`` policy, skipping the repair when nothing failed."""
    valid = _valid_mask(predictions)
    n_valid = int(jnp.sum(valid))
    n_members = state.n_members
    if n_valid == n_members:
        return members, predictions, n_valid
    if n_valid < 2:
        raise EKIError(
            f"evaluate: only {n_valid} of {n_members} members produced finite "
            f"predictions at step {state.step}, beta {float(state.beta):g}. At "
            f"least 2 are required: a single valid member has no anomalies.",
            state=state,
        )
    if on_failure == "raise":
        failed = [int(index) for index in jnp.flatnonzero(~valid)]
        raise EKIError(
            f"evaluate: {n_members - n_valid} of {n_members} members produced "
            f"non-finite predictions at step {state.step}, beta "
            f"{float(state.beta):g}, and on_failure='raise'. Failed members: "
            f"{failed}.",
            state=state,
        )
    repaired, repaired_predictions = repair_failed_members(
        ensemble=members, predictions=predictions, valid=valid
    )
    return repaired, repaired_predictions, n_valid


def _record(evaluation: Evaluation, increment: Array) -> HistoryRecord:
    """Build the step's record from the evaluation and the chosen increment."""
    misfits = evaluation.misfits
    return HistoryRecord(
        step=jnp.asarray(evaluation.step),
        n_valid=jnp.asarray(evaluation.n_valid),
        beta=evaluation.beta,
        increment=increment,
        beta_next=evaluation.beta + increment,
        misfit_mean=jnp.mean(misfits),
        misfit_min=jnp.min(misfits),
        misfit_max=jnp.max(misfits),
        centre_misfit=evaluation.centre_misfit,
        spread=evaluation.rms_parameter_spread,
        ess=effective_sample_size(misfits, increment),
    )


def _terminal_record(evaluation: Evaluation) -> HistoryRecord:
    """The record of an evaluation whose update was discarded.

    ``increment`` is exactly ``0.0`` and ``beta_next == beta``; ``ess`` is
    the literal ``float(J)``, written here rather than obtained from
    :func:`~pyeki.eki.effective_sample_size`, since ``exp(log J)`` is not
    ``J`` in floating point. It appears at most once, always last, and in
    exactly two cases: a stopping rule fired, or a schedule's
    ``next_increment`` returned ``None``. A zero increment in a record
    therefore means "evaluated, then stopped".
    """
    record = _record(evaluation, jnp.zeros_like(evaluation.beta))
    return dataclasses.replace(
        record, ess=jnp.asarray(float(evaluation.n_members))
    )


def _log_step(record: HistoryRecord) -> None:
    """One ``INFO`` record per step: the step, the level, the increment, the misfit."""
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "step %d: beta %g -> %g, increment %g, mean misfit %g",
            int(record.step),
            float(record.beta),
            float(record.beta_next),
            float(record.increment),
            float(record.misfit_mean),
        )


# ---------------------------------------------------------------------------
# private: call-time validation
# ---------------------------------------------------------------------------


def _check_state(state, where: str) -> None:
    if not isinstance(state, EKIState):
        raise TypeError(
            f"{where}: state must be an EKIState, got {type(state).__name__}"
        )
    if state.batch_shape != ():
        raise ValueError(
            f"{where}: {state!r} is a vmapped family. A run cannot be traced, so "
            f"a family of runs is a Python loop over run(), not a jax.vmap."
        )


def _check_problem(where: str, y, noise_cov):
    """Validate the problem's shapes once, and ``y``'s finiteness."""
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{where}: noise_cov must be a pyeki.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
    if noise_cov.batch_shape != ():
        raise ValueError(
            f"{where}: {noise_cov!r} is a vmapped family; a run binds one noise "
            f"covariance."
        )
    # Demanded here rather than at the first whitening, so that a covariance
    # without a cheap whitener costs no forward-model evaluations. The
    # UnsupportedOpError is constructed by the operator that lacks the
    # operation and propagated unmodified.
    noise_cov._require("whiten")
    n_obs = noise_cov.shape[0]
    y = jnp.asarray(y)
    if y.ndim != 1 or y.shape[0] != n_obs:
        raise ValueError(
            f"{where}: expected y of shape ({n_obs},) to match {noise_cov!r}, "
            f"got shape {y.shape}"
        )
    if not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError(
            f"{where}: y must be finite. A non-finite observation otherwise "
            f"surfaces as a full-budget run of nan updates or as an "
            f"'increment not finite' error, neither of which names y."
        )
    return y, n_obs


def _check_predictions(predictions, n_members: int, n_obs: int) -> Array:
    """Validate the forward model's output: the one check that runs every step."""
    predictions = jnp.asarray(predictions)
    if predictions.shape != (n_members, n_obs):
        raise ValueError(
            f"evaluate: the forward model returned shape {predictions.shape}, "
            f"expected ({n_members}, {n_obs})"
        )
    if not jnp.issubdtype(predictions.dtype, jnp.floating):
        raise ValueError(
            f"evaluate: the forward model returned dtype {predictions.dtype}, "
            f"expected a real floating dtype"
        )
    return predictions


def _check_increment(where: str, increment) -> Array:
    """Require a finite, strictly positive scalar, as a 0-d float array."""
    value = jnp.asarray(increment)
    if value.ndim != 0:
        raise ValueError(
            f"{where}: the increment must be a scalar, got shape {value.shape}"
        )
    if not jnp.issubdtype(value.dtype, jnp.floating):
        value = value.astype(jnp.result_type(float))
    if not bool(jnp.isfinite(value) & (value > 0.0)):
        raise ValueError(
            f"{where}: the increment must be finite and strictly positive, got "
            f"{value}. A zero increment leaves the ensemble unchanged while beta "
            f"never advances, so an adaptive ladder would spin until max_steps."
        )
    return value


def _check_provenance(state: EKIState, evaluation) -> None:
    """Require that the evaluation came from this state."""
    if not isinstance(evaluation, Evaluation):
        raise TypeError(
            f"apply: evaluation must be an Evaluation, got "
            f"{type(evaluation).__name__}"
        )
    if evaluation.step != state.step or not bool(evaluation.beta == state.beta):
        raise ValueError(
            f"apply: the evaluation belongs to a different state — evaluation at "
            f"step {evaluation.step}, beta {float(evaluation.beta):g}; state at "
            f"step {state.step}, beta {float(state.beta):g}. Pairing the two "
            f"would take the key split from one and the members from the other "
            f"and return finite, plausible nonsense."
        )


def _check_on_failure(on_failure) -> None:
    if on_failure not in _ON_FAILURE:
        raise ValueError(
            f"on_failure: must be one of {_ON_FAILURE}, got {on_failure!r}. An "
            f"unrecognized value raises rather than falling back to 'repair': a "
            f"typo such as 'Raise' must not quietly select the opposite "
            f"behaviour."
        )


def _check_max_steps(where: str, max_steps) -> None:
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError(
            f"{where}: max_steps must be a positive int, got {max_steps!r}"
        )


def _check_budget_against_bound(schedule, max_steps: int) -> None:
    """Refuse a bound too small for the schedule's own floor-bound worst case.

    A schedule exposing ``beta_target`` and a floor implies a worst case of
    ``ceil(beta_target / min_increment)`` rungs, and this raises before the
    first forward evaluation when ``max_steps`` is below it. A schedule that
    does not expose a floor is not checked.
    """
    beta_target = getattr(schedule, "beta_target", None)
    floor = getattr(schedule, "min_increment", None)
    if beta_target is None or floor is None:
        return
    try:
        worst_case = math.ceil(float(beta_target) / float(floor))
    except (TypeError, ValueError, ZeroDivisionError):
        return
    if worst_case > max_steps:
        raise ValueError(
            f"run: max_steps={max_steps} cannot accommodate {schedule!r}, whose "
            f"floor of {floor} against a budget of {beta_target} needs up to "
            f"{worst_case} rungs. Raise max_steps to at least {worst_case}, or "
            f"raise min_increment. Checked here so that a run cannot spend its "
            f"whole evaluation budget and then report an EKIError on precisely "
            f"the badly-conditioned problems the floor exists to rescue."
        )
