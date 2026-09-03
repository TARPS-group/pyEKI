"""Conformance checks for policies, and for a forward model of your own.

Call the check matching your policy's axis on an instance of it, or
:func:`check_forward_model` on a model, to verify it against the requirements
:mod:`pyeki.eki` places on that axis.

============================= ==============================================
function                      checks
============================= ==============================================
:func:`check_schedule`        the two attributes, the increment's domain,
                              and purity
:func:`check_update`          shape and dtype, determinism, the subspace
                              property, and ``jit``-safety
:func:`check_inflation`       shape preservation, purity, and mean
                              preservation
:func:`check_stopping_rule`   a Python ``bool`` is returned, and purity
:func:`check_forward_model`   shape, dtype, row independence and determinism
                              of a forward model
:func:`synthetic_evaluation`  a small :class:`~pyeki.eki.Evaluation` to run
                              the checks against
============================= ==============================================

Two checks are conditional on a declaration, since a rule may legitimately
break the property:

==================== ================== =====================================
attribute            set it on          suppresses
==================== ================== =====================================
``leaves_span``      an update rule     the subspace check, for a rule that
                                        deliberately leaves the ensemble's
                                        affine span
``changes_mean``     an inflation       the mean-preservation check
==================== ================== =====================================

Both default to ``False`` when absent, so a rule that satisfies the property
declares nothing.

The two declarations above are attributes on the policy.
:func:`check_forward_model` instead takes ``stochastic`` as an argument, since
a bare function has nowhere to put an attribute.

Notes
-----
The behaviour these checks verify is specified by the "Ensemble Kalman
Inversion contract" page of the documentation.

The four policy checks each take a policy and a small
:class:`~pyeki.eki.Evaluation`, which :func:`synthetic_evaluation` builds, so
testing a schedule never means running a forward model.
:func:`check_forward_model` is the exception, and is here for two reasons. A
forward model is not a policy and has no protocol to conform to — the layer
defines no base class for one, and this check constructs no type and registers
nothing. But the obligations on the layer's one external callable are the
layer's own, and one of them, row independence, is invisible from inside a run
and visible from outside it.

Purity is the reason the harness exists. A policy holding state across steps
silently breaks resumption, and that is a failure a test suite can catch in
its own package's policies and cannot catch in a user's. Calling a policy
twice on one evaluation and comparing is two lines.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..linalg import PSDDiagonal, PSDLinOp
from .values import Evaluation

__all__ = [
    "check_forward_model",
    "check_inflation",
    "check_schedule",
    "check_stopping_rule",
    "check_update",
    "synthetic_evaluation",
]

_ATOL = 1e-9


def _close(got, want, what: str, atol: float = _ATOL) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    err = np.abs(got - want).max() if got.size else 0.0
    assert err <= atol, f"{what}: max abs err {err:.3e}"


def _identical(got, want, what: str) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    assert np.array_equal(got, want), f"{what}: not bit-identical"


def synthetic_evaluation(
    *,
    n_members: int = 6,
    u_dim: int = 3,
    v_dim: int = 4,
    step: int = 0,
    beta: float = 0.25,
    seed: int = 0,
) -> Evaluation:
    """A small :class:`~pyeki.eki.Evaluation` to run the checks against.

    A user testing their own schedule should not have to run a forward model
    to get one. The arrays are pseudo-random and independent of one another —
    there is no observation to make the residuals agree with the predictions —
    so this is a shape-and-purity fixture, not a physically consistent step.
    The residuals do have spread, so a schedule's criterion has something to
    measure.

    Parameters
    ----------
    n_members, u_dim, v_dim
        The sizes :math:`J`, :math:`P` and :math:`N`. Keyword-only.
    step, beta
        The step index and the level entering the step. Keyword-only.
    seed
        Seeds the NumPy generator that fills the arrays. Keyword-only.

    Returns
    -------
    Evaluation
        Unbatched, with every member valid.
    """
    rng = np.random.default_rng(seed)
    ensemble = rng.normal(size=(n_members, u_dim))
    predictions = rng.normal(size=(n_members, v_dim))
    residuals = rng.normal(size=(n_members, v_dim))
    anomalies = ensemble - ensemble.mean(axis=0)
    spread = np.linalg.norm(anomalies) / np.sqrt((n_members - 1) * u_dim)
    return Evaluation(
        step=step,
        beta=beta,
        ensemble=jnp.asarray(ensemble),
        predictions=jnp.asarray(predictions),
        whitened_residuals=jnp.asarray(residuals),
        rms_parameter_spread=jnp.asarray(spread),
        n_valid=n_members,
    )


def check_schedule(schedule, evaluation: Evaluation | None = None) -> None:
    """Check a :class:`~pyeki.eki.Schedule` against its protocol.

    Verifies that ``n_steps`` and ``beta_target`` are present, of the right
    types, and unchanged by reads; that ``next_increment`` returns either
    ``None`` or a scalar, finite, strictly positive value; and that it is
    **pure**, by calling it twice on the same evaluation and comparing
    bit-exactly.

    Parameters
    ----------
    schedule
        The schedule to check.
    evaluation
        The evaluation to call it on, defaulting to
        :func:`synthetic_evaluation`.

    Raises
    ------
    AssertionError
        On any violation, naming which.
    """
    evaluation = synthetic_evaluation() if evaluation is None else evaluation
    name = repr(schedule)

    n_steps = schedule.n_steps
    assert n_steps is None or (
        type(n_steps) is int and n_steps >= 1
    ), f"{name}.n_steps: must be None or a positive int, got {n_steps!r}"
    assert n_steps == schedule.n_steps, (
        f"{name}.n_steps: changed between reads; it is static metadata on a "
        f"frozen policy, not state"
    )

    beta_target = schedule.beta_target
    assert beta_target is None or (
        isinstance(beta_target, (int, float))
        and not isinstance(beta_target, bool)
        and beta_target > 0.0
    ), f"{name}.beta_target: must be None or a positive real, got {beta_target!r}"
    assert beta_target == schedule.beta_target, (
        f"{name}.beta_target: changed between reads; it is static metadata on "
        f"a frozen policy, not state"
    )

    first = schedule.next_increment(evaluation)
    second = schedule.next_increment(evaluation)
    if first is None:
        assert second is None, f"{name}.next_increment: not pure across two calls"
        return
    value = jnp.asarray(first)
    assert value.ndim == 0, (
        f"{name}.next_increment: must return a scalar, got shape {value.shape}"
    )
    assert bool(jnp.isfinite(value) & (value > 0.0)), (
        f"{name}.next_increment: must return a finite, strictly positive value, "
        f"got {value}"
    )
    _identical(
        value,
        jnp.asarray(second),
        f"{name}.next_increment: not pure across two calls on one evaluation",
    )


def check_update(
    update,
    key=None,
    *,
    ensemble=None,
    predictions=None,
    y=None,
    noise_cov: PSDLinOp | None = None,
    increment=0.25,
    step: int = 0,
    beta=0.25,
) -> None:
    """Check an :class:`~pyeki.eki.EnsembleUpdate` against its protocol.

    Verifies that the result is ``(J, P)`` with the incoming dtype; that the
    rule is deterministic given its key; that new members lie in the
    ensemble's own affine span, unless the rule declares ``leaves_span =
    True``; and that it is ``jit``-safe with static shapes.

    Parameters
    ----------
    update
        The update rule to check.
    key
        A typed PRNG key, defaulting to ``jax.random.key(0)``.
    ensemble, predictions, y, noise_cov, increment, step, beta
        The operands, each defaulting to a small synthetic one. Keyword-only.

    Raises
    ------
    AssertionError
        On any violation, naming which.
    """
    key = jax.random.key(0) if key is None else key
    rng = np.random.default_rng(1)
    if ensemble is None:
        ensemble = jnp.asarray(rng.normal(size=(6, 3)))
    if predictions is None:
        predictions = jnp.asarray(rng.normal(size=(ensemble.shape[0], 4)))
    v_dim = predictions.shape[-1]
    if noise_cov is None:
        noise_cov = PSDDiagonal(jnp.full((v_dim,), 0.5))
    if y is None:
        y = jnp.asarray(rng.normal(size=(v_dim,)))
    increment = jnp.asarray(increment, dtype=ensemble.dtype)
    beta = jnp.asarray(beta, dtype=ensemble.dtype)
    name = repr(update)

    def call(rule, key_, members, preds, obs, cov, delta, level):
        return rule(
            key_,
            ensemble=members,
            predictions=preds,
            y=obs,
            noise_cov=cov,
            increment=delta,
            step=step,
            beta=level,
        )

    operands = (key, ensemble, predictions, y, noise_cov, increment, beta)
    updated = jnp.asarray(call(update, *operands))
    assert updated.shape == ensemble.shape, (
        f"{name}: returned shape {updated.shape}, expected {ensemble.shape}"
    )
    assert updated.dtype == ensemble.dtype, (
        f"{name}: returned dtype {updated.dtype}, expected {ensemble.dtype}"
    )
    _identical(
        updated,
        jnp.asarray(call(update, *operands)),
        f"{name}: not deterministic given its key",
    )

    if not getattr(update, "leaves_span", False):
        anomalies = np.asarray(ensemble - ensemble.mean(axis=0))
        moved = np.asarray(updated) - np.asarray(ensemble.mean(axis=0))
        basis, *_ = np.linalg.svd(anomalies.T, full_matrices=False)
        rank = int(np.linalg.matrix_rank(anomalies))
        basis = basis[:, :rank]
        residual = moved - moved @ basis @ basis.T
        assert np.abs(residual).max() <= 1e-8 * max(1.0, np.abs(moved).max()), (
            f"{name}: left the ensemble's affine span. Declare "
            f"`leaves_span = True` on the rule if that is intended."
        )

    jitted = jax.jit(call) if _is_pytree(update) else jax.jit(call, static_argnums=(0,))
    _close(
        jitted(update, *operands),
        updated,
        f"{name}: differs under jax.jit",
    )


def check_inflation(inflation, key=None, ensemble=None, *, step: int = 0, beta=0.25):
    """Check an :class:`~pyeki.eki.Inflation` against its protocol.

    Verifies shape and dtype preservation; purity, by calling twice on the
    same arguments and comparing bit-exactly; and that the ensemble mean is
    preserved, unless the rule declares ``changes_mean = True``.

    Parameters
    ----------
    inflation
        The inflation to check.
    key
        A typed PRNG key, defaulting to ``jax.random.key(0)``.
    ensemble
        The ``(J, P)`` ensemble to inflate, defaulting to a small synthetic
        one.
    step, beta
        The step index and level to pass. Keyword-only.

    Raises
    ------
    AssertionError
        On any violation, naming which.
    """
    key = jax.random.key(0) if key is None else key
    if ensemble is None:
        ensemble = jnp.asarray(np.random.default_rng(2).normal(size=(6, 3)))
    beta = jnp.asarray(beta, dtype=ensemble.dtype)
    name = repr(inflation)

    inflated = jnp.asarray(
        inflation(key, ensemble=ensemble, step=step, beta=beta)
    )
    assert inflated.shape == ensemble.shape, (
        f"{name}: returned shape {inflated.shape}, expected {ensemble.shape}. "
        f"An Inflation is shape preserving."
    )
    assert inflated.dtype == ensemble.dtype, (
        f"{name}: returned dtype {inflated.dtype}, expected {ensemble.dtype}. "
        f"These members are what the forward model is called on, so a driver "
        f"rejects a narrower or non-floating one."
    )
    _identical(
        inflated,
        jnp.asarray(inflation(key, ensemble=ensemble, step=step, beta=beta)),
        f"{name}: not pure across two calls on one ensemble",
    )
    if not getattr(inflation, "changes_mean", False):
        want = np.asarray(jnp.mean(ensemble, axis=0))
        got = np.asarray(jnp.mean(inflated, axis=0))
        assert np.abs(got - want).max() <= 1e-10 * max(1.0, np.abs(want).max()), (
            f"{name}: did not preserve the ensemble mean. Declare "
            f"`changes_mean = True` on the rule if that is intended."
        )


def check_stopping_rule(stop, evaluation: Evaluation | None = None) -> None:
    """Check a :class:`~pyeki.eki.StoppingRule` against its protocol.

    Verifies that a Python ``bool`` is returned — not a 0-d array, which is
    truthy in a way that hides a traced value — and that the rule is pure.

    Parameters
    ----------
    stop
        The stopping rule to check.
    evaluation
        The evaluation to call it on, defaulting to
        :func:`synthetic_evaluation`.

    Raises
    ------
    AssertionError
        On any violation, naming which.
    """
    evaluation = synthetic_evaluation() if evaluation is None else evaluation
    name = repr(stop)
    first = stop(evaluation)
    assert type(first) is bool, (
        f"{name}: must return a Python bool, got {type(first).__name__}"
    )
    assert first == stop(evaluation), (
        f"{name}: not pure across two calls on one evaluation"
    )


def check_forward_model(
    forward,
    *,
    u_dim: int,
    v_dim: int,
    n_members: int = 6,
    seed: int = 0,
    stochastic: bool = False,
) -> None:
    """Check a forward model of your own from outside a run.

    A forward model is any callable from a ``(J, P)`` ensemble to ``(J, N)``
    predictions; there is no base class and nothing to register. This checks
    the obligations that can be checked from outside — including **row
    independence**, which no run detects — on an ensemble of pseudo-random
    parameters, plus determinism, which a run *permits* but does not require:
    pass ``stochastic=True`` for a model that is legitimately not
    deterministic.

    ============================================ ============================
    checked                                      not checked
    ============================================ ============================
    the return is array-like of shape ``(J, N)`` failure signalling on your
    at two ensemble sizes                        model's own error paths
    the return's dtype is real floating and no   whether a non-finite row
    narrower than the argument's                 *should* have been produced
    determinism, by calling twice                cost, side effects, device
    row independence, by permuting the members   placement
    and by re-evaluating a subset of them
    ============================================ ============================

    **This calls the model five times** — twice when ``stochastic=True`` —
    which for a real forward model is five evaluations of the expensive thing.
    Check a cheap configuration of it: a coarse grid, a short horizon, a stub
    solver.

    Parameters
    ----------
    forward
        The forward model: a callable taking one positional ``(J, P)`` array.
    u_dim, v_dim
        The sizes :math:`P` and :math:`N` the model is being checked at.
        Keyword-only.
    n_members
        The ensemble size :math:`J` to check at; the model is also called at
        ``J + 1``, since a model whose rows are independent cannot depend on
        how many there are. Keyword-only.
    seed
        Seeds the NumPy generator that fills the ensemble. Keyword-only.
    stochastic
        Declare that the model is not deterministic, skipping the row
        independence and determinism checks. Keyword-only.

    Raises
    ------
    AssertionError
        On any violation, naming which.

    Notes
    -----
    Non-finite predictions are *not* a violation: a non-finite row is how a
    forward model signals a failed member, so this check accepts them, and
    compares results in a way that treats two ``nan`` s as equal. A model that
    fails for every member of a pseudo-random ensemble still passes here and
    still cannot drive a run — read the shapes it returns, not only this
    check's silence.

    There is no check that the model left its argument alone, because a
    ``jax.Array`` cannot be written into: a wrapper that modifies
    ``np.asarray(ensemble)`` — a read-only view — raises
    ``assignment destination is read-only`` from wherever it wrote, without
    needing a check here.

    Row independence is checked twice, because the two ways of breaking it
    are caught by different comparisons and neither catches both: for a model
    whose row :math:`j` depends only on row :math:`j`,
    ``forward(ensemble[perm])`` is exactly ``forward(ensemble)[perm]``, since
    the rows are the same set of members either way — this is a **bit-exact**
    comparison, and it catches a model that is order-dependent across rows,
    such as one writing into a shared accumulator.

    A symmetric coupling — normalizing by the ensemble mean, say — survives a
    permutation, so the second check re-evaluates two of the members *without*
    the others: row :math:`j` must come out the same alongside a different
    set. That comparison is to a **tolerance** rather than bit-exact, because
    a differently shaped batch legitimately takes a different matmul kernel
    and rounds differently in the last bits, while a coupling changes the
    answer by :math:`O(1)`.

    Both are necessary conditions rather than sufficient ones. The subset
    check evaluates two members together, never one, since a run never calls
    a model with fewer than two.
    """
    if n_members < 3:
        raise ValueError(
            f"check_forward_model: n_members must be at least 3, got "
            f"{n_members}. Both row-independence comparisons need a subset "
            f"that is smaller than the ensemble and a permutation that is not "
            f"the identity, and at n_members < 3 neither exists — the checks "
            f"would pass a coupled model."
        )
    if u_dim < 1 or v_dim < 1:
        raise ValueError(
            f"check_forward_model: u_dim and v_dim must be at least 1, got "
            f"u_dim={u_dim}, v_dim={v_dim}"
        )
    name = getattr(forward, "__name__", None) or repr(forward)
    rng = np.random.default_rng(seed)
    ensemble = jnp.asarray(rng.normal(size=(n_members, u_dim)))

    predictions = _forward_call(forward, ensemble, name, u_dim, v_dim)
    # This assertion must precede the width check below: jnp.finfo raises
    # ValueError on a non-inexact dtype, so an integer return would surface as
    # that rather than as the diagnosis it deserves.
    assert jnp.issubdtype(predictions.dtype, jnp.floating), (
        f"{name}: returned dtype {predictions.dtype}, which is not a real "
        f"floating type; an integer or complex return is a ValueError in a run"
    )
    assert jnp.finfo(predictions.dtype).bits >= jnp.finfo(ensemble.dtype).bits, (
        f"{name}: returned {predictions.dtype} for a {ensemble.dtype} "
        f"ensemble; a narrower return is promoted, with a warning, and has "
        f"already lost the digits by then"
    )

    wider = jnp.asarray(rng.normal(size=(n_members + 1, u_dim)))
    _forward_call(forward, wider, name, u_dim, v_dim)

    if stochastic:
        return

    # Determinism first: a stochastic model breaks the permutation check too,
    # and "not deterministic" is the diagnosis it should get.
    _identical_allowing_nan(
        _forward_call(forward, ensemble, name, u_dim, v_dim),
        predictions,
        f"{name}: not deterministic across two calls on one ensemble. A "
        f"stochastic forward model is legal — declare it with "
        f"stochastic=True — but it damps the gain and costs extra evaluations",
    )
    # Never the identity: an identity permutation would make the comparison
    # below compare a result with itself, which asserts nothing. At small
    # n_members a fair draw returns it often.
    permutation = np.asarray(rng.permutation(n_members))
    while np.array_equal(permutation, np.arange(n_members)):
        permutation = np.asarray(rng.permutation(n_members))
    permutation = jnp.asarray(permutation)
    _identical_allowing_nan(
        _forward_call(forward, ensemble[permutation, :], name, u_dim, v_dim),
        predictions[permutation, :],
        f"{name}: row j of the return does not depend on row j of the "
        f"argument alone — permuting the members changed more than the order "
        f"of the predictions, so the model is order-dependent across rows. A "
        f"shared accumulator written to in row order does this, and a run "
        f"cannot detect it",
    )
    subset = jnp.asarray([1, n_members - 1])
    _close_allowing_nan(
        _forward_call(forward, ensemble[subset, :], name, u_dim, v_dim),
        predictions[subset, :],
        f"{name}: a member's prediction changed when it was evaluated "
        f"alongside different members, so row j of the return does not depend "
        f"on row j of the argument alone. A model that normalizes across the "
        f"ensemble does this, and a run cannot detect it",
    )


def _forward_call(forward, ensemble, name: str, u_dim: int, v_dim: int):
    """Call the model once and check the shape of what came back."""
    n_members = ensemble.shape[0]
    returned = forward(ensemble)
    try:
        predictions = jnp.asarray(returned)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"{name}: returned {type(returned).__name__}, which is not "
            f"array-like; the return may be a jax.Array, a NumPy array or a "
            f"nested Python sequence"
        ) from exc
    assert predictions.shape == (n_members, v_dim), (
        f"{name}: called with a ({n_members}, {u_dim}) ensemble and returned "
        f"shape {predictions.shape}, expected ({n_members}, {v_dim}). The "
        f"model is called once with the whole ensemble, not once per member; "
        f"wrap a per-member function with jax.vmap or a loop"
    )
    return predictions


def _identical_allowing_nan(got, want, what: str) -> None:
    """Bit-identity, counting two ``nan`` s as equal: failed rows are legal."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    assert np.array_equal(got, want, equal_nan=True), what


def _close_allowing_nan(got, want, what: str, rtol: float = 1e-8) -> None:
    """Agreement to a tolerance, with the non-finite pattern matching exactly.

    For comparisons across differently shaped batches, where the last bits
    legitimately differ and a genuine coupling differs by ``O(1)``.

    The tolerance is **per element**, not against a global maximum. A single
    global scale would be set by the largest prediction, so a model whose
    observables span orders of magnitude — a pressure beside a mass fraction,
    say — could couple its small observables freely and still pass.
    """
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    finite = np.isfinite(want)
    assert np.array_equal(finite, np.isfinite(got)), (
        f"{what} (the non-finite entries are in different places)"
    )
    if not finite.any():
        return
    tolerance = rtol * np.maximum(1.0, np.abs(want[finite]))
    excess = np.abs(got[finite] - want[finite]) - tolerance
    assert excess.max() <= 0.0, (
        f"{what} (worst element exceeds its tolerance by "
        f"{float(excess.max()):.3e})"
    )


def _is_pytree(obj) -> bool:
    """Whether ``obj`` is a registered pytree, and so a traced ``jit`` operand.

    A rule that is a registered pytree crosses the boundary as data, which is
    what the layer requires of one holding arrays. A rule that is a plain
    function is not a pytree and can only be static.
    """
    leaves = jax.tree.leaves(obj)
    return not (len(leaves) == 1 and leaves[0] is obj)
