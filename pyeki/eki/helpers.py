"""The array-level pieces of an EKI run, usable outside one.

The three quantities a run computes for itself, exposed so that a custom
schedule, a validation set, or a hand-driven loop can compute them the same
way.

============================== =============================================
function                       computes
============================== =============================================
:func:`misfits`                :math:`\\tfrac12\\lVert W(y - v)\\rVert^2`,
                               batched
:func:`effective_sample_size`  the effective sample size of the tempering
                               weights :math:`e^{-\\delta\\Phi}`
:func:`repair_failed_members`  the mean-preserving repair that moves failed
                               members to the valid centre
============================== =============================================

The misfit carries the factor :math:`\\tfrac12` and is measured against the
**base** noise covariance, never a tempered one. Every criterion, diagnostic
and stopping rule in :mod:`pyeki.eki` is written in terms of that quantity, so
the convention is fixed in exactly one place.

Notes
-----
The behaviour of this module is specified by the "Ensemble Kalman Inversion
contract" page of the documentation, which is normative.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import logsumexp

from ..linalg import PSDLinOp, value_check

__all__ = ["effective_sample_size", "misfits", "repair_failed_members"]


def misfits(y, predictions, noise_cov) -> Array:
    """The whitened misfit of each prediction: ``(N,), (..., N) -> (...)``.

    .. math::

        \\Phi(v) \\;=\\; \\tfrac12 \\bigl\\lVert W(y - v) \\bigr\\rVert^2
        \\;=\\; \\tfrac12 (y - v)^\\top R^{-1} (y - v) ,

    with :math:`W` any whitener of ``noise_cov``. The value does not depend on
    which whitener the operator chose, by the operator layer's ``whiten``
    guarantee.

    Parameters
    ----------
    y
        The observation, a ``(N,)`` array.
    predictions
        Predicted observations, ``(..., N)`` — vectors along the trailing
        axis, any number of leading batch axes, as the operator layer's batch
        contract has it. An ensemble's ``(J, N)`` predictions give ``(J,)``
        misfits.
    noise_cov
        The observation-noise covariance :math:`R`, a
        :class:`~pyeki.linalg.PSDLinOp` of side ``N`` supporting ``whiten``.
        Pass the **base** covariance: a tempered :math:`R/\\delta` would
        rescale every misfit by :math:`\\delta`.

    Returns
    -------
    Array
        Shape ``(...)``, the batch axes of ``predictions``.

    Raises
    ------
    UnsupportedOpError
        If ``noise_cov`` does not support ``whiten``.
    TypeError
        If ``noise_cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
    ValueError
        If ``y`` is not ``(N,)``, if ``predictions``' trailing axis is not
        ``N``, or if ``noise_cov`` is a vmapped family.

    Notes
    -----
    The factor of :math:`\\tfrac12` is what makes :math:`e^{-\\beta\\Phi}` the
    tempered likelihood and :math:`2\\overline{\\Phi} \\approx N` the
    well-specified-fit benchmark. Halving or doubling it silently rescales
    every schedule parameter in the layer.
    """
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"misfits: noise_cov must be a pyeki.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
    n_obs = noise_cov.shape[0]
    y = jnp.asarray(y)
    if y.ndim != 1 or y.shape[0] != n_obs:
        raise ValueError(
            f"misfits: expected y of shape ({n_obs},) to match {noise_cov!r}, "
            f"got shape {y.shape}"
        )
    predictions = jnp.asarray(predictions)
    if predictions.ndim < 1 or predictions.shape[-1] != n_obs:
        raise ValueError(
            f"misfits: expected predictions of core shape (..., {n_obs}), got "
            f"shape {predictions.shape}"
        )
    return _misfits_from_residuals(noise_cov.whiten(y - predictions))


def effective_sample_size(misfits, increment) -> Array:
    """The effective sample size of the tempering weights: ``(J,), scalar -> 0-d``.

    For weights :math:`w_j(\\delta) = e^{-\\delta\\Phi_j}`,

    .. math::

        \\mathrm{ESS}(\\delta)
        = \\frac{\\bigl(\\sum_j w_j\\bigr)^2}{\\sum_j w_j^2}
        \\;=\\; \\exp\\bigl(2\\,\\mathrm{lse}(-\\delta\\Phi)
                            - \\mathrm{lse}(-2\\delta\\Phi)\\bigr)
        \\;\\in\\; [1, J] ,

    computed by the second, log-space form with ``lse`` a max-shifted
    log-sum-exp. The bracketing interval holds up to round-off rather than
    exactly: the value at :math:`\\delta = 0` is ``exp(log J)``, which for
    most :math:`J` sits an ulp either side of :math:`J`. It is monotone
    non-increasing in :math:`\\delta`, strictly so unless every misfit is
    equal.

    Parameters
    ----------
    misfits
        The per-member misfits :math:`\\Phi_j`, a ``(J,)`` array.
    increment
        The tempering increment :math:`\\delta`, a scalar. May be traced.

    Returns
    -------
    Array
        A 0-d array.

    Raises
    ------
    ValueError
        If ``misfits`` is not rank 1, or ``increment`` is not a scalar.

    Notes
    -----
    The log-space form is a requirement, not an optimization.
    :math:`\\delta\\Phi_j` is routinely in the hundreds early in a run, where
    naive exponentiation underflows every weight to zero and the ratio
    returns ``nan``. The ratio is unchanged by a common shift of every
    :math:`\\Phi_j`, which is why the max-shifted form computes the same
    quantity rather than an approximation to it.
    """
    misfits = jnp.asarray(misfits)
    if misfits.ndim != 1 or misfits.shape[0] < 1:
        raise ValueError(
            f"effective_sample_size: expected misfits of shape (J,) with "
            f"J at least 1, got shape {misfits.shape}"
        )
    increment = jnp.asarray(increment)
    if increment.ndim != 0:
        raise ValueError(
            f"effective_sample_size: expected a scalar increment, got shape "
            f"{increment.shape}"
        )
    return _ess_from_misfits(misfits, increment)


def repair_failed_members(*, ensemble, predictions, valid):
    """Move failed members to the valid centre.

    ``(J, P), (J, N), (J,) bool -> (J, P), (J, N)``.

    With :math:`m_j \\in \\{0, 1\\}` the validity indicator and
    :math:`\\hat u, \\hat v` the means over the valid members alone,

    .. math::

        u_j \\longmapsto \\hat u + m_j (u_j - \\hat u) ,

    and identically for :math:`v_j`: **failed members are moved to the valid
    centre and valid members are left exactly where they are**, bit for bit.
    The ensemble size is unchanged, so every downstream shape stays static.

    Parameters
    ----------
    ensemble
        The members, a ``(J, P)`` array. Keyword-only.
    predictions
        Their predictions, a ``(J, N)`` array, member-aligned with
        ``ensemble``. Keyword-only.
    valid
        A ``(J,)`` boolean array, ``True`` where the member's prediction is
        finite. At least two entries must be ``True``. Keyword-only.

    Returns
    -------
    tuple of Array
        The repaired ``(J, P)`` ensemble and ``(J, N)`` predictions.

    Raises
    ------
    ValueError
        If ``ensemble`` or ``predictions`` is not rank 2, if ``valid`` is not
        a rank-1 boolean array, if the three disagree on :math:`J`, or if
        fewer than two members are valid — a single member has no anomalies.
        That last check reads a concrete value, so it is skipped on a traced
        ``valid``, as the operator layer's value checks are.

    Notes
    -----
    Three identities hold exactly, and are what make the repair usable inside
    a conditioning step. The all-:math:`J` mean equals :math:`\\hat u`; the
    all-:math:`J` empirical covariance with divisor :math:`J-1` equals the
    valid-member covariance with divisor :math:`J_v-1` scaled by
    :math:`(J_v-1)/(J-1) < 1`, and likewise the cross-covariance, both blocks
    carrying the same factor; and the repaired members' residual is
    :math:`y - \\hat v`, so they rejoin the ensemble rather than being lost.

    The damping is the intended trade. Conditioning with both blocks scaled
    by :math:`c = (J_v-1)/(J-1)` gives the gain at the shorter increment
    :math:`c\\delta`, which for the mean is a slightly smaller step. The
    alternative that rescales every anomaly by :math:`\\sqrt{(J-1)/(J_v-1)}`
    makes the moments exact instead, at the cost of moving the *surviving*
    members outward by a data-dependent factor larger than the inflation
    factors practitioners use — a silent inflation, which is why it is not
    what this function does.

    Both blocks must be repaired with the same mask. Repairing one and not
    the other corrupts the cross-covariance with no exception raised, which
    is why the two are returned together rather than separately.

    Keyword-only because ``ensemble`` and ``predictions`` are arrays whose
    shapes coincide whenever :math:`P = N`, so a positional signature would
    let them be transposed with no error at all.
    """
    ensemble = jnp.asarray(ensemble)
    predictions = jnp.asarray(predictions)
    valid = jnp.asarray(valid)
    if ensemble.ndim != 2:
        raise ValueError(
            f"repair_failed_members: expected ensemble of shape (J, P), got "
            f"shape {ensemble.shape}"
        )
    if predictions.ndim != 2:
        raise ValueError(
            f"repair_failed_members: expected predictions of shape (J, N), got "
            f"shape {predictions.shape}"
        )
    if valid.ndim != 1 or valid.dtype != jnp.bool_:
        raise ValueError(
            f"repair_failed_members: expected valid to be a boolean array of "
            f"shape (J,), got shape {valid.shape} of dtype {valid.dtype}"
        )
    n_members = ensemble.shape[0]
    if predictions.shape[0] != n_members or valid.shape[0] != n_members:
        raise ValueError(
            f"repair_failed_members: ensemble, predictions and valid must agree "
            f"on the member axis, got shapes {ensemble.shape}, "
            f"{predictions.shape} and {valid.shape}"
        )
    n_valid = _concrete_int(jnp.sum(valid))
    if n_valid is not None and n_valid < 2:
        raise ValueError(
            f"repair_failed_members: at least 2 valid members are required, got "
            f"{n_valid}. A single valid member has no anomalies."
        )
    return _repair(ensemble, predictions, valid)


# ---------------------------------------------------------------------------
# private: the traced cores, and the checks the layer shares
# ---------------------------------------------------------------------------


@jax.jit
def _misfits_from_residuals(whitened_residuals: Array) -> Array:
    """The misfit of each row of an already-whitened residual block."""
    return 0.5 * jnp.sum(whitened_residuals**2, axis=-1)


@jax.jit
def _ess_from_misfits(misfits: Array, increment: Array) -> Array:
    """The log-space effective sample size, with no shape checking."""
    log_w = -increment * misfits
    return jnp.exp(2.0 * logsumexp(log_w) - logsumexp(2.0 * log_w))


@jax.jit
def _repair(ensemble: Array, predictions: Array, valid: Array):
    """The repair itself: ``jnp.where``, so a no-failure call is bit-exact."""
    weights = valid.astype(ensemble.dtype)
    denominator = jnp.sum(weights)
    mask = valid[:, None]
    u_hat = jnp.sum(jnp.where(mask, ensemble, 0.0), axis=0) / denominator
    v_hat = jnp.sum(jnp.where(mask, predictions, 0.0), axis=0) / denominator
    return jnp.where(mask, ensemble, u_hat), jnp.where(mask, predictions, v_hat)


def _concrete_int(x) -> int | None:
    """``int(x)`` where that can be read, ``None`` under a trace."""
    try:
        return int(x)
    except (jax.errors.ConcretizationTypeError, TypeError):
        return None


def _check_field_rank(cls_name: str, field_name: str, value, core_ndim: int) -> None:
    """Require an array field of exactly its own rank, with positive sizes.

    Objects in this layer are unbatched; a family is built with
    :func:`jax.vmap` over the pytree, never by storing extra leading axes.
    Enforcing that in the constructor is safe because pytree unflattening
    bypasses it.
    """
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        raise TypeError(
            f"{cls_name}.{field_name}: expected an array of rank {core_ndim}, got "
            f"{type(value).__name__}, which has no shape to check. Pass a JAX "
            f"array — jnp.asarray() on a nested list."
        )
    if ndim != core_ndim:
        raise ValueError(
            f"{cls_name}.{field_name}: expected an array of rank {core_ndim}, got "
            f"shape {value.shape}. Objects in pyeki.eki are unbatched; build a "
            f"family with jax.vmap over the pytree, not with extra leading axes."
        )
    if any(size < 1 for size in value.shape):
        raise ValueError(
            f"{cls_name}.{field_name}: core sizes must be positive, got shape "
            f"{value.shape}."
        )


def _check_not_vmap_family(obj, operation: str) -> None:
    """Refuse an operation on a vmapped family, before any other check."""
    batch = obj.batch_shape
    if batch != ():
        raise ValueError(
            f"{obj!r}.{operation}: a vmapped family cannot be used directly; its "
            f"batch shape is {batch}. Apply it under jax.vmap, member by member."
        )


def _anomalies(x: Array) -> Array:
    """Deviations from the sample mean over the member axis, formed stably.

    Mathematically :math:`x - \\mathbf{1}\\bar x^\\top`, computed by removing
    the first member before averaging, so that identical members give exactly
    zero rather than round-off and the cancellation is governed by the spread
    rather than by the magnitude. The layer's one piece of ensemble
    arithmetic; everything else is dispatched to :mod:`pyeki.gauss` or
    :mod:`pyeki.linalg`.
    """
    shifted = x - x[..., :1, :]
    return shifted - jnp.mean(shifted, axis=-2, keepdims=True)


def _check_finite(where: str, name: str, x) -> None:
    """Require a finite array, only when debug checks are enabled."""
    value_check(
        x,
        lambda arr: bool(jnp.all(jnp.isfinite(arr))),
        f"{where}: {name} must be finite.",
    )
