r"""The update protocols, the default approximation, and the one-call update."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

from jax import Array

from ..distribution import Ensemble, Gaussian
from ..linalg import PSDLinOp
from . import _common as c

__all__ = ["update", "gaussian_approximation", "UpdateRule", "ParticleUpdate"]


def update(
    ensemble: Ensemble,
    given: Mapping[str, Array] | None = None,
    /,
    *,
    update_rule: UpdateRule,
    noise: Mapping[str, PSDLinOp] | None = None,
    approximation: Callable[[Ensemble, dict], Gaussian] | None = None,
    key=None,
    **given_values: Array,
) -> Ensemble:
    r"""One ensemble Kalman update: approximate, build, call.

    Returns particles approximating the conditional distribution of every
    non-given block of ``ensemble`` at the given values :math:`y^*`. It is

    .. code-block:: python

        update_rule.build(ensemble, approximation(ensemble, noise), names)(
            values, key=key)

    with ``names`` the given blocks in the ensemble's block order. With
    ``noise``, each given value :math:`y^*_c` is understood as a realization
    of :math:`x_c + e_c` with :math:`e_c \sim \mathcal N(0, R_c)`, while
    ``ensemble[c]`` holds the noise-free :math:`x_c`; without it, the given
    blocks are conditioned on exactly.

    Parameters
    ----------
    ensemble : Ensemble
        Unweighted, finite, holding every given block and at least one other.
    given : Mapping[str, Array], optional
        Positional-only. :math:`y^*` for each given block, exactly ``(d_c,)``.
    update_rule : UpdateRule
        Keyword-only and required: :class:`SymmetricSquareRoot`,
        :class:`Matheron`, or any object with a ``build`` method. There is no
        default; which rule to use is a modeling choice.
    noise : Mapping[str, PSDLinOp], optional
        Keyword-only. The known noise covariance :math:`R_c` of given blocks,
        used only through ``whiten``. Without it, only
        :class:`SymmetricSquareRoot` applies, with more particles than the
        given blocks' total dimension.
    approximation : callable, optional
        Keyword-only. ``(ensemble, noise) -> Gaussian``, with ``noise`` a
        ``dict``; :func:`gaussian_approximation` by default. The extension
        point for other covariance estimators.
    key : jax.random key, optional
        Keyword-only. Passed whole to the built update; required by
        :class:`Matheron`.
    **given_values : Array
        The values as keywords: ``update(ens, g=y, update_rule=..., noise=...)``.
        A block whose name is one of this function's parameters must go in
        ``given``.

    Returns
    -------
    Ensemble
        Unweighted, with the ensemble's particle count and dtype, over the
        non-given blocks in the approximation's block order.

    Raises
    ------
    ValueError
        If the ensemble is weighted (resample first, with
        :func:`enskit.distribution.resample`), no value is given, no target
        remains, ``noise`` names a block that is not given, or the rule's
        result is not an unweighted ensemble of the right count over the
        targets. In debug mode, also if a particle is not finite.
    KeyError
        If a value or a noise covariance names a block that is not in the
        ensemble.
    TypeError
        If an argument has the wrong type, a block is given twice, the
        approximation does not return a :class:`~enskit.distribution.Gaussian`,
        or the rule's result has another dtype.

    Examples
    --------
    >>> post = update(ens, g=y, noise={"g": R}, update_rule=Matheron(), key=key)

    The same, in three stages, reusing the build for two values:

    >>> approx = gaussian_approximation(ens, {"g": R})
    >>> built = Matheron().build(ens, approx, ("g",))
    >>> post_a, post_b = built(g=y_a, key=k1), built(g=y_b, key=k2)
    """
    where = "kalman.update"
    c.guard(ensemble, where)
    c.check_ensemble(where, ensemble)
    values = c.merge_blocks(where, given, given_values)
    if not values:
        raise ValueError(f"{where}: at least one given value is required")
    for name in values:
        if name not in ensemble.names:
            raise KeyError(
                f"{where}: no block named {name!r}; the blocks are {ensemble.names}"
            )
    if not callable(getattr(update_rule, "build", None)):
        raise TypeError(
            f"{where}: update_rule must have a build method (an UpdateRule), got "
            f"{type(update_rule).__name__}"
        )
    noise = _check_noise(where, ensemble, values, noise)
    build_approximation = (
        gaussian_approximation if approximation is None else approximation
    )
    if not callable(build_approximation):
        raise TypeError(
            f"{where}: approximation must be a callable (ensemble, noise) -> "
            f"Gaussian, got {type(approximation).__name__}"
        )
    c.check_key(where, key, required=False)
    c.check_unweighted(where, ensemble)
    names = tuple(n for n in ensemble.names if n in values)
    targets = tuple(n for n in ensemble.names if n not in values)
    if not targets:
        raise ValueError(
            f"{where}: every block is given; at least one target block must remain"
        )
    c.check_particles_finite(where, ensemble)
    approx = build_approximation(ensemble, noise)
    if not isinstance(approx, Gaussian):
        raise TypeError(
            f"{where}: the approximation must return a Gaussian, got "
            f"{type(approx).__name__}"
        )
    result = update_rule.build(ensemble, approx, names)(values, key=key)
    _check_result(where, update_rule, ensemble, approx, names, result)
    return result


def gaussian_approximation(
    ensemble: Ensemble, noise: Mapping[str, PSDLinOp] | None = None
):
    r"""The joint Gaussian an update conditions: moment match plus known noise.

    .. math::

        \hat p(x, y) = \mathcal N\big(\bar z,\
            \hat C + \operatorname{blockdiag}(0, R)\big),

    the Gaussian with the particles' sample mean :math:`\bar z` and sample
    covariance :math:`\hat C` jointly over every block
    (:meth:`~enskit.distribution.Ensemble.project`), with the known noise
    :math:`R_c` = ``noise[c]`` added to each named block *as a covariance*
    (:meth:`~enskit.distribution.Gaussian.add_noise`), never as sampled
    perturbations, which would estimate the same joint with more variance.
    It is the default of every ``approximation=`` hook, and the starting
    point for a modified approximation.

    Parameters
    ----------
    ensemble : Ensemble
        The particles. Every particle of positive weight must be finite.
    noise : Mapping[str, PSDLinOp], optional
        Known noise covariances, by block name; each of its block's side.

    Returns
    -------
    ~enskit.distribution.EnsembleGaussian or ~enskit.distribution.Gaussian
        For an unweighted ensemble, an
        :class:`~enskit.distribution.EnsembleGaussian` aligned with the
        particles: latent coordinate :math:`j` is particle :math:`j`. Its
        covariance always uses the unbiased divisor :math:`J - 1`. A
        weighted ensemble gives a plain
        :class:`~enskit.distribution.Gaussian`, which no shipped rule accepts.

    Raises
    ------
    TypeError
        If ``ensemble`` is not an :class:`~enskit.distribution.Ensemble`,
        ``noise`` is not a mapping, or a covariance is not a
        :class:`~enskit.linalg.PSDLinOp`.
    KeyError
        If ``noise`` names a block that is not in the ensemble.
    ValueError
        If a covariance has the wrong side. In debug mode, the checks of
        :meth:`~enskit.distribution.Ensemble.project`.
    """
    where = "kalman.gaussian_approximation"
    c.guard(ensemble, where)
    c.check_ensemble(where, ensemble)
    if noise is not None and not isinstance(noise, Mapping):
        raise TypeError(
            f"{where}: noise must be a mapping from block name to PSDLinOp, got "
            f"{type(noise).__name__}"
        )
    noise = dict(noise or {})
    for name in noise:
        if name not in ensemble.names:
            raise KeyError(
                f"{where}: noise names {name!r}, which is not a block; the blocks are "
                f"{ensemble.names}"
            )
    approx = ensemble.project()
    return approx.add_noise(noise) if noise else approx


@runtime_checkable
class UpdateRule(Protocol):
    r"""Builds a :class:`ParticleUpdate`; writing one is how a new update is added.

    An update rule is a factory with one method, ``build``. It receives the
    particles, a joint Gaussian approximation of them, and the *names* of the
    given blocks, but never their values, and returns a callable that takes
    the values. Everything that does not depend on the values is computed in
    ``build``, once. A rule is accepted wherever a shipped rule is.

    The contract between caller and rule:

    - ``particles`` is an unweighted :class:`~enskit.distribution.Ensemble`
      containing every block of the approximation.
    - ``approximation`` is a :class:`~enskit.distribution.Gaussian` over the
      same blocks, of the same dimensions and dtype. Its given blocks carry
      their known noise as independent terms.
    - Every block of the approximation not named in ``given`` is a target.
    - **If** the approximation is an
      :class:`~enskit.distribution.EnsembleGaussian` with the particles'
      count, it is *aligned* with these particles, as
      :func:`gaussian_approximation` guarantees: its realized particles,
      without independent terms, are the particles,

      .. math::

          m_b + \sqrt{\delta}\, F_b e_j = x^{(b)}_j, \qquad j = 1, \dots, J ,

      with :math:`\delta` its
      :attr:`~enskit.distribution.EnsembleGaussian.divisor`.

      A rule may rely on that for a faster path. A modified approximation is
      generally a plain :class:`~enskit.distribution.Gaussian`, and a rule
      that needs alignment refuses it.

    A rule may refuse an approximation it does not support, at build, with
    ``ValueError``. :func:`enskit.testing.check_update_rule` checks a rule.

    Examples
    --------
    The deterministic EnKF (Sakov & Oke, 2008) in a few lines: full gain on
    the mean, half gain on the anomalies.

    >>> class DEnKF:
    ...     def build(self, particles, approximation, given):
    ...         cmap = approximation.conditional_map(given)
    ...         halfway = particles.assign(
    ...             {c: particles.mean(c) + 0.5 * particles.anomalies(c)
    ...              for c in given})
    ...         def call(values=None, /, *, key=None, **block_values):
    ...             return cmap(halfway, values, **block_values).marginal(
    ...                 *cmap.targets)
    ...         return call

    References
    ----------
    Sakov, P. & Oke, P. R. (2008). A deterministic formulation of the
    ensemble Kalman filter: an alternative to ensemble square root filters.
    *Tellus A*, 60(2), 361–371.
    """

    def build(
        self, particles: Ensemble, approximation: Gaussian, given: tuple[str, ...]
    ) -> ParticleUpdate:
        """Build the update of ``particles`` conditioned on the blocks named ``given``."""
        ...


@runtime_checkable
class ParticleUpdate(Protocol):
    r"""A built update: called with the given values, returns the updated particles.

    Bound to the particles, approximation and given names it was built from.
    Calling it again with other values reuses everything that does not depend
    on the values. An implementation must satisfy:

    1. The values are a positional-only mapping, keywords, or both, with
       exactly one ``(d_c,)`` array for each given block; a missing, extra or
       misshapen value raises.
    2. The result is an unweighted :class:`~enskit.distribution.Ensemble`
       with the particles' count, over the target blocks in the
       approximation's block order, of the particles' dtype.
    3. One built update called at several values gives what a fresh build
       gives for each.
    4. A key is consumed whole, and the result is deterministic given it; an
       update that needs a key raises ``ValueError`` without one.
    5. Building and calling run under :func:`jax.jit` and :func:`jax.vmap`.
    6. With particles whose sample moments equal a linear-Gaussian joint's,
       the result's sample mean equals the exact conditional mean (in
       expectation over the key, for a stochastic update).
    """

    def __call__(
        self, values: Mapping[str, Array] | None = None, /, *, key=None, **block_values
    ) -> Ensemble:
        """Return the updated particles over the target blocks."""
        ...


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _check_noise(where: str, ensemble: Ensemble, values: dict, noise) -> dict:
    """``noise`` as a dict naming given blocks only."""
    if noise is None:
        return {}
    if not isinstance(noise, Mapping):
        raise TypeError(
            f"{where}: noise must be a mapping from given block to PSDLinOp, got "
            f"{type(noise).__name__}"
        )
    for name in noise:
        if name not in ensemble.names:
            raise KeyError(
                f"{where}: noise names {name!r}, which is not a block; the blocks are "
                f"{ensemble.names}"
            )
        if name not in values:
            raise ValueError(
                f"{where}: noise names {name!r}, which is not a given block. Noise "
                f"is the known error of a given value; to add a covariance to a "
                f"target, use add_noise in a custom approximation."
            )
        if not isinstance(noise[name], PSDLinOp):
            raise TypeError(
                f"{where}: the noise of block {name!r} must be a PSDLinOp, got "
                f"{type(noise[name]).__name__}"
            )
    return dict(noise)


def _check_result(where, rule, ensemble, approx, names, result) -> None:
    """The rule's result: an unweighted Ensemble over the targets, of the right
    count and dtype. Static, so it always runs."""
    who = f"{where}: the update rule {type(rule).__name__}"
    if not isinstance(result, Ensemble):
        raise TypeError(f"{who} returned {type(result).__name__}, not an Ensemble")
    targets = tuple(n for n in approx.names if n not in names)
    if result.is_weighted:
        raise ValueError(f"{who} returned a weighted ensemble")
    if result.n_particles != ensemble.n_particles:
        raise ValueError(
            f"{who} returned {result.n_particles} particles; the ensemble has "
            f"{ensemble.n_particles}"
        )
    if result.names != targets:
        raise ValueError(
            f"{who} returned blocks {result.names}; the target blocks, in the "
            f"approximation's order, are {targets}"
        )
    dtype = ensemble[ensemble.names[0]].dtype
    for name in targets:
        if result.dims[name] != ensemble.dims[name]:
            raise ValueError(
                f"{who} returned block {name!r} of dimension {result.dims[name]}; "
                f"the ensemble's has {ensemble.dims[name]}"
            )
        got = result[name].dtype
        if got != dtype:
            raise TypeError(
                f"{who} returned block {name!r} of dtype {got}; the ensemble has "
                f"{dtype}. A rule that changes the dtype would change it for every "
                f"later update."
            )
