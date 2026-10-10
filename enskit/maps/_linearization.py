r""":func:`statistical_linearization` and its result, :class:`Linearization`."""
from __future__ import annotations

from typing import NamedTuple

from jax import Array

from ..distribution import Ensemble, Gaussian
from ..linalg import PSDLinOp
from ._pushforward import _input_names
from ._structured import Linear

__all__ = ["statistical_linearization", "Linearization"]


def statistical_linearization(
    dist, *, inputs, output, min_norm: bool = False, unbiased: bool = True
):
    r"""The affine map that best predicts block ``output`` from ``inputs``.

    For :math:`(x, y)` distributed as ``dist``, with :math:`x` the input
    blocks and :math:`y` the output block, the statistical linear regression
    of :math:`y` on :math:`x` is the affine map minimizing the mean squared
    error, and :math:`\Omega` is the covariance of what it leaves over:

    .. math::

        A = C_{yx} C_{xx}^{-1}, \qquad b = m_y - A m_x, \qquad
        \Omega = \operatorname{cov}(y - Ax - b) = C_{yy} - A C_{xy} .

    When :math:`y = f(x)` this is a linearization of :math:`f` with respect
    to the distribution of :math:`x`. If :math:`x \sim \mathcal N(m, C)` and
    :math:`f` is differentiable, Stein's lemma gives
    :math:`C_{yx} = \mathbb E[Df(x)]\,C`, so :math:`A = \mathbb E[Df(x)]`,
    the average Jacobian. For any other distribution of :math:`x`, :math:`A`
    is the least-squares slope and not, in general, an average Jacobian.

    ``dist`` must already hold the output block. To linearize a simulator
    ``f`` about an ensemble, push the ensemble through it first:

    .. code-block:: python

        ens = maps.pushforward(ens, f, inputs="x", output="y")
        fit = maps.statistical_linearization(ens, inputs="x", output="y")

    An :class:`~enskit.distribution.Ensemble` is replaced by its projection,
    :meth:`~enskit.distribution.Ensemble.project` with the given
    ``unbiased``, so :math:`C` and :math:`m` are the particles' moments and
    the fit is the least-squares regression of the particles' outputs on
    their inputs. :math:`A` and :math:`b` do not depend on the divisor;
    :math:`\Omega` is proportional to its reciprocal. A
    :class:`~enskit.distribution.Gaussian` is regressed as it is; this is
    how a ridge regression is computed (see Notes). The computation is
    :meth:`~enskit.distribution.Gaussian.regression`, whose three cases
    apply.

    Parameters
    ----------
    dist : enskit.distribution.Ensemble or enskit.distribution.Gaussian
        The joint distribution of the inputs and the output.
    inputs : str or sequence of str
        Keyword-only. The input blocks, at least one, in the order the
        returned map takes them.
    output : str
        Keyword-only. The output block, which must not be an input.
    min_norm : bool
        Keyword-only. Whether to return the minimum-norm solution when the
        least-squares solution is not unique: for an unweighted ensemble,
        when the inputs' total dimension exceeds ``n_particles - 1``.
    unbiased : bool
        Keyword-only. For an ensemble, the divisor of its projection:
        ``True`` (the default) for :math:`J - 1` or :math:`1 - \sum_j
        w_j^2`, ``False`` for :math:`J` or :math:`1`, the moments of the
        particles' empirical distribution itself. A Gaussian's covariance
        is used as it is (an :class:`~enskit.distribution.EnsembleGaussian`
        keeps the divisor it was projected with), so a Gaussian accepts only
        the default.

    Returns
    -------
    Linearization
        ``map`` is the :class:`Linear` map :math:`x \mapsto Ax + b`, with
        one operator per input; ``residual_cov`` is :math:`\Omega`;
        ``residuals`` is the ``(n_particles, d_y)`` array
        :math:`y_j - Ax_j - b` for an ensemble, and ``None`` for a Gaussian.

    Raises
    ------
    TypeError
        If ``dist`` is not a distribution, a name is not a ``str``, or
        ``min_norm`` or ``unbiased`` is not a ``bool``.
    KeyError
        If an input or the output is not a block.
    ValueError
        If no input is given, a name is repeated, the output is an input,
        ``dist`` is a vmapped family, the solution is not unique and
        ``min_norm`` is ``False``, ``dist`` is a weighted ensemble whose
        inputs' total dimension exceeds ``n_particles - 1``, or
        ``unbiased`` is ``False`` for a Gaussian. Otherwise as
        :meth:`~enskit.distribution.Gaussian.regression`.

    Notes
    -----
    **The residuals depend on the regime.** With :math:`J` particles and
    :math:`d_x` input coordinates:

    - when :math:`d_x \ge J - 1` the fit interpolates the particles: the
      residuals and :math:`\Omega` are zero up to round-off, whatever
      :math:`f` is, and say nothing about its nonlinearity. Every
      interpolating :math:`A` agrees on the span of the input anomalies;
      ``min_norm=True`` sets it to zero off that span;
    - when :math:`d_x < J - 1`, :math:`\Omega` uses the projection's
      divisor :math:`\delta`, :math:`J - 1` by default or :math:`J` with
      ``unbiased=False``. For a linear model with independent errors of
      covariance :math:`\Sigma`,
      :math:`\mathbb E\,\Omega = \frac{J - 1 - d_x}{\delta}\,\Sigma`; scale by
      the inverse factor for an unbiased estimate of :math:`\Sigma`.

    Without regularization, pushing the inputs' projection through ``map``
    and then :class:`AdditiveNoise` of :math:`\Omega` reproduces the
    projection of inputs and output exactly, cross-covariance included.
    :math:`\Omega` is singular in general and has no ``whiten``, so as an
    independent term it can be neither conditioned on nor combined with a
    factor row by :meth:`~enskit.distribution.Gaussian.cov`; move it into the
    shared factor with :meth:`~enskit.distribution.Gaussian.absorb`, or add
    it to a nonsingular noise covariance :math:`R` as
    ``linalg.LowRankUpdate(R, fit.residual_cov.factor())``.

    **Ridge regression** is ``dist = ens.project().add_noise(x=Lam)``: the
    inputs' term :math:`\Lambda` gives
    :math:`A = \hat C_{yx}(\hat C_{xx} + \Lambda)^{-1}` and :math:`\Omega`
    the conditional covariance of the output given the noisy inputs. The
    result then has no ``residuals``; computed against the particles,
    ``ens[output] - fit.map(ens[x])``, they are not distributed with
    covariance :math:`\Omega`, since the coefficients are shrunk.

    References
    ----------
    Lefebvre, T., Bruyninckx, H. & De Schutter, J. (2002). Comment on "A new
    method for the nonlinear transformation of means and covariances in
    filters and estimators". *IEEE Transactions on Automatic Control*, 47(8),
    1406–1409.

    Stein, C. M. (1981). Estimation of the mean of a multivariate normal
    distribution. *The Annals of Statistics*, 9(6), 1135–1151.
    """
    where = "statistical_linearization"
    if not isinstance(dist, (Ensemble, Gaussian)):
        raise TypeError(
            f"{where}: dist must be an Ensemble or a Gaussian, got {type(dist).__name__}"
        )
    if dist.batch_shape != ():
        raise ValueError(
            f"{where}: {dist!r} is a vmapped family with batch shape "
            f"{dist.batch_shape}; apply statistical_linearization under jax.vmap, "
            f"one member at a time."
        )
    if not isinstance(output, str):
        raise TypeError(f"{where}: output must be a str, got {type(output).__name__}")
    if output not in dist.names:
        raise KeyError(
            f"{where}: output {output!r} is not a block; the blocks are {dist.names}"
        )
    if inputs is None:
        raise ValueError(f"{where}: at least one input block is required")
    inputs = _input_names(where, dist, inputs)
    if output in inputs:
        raise ValueError(f"{where}: block {output!r} is both an input and the output")
    if not isinstance(min_norm, bool):
        raise TypeError(
            f"{where}: min_norm must be a Python bool, got "
            f"{type(min_norm).__module__}.{type(min_norm).__name__}"
        )
    if not isinstance(unbiased, bool):
        raise TypeError(
            f"{where}: unbiased must be a Python bool, got "
            f"{type(unbiased).__module__}.{type(unbiased).__name__}"
        )
    ensemble = dist if isinstance(dist, Ensemble) else None
    if ensemble is None and not unbiased:
        raise ValueError(
            f"{where}: unbiased=False selects the divisor of an ensemble's "
            f"projection; {dist!r} is a Gaussian, whose covariance is used as it "
            f"is"
        )
    if ensemble is not None and ensemble.is_weighted:
        N, J = sum(ensemble.dims[n] for n in inputs), ensemble.n_particles
        if N > J - 1:
            raise ValueError(
                f"{where}: the inputs have N = {N} coordinates and the ensemble "
                f"J = {J} weighted particles; a weighted projection has rank at "
                f"most J - 1 = {J - 1}, so its least-squares coefficients are not "
                f"unique and its minimum-norm solution is not supported. Resample "
                f"first (distribution.resample), or pass "
                f"ens.project().add_noise(...) for a ridge regression."
            )
    g = ensemble.project(unbiased=unbiased) if ensemble is not None else dist
    reg = g.regression(output, given=inputs, min_norm=min_norm)
    coefs = reg.coefficients
    op = coefs[inputs[0]] if len(inputs) == 1 else {n: coefs[n] for n in inputs}
    lin = Linear(op, reg.intercept)
    residuals = None
    if ensemble is not None:
        residuals = ensemble[output] - lin(*(ensemble[n] for n in inputs))
    return Linearization(map=lin, residual_cov=reg.residual_cov, residuals=residuals)


class Linearization(NamedTuple):
    r"""An affine fit of one block on others, from :func:`statistical_linearization`.

    .. math::

        y \approx Ax + b + e, \qquad \operatorname{cov}(e) = \Omega .

    A named tuple, and so a pytree. ``map`` is the :class:`Linear` map
    :math:`x \mapsto Ax + b`, whose ``op`` is :math:`A` (a mapping of
    operators, one per input block, when there are several) and whose
    ``shift`` is :math:`b`. ``residual_cov`` is :math:`\Omega`, a
    :class:`~enskit.linalg.PSDLinOp`. ``residuals`` is the
    ``(n_particles, d_y)`` array :math:`y_j - Ax_j - b` when the fit was of
    an :class:`~enskit.distribution.Ensemble`, and ``None`` otherwise.
    """

    map: Linear
    residual_cov: PSDLinOp
    residuals: Array | None
