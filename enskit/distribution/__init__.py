r"""Probability distributions over named vector blocks.

This module provides the two distributions every ensemble Kalman method is
built from, the crossings between them, and the exact operations on them.

=============================== ===============================================
object                          represents
=============================== ===============================================
:class:`Ensemble`               an empirical distribution: ``n_particles``
                                points, optionally weighted, over named blocks
:class:`Gaussian`               a joint Gaussian over named blocks, held as a
                                shared factor plus an independent term per
                                block
:class:`EnsembleGaussian`       a Gaussian whose latent coordinates are an
                                ensemble's particles
:class:`Regression`             the regression of one block on others in a
                                Gaussian, from :meth:`Gaussian.regression`
:class:`ConditionalMap`         protocol: a pointwise map carrying samples of a
                                joint to samples of a conditional
:class:`MatheronMap`            the exact conditional of a :class:`Gaussian`,
                                as the affine map :math:`x \mapsto x + K(y^* - y)`
:class:`SquareRootMap`          the square-root update of an
                                :class:`EnsembleGaussian`'s particle set
:func:`reweight`                multiply an ensemble's weights by factors, in
                                log space
:func:`effective_sample_size`   :math:`1 / \sum_j w_j^2` for normalized weights
:func:`resample`                draw an unweighted ensemble from a weighted one
:func:`exact_moment_ensemble`   particles whose sample moments equal a
                                Gaussian's exactly
=============================== ===============================================

Conventions shared by everything in the module:

- **Blocks are named vectors.** A block is a ``str`` name and a dimension
  :math:`d_b \ge 1`. A block's value is a ``(d_b,)`` array; an ensemble
  stores block :math:`b` as a ``(n_particles, d_b)`` array, one particle per
  row. Names are ordered: the order in which blocks were given is the order
  every method reports them in.
- **The particle axis is a batch axis** in the sense of
  :mod:`enskit.linalg`, so ``op.matvec(ensemble["x"])`` applies an operator
  to every particle. **Factors are column-wise**: a factor of a
  :math:`d`-dimensional covariance is a ``(d, k)`` operator, so particles
  become a factor through a transpose and the divisor's square root.
- **Distributions are unbatched.** A family of distributions is expressed
  with :func:`jax.vmap`; a pytree rebuilt with stacked leaves is a vmapped
  family that refuses every method until applied under :func:`jax.vmap`.
  The one argument that takes leading batch axes is the value of
  :meth:`Gaussian.log_density`.
- **Block values may be passed as keywords.** Every argument that is a set of
  block values or covariances accepts a positional-only mapping, keywords,
  or both: ``g.condition(y=y_obs)`` is ``g.condition({"y": y_obs})``. The
  mapping form is required for a name that is not a Python identifier or
  that collides with one of the function's own parameters, such as ``key``.
  Where the order of the given blocks matters to a result, it is the
  distribution's block order, not the order the arguments were written in.
- **Samples and particles.** A *sample* is a draw from any distribution; a
  *particle* is an element of an :class:`Ensemble`. An ensemble's particles
  are treated as samples wherever a function asks for samples.
- **Anomalies are raw deviations from the mean.** A covariance computed from
  particles divides by :math:`J - 1` (unweighted) or
  :math:`1 - \sum_j w_j^2` (weighted) by default, which treats them as
  samples. ``unbiased=False``, accepted by :meth:`Ensemble.cov` and
  :meth:`Ensemble.project`, divides by :math:`J` or :math:`1` instead,
  giving the moments of the empirical distribution itself. An
  :class:`EnsembleGaussian` stores its divisor, and every operation that
  reads particles out of it uses that divisor.
- **Randomness enters through an explicit typed key**
  (:func:`jax.random.key`), consumed whole. Nothing stores or advances a key,
  and every draw is pinned.
- **Scalars are returned as 0-d JAX arrays**, never Python floats.

Every conditioning path computes one whitened factor
:math:`S = (WF_c)^\top`, one :class:`~enskit.linalg.IdentityPlusGram` of
it, and so one thin singular value decomposition; see
:meth:`Gaussian.condition`.

Notes
-----
The behavior of this module is specified by the "Distribution contract" page
of the documentation, which is normative. All classes are frozen pytrees:
construction validates, unflattening bypasses the constructor, objects
compare by identity and are never ``static_argnums``. Whether a block has a
factor row or an independent term, and whether an ensemble is weighted, is
part of the tree structure, so a loop carry must keep one structure.

Value preconditions (finiteness, centering, rank, concentrated weights) are
checked only in debug mode (:func:`enskit.linalg.debug_checks`), on
concrete arrays. Every conditioning path then runs two checks: a *whitening
check* right after whitening, naming the given block whose independent term
is the likely cause, and a *result check* on what it returns.
"""
from ._ensemble import Ensemble
from ._gaussian import EnsembleGaussian, Gaussian
from ._maps import ConditionalMap, MatheronMap, SquareRootMap
from ._regression import Regression
from ._weights import effective_sample_size, exact_moment_ensemble, resample, reweight

__all__ = [
    "Ensemble",
    "Gaussian",
    "EnsembleGaussian",
    "Regression",
    "ConditionalMap",
    "MatheronMap",
    "SquareRootMap",
    "reweight",
    "effective_sample_size",
    "resample",
    "exact_moment_ensemble",
]

# The modules above are private, so the public names report the package they
# are imported from, in tracebacks, ``type()`` and pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
