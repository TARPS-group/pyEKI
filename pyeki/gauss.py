"""Joint Gaussian distributions and the conditioning built on them.

The layer represents a joint Gaussian over two blocks, :math:`u` and
:math:`v`, and conditions it on a noisy observation of the second. Every
conditioning path routes through one algorithm, the whitened-SVD kernel
described below, so nothing here forms a matrix of either block's dimension.

============================ ==================================================
object                       represents
============================ ==================================================
:class:`Gaussian`            one Gaussian distribution: a mean vector and a
                             :class:`~pyeki.linalg.PSDLinOp` covariance
:class:`GaussianJoint`       a joint Gaussian over the two blocks, held as a
                             mean pair and a joint factor; the home of every
                             conditioning identity
:class:`EmpiricalJoint`      :math:`J` paired samples, and the two updates
                             that carry them to updated samples
:func:`gain_weights`         the Kalman-gain weights for a whitened residual
:func:`sqrt_transform`       the deterministic square-root update transform
============================ ==================================================

The two joint objects divide the work: :class:`GaussianJoint` owns the
mathematics, returning distributions or transporting realizations handed to
it, and :class:`EmpiricalJoint` owns the samples, returning updated samples
aligned with the ones it holds. A sample set becomes a joint Gaussian through
:meth:`EmpiricalJoint.to_gaussian_joint`, which fits a Gaussian to the
samples' moments; that call is the one place the fit happens, and it is
written out at the call site rather than hidden inside a conditioning method.

Conventions shared by everything in the module:

- **Samples are stored row-wise**: a sample block is a ``(J, dim)`` array,
  one draw per row. That is what :func:`jax.vmap` produces and what the
  operator layer's batch contract treats as a batch of vectors, so sample
  blocks flow between the two layers with no transposes.
- **Factors are stored column-wise**, as the operator layer's ``factor``
  returns them: a factor of an ``n``-dimensional covariance is an ``(n, k)``
  operator. Converting between the two conventions therefore carries a
  transpose *and* the divisor's square root; :class:`GaussianJoint` states
  the conversion where it uses it.
- **Anomalies are raw deviations from the sample mean**, with no
  normalization folded in, and **empirical covariances use the divisor**
  :math:`J - 1`.
- **Vectors passed to methods carry no batch axes**: an observation is a
  ``(N,)`` array and a mean a ``(dim,)`` array, exactly. A family of distributions or
  updates is expressed with :func:`jax.vmap` over the pytree, never with
  stored batch axes. The two conditioning primitives are the exception —
  they are array-level and follow the operator layer's batch contract — as
  is the evaluation point of :meth:`Gaussian.log_density`.
- **Noise covariances are** :class:`~pyeki.linalg.PSDLinOp` **s**, used only
  through ``whiten``, so a noise operator with no factorization at all
  drives every update.

All three classes are frozen-dataclass pytrees whose fields are their whole
state, exactly like operators: they compare by identity, they are never
valid ``static_argnums``, and a pytree reconstruction with batched leaves
produces a *vmapped family*, which reports its ``batch_shape`` and refuses
every method and array-computing property until it is applied under
:func:`jax.vmap`; the static size properties and ``repr`` still answer.

Notes
-----
The behaviour of this module is specified by the "Joint Gaussian contract"
page of the documentation, which is normative; the user guide's
"Conditioning" page explains when to reach for each piece.

The conditioning kernel. A joint Gaussian is held as a **joint factor**: a
single factor :math:`F` of the joint covariance, cut into the two row blocks
:math:`F_u \\in \\mathbb{R}^{P \\times k}` and
:math:`F_v \\in \\mathbb{R}^{N \\times k}` that drive the two blocks from one
shared latent vector,

.. math::

    \\begin{pmatrix} u \\\\ v \\end{pmatrix}
    = \\begin{pmatrix} \\bar u \\\\ \\bar v \\end{pmatrix}
    + \\begin{pmatrix} F_u \\\\ F_v \\end{pmatrix} \\xi,
    \\qquad \\xi \\sim \\mathcal{N}(0, I_k),

so that :math:`C_{uu} = F_u F_u^\\top`, :math:`C_{uv} = F_u F_v^\\top` and
:math:`C_{vv} = F_v F_v^\\top`. With :math:`W` a whitener of the noise
covariance :math:`R`, write

.. math::

    S = \\bigl(W F_v\\bigr)^\\top \\in \\mathbb{R}^{k \\times N},
    \\qquad S = U \\Sigma V^\\top \\ \\text{(thin SVD)}.

The Kalman gain :math:`K = C_{uv}(C_{vv} + R)^{-1}` applied to a residual
:math:`r` is then a combination of :math:`F_u`'s own columns,

.. math::

    K r = F_u\\, w, \\qquad
    w = U \\operatorname{diag}\\!\\Bigl(\\frac{\\sigma_i}{1+\\sigma_i^2}\\Bigr)
        V^\\top (W r),

and the posterior covariance is the same factor multiplied on the right by
:math:`T = (I_k + S S^\\top)^{-1/2}`, built from the same decomposition:

.. math::

    C_{\\text{post}} = C_{uu} - K C_{vu}
    = \\bigl(F_u T\\bigr)\\bigl(F_u T\\bigr)^\\top .

Neither :math:`S^\\top S` nor :math:`S S^\\top` is ever formed: their
condition numbers are the squares of :math:`S`'s. Each class method
computes one SVD, uses it for both pieces, and discards it; each public
primitive computes its own.
"""
from __future__ import annotations

import math
from dataclasses import field

import jax
import jax.numpy as jnp
from jax import Array

from .linalg import Dense, LinOp, PSDLinOp, PSDLowRank, dense_matvec, value_check
from .linalg.base import _broadcast_batch, _pytree_dataclass

__all__ = [
    "EmpiricalJoint",
    "Gaussian",
    "GaussianJoint",
    "gain_weights",
    "sqrt_transform",
]


# ---------------------------------------------------------------------------
# the conditioning primitives
# ---------------------------------------------------------------------------


def gain_weights(s: Array, b: Array) -> Array:
    """Sample weights for a whitened residual: the shared conditioning core.

    A pure matrix function of its arguments — no divisor, no whitening and no
    randomness folded in. For the thin SVD :math:`s = U\\Sigma V^\\top`,

    .. math::

        \\texttt{gain\\_weights}(s, b)
        = U \\operatorname{diag}\\!\\Bigl(\\frac{\\sigma_i}{1+\\sigma_i^2}\\Bigr)
          V^\\top b
        = s\\,(s^\\top s + I_N)^{-1} b ,

    the second form showing that the result is a function of ``s`` alone,
    invariant to the SVD's sign and degenerate-rotation freedom.

    In conditioning, ``s`` is the whitened factor
    :math:`S = (W F_v)^\\top` of the observed block and ``b`` a whitened
    residual :math:`W r`, and the returned weights give the gain applied to
    that residual as a combination of the other block's factor columns,
    :math:`K r = F_u w`. The multipliers are bounded by
    :math:`\\sigma/(1+\\sigma^2) \\le 1/2` for every :math:`\\sigma \\ge 0`,
    so the gain cannot blow up however collapsed or ill-conditioned
    :math:`s` becomes, and there is no regularization parameter to tune.

    Parameters
    ----------
    s
        Array of shape ``(k, N)``, exactly 2-D, both sizes at least 1. It
        plays the operator's role and carries no batch axes; a family of
        local analyses is a :func:`jax.vmap` over this function.
    b
        Array of shape ``(..., N)`` — whitened residuals along the trailing
        axis, any number of leading batch axes, carried through.

    Returns
    -------
    Array
        Shape ``(..., k)``, the batch axes of ``b`` preserved.

    Raises
    ------
    ValueError
        If ``s`` is not 2-D with positive sizes, or ``b``'s trailing axis is
        not ``N``. In debug mode, also if either is not finite.

    Notes
    -----
    One SVD per call: batch the residuals of an update into a single call
    rather than looping, since the :math:`J` per-sample residuals of a
    stochastic update are one ``(J, N)`` operand.

    Callers own the semantics of ``s`` and ``b``. The function cannot check
    that they are whitened and read off a single joint factor as
    conditioning requires, which is why the methods of
    :class:`GaussianJoint` — where those conventions are enforced — are the
    default interface and this is the escape hatch.

    Differentiable wherever the singular values of ``s`` are distinct and
    nonzero. At exactly repeated or exactly zero singular values — an
    exactly collapsed ``s``, or the zero-padded columns a masked local
    analysis may produce — the SVD's gradient is ``nan`` even though this
    function is smooth there, equalling the rational form above. The
    float-generic degeneracy of mean-centering (:math:`\\sigma_{\\min} \\sim
    10^{-16}` when :math:`N \\ge k`) is not an exact tie and differentiates
    finitely.
    """
    s = jnp.asarray(s)
    if s.ndim != 2 or any(size < 1 for size in s.shape):
        raise ValueError(
            f"gain_weights: expected s of shape (k, N), exactly 2-D with both "
            f"sizes at least 1, got shape {s.shape}"
        )
    b = _check_batched_operand("gain_weights", "b", b, s.shape[1])
    _check_finite("gain_weights", "s", s)
    _check_finite("gain_weights", "b", b)
    U, sigma, Vt = _thin_svd(s)
    return _weights_from_svd(U, sigma, Vt, b)


def sqrt_transform(s: Array) -> Array:
    """The deterministic square-root update transform: the shared conditioning core.

    A pure matrix function of its argument. For the thin SVD
    :math:`s = U\\Sigma V^\\top` with :math:`\\rho = \\min(k, N)`,

    .. math::

        \\texttt{sqrt\\_transform}(s) = (I_k + s s^\\top)^{-1/2}
        = I_k + U\\bigl((I_\\rho + \\Sigma^2)^{-1/2} - I_\\rho\\bigr)U^\\top ,

    which is symmetric, and exact at every rank: the second form is how it
    is computed, and it is what this function returns for any correct thin
    SVD, elementwise.

    In conditioning, ``s`` is the whitened factor :math:`S = (W F_v)^\\top`
    of the observed block, and multiplying the other block's factor on the
    right by the result gives the posterior covariance exactly,

    .. math::

        \\bigl(F_u T\\bigr)\\bigl(F_u T\\bigr)^\\top = C_{uu} - K C_{vu} ,

    an identity in exact arithmetic rather than an approximation. Neither
    :math:`s s^\\top` nor :math:`s^\\top s` is formed.

    Parameters
    ----------
    s
        Array of shape ``(k, N)``, exactly 2-D, both sizes at least 1. No
        batch axes, and no centering requirement: on general ``s`` the result
        is still :math:`(I + ss^\\top)^{-1/2}`.

    Returns
    -------
    Array
        Shape ``(k, k)``, symmetric to round-off.

    Raises
    ------
    ValueError
        If ``s`` is not 2-D with positive sizes. In debug mode, also if it
        is not finite.

    Notes
    -----
    :math:`T\\mathbf{1} = \\mathbf{1}` — so a centred factor stays centred
    and the posterior mean is not silently shifted — follows from
    :math:`s^\\top \\mathbf{1} = 0`, which holds exactly when the factor the
    whitening came from is centred. A factor read off a sample set is, which
    is what makes :meth:`EmpiricalJoint.transform_update` a sample-to-sample
    map. On general ``s``, :math:`T\\mathbf{1}` is whatever that matrix makes
    it.

    Differentiability carries the caveat documented on
    :func:`gain_weights`; restoring gradients everywhere would need a
    Fréchet derivative of :math:`A \\mapsto A^{-1/2}`, materially more work
    than that function's rational form, and no conditioning path in this
    layer requires it.
    """
    s = jnp.asarray(s)
    if s.ndim != 2 or any(size < 1 for size in s.shape):
        raise ValueError(
            f"sqrt_transform: expected s of shape (k, N), exactly 2-D with both "
            f"sizes at least 1, got shape {s.shape}"
        )
    _check_finite("sqrt_transform", "s", s)
    U, sigma, _ = _thin_svd(s)
    return _transform_from_svd(U, sigma, s.shape[0])


# ---------------------------------------------------------------------------
# a single Gaussian distribution
# ---------------------------------------------------------------------------


@_pytree_dataclass
class Gaussian:
    """A Gaussian distribution :math:`\\mathcal{N}(m, C)`.

    The prior a caller supplies, and the posterior
    :meth:`GaussianJoint.condition` returns.

    Each method requires specific operations of the covariance, and an
    unsupported one raises the operator layer's
    :class:`~pyeki.linalg.UnsupportedOpError` from the inner call. ``cov`` is
    a public field, so gate on ``gaussian.cov.supports("factor")`` exactly as
    you would on an operator.

    Parameters
    ----------
    mean
        The mean, a ``(n,)`` array.
    cov
        The covariance, a :class:`~pyeki.linalg.PSDLinOp` of side ``n``.

    Raises
    ------
    ValueError
        If ``mean`` is not rank 1, if the two disagree on ``n``, or if
        ``cov`` is a vmapped family. In debug mode, also if ``mean`` is not
        finite.
    TypeError
        If ``cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
    """

    mean: Array
    cov: PSDLinOp

    def __post_init__(self) -> None:
        _check_field_rank("Gaussian", "mean", self.mean, 1)
        if not isinstance(self.cov, PSDLinOp):
            raise TypeError(
                f"Gaussian.cov: must be a pyeki.linalg.PSDLinOp, got "
                f"{type(self.cov).__name__}"
            )
        if self.cov.batch_shape != ():
            raise ValueError(
                f"Gaussian.cov: {self.cov!r} is a vmapped family; construct a "
                f"family of distributions with jax.vmap over the constructor, "
                f"not from a family covariance."
            )
        if self.cov.shape[0] != self.mean.shape[-1]:
            raise ValueError(
                f"Gaussian: mean of length {self.mean.shape[-1]} disagrees with "
                f"{self.cov!r} of side {self.cov.shape[0]}"
            )
        _check_finite("Gaussian", "mean", self.mean)

    @classmethod
    def from_samples(cls, samples) -> Gaussian:
        """The Gaussian fit to a set of samples: their empirical moments.

        The one-block counterpart of :meth:`GaussianJoint.from_samples`,
        which fits a joint to two row-aligned blocks. Use it to read a block
        of samples' moments as a distribution — per-coordinate variances
        through ``cov.diag()``, fresh draws through :meth:`sample`.

        With :math:`\\mathsf{X}` the ``(J, n)`` sample matrix and
        :math:`A = \\mathsf{X} - \\mathbf{1}_J\\bar x^\\top` its anomalies,

        .. math::

            \\bar x = \\frac{1}{J}\\sum_j x_j, \\qquad
            \\widehat{C} = \\frac{A^\\top A}{J-1} = FF^\\top,
            \\qquad F = \\frac{A^\\top}{\\sqrt{J-1}} .

        Parameters
        ----------
        samples
            A ``(J, n)`` array, one sample per row, with :math:`J \\ge 2`.

        Returns
        -------
        Gaussian
            Mean the sample mean; covariance the empirical covariance with
            the package's :math:`J-1` divisor, held as a
            :class:`~pyeki.linalg.PSDLowRank` whose factor is
            :math:`A^\\top/\\sqrt{J-1}`.

        Raises
        ------
        ValueError
            If ``samples`` is not rank 2, or if :math:`J < 2` — a single
            sample has no anomalies. In debug mode, also if it is not finite.
        TypeError
            If ``samples`` has no shape to check.

        Notes
        -----
        The covariance is never formed as an :math:`n \\times n` matrix. The
        stored factor is the scaled anomaly matrix, so the operator *is* the
        empirical covariance exactly, and costs :math:`O(nJ)` to hold rather
        than :math:`O(n^2)`.

        Its rank is at most :math:`J-1`, so it is singular whenever
        :math:`J - 1 < n` — the usual regime for this layer — and
        :class:`~pyeki.linalg.PSDLowRank` accordingly provides ``diag`` and
        ``factor`` and withholds ``solve``, ``whiten`` and ``logdet``.
        :meth:`log_density` therefore raises
        :class:`~pyeki.linalg.UnsupportedOpError` on the result, which is
        correct rather than restrictive: a density against a singular
        covariance is not defined.

        Anomalies are formed with the same centring the conditioning methods
        use, so identical samples give exactly zero spread rather than
        round-off, and the cancellation is governed by the spread rather than
        by the magnitude.

        This is a *fit*, not a conditioning result. Nothing about the samples
        is assumed beyond their shape, and a Gaussian fit to a non-Gaussian
        sample describes only its first two moments.
        """
        ndim = getattr(samples, "ndim", None)
        if ndim is None:
            raise TypeError(
                f"Gaussian.from_samples: samples must be an array of rank 2, got "
                f"{type(samples).__name__}, which has no shape to check. Pass a "
                f"JAX array — jnp.asarray() on a nested list."
            )
        if ndim != 2:
            raise ValueError(
                f"Gaussian.from_samples: samples must be rank 2, got shape "
                f"{samples.shape}"
            )
        samples = jnp.asarray(samples)
        n_samples = samples.shape[0]
        if n_samples < 2:
            raise ValueError(
                f"Gaussian.from_samples: at least 2 samples are required, got "
                f"{n_samples}. A single sample has no anomalies."
            )
        _check_finite("Gaussian.from_samples", "samples", samples)
        factor = _centered(samples).T / math.sqrt(n_samples - 1)
        return cls(jnp.mean(samples, axis=-2), PSDLowRank(factor))

    @property
    def dim(self) -> int:
        """The dimension, the trailing core size of ``mean``."""
        return int(self.mean.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return _broadcast_batch(
            "Gaussian", tuple(self.mean.shape[:-1]), self.cov.batch_shape
        )

    def sample(self, key, n_samples: int) -> Array:
        """Draw independent samples: ``(n_samples, n)``.

        The draw is pinned elementwise, not merely distributionally: with
        ``L`` the operator ``cov.factor()`` returns and ``k`` its width, the
        result is exactly

        .. code-block:: python

            mean + L.matvec(jax.random.normal(key, (n_samples, k)))

        one batched ``matvec`` under the operator layer's batch contract.

        Parameters
        ----------
        key
            A JAX PRNG key, consumed whole: splitting is the caller's.
        n_samples
            How many samples to draw. A Python ``int``, at least 1 — it
            determines an output shape, so it can never be traced.

        Returns
        -------
        Array
            Shape ``(n_samples, n)``.

        Raises
        ------
        UnsupportedOpError
            If ``cov`` does not support ``factor``.
        TypeError
            If ``n_samples`` is not a Python ``int`` — ``bool`` and NumPy
            integers included.
        ValueError
            If ``n_samples`` is below 1, or this is a vmapped family.

        Notes
        -----
        Pinning the draw makes sampling reproducible for a fixed covariance
        representation. What is *not* pinned is anything across
        representations: two operators for the same matrix may return
        different factors, hence different samples from the same key. Only
        the distribution is shared.
        """
        _check_not_vmap_family(self, "sample")
        _require_cov_ops(self.cov, "factor")
        if type(n_samples) is not int:
            raise TypeError(
                f"{self!r}.sample: n_samples must be a Python int, got "
                f"{type(n_samples).__name__}. It determines an output shape, so "
                f"it can never be traced."
            )
        if n_samples < 1:
            raise ValueError(
                f"{self!r}.sample: n_samples must be at least 1, got {n_samples}"
            )
        L = self.cov.factor()
        eps = jax.random.normal(key, (n_samples, L.shape[1]))
        return self.mean + L.matvec(eps)

    def log_density(self, x) -> Array:
        """The log-density at ``x``, batched: ``(..., n) -> (...)``.

        .. math::

            \\log p(x) = -\\tfrac{1}{2}\\Bigl(n\\log 2\\pi + \\log\\det C
            + \\lVert W(x - m) \\rVert^2 \\Bigr),

        correct for every valid whitener by the invariant
        :math:`\\lVert W r \\rVert^2 = r^\\top C^{-1} r` that the operator
        contract guarantees.

        Parameters
        ----------
        x
            Evaluation points, shape ``(..., n)`` — the one argument of this
            class that follows the operator layer's batch contract.

        Returns
        -------
        Array
            Shape ``(...)``, each element a 0-d real JAX scalar.

        Raises
        ------
        UnsupportedOpError
            If ``cov`` does not support ``whiten``, or does not support
            ``logdet``, checked in that order.
        ValueError
            If ``x`` has no axes or its trailing axis is not ``n``, if this is
            a vmapped family, or — in debug mode — if ``x`` is not finite.

        Notes
        -----
        Precondition: :math:`C` is nonsingular. A singular covariance yields
        ``nan``/``inf`` downstream of the operator layer, per its
        value-precondition convention.

        The core-shape check on ``x`` runs before anything is computed: a
        shorter ``x`` would otherwise broadcast against ``mean`` and return a
        finite, plausible, wrong number.
        """
        _check_not_vmap_family(self, "log_density")
        _require_cov_ops(self.cov, "whiten", "logdet")
        where = f"{self!r}.log_density"
        x = _check_batched_operand(where, "x", x, self.dim)
        _check_finite(where, "x", x)
        whitened = self.cov.whiten(x - self.mean)
        quadratic = jnp.sum(whitened * whitened, axis=-1)
        return -0.5 * (
            self.dim * math.log(2.0 * math.pi) + self.cov.logdet() + quadratic
        )

    def __repr__(self) -> str:
        """The type name and dimension, as ``Gaussian(dim=12)``; never raises."""
        try:
            base = f"Gaussian(dim={self.dim})"
            batch = self.batch_shape
        except Exception:
            return "<Gaussian (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base


# ---------------------------------------------------------------------------
# the joint Gaussian, and every conditioning identity
# ---------------------------------------------------------------------------


@_pytree_dataclass
class GaussianJoint:
    """A joint Gaussian over the two blocks, held as a joint factor.

    The distribution is

    .. math::

        \\begin{pmatrix} u \\\\ v \\end{pmatrix} \\sim \\mathcal{N}\\!\\left(
        \\begin{pmatrix} \\bar u \\\\ \\bar v \\end{pmatrix},
        \\begin{pmatrix} C_{uu} & C_{uv} \\\\ C_{vu} & C_{vv} \\end{pmatrix}
        \\right),

    with the covariance given by a **joint factor**: a single factor of the
    whole block matrix, cut into the two row blocks
    :math:`F_u \\in \\mathbb{R}^{P \\times k}` and
    :math:`F_v \\in \\mathbb{R}^{N \\times k}`, which drive the two blocks
    from one shared latent vector,

    .. math::

        \\begin{pmatrix} u \\\\ v \\end{pmatrix}
        = \\begin{pmatrix} \\bar u \\\\ \\bar v \\end{pmatrix}
        + \\begin{pmatrix} F_u \\\\ F_v \\end{pmatrix} \\xi,
        \\qquad \\xi \\sim \\mathcal{N}(0, I_k),

    so that

    .. math::

        C_{uu} = F_u F_u^\\top, \\qquad
        C_{uv} = F_u F_v^\\top, \\qquad
        C_{vv} = F_v F_v^\\top ,

    all three at once and with no side condition to check. The shared
    :math:`\\xi` is the whole content of the representation: two factors
    chosen independently, one of :math:`C_{uu}` and one of :math:`C_{vv}`,
    would say nothing at all about :math:`C_{uv}`.

    Both row blocks are :class:`~pyeki.linalg.LinOp` s, so a structured
    covariance stays structured: :meth:`condition` applies :math:`F_u` only
    through ``matvec`` and ``matmat``, and materializes :math:`F_v` as an
    ``(N, k)`` array, which the singular value decomposition needs.

    Use a classmethod rather than the constructor:

    ====================================== ===================================
    constructor                            builds
    ====================================== ===================================
    :meth:`from_linear_map`                the joint of :math:`u` and
                                           :math:`Gu` for a linear map
                                           :math:`G`
    :meth:`from_samples`                   the joint fitted to paired
                                           samples' empirical moments
    :meth:`from_factors`                   a joint from row blocks supplied
                                           directly
    ====================================== ===================================

    Parameters
    ----------
    u_mean
        The mean :math:`\\bar u` of the block to be updated, a ``(P,)``
        array. Keyword-only.
    v_mean
        The mean :math:`\\bar v` of the observed block, a ``(N,)`` array.
        Keyword-only.
    u_factor
        The row block :math:`F_u`, a :class:`~pyeki.linalg.LinOp` of shape
        ``(P, k)``. Keyword-only.
    v_factor
        The row block :math:`F_v`, a :class:`~pyeki.linalg.LinOp` of shape
        ``(N, k)``, sharing the latent width ``k``. Keyword-only.

    Raises
    ------
    ValueError
        If either mean is not rank 1, if a factor's shape disagrees with its
        own mean, if the two factors disagree on ``k``, or if any field is a
        vmapped family. In debug mode, also if either mean is not finite.
    TypeError
        If either factor is not a :class:`~pyeki.linalg.LinOp`, or either mean
        has no shape to check.

    Notes
    -----
    All four fields are **keyword-only**. The two means, and the two
    factors, are pairs of like-shaped objects that agree on their trailing
    size, so exchanging a pair is valid whenever :math:`P = N` and no check
    can detect it: conditioning would then run on the wrong blocks and
    return finite, plausible numbers. Naming the fields at the call site is
    the only thing that rules it out.

    Nothing is factorized here, deliberately: the decomposition the kernel
    needs is of :math:`W F_v`, and the noise operator arrives per call and
    may differ on every call. Each method computes its own decomposition,
    uses it, and discards it, never caching it on the instance — which
    under ``jit`` would write the cache to a temporary copy and silently
    re-factorize on every call.

    Both conditioning methods degrade gracefully when :math:`F_v` is zero:
    :meth:`condition` returns the :math:`u` marginal's own moments and
    :meth:`pathwise` returns its ``u`` argument unchanged. A collapsed
    observed block is a no-op, not ``nan``, for finite inputs.
    """

    u_mean: Array = field(kw_only=True)
    v_mean: Array = field(kw_only=True)
    u_factor: LinOp = field(kw_only=True)
    v_factor: LinOp = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_field_rank("GaussianJoint", "u_mean", self.u_mean, 1)
        _check_field_rank("GaussianJoint", "v_mean", self.v_mean, 1)
        _check_factor_field("GaussianJoint", "u_factor", self.u_factor, "u_mean",
                            self.u_mean.shape[-1])
        _check_factor_field("GaussianJoint", "v_factor", self.v_factor, "v_mean",
                            self.v_mean.shape[-1])
        if self.u_factor.shape[1] != self.v_factor.shape[1]:
            raise ValueError(
                f"GaussianJoint: the two factors must share a latent width, got "
                f"{self.u_factor!r} of width {self.u_factor.shape[1]} and "
                f"{self.v_factor!r} of width {self.v_factor.shape[1]}. They are "
                f"the row blocks of one joint factor, not two independent "
                f"factorizations."
            )
        _check_finite("GaussianJoint", "u_mean", self.u_mean)
        _check_finite("GaussianJoint", "v_mean", self.v_mean)

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_factors(cls, *, u_mean, v_mean, u_factor, v_factor) -> GaussianJoint:
        """A joint from row blocks supplied directly: the escape hatch.

        Use it when the two row blocks are already in hand and coherent —
        derived from one factorization of the joint covariance. Nothing here
        can check that they are: any two operators of matching latent width
        are accepted, and an incoherent pair describes a different
        distribution than the caller intends, with no exception.
        :meth:`from_linear_map` and :meth:`from_samples` construct coherent
        pairs by their arithmetic and are the safe routes.

        Parameters
        ----------
        u_mean, v_mean
            The two means, ``(P,)`` and ``(N,)`` arrays.
        u_factor, v_factor
            The two row blocks, of shapes ``(P, k)`` and ``(N, k)``. A
            :class:`~pyeki.linalg.LinOp`, or an array, which is wrapped as a
            :class:`~pyeki.linalg.Dense`.

        Returns
        -------
        GaussianJoint

        Raises
        ------
        ValueError, TypeError
            As the class documents.
        """
        where = "GaussianJoint.from_factors"
        return cls(
            u_mean=_as_vector(where, "u_mean", u_mean),
            v_mean=_as_vector(where, "v_mean", v_mean),
            u_factor=_as_factor(where, "u_factor", u_factor),
            v_factor=_as_factor(where, "v_factor", v_factor),
        )

    @classmethod
    def from_samples(cls, *, u_samples, v_samples) -> GaussianJoint:
        """The joint Gaussian fitted to paired samples' empirical moments.

        Matches moments to the samples: the means are the sample means and
        the covariance blocks are the empirical covariances with the
        package's :math:`J-1` divisor,

        .. math::

            \\bar u = \\frac{1}{J}\\sum_j u_j, \\qquad
            \\widehat{C}_{uv} = \\frac{A_u^\\top A_v}{J-1},

        and likewise for the other blocks, where
        :math:`A_u = \\mathsf{U} - \\mathbf{1}_J \\bar u^\\top` is the
        anomaly matrix. The joint factor is the scaled anomalies,

        .. math::

            F_u = \\frac{A_u^\\top}{\\sqrt{J-1}}, \\qquad
            F_v = \\frac{A_v^\\top}{\\sqrt{J-1}},
            \\qquad k = J ,

        which reproduces all three blocks exactly and forms none of them.
        The result is a Gaussian *fit to* the samples, not the equal-weight
        point-mass distribution of the samples themselves.

        Parameters
        ----------
        u_samples
            The block to be updated, a ``(J, P)`` array, one sample per row.
            Keyword-only.
        v_samples
            The observed block, a ``(J, N)`` array row-aligned with
            ``u_samples``: row :math:`j` of each belongs to the same sample.
            Keyword-only.

        Returns
        -------
        GaussianJoint
            Of latent width :math:`k = J`.

        Raises
        ------
        ValueError
            If either array is not rank 2, if they disagree on :math:`J`, or
            if :math:`J < 2` — a single sample has no anomalies. In debug
            mode, also if either is not finite.
        TypeError
            If either argument has no shape to check.

        Notes
        -----
        The factor this builds is **centred**, :math:`F\\mathbf{1}_J = 0`,
        because anomalies sum to zero. That is what makes the latent index a
        sample index and lets :class:`EmpiricalJoint` read updated samples
        off a conditioned factor; see
        :meth:`EmpiricalJoint.transform_update`.

        Anomalies are formed by removing the first row before averaging, so
        identical samples give exactly zero spread rather than round-off.
        The alternative is not a ``nan`` but a wrong finite update: the gain
        amplifies spurious anomalies of about
        :math:`\\varepsilon\\lvert\\bar v\\rvert` into an update of order 1
        once the samples are of order :math:`10^{23}`.
        """
        u_samples, v_samples = _check_sample_pair(
            "GaussianJoint", u_samples, v_samples
        )
        scale = math.sqrt(u_samples.shape[0] - 1)
        return cls(
            u_mean=jnp.mean(u_samples, axis=-2),
            v_mean=jnp.mean(v_samples, axis=-2),
            u_factor=Dense(_centered(u_samples).swapaxes(-1, -2) / scale),
            v_factor=Dense(_centered(v_samples).swapaxes(-1, -2) / scale),
        )

    @classmethod
    def from_linear_map(cls, u_marginal: Gaussian, linear_map) -> GaussianJoint:
        """The joint of :math:`u` and :math:`v = G u`, for a linear map :math:`G`.

        With :math:`u \\sim \\mathcal{N}(m_0, C_0)` and :math:`C_0 = L
        L^\\top` for the factor :math:`L` that ``u_marginal.cov.factor()``
        returns, the joint of the pair is

        .. math::

            \\begin{pmatrix} u \\\\ Gu \\end{pmatrix} \\sim
            \\mathcal{N}\\!\\left(
            \\begin{pmatrix} m_0 \\\\ G m_0 \\end{pmatrix},
            \\begin{pmatrix} C_0 & C_0 G^\\top \\\\
            G C_0 & G C_0 G^\\top \\end{pmatrix} \\right),
            \\qquad
            F_u = L, \\quad F_v = G L .

        Conditioning it on :math:`y = Gu + \\eta` therefore gives the
        closed-form Gaussian posterior, computed through the same kernel as
        every other path here: nothing is inverted, no matrix of either
        block's dimension is formed, and a singular :math:`C_0` is fine.

        Parameters
        ----------
        u_marginal
            The marginal over :math:`u`, a :class:`Gaussian` whose
            covariance supports ``factor``.
        linear_map
            The map :math:`G`, a :class:`~pyeki.linalg.LinOp` of shape
            ``(N, P)``.

        Returns
        -------
        GaussianJoint
            Of latent width :math:`k` equal to the width of
            ``u_marginal.cov.factor()``.

        Raises
        ------
        UnsupportedOpError
            If ``u_marginal.cov`` does not support ``factor``.
        TypeError
            If ``u_marginal`` is not a :class:`Gaussian`, or ``linear_map``
            is not a :class:`~pyeki.linalg.LinOp`.
        ValueError
            If ``linear_map``'s input size is not ``u_marginal.dim``, or if
            either argument is a vmapped family.

        Notes
        -----
        **The observation noise is not part of this joint.** It is supplied
        per call, as ``condition(y, noise_cov)``, so one joint can be
        conditioned against a succession of noise operators — including
        scalings of a single operator whose scale is itself traced — while
        re-factorizing nothing.

        :math:`F_v = GL` is materialized here, as an ``(N, k)`` array built
        from ``k`` applications of ``linear_map``. That is the factorization
        this constructor owes the kernel, and doing it once at construction
        is why the shape of :math:`G` never enters a conditioning call.
        """
        if not isinstance(u_marginal, Gaussian):
            raise TypeError(
                f"GaussianJoint.from_linear_map: u_marginal must be a "
                f"pyeki.gauss.Gaussian, got {type(u_marginal).__name__}"
            )
        if not isinstance(linear_map, LinOp):
            raise TypeError(
                f"GaussianJoint.from_linear_map: linear_map must be a "
                f"pyeki.linalg.LinOp, got {type(linear_map).__name__}"
            )
        _check_not_vmap_family(u_marginal, "as the u_marginal of a GaussianJoint")
        if linear_map.batch_shape != ():
            raise ValueError(
                f"GaussianJoint.from_linear_map: linear_map is a vmapped family "
                f"with batch shape {linear_map.batch_shape}; build a family of "
                f"joints with jax.vmap over this constructor."
            )
        if linear_map.shape[1] != u_marginal.dim:
            raise ValueError(
                f"GaussianJoint.from_linear_map: {linear_map!r} takes vectors of "
                f"length {linear_map.shape[1]}, but u_marginal has dimension "
                f"{u_marginal.dim}"
            )
        _require_cov_ops(u_marginal.cov, "factor")
        u_factor = u_marginal.cov.factor()
        return cls(
            u_mean=u_marginal.mean,
            v_mean=linear_map.matvec(u_marginal.mean),
            u_factor=u_factor,
            v_factor=Dense(linear_map.matmat(u_factor.to_dense())),
        )

    # -- sizes and marginals ----------------------------------------------------
    @property
    def u_dim(self) -> int:
        """The dimension :math:`P` of the block to be updated."""
        return int(self.u_mean.shape[-1])

    @property
    def v_dim(self) -> int:
        """The dimension :math:`N` of the observed block."""
        return int(self.v_mean.shape[-1])

    @property
    def latent_dim(self) -> int:
        """The latent width :math:`k`: the number of columns of the joint factor."""
        return int(self.u_factor.shape[1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return _broadcast_batch(
            "GaussianJoint",
            tuple(self.u_mean.shape[:-1]),
            tuple(self.v_mean.shape[:-1]),
            self.u_factor.batch_shape,
            self.v_factor.batch_shape,
        )

    @property
    def u_marginal(self) -> Gaussian:
        """The marginal :math:`\\mathcal{N}(\\bar u, F_u F_u^\\top)` over :math:`u`.

        The covariance is a :class:`~pyeki.linalg.PSDLowRank` holding
        :math:`F_u`, so its rank is at most :math:`k` and it materializes
        the factor as a ``(P, k)`` array. It supports
        :meth:`Gaussian.sample` and ``diag``, and not
        :meth:`Gaussian.log_density`.
        """
        _check_not_vmap_family(self, "u_marginal")
        return Gaussian(self.u_mean, PSDLowRank(self.u_factor.to_dense()))

    @property
    def v_marginal(self) -> Gaussian:
        """The marginal :math:`\\mathcal{N}(\\bar v, F_v F_v^\\top)` over :math:`v`.

        The *noise-free* marginal: it is the distribution of :math:`v`, not
        of an observation of it, so the observation noise :math:`R` does not
        appear. As with :attr:`u_marginal`, the covariance is a
        :class:`~pyeki.linalg.PSDLowRank`.
        """
        _check_not_vmap_family(self, "v_marginal")
        return Gaussian(self.v_mean, PSDLowRank(self.v_factor.to_dense()))

    # -- conditioning -----------------------------------------------------------
    def condition(self, y, noise_cov) -> Gaussian:
        """Condition on :math:`y = v + \\eta`: the posterior over :math:`u`.

        Standard Gaussian conditioning, for the observation model
        :math:`y = v + \\eta` with :math:`\\eta \\sim \\mathcal{N}(0, R)`.
        With :math:`W` a whitener of :math:`R`,
        :math:`S = (W F_v)^\\top` and :math:`K = C_{uv}(C_{vv} + R)^{-1}`,
        the result is

        .. math::

            m_{\\text{post}} = \\bar u + K(y - \\bar v)
            = \\bar u + F_u\\,
              \\texttt{gain\\_weights}\\bigl(S,\\, W(y - \\bar v)\\bigr),

        .. math::

            C_{\\text{post}} = C_{uu} - K C_{vu}
            = \\bigl(F_u T\\bigr)\\bigl(F_u T\\bigr)^\\top,
            \\qquad T = \\texttt{sqrt\\_transform}(S) .

        So conditioning multiplies the joint factor's :math:`u` block on the
        right by :math:`T`, and the posterior is returned in that form: a
        :class:`~pyeki.linalg.PSDLowRank` holding the ``(P, k)`` array
        :math:`F_u T`, never a dense :math:`P \\times P` matrix.

        Parameters
        ----------
        y
            The observation, a ``(N,)`` array.
        noise_cov
            The observation-noise covariance :math:`R`, a
            :class:`~pyeki.linalg.PSDLinOp` of side ``N`` supporting
            ``whiten``.

        Returns
        -------
        Gaussian
            The posterior over :math:`u`, of dimension :math:`P`.

        Raises
        ------
        UnsupportedOpError
            If ``noise_cov`` does not support ``whiten``.
        TypeError
            If ``noise_cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
        ValueError
            If ``y`` is not ``(N,)``, ``noise_cov``'s side is not ``N``,
            this or ``noise_cov`` is a vmapped family, or — in debug mode —
            if ``y`` or the returned value is not finite.

        Notes
        -----
        The returned covariance is honest about rank:
        :math:`\\operatorname{rank}(C_{\\text{post}}) \\le \\min(k, P)`, so
        it is singular whenever :math:`k < P`. The posterior therefore
        supports :meth:`Gaussian.sample` — the factor is the stored
        representation — but not :meth:`Gaussian.log_density`, which raises
        :class:`~pyeki.linalg.UnsupportedOpError` from the covariance. When
        :math:`k \\ge P` the density exists mathematically; the static
        capability choice still raises, and a caller wanting it densifies
        the covariance deliberately.

        Precondition: ``noise_cov`` is nonsingular, which is ``whiten``'s
        own precondition. Nothing here can detect a singular one before the
        fact, so it surfaces as ``nan`` — or, in debug mode, as the result
        check applied to the two halves of the answer.

        Whitening costs :math:`k + 1` applications of :math:`W`, from a
        single call on the stacked columns :math:`[F_v \\mid y - \\bar v]`.
        """
        _check_not_vmap_family(self, "condition")
        where = f"{self!r}.condition"
        y = _validate_conditioning_call(where, y, noise_cov, self.v_dim)
        mean, factor = self._posterior_mean_and_factor(y, noise_cov)
        # Both halves, checked before the covariance and the distribution are
        # built, so the diagnosis names this call rather than a constructor
        # further down.
        _check_finite(where, "the posterior mean", mean, cause=_SINGULAR_NOISE)
        _check_finite(
            where, "the posterior covariance factor", factor, cause=_SINGULAR_NOISE
        )
        return Gaussian(mean, PSDLowRank(factor))

    def pathwise(self, *, u, v, whitened_noise, y, noise_cov) -> Array:
        """Transport realizations of the joint to the posterior: Matheron's rule.

        The affine map

        .. math::

            \\Phi(u, v, \\eta) = u + K\\bigl(y - v - \\eta\\bigr),
            \\qquad K = C_{uv}\\bigl(C_{vv} + R\\bigr)^{-1},

        which depends on the joint only through :math:`K` — that is, only
        through its moments. If :math:`(u, v)` is distributed as the joint
        and :math:`\\eta \\sim \\mathcal{N}(0, R)` independently, then
        :math:`\\Phi(u, v, \\eta)` is distributed as the posterior
        :meth:`condition` returns:

        .. math::

            \\operatorname{Cov}\\Phi
            = C_{uu} - K C_{vu} - C_{uv}K^\\top + K(C_{vv} + R)K^\\top
            = C_{uu} - K C_{vu},

        since :math:`K(C_{vv} + R) = C_{uv}`. Each realization is
        transported independently of every other, so a batch of them is one
        call rather than a loop.

        The noise enters in **whitened** coordinates: with
        :math:`\\varepsilon = W\\eta \\sim \\mathcal{N}(0, I_N)` and
        :math:`S = (W F_v)^\\top`, what is computed is

        .. math::

            \\Phi = u + F_u\\,
            \\texttt{gain\\_weights}\\bigl(S,\\, W(y - v) - \\varepsilon\\bigr).

        Parameters
        ----------
        u
            Realizations of the first block, shape ``(..., P)``.
            Keyword-only.
        v
            The paired realizations of the observed block, shape
            ``(..., N)``, broadcasting against ``u``'s batch axes.
            Keyword-only.
        whitened_noise
            Standard normal draws :math:`\\varepsilon`, shape ``(..., N)``.
            **Whitened**: the method applies :math:`W^{-1}` to nothing, so
            these are draws in whitened coordinates, not draws from
            :math:`R`. Keyword-only.
        y
            The observation, a ``(N,)`` array. Keyword-only.
        noise_cov
            The observation-noise covariance :math:`R`, a
            :class:`~pyeki.linalg.PSDLinOp` of side ``N`` supporting
            ``whiten``. Keyword-only.

        Returns
        -------
        Array
            Shape ``(..., P)``, the broadcast batch shape of the three
            realization arguments.

        Raises
        ------
        UnsupportedOpError
            If ``noise_cov`` does not support ``whiten``.
        TypeError
            If ``noise_cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
        ValueError
            If any operand's trailing size is wrong, if ``noise_cov``'s side
            is not ``N``, if this or ``noise_cov`` is a vmapped family, or —
            in debug mode — if an operand or the result is not finite.

        Notes
        -----
        Every argument is **keyword-only**. ``v`` and ``whitened_noise`` are
        both ``(..., N)`` and so are freely exchangeable, and ``u`` joins
        them when :math:`P = N`; a positional call would make a silent
        wrong answer out of an argument-order slip.

        A whitened perturbation must never *also* be pushed through
        ``factor()`` in the same update: :math:`WL` has orthonormal rows but
        is not the identity, so mixing the two representations of one
        :math:`\\varepsilon` corrupts the joint law of the result while
        every marginal statistic still looks right. Draw
        ``whitened_noise`` once, and use it only here.

        Whitening costs :math:`k` applications of :math:`W` for :math:`S`,
        plus one per realization for the residuals. Unlike
        :meth:`condition`, the residuals cannot come free from :math:`S`:
        ``v`` is arbitrary data rather than the joint's own factor.
        :meth:`EmpiricalJoint.pathwise_update`, whose realizations *are* the
        factor, spends :math:`J + 1` in total instead.
        """
        _check_not_vmap_family(self, "pathwise")
        where = f"{self!r}.pathwise"
        y = _validate_conditioning_call(where, y, noise_cov, self.v_dim)
        u = _check_batched_operand(where, "u", u, self.u_dim)
        v = _check_batched_operand(where, "v", v, self.v_dim)
        eps = _check_batched_operand(where, "whitened_noise", whitened_noise, self.v_dim)
        _check_operands_broadcast(
            where,
            ("u", u.shape[:-1]),
            ("v", v.shape[:-1]),
            ("whitened_noise", eps.shape[:-1]),
        )
        _check_finite(where, "u", u)
        _check_finite(where, "v", v)
        _check_finite(where, "whitened_noise", eps)
        s = self._whitened_factor(noise_cov)
        b = noise_cov.whiten(y - v) - eps
        # the private kernel rather than gain_weights, so that a non-finite
        # result is diagnosed against this call rather than against a
        # primitive the caller never invoked
        U, sigma, Vt = _thin_svd(s)
        transported = u + self.u_factor.matvec(_weights_from_svd(U, sigma, Vt, b))
        _check_finite(
            where, "the transported realizations", transported, cause=_SINGULAR_NOISE
        )
        return transported

    def __repr__(self) -> str:
        """As ``GaussianJoint(u_dim=12, v_dim=40, latent_dim=100)``; never raises."""
        try:
            base = (
                f"GaussianJoint(u_dim={self.u_dim}, v_dim={self.v_dim}, "
                f"latent_dim={self.latent_dim})"
            )
            batch = self.batch_shape
        except Exception:
            return "<GaussianJoint (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base

    # -- private: the whitened-SVD assembly --
    def _whitened_factor(self, noise_cov) -> Array:
        """The whitened factor :math:`S = (W F_v)^\\top`, a ``(k, N)`` array.

        One ``whiten_mat`` call on the ``(N, k)`` factor, for :math:`k`
        applications of :math:`W`. :meth:`pathwise` wants exactly this and
        no mean residual, so taking it from
        :meth:`_whitened_factor_and_residual` would whiten one vector more
        than the call needs.
        """
        return noise_cov.whiten_mat(self.v_factor.to_dense()).swapaxes(-1, -2)

    def _whitened_factor_and_residual(self, y, noise_cov) -> tuple[Array, Array]:
        """:math:`S` and :math:`W(y - \\bar v)`, from a single ``whiten_mat`` call.

        Returns shapes ``(k, N)`` and ``(N,)``, whitening the stacked
        columns :math:`[F_v \\mid y - \\bar v]` in one call, for
        :math:`k + 1` applications of :math:`W`. Two calls would cost the
        same here; one is used because the same grouping is what makes
        :meth:`EmpiricalJoint.pathwise_update` cost :math:`J + 1` rather
        than :math:`2J`, and having one helper keeps the two paths in step.

        The difference :math:`y - \\bar v` is formed *before* the whitener
        is applied. Whitening is linear, so the orders agree in exact
        arithmetic — but not in stability: whitening first makes the
        cancellation ratio :math:`\\lVert W\\bar v\\rVert / \\lVert W F_v
        \\rVert` in place of :math:`\\lVert \\bar v\\rVert / \\lVert F_v
        \\rVert`, so the error grows with :math:`\\kappa(W) =
        \\sqrt{\\kappa(R)}` whenever the mean of the observed block is
        aligned with a precise direction of the noise.
        """
        columns = jnp.concatenate(
            [self.v_factor.to_dense(), (y - self.v_mean)[..., None]], axis=-1
        )
        whitened = noise_cov.whiten_mat(columns)
        return whitened[..., :-1].swapaxes(-1, -2), whitened[..., -1]

    def _posterior_mean_and_factor(self, y, noise_cov) -> tuple[Array, Array]:
        """The posterior mean and covariance factor :math:`F_u T`, from one SVD.

        Shared by :meth:`condition` and
        :meth:`EmpiricalJoint.transform_update`, which are two
        representations of the same posterior, so computing it once makes
        them agree by construction.
        """
        s, whitened_residual = self._whitened_factor_and_residual(y, noise_cov)
        U, sigma, Vt = _thin_svd(s)
        mean = self.u_mean + self.u_factor.matvec(
            _weights_from_svd(U, sigma, Vt, whitened_residual)
        )
        transform = _transform_from_svd(U, sigma, self.latent_dim)
        return mean, self.u_factor.matmat(transform)



# ---------------------------------------------------------------------------
# paired samples, and the two updates that carry them forward
# ---------------------------------------------------------------------------


@_pytree_dataclass
class EmpiricalJoint:
    """:math:`J` paired samples, and the two updates that carry them forward.

    Holds the samples and nothing else. Its two update methods condition on
    :math:`y = v + \\eta` with :math:`\\eta \\sim \\mathcal{N}(0, R)` and
    return a ``(J, P)`` array of updated samples, row :math:`j` updating
    row :math:`j`; they update :math:`u` only, leaving a caller who needs a
    matching :math:`v` to recompute it.

    Reading a *distribution* out of the samples is
    :meth:`to_gaussian_joint`, which fits a Gaussian to their moments and
    hands back a :class:`GaussianJoint`. That call is deliberately not
    hidden inside a method here: conditioning samples means conditioning a
    Gaussian fitted to them, and the fit belongs in the source text::

        posterior = joint.to_gaussian_joint().condition(y, noise_cov)

    Parameters
    ----------
    u_samples
        The block to be updated, a ``(J, P)`` array, one sample per row.
        Keyword-only.
    v_samples
        The block that is observed, a ``(J, N)`` array row-aligned with
        ``u_samples``: row :math:`j` of each belongs to the same sample.
        Keyword-only.

    Raises
    ------
    ValueError
        If either array is not rank 2, if they disagree on :math:`J`, or if
        :math:`J < 2` — a single sample has no anomalies. In debug mode,
        also if either is not finite.
    TypeError
        If either field has no shape to check — a list, a tuple, a scalar.

    Notes
    -----
    Both fields are **keyword-only**. They are two arrays of the same rank
    whose sizes agree on the sample axis, so a swap is a shape-valid
    mistake that no check can catch when :math:`P = N`: the update would be
    computed from the wrong blocks and return finite, plausible numbers.
    Naming them at the call site is the only thing that rules it out. The
    cost is that a family is built through a lambda rather than by mapping
    the constructor directly::

        jax.vmap(lambda u, v: EmpiricalJoint(u_samples=u, v_samples=v))(U, V)

    The four array properties — the two means and the two anomaly
    matrices — raise ``ValueError`` on a vmapped family, as the methods do.

    Both updates degrade gracefully when the :math:`v` anomalies are zero:
    they return ``u_samples`` unchanged — bit-exactly for
    :meth:`pathwise_update`, which adds to the samples themselves, and to
    within a unit in the last place for :meth:`transform_update`, which
    rebuilds them from the mean and the anomalies. A collapsed sample block
    is a no-op, not ``nan``, for finite inputs.
    """

    u_samples: Array = field(kw_only=True)
    v_samples: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_sample_pair("EmpiricalJoint", self.u_samples, self.v_samples)

    @property
    def n_samples(self) -> int:
        """The number of samples :math:`J`."""
        return int(self.u_samples.shape[-2])

    @property
    def u_dim(self) -> int:
        """The dimension :math:`P` of the block to be updated."""
        return int(self.u_samples.shape[-1])

    @property
    def v_dim(self) -> int:
        """The dimension :math:`N` of the observed block."""
        return int(self.v_samples.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        return _broadcast_batch(
            "EmpiricalJoint",
            tuple(self.u_samples.shape[:-2]),
            tuple(self.v_samples.shape[:-2]),
        )

    @property
    def u_mean(self) -> Array:
        """The sample mean :math:`\\bar u`, a ``(P,)`` array."""
        _check_not_vmap_family(self, "u_mean")
        return jnp.mean(self.u_samples, axis=-2)

    @property
    def v_mean(self) -> Array:
        """The sample mean :math:`\\bar v`, a ``(N,)`` array."""
        _check_not_vmap_family(self, "v_mean")
        return jnp.mean(self.v_samples, axis=-2)

    @property
    def u_anomalies(self) -> Array:
        """The anomalies :math:`A_u = \\mathsf{U} - \\mathbf{1}_J\\bar u^\\top`.

        A ``(J, P)`` array.
        """
        _check_not_vmap_family(self, "u_anomalies")
        return _centered(self.u_samples)

    @property
    def v_anomalies(self) -> Array:
        """The anomalies :math:`A_v`, a ``(J, N)`` array."""
        _check_not_vmap_family(self, "v_anomalies")
        return _centered(self.v_samples)

    def to_gaussian_joint(self) -> GaussianJoint:
        """The joint Gaussian fitted to these samples' moments.

        Delegates to :meth:`GaussianJoint.from_samples`, so the result has
        the sample means and the empirical covariances with the package's
        :math:`J-1` divisor, held as the joint factor
        :math:`F = A^\\top/\\sqrt{J-1}` of latent width :math:`k = J`.

        Returns
        -------
        GaussianJoint

        Raises
        ------
        ValueError
            If this is a vmapped family.

        Notes
        -----
        The conversion **loses nothing**: with the mean kept and the factor
        of width :math:`J`, the samples are recovered exactly as
        :math:`u_j = \\bar u + \\sqrt{J-1}\\,(F_u)_{\\cdot j}`. What it drops
        is the *reading* of the latent index as a sample index — which is
        why the two updates below live here and not on the joint.

        The joint is derived on each call, not stored. It is an
        :math:`O(J(P+N))` centre-and-scale that fuses under ``jit``, and
        holding both representations would hold the same numbers twice.
        """
        _check_not_vmap_family(self, "to_gaussian_joint")
        return GaussianJoint.from_samples(
            u_samples=self.u_samples, v_samples=self.v_samples
        )

    def transform_update(self, y, noise_cov) -> Array:
        """The deterministic (square-root) update: ``(J, P)``.

        The posterior of the fitted joint Gaussian, read back as samples.
        Conditioning multiplies the factor on the right by :math:`T =
        \\texttt{sqrt\\_transform}(S)`, and because that factor is centred
        the conditioned one is too, so its columns are again a sample set:

        .. math::

            u_j' = m_{\\text{post}}
            + \\sqrt{J-1}\\,\\bigl(F_u T\\bigr)_{\\cdot j},
            \\qquad
            m_{\\text{post}} = \\bar u + F_u\\,
            \\texttt{gain\\_weights}\\bigl(S,\\, W(y - \\bar v)\\bigr).

        No randomness, and no ``key``. The returned block is an *exact*
        representation of the posterior: its sample mean equals the
        posterior mean and its sample covariance, with the divisor
        :math:`J-1`, equals the posterior covariance, both in exact
        arithmetic. Equivalently, in terms of the anomalies,
        :math:`u_j' = m_{\\text{post}} + (T A_u)_j`.

        Parameters
        ----------
        y
            The observation, a ``(N,)`` array.
        noise_cov
            The observation-noise covariance :math:`R`, a
            :class:`~pyeki.linalg.PSDLinOp` of side ``N`` supporting
            ``whiten``.

        Returns
        -------
        Array
            Shape ``(J, P)``, row :math:`j` updating row :math:`j` of
            ``u_samples``.

        Raises
        ------
        UnsupportedOpError
            If ``noise_cov`` does not support ``whiten``.
        TypeError
            If ``noise_cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
        ValueError
            If ``y`` is not ``(N,)``, ``noise_cov``'s side is not ``N``,
            this or ``noise_cov`` is a vmapped family, or — in debug mode —
            if ``y`` or the returned value is not finite.

        Notes
        -----
        This is exactly ``to_gaussian_joint().condition(y, noise_cov)``
        followed by the reading above, sharing that method's single
        decomposition. It is a method here, rather than a function of the
        returned posterior, because the reading is valid only for a centred
        factor. On a joint built any other way it returns, silently, a
        sample set whose mean is displaced by :math:`\\sqrt{k-1}\\,F_uT
        \\mathbf{1}_k/k` and whose sample covariance falls short of the
        posterior's by the rank-one term
        :math:`(F_uT\\mathbf{1}_k)(F_uT\\mathbf{1}_k)^\\top/k` — measured at
        18% of the covariance's own scale on a
        :meth:`~GaussianJoint.from_linear_map` joint.

        Which update to use is the caller's decision, not this layer's.
        This one replaces sampling noise with an exact transform;
        :meth:`pathwise_update` matches the posterior moments only in
        expectation, at the usual :math:`O(J^{-1/2})` rate.
        """
        _check_not_vmap_family(self, "transform_update")
        where = f"{self!r}.transform_update"
        y = _validate_conditioning_call(where, y, noise_cov, self.v_dim)
        joint = self.to_gaussian_joint()
        mean, factor = joint._posterior_mean_and_factor(y, noise_cov)
        updated = mean + math.sqrt(self.n_samples - 1) * factor.swapaxes(-1, -2)
        _check_finite(where, "the updated block", updated, cause=_SINGULAR_NOISE)
        return updated

    def pathwise_update(self, key, y, noise_cov) -> Array:
        """The stochastic (perturbed-observation) update: ``(J, P)``.

        Matheron's rule applied to the samples themselves, one fresh
        perturbation each:

        .. math::

            u_j' = u_j + F_u\\, w_j, \\qquad
            w_j = \\texttt{gain\\_weights}\\bigl(S,\\, b_j\\bigr), \\qquad
            b_j = W(y - v_j) - \\varepsilon_j,

        with :math:`\\varepsilon_j` the rows of the pinned draw
        ``jax.random.normal(key, (J, N))`` and :math:`F_u =
        A_u^\\top/\\sqrt{J-1}`. The perturbation enters *only* in whitened
        space, so equivalently :math:`u_j' = u_j + K(y - v_j - \\eta_j)`
        with :math:`\\eta_j = W^{-1}\\varepsilon_j \\sim \\mathcal{N}(0, R)`.

        Parameters
        ----------
        key
            A JAX PRNG key, consumed whole: splitting is the caller's.
        y
            The observation, a ``(N,)`` array.
        noise_cov
            The observation-noise covariance :math:`R`, a
            :class:`~pyeki.linalg.PSDLinOp` of side ``N`` supporting
            ``whiten``.

        Returns
        -------
        Array
            Shape ``(J, P)``, row :math:`j` updating row :math:`j` of
            ``u_samples``.

        Raises
        ------
        UnsupportedOpError
            If ``noise_cov`` does not support ``whiten``.
        TypeError
            If ``noise_cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
        ValueError
            If ``y`` is not ``(N,)``, ``noise_cov``'s side is not ``N``,
            this or ``noise_cov`` is a vmapped family, or — in debug mode —
            if ``y`` or the returned value is not finite.

        Notes
        -----
        The method **neither accepts nor exposes the perturbations**: no
        ``eps`` argument, and no perturbed observations in the return value.
        A perturbation used through the whitened shortcut must never also be
        pushed through ``factor()`` in the same update — :math:`WL` has
        orthonormal rows but is not the identity, so mixing the two
        representations of the same :math:`\\varepsilon` corrupts the joint
        law of the update while every marginal statistic still looks right.
        The surest way to prevent that is for the same code to own both the
        draw and its single use. A caller who needs the perturbations
        materialized wants :meth:`GaussianJoint.pathwise`, which takes them
        explicitly, or :func:`gain_weights` directly.

        Because the samples *are* the fitted joint's factor, the whitened
        residuals follow from the whitened factor and the update costs
        :math:`J + 1` applications of :math:`W`, not :math:`2J`. Routing
        the same samples through :meth:`GaussianJoint.pathwise` computes the
        same thing, to round-off, and spends :math:`2J`.

        The output's sample mean and sample covariance, with the divisor
        :math:`J-1`, are unbiased estimators of the posterior moments
        :meth:`transform_update` represents exactly; the mean has variance
        :math:`K R K^\\top / J`. Individual samples are not posterior draws:
        conditional on the sample block, sample :math:`j` is distributed
        :math:`\\mathcal{N}(u_j + K(y - v_j),\\, K R K^\\top)`.

        Precondition: ``noise_cov`` is nonsingular, which is ``whiten``'s
        own precondition. Nothing here can detect a singular one before the
        fact, so it surfaces as ``nan`` — or, in debug mode, as the result
        check applied to what this returns.
        """
        _check_not_vmap_family(self, "pathwise_update")
        where = f"{self!r}.pathwise_update"
        y = _validate_conditioning_call(where, y, noise_cov, self.v_dim)
        joint = self.to_gaussian_joint()
        eps = jax.random.normal(key, (self.n_samples, self.v_dim))
        s, whitened_residual = joint._whitened_factor_and_residual(y, noise_cov)
        U, sigma, Vt = _thin_svd(s)
        # W(y - v_j) = W(y - v_bar) - sqrt(J-1) S_j., because these samples are
        # the factor: v_j - v_bar is exactly sqrt(J-1) times column j of F_v.
        b = whitened_residual - math.sqrt(self.n_samples - 1) * s - eps
        updated = self.u_samples + joint.u_factor.matvec(
            _weights_from_svd(U, sigma, Vt, b)
        )
        _check_finite(where, "the updated block", updated, cause=_SINGULAR_NOISE)
        return updated

    def __repr__(self) -> str:
        """As ``EmpiricalJoint(n_samples=100, u_dim=12, v_dim=40)``; never raises."""
        try:
            base = (
                f"EmpiricalJoint(n_samples={self.n_samples}, u_dim={self.u_dim}, "
                f"v_dim={self.v_dim})"
            )
            batch = self.batch_shape
        except Exception:
            return "<EmpiricalJoint (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base


# ---------------------------------------------------------------------------
# private helpers: validation, and the conditioning kernel
# ---------------------------------------------------------------------------


def _check_field_rank(cls_name: str, field_name: str, value, core_ndim: int) -> None:
    """Require an array field of exactly its own rank, with positive sizes.

    Objects in this layer are unbatched; a family is built with
    :func:`jax.vmap` over the pytree, never by storing extra leading axes.
    Enforcing that in the constructor is safe because pytree unflattening
    bypasses it.

    Raises
    ------
    TypeError
        If the field has no shape at all — a list, a tuple, a scalar. These
        checks are unconditional, so a field that cannot be inspected is
        rejected rather than waved through: passing it silences every check
        that follows, including those on the *other* field, and yields an
        object whose every accessor raises ``AttributeError`` instead. Arrays
        that merely are not JAX arrays, such as NumPy's, have a shape and are
        accepted.
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
            f"shape {value.shape}. Objects in pyeki.gauss are unbatched; build a "
            f"family with jax.vmap over the pytree, not with extra leading axes."
        )
    if any(size < 1 for size in value.shape):
        raise ValueError(
            f"{cls_name}.{field_name}: core sizes must be positive, got shape "
            f"{value.shape}."
        )


def _check_factor_field(
    cls_name: str, field_name: str, value, mean_name: str, size: int
) -> None:
    """Require a factor field: an operator, unbatched, of the mean's dimension.

    Raises
    ------
    TypeError
        If the field is not a :class:`~pyeki.linalg.LinOp`. The row blocks
        of a joint factor are operators, so that a structured covariance
        keeps its structure; an array is wrapped by
        :meth:`GaussianJoint.from_factors`, not here.
    ValueError
        If the field is a vmapped family, or its output size disagrees with
        the mean it belongs to.
    """
    if not isinstance(value, LinOp):
        raise TypeError(
            f"{cls_name}.{field_name}: must be a pyeki.linalg.LinOp, got "
            f"{type(value).__name__}. Wrap an array with pyeki.linalg.Dense, or "
            f"use {cls_name}.from_factors, which wraps it for you."
        )
    if value.batch_shape != ():
        raise ValueError(
            f"{cls_name}.{field_name}: {value!r} is a vmapped family; build a "
            f"family of joints with jax.vmap over a constructor, not from a "
            f"family factor."
        )
    if value.shape[0] != size:
        raise ValueError(
            f"{cls_name}.{field_name}: {value!r} has output size "
            f"{value.shape[0]}, which disagrees with {mean_name} of length "
            f"{size}"
        )
    # As PSDLowRank checks its own factor: a non-finite factor makes every
    # result nan with no exception, and the conditioning result check would
    # then name a singular noise_cov, which is not the cause. The check reads
    # the operator's own array leaves rather than its dense form, which would
    # allocate O(P k) on every construction whether or not checks are enabled.
    for leaf in jax.tree_util.tree_leaves(value):
        _check_finite(cls_name, field_name, leaf)


def _as_vector(where: str, name: str, value) -> Array:
    """Convert a mean field to an array, naming the call if it cannot be."""
    try:
        return jnp.asarray(value)
    except (TypeError, ValueError) as e:
        kind = ValueError if isinstance(e, ValueError) else TypeError
        raise kind(f"{where}: {name} must be an array of shape (n,)") from e


def _as_factor(where: str, name: str, value) -> LinOp:
    """Accept an operator as a factor row block, or wrap an array as one."""
    if isinstance(value, LinOp):
        return value
    try:
        return Dense(jnp.asarray(value))
    except (TypeError, ValueError) as e:
        # Never reconstruct the caught type: an exception class whose __init__
        # takes more than one argument raises from the re-raise itself, losing
        # the diagnosis. The base class of the category always accepts a string.
        kind = ValueError if isinstance(e, ValueError) else TypeError
        raise kind(
            f"{where}: {name} must be a pyeki.linalg.LinOp of shape (n, k), "
            f"or an array of that shape"
        ) from e


def _check_sample_pair(cls_name: str, u_samples, v_samples) -> tuple[Array, Array]:
    """Require two row-aligned sample blocks of at least two samples.

    Shared by :class:`EmpiricalJoint`'s construction and
    :meth:`GaussianJoint.from_samples`, so the two agree on what a sample
    pair is. Returns the pair as arrays.
    """
    _check_field_rank(cls_name, "u_samples", u_samples, 2)
    _check_field_rank(cls_name, "v_samples", v_samples, 2)
    u_shape, v_shape = u_samples.shape, v_samples.shape
    if u_shape[0] != v_shape[0]:
        raise ValueError(
            f"{cls_name}: u_samples and v_samples must have the same number of "
            f"samples, got shapes {u_shape} and {v_shape}"
        )
    if u_shape[0] < 2:
        raise ValueError(
            f"{cls_name}: at least 2 samples are required, got {u_shape[0]}. A "
            f"single sample has no anomalies."
        )
    _check_finite(cls_name, "u_samples", u_samples)
    _check_finite(cls_name, "v_samples", v_samples)
    return jnp.asarray(u_samples), jnp.asarray(v_samples)


def _validate_conditioning_call(where: str, y, noise_cov, v_dim: int) -> Array:
    """Run the shared checks of a conditioning call; return the validated ``y``.

    ``where`` describes the call, so a method of :class:`EmpiricalJoint` that
    delegates to :class:`GaussianJoint` still names itself in the diagnosis.

    In the operator layer's order, with one departure the type system forces:
    ``noise_cov``'s type and family — which must precede the capability
    check, since ``supports`` cannot be consulted on an object not known to
    be an operator — then the capability check, then the remaining operand
    shapes, then, in debug mode, operand value checks. The family guard on
    the object itself runs before this, in the calling method, and result
    checks run last, once there is a result.
    """
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{where}: noise_cov must be a pyeki.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
    if noise_cov.batch_shape != ():
        raise ValueError(
            f"{where}: noise_cov is a vmapped family with batch shape "
            f"{noise_cov.batch_shape}; apply a family of noise operators with "
            f"jax.vmap over the method, not directly."
        )
    _require_cov_ops(noise_cov, "whiten")
    _check_noise_cov_dim(where, noise_cov, v_dim)
    y = _check_unbatched_operand(where, "y", y, v_dim)
    _check_finite(where, "y", y)
    return y


def _check_operands_broadcast(where: str, *named_batches) -> None:
    """Require operands whose batch axes broadcast against one another.

    The trailing-size checks pass independently, so without this a batch-shape
    disagreement surfaces from inside JAX as an unattributed broadcasting
    error rather than as a diagnosis naming the arguments.
    """
    try:
        jnp.broadcast_shapes(*(batch for _, batch in named_batches))
    except ValueError:
        described = ", ".join(
            f"{name} {tuple(batch)}" for name, batch in named_batches
        )
        raise ValueError(
            f"{where}: the batch axes of {described} do not broadcast"
        ) from None


def _check_batched_operand(where: str, name: str, x, size: int) -> Array:
    """Require an operand of shape ``(..., size)``, batch axes allowed.

    Without this check a short operand broadcasts against the stored arrays
    and the call returns a finite, plausible, wrong number.
    """
    x = jnp.asarray(x)
    if x.ndim < 1 or x.shape[-1] != size:
        raise ValueError(
            f"{where}: expected {name} of core shape (..., {size}), got shape "
            f"{x.shape}"
        )
    return x


def _check_unbatched_operand(where: str, name: str, x, size: int) -> Array:
    """Require an operand of shape exactly ``(size,)``, with no batch axes.

    Unlike operator operands, the vectors these classes take carry no batch
    axes: a family of updates is :func:`jax.vmap` over the method.
    """
    x = jnp.asarray(x)
    if x.ndim != 1 or x.shape[0] != size:
        raise ValueError(
            f"{where}: expected {name} of shape ({size},), got shape {x.shape}. "
            f"This argument takes no batch axes; map a family of calls with "
            f"jax.vmap over the method."
        )
    return x


def _check_not_vmap_family(obj, operation: str) -> None:
    """Refuse an operation on a vmapped family, before any other check."""
    batch = obj.batch_shape
    if batch != ():
        raise ValueError(
            f"{obj!r}.{operation}: a vmapped family cannot be used directly; its "
            f"batch shape is {batch}. Apply it under jax.vmap, one at a time."
        )


def _check_noise_cov_dim(where: str, noise_cov, size: int) -> None:
    """Require a noise covariance of the dimension the observed block has.

    Noise covariances are square, so one dimension describes them.
    """
    if noise_cov.shape[0] != size:
        raise ValueError(
            f"{where}: expected noise_cov of dimension {size}, got "
            f"{noise_cov!r} of shape {noise_cov.shape}"
        )


def _require_cov_ops(cov, *names: str) -> None:
    """Demand operations of a covariance, in order, raising from the operator layer.

    The ``UnsupportedOpError`` is constructed by the operator that lacks the
    operation and propagated unmodified: this layer never wraps it and never
    falls back to dense linear algebra on the caller's behalf.
    """
    for name in names:
        cov._require(name)


def _check_finite(where: str, name: str, x, *, cause: str | None = None) -> None:
    """Require a finite array, only when debug checks are enabled.

    Used both on arguments the caller supplied and on values this layer
    produced. ``cause`` appends a hint to the message, for the results, where
    the reason is not visible at the point the check fires.
    """
    hint = f" {cause}" if cause else ""
    value_check(
        x,
        lambda arr: bool(jnp.all(jnp.isfinite(arr))),
        f"{where}: {name} must be finite.{hint}",
    )


#: Why a conditioning result goes non-finite, which the check cannot itself see.
_SINGULAR_NOISE = (
    "The likeliest cause is a singular noise_cov: whiten's precondition is "
    "that the noise covariance is nonsingular, and nothing here can detect a "
    "violation before the fact."
)


def _centered(x: Array) -> Array:
    """Deviations from the sample mean over the sample axis, formed stably.

    Mathematically :math:`x - \\mathbf{1}\\bar x^\\top`, computed by removing
    the first row before averaging. Two things follow, neither of them true
    of a direct subtraction of :func:`jax.numpy.mean`:

    - **Identical rows give exactly zero.** ``jnp.mean`` of :math:`J`
      bit-identical rows sums and divides, which does not in general return
      the value it was given, so a collapsed sample block otherwise acquires
      spurious anomalies of about :math:`\\varepsilon\\lvert\\bar x\\rvert`.
      The gain amplifies those into a wrong, finite, ``nan``-free update once
      the rows are large: at :math:`v \\equiv 6\\times10^{23}` the spurious
      update is of order 1 where the exact answer is no update at all.
    - **Cancellation is governed by the spread, not the magnitude.** The error
      is :math:`O(\\varepsilon \\max_j \\lvert x_j - x_0 \\rvert)` rather than
      :math:`O(\\varepsilon \\max_j \\lvert x_j \\rvert)`, which matters
      whenever the mean is large relative to the anomalies — the regime a
      nearly collapsed sample block sits in.
    """
    shifted = x - x[..., :1, :]
    return shifted - jnp.mean(shifted, axis=-2, keepdims=True)


def _thin_svd(s: Array) -> tuple[Array, Array, Array]:
    """Thin SVD :math:`s = U \\Sigma V^\\top`, as ``(U, sigma, Vt)``.

    Shapes are ``(k, rho)``, ``(rho,)`` and ``(rho, N)`` for
    ``rho = min(k, N)``. The single SVD call site of the module: a class
    method calls this once and feeds both pieces below, so "one method call,
    one SVD" holds by construction.
    """
    return jnp.linalg.svd(s, full_matrices=False)


def _weights_from_svd(U: Array, sigma: Array, Vt: Array, b: Array) -> Array:
    """Apply the gain multipliers in the whitened SVD basis.

    Computes :math:`U \\operatorname{diag}(\\sigma_i/(1+\\sigma_i^2)) V^\\top b`,
    contracting the trailing axis of ``b`` and carrying its batch axes.
    Neither :math:`s^\\top s` nor :math:`s s^\\top` appears.
    """
    coefficients = dense_matvec(Vt, b) * (sigma / (1.0 + sigma**2))
    return dense_matvec(U, coefficients)


def _transform_from_svd(U: Array, sigma: Array, latent_dim: int) -> Array:
    """Assemble :math:`T = I_k + U((I+\\Sigma^2)^{-1/2} - I)U^\\top`.

    The identity completion is what makes this exact at every rank: for a
    thin SVD the naive :math:`U(I+\\Sigma^2)^{-1/2}U^\\top` omits the identity
    on the orthogonal complement of :math:`U`'s columns and is simply wrong
    whenever :math:`\\rho < k`.

    :math:`T\\mathbf{1} = \\mathbf{1}` survives floating point for a
    centred factor because the modifier decays *quadratically*: the
    numerically-zero singular value's column of :math:`U` need not be
    orthogonal to :math:`\\mathbf{1}`, but its modifier is
    :math:`O(\\sigma_i^2)`, so the induced mean shift is
    :math:`O((\\varepsilon\\sigma_{\\max})^2)` rather than
    :math:`O(\\varepsilon\\sigma_{\\max})`. Computed as written, the modifier
    is in fact exactly ``0.0`` once :math:`\\sigma_i^2` falls below the
    resolution of ``1.0``, which is the same bound reached the short way.
    """
    modifier = 1.0 / jnp.sqrt(1.0 + sigma**2) - 1.0
    # (k, rho) @ (rho, k): both operands are exactly 2-D, so this is the
    # plain matrix product, not a batch of vectors.
    return jnp.eye(latent_dim, dtype=U.dtype) + (U * modifier) @ U.swapaxes(-1, -2)

