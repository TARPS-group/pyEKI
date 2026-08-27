"""Joint Gaussian distributions and the conditioning EKI is built from.

The layer represents a joint Gaussian over two blocks — in Ensemble Kalman
Inversion, the parameters and the predicted observations — and conditions it
on a noisy observation of the second block. Every conditioning path routes
through one algorithm, the whitened-SVD kernel described below, so nothing
here forms a matrix of the parameter or observation dimension.

============================ ==================================================
object                       represents
============================ ==================================================
:class:`Gaussian`            one Gaussian distribution: a mean vector and a
                             :class:`~pyeki.linalg.PSDLinOp` covariance
:class:`EnsembleJoint`       the joint Gaussian with an ensemble's empirical
                             moments, held as :math:`J` paired samples
:func:`gain_weights`         the Kalman-gain weights for a whitened residual
:func:`sqrt_transform`       the deterministic square-root update transform
============================ ==================================================

Conventions shared by everything in the module:

- **Samples are stored row-wise**: an ensemble is a ``(J, dim)`` array, one
  member per row. This is what a :func:`jax.vmap`-ed forward model produces
  and what the operator layer's batch contract treats as a batch of vectors,
  so ensembles flow between the two layers with no transposes.
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

Both classes are frozen-dataclass pytrees whose fields are their whole
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

The conditioning kernel. With :math:`A_v` the prediction anomalies,
:math:`W` a whitener of the noise covariance :math:`R`, and

.. math::

    S = \\frac{1}{\\sqrt{J-1}} A_v W^\\top \\in \\mathbb{R}^{J \\times N},
    \\qquad S = U \\Sigma V^\\top \\ \\text{(thin SVD)},

the Kalman gain :math:`K = \\widehat{C}_{uv}(\\widehat{C}_{vv} + R)^{-1}`
applied to a residual :math:`r` is a combination of the ensemble's own
:math:`u`-anomalies,

.. math::

    K r = \\frac{1}{\\sqrt{J-1}} A_u^\\top w, \\qquad
    w = U \\operatorname{diag}\\!\\Bigl(\\frac{\\sigma_i}{1+\\sigma_i^2}\\Bigr)
        V^\\top (W r),

and the deterministic update transforms the anomalies by
:math:`T = (I_J + S S^\\top)^{-1/2}`, built from the same decomposition.
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

from .linalg import PSDLinOp, PSDLowRank, dense_matvec, value_check
from .linalg.base import _broadcast_batch, _pytree_dataclass

__all__ = ["EnsembleJoint", "Gaussian", "gain_weights", "sqrt_transform"]


# ---------------------------------------------------------------------------
# the conditioning primitives
# ---------------------------------------------------------------------------


def gain_weights(s: Array, b: Array) -> Array:
    """Ensemble weights for a whitened residual: the shared conditioning core.

    A pure matrix function of its arguments — no divisor, no whitening and no
    randomness folded in. For the thin SVD :math:`s = U\\Sigma V^\\top`,

    .. math::

        \\texttt{gain\\_weights}(s, b)
        = U \\operatorname{diag}\\!\\Bigl(\\frac{\\sigma_i}{1+\\sigma_i^2}\\Bigr)
          V^\\top b
        = s\\,(s^\\top s + I_N)^{-1} b ,

    the second form showing that the result is a function of ``s`` alone,
    invariant to the SVD's sign and degenerate-rotation freedom.

    In conditioning, ``s`` is the scaled whitened anomaly matrix
    :math:`A_v W^\\top/\\sqrt{J-1}` and ``b`` a whitened residual
    :math:`W r`, and the returned weights give the gain applied to that
    residual as a combination of the ensemble's own anomalies,
    :math:`K r = A_u^\\top w/\\sqrt{J-1}`. The multipliers are bounded by
    :math:`\\sigma/(1+\\sigma^2) \\le 1/2` for every :math:`\\sigma \\ge 0`,
    so the gain cannot blow up however collapsed or ill-conditioned the
    ensemble becomes, and there is no regularization parameter to tune.

    Parameters
    ----------
    s
        Array of shape ``(J, N)``, exactly 2-D, both sizes at least 1. It
        plays the operator's role and carries no batch axes; a family of
        local analyses is a :func:`jax.vmap` over this function.
    b
        Array of shape ``(..., N)`` — whitened residuals along the trailing
        axis, any number of leading batch axes, carried through.

    Returns
    -------
    Array
        Shape ``(..., J)``, the batch axes of ``b`` preserved.

    Raises
    ------
    ValueError
        If ``s`` is not 2-D with positive sizes, or ``b``'s trailing axis is
        not ``N``. In debug mode, also if either is not finite.

    Notes
    -----
    One SVD per call: batch the residuals of an update into a single call
    rather than looping, since its :math:`J` per-member residuals are one
    ``(J, N)`` operand.

    Callers own the semantics of ``s`` and ``b``. The function cannot check
    that they are scaled, whitened and centered as conditioning requires,
    which is why the methods of :class:`EnsembleJoint` — where those
    conventions are enforced — are the default interface and this is the
    escape hatch.

    Differentiable wherever the singular values of ``s`` are distinct and
    nonzero. At exactly repeated or exactly zero singular values — an
    exactly collapsed ``s``, or the zero-padded columns a masked local
    analysis may produce — the SVD's gradient is ``nan`` even though this
    function is smooth there, equalling the rational form above. The
    float-generic degeneracy of mean-centering (:math:`\\sigma_{\\min} \\sim
    10^{-16}` when :math:`N \\ge J`) is not an exact tie and differentiates
    finitely.
    """
    s = jnp.asarray(s)
    if s.ndim != 2 or any(size < 1 for size in s.shape):
        raise ValueError(
            f"gain_weights: expected s of shape (J, N), exactly 2-D with both "
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
    :math:`s = U\\Sigma V^\\top` with :math:`\\rho = \\min(J, N)`,

    .. math::

        \\texttt{sqrt\\_transform}(s) = (I_J + s s^\\top)^{-1/2}
        = I_J + U\\bigl((I_\\rho + \\Sigma^2)^{-1/2} - I_\\rho\\bigr)U^\\top ,

    which is symmetric, and exact at every rank: the second form is how it
    is computed, and it is what this function returns for any correct thin
    SVD, elementwise.

    In conditioning, ``s`` is the scaled whitened anomaly matrix and the
    transformed anomalies :math:`T A_u` have empirical covariance exactly
    equal to the posterior covariance of the fitted joint Gaussian,

    .. math::

        \\frac{(T A_u)^\\top (T A_u)}{J-1}
        = \\widehat{C}_{uu} - K \\widehat{C}_{vu} ,

    an identity in exact arithmetic rather than an approximation in
    :math:`J`. Neither :math:`s s^\\top` nor :math:`s^\\top s` is formed.

    Parameters
    ----------
    s
        Array of shape ``(J, N)``, exactly 2-D, both sizes at least 1. No
        batch axes, and no centering requirement: on general ``s`` the result
        is still :math:`(I + ss^\\top)^{-1/2}`.

    Returns
    -------
    Array
        Shape ``(J, J)``, symmetric to round-off.

    Raises
    ------
    ValueError
        If ``s`` is not 2-D with positive sizes. In debug mode, also if it
        is not finite.

    Notes
    -----
    :math:`T\\mathbf{1} = \\mathbf{1}` — so transformed anomalies still sum
    to zero and a posterior ensemble's mean is not silently shifted —
    follows from :math:`\\mathbf{1}^\\top s = 0` and holds only for such
    mean-centered ``s``, which is the only case the conditioning kernel
    produces. On general ``s``, :math:`T\\mathbf{1}` is whatever that matrix
    makes it.

    Differentiability carries the caveat documented on
    :func:`gain_weights`; restoring gradients everywhere would need a
    Fréchet derivative of :math:`A \\mapsto A^{-1/2}`, materially more work
    than that function's rational form, and no conditioning path in this
    layer requires it.
    """
    s = jnp.asarray(s)
    if s.ndim != 2 or any(size < 1 for size in s.shape):
        raise ValueError(
            f"sqrt_transform: expected s of shape (J, N), exactly 2-D with both "
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
    :meth:`EnsembleJoint.condition` returns.

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

        The one-block counterpart of :class:`EnsembleJoint`, which fits a
        joint to two member-aligned blocks. Use it to read an ensemble's
        moments as a distribution — per-coordinate variances through
        ``cov.diag()``, fresh draws through :meth:`sample`.

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

        Notes
        -----
        The covariance is never formed as an :math:`n \\times n` matrix. The
        stored factor is the scaled anomaly matrix, so the operator *is* the
        empirical covariance exactly, and costs :math:`O(nJ)` to hold rather
        than :math:`O(n^2)`.

        Its rank is at most :math:`J-1`, so it is singular whenever
        :math:`J - 1 < n` — the usual ensemble regime — and
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
        samples = jnp.asarray(samples)
        if samples.ndim != 2:
            raise ValueError(
                f"Gaussian.from_samples: samples must be rank 2, got shape "
                f"{samples.shape}"
            )
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
    def n(self) -> int:
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
        x = _check_batched_operand(where, "x", x, self.n)
        _check_finite(where, "x", x)
        whitened = self.cov.whiten(x - self.mean)
        quadratic = jnp.sum(whitened * whitened, axis=-1)
        return -0.5 * (
            self.n * math.log(2.0 * math.pi) + self.cov.logdet() + quadratic
        )

    def __repr__(self) -> str:
        """The type name and dimension, as ``Gaussian(n=12)``; never raises."""
        try:
            base = f"Gaussian(n={self.n})"
            batch = self.batch_shape
        except Exception:
            return "<Gaussian (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base


# ---------------------------------------------------------------------------
# the joint Gaussian of an ensemble's empirical moments
# ---------------------------------------------------------------------------


@_pytree_dataclass
class EnsembleJoint:
    """The joint Gaussian determined by an ensemble's empirical moments.

    Held in sample form: the object EKI builds once per step, uses for one
    update, and discards. Its distribution is

    .. math::

        \\mathcal{N}\\!\\left(
        \\begin{pmatrix}\\bar u\\\\ \\bar v\\end{pmatrix},
        \\begin{pmatrix}\\widehat{C}_{uu} & \\widehat{C}_{uv}\\\\
        \\widehat{C}_{vu} & \\widehat{C}_{vv}\\end{pmatrix}\\right),
        \\qquad \\widehat{C}_{uv} = \\frac{A_u^\\top A_v}{J-1},

    a Gaussian *fit to* the members rather than the equal-weight point-mass
    distribution of the members themselves: every conditioning method below
    is exact Gaussian conditioning applied to this fitted Gaussian. The
    samples are the representation from which its moments are read, and no
    moment matrix is ever formed.

    All three conditioning methods condition on the same observation model,
    :math:`y = v + \\eta` with :math:`\\eta \\sim \\mathcal{N}(0, R)`, and
    share the trailing arguments ``y`` and ``noise_cov``. The two update
    methods return updated members and update :math:`u` only, since EKI
    re-evaluates the forward model to get the next step's :math:`v`;
    :meth:`condition` returns the same posterior as a distribution.

    Parameters
    ----------
    u_samples
        The block to be updated — in EKI, the parameters — a ``(J, P)``
        array, one member per row. Keyword-only.
    v_samples
        The observed block — in EKI, the predicted observations — a
        ``(J, N)`` array, member-aligned with ``u_samples``: row :math:`j`
        of each belongs to the same member. Keyword-only.

    Raises
    ------
    ValueError
        If either array is not rank 2, if they disagree on :math:`J`, or if
        :math:`J < 2` — a single sample has no anomalies. In debug mode,
        also if either is not finite.

    Notes
    -----
    Both fields are **keyword-only**. They are two arrays of the same rank
    whose sizes agree on the member axis, so a swap is a shape-valid mistake
    that no check can catch when :math:`P = N`: the update would be computed
    from the wrong blocks and return finite, plausible numbers. Naming them
    at the call site is the only thing that rules it out. The cost is that a
    family is built through a lambda rather than by mapping the constructor
    directly::

        jax.vmap(lambda u, v: EnsembleJoint(u_samples=u, v_samples=v))(U, V)

    The four array properties — the two means and the two anomaly matrices —
    raise ``ValueError`` on a vmapped family, as the methods do.

    Nothing is factorized at construction, deliberately: the SVD of
    :math:`S` depends on the noise operator, which arrives per update call
    and changes at every tempering step. Each method computes its SVD once,
    uses it, and discards it, never caching it on the instance — which under
    ``jit`` would write the cache to a temporary copy and silently
    re-factorize on every call.

    All three methods degrade gracefully when the prediction anomalies are
    zero: the updates return ``u_samples`` unchanged and :meth:`condition`
    returns the prior marginal's moments. A collapsed ensemble is a no-op,
    not ``nan``, for finite inputs.
    """

    u_samples: Array = field(kw_only=True)
    v_samples: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_field_rank("EnsembleJoint", "u_samples", self.u_samples, 2)
        _check_field_rank("EnsembleJoint", "v_samples", self.v_samples, 2)
        u_shape, v_shape = self.u_samples.shape, self.v_samples.shape
        if u_shape[0] != v_shape[0]:
            raise ValueError(
                f"EnsembleJoint: u_samples and v_samples must have the same "
                f"number of members, got shapes {u_shape} and {v_shape}"
            )
        if u_shape[0] < 2:
            raise ValueError(
                f"EnsembleJoint: at least 2 members are required, got "
                f"{u_shape[0]}. A single sample has no anomalies."
            )
        _check_finite("EnsembleJoint", "u_samples", self.u_samples)
        _check_finite("EnsembleJoint", "v_samples", self.v_samples)

    @property
    def n_members(self) -> int:
        """The ensemble size :math:`J`."""
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
            "EnsembleJoint",
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
        """The anomalies :math:`A_u = \\mathsf{U} - \\mathbf{1}\\bar u^\\top`.

        A ``(J, P)`` array.
        """
        _check_not_vmap_family(self, "u_anomalies")
        return _centered(self.u_samples)

    @property
    def v_anomalies(self) -> Array:
        """The anomalies :math:`A_v`, a ``(J, N)`` array."""
        _check_not_vmap_family(self, "v_anomalies")
        return _centered(self.v_samples)


    def pathwise_update(self, key, y, noise_cov) -> Array:
        """The stochastic (perturbed-observation) update: ``(J, P)``.

        Pathwise conditioning of the joint's own samples, one fresh
        perturbation per member:

        .. math::

            u_j' = u_j + \\frac{1}{\\sqrt{J-1}} A_u^\\top w_j, \\qquad
            w_j = \\texttt{gain\\_weights}(S, b_j), \\qquad
            b_j = W(y - v_j) - \\varepsilon_j,

        with :math:`\\varepsilon_j` the rows of the pinned draw
        ``jax.random.normal(key, (J, N))``. The perturbation enters *only* in
        whitened space: :math:`b_j` is the whitened residual against the
        perturbed prediction :math:`v_j + W^{-1}\\varepsilon_j`, so
        equivalently :math:`u_j' = u_j + K(y - v_j - \\eta_j)` with
        :math:`\\eta_j = W^{-1}\\varepsilon_j \\sim \\mathcal{N}(0, R)`.

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
            Shape ``(J, P)``, row :math:`j` updating member :math:`j`.

        Raises
        ------
        UnsupportedOpError
            If ``noise_cov`` does not support ``whiten``.
        TypeError
            If ``noise_cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
        ValueError
            If ``y`` is not ``(N,)``, ``noise_cov``'s side is not ``N``,
            either is a vmapped family, or — in debug mode — ``y`` or the
            returned value is not finite.

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
        draw and its single use. A caller who needs perturbed observations
        materialized is building a different algorithm, and should use
        :func:`gain_weights` directly, drawing its own :math:`\\varepsilon`
        and choosing one representation for it.

        The output's sample mean and sample covariance (divisor
        :math:`J - 1`) are unbiased estimators of the posterior moments
        :meth:`transform_update` represents exactly; the mean has variance
        :math:`K R K^\\top / J`. Individual members are not posterior draws:
        conditional on the ensemble, member :math:`j` is distributed
        :math:`\\mathcal{N}(u_j + K(y - v_j),\\, K R K^\\top)`.

        Precondition: ``noise_cov`` is nonsingular, which is ``whiten``'s own
        precondition. Nothing here can detect a singular one before the fact,
        so it surfaces as ``nan`` — or, in debug mode, as the result check
        that all three conditioning methods apply to what they return.
        """
        y, where = self._validate_call("pathwise_update", y, noise_cov)
        whitened_anomalies, whitened_residual = self._whiten_anomalies_and_residual(
            y, noise_cov
        )
        U, sigma, Vt = self._conditioning_svd(whitened_anomalies)
        eps = jax.random.normal(key, (self.n_members, self.v_dim))
        # W(y - v_j) - eps_j, the per-member residual assembled from the mean
        # residual and the anomalies rather than whitened a second time.
        b = whitened_residual - whitened_anomalies - eps
        members = self.u_samples + self._combine_anomalies(
            _weights_from_svd(U, sigma, Vt, b)
        )
        _check_finite(where, "the updated ensemble", members, cause=_SINGULAR_NOISE)
        return members

    def transform_update(self, y, noise_cov) -> Array:
        """The deterministic (square-root) update: ``(J, P)``.

        The moment-form posterior of the fitted joint Gaussian, returned in
        ensemble representation:

        .. math::

            u_j' = \\underbrace{\\bar u + \\frac{1}{\\sqrt{J-1}} A_u^\\top
            \\texttt{gain\\_weights}\\bigl(S, W(y - \\bar v)\\bigr)}
            _{m_{\\text{post}}} + (T A_u)_j, \\qquad
            T = \\texttt{sqrt\\_transform}(S).

        No randomness, and no ``key``. The returned ensemble is an *exact*
        representation of the moment posterior: its sample mean equals the
        posterior mean and its sample covariance (divisor :math:`J - 1`)
        equals the posterior covariance, both in exact arithmetic. Because
        :math:`T\\mathbf{1} = \\mathbf{1}`, the transformed anomalies remain
        centered, so the two summands really are the posterior mean and the
        posterior anomalies.

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
            Shape ``(J, P)``, row :math:`j` updating member :math:`j`.

        Raises
        ------
        UnsupportedOpError
            If ``noise_cov`` does not support ``whiten``.
        TypeError
            If ``noise_cov`` is not a :class:`~pyeki.linalg.PSDLinOp`.
        ValueError
            If ``y`` is not ``(N,)``, ``noise_cov``'s side is not ``N``,
            either is a vmapped family, or — in debug mode — ``y`` or the
            returned value is not finite.

        Notes
        -----
        Which update to use is the EKI layer's decision, not this layer's.
        This one replaces sampling noise with an exact transform of the
        anomalies; :meth:`pathwise_update` matches the posterior moments only
        in expectation, at the usual :math:`O(J^{-1/2})` rate.
        """
        y, where = self._validate_call("transform_update", y, noise_cov)
        mean, transform = self._posterior_mean_and_transform(y, noise_cov)
        # (J, J) @ (J, P): both operands are exactly 2-D, so this is the
        # plain matrix product, not a batch of vectors.
        members = mean + transform @ self.u_anomalies
        _check_finite(where, "the updated ensemble", members, cause=_SINGULAR_NOISE)
        return members

    def condition(self, y, noise_cov) -> Gaussian:
        """Moment-form conditioning: the posterior as a :class:`Gaussian`.

        The same posterior :meth:`transform_update` represents as members,
        for sampling at any size and for diagnostics:

        .. math::

            m_{\\text{post}} = \\bar u + \\frac{1}{\\sqrt{J-1}} A_u^\\top
            \\texttt{gain\\_weights}\\bigl(S, W(y - \\bar v)\\bigr),
            \\qquad
            C_{\\text{post}} = F F^\\top, \\quad
            F = \\frac{(T A_u)^\\top}{\\sqrt{J-1}} \\in
            \\mathbb{R}^{P \\times J}.

        The covariance is returned in structured form, a
        :class:`~pyeki.linalg.PSDLowRank` holding :math:`F` — never a dense
        :math:`P \\times P` matrix. Its exact relationship to the transform
        update is

        .. math::

            \\texttt{transform\\_update}(y, R)_j
            = m_{\\text{post}} + \\sqrt{J-1}\\, F_{\\cdot j}.

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
            either is a vmapped family, or — in debug mode — ``y`` or the
            returned value is not finite.

        Notes
        -----
        The returned covariance is honest about rank:
        :math:`\\operatorname{rank}(C_{\\text{post}}) \\le J - 1`, so it is
        singular whenever :math:`J - 1 < P`, the usual EKI regime. The
        posterior therefore supports :meth:`Gaussian.sample` — the factor is
        the stored representation — but not
        :meth:`Gaussian.log_density`, which raises
        :class:`~pyeki.linalg.UnsupportedOpError` from the covariance. In the
        usual regime that is forced: a rank-deficient Gaussian has no density
        on :math:`\\mathbb{R}^P`, and the capability system says so instead of
        returning ``-inf`` or ``nan``. When :math:`J - 1 \\ge P` the posterior
        is typically full rank and the density exists mathematically; the
        static capability choice still raises, and a caller wanting that
        density densifies the covariance deliberately.
        """
        y, where = self._validate_call("condition", y, noise_cov)
        mean, transform = self._posterior_mean_and_transform(y, noise_cov)
        factor = (transform @ self.u_anomalies).swapaxes(-1, -2) / math.sqrt(
            self.n_members - 1
        )
        # Both halves of the result, checked before the covariance and the
        # distribution are built, so the diagnosis names this call rather
        # than a constructor further down.
        _check_finite(where, "the posterior mean", mean, cause=_SINGULAR_NOISE)
        _check_finite(
            where, "the posterior covariance factor", factor, cause=_SINGULAR_NOISE
        )
        return Gaussian(mean, PSDLowRank(factor))

    def __repr__(self) -> str:
        """As ``EnsembleJoint(n_members=100, u_dim=12, v_dim=40)``; never raises."""
        try:
            base = (
                f"EnsembleJoint(n_members={self.n_members}, u_dim={self.u_dim}, "
                f"v_dim={self.v_dim})"
            )
            batch = self.batch_shape
        except Exception:
            return "<EnsembleJoint (unprintable leaves)>"
        return f"vmapped({base}, batch={batch})" if batch != () else base

    # -- private: call validation, and the whitened-SVD assembly --
    def _validate_call(self, method: str, y, noise_cov) -> tuple[Array, str]:
        """Run the shared checks of a conditioning call.

        Returns the validated ``y`` and the call's description, which the
        method reuses for its result check rather than rebuilding it.

        In the operator layer's order, with one departure the type system
        forces: the family guard, then ``noise_cov``'s type and family —
        which must precede the capability check, since ``supports`` cannot
        be consulted on an object not known to be an operator — then the
        capability check, then the remaining operand shapes, then, in debug
        mode, operand value checks. Result checks run last, in the method,
        once there is a result to check.
        """
        _check_not_vmap_family(self, method)
        where = f"{self!r}.{method}"
        if not isinstance(noise_cov, PSDLinOp):
            raise TypeError(
                f"{where}: noise_cov must be a pyeki.linalg.PSDLinOp, got "
                f"{type(noise_cov).__name__}"
            )
        if noise_cov.batch_shape != ():
            raise ValueError(
                f"{where}: noise_cov is a vmapped family with batch shape "
                f"{noise_cov.batch_shape}; apply a family of noise operators "
                f"with jax.vmap over the method, not directly."
            )
        _require_cov_ops(noise_cov, "whiten")
        _check_noise_cov_dim(where, noise_cov, self.v_dim)
        y = _check_unbatched_operand(where, "y", y, self.v_dim)
        _check_finite(where, "y", y)
        return y, where

    def _whiten_anomalies_and_residual(self, y, noise_cov):
        """Apply :math:`W` to the prediction anomalies and to the mean residual.

        Returns :math:`(W a_j` for each :math:`j`, :math:`W(y - \\bar v))`, of
        shapes ``(J, N)`` and ``(N,)``, from a single ``whiten`` call on the
        :math:`J + 1` stacked rows :math:`[A_v;\\, y - \\bar v]`. Every
        whitened quantity the kernel needs follows: :math:`S` is the first
        block over :math:`\\sqrt{J-1}`, the deterministic paths want the
        second as it stands, and the stochastic path's per-member residual is
        :math:`W(y - v_j) = W(y - \\bar v) - W a_j`. Whitening the anomalies
        and the residuals in two calls would instead cost :math:`2J`
        applications of :math:`W` in the stochastic update, which for a dense
        whitener is the dominant term.

        Centering and differencing happen *before* the whitener is applied,
        deliberately. Whitening is linear, so the orders agree in exact
        arithmetic — but they are not equally stable. Centering *whitened*
        predictions makes the cancellation ratio
        :math:`\\lVert W\\bar v\\rVert / \\lVert W a_j \\rVert` in place of
        :math:`\\lVert \\bar v\\rVert / \\lVert a_j \\rVert`, so the error
        grows with :math:`\\kappa(W) = \\sqrt{\\kappa(R)}` whenever the
        prediction mean is aligned with a precise direction of the noise.
        Whitening last costs nothing and avoids it.
        """
        anomalies = _centered(self.v_samples)
        residual = y - jnp.mean(self.v_samples, axis=-2)
        stacked = jnp.concatenate([anomalies, residual[None, :]], axis=-2)
        whitened = noise_cov.whiten(stacked)
        return whitened[..., :-1, :], whitened[..., -1, :]

    def _conditioning_svd(self, whitened_anomalies: Array):
        """The thin SVD of :math:`S = A_v W^\\top/\\sqrt{J-1}`, computed once."""
        return _thin_svd(whitened_anomalies / math.sqrt(self.n_members - 1))

    def _posterior_mean_and_transform(self, y, noise_cov) -> tuple[Array, Array]:
        """The posterior mean and the transform :math:`T`, from a single SVD.

        Shared by :meth:`transform_update` and :meth:`condition`, which are
        two representations of the same posterior, so computing it once makes
        them agree by construction.
        """
        whitened_anomalies, whitened_residual = self._whiten_anomalies_and_residual(
            y, noise_cov
        )
        U, sigma, Vt = self._conditioning_svd(whitened_anomalies)
        mean = self.u_mean + self._combine_anomalies(
            _weights_from_svd(U, sigma, Vt, whitened_residual)
        )
        return mean, _transform_from_svd(U, sigma, self.n_members)

    def _combine_anomalies(self, weights: Array) -> Array:
        """Apply the :math:`u`-anomalies to weights: :math:`A_u^\\top w/\\sqrt{J-1}`.

        Carries any batch axes of ``weights``, so one call handles both a
        single weight vector and the :math:`J` per-member vectors of a
        stochastic update.
        """
        anomalies = self.u_anomalies.swapaxes(-1, -2)  # (P, J)
        return dense_matvec(anomalies, weights) / math.sqrt(self.n_members - 1)


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
            f"batch shape is {batch}. Apply it under jax.vmap, member by member."
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
    """Deviations from the sample mean over the member axis, formed stably.

    Mathematically :math:`x - \\mathbf{1}\\bar x^\\top`, computed by removing
    the first member before averaging. Two things follow, neither of them true
    of a direct subtraction of :func:`jax.numpy.mean`:

    - **Identical members give exactly zero.** ``jnp.mean`` of :math:`J`
      bit-identical rows sums and divides, which does not in general return
      the value it was given, so a collapsed ensemble otherwise acquires
      spurious anomalies of about :math:`\\varepsilon\\lvert\\bar x\\rvert`.
      The gain amplifies those into a wrong, finite, ``nan``-free update once
      the members are large: at :math:`v \\equiv 6\\times10^{23}` the spurious
      update is of order 1 where the exact answer is no update at all.
    - **Cancellation is governed by the spread, not the magnitude.** The error
      is :math:`O(\\varepsilon \\max_j \\lvert x_j - x_0 \\rvert)` rather than
      :math:`O(\\varepsilon \\max_j \\lvert x_j \\rvert)`, which matters
      whenever the mean is large relative to the anomalies — a converged
      ensemble, late in a tempering run.
    """
    shifted = x - x[..., :1, :]
    return shifted - jnp.mean(shifted, axis=-2)


def _thin_svd(s: Array) -> tuple[Array, Array, Array]:
    """Thin SVD :math:`s = U \\Sigma V^\\top`, as ``(U, sigma, Vt)``.

    Shapes are ``(J, rho)``, ``(rho,)`` and ``(rho, N)`` for
    ``rho = min(J, N)``. The single SVD call site of the module: a class
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


def _transform_from_svd(U: Array, sigma: Array, n_members: int) -> Array:
    """Assemble :math:`T = I_J + U((I+\\Sigma^2)^{-1/2} - I)U^\\top`.

    The identity completion is what makes this exact at every rank: for a
    thin SVD the naive :math:`U(I+\\Sigma^2)^{-1/2}U^\\top` omits the identity
    on the orthogonal complement of :math:`U`'s columns and is simply wrong
    whenever :math:`\\rho < J`.

    :math:`T\\mathbf{1} = \\mathbf{1}` survives floating point for
    mean-centered input because the modifier decays *quadratically*: the
    numerically-zero singular value's column of :math:`U` need not be
    orthogonal to :math:`\\mathbf{1}`, but its modifier is
    :math:`O(\\sigma_i^2)`, so the induced mean shift is
    :math:`O((\\varepsilon\\sigma_{\\max})^2)` rather than
    :math:`O(\\varepsilon\\sigma_{\\max})`. Computed as written, the modifier
    is in fact exactly ``0.0`` once :math:`\\sigma_i^2` falls below the
    resolution of ``1.0``, which is the same bound reached the short way.
    """
    modifier = 1.0 / jnp.sqrt(1.0 + sigma**2) - 1.0
    # (J, rho) @ (rho, J): both operands are exactly 2-D, so this is the
    # plain matrix product, not a batch of vectors.
    return jnp.eye(n_members, dtype=U.dtype) + (U * modifier) @ U.swapaxes(-1, -2)

