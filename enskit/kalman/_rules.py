r"""The two shipped update rules and the particle updates they build."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..distribution import Ensemble
from . import _common as c

__all__ = ["SymmetricSquareRoot", "Matheron"]


@c.pytree_class(data=(), meta=())
class SymmetricSquareRoot:
    r"""The deterministic square-root rule (ETKF form): realize after condition.

    An :class:`UpdateRule`. For noisy given blocks, ``build`` returns
    ``approximation.square_root_map(given)``, a
    :class:`~enskit.distribution.SquareRootMap`. Called with :math:`y^*` it
    returns, for each target block :math:`x`,

    .. math::

        x_j' = m_x + F_x w + \sqrt{\delta}\, F_x T e_j \;\big[+\, L_x \eta^{(x)}_j\big],
        \qquad w = A^{-1} S\, W(y^* - m_c), \qquad T = A^{-1/2},

    with :math:`S = (WF_c)^\top`, :math:`A = I_k + SS^\top`, :math:`W` a
    whitener of the given blocks' noise, :math:`e_j` the :math:`j`-th unit
    vector, :math:`\delta` the approximation's divisor
    (:attr:`~enskit.distribution.EnsembleGaussian.divisor`, :math:`J - 1`
    from :func:`gaussian_approximation`), and the bracketed draw from a
    target's independent term :math:`D_x = L_xL_x^\top` present only when it
    has one. :math:`T` and the conditional anomalies do not depend on
    :math:`y^*` and are built once.

    When no given block has an independent term (exact values), the built
    update returns ``approximation.condition(values).realize_particles(key=key)``,
    which factorizes at every call and needs :math:`N \le J - 1` for given
    blocks of total dimension :math:`N`.

    For particles whose sample moments equal a linear-Gaussian joint's, and
    targets without independent terms, the updated particles' mean and
    covariance (divisor :math:`\delta`) equal the exact conditional's. The
    call needs a key only if a target block has an independent term, which is
    then sampled.

    Raises
    ------
    ValueError
        At build: if the particles are weighted, the approximation is not an
        :class:`~enskit.distribution.EnsembleGaussian` with the particles'
        count, its blocks are not the particles', the given blocks mix noisy
        and exact, exact values exceed :math:`J - 1` dimensions, or no target
        remains; in debug mode, if a particle is not finite or the
        approximation is not aligned with the particles.
    UnsupportedOpError
        At build, if a given block's term cannot ``whiten`` or a target's term
        cannot ``factor``.

    Notes
    -----
    Exact moments do not mean the right shape. On a nonlinear problem the
    square-root update keeps the particles' arrangement, rotated and scaled,
    and can leave a few particles carrying most of the spread; the stochastic
    rule :class:`Matheron` does not. "Symmetric" names the choice of square
    root, :math:`T = T^\top`, which is what preserves the mean.

    References
    ----------
    Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
    with the ensemble transform Kalman filter. Part I: Theoretical aspects.
    *Monthly Weather Review*, 129(3), 420–436.

    Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
    assimilation for spatiotemporal chaos: a local ensemble transform Kalman
    filter. *Physica D*, 230(1–2), 112–126.

    Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M. & Whitaker,
    J. S. (2003). Ensemble square root filters. *Monthly Weather Review*,
    131(7), 1485–1490.

    Wang, X., Bishop, C. H. & Julier, S. J. (2004). Which is better, an
    ensemble of positive–negative pairs or a centered spherical simplex
    ensemble? *Monthly Weather Review*, 132(7), 1590–1605.
    """

    def build(self, particles, approximation, given):
        """Build the update of ``particles`` conditioned on the blocks named ``given``."""
        where = "SymmetricSquareRoot.build"
        given, targets = c.check_build_arguments(where, particles, approximation, given)
        if not c.is_aligned_type(particles, approximation):
            raise ValueError(
                f"{where}: the approximation must be an EnsembleGaussian with the "
                f"particles' count ({particles.n_particles}), aligned with them, as "
                f"gaussian_approximation builds; got {approximation!r}. A modified "
                f"approximation (a plain Gaussian) needs a rule that does not rely "
                f"on alignment, such as Matheron."
            )
        noisy = [approximation.block_cov(n) is not None for n in given]
        if any(noisy) and not all(noisy):
            raise ValueError(
                f"{where}: the given blocks {given} mix noisy blocks (with an "
                f"independent term) and exact ones (without); condition on one kind "
                f"at a time"
            )
        if not any(noisy):
            N = sum(approximation.dims[n] for n in given)
            if N > particles.n_particles - 1:
                raise ValueError(
                    f"{where}: exact values of total dimension {N} need more than "
                    f"{N} particles; there are {particles.n_particles}. Pass the "
                    f"values' noise instead."
                )
            for name in targets:
                D = approximation.block_cov(name)
                if D is not None:
                    D._require("factor")
        else:
            for name in approximation.names:
                D = approximation.block_cov(name)
                if D is not None:
                    D._require("whiten" if name in given else "factor")
        c.check_particles_finite(where, particles)
        c.check_alignment(where, particles, approximation)
        if any(noisy):
            return approximation.square_root_map(given)
        return c.build(
            _ExactSquareRootUpdate,
            given=given,
            targets=targets,
            n_particles=particles.n_particles,
            approximation=approximation,
        )

    def __repr__(self) -> str:
        return "SymmetricSquareRoot()"


@c.pytree_class(data=(), meta=())
class Matheron:
    r"""The stochastic rule (perturbed values, Matheron's rule): transport the particles.

    An :class:`UpdateRule`. The built update moves each particle by

    .. math::

        x_j' = x_j + e_{x,j} + K\big(y^* - g_j - e_j\big),
        \qquad K = \operatorname{cov}(x, y)\operatorname{cov}(y, y)^{-1},

    for every target block :math:`x`, with :math:`g_j` the particle's given
    values, :math:`e_j \sim \mathcal N(0, R)` the given blocks' noise and
    :math:`e_{x,j} = L_x\eta^{(x)}_j` a draw from the target's independent term
    :math:`D_x = L_xL_x^\top` (zero when it has none). The gain is never
    formed: :math:`K r = F_x A^{-1} S\, W r` through the approximation's
    :class:`~enskit.distribution.MatheronMap`, and :math:`e_j` is drawn in
    whitened coordinates, :math:`We_j = \varepsilon_j` standard normal, so the
    noise covariances need only ``whiten``. It works with any approximation,
    by one of two paths chosen at build:

    - **aligned** (an :class:`~enskit.distribution.EnsembleGaussian` with the
      particles' count): the whitened residuals are read off the
      approximation's factor,
      :math:`W(y^* - g_j) = W(y^* - m_c) - \sqrt{\delta}\, S_{j\cdot}^\top`,
      with :math:`\delta` the approximation's divisor
      (:meth:`~enskit.distribution.MatheronMap.particle_coefficients`), for
      :math:`J + 1` whitened vectors per update;
    - **general** (any other :class:`~enskit.distribution.Gaussian`, such as a
      hybrid covariance): the map is applied to the particles, whitening each
      residual, for :math:`k + J` whitened vectors with :math:`k` the
      approximation's latent width.

    The two agree to round-off for the same key. The call's key is required
    and is split as ``k_targets, k_noise = jax.random.split(key)``: the target
    draws are ``normal(keys[i], (J, w_x))`` with
    ``keys = jax.random.split(k_targets, n_T)`` over the :math:`n_T` targets
    with a term, in block order, and the given blocks' noise is the map's
    ``normal(k_noise, (J, N))``.

    For a linear-Gaussian problem and an approximation with divisor
    :math:`J - 1`, as :func:`gaussian_approximation` builds, the updated
    particles' sample mean and covariance (divisor :math:`J - 1`) are
    unbiased, over the key, for the conditional moments of the fitted
    Gaussian; their sampling error is of
    order :math:`1/\sqrt{J}`.

    Raises
    ------
    ValueError
        At build: if the particles are weighted, a given block has no
        independent term (the perturbations are draws of it; for exact values
        use :class:`SymmetricSquareRoot`), the approximation's blocks are not
        the particles', or no target remains; in debug mode, if a particle is
        not finite or an aligned-type approximation is not aligned. At call:
        if no key is given.
    UnsupportedOpError
        At build, if a given block's term cannot ``whiten`` or a target's term
        cannot ``factor``.

    References
    ----------
    Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in
    the ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.

    Houtekamer, P. L. & Mitchell, H. L. (1998). Data assimilation using an
    ensemble Kalman filter technique. *Monthly Weather Review*, 126(3),
    796–811.

    Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
    M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
    Machine Learning Research*, 22(105), 1–47.
    """

    def build(self, particles, approximation, given):
        """Build the update of ``particles`` conditioned on the blocks named ``given``."""
        where = "Matheron.build"
        given, targets = c.check_build_arguments(where, particles, approximation, given)
        exact = tuple(n for n in given if approximation.block_cov(n) is None)
        if exact:
            raise ValueError(
                f"{where}: given blocks {exact} have no independent term. The "
                f"stochastic rule's perturbations are draws of the given blocks' "
                f"noise: pass it (noise= in update, or add_noise), or use "
                f"SymmetricSquareRoot for exact values."
            )
        for name in approximation.names:
            D = approximation.block_cov(name)
            if D is not None:
                D._require("whiten" if name in given else "factor")
        aligned = c.is_aligned_type(particles, approximation)
        c.check_particles_finite(where, particles)
        if aligned:
            c.check_alignment(where, particles, approximation)
        cmap = approximation.conditional_map(given)
        target_terms = tuple(approximation.block_cov(n) for n in targets)
        return c.build(
            _MatheronUpdate,
            given=given,
            targets=targets,
            n_particles=particles.n_particles,
            aligned=aligned,
            cmap=cmap,
            particles=particles,
            target_rows=tuple(approximation.factor(n) for n in targets),
            target_factors=tuple(None if D is None else D.factor() for D in target_terms),
        )

    def __repr__(self) -> str:
        return "Matheron()"


# ---------------------------------------------------------------------------
# private: the built updates
# ---------------------------------------------------------------------------


@c.pytree_class(
    data=("cmap", "particles", "target_rows", "target_factors"),
    meta=("given", "targets", "n_particles", "aligned"),
)
class _MatheronUpdate:
    """The particle update :class:`Matheron` builds."""

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return jnp.broadcast_shapes(self.cmap.batch_shape, self.particles.batch_shape)

    def __call__(self, values=None, /, *, key=None, **block_values) -> Ensemble:
        where = f"{self!r}.__call__"
        c.guard(self, where)
        c.check_key(
            where,
            key,
            required=True,
            why=": the stochastic rule draws the given blocks' noise",
        )
        k_targets, k_noise = jax.random.split(key)
        J, P = self.n_particles, self.particles
        dtype = P[P.names[0]].dtype
        sampled = tuple(L for L in self.target_factors if L is not None)
        keys = jax.random.split(k_targets, len(sampled)) if sampled else None
        draws, i = [], 0
        for L in self.target_factors:
            if L is None:
                draws.append(None)
            else:
                draws.append(L.matvec(jax.random.normal(keys[i], (J, L.shape[1]), dtype)))
                i += 1
        if self.aligned:
            w = self.cmap.particle_coefficients(values, key=k_noise, **block_values)
            blocks = []
            for name, F, e in zip(self.targets, self.target_rows, draws, strict=True):
                x = P[name]
                if F is not None:
                    x = x + F.matvec(w)
                if e is not None:
                    x = x + e
                blocks.append(x)
        else:
            moved = self.cmap(P, values, key=k_noise, **block_values)
            blocks = []
            for name, e in zip(self.targets, draws, strict=True):
                x = moved[name]
                blocks.append(x if e is None else x + e)
        for name, x in zip(self.targets, blocks, strict=True):
            if x.dtype != dtype:
                raise TypeError(
                    f"{where}: updated block {name!r} has dtype {x.dtype}, the "
                    f"particles {dtype}: an operator of the approximation (a "
                    f"target's term or factor row) is of another dtype"
                )
            c.result_check(where, f"updated block {name!r}", x)
        return c.build(
            Ensemble, names=self.targets, _blocks=tuple(blocks), _log_weights=None
        )

    def __repr__(self) -> str:
        return c.safe_repr(
            lambda: (
                f"MatheronUpdate(given={self.given}, targets={self.targets}, "
                f"n_particles={self.n_particles}, aligned={self.aligned})"
            ),
            "MatheronUpdate",
            lambda: self.batch_shape,
        )


@c.pytree_class(data=("approximation",), meta=("given", "targets", "n_particles"))
class _ExactSquareRootUpdate:
    """The particle update :class:`SymmetricSquareRoot` builds for exact values."""

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.approximation.batch_shape

    def __call__(self, values=None, /, *, key=None, **block_values) -> Ensemble:
        where = f"{self!r}.__call__"
        c.guard(self, where)
        vals = c.merge_blocks(where, values, block_values)
        for name in vals:
            if name not in self.given:
                raise KeyError(
                    f"{where}: {name!r} is not a given block; the given blocks are "
                    f"{self.given}"
                )
        missing = tuple(n for n in self.given if n not in vals)
        if missing:
            raise ValueError(
                f"{where}: values are missing for the given blocks {missing}"
            )
        return self.approximation.condition(vals).realize_particles(key=key)

    def __repr__(self) -> str:
        return c.safe_repr(
            lambda: (
                f"ExactSquareRootUpdate(given={self.given}, targets={self.targets}, "
                f"n_particles={self.n_particles})"
            ),
            "ExactSquareRootUpdate",
            lambda: self.batch_shape,
        )
