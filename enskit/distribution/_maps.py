r"""Conditional maps: the :class:`ConditionalMap` protocol and its two shipped forms."""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array

from ..linalg import IdentityPlusGram, LinOp, PSDLinOp, static_field
from . import _common as c
from ._common import distribution_class
from ._ensemble import Ensemble

__all__ = ["ConditionalMap", "MatheronMap", "SquareRootMap"]


@runtime_checkable
class ConditionalMap(Protocol):
    r"""A pointwise map carrying samples of a joint to samples of a conditional.

    Built from a joint distribution and the *names* of the given blocks;
    called with samples of the joint (an :class:`Ensemble`, whose particles
    are such samples) and a value :math:`y^*` for the given blocks. Returns
    samples whose distribution approximates, or for a Gaussian joint equals,
    the conditional of the targets at :math:`y^*`. :class:`MatheronMap` is
    the affine instance; a nonlinear triangular transport map fitted to
    particles is the same protocol with a different fitting procedure.

    An implementation must satisfy:

    1. ``given`` and ``targets`` are disjoint tuples of block names, fixed at
       build.
    2. Values are given as a positional mapping, keywords, or both: one
       exactly ``(d_c,)`` array for each given block and no other; anything
       else raises.
    3. ``samples`` contains every given and every target block. The result
       has every target block replaced, the given blocks dropped, any other
       block passed through unchanged and in its position, the weights kept,
       and the same number of particles.
    4. It is **pointwise**: the image of sample :math:`j` depends only on
       sample :math:`j`, :math:`y^*`, and row :math:`j` of whatever the map
       draws from its key. Without a key, permuting the samples permutes the
       result exactly. A map that couples samples, as
       :class:`SquareRootMap` does, is not a ``ConditionalMap``.
    5. A key is consumed whole, and the map is deterministic given it.
    """

    given: tuple[str, ...]
    targets: tuple[str, ...]

    def __call__(
        self, samples: Ensemble, values=None, /, *, key=None, **block_values
    ) -> Ensemble:
        """Transport ``samples`` of the joint to the conditional at ``values``."""
        ...


@distribution_class
class MatheronMap:
    r"""The exact conditional of a :class:`Gaussian` as an affine transport map.

    .. math::

        T_{y^*}(x, y) = x + K(y^* - y) = x + F_x\, A^{-1} S\, W(y^* - y),
        \qquad K = \operatorname{cov}(x, y)\operatorname{cov}(y, y)^{-1},

    for every target block :math:`x` with a factor row, with
    :math:`S = (W F_c)^\top`, :math:`A = I_k + SS^\top` and :math:`W` the
    whitener of the given blocks' independent terms; targets without a
    factor row pass through unchanged, their gain being zero. If
    :math:`(x, y)` is a sample of the Gaussian, :math:`T_{y^*}(x, y)` is a
    sample of its conditional at :math:`y^*`, whose covariance is

    .. math::

        \operatorname{cov} T = C_{xx} - KC_{yx} - C_{xy}K^\top + KC_{yy}K^\top
        = C_{xx} - K C_{yx} .

    Built by :meth:`Gaussian.conditional_map`; not constructed directly. It
    holds the :class:`~enskit.linalg.IdentityPlusGram` of :math:`S` (none at
    :math:`k = 0`, where the map moves nothing), the given blocks' terms and
    means, and the targets' factor rows. A :class:`ConditionalMap`. Its
    static attributes are ``given`` and ``targets``, the given and target
    blocks in block order; ``latent_dim``, the latent width :math:`k` of the
    Gaussian it was built from; and ``n_particles`` and ``divisor``,
    :math:`J` and the stored divisor :math:`\delta` when built from an
    :class:`EnsembleGaussian`, which enable :meth:`particle_coefficients`,
    and ``None`` otherwise.

    Notes
    -----
    Matheron's rule (Journel & Huijbregts, 1978), known in machine learning
    as pathwise conditioning (Wilson et al., 2021); applied to ensemble
    particles with perturbed values it is the stochastic ensemble Kalman
    update (Burgers et al., 1998).

    Applied to particles fitted by the Gaussian with the divisor
    :math:`J - 1`, rather than to its own samples, the output's sample mean
    and covariance (divisor :math:`J - 1`) are unbiased for the
    conditional's moments under the noise draw, but individual images are
    not conditional samples: given the particles, image :math:`j` is
    distributed
    :math:`\mathcal N\big(x_j + K(y^* - y_j),\ K D_c K^\top\big)`.

    The map never accepts or returns its perturbations. A perturbation used
    through the whitened shortcut must never also be pushed through
    ``factor()`` in the same transport: :math:`WL` has orthonormal rows but
    is not the identity, so mixing the two representations corrupts the
    joint law of the result while every marginal statistic still looks
    right.

    References
    ----------
    Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in
    the ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.

    Journel, A. G. & Huijbregts, C. J. (1978). *Mining Geostatistics*.
    Academic Press.

    Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
    M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
    Machine Learning Research*, 22(105), 1–47.
    """

    given: tuple[str, ...] = static_field()
    targets: tuple[str, ...] = static_field()
    latent_dim: int = static_field()
    n_particles: int | None = static_field()
    divisor: int | None = static_field()
    target_dims: tuple[int, ...] = static_field()
    gram: IdentityPlusGram | None
    _given_covs: tuple[PSDLinOp, ...]
    _given_means: tuple[Array, ...]
    _target_rows: tuple[LinOp | None, ...]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a map built directly."""
        shapes = [tuple(m.shape[:-1]) for m in self._given_means]
        shapes += [D.batch_shape for D in self._given_covs]
        shapes += [F.batch_shape for F in self._target_rows if F is not None]
        if self.gram is not None:
            shapes.append(self.gram.batch_shape)
        return c.broadcast_batch("MatheronMap", *shapes)

    def __call__(self, samples, values=None, /, *, key=None, **block_values) -> Ensemble:
        r"""Apply the map to every sample.

        For each sample :math:`j` it whitens the residual
        :math:`y^*_c - y_{j,c}` of each given block (differenced before
        whitening), forms :math:`b_j = W(y^* - y_j)`, and, when ``key`` is
        given, subtracts :math:`\varepsilon_j`, row :math:`j` of one pinned
        ``normal(key, (n, N))`` draw, its columns the given blocks' whitened
        coordinates in block order:

        .. math::

            W\big(y^* - (y_j + e_j)\big) = W(y^* - y_j) - \varepsilon_j, \qquad
            e_j = W^{-1}\varepsilon_j \sim \mathcal N(0, D_c) .

        One ``solve_factor`` call on the batch gives the coefficients
        :math:`w_j = A^{-1}Sb_j`, and :math:`x_j' = x_j + F_x w_j`.

        Parameters
        ----------
        samples : Ensemble
            Must contain every given and every target block, of the map's
            dtype; other blocks pass through.
        values : Mapping[str, Array], optional
            Positional-only. :math:`y^*` for each given block, exactly
            ``(d_c,)``.
        key
            Keyword-only typed key. Its presence says what the samples hold:
            without one, each sample's given blocks are taken to include
            their independent noise (joint samples); with one, they are taken
            to be noise-free, and the map draws the noise, in whitened
            coordinates, so the terms need only ``whiten``.
        **block_values : Array
            The values, as keywords.

        Returns
        -------
        Ensemble
            ``samples`` with every target block moved, the given blocks
            dropped, other blocks unchanged and the weights kept.

        Raises
        ------
        KeyError
            If a value names a block that is not given.
        ValueError
            If a given block's value is missing or not ``(d_c,)``, or
            ``samples`` lacks a given or target block or has one of the wrong
            dimension. In debug mode, also if a value or sample is not
            finite, a whitened residual is not finite, or a moved block is
            not finite.
        TypeError
            If ``samples`` is not an :class:`Ensemble` of the map's dtype,
            ``key`` is not a typed key, or a block is given twice.
        """
        where = f"{self!r}.__call__"
        w, samples = self._coefficients(where, samples, values, block_values, key)
        moved = {}
        for name, F in zip(self.targets, self._target_rows, strict=True):
            if F is not None and w is not None:
                moved[name] = samples[name] + F.matvec(w)
                c.result_check(where, f"moved block {name!r}", moved[name])
        pairs = [
            (n, moved.get(n, a))
            for n, a in zip(samples.names, samples._blocks, strict=True)
            if n not in self.given
        ]
        return c.build(
            Ensemble,
            names=tuple(n for n, _ in pairs),
            _blocks=tuple(a for _, a in pairs),
            _log_weights=samples._log_weights,
        )

    def coefficients(self, samples, values=None, /, *, key=None, **block_values) -> Array:
        r"""The latent coefficients :math:`w_j` of the map, ``(n, k)``, unapplied.

        :math:`K(y^* - y_j - e_j) = F_x w_j` for every target :math:`x`, with
        :math:`w_j = A^{-1}SW(y^* - y_j - e_j)` computed as in
        :meth:`__call__`, which takes the same arguments and raises the same
        errors. The escape hatch for code that works in latent space. At
        :math:`k = 0` the result is an ``(n, 0)`` array of zeros.
        """
        where = f"{self!r}.coefficients"
        w, samples = self._coefficients(where, samples, values, block_values, key)
        if w is None:
            return jnp.zeros((samples.n_particles, 0), self._dtype)
        c.result_check(where, "coefficients", w)
        return w

    def particle_coefficients(self, values=None, /, *, key, **block_values) -> Array:
        r"""The coefficients for the particles of the Gaussian the map came from.

        Reads the particles' whitened residuals off the whitened factor
        instead of whitening each particle:

        .. math::

            b_j = W(y^* - m_c) - \sqrt{\delta}\,S_{j\cdot}^\top - \varepsilon_j,
            \qquad w_j = A^{-1} S\, b_j ,

        with :math:`\delta` the Gaussian's divisor, so that
        :math:`\sqrt{\delta}\,S_{j\cdot}^\top` is particle :math:`j`'s
        whitened given anomaly, and :math:`\varepsilon` the same pinned
        ``normal(key, (J, N))`` draw as :meth:`__call__`. It whitens one
        vector per given block, so transporting :math:`J` particles costs
        :math:`J + 1` whitened vectors with the map's build, against
        :math:`2J` through :meth:`__call__`. It agrees with
        ``coefficients(g.realize_particles(exclude_block_covs=given), values,
        key=key)`` to round-off, not bit-exactly.

        The caller applies the coefficients to its own particles,
        :math:`x_j + F_x w_j`, so particles whose given anomalies are exactly
        zero come back bit-identical.

        Parameters
        ----------
        values : Mapping[str, Array], optional
            Positional-only. :math:`y^*` for each given block, exactly
            ``(d_c,)``.
        key
            Keyword-only typed key, required: the particles' given blocks are
            noise-free by construction, and without the draw the sample
            covariance would fall short of the conditional's by
            :math:`KD_cK^\top`.
        **block_values : Array
            The values, as keywords.

        Returns
        -------
        Array
            The ``(J, k)`` coefficients, :math:`k = J`.

        Raises
        ------
        ValueError
            If ``key`` is ``None``, the map was built from a plain
            :class:`Gaussian`, or a value is missing or not ``(d_c,)``. In
            debug mode, as for :meth:`__call__`.
        KeyError
            If a value names a block that is not given.
        TypeError
            If ``key`` is not a typed key, or a block is given twice.
        """
        where = f"{self!r}.particle_coefficients"
        c.guard(self, "particle_coefficients")
        vals = _value_names(where, self.given, values, block_values)
        c.check_key(
            where,
            key,
            required=True,
            why=(
                ": the particles' given blocks are noise-free, so the map draws "
                "their noise"
            ),
        )
        if self.n_particles is None:
            raise ValueError(
                f"{where}: this map was built from a plain Gaussian, whose latent "
                f"coordinates are not particles; call the map on samples instead"
            )
        y = _values(where, self.given, self._given_means, vals)
        rw = _whitened_residuals(
            where, self.given, self._given_covs, self._given_means, y
        )
        J = self.n_particles
        eps = jax.random.normal(key, (J, rw.shape[-1]), self._dtype)
        b = rw - math.sqrt(self.divisor) * self.gram.S - eps
        w = self.gram.solve_factor(b)
        c.result_check(where, "coefficients", w)
        return w

    def __repr__(self) -> str:
        """Type name and static sizes, never array contents; never raises."""
        return c.safe_repr(
            lambda: (
                f"MatheronMap(given={self.given}, targets={self.targets}, "
                f"latent_dim={self.latent_dim})"
            ),
            "MatheronMap",
            lambda: self.batch_shape,
        )

    # -- private -----------------------------------------------------------------
    @property
    def _dtype(self):
        return self._given_means[0].dtype

    def _coefficients(self, where, samples, values, block_values, key):
        """Validate a call and compute ``(w, samples)``; ``w`` is None at k = 0."""
        c.guard(self, where.rsplit(".", 1)[-1])
        if not isinstance(samples, Ensemble):
            raise TypeError(
                f"{where}: samples must be an Ensemble, got {type(samples).__name__}"
            )
        c.guard(samples, "the samples of a map")
        vals = _value_names(where, self.given, values, block_values)
        c.check_key(where, key, required=False)
        missing = tuple(n for n in self.given + self.targets if n not in samples.names)
        if missing:
            raise ValueError(
                f"{where}: samples must contain every given and target block; "
                f"missing {missing}"
            )
        y = _values(where, self.given, self._given_means, vals, finite=False)
        dims = dict(
            zip(self.given, (m.shape[-1] for m in self._given_means), strict=True)
        )
        dims.update(zip(self.targets, self.target_dims, strict=True))
        for name, d in dims.items():
            if samples.dims[name] != d:
                raise ValueError(
                    f"{where}: samples block {name!r} has dimension "
                    f"{samples.dims[name]}, the map's block has {d}"
                )
        if samples._dtype != self._dtype:
            raise TypeError(
                f"{where}: samples have dtype {samples._dtype}, the map has "
                f"{self._dtype}"
            )
        for name in self.given:
            c.check_finite(where, f"the value of block {name!r}", y[name])
        for name in dims:
            c.check_finite(where, f"samples block {name!r}", samples[name])
        if self.gram is None:
            return None, samples
        b = jnp.concatenate(
            [
                _whiten(where, n, D, y[n] - samples[n])
                for n, D in zip(self.given, self._given_covs, strict=True)
            ],
            axis=-1,
        )
        if key is not None:
            b = b - jax.random.normal(key, b.shape, self._dtype)
        return self.gram.solve_factor(b), samples


@distribution_class
class SquareRootMap:
    r"""Moves the particle set of the :class:`EnsembleGaussian` it was built from.

    Built by :meth:`EnsembleGaussian.square_root_map`. Called with
    :math:`y^*`, it returns, for each target block :math:`x`,

    .. math::

        x_j' = m_x + F_x\, A^{-1}S\,W(y^* - m_c) + \sqrt{\delta}\,F_x T e_j
        \;\big[+\, L_x\,\eta^{(x)}_j\big], \qquad j = 1, \dots, J,

    with :math:`A = I_k + SS^\top`, :math:`T = A^{-1/2}` and :math:`\delta`
    the Gaussian's divisor (:attr:`EnsembleGaussian.divisor`): the value of
    ``g.condition(values).realize_particles(key=key)``, with everything that
    does not depend on :math:`y^*` computed once, at build. A call whitens
    one vector per given block. A target with no factor row realizes as its
    mean; a target with an independent term :math:`D_x = L_xL_x^\top` is
    sampled, with the draw of :meth:`EnsembleGaussian.realize_particles`
    over the targets.

    It has no samples argument: it acts through each particle's latent
    coordinate :math:`e_j`, so it is defined only for the particles the
    Gaussian was built from. It is therefore not a :class:`ConditionalMap`.
    Its static attributes are ``given`` and ``targets``, the given and target
    blocks in block order, and ``n_particles``, :math:`J`.

    Notes
    -----
    This is the symmetric square-root update of the ensemble transform
    Kalman filter (Bishop et al., 2001; Hunt et al., 2007), written in the
    latent coordinates of the projected Gaussian. :math:`T` is symmetric,
    which is what makes the update preserve the particles' mean (Wang et
    al., 2004). For a linear-Gaussian joint whose particles' moments equal
    the joint's, and whose targets have no independent terms, the output's
    sample mean and covariance (divisor :math:`\delta`) equal the exact
    conditional's.

    References
    ----------
    Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive
    sampling with the ensemble transform Kalman filter. Part I: Theoretical
    aspects. *Monthly Weather Review*, 129(3), 420–436.

    Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
    assimilation for spatiotemporal chaos: a local ensemble transform Kalman
    filter. *Physica D*, 230(1–2), 112–126.

    Wang, X., Bishop, C. H. & Julier, S. J. (2004). Which is better, an
    ensemble of positive–negative pairs or a centered spherical simplex
    ensemble? *Monthly Weather Review*, 132(7), 1590–1605.
    """

    given: tuple[str, ...] = static_field()
    targets: tuple[str, ...] = static_field()
    n_particles: int = static_field()
    gram: IdentityPlusGram
    _given_covs: tuple[PSDLinOp, ...]
    _given_means: tuple[Array, ...]
    _target_means: tuple[Array, ...]
    _target_rows: tuple[LinOp | None, ...]
    _target_anomalies: tuple[Array | None, ...]
    _target_covs: tuple[PSDLinOp | None, ...]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a map built directly."""
        shapes = [self.gram.batch_shape]
        shapes += [tuple(m.shape[:-1]) for m in self._given_means + self._target_means]
        shapes += [D.batch_shape for D in self._given_covs]
        shapes += [
            op.batch_shape
            for op in self._target_rows + self._target_covs
            if op is not None
        ]
        shapes += [tuple(a.shape[:-2]) for a in self._target_anomalies if a is not None]
        return c.broadcast_batch("SquareRootMap", *shapes)

    def __call__(self, values=None, /, *, key=None, **block_values) -> Ensemble:
        """The updated particles over the target blocks.

        Parameters
        ----------
        values : Mapping[str, Array], optional
            Positional-only. :math:`y^*` for each given block, exactly
            ``(d_c,)``.
        key
            Keyword-only typed key, required exactly when a target block has
            an independent term, which is then sampled; otherwise ignored.
        **block_values : Array
            The values, as keywords.

        Returns
        -------
        Ensemble
            Unweighted, over the targets in block order.

        Raises
        ------
        KeyError
            If a value names a block that is not given.
        ValueError
            If a value is missing or not ``(d_c,)``, or a needed key is
            missing. In debug mode, also if a value, a whitened residual or
            an updated particle is not finite.
        TypeError
            If ``key`` is not a typed key, or a block is given twice.
        """
        where = f"{self!r}.__call__"
        c.guard(self, "__call__")
        vals = _value_names(where, self.given, values, block_values)
        sampled = tuple(
            n
            for n, D in zip(self.targets, self._target_covs, strict=True)
            if D is not None
        )
        c.check_key(
            where, key, required=bool(sampled),
            why=f": target blocks {sampled} have independent terms to sample",
        )
        y = _values(where, self.given, self._given_means, vals)
        rw = _whitened_residuals(
            where, self.given, self._given_covs, self._given_means, y
        )
        w = self.gram.solve_factor(rw)
        keys = jax.random.split(key, len(sampled)) if sampled else None
        J, dtype = self.n_particles, self._dtype
        blocks, i = [], 0
        for name, m, F, a, D in zip(
            self.targets,
            self._target_means,
            self._target_rows,
            self._target_anomalies,
            self._target_covs,
            strict=True,
        ):
            if F is not None:
                x = (m + F.matvec(w)) + a
            else:
                x = jnp.broadcast_to(m, (J, m.shape[-1]))
            if D is not None:
                L = D.factor()
                x = x + L.matvec(jax.random.normal(keys[i], (J, L.shape[1]), dtype))
                i += 1
            c.result_check(where, f"updated block {name!r}", x)
            blocks.append(x)
        return c.build(
            Ensemble, names=self.targets, _blocks=tuple(blocks), _log_weights=None
        )

    def __repr__(self) -> str:
        """Type name and static sizes, never array contents; never raises."""
        return c.safe_repr(
            lambda: (
                f"SquareRootMap(given={self.given}, targets={self.targets}, "
                f"n_particles={self.n_particles})"
            ),
            "SquareRootMap",
            lambda: self.batch_shape,
        )

    @property
    def _dtype(self):
        return self._given_means[0].dtype


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _value_names(where: str, given: tuple[str, ...], values, block_values) -> dict:
    """Merge the values and check they name exactly the given blocks."""
    vals = c.merge_blocks(where, values, block_values)
    for name in vals:
        if name not in given:
            raise KeyError(
                f"{where}: {name!r} is not a given block; the given blocks are {given}"
            )
    missing = tuple(n for n in given if n not in vals)
    if missing:
        raise ValueError(f"{where}: values are missing for the given blocks {missing}")
    return vals


def _values(where: str, given, means, vals, *, finite: bool = True) -> dict:
    """Check each value's core shape, then (if ``finite``) its finiteness."""
    dtype = means[0].dtype
    y = {
        n: c.check_value(where, n, vals[n], m.shape[-1], dtype)
        for n, m in zip(given, means, strict=True)
    }
    if finite:
        for name in given:
            c.check_finite(where, f"the value of block {name!r}", y[name])
    return y


def _whiten(where: str, name: str, D: PSDLinOp, r: Array) -> Array:
    """Whiten one given block's residuals, then run the whitening check."""
    out = D.whiten(r)
    c.whitening_check(where, name, out)
    return out


def _whitened_residuals(where, given, covs, means, y) -> Array:
    """``W(y* - m_c)``, one whitened vector per given block, concatenated."""
    return jnp.concatenate(
        [
            _whiten(where, n, D, y[n] - m)
            for n, D, m in zip(given, covs, means, strict=True)
        ]
    )
