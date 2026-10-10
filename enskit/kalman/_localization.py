r"""Domain localization: the geometry, the taper, and the localized rule."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..distribution import Ensemble
from ..linalg import (
    Identity,
    IdentityPlusGram,
    PSDDiagonal,
    PSDScaled,
    dense_matvec,
    value_check,
)
from . import _common as c
from ._rules import Matheron, SymmetricSquareRoot

__all__ = ["LocalizedUpdateRule", "DomainLocalization", "gaspari_cohn"]

#: What the whitening check says about the likely cause.
_WHITENING_CAUSE = (
    "The likeliest cause is a noise variance that is zero, or values so large "
    "relative to the noise that whitening overflows."
)


@c.pytree_class(data=("update_rule", "localization"), meta=())
class LocalizedUpdateRule:
    r"""Domain localization around :class:`SymmetricSquareRoot` or :class:`Matheron`.

    An :class:`UpdateRule`. The built update replaces the wrapped rule's one
    global update by one local update per coordinate :math:`p` of every
    located target block: the wrapped rule's update of that coordinate given
    only its neighborhood :math:`\mathcal N_p = (i_1, \dots, i_K)` of given
    coordinates, with noise variances :math:`r_{i_k} / \rho_{pk}` in place of
    :math:`r_{i_k}`, from ``localization``. With
    :math:`S = (WF_c)^\top \in \mathbb R^{J \times N}` the whitened factor of
    the given blocks and :math:`f_p \in \mathbb R^J` row :math:`p` of the
    target's factor row, the local problem's whitened factor is

    .. math::

        S_p = \Big[\sqrt{\rho_{p1}}\, S_{\cdot i_1}\ \cdots\
              \sqrt{\rho_{pK}}\, S_{\cdot i_K}\Big],
        \qquad A_p = I_J + S_p S_p^\top ,

    and particle :math:`j`'s coordinate :math:`p` is updated by

    .. math::

        x'_{jp} &= m_{x,p} + f_p^\top A_p^{-1} S_p\, b_p
                   + \sqrt{\delta}\,\big(A_p^{-1/2} f_p\big)_j,
        &\quad (b_p)_k &= \sqrt{\rho_{pk}}\,\big(W(y^* - m_c)\big)_{i_k}
        &\quad &\text{(square root)},\\
        x'_{jp} &= x_{jp} + f_p^\top A_p^{-1} S_p\, b_{pj},
        &\quad (b_{pj})_k &= \sqrt{\rho_{pk}}\,\big(W(y^* - g_j)\big)_{i_k}
                             - \varepsilon_{j i_k}
        &\quad &\text{(stochastic)},

    with :math:`\delta` the approximation's divisor
    (:attr:`~enskit.distribution.EnsembleGaussian.divisor`),
    :math:`m_{x,p}` the target's mean, :math:`g_j` particle :math:`j`'s
    given values and :math:`\varepsilon` one standard normal ``(J, N)`` draw
    shared by every local update. A target block whose locations are
    ``None`` gets the wrapped rule's global update, with the same draw.
    Everything but :math:`y^*` and the draw is computed at build, so a call
    only gathers and contracts.

    Parameters
    ----------
    update_rule : SymmetricSquareRoot or Matheron
        The rule applied to every local problem.
    localization : DomainLocalization
        The locations, taper, radius and neighborhood size.

    Raises
    ------
    TypeError
        If ``update_rule`` is not one of the two rules or ``localization`` is
        not a :class:`DomainLocalization`; at build, if a given block's noise
        is not row-local: a :class:`~enskit.linalg.PSDDiagonal` or
        :class:`~enskit.linalg.Identity`, possibly scaled (as the tempered
        noise ``R * (1 / delta)`` of an algorithm is), or whitens to another
        dtype than the particles'.
    ValueError
        At build: if the approximation is not an
        :class:`~enskit.distribution.EnsembleGaussian` with the particles'
        count, a given block has no noise, a target block has an independent
        term, a target block is missing from ``target_coords``, the given
        blocks are not the localization's, or a block's locations are of the
        wrong size; the build checks of the wrapped rules otherwise; in debug
        mode, if a particle is not finite or the approximation is not aligned.
        At call, around :class:`Matheron`, if no key is given.
    KeyError
        At build, if ``target_coords`` names a block the approximation does
        not have.

    Notes
    -----
    With :math:`K = N` and every weight 1, every :math:`A_p` is :math:`A`
    with the columns of :math:`S` permuted, and the localized update equals
    the wrapped rule's global one to round-off, for the same key. The
    stochastic draw is pinned as :class:`Matheron`'s: ``k_targets, k_noise =
    jax.random.split(key)`` and :math:`\varepsilon` =
    ``normal(k_noise, (J, N))``. In each local problem's whitened coordinates
    :math:`\varepsilon_{ji_k}` is a draw of noise of variance
    :math:`r_{i_k}/\rho_{pk}`, so each coordinate's spread is its local
    problem's.

    A zero weight gives a zero column of :math:`S_p` and an exactly zero
    singular value, at which :class:`~enskit.linalg.IdentityPlusGram`'s
    derivative rules stay finite. The square root of a weight is taken with a
    zero derivative at zero, so the derivative with respect to a weight that
    is exactly zero is zero by convention; for a taper that reaches zero with
    zero slope, such as :func:`gaspari_cohn`, derivatives in the radius are
    exact.

    References
    ----------
    Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
    assimilation for spatiotemporal chaos: a local ensemble transform Kalman
    filter. *Physica D*, 230(1–2), 112–126.

    Ott, E., Hunt, B. R., Szunyogh, I., Zimin, A. V., Kostelich, E. J.,
    Corazza, M., Kalnay, E., Patil, D. J. & Yorke, J. A. (2004). A local
    ensemble Kalman filter for atmospheric data assimilation. *Tellus A*,
    56(5), 415–428.

    Sakov, P. & Bertino, L. (2011). Relation between two common localisation
    methods for the EnKF. *Computational Geosciences*, 15(2), 225–237.
    """

    def __init__(self, update_rule, localization) -> None:
        where = "LocalizedUpdateRule"
        if not isinstance(update_rule, (SymmetricSquareRoot, Matheron)):
            raise TypeError(
                f"{where}: update_rule must be SymmetricSquareRoot() or Matheron(), "
                f"got {type(update_rule).__name__}"
            )
        if not isinstance(localization, DomainLocalization):
            raise TypeError(
                f"{where}: localization must be a DomainLocalization, got "
                f"{type(localization).__name__}"
            )
        c.guard(localization, where)
        object.__setattr__(self, "update_rule", update_rule)
        object.__setattr__(self, "localization", localization)

    def build(self, particles, approximation, given):
        """Build the localized update of ``particles`` given the blocks ``given``."""
        where = "LocalizedUpdateRule.build"
        given, targets = c.check_build_arguments(where, particles, approximation, given)
        if not c.is_aligned_type(particles, approximation):
            raise ValueError(
                f"{where}: the approximation must be an EnsembleGaussian with the "
                f"particles' count ({particles.n_particles}), aligned with them, as "
                f"gaussian_approximation builds; got {approximation!r}. The local "
                f"updates read the given blocks' whitened residuals off its factor."
            )
        loc = self.localization
        c.guard(loc, where)
        dims = approximation.dims
        loc_given = _check_given(where, loc, given)
        _check_targets(where, loc, approximation.names, given, targets)
        for name, coords in zip(loc_given, loc._given_coords, strict=True):
            _check_rows(where, "given_coords", name, coords, dims[name], loc)
        for name, coords in zip(loc.target_names, loc._target_coords, strict=True):
            if coords is not None:
                _check_rows(where, "target_coords", name, coords, dims[name], loc)
        termed = tuple(n for n in targets if approximation.block_cov(n) is not None)
        if termed:
            raise ValueError(
                f"{where}: target blocks {termed} have independent terms, which a "
                f"localized update does not draw"
            )
        for name in given:
            D = approximation.block_cov(name)
            if D is None:
                raise ValueError(
                    f"{where}: given block {name!r} has no independent term. "
                    f"Localization tapers the given blocks' noise: pass it (noise= "
                    f"in update, or add_noise)."
                )
            if not _row_local(D):
                raise TypeError(
                    f"{where}: the noise of given block {name!r} must be row-local, "
                    f"a PSDDiagonal or Identity, possibly scaled; got {D!r}. A local "
                    f"problem needs the noise restricted to a neighborhood, which "
                    f"is not an operator-layer operation for a correlated block."
                )
        c.check_particles_finite(where, particles)
        c.check_alignment(where, particles, approximation)

        J = particles.n_particles
        scale = math.sqrt(approximation.divisor)
        dtype = particles[particles.names[0]].dtype
        stochastic = isinstance(self.update_rule, Matheron)
        eye = jnp.eye(J, dtype=dtype)

        # The whitened factor S, (J, N), its columns in the approximation's order.
        columns = []
        for name in given:
            F = approximation.factor(name)
            rows = jnp.zeros((J, dims[name]), dtype) if F is None else F.matvec(eye)
            whitened = approximation.block_cov(name).whiten(rows)
            if whitened.dtype != dtype:
                raise TypeError(
                    f"{where}: the noise of given block {name!r} whitens to dtype "
                    f"{whitened.dtype}, the particles have {dtype}. Operators carry "
                    f"no dtype; build the noise in the particles' dtype (a scalar "
                    f"scale promotes it)."
                )
            _whitening_check(where, name, "whitened factor", whitened)
            columns.append(whitened)
        S = jnp.concatenate(columns, axis=1)

        # Neighbor indices count in the localization's order of the given
        # blocks; ``perm`` carries them to the approximation's.
        offsets, start = {}, 0
        for name in given:
            offsets[name] = start
            start += dims[name]
        perm = jnp.asarray(
            np.concatenate(
                [np.arange(offsets[n], offsets[n] + dims[n]) for n in loc_given]
            )
        )

        def local(f, idx, root):
            Sp = S[:, idx] * root
            G = IdentityPlusGram(Sp)
            gain = dense_matvec(Sp.T, G.solve(f))
            anomalies = None if stochastic else G.inverse_sqrt().matvec(f)
            return gain, anomalies

        kinds, neighbors, gains_res, gains_noise, anomalies, rows = [], [], [], [], [], []
        located = dict(zip(loc.target_names, loc._target_coords, strict=True))
        weights = dict(zip(loc.located, loc._weights, strict=True))
        nbrs = dict(zip(loc.located, loc._neighbors, strict=True))
        for name in targets:
            F = approximation.factor(name)
            # Every target has a factor row: an aligned EnsembleGaussian refuses
            # a block with neither a row nor a term, and target terms are refused.
            kind = "global" if located[name] is None else "local"
            kinds.append(kind)
            entry = dict(idx=None, res=None, noise=None, anom=None, row=None)
            if kind == "local":
                idx = perm[nbrs[name]]
                root = _safe_sqrt(weights[name]).astype(dtype)
                gain, anom = jax.vmap(local)(F.matvec(eye).T, idx, root)
                entry.update(idx=idx, res=root * gain)
                if stochastic:
                    entry["noise"] = gain
                else:
                    entry["anom"] = scale * anom.T
            elif kind == "global":
                entry["row"] = F
            neighbors.append(entry["idx"])
            gains_res.append(entry["res"])
            gains_noise.append(entry["noise"])
            anomalies.append(entry["anom"])
            rows.append(entry["row"])

        gram = None
        if "global" in kinds:
            gram = IdentityPlusGram(S)
            if not stochastic:
                T = gram.inverse_sqrt().matvec(eye)
                for i, kind in enumerate(kinds):
                    if kind == "global":
                        anomalies[i] = scale * rows[i].matvec(T)

        return c.build(
            _LocalizedUpdate,
            S=S,
            given_covs=tuple(approximation.block_cov(n) for n in given),
            given_means=tuple(approximation.mean(n) for n in given),
            particles=particles.marginal(*targets) if stochastic else None,
            target_means=None
            if stochastic
            else tuple(approximation.mean(n) for n in targets),
            neighbors=tuple(neighbors),
            gains_res=tuple(gains_res),
            gains_noise=tuple(gains_noise),
            anomalies=tuple(anomalies),
            gram=gram,
            target_rows=tuple(rows),
            rule=type(self.update_rule).__name__,
            given=given,
            targets=targets,
            given_dims=tuple(dims[n] for n in given),
            n_particles=J,
            divisor=approximation.divisor,
            dtype=jnp.dtype(dtype),
            kinds=tuple(kinds),
        )

    def __repr__(self) -> str:
        return f"LocalizedUpdateRule({self.update_rule!r}, DomainLocalization(...))"


@c.pytree_class(
    data=("_target_coords", "_given_coords", "radius", "_neighbors", "_weights"),
    meta=(
        "target_names",
        "located",
        "given_names",
        "max_neighbors",
        "taper_name",
    ),
)
class DomainLocalization:
    r"""The geometry of domain localization: locations and neighborhoods.

    Holds a location :math:`c_p \in \mathbb R^q` for every coordinate of
    every located target block and :math:`c_i` for every given coordinate,
    and computes, once, each target coordinate's neighborhood
    :math:`\mathcal N_p = (i_1, \dots, i_K)`, the :math:`K` given coordinates
    nearest it (nearest first, ties to the lower index in the order of
    ``given_coords``), and their weights

    .. math::

        \rho_{pk} = \rho\big(d(c_p, c_{i_k}) / L\big), \qquad k = 1, \dots, K,

    with :math:`d` the distance, :math:`\rho` the taper and :math:`L` the
    radius. :class:`LocalizedUpdateRule` reads them: given coordinate
    :math:`i_k` enters the local update of :math:`p` with its noise variance
    :math:`r_{i_k}` inflated to :math:`r_{i_k}/\rho_{pk}`, so a neighbor with
    :math:`\rho_{pk} = 0` has no influence. Every neighborhood has the same
    size, so the local updates vectorize; a given coordinate outside the
    :math:`K` nearest has no influence even within the radius.

    Parameters
    ----------
    target_coords : Mapping[str, Array | None]
        For each target block, a ``(d_x, q)`` array of locations, one row per
        coordinate, or ``None`` for a block without a location, such as a
        global parameter, which gets the global update. Every target block
        must be named, and at least one must have locations.
    given_coords : Mapping[str, Array] or Array
        ``(d_c, q)`` locations of each given block's coordinates, or one bare
        ``(N, q)`` array, accepted when exactly one block is given.
    radius : float or Array
        Keyword-only. :math:`L`, a real scalar, positive and finite.
    max_neighbors : int
        Keyword-only. :math:`K`, an integer from 1 to the number :math:`N` of
        given coordinates.
    taper : callable, optional
        Keyword-only. :math:`\rho`, from an array of :math:`r = d/L` to an
        array of the same shape with values in :math:`[0, 1]`;
        :func:`gaspari_cohn` by default.
    distance : callable, optional
        Keyword-only. ``(point (q,), points (m, q)) -> (m,)``; Euclidean by
        default. Pass a periodic distance for a periodic domain.

    Raises
    ------
    TypeError
        If ``target_coords`` is not a mapping, a name is not a ``str``, an
        array is complex or boolean, ``max_neighbors`` is not an ``int``, or
        ``taper`` or ``distance`` is not callable.
    ValueError
        If an array is not 2-D with at least one row, the arrays' location
        dimensions differ, no target block has locations, ``given_coords`` is
        empty, ``radius`` is not a scalar, ``max_neighbors`` is outside
        :math:`[1, N]`, or the distance or taper returns the wrong shape; in
        debug mode, if a location, the radius or a distance is not finite,
        the radius is not positive, a distance is negative, or a weight is
        outside :math:`[0, 1]`.

    Notes
    -----
    The distances and weights are computed in the locations' floating dtype,
    in time and memory proportional to :math:`d_x N` for each located block.
    Any taper with values in :math:`[0, 1]` gives a valid update, since a
    weight only scales a noise variance. That a taper must be a
    positive-definite function is a requirement of covariance localization,
    where it multiplies a covariance, and not of this one. The radius enters
    only the weights, so a localization built inside a traced function is
    differentiable in it.

    References
    ----------
    Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
    assimilation for spatiotemporal chaos: a local ensemble transform Kalman
    filter. *Physica D*, 230(1–2), 112–126.

    Ott, E., Hunt, B. R., Szunyogh, I., Zimin, A. V., Kostelich, E. J.,
    Corazza, M., Kalnay, E., Patil, D. J. & Yorke, J. A. (2004). A local
    ensemble Kalman filter for atmospheric data assimilation. *Tellus A*,
    56(5), 415–428.
    """

    def __init__(
        self,
        target_coords: Mapping[str, Array | None],
        given_coords,
        *,
        radius,
        max_neighbors: int,
        taper: Callable | None = None,
        distance: Callable | None = None,
    ) -> None:
        where = "DomainLocalization"
        taper = gaspari_cohn if taper is None else taper
        if not isinstance(target_coords, Mapping):
            raise TypeError(
                f"{where}: target_coords must be a mapping from block name to "
                f"locations or None, got {type(target_coords).__name__}"
            )
        target_names, raw_targets = _names_and_values(
            where, "target_coords", target_coords
        )
        if isinstance(given_coords, Mapping):
            given_names, raw_given = _names_and_values(
                where, "given_coords", given_coords
            )
        else:
            given_names, raw_given = None, (given_coords,)
        if any(v is None for v in raw_given):
            raise TypeError(
                f"{where}: every given block needs locations; given_coords holds None"
            )
        if isinstance(max_neighbors, bool):
            raise TypeError(f"{where}: max_neighbors must be an int, got bool")
        try:
            max_neighbors = operator.index(max_neighbors)
        except TypeError:
            raise TypeError(
                f"{where}: max_neighbors must be an int, got "
                f"{type(max_neighbors).__name__}"
            ) from None
        for what, fn in (("taper", taper), ("distance", distance)):
            if fn is not None and not callable(fn):
                raise TypeError(f"{where}: {what} must be callable, got {fn!r}")
        if given_names == ():
            raise ValueError(f"{where}: given_coords names no block")

        labels = [f"target_coords[{n!r}]" for n in target_names]
        labels += (
            ["given_coords"]
            if given_names is None
            else [f"given_coords[{n!r}]" for n in given_names]
        )
        arrays = [None if v is None else jnp.asarray(v) for v in raw_targets + raw_given]
        present = [
            (lab, a) for lab, a in zip(labels, arrays, strict=True) if a is not None
        ]
        for label, a in present:
            if not (
                jnp.issubdtype(a.dtype, jnp.floating)
                or jnp.issubdtype(a.dtype, jnp.integer)
            ):
                raise TypeError(f"{where}: {label} must be real, got dtype {a.dtype}")
        for label, a in present:
            if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
                raise ValueError(
                    f"{where}: {label} must be a (rows, q) array with at least one "
                    f"row and q >= 1, got shape {a.shape}"
                )
        qs = {a.shape[1] for _, a in present}
        if len(qs) != 1:
            raise ValueError(
                f"{where}: every array of locations must have the same number of "
                f"columns q; got "
                + ", ".join(f"{lab}: {a.shape[1]}" for lab, a in present)
            )
        if all(a is None for a in arrays[: len(target_names)]):
            raise ValueError(
                f"{where}: no target block has locations; at least one must, or the "
                f"wrapped rule alone is the update"
            )
        dtype = jnp.result_type(*(a.dtype for _, a in present))
        if not jnp.issubdtype(dtype, jnp.floating):
            dtype = jnp.asarray(0.0).dtype
        arrays = [None if a is None else a.astype(dtype) for a in arrays]
        tcoords = tuple(arrays[: len(target_names)])
        gcoords = tuple(arrays[len(target_names) :])
        radius = c.check_scalar(where, "radius", radius, dtype)
        N = sum(g.shape[0] for g in gcoords)
        if not 1 <= max_neighbors <= N:
            raise ValueError(
                f"{where}: max_neighbors must be from 1 to the number of given "
                f"coordinates, {N}; got {max_neighbors}"
            )

        for label, a in present:
            value_check(
                a,
                lambda x: bool(jnp.all(jnp.isfinite(x))),
                f"{where}: {label} must be finite",
            )
        value_check(
            radius,
            lambda r: bool(jnp.isfinite(r) & (r > 0)),
            f"{where}: radius must be finite and positive",
        )

        dist_fn = _euclidean if distance is None else distance
        points = jnp.concatenate(gcoords, axis=0)
        located, neighbors, weights = [], [], []
        for name, coords in zip(target_names, tcoords, strict=True):
            if coords is None:
                continue
            d = jnp.asarray(jax.vmap(lambda p: dist_fn(p, points))(coords))
            if d.shape != (coords.shape[0], N):
                raise ValueError(
                    f"{where}: distance must map a (q,) point and (m, q) points to "
                    f"(m,) distances; for block {name!r} it gave shape "
                    f"{d.shape[1:]} for m = {N}"
                )
            value_check(
                d,
                lambda x: bool(jnp.all(jnp.isfinite(x) & (x >= 0))),
                f"{where}: the distances from block {name!r} must be finite and "
                f"nonnegative",
            )
            neg, idx = jax.lax.top_k(-d, max_neighbors)
            w = jnp.asarray(taper(-neg / radius))
            if w.shape != idx.shape:
                raise ValueError(
                    f"{where}: taper must return an array of its argument's shape; "
                    f"it returned {w.shape} for {idx.shape}"
                )
            value_check(
                w,
                lambda x: bool(jnp.all(jnp.isfinite(x) & (x >= 0) & (x <= 1))),
                f"{where}: the taper's weights for block {name!r} must be finite and "
                f"in [0, 1]. A weight above 1 would shrink a noise variance below "
                f"the known one.",
            )
            located.append(name)
            neighbors.append(idx)
            weights.append(w)

        for field, value in (
            ("_target_coords", tcoords),
            ("_given_coords", gcoords),
            ("radius", radius),
            ("_neighbors", tuple(neighbors)),
            ("_weights", tuple(weights)),
            ("target_names", target_names),
            ("located", tuple(located)),
            ("given_names", given_names),
            ("max_neighbors", max_neighbors),
            ("taper_name", getattr(taper, "__name__", type(taper).__name__)),
        ):
            object.__setattr__(self, field, value)

    @property
    def target_coords(self) -> dict:
        """The locations of each target block, ``None`` for an unlocated one."""
        return dict(zip(self.target_names, self._target_coords, strict=True))

    @property
    def given_coords(self):
        """The given blocks' locations: a mapping, or the bare array."""
        if self.given_names is None:
            return self._given_coords[0]
        return dict(zip(self.given_names, self._given_coords, strict=True))

    @property
    def neighbors(self) -> dict:
        """For each located block, the ``(d_x, K)`` neighbor indices :math:`i_k`.

        Indices into the given coordinates concatenated in the order of
        ``given_coords``, nearest first.
        """
        return dict(zip(self.located, self._neighbors, strict=True))

    @property
    def weights(self) -> dict:
        """For each located block, the ``(d_x, K)`` weights :math:`\rho_{pk}`."""
        return dict(zip(self.located, self._weights, strict=True))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a localization built directly."""
        return tuple(jnp.shape(self.radius))

    def __repr__(self) -> str:
        def text():
            targets = {
                n: (None if a is None else a.shape[-2])
                for n, a in zip(self.target_names, self._target_coords, strict=True)
            }
            if self.given_names is None:
                given = self._given_coords[0].shape[-2]
            else:
                given = {
                    n: a.shape[-2]
                    for n, a in zip(self.given_names, self._given_coords, strict=True)
                }
            q = self._given_coords[0].shape[-1]
            return (
                f"DomainLocalization(target_dims={targets}, given_dims={given}, "
                f"coord_dim={q}, max_neighbors={self.max_neighbors}, "
                f"taper={self.taper_name})"
            )

        return c.safe_repr(text, "DomainLocalization", lambda: self.batch_shape)


def gaspari_cohn(r) -> Array:
    r"""The Gaspari–Cohn fifth-order taper, on :math:`r` = distance / radius.

    With :math:`z = 2|r|`,

    .. math::

        \rho(r) = \begin{cases}
          -\tfrac14 z^5 + \tfrac12 z^4 + \tfrac58 z^3 - \tfrac53 z^2 + 1,
            & 0 \le z \le 1,\\[2pt]
          \tfrac1{12} z^5 - \tfrac12 z^4 + \tfrac58 z^3 + \tfrac53 z^2 - 5z + 4
            - \tfrac{2}{3z}, & 1 < z \le 2,\\[2pt]
          0, & z > 2 .
        \end{cases}

    Equal to 1 at :math:`r = 0`, decreasing to exactly 0 at :math:`|r| = 1`
    and zero beyond, and a positive-definite correlation function in up to
    three dimensions. Elementwise, with a finite derivative everywhere.

    Parameters
    ----------
    r : Array
        Distances divided by the radius, of any shape. An integer array is
        converted to the default floating dtype.

    Returns
    -------
    Array
        :math:`\rho(r)`, of ``r``'s shape and floating dtype, in
        :math:`[0, 1]`.

    Notes
    -----
    The second branch rounds to slightly negative values near :math:`z = 2`
    (about :math:`-10^{-6}` in float32), so the result is clamped at zero.

    References
    ----------
    Gaspari, G. & Cohn, S. E. (1999). Construction of correlation functions
    in two and three dimensions. *Quarterly Journal of the Royal
    Meteorological Society*, 125(554), 723–757.
    """
    r = jnp.asarray(r)
    if not jnp.issubdtype(r.dtype, jnp.floating):
        r = r.astype(jnp.asarray(0.0).dtype)
    z = 2.0 * jnp.abs(r)
    inner = (((-0.25 * z + 0.5) * z + 0.625) * z - 5.0 / 3.0) * z * z + 1.0
    zo = jnp.where(z > 1.0, z, 2.0)  # the second branch, never evaluated at z <= 1
    outer = (
        ((((zo / 12.0 - 0.5) * zo + 0.625) * zo + 5.0 / 3.0) * zo - 5.0) * zo
        + 4.0
        - 2.0 / (3.0 * zo)
    )
    return jnp.where(z <= 1.0, inner, jnp.where(z < 2.0, jnp.maximum(outer, 0.0), 0.0))


# ---------------------------------------------------------------------------
# private: the built update
# ---------------------------------------------------------------------------


@c.pytree_class(
    data=(
        "S",
        "given_covs",
        "given_means",
        "particles",
        "target_means",
        "neighbors",
        "gains_res",
        "gains_noise",
        "anomalies",
        "gram",
        "target_rows",
    ),
    meta=(
        "rule", "given", "targets", "given_dims", "n_particles", "divisor", "dtype",
        "kinds",
    ),
)
class _LocalizedUpdate:
    """The particle update :class:`LocalizedUpdateRule` builds."""

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.S.shape[:-2])

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
        stochastic = self.rule == "Matheron"
        c.check_key(
            where,
            key,
            required=stochastic,
            why=": the stochastic rule draws the given blocks' noise",
        )
        dtype = self.dtype
        whitened = []
        for name, d, D, m in zip(
            self.given, self.given_dims, self.given_covs, self.given_means, strict=True
        ):
            y = jnp.asarray(vals[name])
            if not (
                jnp.issubdtype(y.dtype, jnp.floating)
                or jnp.issubdtype(y.dtype, jnp.integer)
            ):
                raise TypeError(
                    f"{where}: the value of block {name!r} must be real, got dtype "
                    f"{y.dtype}"
                )
            if y.shape != (d,):
                raise ValueError(
                    f"{where}: the value of block {name!r} must have shape ({d},), got "
                    f"{y.shape}. A family of values is a jax.vmap over this call."
                )
            y = y.astype(dtype)
            value_check(
                y,
                lambda a: bool(jnp.all(jnp.isfinite(a))),
                f"{where}: the value of block {name!r} must be finite",
            )
            w = D.whiten(y - m)
            _whitening_check(where, name, "whitened residual", w)
            whitened.append(w)
        b = jnp.concatenate(whitened)
        if stochastic:
            _, k_noise = jax.random.split(key)
            eps = jax.random.normal(k_noise, self.S.shape, dtype)
            resid = b - math.sqrt(self.divisor) * self.S

        blocks = []
        for i, (name, kind) in enumerate(zip(self.targets, self.kinds, strict=True)):
            idx, res, anom = self.neighbors[i], self.gains_res[i], self.anomalies[i]
            if stochastic:
                x = self.particles[name]
                if kind == "local":
                    x = (
                        x
                        + jnp.einsum("jpk,pk->jp", resid[:, idx], res)
                        - jnp.einsum("jpk,pk->jp", eps[:, idx], self.gains_noise[i])
                    )
                else:
                    w = self.gram.solve_factor(resid - eps)
                    x = x + self.target_rows[i].matvec(w)
            else:
                m = self.target_means[i]
                if kind == "local":
                    x = m + jnp.sum(res * b[idx], axis=-1) + anom
                else:
                    x = m + self.target_rows[i].matvec(self.gram.solve_factor(b)) + anom
            if x.dtype != dtype:
                raise TypeError(
                    f"{where}: updated block {name!r} has dtype {x.dtype}, the "
                    f"particles {dtype}: an operator of the approximation (a noise "
                    f"covariance or a factor row) is of another dtype"
                )
            c.result_check(where, f"updated block {name!r}", x)
            blocks.append(x)
        return c.build(
            Ensemble, names=self.targets, _blocks=tuple(blocks), _log_weights=None
        )

    def __repr__(self) -> str:
        located = tuple(
            t for t, k in zip(self.targets, self.kinds, strict=True) if k == "local"
        )
        return c.safe_repr(
            lambda: (
                f"LocalizedUpdate(rule={self.rule}(), given={self.given}, "
                f"targets={self.targets}, n_particles={self.n_particles}, "
                f"located={located})"
            ),
            "LocalizedUpdate",
            lambda: self.batch_shape,
        )


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _euclidean(point: Array, points: Array) -> Array:
    """Euclidean distances from one ``(q,)`` point to ``(m, q)`` points.

    With a zero derivative, not ``nan``, where two locations coincide.
    """
    return _safe_sqrt(jnp.sum((points - point) ** 2, axis=-1))


def _safe_sqrt(w: Array) -> Array:
    """The square root, with a zero derivative rather than an infinite one at 0."""
    positive = w > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, w, 1.0)), 0.0)


def _row_local(D) -> bool:
    """Whether a noise covariance is diagonal: its whitener acts row by row."""
    while isinstance(D, PSDScaled):
        D = D.op
    return isinstance(D, (PSDDiagonal, Identity))


def _names_and_values(where: str, field: str, mapping: Mapping):
    names, values = [], []
    for name, value in mapping.items():
        if not isinstance(name, str):
            raise TypeError(
                f"{where}: the names in {field} must be str, got {type(name).__name__}"
            )
        names.append(name)
        values.append(value)
    return tuple(names), tuple(values)


def _check_given(where: str, loc: DomainLocalization, given) -> tuple[str, ...]:
    """The given blocks in the localization's order, checked against ``given``."""
    if loc.given_names is None:
        if len(given) != 1:
            raise ValueError(
                f"{where}: the localization's given_coords is one bare array, which "
                f"locates exactly one given block; the given blocks are {given}. "
                f"Pass given_coords as a mapping from block name to locations."
            )
        return given
    if set(loc.given_names) != set(given):
        raise ValueError(
            f"{where}: the localization locates the given blocks "
            f"{loc.given_names}, but the given blocks are {given}"
        )
    return loc.given_names


def _check_targets(where, loc, names, given, targets) -> None:
    for name in loc.target_names:
        if name not in names:
            raise KeyError(
                f"{where}: target_coords names {name!r}, which is not a block; the "
                f"blocks are {names}"
            )
        if name in given:
            raise ValueError(
                f"{where}: target_coords names {name!r}, which is a given block"
            )
    missing = tuple(n for n in targets if n not in loc.target_names)
    if missing:
        raise ValueError(
            f"{where}: target blocks {missing} are missing from target_coords. Name "
            f"every target block, with None for one that gets the global update."
        )


def _check_rows(where, field, name, coords, dim, loc) -> None:
    bare = field == "given_coords" and loc.given_names is None
    label = field if bare else f"{field}[{name!r}]"
    if coords.shape[-2] != dim:
        raise ValueError(
            f"{where}: {label} has {coords.shape[-2]} rows, but block {name!r} has "
            f"dimension {dim}: one location per coordinate"
        )


def _whitening_check(where: str, name: str, what: str, value) -> None:
    value_check(
        value,
        lambda a: bool(jnp.all(jnp.isfinite(a))),
        f"{where}: the {what} of given block {name!r} is not finite. {_WHITENING_CAUSE}",
    )
