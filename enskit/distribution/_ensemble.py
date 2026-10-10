r"""The empirical distribution :class:`Ensemble`."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from ..linalg import Dense, LinOp, PSDLowRank, product, static_field
from . import _common as c
from ._common import distribution_class

__all__ = ["Ensemble"]


@distribution_class
class Ensemble:
    r"""An empirical distribution over named blocks, optionally weighted.

    Holds :math:`J` *particles* :math:`x_1, \dots, x_J`. Block ``b`` is
    stored as a ``(J, d_b)`` array whose row ``j`` is particle ``j``'s value
    of that block; every block has the same particles in the same order. The
    distribution is

    .. math::

        \hat p = \sum_{j=1}^{J} w_j\, \delta_{x_j}, \qquad
        w_j = \frac{\exp(\ell_j)}{\sum_{i=1}^{J} \exp(\ell_i)},

    with log weights :math:`\ell_j`, or :math:`w_j = 1/J` when unweighted.

    Parameters
    ----------
    blocks : Mapping[str, Array], optional
        Positional-only. One exactly 2-D ``(J, d_b)`` array per block, with
        ``d_b >= 1``, all with the same ``J >= 2`` and the same real floating
        dtype. The mapping's order is the block order. Arrays are stored as
        given, never converted to another dtype.
    log_weights : Array, optional
        Keyword-only. ``(J,)`` unnormalized log weights :math:`\ell`, of the
        blocks' dtype. ``None`` means equal weights, and differs from an
        array of equal values only in :attr:`is_weighted`.
    **block_arrays : Array
        Blocks given as keywords, appended after those in ``blocks`` in the
        order written. A block named ``log_weights`` must be given in
        ``blocks``.

    Raises
    ------
    ValueError
        If there are no blocks, a block is not 2-D or has no columns, the
        blocks disagree on ``J``, ``J < 2``, or ``log_weights`` is not
        ``(J,)``.
    TypeError
        If a block is not a real floating array, the blocks' dtypes differ,
        ``log_weights`` has another dtype, or a block is given twice.

    Notes
    -----
    Particles need not be finite. A non-finite row is how a failed particle
    is marked, and :attr:`all_finite` finds them. The moments leave out every
    particle whose weight is zero, so a failed particle given log weight
    :math:`-\infty` affects no mean, covariance or projection, and no other
    particle's anomaly.

    Examples
    --------
    >>> ens = Ensemble(x=x, theta=theta)           # (J, 40), (J, 1)
    >>> ens["x"].shape
    (J, 40)
    >>> ens.marginal("theta")
    Ensemble(n_particles=J, blocks={'theta': 1}, weighted=False)
    """

    names: tuple[str, ...] = static_field()
    _blocks: tuple[Array, ...]
    _log_weights: Array | None

    def __init__(self, blocks=None, /, *, log_weights=None, **block_arrays) -> None:
        where = "Ensemble"
        arrays = c.merge_blocks(where, blocks, block_arrays)
        if not arrays:
            raise ValueError(f"{where}: at least one block is required")
        first = None
        stored = []
        for name, value in arrays.items():
            arr = jnp.asarray(value)
            if first is None:
                _check_first_block(where, name, arr)
                first = arr
            else:
                _check_block(where, name, arr, first.shape[0], first.dtype)
            stored.append(arr)
        n_particles, dtype = first.shape[0], first.dtype
        if log_weights is not None:
            log_weights = jnp.asarray(log_weights)
            if log_weights.dtype != dtype:
                raise TypeError(
                    f"{where}: log_weights must have the blocks' dtype {dtype}, got "
                    f"{log_weights.dtype}"
                )
            if log_weights.shape != (n_particles,):
                raise ValueError(
                    f"{where}: log_weights must have shape ({n_particles},), got "
                    f"{log_weights.shape}"
                )
        object.__setattr__(self, "names", tuple(arrays))
        object.__setattr__(self, "_blocks", tuple(stored))
        object.__setattr__(self, "_log_weights", log_weights)

    # -- introspection ---------------------------------------------------------
    @property
    def dims(self) -> dict[str, int]:
        """Each block's dimension, keyed by name, in order."""
        return {
            n: int(a.shape[-1]) for n, a in zip(self.names, self._blocks, strict=True)
        }

    @property
    def n_particles(self) -> int:
        """The number of particles :math:`J` (rows of every block)."""
        return int(self._blocks[0].shape[-2])

    @property
    def is_weighted(self) -> bool:
        """Whether this ensemble carries ``log_weights``."""
        return self._log_weights is not None

    @property
    def log_weights(self) -> Array | None:
        """The stored unnormalized log weights, or ``None``."""
        return self._log_weights

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        shapes = [tuple(a.shape[:-2]) for a in self._blocks]
        if self._log_weights is not None:
            shapes.append(tuple(self._log_weights.shape[:-1]))
        return c.broadcast_batch("Ensemble", *shapes)

    @property
    def weights(self) -> Array:
        r"""Normalized weights :math:`w`, ``(J,)``; :math:`1/J` each when unweighted.

        Computed as a max-shifted softmax of the log weights, so log weights
        far outside the floating-point range normalize correctly.
        """
        c.guard(self, "weights")
        if self._log_weights is None:
            return jnp.full(
                (self.n_particles,), 1.0 / self.n_particles, dtype=self._dtype
            )
        return jax.nn.softmax(self._log_weights)

    @property
    def all_finite(self) -> Array:
        """``(J,)`` boolean: particle ``j`` is finite in every block.

        Non-finite rows are how the layers above mark failed particles; this
        is how such particles are found. It reads values, so it is an array,
        never a Python ``bool``.
        """
        c.guard(self, "all_finite")
        out = jnp.ones((self.n_particles,), dtype=bool)
        for arr in self._blocks:
            out = out & jnp.all(jnp.isfinite(arr), axis=-1)
        return out

    def __getitem__(self, name: str) -> Array:
        """Block ``name`` as a ``(J, d)`` array.

        Raises
        ------
        KeyError
            If there is no such block; the message lists the blocks.
        """
        c.guard(self, "__getitem__")
        c.check_known(f"{self!r}[...]", self.names, (name,))
        return self._blocks[self.names.index(name)]

    # -- structure ---------------------------------------------------------------
    def marginal(self, *names: str) -> Ensemble:
        """The ensemble restricted to ``names``, in the order given.

        Weights are kept. Marginalizing an empirical distribution is dropping
        coordinates, so the particles are unchanged.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If no name is given, or a name is repeated.
        """
        where = f"{self!r}.marginal"
        c.guard(self, "marginal")
        names = c.check_name_list(where, self.names, names)
        if not names:
            raise ValueError(f"{where}: at least one block name is required")
        return self._with([(n, self[n]) for n in names])

    def drop(self, *names: str) -> Ensemble:
        """The ensemble without ``names``, in block order; weights are kept.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If a name is repeated, or no block would remain.
        """
        where = f"{self!r}.drop"
        c.guard(self, "drop")
        dropped = c.check_name_list(where, self.names, names)
        kept = [n for n in self.names if n not in dropped]
        if not kept:
            raise ValueError(f"{where}: at least one block must remain")
        return self._with([(n, self[n]) for n in kept])

    def assign(self, blocks=None, /, **block_arrays) -> Ensemble:
        """Replace existing blocks or append new ones; weights are kept.

        Blocks may be given as a mapping, as keywords, or both
        (``ens.assign(x_prev=ens["x"])``). A replaced block keeps its
        position; new blocks are appended in the order given.

        Raises
        ------
        ValueError
            If an array is not ``(J, d)`` with this ensemble's ``J`` and
            ``d >= 1``.
        TypeError
            If an array's dtype is not this ensemble's, or a block is given
            twice.
        """
        where = f"{self!r}.assign"
        c.guard(self, "assign")
        arrays = c.merge_blocks(where, blocks, block_arrays)
        checked = {}
        for name, value in arrays.items():
            arr = jnp.asarray(value)
            _check_block(where, name, arr, self.n_particles, self._dtype)
            checked[name] = arr
        pairs = [
            (n, checked.get(n, a)) for n, a in zip(self.names, self._blocks, strict=True)
        ]
        pairs += [(n, a) for n, a in checked.items() if n not in self.names]
        return self._with(pairs)

    def rename(self, mapping=None, /, **new_names) -> Ensemble:
        """Rename blocks (``old=new``); positions and weights are kept.

        Raises
        ------
        KeyError
            If an old name is not a block.
        ValueError
            If the new names are not distinct, or one collides with a block
            that is not itself being renamed. A swap is allowed.
        TypeError
            If a new name is not a ``str``, or a block is given twice.
        """
        where = f"{self!r}.rename"
        c.guard(self, "rename")
        names = _renamed(where, self.names, mapping, new_names)
        return c.build(
            Ensemble, names=names, _blocks=self._blocks, _log_weights=self._log_weights
        )

    def pipe(self, f, *args, **kwargs):
        """Return ``f(self, *args, **kwargs)``.

        Lets an operation defined in a layer above this one be chained
        without this layer importing it, as in
        ``ens.pipe(maps.pushforward, ...)``.
        """
        c.guard(self, "pipe")
        return f(self, *args, **kwargs)

    # -- moments -------------------------------------------------------------------
    def mean(self, name: str) -> Array:
        r"""The (weighted) mean of block ``name``, ``(d,)``.

        .. math::

            \bar x = \sum_{j=1}^{J} w_j x_j .

        Computed as a reference particle :math:`x_r` plus the mean of the
        differences from it, so identical particles have exactly their common
        value as mean. The reference is the first particle when unweighted,
        and a particle of largest weight when weighted. Particles of weight
        zero are left out of the sum, so a non-finite particle of weight zero
        does not affect the mean.
        """
        c.guard(self, "mean")
        x = self._block(f"{self!r}.mean", name)
        return self._reference(x)[..., 0, :] + jnp.squeeze(
            self._mean_of_differences(x), axis=-2
        )

    def anomalies(self, name: str) -> Array:
        r"""Raw deviations from the mean, ``(J, d)``.

        .. math::

            a_j = x_j - \bar x = (x_j - x_r) - \sum_{i : w_i > 0} w_i (x_i - x_r),

        computed in the second form, with the reference particle :math:`x_r`
        of :meth:`mean`, so particles that are identical give exactly zero
        anomalies however large their magnitude. A particle of weight zero
        has its own anomaly, which is non-finite if the particle is, and does
        not affect the others.
        """
        c.guard(self, "anomalies")
        x = self._block(f"{self!r}.anomalies", name)
        diffs = x - self._reference(x)
        return diffs - self._mean_of_differences(x)

    def cov(self, a: str, b: str | None = None, *, unbiased: bool = True) -> LinOp:
        r"""The covariance of block ``a`` (with ``b``, if given), as an operator.

        .. math::

            \hat C_{ab} = \frac{1}{\delta_w}
                \sum_{j=1}^{J} w_j\, a_j^{(a)} \big(a_j^{(b)}\big)^\top,
            \qquad
            \delta_w = \begin{cases}
                1 - \sum_j w_j^2 & \text{unbiased (the default)},\\
                1 & \text{otherwise},
            \end{cases}

        which with :math:`w_j = 1/J` is :math:`\frac{1}{\delta}\sum_j a_j
        a_j^\top` with :math:`\delta = J - 1` or :math:`J`. The unbiased
        divisor treats the particles as samples; the other gives the
        covariance of :math:`\hat p` itself. Nothing of size
        ``(d_a, d_b)`` is formed: the result applies the scaled anomalies.

        Parameters
        ----------
        a, b : str
            Block names; ``b`` defaults to ``a``.
        unbiased : bool
            Keyword-only. ``True`` (the default) for the divisor
            :math:`J - 1` or :math:`1 - \sum_j w_j^2`, ``False`` for
            :math:`J` or :math:`1`.

        Returns
        -------
        LinOp
            For ``cov(a)``, or ``cov(a, a)``, a
            :class:`~enskit.linalg.PSDLowRank` of the projected factor row
            (see :meth:`project`), of width ``J``. For ``cov(a, b)`` with
            ``a != b``, the product operator :math:`F_a F_b^\top`, of shape
            ``(d_a, d_b)``.

        Raises
        ------
        KeyError
            If a name is not a block.
        TypeError
            If ``unbiased`` is not a ``bool``.
        ValueError
            In debug mode, if a particle of positive weight is not finite in
            block ``a`` or ``b``. Otherwise the result is then ``nan``.
        """
        where = f"{self!r}.cov"
        c.guard(self, "cov")
        c.check_unbiased(where, unbiased)
        c.check_known(where, self.names, (a,) if b is None else (a, b))
        self._check_failed_particles(where, (a,) if b is None else (a, b))
        F_a = self._factor_row(a, unbiased)
        if b is None or b == a:
            return PSDLowRank(F_a)
        return product(Dense(F_a), Dense(self._factor_row(b, unbiased).T))

    def project(self, *, unbiased: bool = True):
        r"""The moment-matching Gaussian: this ensemble's mean and covariance.

        The covariance is :meth:`cov`'s, with the same ``unbiased``.

        For an unweighted ensemble the result is an
        :class:`~enskit.distribution.EnsembleGaussian` with :math:`k = J`,
        no independent terms, and divisor :math:`\delta = J - 1` (unbiased)
        or :math:`J`:

        .. math::

            m_b = \bar x^{(b)}, \qquad
            F_b = \frac{1}{\sqrt{\delta}} \big[a_1^{(b)}, \dots, a_J^{(b)}\big]
            \in \mathbb R^{d_b \times J},

        each row a :class:`~enskit.linalg.Dense`. Its latent coordinate
        :math:`j` belongs to particle :math:`j`, which is what lets
        :meth:`~enskit.distribution.EnsembleGaussian.realize_particles`
        return these particles; the result carries :math:`\delta` for that
        purpose.

        For a weighted ensemble the result is a plain
        :class:`~enskit.distribution.Gaussian` with :math:`k = J` and

        .. math::

            m_b = \sum_j w_j x_j^{(b)}, \qquad
            (F_b)_{:,j} = \frac{\sqrt{w_j}\,a_j^{(b)}}{\sqrt{\delta_w}},

        with :math:`\delta_w = 1 - \sum_i w_i^2` (unbiased) or :math:`1`,
        :math:`\sqrt{w_j} = \exp\big((\ell_j - \operatorname{lse}(\ell))/2\big)`
        and :math:`1 - \sum_i w_i^2 = -\operatorname{expm1}\big(
        \operatorname{lse}(2\ell) - 2\operatorname{lse}(\ell)\big)`,
        so a particle of weight zero contributes an exact zero column and
        the divisor stays accurate when one weight dominates.

        Parameters
        ----------
        unbiased : bool
            Keyword-only. ``True`` (the default) for the divisor
            :math:`J - 1` or :math:`1 - \sum_j w_j^2`, ``False`` for the
            moments of :math:`\hat p` itself, divisor :math:`J` or :math:`1`.

        Returns
        -------
        EnsembleGaussian or Gaussian
            Over the same blocks, in the same order; an
            :class:`~enskit.distribution.EnsembleGaussian` exactly when the
            ensemble is unweighted, whichever the divisor.

        Raises
        ------
        TypeError
            If ``unbiased`` is not a ``bool``.
        ValueError
            In debug mode, if the weights are concentrated on one particle to
            working precision (:math:`1 - \sum_i w_i^2` rounds to zero) and
            ``unbiased`` is true, where the weighted covariance is undefined;
            and if a particle of positive weight is not finite. Otherwise the
            means or factor rows are then ``inf`` or ``nan``.

        Notes
        -----
        Projection is the one place a Gaussian approximation of particles
        enters, and it is always written at the call site rather than hidden
        inside a conditioning routine.
        """
        from ._gaussian import EnsembleGaussian, Gaussian

        where = f"{self!r}.project"
        c.guard(self, "project")
        c.check_unbiased(where, unbiased)
        self._check_failed_particles(where, self.names)
        if self.is_weighted and unbiased:
            c.lazy_value_check(
                self._weight_pieces()[2],
                lambda d: d > 0,
                lambda: (
                    f"{where}: the weights are concentrated on one particle to "
                    f"working precision (effective sample size "
                    f"{float(_ess(self._log_weights)):.17g}), so the unbiased "
                    f"weighted covariance is undefined; unbiased=False gives the "
                    f"empirical distribution's covariance"
                ),
            )
        means = tuple(self.mean(n) for n in self.names)
        factors = tuple(Dense(self._factor_row(n, unbiased)) for n in self.names)
        covs = (None,) * len(self.names)
        J = self.n_particles
        if self.is_weighted:
            return c.build(
                Gaussian, names=self.names, latent_dim=J,
                _means=means, _factors=factors, _block_covs=covs,
            )
        return c.build(
            EnsembleGaussian, names=self.names, latent_dim=J, n_particles=J,
            unbiased=unbiased, _means=means, _factors=factors, _block_covs=covs,
        )

    def __repr__(self) -> str:
        """Type name and static sizes, never array contents; never raises."""
        return c.safe_repr(
            lambda: (
                f"Ensemble(n_particles={self.n_particles}, blocks={self.dims}, "
                f"weighted={self.is_weighted})"
            ),
            "Ensemble",
            lambda: self.batch_shape,
        )

    # -- private ---------------------------------------------------------------------
    @property
    def _dtype(self):
        return self._blocks[0].dtype

    def _block(self, where: str, name: str) -> Array:
        c.check_known(where, self.names, (name,))
        return self._blocks[self.names.index(name)]

    def _with(self, pairs) -> Ensemble:
        return c.build(
            Ensemble,
            names=tuple(n for n, _ in pairs),
            _blocks=tuple(a for _, a in pairs),
            _log_weights=self._log_weights,
        )

    def _reference(self, x: Array) -> Array:
        """The reference particle ``x_r``, ``(1, d)``: the first one when
        unweighted, one of largest weight when weighted, so never one of
        weight zero."""
        if self._log_weights is None:
            return x[..., :1, :]
        r = jnp.argmax(self._log_weights, axis=-1)
        return jnp.take(x, r[..., None], axis=-2)

    def _positive_weight(self) -> Array:
        """``(J,)`` boolean: ``w_j > 0``, as :attr:`weights` reports it."""
        return jax.nn.softmax(self._log_weights) > 0

    def _mean_of_differences(self, x: Array) -> Array:
        """The weighted mean of ``x_j - x_r``, with the particle axis kept.

        Particles of weight zero are masked out of the differences before
        they are weighted: ``0 * nan`` is ``nan``, and masking the product
        instead would still give a ``nan`` derivative with respect to the
        log weights.
        """
        diffs = x - self._reference(x)
        if self._log_weights is None:
            return jnp.mean(diffs, axis=-2, keepdims=True)
        w = jax.nn.softmax(self._log_weights)[..., :, None]
        diffs = jnp.where(self._positive_weight()[..., :, None], diffs, 0)
        return jnp.sum(w * diffs, axis=-2, keepdims=True)

    def _weight_pieces(self):
        """``(w, sqrt(w), 1 - sum(w**2))``, each computed in log space."""
        lw = self._log_weights
        lse = _lse(lw)
        w = jnp.exp(lw - lse)
        sqrt_w = jnp.exp((lw - lse) / 2)
        divisor = -jnp.expm1(_lse(2 * lw) - 2 * lse)
        return w, sqrt_w, divisor

    def _factor_row(self, name: str, unbiased: bool) -> Array:
        """The projected factor row of block ``name``, ``(d, J)``."""
        a = self.anomalies(name)
        if self._log_weights is None:
            return a.T / math.sqrt(c.particle_divisor(self.n_particles, unbiased))
        _, sqrt_w, divisor = self._weight_pieces()
        a = jnp.where(self._positive_weight()[:, None], a, 0)
        row = (sqrt_w[:, None] * a).T
        return row / jnp.sqrt(divisor) if unbiased else row

    def _check_failed_particles(self, where: str, names) -> None:
        """Debug-mode check that every particle of positive weight is finite
        in the named blocks."""

        def failed(a):
            """``(J,)`` boolean: a particle of positive weight is not finite."""
            out = ~jnp.all(jnp.isfinite(a), axis=-1)
            if self._log_weights is not None:
                out = out & self._positive_weight()
            return out

        for name in names:
            x = self._block(where, name)
            c.lazy_value_check(
                x,
                lambda a: ~jnp.any(failed(a)),
                lambda x=x, name=name: (
                    f"{where}: {int(jnp.sum(failed(x)))} "
                    f"particle(s) of positive weight are not finite in block "
                    f"{name!r}. Failed particles are found with all_finite; "
                    f"repair them, or give them weight zero with "
                    f"reweight(ensemble, jnp.where(ensemble.all_finite, 0.0, "
                    f"-jnp.inf))"
                ),
            )


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _check_first_block(where: str, name: str, arr) -> None:
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        raise TypeError(
            f"{where}: block {name!r} must be a real floating array, got dtype "
            f"{arr.dtype}"
        )
    if arr.ndim != 2:
        raise ValueError(
            f"{where}: block {name!r} must be a (n_particles, d) array of rank "
            f"exactly 2, got shape {arr.shape}. Ensembles are unbatched; a family "
            f"is a jax.vmap."
        )
    if arr.shape[1] < 1:
        raise ValueError(f"{where}: block {name!r} has no columns, shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(
            f"{where}: at least 2 particles are required, got {arr.shape[0]}. A "
            f"single particle has no anomalies."
        )


def _check_block(where: str, name: str, arr, n_particles: int, dtype) -> None:
    if not jnp.issubdtype(arr.dtype, jnp.floating):
        raise TypeError(
            f"{where}: block {name!r} must be a real floating array, got dtype "
            f"{arr.dtype}"
        )
    if arr.ndim != 2 or arr.shape[0] != n_particles or arr.shape[1] < 1:
        raise ValueError(
            f"{where}: block {name!r} must have shape ({n_particles}, d) with "
            f"d >= 1, got {arr.shape}"
        )
    if arr.dtype != dtype:
        raise TypeError(
            f"{where}: block {name!r} has dtype {arr.dtype}, but the blocks have "
            f"dtype {dtype}; every block shares one dtype"
        )


def _renamed(where: str, names: tuple[str, ...], mapping, new_names) -> tuple[str, ...]:
    """The block names after renaming ``old=new`` pairs; validated."""
    pairs = c.merge_blocks(where, mapping, new_names)
    c.check_known(where, names, pairs)
    for new in pairs.values():
        if not isinstance(new, str):
            raise TypeError(f"{where}: new names must be str, got {type(new).__name__}")
    targets = list(pairs.values())
    if len(set(targets)) != len(targets):
        raise ValueError(f"{where}: the new names {tuple(targets)} are not distinct")
    for new in targets:
        if new in names and new not in pairs:
            raise ValueError(
                f"{where}: the new name {new!r} collides with a block that is not "
                f"being renamed"
            )
    return tuple(pairs.get(n, n) for n in names)


def _ess(log_weights: Array) -> Array:
    """``exp(2 lse(l) - lse(2 l))``."""
    return jnp.exp(2 * _lse(log_weights) - _lse(2 * log_weights))


def _lse(x: Array) -> Array:
    """``log(sum(exp(x)))``, accurate when one entry dominates.

    The largest entry is factored out and the rest summed through ``log1p``:
    ``log(1 + t)`` of a small ``t`` loses ``eps / t`` relative accuracy,
    which ``-expm1(lse(2 l) - 2 lse(l))`` would then expose, since its two
    terms nearly cancel.
    """
    i = jnp.argmax(x, axis=-1)
    m = x[i]
    others = jnp.where(jnp.arange(x.shape[-1]) == i, 0.0, jnp.exp(x - m))
    return m + jnp.log1p(jnp.sum(others, axis=-1))
