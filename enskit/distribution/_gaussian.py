r"""The joint Gaussian :class:`Gaussian` and its subclass :class:`EnsembleGaussian`."""
from __future__ import annotations

import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import Array

from ..linalg import (
    Dense,
    IdentityPlusGram,
    LinOp,
    LowRankUpdate,
    PSDLinOp,
    PSDLowRank,
    Zero,
    block_diag,
    dense_matvec,
    hstack,
    product,
    static_field,
    tri_solve,
    value_check,
)
from . import _common as c
from ._common import distribution_class
from ._ensemble import Ensemble, _renamed
from ._regression import Regression, regression

__all__ = ["Gaussian", "EnsembleGaussian"]


@distribution_class
class Gaussian:
    r"""A joint Gaussian distribution over named vector blocks.

    Each block :math:`b` has a mean :math:`m_b`, a row :math:`F_b` of a
    **shared factor** and an **independent term** :math:`D_b`, either of
    which may be absent:

    .. math::

        x_b = m_b + F_b\,\xi + e_b, \qquad
        \xi \sim \mathcal N(0, I_k), \qquad
        e_b \sim \mathcal N(0, D_b),

    with the :math:`e_b` independent of :math:`\xi` and of each other, so
    that

    .. math::

        \operatorname{cov}(x_a, x_b) = F_a F_b^\top \ (a \ne b), \qquad
        \operatorname{cov}(x_b, x_b) = F_b F_b^\top + D_b .

    :math:`F_b` is an operator of shape ``(d_b, k)``; every block shares the
    latent width :math:`k`, which may be zero. :math:`D_b` is a positive
    semi-definite operator of shape ``(d_b, d_b)``. A block with no factor
    row is independent of the latent vector; a block with no independent
    term has :math:`D_b = 0`; a block must have at least one of the two.

    This one representation covers every Gaussian an ensemble method meets:

    =================================  =============================  ==================
    distribution                       shared factor                  independent terms
    =================================  =============================  ==================
    a prior :math:`\mathcal N(m, C)`   none (:math:`k = 0`)           :math:`C`
    an ensemble's moment match         :math:`A^\top/\sqrt{\delta}`   none
    the same, plus known noise on y    :math:`A^\top/\sqrt{\delta}`   :math:`R` on ``y``
    a linear-Gaussian joint (u, y)     :math:`[L;\ HL]`               :math:`R` on ``y``
    =================================  =============================  ==================

    where :math:`A` holds an ensemble's anomalies row-wise, :math:`\delta` is
    the divisor of :meth:`Ensemble.project`, :math:`LL^\top = C` and
    :math:`H` is a linear map. The moment match
    is an :class:`EnsembleGaussian`, the subclass whose latent coordinates
    are an ensemble's particles.

    Parameters
    ----------
    means : Mapping[str, Array]
        One exactly 1-D ``(d_b,)`` array per block, ``d_b >= 1``, all of one
        real floating dtype. Its keys are the block names, and its order is
        the block order.
    factors : Mapping[str, LinOp or Array], optional
        Keyword-only. Factor rows of shape ``(d_b, k)`` with one shared
        ``k``, keyed by a subset of the block names. Arrays are wrapped in
        :class:`~enskit.linalg.Dense`.
    block_covs : Mapping[str, PSDLinOp], optional
        Keyword-only. Independent terms of shape ``(d_b, d_b)``, keyed by a
        subset of the block names.
    latent_dim : int, optional
        Keyword-only. The latent width :math:`k`, a Python ``int >= 0``;
        inferred from the factor rows when omitted, and 0 when there are
        none. Given together with factor rows, it must equal their width.

    Raises
    ------
    ValueError
        If ``means`` is empty, a mean is not 1-D, a factor row or term
        disagrees with its block's dimension, the rows disagree on ``k`` or
        with ``latent_dim``, a key of ``factors`` or ``block_covs`` names no
        block, a block has neither a factor row nor a term, or an operator is
        a vmapped family. In debug mode, also if a mean is not finite.
    TypeError
        If a factor row is not a :class:`~enskit.linalg.LinOp` or floating
        array, a term is not a :class:`~enskit.linalg.PSDLinOp`, the means
        are not floating or their dtypes differ, or ``latent_dim`` is not an
        ``int``.

    Notes
    -----
    The constructor stores what it is given and computes nothing. Methods
    that keep the latent space return a Gaussian of the same kind, so an
    :class:`EnsembleGaussian` stays one; :meth:`absorb`, :meth:`compress`,
    and :meth:`add_noise` on a block that already has a term return a plain
    :class:`Gaussian`.

    ``factors`` and ``block_covs`` are keyword-only because a factor row may
    be any operator, a PSD one included: passed in the wrong position, a
    covariance :math:`C` would become a factor row with covariance
    :math:`CC^\top`, silently, whenever :math:`k = d_b`.

    The representation is a factor rather than three covariance blocks
    because a factor keeps every block's covariance and every
    cross-covariance consistent by construction, keeps sample covariances low
    rank, and gives conditioning a form that never squares the factor. The
    independent terms are kept apart because they carry known structure: a
    noise covariance used only through ``whiten``, or a prior with a
    Kronecker or diagonal form.
    """

    names: tuple[str, ...] = static_field()
    latent_dim: int = static_field()
    _means: tuple[Array, ...]
    _factors: tuple[LinOp | None, ...]
    _block_covs: tuple[PSDLinOp | None, ...]

    def __init__(self, means, *, factors=None, block_covs=None, latent_dim=None) -> None:
        _init(self, "Gaussian", means, factors, block_covs, latent_dim, "latent_dim")

    # -- construction --------------------------------------------------------------
    @classmethod
    def independent(cls, blocks=None, /, **block_specs) -> Gaussian:
        r"""Mutually independent blocks, each given as a ``(mean, cov)`` pair.

        .. math::

            x_b \sim \mathcal N(m_b, C_b) \ \text{independently for each } b .

        The result is a plain :class:`Gaussian` with no shared factor
        (:math:`k = 0`); each covariance becomes that block's independent
        term. This is how a prior is written:
        ``Gaussian.independent(u=(m0, C0))``.

        Parameters
        ----------
        blocks : Mapping[str, tuple[Array, PSDLinOp]], optional
            Positional-only. Each block's ``(mean, cov)``.
        **block_specs : tuple[Array, PSDLinOp]
            The same, as keywords.

        Raises
        ------
        TypeError
            If a specification is not a ``(mean, cov)`` pair, or as for the
            constructor.
        ValueError
            As for the constructor.
        """
        where = "Gaussian.independent"
        specs = c.merge_blocks(where, blocks, block_specs)
        means, covs = {}, {}
        for name, spec in specs.items():
            if not isinstance(spec, tuple) or len(spec) != 2:
                raise TypeError(
                    f"{where}: block {name!r} must be given as a (mean, cov) pair, "
                    f"got {type(spec).__name__}"
                )
            means[name], covs[name] = spec
        return Gaussian(means, block_covs=covs, latent_dim=0)

    # -- introspection ---------------------------------------------------------------
    @property
    def dims(self) -> dict[str, int]:
        """Each block's dimension, keyed by name, in order."""
        return {n: int(m.shape[-1]) for n, m in zip(self.names, self._means, strict=True)}

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """The family's batch shape, ``()`` for a directly constructed object."""
        shapes = [tuple(m.shape[:-1]) for m in self._means]
        shapes += [op.batch_shape for op in self._factors if op is not None]
        shapes += [op.batch_shape for op in self._block_covs if op is not None]
        return c.broadcast_batch(type(self).__name__, *shapes)

    def mean(self, name: str) -> Array:
        """Block ``name``'s mean, ``(d,)``.

        Raises
        ------
        KeyError
            If there is no such block.
        """
        c.guard(self, "mean")
        return self._means[self._index(f"{self!r}.mean", name)]

    def factor(self, name: str) -> LinOp | None:
        """Block ``name``'s row of the shared factor, ``(d, k)``, or ``None``.

        Raises
        ------
        KeyError
            If there is no such block.
        """
        c.guard(self, "factor")
        return self._factors[self._index(f"{self!r}.factor", name)]

    def block_cov(self, name: str) -> PSDLinOp | None:
        """Block ``name``'s independent term, ``(d, d)``, or ``None``.

        Raises
        ------
        KeyError
            If there is no such block.
        """
        c.guard(self, "block_cov")
        return self._block_covs[self._index(f"{self!r}.block_cov", name)]

    def cov(self, a: str, b: str | None = None) -> LinOp:
        r"""The covariance of block ``a`` (with ``b``, if given), as an operator.

        Nothing of size ``(d_a, d_b)`` is formed by the call.

        ===============================  ==========================================
        block ``a`` has                  ``cov(a)``
        ===============================  ==========================================
        :math:`F_a` and :math:`D_a`      :class:`~enskit.linalg.LowRankUpdate`
                                         ``(D_a, F_a)``, :math:`D_a + F_aF_a^\top`
        :math:`D_a` only                 :math:`D_a` itself
        :math:`F_a` only                 :class:`~enskit.linalg.PSDLowRank` of
                                         ``F_a.to_dense()``, materializing the row
        ===============================  ==========================================

        For ``a != b``, ``cov(a, b)`` is :math:`F_a F_b^\top` as
        ``product(F_a, F_b.T)``, or a :class:`~enskit.linalg.Zero` of the
        distribution's dtype when either row is absent. ``cov(a, a)`` is
        ``cov(a)``.

        Raises
        ------
        KeyError
            If a name is not a block.
        UnsupportedOpError
            For ``cov(a)`` of a block with both parts whose term cannot
            ``whiten``, which :class:`~enskit.linalg.LowRankUpdate` needs.

        Notes
        -----
        The ``LowRankUpdate``'s ``solve`` is a Woodbury solve that loses
        accuracy when :math:`F_aF_a^\top` dominates :math:`D_a` by many
        orders of magnitude; its ``whiten`` and ``logdet`` do not.
        """
        where = f"{self!r}.cov"
        c.guard(self, "cov")
        c.check_known(where, self.names, (a,) if b is None else (a, b))
        i = self.names.index(a)
        F_a, D_a = self._factors[i], self._block_covs[i]
        if b is None or b == a:
            if F_a is not None and D_a is not None:
                return LowRankUpdate(D_a, F_a)
            if D_a is not None:
                return D_a
            return PSDLowRank(F_a.to_dense())
        F_b = self._factors[self.names.index(b)]
        if F_a is None or F_b is None:
            return Zero(self.dims[a], self.dims[b], dtype=self._dtype)
        return product(F_a, F_b.T)

    # -- structure ---------------------------------------------------------------------
    def marginal(self, *names: str) -> Gaussian:
        """The marginal over ``names``, in the order given.

        Exact and free: the named blocks' means, factor rows and terms are
        kept and the rest discarded. The latent width is unchanged, and the
        result is of the same kind.

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
        return self._select(names)

    def drop(self, *names: str) -> Gaussian:
        """The marginal over every block not in ``names``, in block order.

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
        kept = tuple(n for n in self.names if n not in dropped)
        if not kept:
            raise ValueError(f"{where}: at least one block must remain")
        return self._select(kept)

    def rename(self, mapping=None, /, **new_names) -> Gaussian:
        """Rename blocks (``old=new``); positions and the latent space are kept.

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
        return self._with(names, self._means, self._factors, self._block_covs)

    def pipe(self, f, *args, **kwargs):
        """Return ``f(self, *args, **kwargs)``; see :meth:`Ensemble.pipe`."""
        c.guard(self, "pipe")
        return f(self, *args, **kwargs)

    def add_noise(self, covs=None, /, **block_covs) -> Gaussian:
        r"""Add independent Gaussian noise to the named blocks, in place.

        .. math::

            x_b \mapsto x_b + e_b, \qquad e_b \sim \mathcal N(0, R_b)
            \ \text{independent of everything else,}

        for each named block :math:`b`, with :math:`R_b` given as a mapping
        or as keywords: ``g.add_noise(y=R)``.

        On a block with no independent term this sets :math:`D_b = R_b` and
        changes nothing else: the latent space is unchanged, so an
        :class:`EnsembleGaussian` stays one, and :math:`R_b` is used
        afterward only through what later operations ask of it. On a block
        that already has one, the existing term is first moved into the
        shared factor (:meth:`absorb`), which needs its ``factor``, and
        :math:`R_b` becomes the new term; the result is then a plain
        :class:`Gaussian`.

        The noise-free block is not kept. To keep it, push it to a new block
        through :class:`enskit.maps.AdditiveNoise` instead.

        Raises
        ------
        KeyError
            If a name is not a block.
        TypeError
            If a covariance is not a :class:`~enskit.linalg.PSDLinOp`, or a
            block is given twice.
        ValueError
            If no block is given, a covariance's side is not its block's
            dimension, or it is a vmapped family.
        UnsupportedOpError
            If an existing term to absorb cannot ``factor``.
        """
        where = f"{self!r}.add_noise"
        c.guard(self, "add_noise")
        noise = c.merge_blocks(where, covs, block_covs)
        if not noise:
            raise ValueError(f"{where}: at least one block is required")
        c.check_known(where, self.names, noise)
        for name, R in noise.items():
            _check_term(where, name, R)
        absorbed = tuple(
            n
            for n in self.names
            if n in noise and self._block_covs[self.names.index(n)] is not None
        )
        for name in absorbed:
            c.require(self._block_covs[self.names.index(name)], "factor")
        for name, R in noise.items():
            _check_term_shape(where, name, R, self.dims[name])
        g = self._absorb(absorbed) if absorbed else self
        terms = tuple(
            noise.get(n, D) for n, D in zip(g.names, g._block_covs, strict=True)
        )
        return g._with(g.names, g._means, g._factors, terms)

    def absorb(self, *names: str) -> Gaussian:
        r"""Move the independent terms of ``names`` into the shared factor.

        The distribution is unchanged; the representation now lets a later
        map of block :math:`b` stay correlated with :math:`b`. For the named
        blocks in block order, with :math:`L_b` = ``D_b.factor()`` of width
        :math:`w_b`, the latent space grows to :math:`k' = k + \sum_b w_b`,
        block :math:`b`'s columns appended in that order:

        .. math::

            F_b \mapsto \big[\,F_b \ \ 0 \ \ L_b \ \ 0\,\big], \qquad
            F_a \mapsto \big[\,F_a \ \ 0\,\big] \ (a \text{ not named}),
            \qquad D_b \mapsto 0 .

        The zero columns are :class:`~enskit.linalg.Zero` blocks of the
        distribution's dtype inside an :class:`~enskit.linalg.HStack`; they
        store no array and are skipped when applied. A block with no factor
        row and no term named keeps no row, and a row that would be a single
        operator is that operator itself.

        Returns
        -------
        Gaussian
            A plain :class:`Gaussian`: the latent coordinates are no longer
            particles.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If no name is given, a name is repeated, or a named block has no
            independent term.
        UnsupportedOpError
            If a named term cannot ``factor``; checked for every named block
            before any work.
        """
        where = f"{self!r}.absorb"
        c.guard(self, "absorb")
        names = c.check_name_list(where, self.names, names)
        if not names:
            raise ValueError(f"{where}: at least one block name is required")
        for name in names:
            if self._block_covs[self.names.index(name)] is None:
                raise ValueError(
                    f"{where}: block {name!r} has no independent term to absorb"
                )
        ordered = tuple(n for n in self.names if n in names)
        for name in ordered:
            c.require(self._block_covs[self.names.index(name)], "factor")
        return self._absorb(ordered)

    def compress(self, *, max_dim: int = 4096) -> Gaussian:
        r"""Re-factor the shared factor to width at most the rows' total dimension.

        Let :math:`F` be the ``(D_F, k)`` array stacking every factor row,
        :math:`D_F` the total dimension of the blocks that have one. When
        :math:`k > D_F`, with the thin QR decomposition :math:`F^\top = QR`,

        .. math::

            F F^\top = R^\top R, \qquad F \mapsto R^\top \in \mathbb R^{D_F \times D_F},

        and the rows of :math:`R^\top`, as :class:`~enskit.linalg.Dense`,
        replace the factor rows. When :math:`k \le D_F` nothing is computed.
        The distribution is unchanged either way.

        Use it where repeated :meth:`absorb` would otherwise grow the latent
        width without bound.

        Parameters
        ----------
        max_dim : int
            Keyword-only. A Python ``int >= 1``.

        Returns
        -------
        Gaussian
            Always a plain :class:`Gaussian`, whether or not it compressed.

        Raises
        ------
        ValueError
            Before allocating, when compression would happen
            (:math:`k > D_F`) and :math:`D_F >` ``max_dim`` or
            :math:`D_F k >` ``max_dim**2``; or if ``max_dim < 1``.
        TypeError
            If ``max_dim`` is not an ``int``.

        Notes
        -----
        Compression materializes every factor row, so a structured row (a
        Kronecker factor, say) does not survive it. When no block has a
        factor row but :math:`k > 0`, as for a marginal over blocks without
        rows, :math:`D_F = 0` and the result has latent width 0.
        """
        where = f"{self!r}.compress"
        c.guard(self, "compress")
        if type(max_dim) is not int:
            raise TypeError(
                f"{where}: max_dim must be a Python int, got {type(max_dim).__name__}"
            )
        if max_dim < 1:
            raise ValueError(f"{where}: max_dim must be at least 1, got {max_dim}")
        rows = [
            (n, F)
            for n, F in zip(self.names, self._factors, strict=True)
            if F is not None
        ]
        D_F, k = sum(F.shape[0] for _, F in rows), self.latent_dim
        if k <= D_F:
            return self._with(
                self.names, self._means, self._factors, self._block_covs, plain=True
            )
        if D_F == 0:
            return self._with(
                self.names, self._means, self._factors, self._block_covs,
                latent_dim=0, plain=True,
            )
        if D_F > max_dim or D_F * k > max_dim**2:
            raise ValueError(
                f"{where}: compressing stacks a ({D_F}, {k}) factor, over "
                f"max_dim={max_dim}; raise max_dim deliberately if the cost is wanted"
            )
        F = jnp.concatenate([row.to_dense() for _, row in rows], axis=0)
        _, R = jnp.linalg.qr(F.T)
        Rt = R.T
        new_rows, offset = {}, 0
        for name, row in rows:
            d = row.shape[0]
            new_rows[name] = Dense(Rt[offset : offset + d])
            offset += d
        factors = tuple(new_rows.get(n) for n in self.names)
        return self._with(
            self.names, self._means, factors, self._block_covs, latent_dim=D_F, plain=True
        )

    # -- conditioning -----------------------------------------------------------------
    def condition(self, values=None, /, **block_values) -> Gaussian:
        r"""The exact conditional distribution of the other blocks given ``values``.

        Two cases, chosen by the given blocks' structure.

        **Noisy values: every given block has an independent term.** Write
        :math:`F_c` and :math:`D_c` for the given blocks' stacked factor rows
        (a block without one contributes zero rows) and block-diagonal terms,
        :math:`W` for the whitener :math:`D_c`'s ``whiten`` applies, and

        .. math::

            S = (W F_c)^\top \in \mathbb R^{k \times N}, \qquad
            S = U \Sigma V^\top \ \text{(thin SVD)}, \qquad A = I_k + S S^\top .

        For each target block :math:`x` with a factor row,

        .. math::

            m_x' = m_x + F_x w, \qquad
            w = A^{-1} S\, W(y^* - m_c)
              = U \operatorname{diag}\!\Big(\frac{\sigma_i}{1+\sigma_i^2}\Big)
                V^\top W(y^* - m_c), \qquad
            F_x' = F_x T, \qquad T = A^{-1/2},

        and its term is unchanged. One
        :class:`~enskit.linalg.IdentityPlusGram` serves both; :math:`F_x'` is
        the product operator ``product(F_x, T)``, so a structured row is never
        densified, and repeated conditioning nests products
        (:meth:`compress` collapses them). A target with no factor row is
        independent of the given blocks and is returned unchanged; at
        :math:`k = 0` the result is the marginal over the targets.

        **Exact values: no given block has an independent term.** With the
        thin QR decomposition :math:`F_c^\top = QR`,
        :math:`Q \in \mathbb R^{k \times N}`,

        .. math::

            m_x' = m_x + F_x w, \qquad w = Q R^{-\top}(y^* - m_c), \qquad
            F_x' = F_x (I_k - Q Q^\top),

        with :math:`F_x'` materialized as the ``(d_x, k)`` array
        :math:`F_x - (F_x Q) Q^\top`, a :class:`~enskit.linalg.Dense`. It
        requires :math:`F_c` of full row rank :math:`N`, so :math:`N \le k`,
        and :math:`N \le J - 1` for an :class:`EnsembleGaussian`.

        Parameters
        ----------
        values : Mapping[str, Array], optional
            Positional-only. The given blocks and their values, each exactly
            ``(d_c,)``. A family of values is a :func:`jax.vmap` over this
            method.
        **block_values : Array
            The same, as keywords: ``g.condition(y=y_obs)``.

        Returns
        -------
        Gaussian
            Over the remaining blocks, in block order, of the same kind and
            on the same latent space. An :class:`EnsembleGaussian` stays one:
            :math:`T\mathbf 1 = \mathbf 1` and
            :math:`(I - QQ^\top)\mathbf 1 = \mathbf 1` when the factor is
            centered.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If no block is given or none would remain, a value's shape is not
            ``(d_c,)``, some but not all given blocks have independent terms,
            or the exact case's size conditions fail. In debug mode, also if
            a value is not finite, the exact case is rank deficient
            (:math:`\min_i |R_{ii}| \le \max(N, k)\,\varepsilon\max_i|R_{ii}|`),
            a whitened quantity is not finite (naming the given block), or a
            result is not finite.
        TypeError
            If a block is given twice.
        UnsupportedOpError
            If a given block's term cannot ``whiten``.

        Notes
        -----
        Each given block's factor row and residual are whitened together,
        differenced first: ``W [F_c | y* - m_c]`` in one ``whiten_mat`` call,
        :math:`k + 1` vectors. Whitening before differencing loses accuracy
        in proportion to :math:`\sqrt{\kappa(D_c)}` when the mean lies along a
        precise direction of the noise. A given block with no factor row
        whitens its residual alone.

        Neither :math:`S S^\top` nor :math:`S^\top S` is formed: forming
        either rounds away every :math:`\sigma_i < \sqrt\varepsilon\,
        \sigma_{\max}`, which are the singular values with the largest gain
        multipliers. The multiplier :math:`\sigma/(1 + \sigma^2)` is at most
        1/2, so the gain is bounded however collapsed the factor is.

        When :math:`k > N` the conditional covariance loses about
        :math:`\varepsilon\sigma_{\max}` relative accuracy in the directions
        the given blocks constrain, from the thin basis of
        :class:`~enskit.linalg.IdentityPlusGram`; the mean does not.
        """
        where = f"{self!r}.condition"
        c.guard(self, "condition")
        vals = c.merge_blocks(where, values, block_values)
        if not vals:
            raise ValueError(f"{where}: at least one block value is required")
        c.check_known(where, self.names, vals)
        given = tuple(n for n in self.names if n in vals)
        targets = tuple(n for n in self.names if n not in vals)
        noisy = self._given_kind(where, given)
        if not noisy:
            self._check_exact_sizes(where, given)
        if not targets:
            raise ValueError(f"{where}: at least one block must remain as a target")
        if noisy:
            for name in given:
                c.require(self._term(name), "whiten")
        y = self._values(where, given, vals)
        if noisy:
            return self._condition_noisy(where, given, targets, y)
        return self._condition_exact(where, given, targets, y)

    def conditional_map(self, given):
        r"""The exact conditional of this Gaussian as a pointwise transport map.

        For the given blocks :math:`y` and the remaining blocks :math:`x`,
        the map is

        .. math::

            T_{y^*}(x, y) = x + K\,(y^* - y), \qquad
            K = \operatorname{cov}(x, y)\,\operatorname{cov}(y, y)^{-1},

        built now from the block *names*; the value :math:`y^*` is supplied
        each time the map is called. Applied to joint samples of this
        Gaussian it returns exact samples of the conditional at :math:`y^*`
        (Matheron's rule). The gain is never formed: the map holds the
        whitened factor's :class:`~enskit.linalg.IdentityPlusGram`, computed
        here from :math:`k` whitened vectors per given block.

        Parameters
        ----------
        given : str or sequence of str
            The names of the given blocks. Every one must have an independent
            term: there is no exact-value conditional map.

        Returns
        -------
        MatheronMap
            Its targets are every other block, in block order.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If a given block has no independent term, a name is repeated, or
            no target remains.
        UnsupportedOpError
            If a given block's term cannot ``whiten``.
        """
        from ._maps import MatheronMap

        where = f"{self!r}.conditional_map"
        c.guard(self, "conditional_map")
        given = c.given_names(where, self.names, given)
        targets = self._noisy_targets(where, given, "conditional_map")
        for name in given:
            c.require(self._term(name), "whiten")
        gram = None
        if self.latent_dim > 0:
            gram = IdentityPlusGram(self._whiten_given(where, given, None)[0])
        return c.build(
            MatheronMap,
            given=given,
            targets=targets,
            latent_dim=self.latent_dim,
            n_particles=getattr(self, "n_particles", None),
            divisor=getattr(self, "divisor", None),
            target_dims=tuple(self.dims[n] for n in targets),
            gram=gram,
            _given_covs=tuple(self._term(n) for n in given),
            _given_means=tuple(self._means[self.names.index(n)] for n in given),
            _target_rows=tuple(self._factors[self.names.index(n)] for n in targets),
        )

    def regression(self, target: str, *, given, min_norm: bool = False) -> Regression:
        r"""The regression of block ``target`` on the ``given`` blocks.

        The conditional distribution of :math:`y` = ``target`` given the
        blocks :math:`x` = ``given`` is Gaussian with a mean affine in the
        given values and a covariance that does not depend on them:

        .. math::

            y \mid x \sim \mathcal N(Ax + c,\ \Omega), \qquad
            A = C_{yx} C_{xx}^{-1}, \qquad c = m_y - A m_x, \qquad
            \Omega = C_{yy} - A C_{xy},

        where :math:`C` is this Gaussian's covariance and :math:`x` stacks
        the given blocks in block order. :math:`A` minimizes
        :math:`\mathbb E\lVert y - Ax - c\rVert^2`, and :math:`\Omega` is the
        covariance of the residual :math:`y - Ax - c`. For the projection of
        an ensemble (:meth:`Ensemble.project`) this is the least-squares
        regression of the particles' ``target`` block on their ``given``
        blocks.

        Write :math:`F_c` for the given blocks' stacked factor rows,
        :math:`N` for their total dimension, :math:`k` for the latent
        width, and :math:`F_y` for the target's factor row. Three cases,
        chosen by the given blocks' structure:

        **Every given block has an independent term** :math:`D_c`. With
        :math:`S = (WF_c)^\top` and its
        :class:`~enskit.linalg.IdentityPlusGram` :math:`I + SS^\top`, as
        in :meth:`condition`,

        .. math::

            A = F_y (I_k + SS^\top)^{-1} F_c^\top D_c^{-1}
              = C_{yx}(F_cF_c^\top + D_c)^{-1}, \qquad
            \Omega = F_y (I_k + SS^\top)^{-1} F_y^\top + D_y .

        Adding a term :math:`\Lambda` to the given blocks of an ensemble's
        projection, ``ens.project().add_noise(x=Lam)``, makes this the ridge
        regression :math:`A = \hat C_{yx}(\hat C_{xx} + \Lambda)^{-1}`.

        **No given block has a term, and** :math:`N \le r`, where
        :math:`r = J - 1` for an :class:`EnsembleGaussian` and :math:`r = k`
        otherwise. With the thin QR decomposition :math:`F_c^\top = QR`, the
        unique least-squares solution is

        .. math::

            A = F_y Q R^{-\top}, \qquad
            \Omega = F_y (I_k - QQ^\top) F_y^\top + D_y .

        **No given block has a term, and** :math:`N > r`. The coefficients
        are not unique: every :math:`A` with :math:`AF_c = F_y` fits
        exactly. With ``min_norm=True`` the result is the one of least
        Frobenius norm, :math:`A = F_y F_c^{+}`. Computed with :math:`H`, an
        orthonormal basis of :math:`\mathbf 1^\perp` for an
        :class:`EnsembleGaussian` (whose factor rows annihilate
        :math:`\mathbf 1`) and :math:`H = I_k` otherwise, and the thin QR
        decomposition :math:`F_c H = QR`:

        .. math::

            A = F_y H R^{-1} Q^\top, \qquad
            \Omega = (F_y - AF_c)(F_y - AF_c)^\top + D_y = D_y
            \ \text{(to round-off)}.

        Parameters
        ----------
        target : str
            The block regressed.
        given : str or sequence of str
            Keyword-only. The blocks it is regressed on, at least one, not
            including ``target``.
        min_norm : bool
            Keyword-only. Whether to return the minimum-norm solution when
            the coefficients are not unique. Ignored when they are.

        Returns
        -------
        Regression
            ``coefficients`` maps each given block, in block order, to its
            ``(d_y, d_b)`` operator :math:`A_b`; ``intercept`` is :math:`c`;
            ``residual_cov`` is :math:`\Omega`, as :meth:`cov` returns a
            block's covariance.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If ``target`` is also given, no block is given, a name is
            repeated, some but not all given blocks have independent terms,
            or the coefficients are not unique and ``min_norm`` is
            ``False``. In debug mode, also if the given factor rows (or,
            for the minimum-norm solution, their columns) are rank deficient
            to working precision, or a result is not finite.
        TypeError
            If ``target`` is not a ``str``, or ``min_norm`` not a ``bool``.
        UnsupportedOpError
            If a given block with a factor row has a term that cannot
            ``whiten`` and ``solve``.

        Notes
        -----
        The coefficients are held as operators of width at most :math:`k`,
        never as dense ``(d_y, d_b)`` arrays, except in the unique
        least-squares case, where :math:`N \le k`. The residual factor row
        is materialized as a ``(d_y, k)`` array in the cases without
        independent terms, as in :meth:`condition`.

        The minimum-norm solution depends on the coordinates of the given
        blocks: rescaling a block changes which interpolating :math:`A` has
        least norm. A ridge term :math:`\varepsilon M`, as
        :math:`\varepsilon \to 0`, selects the solution minimizing
        :math:`\operatorname{tr}(AMA^\top)` instead.

        With independent terms, the coefficients are computed as
        :math:`(I + SS^\top)^{-1}` applied to :math:`F_c^\top D_c^{-1}`, which
        loses relative accuracy like :math:`\varepsilon(1 + \sigma_{\max}^2)`
        for the largest singular value :math:`\sigma_{\max}` of :math:`S`:
        a ridge term that is tiny relative to the given blocks' spread is
        better dropped, with ``min_norm=True`` when :math:`N > r`.

        Rank is checked only in debug mode, as for :meth:`condition`. Outside
        it, given factor rows of lower rank than the case assumes give
        non-finite or wrong coefficients without raising: duplicated
        particles (after resampling, for example), a given coordinate that
        is constant across particles, an :class:`EnsembleGaussian`
        conditioned exactly on other blocks, or a weighted projection, whose
        rank is at most :math:`J - 1` and less with zero weights.
        """
        return regression(self, target, given, min_norm)

    def log_density(self, values=None, /, **block_values) -> Array:
        r"""Log density of the marginal over exactly the blocks in ``values``.

        Blocks not named are marginalized out, so ``g.log_density(a=v)`` is
        the density of the ``a`` marginal and ``g.log_density(a=v, b=w)``
        that of the joint.

        **Noisy values** (every named block has an independent term). With
        the notation of :meth:`condition` over the named blocks and
        :math:`C = D_c + F_cF_c^\top`,

        .. math::

            \log p(v) = -\tfrac12\Big(\lVert W_C(v - m_c)\rVert^2
              + \log\det D_c + \textstyle\sum_i \log(1 + \sigma_i^2)
              + N\log 2\pi\Big),
            \qquad W_C = (I_N + S^\top S)^{-1/2}\,W,

        computed through :class:`~enskit.linalg.LowRankUpdate` ``(D_c,
        F_c)``: the quadratic term is its ``whiten`` and the determinant its
        ``logdet``, from one decomposition. At :math:`k = 0`, or when no named
        block has a factor row, :math:`C = D_c`.

        **Exact values** (no named block has one). With
        :math:`F_c^\top = QR` and the size conditions of :meth:`condition`,

        .. math::

            \log p(v) = -\tfrac12\Big(\lVert R^{-\top}(v - m_c)\rVert^2
              + 2\textstyle\sum_i \log|R_{ii}| + N \log 2\pi\Big).

        Parameters
        ----------
        values : Mapping[str, Array], optional
            Positional-only. Each ``(*batch, d_b)``, with one batch shape
            shared by all of them; values are not broadcast against each
            other.
        **block_values : Array
            The same, as keywords.

        Returns
        -------
        Array
            Shape ``batch``; a 0-d array when unbatched, never a Python float.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If no block is named, the values' trailing sizes or batch shapes
            disagree, the named blocks are mixed, or the exact size
            conditions fail. In debug mode, also as for :meth:`condition`.
        UnsupportedOpError
            If a named term cannot ``whiten`` or ``logdet``, checked block by
            block, in that order.

        Notes
        -----
        The quadratic term is never computed as
        :math:`\lVert b\rVert^2 - \langle Sb, A^{-1}Sb\rangle` with
        :math:`b = W(v - m_c)`, the form the determinant identity suggests:
        the two terms nearly cancel, and the relative error grows like
        :math:`\varepsilon\sigma_{\max}^2`. The route through
        ``LowRankUpdate.whiten`` is exact to round-off when :math:`N \le k`,
        and loses about :math:`\varepsilon\sigma_{\max}` when :math:`N > k`.

        This is the evidence of values under a joint, and the objective for
        tuning hyperparameters: its first derivatives are finite at
        degenerate spectra.
        """
        where = f"{self!r}.log_density"
        c.guard(self, "log_density")
        vals = c.merge_blocks(where, values, block_values)
        if not vals:
            raise ValueError(f"{where}: at least one block value is required")
        c.check_known(where, self.names, vals)
        named = tuple(n for n in self.names if n in vals)
        noisy = self._given_kind(where, named)
        if not noisy:
            self._check_exact_sizes(where, named)
        else:
            for name in named:
                c.require(self._term(name), "whiten", "logdet")
        v = self._batched_values(where, named, vals)
        diff = jnp.concatenate(
            [v[n] - self._means[self.names.index(n)] for n in named], axis=-1
        )
        N = diff.shape[-1]
        if noisy:
            quad, logdet = self._noisy_density_terms(where, named, diff)
        else:
            Q, R = self._exact_qr(where, named)
            z = tri_solve(R, diff, lower=False, trans=1)
            quad = jnp.sum(z * z, axis=-1)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diagonal(R))))
        out = -0.5 * (quad + logdet + N * math.log(2.0 * math.pi))
        c.result_check(where, "log density", out)
        return out

    # -- sampling ----------------------------------------------------------------------
    def sample(self, key, n_particles: int) -> Ensemble:
        r"""Draw ``n_particles`` independent samples as an unweighted :class:`Ensemble`.

        .. math::

            x_j^{(b)} = m_b + F_b\,\xi_j + L_b\,\eta_j^{(b)}, \qquad
            \xi_j \sim \mathcal N(0, I_k),\quad
            \eta_j^{(b)} \sim \mathcal N(0, I_{w_b}),

        with :math:`L_b` = ``D_b.factor()`` of width :math:`w_b`, and either
        term absent when the block lacks it. Over every block, in block order.

        The draw is pinned: ``keys = jax.random.split(key, 1 + n_D)``, with
        :math:`n_D` the number of blocks with an independent term;
        ``normal(keys[0], (n_particles, k))`` for :math:`\xi` when
        :math:`k > 0`, and ``normal(keys[1 + i], (n_particles, w_b))`` for
        the :math:`i`-th block with a term, in block order, all in the
        distribution's dtype.

        Parameters
        ----------
        key
            A typed key (:func:`jax.random.key`), consumed whole.
        n_particles : int
            A Python ``int >= 2``, the :class:`Ensemble` minimum.

        Raises
        ------
        TypeError
            If ``n_particles`` is not an ``int``, or ``key`` is not a typed
            key.
        ValueError
            If ``n_particles < 2``, or ``key`` is ``None``.
        UnsupportedOpError
            If an independent term cannot ``factor``; checked before drawing.
        """
        where = f"{self!r}.sample"
        c.guard(self, "sample")
        c.check_n_particles(where, n_particles)
        c.check_key(where, key, required=True)
        for D in self._block_covs:
            if D is not None:
                c.require(D, "factor")
        n_D = sum(D is not None for D in self._block_covs)
        keys = jax.random.split(key, 1 + n_D)
        dtype = self._dtype
        xi = None
        if self.latent_dim > 0:
            xi = jax.random.normal(keys[0], (n_particles, self.latent_dim), dtype)
        blocks, i = [], 0
        for m, F, D in zip(self._means, self._factors, self._block_covs, strict=True):
            x = jnp.broadcast_to(m, (n_particles, m.shape[-1]))
            if F is not None:
                x = x + F.matvec(xi)
            if D is not None:
                L = D.factor()
                x = x + L.matvec(
                    jax.random.normal(keys[1 + i], (n_particles, L.shape[1]), dtype)
                )
                i += 1
            blocks.append(x)
        return c.build(
            Ensemble, names=self.names, _blocks=tuple(blocks), _log_weights=None
        )

    def __repr__(self) -> str:
        """Type name and static sizes, never array contents; never raises."""
        return c.safe_repr(
            lambda: (
                f"Gaussian(blocks={self.dims}, latent_dim={self.latent_dim}, "
                f"block_covs={self._term_names()})"
            ),
            "Gaussian",
            lambda: self.batch_shape,
        )

    # -- private ---------------------------------------------------------------------
    @property
    def _dtype(self):
        return self._means[0].dtype

    def _index(self, where: str, name: str) -> int:
        c.check_known(where, self.names, (name,))
        return self.names.index(name)

    def _term(self, name: str) -> PSDLinOp | None:
        return self._block_covs[self.names.index(name)]

    def _term_names(self) -> tuple[str, ...]:
        return tuple(
            n for n, D in zip(self.names, self._block_covs, strict=True) if D is not None
        )

    def _with(
        self, names, means, factors, covs, *, latent_dim=None, plain=False
    ) -> Gaussian:
        """A result of the same kind, built without the constructor."""
        fields = dict(
            names=tuple(names),
            latent_dim=self.latent_dim if latent_dim is None else latent_dim,
            _means=tuple(means),
            _factors=tuple(factors),
            _block_covs=tuple(covs),
        )
        if isinstance(self, EnsembleGaussian) and not plain:
            return c.build(
                EnsembleGaussian,
                n_particles=self.n_particles,
                unbiased=self.unbiased,
                **fields,
            )
        return c.build(Gaussian, **fields)

    def _select(self, names) -> Gaussian:
        idx = [self.names.index(n) for n in names]
        return self._with(
            names,
            [self._means[i] for i in idx],
            [self._factors[i] for i in idx],
            [self._block_covs[i] for i in idx],
        )

    def _given_kind(self, where: str, given) -> bool:
        """Whether the given blocks are noisy; ``ValueError`` when mixed."""
        has = [self._term(n) is not None for n in given]
        if all(has):
            return True
        if not any(has):
            return False
        noisy = tuple(n for n, h in zip(given, has, strict=True) if h)
        exact = tuple(n for n, h in zip(given, has, strict=True) if not h)
        raise ValueError(
            f"{where}: the given blocks are mixed: {noisy} have independent terms "
            f"and {exact} do not. Noisy and exact values do not combine in one call."
        )

    def _noisy_targets(self, where: str, given, method: str) -> tuple[str, ...]:
        for name in given:
            if self._term(name) is None:
                raise ValueError(
                    f"{where}: given block {name!r} has no independent term; a "
                    f"{method} needs one on every given block. For exact values, "
                    f"use condition(...)."
                )
        targets = tuple(n for n in self.names if n not in given)
        if not targets:
            raise ValueError(f"{where}: at least one block must remain as a target")
        return targets

    def _check_exact_sizes(self, where: str, given) -> None:
        N = sum(self.dims[n] for n in given)
        k = self.latent_dim
        if N > k:
            raise ValueError(
                f"{where}: exact values need the given factor rows to have full row "
                f"rank, so N <= k; got N = {N} given coordinates and latent width "
                f"k = {k}"
            )
        if isinstance(self, EnsembleGaussian) and N > self.n_particles - 1:
            raise ValueError(
                f"{where}: exact values on an EnsembleGaussian need N <= J - 1, "
                f"since centered factor rows have rank at most J - 1; got N = {N} "
                f"and J = {self.n_particles}"
            )

    def _values(self, where: str, given, vals) -> dict:
        y = {
            n: c.check_value(where, n, vals[n], self.dims[n], self._dtype) for n in given
        }
        for name in given:
            c.check_finite(where, f"the value of block {name!r}", y[name])
        return y

    def _batched_values(self, where: str, named, vals) -> dict:
        out, batch = {}, None
        for name in named:
            arr = jnp.asarray(vals[name])
            c.check_real(where, f"the value of block {name!r}", arr)
            d = self.dims[name]
            if arr.ndim < 1 or arr.shape[-1] != d:
                raise ValueError(
                    f"{where}: the value of block {name!r} must have shape "
                    f"(*batch, {d}), got {arr.shape}"
                )
            if batch is None:
                batch = arr.shape[:-1]
            elif arr.shape[:-1] != batch:
                raise ValueError(
                    f"{where}: the values must share one batch shape; got {batch} and "
                    f"{arr.shape[:-1]} for block {name!r}. Values are not broadcast "
                    f"against each other."
                )
            out[name] = arr.astype(self._dtype)
        for name in named:
            c.check_finite(where, f"the value of block {name!r}", out[name])
        return out

    def _whiten_given(self, where: str, given, residuals):
        """Whiten the given blocks' factor rows and, if given, residuals.

        Returns ``(S, rw)``: the ``(k, N)`` whitened factor (``None`` at
        ``k = 0``) and the ``(N,)`` whitened residual (``None`` without
        residuals). Each block's row and residual are whitened together, in
        one ``whiten_mat`` call; a block with no factor row whitens only its
        residual. The whitening check runs on each block's output.
        """
        k, dtype = self.latent_dim, self._dtype
        S_cols, r_cols = [], []
        for name in given:
            i = self.names.index(name)
            F, D, d = self._factors[i], self._block_covs[i], self.dims[name]
            r = None if residuals is None else residuals[name]
            if F is not None:
                X = F.to_dense()
                if r is not None:
                    X = jnp.concatenate([X, r[:, None]], axis=-1)
                WX = D.whiten_mat(X)
                c.whitening_check(where, name, WX)
                S_cols.append(WX[:, :k])
                if r is not None:
                    r_cols.append(WX[:, k])
            else:
                if r is not None:
                    wr = D.whiten(r)
                    c.whitening_check(where, name, wr)
                    r_cols.append(wr)
                if k > 0:
                    S_cols.append(jnp.zeros((d, k), dtype))
        S = jnp.concatenate(S_cols, axis=0).T if k > 0 else None
        rw = jnp.concatenate(r_cols) if residuals is not None else None
        return S, rw

    def _condition_noisy(self, where, given, targets, y) -> Gaussian:
        if self.latent_dim == 0:
            return self._select(targets)
        residuals = {n: y[n] - self._means[self.names.index(n)] for n in given}
        S, rw = self._whiten_given(where, given, residuals)
        gram = IdentityPlusGram(S)
        w = gram.solve_factor(rw)
        T = gram.inverse_sqrt()
        means, factors, covs = [], [], []
        for name in targets:
            i = self.names.index(name)
            m, F = self._means[i], self._factors[i]
            if F is not None:
                m = m + F.matvec(w)
                F = product(F, T)
            means.append(m)
            factors.append(F)
            covs.append(self._block_covs[i])
        c.result_check(where, "decomposition's U", gram.U)
        c.result_check(where, "decomposition's singular values", gram.sigma)
        c.result_check(where, "gain coefficients w", w)
        for name, m in zip(targets, means, strict=True):
            c.result_check(where, f"conditional mean of block {name!r}", m)
        return self._with(targets, means, factors, covs)

    def _exact_qr(self, where: str, given):
        """The thin QR of the stacked given rows' transpose, rank-checked."""
        Fc = jnp.concatenate(
            [self._factors[self.names.index(n)].to_dense() for n in given]
        )
        Q, R = jnp.linalg.qr(Fc.T)
        N, k = Fc.shape
        eps = float(jnp.finfo(self._dtype).eps)
        diag = jnp.abs(jnp.diagonal(R))
        value_check(
            diag,
            lambda d: bool(jnp.min(d) > max(N, k) * eps * jnp.max(d)),
            f"{where}: the given factor rows are rank deficient to working "
            f"precision, so the exact conditional is not defined",
        )
        return Q, R

    def _condition_exact(self, where, given, targets, y) -> Gaussian:
        Q, R = self._exact_qr(where, given)
        r = jnp.concatenate([y[n] - self._means[self.names.index(n)] for n in given])
        w = dense_matvec(Q, tri_solve(R, r, lower=False, trans=1))
        means, factors, covs, dense_rows = [], [], [], []
        for name in targets:
            i = self.names.index(name)
            m, F = self._means[i], self._factors[i]
            if F is not None:
                m = m + F.matvec(w)
                Fd = F.to_dense()
                Fd = Fd - (Fd @ Q) @ Q.T
                dense_rows.append((name, Fd))
                F = Dense(Fd)
            means.append(m)
            factors.append(F)
            covs.append(self._block_covs[i])
        c.result_check(where, "diagonal of R", jnp.diagonal(R))
        c.result_check(where, "gain coefficients w", w)
        for name, m in zip(targets, means, strict=True):
            c.result_check(where, f"conditional mean of block {name!r}", m)
        for name, Fd in dense_rows:
            c.result_check(where, f"conditional factor row of block {name!r}", Fd)
        return self._with(targets, means, factors, covs)

    def _noisy_density_terms(self, where, named, diff):
        D = block_diag(*[self._term(n) for n in named])
        rows = [self._factors[self.names.index(n)] for n in named]
        if self.latent_dim > 0 and any(F is not None for F in rows):
            k = self.latent_dim
            Fc = jnp.concatenate(
                [
                    F.to_dense()
                    if F is not None
                    else jnp.zeros((self.dims[n], k), self._dtype)
                    for n, F in zip(named, rows, strict=True)
                ]
            )
            C = LowRankUpdate(D, Dense(Fc))
            offset = 0
            for name in named:
                d = self.dims[name]
                c.whitening_check(where, name, C.gram.S[:, offset : offset + d])
                offset += d
        else:
            C = D
        z = C.whiten(diff)
        label = named[0] if len(named) == 1 else ", ".join(named)
        c.whitening_check(where, label, z)
        return jnp.sum(z * z, axis=-1), C.logdet()

    def _absorb(self, ordered) -> Gaussian:
        """Absorb the terms of ``ordered`` (block order, checked) into the factor."""
        dtype, k = self._dtype, self.latent_dim
        Ls = {n: self._term(n).factor() for n in ordered}
        widths = {n: L.shape[1] for n, L in Ls.items()}
        total = sum(widths.values())
        factors, covs = [], []
        for name, F, D in zip(self.names, self._factors, self._block_covs, strict=True):
            d = self.dims[name]
            if name in Ls:
                pieces = [F] if F is not None else []
                zeros = k if F is None else 0
                for other in ordered:
                    if other == name:
                        if zeros:
                            pieces.append(Zero(d, zeros, dtype=dtype))
                        pieces.append(Ls[name])
                        zeros = 0
                    else:
                        zeros += widths[other]
                if zeros:
                    pieces.append(Zero(d, zeros, dtype=dtype))
                factors.append(hstack(*pieces))
                covs.append(None)
            else:
                factors.append(
                    None if F is None else hstack(F, Zero(d, total, dtype=dtype))
                )
                covs.append(D)
        return self._with(
            self.names, self._means, factors, covs, latent_dim=k + total, plain=True
        )


@distribution_class
class EnsembleGaussian(Gaussian):
    r"""A Gaussian whose latent coordinates are an ensemble's particles.

    The Gaussian of :class:`Gaussian`, with :math:`k = J` and

    .. math::

        F_b = \frac{1}{\sqrt{\delta}}\big[a_1^{(b)}, \dots, a_J^{(b)}\big],
        \qquad F_b \mathbf 1 = 0, \qquad
        \delta = \begin{cases} J - 1 & \text{unbiased},\\ J & \text{otherwise,}
        \end{cases}

    so latent coordinate :math:`j` belongs to particle :math:`j`, and
    :math:`\delta` is the divisor its covariance was formed with (see
    :meth:`Ensemble.project`). The Gaussian stores :math:`\delta`, and every
    operation that reads particles out of its factor rows,
    :math:`a_j^{(b)} = \sqrt{\delta}\,F_b e_j`, uses it.
    :meth:`Ensemble.project` returns one for an unweighted ensemble. Two
    operations need that correspondence and exist only here: reading the
    particles back out (:meth:`realize_particles`), and moving them as a set
    (:meth:`square_root_map`). A Gaussian built any other way is a plain
    :class:`Gaussian` and has neither.

    Every :class:`Gaussian` method that keeps the latent space returns an
    :class:`EnsembleGaussian` with the same ``n_particles`` and ``unbiased``:
    :meth:`~Gaussian.marginal`, :meth:`~Gaussian.drop`,
    :meth:`~Gaussian.rename`, :meth:`~Gaussian.condition` (both cases) and
    :meth:`~Gaussian.add_noise` on blocks with no independent term.
    :meth:`~Gaussian.absorb`, :meth:`~Gaussian.compress`, and
    :meth:`~Gaussian.add_noise` on a block that already has a term return a
    plain :class:`Gaussian`.

    Parameters
    ----------
    means, factors, block_covs
        As for :class:`Gaussian`; ``factors`` and ``block_covs`` are
        keyword-only.
    n_particles : int
        Keyword-only. :math:`J`, a Python ``int >= 2``; the latent width,
        which the factor rows' width must equal.
    unbiased : bool
        Keyword-only. Whether the factor rows were formed with
        :math:`\delta = J - 1` (``True``, the default) or :math:`\delta = J`.

    Raises
    ------
    ValueError
        As for :class:`Gaussian`, and if a factor row's width is not
        ``n_particles``. In debug mode, also if a factor row is not centered:
        :math:`\lVert F_b\mathbf 1\rVert_\infty
        \le 10\,J\,\varepsilon\max_{ij}|(F_b)_{ij}|` fails, computed densely.
    TypeError
        As for :class:`Gaussian`, and if ``n_particles`` is not an ``int`` or
        ``unbiased`` is not a ``bool``.

    Notes
    -----
    Particle alignment is a type rather than a flag because the square-root
    reading is valid only for a centered factor. Applied to an uncentered
    one it returns, silently, a particle set whose mean is displaced and
    whose sample covariance falls short of the conditional's by a rank-one
    term. :meth:`Ensemble.project` centers by construction and is the normal
    way to make one.

    The divisor is stored for the same reason: reading particles out with
    :math:`\sqrt{J-1}` from factor rows formed with :math:`J` returns them
    scaled by :math:`\sqrt{(J-1)/J}` about the mean, without raising.
    """

    n_particles: int = static_field()
    unbiased: bool = static_field()

    def __init__(
        self, means, *, factors=None, block_covs=None, n_particles, unbiased=True
    ) -> None:
        where = "EnsembleGaussian"
        c.check_n_particles(where, n_particles)
        c.check_unbiased(where, unbiased)
        _init(self, where, means, factors, block_covs, n_particles, "n_particles")
        object.__setattr__(self, "n_particles", n_particles)
        object.__setattr__(self, "unbiased", unbiased)
        dtype = self._dtype
        eps = float(jnp.finfo(dtype).eps) if jnp.issubdtype(dtype, jnp.floating) else 0.0
        for name, m, F in zip(self.names, self._means, self._factors, strict=True):
            if F is None:
                continue
            # value_check reads its first argument only to decide whether to
            # run; the row is densified only when it does.
            value_check(
                m,
                lambda _, F=F: bool(_centering_ok(F.to_dense(), n_particles, eps)),
                f"{where}: the factor row of block {name!r} is not centered "
                f"(F 1 != 0); the latent coordinates are not particles. Build it "
                f"with Ensemble.project().",
            )

    @property
    def divisor(self) -> int:
        r""":math:`\delta`: ``n_particles - 1`` if :attr:`unbiased`, else
        ``n_particles``."""
        return c.particle_divisor(self.n_particles, self.unbiased)

    def realize_particles(self, *, key=None, exclude_block_covs=()) -> Ensemble:
        r"""The particles this Gaussian's latent coordinates belong to.

        .. math::

            x_j^{(b)} = m_b + \sqrt{\delta}\,F_b e_j \;\big[\, + L_b \eta_j^{(b)} \big],
            \qquad j = 1, \dots, J,

        with :math:`e_j` the :math:`j`-th unit vector and :math:`\delta` the
        stored :attr:`divisor`, computed as
        :math:`m_b` plus the rows of :math:`\sqrt{\delta}\,F_b^\top`
        (``F_b.to_dense()``, transposed). A block with no factor row realizes
        as its mean. The bracketed draw from :math:`D_b = L_b L_b^\top` is
        added for each block that has an independent term and is not named
        in ``exclude_block_covs``.

        For the result of :meth:`Ensemble.project` this returns the
        ensemble's particles to round-off. After :meth:`~Gaussian.condition`
        it returns the conditioned particles: the deterministic square-root
        reading of the conditional.

        The draw is pinned: ``keys = jax.random.split(key, n_R)``, with
        :math:`n_R` the number of realized blocks with a term that is not
        excluded, and ``normal(keys[i], (J, w_b))`` for the :math:`i`-th in
        block order.

        Parameters
        ----------
        key
            Keyword-only typed key. Required exactly when some block has a
            term that is not excluded; otherwise ignored.
        exclude_block_covs : sequence of str
            Keyword-only. Blocks returned without their independent terms:
            the noise-free part, for instance of given blocks whose noise a
            :class:`MatheronMap` draws itself.

        Returns
        -------
        Ensemble
            Unweighted, over every block, in block order.

        Raises
        ------
        KeyError
            If an excluded name is not a block.
        ValueError
            If an excluded name is repeated or has no independent term, or a
            needed key is missing (naming the blocks that need it).
        TypeError
            If ``key`` is not a typed key.
        UnsupportedOpError
            If a realized term cannot ``factor``.
        """
        where = f"{self!r}.realize_particles"
        c.guard(self, "realize_particles")
        if isinstance(exclude_block_covs, str):
            exclude_block_covs = (exclude_block_covs,)
        excluded = c.check_name_list(where, self.names, exclude_block_covs)
        sampled = tuple(n for n in self._term_names() if n not in excluded)
        c.check_key(
            where, key, required=bool(sampled),
            why=f": blocks {sampled} have independent terms to sample",
        )
        for name in excluded:
            if self._term(name) is None:
                raise ValueError(
                    f"{where}: excluded block {name!r} has no independent term"
                )
        for name in sampled:
            c.require(self._term(name), "factor")
        keys = jax.random.split(key, len(sampled)) if sampled else None
        J, dtype = self.n_particles, self._dtype
        scale = math.sqrt(self.divisor)
        blocks, i = [], 0
        for name, m, F, D in zip(
            self.names, self._means, self._factors, self._block_covs, strict=True
        ):
            if F is not None:
                x = m + scale * F.to_dense().T
            else:
                x = jnp.broadcast_to(m, (J, m.shape[-1]))
            if name in sampled:
                L = D.factor()
                x = x + L.matvec(jax.random.normal(keys[i], (J, L.shape[1]), dtype))
                i += 1
            blocks.append(x)
        return c.build(
            Ensemble, names=self.names, _blocks=tuple(blocks), _log_weights=None
        )

    def square_root_map(self, given):
        r"""The square-root reading of the conditional, as a map from given values.

        Built from the given block *names*; called with their values,
        ``smap(y=y_star)`` returns
        ``self.condition(y=y_star).realize_particles()`` to round-off, with
        everything that does not depend on the value computed here, once:
        :math:`S = (W F_c)^\top`, its
        :class:`~enskit.linalg.IdentityPlusGram`, and each target's
        conditional anomalies :math:`\sqrt{\delta}\,(F_xT)^\top`, with
        :math:`\delta` the stored :attr:`divisor`.

        Parameters
        ----------
        given : str or sequence of str
            The names of the given blocks. Every one must have an independent
            term; for exact values, ``condition(...).realize_particles()`` is
            the route.

        Returns
        -------
        SquareRootMap
            Its targets are every other block, in block order.

        Raises
        ------
        KeyError
            If a name is not a block.
        ValueError
            If a given block has no independent term, a name is repeated, or
            no target remains.
        UnsupportedOpError
            If a given block's term cannot ``whiten``, or a target's term
            cannot ``factor``, checked block by block in block order.
        """
        from ._maps import SquareRootMap

        where = f"{self!r}.square_root_map"
        c.guard(self, "square_root_map")
        given = c.given_names(where, self.names, given)
        targets = self._noisy_targets(where, given, "square_root_map")
        for name in self.names:
            D = self._term(name)
            if name in given:
                c.require(D, "whiten")
            elif D is not None:
                c.require(D, "factor")
        S, _ = self._whiten_given(where, given, None)
        gram = IdentityPlusGram(S)
        T = gram.inverse_sqrt()
        scale = math.sqrt(self.divisor)
        rows, anomalies = [], []
        for name in targets:
            F = self._factors[self.names.index(name)]
            rows.append(F)
            anomalies.append(None if F is None else scale * T.matmat(F.to_dense().T))
        return c.build(
            SquareRootMap,
            given=given,
            targets=targets,
            n_particles=self.n_particles,
            gram=gram,
            _given_covs=tuple(self._term(n) for n in given),
            _given_means=tuple(self._means[self.names.index(n)] for n in given),
            _target_means=tuple(self._means[self.names.index(n)] for n in targets),
            _target_rows=tuple(rows),
            _target_anomalies=tuple(anomalies),
            _target_covs=tuple(self._term(n) for n in targets),
        )

    def __repr__(self) -> str:
        """Type name and static sizes, never array contents; never raises."""
        return c.safe_repr(
            lambda: (
                f"EnsembleGaussian(n_particles={self.n_particles}, blocks={self.dims}, "
                f"block_covs={self._term_names()}"
                + ("" if self.unbiased else ", unbiased=False")
                + ")"
            ),
            "EnsembleGaussian",
            lambda: self.batch_shape,
        )


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _init(obj, where, means, factors, block_covs, latent_dim, latent_name) -> None:
    """Validate and store a Gaussian's fields; shared by both constructors."""
    if not isinstance(means, Mapping):
        raise TypeError(
            f"{where}: means must be a mapping from block name to mean, got "
            f"{type(means).__name__}"
        )
    if not means:
        raise ValueError(f"{where}: at least one block is required")
    names, stored, dtype = [], [], None
    for name, m in means.items():
        if not isinstance(name, str):
            raise TypeError(
                f"{where}: block names must be str, got {type(name).__name__}"
            )
        m = jnp.asarray(m)
        if not jnp.issubdtype(m.dtype, jnp.floating):
            raise TypeError(
                f"{where}: the mean of block {name!r} must be a real floating array, "
                f"got dtype {m.dtype}"
            )
        if m.ndim != 1:
            raise ValueError(
                f"{where}: the mean of block {name!r} must be a (d,) array of rank "
                f"exactly 1, got shape {m.shape}. Distributions are unbatched; a "
                f"family is a jax.vmap."
            )
        if m.shape[0] < 1:
            raise ValueError(f"{where}: block {name!r} has dimension 0")
        if dtype is None:
            dtype = m.dtype
        elif m.dtype != dtype:
            raise TypeError(
                f"{where}: the mean of block {name!r} has dtype {m.dtype}, but the "
                f"means have dtype {dtype}; every block shares one dtype"
            )
        names.append(name)
        stored.append(m)
    dims = {n: int(m.shape[0]) for n, m in zip(names, stored, strict=True)}
    rows = _check_mapping(where, "factors", factors, names)
    terms = _check_mapping(where, "block_covs", block_covs, names)
    width = None
    for name, row in list(rows.items()):
        if not isinstance(row, LinOp):
            try:
                arr = jnp.asarray(row)
            except TypeError:
                raise TypeError(
                    f"{where}: the factor row of block {name!r} must be a LinOp or "
                    f"an array, got {type(row).__name__}"
                ) from None
            if arr.dtype != dtype:
                raise TypeError(
                    f"{where}: the factor row of block {name!r} has dtype "
                    f"{arr.dtype}, but the means have dtype {dtype}; every array "
                    f"shares one dtype"
                )
            if arr.ndim != 2:
                raise ValueError(
                    f"{where}: the factor row of block {name!r} must be a (d, k) "
                    f"array, got shape {arr.shape}"
                )
            row = Dense(arr)
            rows[name] = row
        if row.batch_shape != ():
            raise ValueError(
                f"{where}: the factor row of block {name!r}, {row!r}, is a vmapped "
                f"family; construct a family of distributions with jax.vmap over "
                f"the constructor"
            )
        if row.shape[0] != dims[name]:
            raise ValueError(
                f"{where}: the factor row of block {name!r} has {row.shape[0]} rows, "
                f"but the block has dimension {dims[name]}"
            )
        if width is None:
            width = row.shape[1]
        elif row.shape[1] != width:
            raise ValueError(
                f"{where}: the factor rows disagree on the latent width: block "
                f"{name!r} has {row.shape[1]} columns, an earlier row {width}"
            )
    for name, D in terms.items():
        _check_term(where, name, D)
        _check_term_shape(where, name, D, dims[name])
    if latent_dim is not None and latent_name == "latent_dim":
        if type(latent_dim) is not int:
            raise TypeError(
                f"{where}: latent_dim must be a Python int, got "
                f"{type(latent_dim).__name__}"
            )
        if latent_dim < 0:
            raise ValueError(f"{where}: latent_dim must be at least 0, got {latent_dim}")
    if latent_dim is None:
        latent_dim = 0 if width is None else width
    elif width is not None and width != latent_dim:
        raise ValueError(
            f"{where}: {latent_name}={latent_dim} disagrees with the factor rows' "
            f"width {width}"
        )
    for name in names:
        if name not in rows and name not in terms:
            raise ValueError(
                f"{where}: block {name!r} has neither a factor row nor an independent "
                f"term; a known constant is not a block of a distribution"
            )
    for name, m in zip(names, stored, strict=True):
        c.check_finite(where, f"the mean of block {name!r}", m)
    object.__setattr__(obj, "names", tuple(names))
    object.__setattr__(obj, "latent_dim", latent_dim)
    object.__setattr__(obj, "_means", tuple(stored))
    object.__setattr__(obj, "_factors", tuple(rows.get(n) for n in names))
    object.__setattr__(obj, "_block_covs", tuple(terms.get(n) for n in names))


def _check_mapping(where: str, field: str, mapping, names) -> dict:
    if mapping is None:
        return {}
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"{where}: {field} must be a mapping from block name to operator, got "
            f"{type(mapping).__name__}"
        )
    for name in mapping:
        if name not in names:
            raise ValueError(
                f"{where}: {field} names {name!r}, which is not a block; the blocks "
                f"are {tuple(names)}"
            )
    return dict(mapping)


def _check_term(where: str, name: str, D) -> None:
    if not isinstance(D, PSDLinOp):
        raise TypeError(
            f"{where}: the independent term of block {name!r} must be a PSDLinOp, "
            f"got {type(D).__name__}"
        )
    if D.batch_shape != ():
        raise ValueError(
            f"{where}: the independent term of block {name!r}, {D!r}, is a vmapped "
            f"family; construct a family of distributions with jax.vmap over the "
            f"constructor"
        )


def _check_term_shape(where: str, name: str, D, d: int) -> None:
    if D.shape != (d, d):
        raise ValueError(
            f"{where}: the independent term of block {name!r}, {D!r}, must have "
            f"shape ({d}, {d})"
        )


def _centering_ok(F: Array, J: int, eps: float) -> Array:
    return jnp.max(jnp.abs(jnp.sum(F, axis=-1))) <= 10 * J * eps * jnp.max(jnp.abs(F))
