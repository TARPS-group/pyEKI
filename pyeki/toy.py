"""Toy problems: small forward models, with the priors and data to drive them.

Three self-contained calibration problems, for the package's own tests and for
the documentation. Each bundles a forward model with a prior, an observation
error covariance, a synthetic observation and the parameters that generated
it, so a tutorial or a test spends two lines on setup rather than twelve.

============================ ==================================================
factory                      builds
============================ ==================================================
:func:`linear_gaussian`      :class:`LinearGaussian` — a linear model, whose
                             posterior is available in closed form, at any
                             pair of dimensions
:func:`exponential_decay`    :class:`ExponentialDecay` — two parameters, mildly
                             nonlinear
:func:`restricted_decay`     :class:`RestrictedDecay` — the same decay model
                             with a valid domain, so members outside it fail
============================ ==================================================

**These are not production models, and they are not an interface.** pyEKI
ships no forward models for real use and defines no base class, protocol or
registry for one: a forward model is any callable taking a ``(J, P)`` ensemble
to ``(J, N)`` predictions. What these classes exemplify is that callable and
the problem around it, nothing more.

Using one::

    import jax
    from pyeki import toy
    from pyeki.eki import AdaptiveESSSchedule, EKIState, run

    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    result = run(state, problem.forward, problem.y, problem.noise_cov,
                 schedule=AdaptiveESSSchedule())

    result.mean, problem.u_true      # the answer, and what generated the data

Conventions shared by all three:

- **A problem is a frozen dataclass of plain public values, and is not
  callable.** Pass ``forward``, ``y`` and ``noise_cov`` as three separate
  arguments; nothing in :mod:`pyeki.eki` accepts a problem object in their
  place.
- **Every field is a value a caller could have written themselves**, so a
  problem can be modified by constructing the class directly rather than
  through its factory — with a different noise covariance, for instance.
- **A problem has no ensemble size.** :math:`J` is chosen where the initial
  ensemble is drawn, and a forward model is independent of it.
- **The synthetic observation is fixed at construction** from the factory's
  ``seed``, so a docs build or a test sees the same numbers every time — at
  the package's default float64. With JAX's x64 mode disabled, the draws
  consume randomness differently and every factory builds a *different*
  problem, not merely a less precise one.
- **These models happen to be pure JAX**, hence ``jit``-able, ``vmap``-pable
  and differentiable. That is a convenience of *these* models, chosen to keep
  the test suite cheap, and **not** an obligation on yours: the driver loop is
  ordinary Python precisely so that a subprocess, a scheduler submission or a
  legacy binary is a legal forward model. The user guide's "Writing a forward
  model" page states the whole obligation, and its worked example — an
  external executable wrapped in NumPy — is the realistic one.

Notes
-----
Nothing in :mod:`pyeki.linalg`, :mod:`pyeki.gauss` or :mod:`pyeki.eki` imports
this module, and nothing should: it depends on two of them, so an import in
the other direction would make toy problems load-bearing for the library.

Row independence — row :math:`j` of a model's return depending only on row
:math:`j` of its argument — is the one requirement a forward model must meet
that nothing in a run detects. Each model here is written so that the property
is visible rather than asserted: the linear model applies an operator to the
trailing axis, and the two decay models are :func:`jax.vmap` of a function of
one member. :func:`pyeki.eki.testing.check_forward_model` tests the property
from outside, on a model of your own.

The closed-form posterior of :class:`LinearGaussian` is not computed here. It
is :meth:`pyeki.gauss.GaussianJoint.from_linear_map` followed by
:meth:`~pyeki.gauss.GaussianJoint.condition`, which is where the mathematics
belongs and which is worth seeing: checking a run against a known answer does
not mean writing your own algebra.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

from .gauss import Gaussian, GaussianJoint
from .linalg import Dense, LinOp, PSDDiagonal, PSDLinOp, value_check

__all__ = [
    "ExponentialDecay",
    "LinearGaussian",
    "RestrictedDecay",
    "exponential_decay",
    "linear_gaussian",
    "restricted_decay",
]

# The largest array LinearGaussian.posterior will build, in elements. The
# conditioning forms a (P, k) factor and a (k, k) transform for a prior factor
# of width k, so the cap is on the larger of the two products, and the peak is
# a small multiple of it -- see LinearGaussian.posterior's Notes.
_MAX_POSTERIOR_ELEMENTS = 20_000_000


# ---------------------------------------------------------------------------
# shared validation and per-member models
# ---------------------------------------------------------------------------


def _check_array_field(cls_name: str, field_name: str, value, shape: tuple) -> None:
    """Require an array of exactly ``shape``, finite in debug mode.

    A field that cannot be inspected is rejected rather than waved through:
    a Python list passes a shape check and then fails inside the model, with
    an error naming a tracer rather than the field.
    """
    if getattr(value, "shape", None) is None:
        raise TypeError(
            f"{cls_name}.{field_name}: expected an array of shape {shape}, got "
            f"{type(value).__name__}, which has no shape to check. Pass a JAX "
            f"array — jnp.asarray() on a nested list."
        )
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{cls_name}.{field_name}: expected an array of shape {shape}, got "
            f"shape {tuple(value.shape)}"
        )
    value_check(
        value,
        lambda arr: bool(jnp.all(jnp.isfinite(arr))),
        f"{cls_name}.{field_name}: must be finite",
    )


def _check_not_family(cls_name: str, field_name: str, value) -> None:
    """Reject a vmapped family, naming the object being built.

    A family field is accepted by every size check here and diagnosed much
    later, by the operator, in a message about the operator rather than about
    the problem.
    """
    if value.batch_shape != ():
        raise ValueError(
            f"{cls_name}.{field_name}: {value!r} is a vmapped family with batch "
            f"shape {value.batch_shape}; build a family of problems with "
            f"jax.vmap over a function that constructs one, not from a family "
            f"field."
        )


def _check_problem(
    cls_name: str,
    *,
    u_dim: int,
    v_dim: int,
    prior: Gaussian,
    noise_cov: PSDLinOp,
    y: Array,
    u_true: Array,
) -> None:
    """Validate the four fields every problem carries against its sizes."""
    if not isinstance(prior, Gaussian):
        raise TypeError(
            f"{cls_name}.prior: must be a pyeki.gauss.Gaussian, got "
            f"{type(prior).__name__}. Build one as "
            f"Gaussian(mean, PSDDiagonal(variances))."
        )
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{cls_name}.noise_cov: must be a pyeki.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}. Wrap a dense matrix with "
            f"pyeki.linalg.DensePSD.from_matrix."
        )
    _check_not_family(cls_name, "prior", prior)
    _check_not_family(cls_name, "noise_cov", noise_cov)
    if prior.dim != u_dim:
        raise ValueError(
            f"{cls_name}: the prior has dimension {prior.dim}, but the model "
            f"takes {u_dim} parameters"
        )
    if noise_cov.shape[0] != v_dim:
        raise ValueError(
            f"{cls_name}: {noise_cov!r} has side {noise_cov.shape[0]}, but the "
            f"model returns {v_dim} predictions"
        )
    _check_array_field(cls_name, "y", y, (v_dim,))
    _check_array_field(cls_name, "u_true", u_true, (u_dim,))


def _check_dim(where: str, name: str, value) -> None:
    """Require a positive Python ``int``, excluding ``bool``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{where}: {name} must be an int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{where}: {name} must be at least 1, got {value}")


def _check_scale(where: str, name: str, value) -> None:
    """Require a positive finite scalar: ``nan`` and ``inf`` both fail.

    ``not (value > 0)`` alone catches ``nan`` and lets ``inf`` through, which
    would build a problem whose every array is non-finite.
    """
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{where}: {name} must be a real scalar, got "
            f"{type(value).__name__}"
        ) from exc
    if not (scale > 0.0) or not math.isfinite(scale):
        raise ValueError(
            f"{where}: {name} must be positive and finite, got {value}"
        )


def _check_ensemble(cls_name: str, ensemble, u_dim: int):
    """Require exactly ``(J, u_dim)``: the whole ensemble, one member per row.

    The generalized-ufunc convention would carry any leading batch rank
    through, so a single parameter vector returns a plausible ``(N,)`` and a
    stack of ensembles a plausible ``(B, J, N)``. Neither is what a run passes
    — a run binds one ensemble — and the first is the mistake the forward-model
    guide calls the most common one, so the models say so rather than
    answering. ``jax.vmap`` over the method still works: it passes each slice
    as a two-dimensional ensemble.
    """
    shape = getattr(ensemble, "shape", None)
    if shape is None:
        raise TypeError(
            f"{cls_name}.forward: expected a (J, {u_dim}) array, got "
            f"{type(ensemble).__name__}, which has no shape"
        )
    if len(shape) != 2 or shape[1] != u_dim:
        raise ValueError(
            f"{cls_name}.forward: expected a (J, {u_dim}) ensemble, got shape "
            f"{tuple(shape)}. The model is called once with the whole "
            f"ensemble, one member per row — not with a single parameter "
            f"vector, and never with a further leading axis."
        )
    return ensemble


def _decay(member: Array, times: Array) -> Array:
    """One member of the decay model: ``(2,) -> (N,)``.

    .. math::

        v_i = u_0 \\, e^{-u_1 t_i} .
    """
    return member[0] * jnp.exp(-member[1] * times)


def _restricted_decay(member: Array, times: Array, rate_floor: float) -> Array:
    """One member of the decay model, outside its domain returning ``nan``.

    The rate is clamped inside the valid branch as well as selected outside
    it. ``jnp.where`` evaluates both branches, and for a very negative rate
    the discarded one overflows to ``inf``; the derivative then multiplies
    that by a zero cotangent and returns ``nan``. Clamping first keeps the
    discarded branch finite, so the model differentiates.
    """
    valid = member[1] > rate_floor
    safe = jnp.where(valid, member[1], rate_floor + 1.0)
    prediction = member[0] * jnp.exp(-safe * times)
    return jnp.where(valid, prediction, jnp.nan)


# ---------------------------------------------------------------------------
# the linear-Gaussian problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class LinearGaussian:
    """A linear forward model :math:`v = Gu`, and the problem around it.

    The one problem here whose posterior is available exactly, which makes it
    the problem to check a run against: with a Gaussian prior and a linear
    model, :meth:`posterior` is the answer the run is approximating.

    Build one with :func:`linear_gaussian`, which draws :math:`G`, the true
    parameters and the observation from a seed. Construct the class directly
    to supply any of them yourself.

    Parameters
    ----------
    G
        The linear map, a :class:`~pyeki.linalg.LinOp` of shape ``(N, P)``.
    prior
        The prior over the parameters, a :class:`~pyeki.gauss.Gaussian` of
        dimension ``P`` whose covariance supports ``factor``.
    noise_cov
        The observation error covariance, a
        :class:`~pyeki.linalg.PSDLinOp` of side ``N``.
    y
        The observation, a ``(N,)`` array.
    u_true
        The parameters the observation was generated from, a ``(P,)`` array.

    Raises
    ------
    ValueError
        If the five fields disagree on ``P`` or ``N``.
    TypeError
        If ``G`` is not a :class:`~pyeki.linalg.LinOp`, ``prior`` not a
        :class:`~pyeki.gauss.Gaussian`, or ``noise_cov`` not a
        :class:`~pyeki.linalg.PSDLinOp`.
    """

    G: LinOp = field(kw_only=True)
    prior: Gaussian = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    y: Array = field(kw_only=True)
    u_true: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.G, LinOp):
            raise TypeError(
                f"LinearGaussian.G: must be a pyeki.linalg.LinOp, got "
                f"{type(self.G).__name__}. Wrap a dense matrix as "
                f"pyeki.linalg.Dense(G); the closed-form posterior needs an "
                f"operator."
            )
        _check_not_family("LinearGaussian", "G", self.G)
        _check_problem(
            "LinearGaussian",
            u_dim=self.G.shape[1],
            v_dim=self.G.shape[0],
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            u_true=self.u_true,
        )

    @property
    def u_dim(self) -> int:
        """The number of parameters :math:`P`."""
        return int(self.G.shape[1])

    @property
    def v_dim(self) -> int:
        """The number of predictions :math:`N`."""
        return int(self.G.shape[0])

    def forward(self, ensemble) -> Array:
        """The forward model: ``(J, P)`` members in, ``(J, N)`` predictions out.

        Parameters
        ----------
        ensemble
            The parameters to evaluate, a ``(J, P)`` array — the whole
            ensemble, one member per row, not a single parameter vector.

        Returns
        -------
        Array
            The predictions, ``(J, N)``, row :math:`j` from row :math:`j`.

        Raises
        ------
        ValueError
            If ``ensemble`` is not exactly two-dimensional with ``P`` columns.

        Notes
        -----
        ``G.matvec`` contracts the *trailing* axis, so the leading axis is
        carried through as a batch and the members stay in their rows. Written
        with a dense array instead, the same contraction is ``ensemble @ G.T``;
        ``ensemble @ G`` raises for a rectangular ``G`` and silently returns
        the transposed model's predictions for a square one.

        ``jit``-able and ``vmap``-pable, as a convenience of these models
        rather than a requirement on yours — see :mod:`pyeki.toy`.
        """
        return self.G.matvec(_check_ensemble("LinearGaussian", ensemble, self.u_dim))

    def __repr__(self) -> str:
        """As ``LinearGaussian(u_dim=4, v_dim=8)``; never raises."""
        try:
            return f"LinearGaussian(u_dim={self.u_dim}, v_dim={self.v_dim})"
        except Exception:
            return "<LinearGaussian (unprintable fields)>"

    def posterior(self, beta: float = 1.0) -> Gaussian:
        """The exact posterior at tempering level :math:`\\beta`: a closed form.

        The posterior of the linear-Gaussian problem, conditioning on
        :math:`y` with the noise covariance :math:`R/\\beta` — so ``beta=1.0``
        is the Bayesian posterior and a smaller value is the intermediate
        target a run passes through on the way to it.

        Two lines, both in :mod:`pyeki.gauss`::

            joint = GaussianJoint.from_linear_map(self.prior, self.G)
            return joint.condition(self.y, self.noise_cov / beta)

        Copy them to reach the rest of that object: its ``v_marginal`` is the
        prior predictive distribution, and its ``pathwise`` map transports
        realizations.

        Parameters
        ----------
        beta
            The tempering level :math:`\\beta`, a positive finite scalar.
            Defaults to 1.0. Named as the state and every policy name it, so
            ``problem.posterior(result.beta)`` reads directly.

        Returns
        -------
        Gaussian
            The posterior over the parameters, of dimension ``P``. Its
            covariance is a :class:`~pyeki.linalg.PSDLowRank` holding a
            ``(P, k)`` factor, for ``k`` the width of the prior covariance's
            factor — never a dense ``P``-by-``P`` matrix. Directly comparable
            with :meth:`pyeki.gauss.Gaussian.from_samples` of a run's final
            ensemble, which has the same type and the same covariance
            structure.

        Raises
        ------
        ValueError
            If ``beta`` is not positive and finite, or if the posterior
            factor would exceed this module's element budget.
        UnsupportedOpError
            If the prior covariance does not support ``factor``, or the noise
            covariance does not support ``whiten``.

        Notes
        -----
        Nothing is inverted, so a *singular* prior covariance is fine here —
        the case a precision-form posterior
        :math:`(C_0^{-1} + \\beta G^\\top R^{-1} G)^{-1}` cannot express at
        all.

        The cost is set by ``P`` and the width ``k`` of the prior's factor,
        and the observation dimension ``N`` does not enter it. At a full-rank
        prior in 2000 dimensions the returned factor is ``(2000, 2000)``, or
        32 MB; peak memory is several times that, because the conditioning
        forms a ``(k, k)`` transform and the product of the two before the
        result exists — measured at about 180 MB for that case. The guard is
        on the larger of ``P * k`` and ``k * k``, capped at 20 million
        elements, and it raises rather than allocating.

        The run itself has no such limit, so a high-dimensional problem can be
        inverted where its closed form cannot be written down. The two lines
        above bypass the guard, deliberately: it is a guard on a toy
        convenience, not a limit in :mod:`pyeki.gauss`.
        """
        beta = float(beta)
        if not (beta > 0.0) or not math.isfinite(beta):
            raise ValueError(
                f"LinearGaussian.posterior: beta must be positive and finite, "
                f"got {beta}. The target at beta 0 is the prior itself, which "
                f"is the `prior` field."
            )
        latent_dim = self.prior.cov.factor().shape[1]
        # The conditioning forms a (P, k) factor *and* a (k, k) transform, so
        # bound the larger of the two: a prior factor wider than P would sail
        # past a cap on P * k while building a k * k array.
        elements = max(self.u_dim, latent_dim) * latent_dim
        if elements > _MAX_POSTERIOR_ELEMENTS:
            raise ValueError(
                f"LinearGaussian.posterior: the largest array it would build "
                f"is {max(self.u_dim, latent_dim)}-by-{latent_dim}, "
                f"{elements} elements, above this module's budget of "
                f"{_MAX_POSTERIOR_ELEMENTS}. The closed form is not available "
                f"at this size; the run is."
            )
        joint = GaussianJoint.from_linear_map(self.prior, self.G)
        return joint.condition(self.y, self.noise_cov / beta)


def linear_gaussian(
    *,
    u_dim: int = 4,
    v_dim: int = 8,
    prior_std: float = 1.0,
    noise_std: float = 0.1,
    seed: int = 0,
) -> LinearGaussian:
    """A :class:`LinearGaussian` problem at a chosen pair of dimensions.

    The map's entries are drawn :math:`N(0, 1/P)`, so a prediction is of the
    same order as ``prior_std`` whatever ``P`` is, and the signal-to-noise
    ratio is about ``prior_std / noise_std`` at every size. The true
    parameters are a draw from the prior, and the observation is
    :math:`G u_\\star` plus a draw from the observation error — so the problem
    is well specified, and the posterior really does concentrate near
    ``u_true`` where the data can see it.

    Parameters
    ----------
    u_dim, v_dim
        The number of parameters :math:`P` and predictions :math:`N`.
        Keyword-only.
    prior_std, noise_std
        The prior standard deviation, the same in every parameter, and the
        observation error standard deviation, the same in every prediction.
        Both positive. Keyword-only.
    seed
        Seeds the map, the true parameters and the observation error.
        Keyword-only.

    Returns
    -------
    LinearGaussian

    Raises
    ------
    ValueError
        If a dimension is below 1, or a standard deviation is not positive.

    Notes
    -----
    Pass ``u_dim=2000, v_dim=40`` for a problem where the parameter dimension
    far exceeds any affordable ensemble size — the regime in which a run can
    only represent a :math:`J-1`-dimensional subspace of the answer, and in
    which comparing against :meth:`LinearGaussian.posterior` is the only way
    to see that.
    """
    _check_dim("linear_gaussian", "u_dim", u_dim)
    _check_dim("linear_gaussian", "v_dim", v_dim)
    _check_scale("linear_gaussian", "prior_std", prior_std)
    _check_scale("linear_gaussian", "noise_std", noise_std)
    key_map, key_truth, key_noise = jax.random.split(jax.random.key(seed), 3)
    G = Dense(jax.random.normal(key_map, (v_dim, u_dim)) / jnp.sqrt(u_dim))
    prior = Gaussian(
        jnp.zeros(u_dim), PSDDiagonal(jnp.full(u_dim, float(prior_std) ** 2))
    )
    u_true = prior.sample(key_truth, 1)[0]
    error = noise_std * jax.random.normal(key_noise, (v_dim,))
    return LinearGaussian(
        G=G,
        prior=prior,
        noise_cov=PSDDiagonal(jnp.full(v_dim, float(noise_std) ** 2)),
        y=G.matvec(u_true) + error,
        u_true=u_true,
    )


# ---------------------------------------------------------------------------
# the decay problem, and its restricted variant
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class ExponentialDecay:
    """Two parameters, mildly nonlinear: an amplitude and a decay rate.

    .. math::

        v_i(u) = u_0 \\, e^{-u_1 t_i}, \\qquad i = 1, \\dots, N ,

    for a fixed set of points :math:`t_i`. Two parameters, so results print on
    one line and plot in a plane, and nonlinear enough that how gradually the
    observation is assimilated changes the answer. There is no closed-form
    posterior — use :class:`LinearGaussian` where an exact answer is the
    point.

    Build one with :func:`exponential_decay`.

    Parameters
    ----------
    times
        The points :math:`t_i`, a ``(N,)`` array.
    prior
        The prior over ``(amplitude, rate)``, a
        :class:`~pyeki.gauss.Gaussian` of dimension 2.
    noise_cov
        The observation error covariance, a
        :class:`~pyeki.linalg.PSDLinOp` of side ``N``.
    y
        The observation, a ``(N,)`` array.
    u_true
        The parameters the observation was generated from, a ``(2,)`` array.

    Raises
    ------
    ValueError
        If ``times`` is not rank 1, or the fields disagree on ``N`` or on the
        two parameters.
    TypeError
        If ``prior`` is not a :class:`~pyeki.gauss.Gaussian` or ``noise_cov``
        not a :class:`~pyeki.linalg.PSDLinOp`.
    """

    times: Array = field(kw_only=True)
    prior: Gaussian = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    y: Array = field(kw_only=True)
    u_true: Array = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_times("ExponentialDecay", self.times)
        _check_problem(
            "ExponentialDecay",
            u_dim=2,
            v_dim=int(self.times.shape[0]),
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            u_true=self.u_true,
        )

    @property
    def u_dim(self) -> int:
        """The number of parameters, 2."""
        return 2

    @property
    def v_dim(self) -> int:
        """The number of predictions :math:`N`."""
        return int(self.times.shape[0])

    def forward(self, ensemble) -> Array:
        """The forward model: ``(J, P)`` members in, ``(J, N)`` predictions out.

        Parameters
        ----------
        ensemble
            The parameters to evaluate, a ``(J, 2)`` array — the whole
            ensemble, one member per row, not a single parameter vector.

        Returns
        -------
        Array
            The predictions, ``(J, N)``, row :math:`j` from row :math:`j`.

        Raises
        ------
        ValueError
            If ``ensemble`` is not exactly two-dimensional with 2 columns.

        Notes
        -----
        This is :func:`jax.vmap` of a function of one member, which is the
        wrapper a model written for a single parameter vector needs — and
        which cannot couple the rows even in principle.

        ``jit``-able and ``vmap``-pable, as a convenience of these models
        rather than a requirement on yours — see :mod:`pyeki.toy`.
        """
        _check_ensemble("ExponentialDecay", ensemble, 2)
        return jax.vmap(_decay, in_axes=(0, None))(ensemble, self.times)

    def __repr__(self) -> str:
        """As ``ExponentialDecay(v_dim=12)``; never raises."""
        try:
            return f"ExponentialDecay(v_dim={self.v_dim})"
        except Exception:
            return "<ExponentialDecay (unprintable fields)>"


@dataclass(frozen=True, eq=False, repr=False)
class RestrictedDecay:
    """:class:`ExponentialDecay` with a valid domain, so some members fail.

    The same model, defined only where the decay rate exceeds
    ``rate_floor``:

    .. math::

        v_i(u) = \\begin{cases}
            u_0 \\, e^{-u_1 t_i} & u_1 > \\texttt{rate\\_floor}, \\\\
            \\texttt{nan} & \\text{otherwise.}
        \\end{cases}

    A non-finite prediction row is how a forward model signals that a member
    failed, and a solver refusing to run or diverging on an out-of-domain
    parameter is the ordinary reason for one. Which members fail is a
    deterministic function of the ensemble, so a run is reproducible; the
    *fraction* that fail is set by how much prior mass sits below the floor,
    which makes it a property of the problem rather than an injected rate.

    Build one with :func:`restricted_decay`.

    Parameters
    ----------
    times, prior, noise_cov, y, u_true
        As :class:`ExponentialDecay`.
    rate_floor
        The domain boundary: the model is defined where the rate, the second
        parameter, is strictly greater than this. A Python ``float``.

    Raises
    ------
    ValueError
        As :class:`ExponentialDecay`, and if ``rate_floor`` is not finite, or
        if ``u_true`` is outside the valid domain — a problem whose own true
        parameters fail is a mistake rather than a test case.
    TypeError
        As :class:`ExponentialDecay`.

    Notes
    -----
    Under a Gaussian prior with rate mean :math:`m_1` and standard deviation
    :math:`\\sigma_1`, the expected fraction of failing members is

    .. math::

        \\Phi\\!\\left(
            \\frac{\\texttt{rate\\_floor} - m_1}{\\sigma_1}
        \\right) ,

    so a target failure rate is reached by moving either the floor or the
    prior. The realized fraction varies with the ensemble draw, and falls to
    zero once an ensemble has concentrated above the floor.
    """

    times: Array = field(kw_only=True)
    prior: Gaussian = field(kw_only=True)
    noise_cov: PSDLinOp = field(kw_only=True)
    y: Array = field(kw_only=True)
    u_true: Array = field(kw_only=True)
    rate_floor: float = field(kw_only=True)

    def __post_init__(self) -> None:
        _check_times("RestrictedDecay", self.times)
        _check_problem(
            "RestrictedDecay",
            u_dim=2,
            v_dim=int(self.times.shape[0]),
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            u_true=self.u_true,
        )
        if isinstance(self.rate_floor, bool) or not isinstance(
            self.rate_floor, (int, float)
        ):
            raise TypeError(
                f"RestrictedDecay.rate_floor: must be a Python float, got "
                f"{type(self.rate_floor).__name__}. A string or a bool would "
                f"pass a coercion check and then move the domain boundary, or "
                f"fail inside the model with an error naming a tracer."
            )
        floor = float(self.rate_floor)
        if not math.isfinite(floor):
            raise ValueError(
                f"RestrictedDecay.rate_floor: must be finite, got {floor}"
            )
        # Store the coerced value, so the field is the Python float the class
        # documents rather than whatever numeric type was passed.
        object.__setattr__(self, "rate_floor", floor)
        if not float(self.u_true[1]) > floor:
            raise ValueError(
                f"RestrictedDecay: u_true has rate {float(self.u_true[1])}, which "
                f"is outside the valid domain rate > {floor}, so the "
                f"observation was generated where the model does not evaluate"
            )

    @property
    def u_dim(self) -> int:
        """The number of parameters, 2."""
        return 2

    @property
    def v_dim(self) -> int:
        """The number of predictions :math:`N`."""
        return int(self.times.shape[0])

    def forward(self, ensemble) -> Array:
        """The forward model: ``(J, P)`` in, ``(J, N)`` out, some rows ``nan``.

        Parameters
        ----------
        ensemble
            The parameters to evaluate, a ``(J, 2)`` array — the whole
            ensemble, one member per row.

        Returns
        -------
        Array
            The predictions, ``(J, N)``. A member whose rate is not above
            ``rate_floor`` gets a wholly non-finite row, which a run reads as
            a failed member.

        Raises
        ------
        ValueError
            If ``ensemble`` is not exactly two-dimensional with 2 columns.

        Notes
        -----
        The rows are built by :func:`jax.vmap` of a function of one member, as
        in :class:`ExponentialDecay`.

        A row is wholly non-finite when the member is outside the domain. It
        can *also* go partly non-finite inside the domain, for a
        ``rate_floor`` far below zero: :math:`e^{-u_1 t}` overflows to
        ``inf`` below a rate of about :math:`-236` at the default ``t_max``.
        Either way the member counts as failed, since a run requires every
        entry of a row to be finite. The shipped prior puts that region 237
        standard deviations away, so the default problem never reaches it.

        Returning these same numbers as a NumPy array, or as a nested Python
        list, would give a bit-identical run: the return may be any array-like
        of the right shape. That is what lets a wrapper around an external
        code assemble its rows in Python without converting — and it is how a
        real failing model, which is not pure JAX, meets the same obligation
        this one meets with :func:`jax.numpy.where`.
        """
        _check_ensemble("RestrictedDecay", ensemble, 2)
        return jax.vmap(_restricted_decay, in_axes=(0, None, None))(
            ensemble, self.times, self.rate_floor
        )

    def __repr__(self) -> str:
        """As ``RestrictedDecay(v_dim=12, rate_floor=0.0)``; never raises."""
        try:
            return (
                f"RestrictedDecay(v_dim={self.v_dim}, "
                f"rate_floor={self.rate_floor})"
            )
        except Exception:
            return "<RestrictedDecay (unprintable fields)>"


def _check_times(cls_name: str, times) -> None:
    """Require a rank-1 array of at least one point, finite in debug mode.

    Checked through the object's own ``shape`` rather than through
    :func:`jax.numpy.shape`, which accepts anything shape-like — a Python
    list among them — and is deprecated for non-arrays.
    """
    shape = getattr(times, "shape", None)
    if shape is None:
        raise TypeError(
            f"{cls_name}.times: expected a rank-1 array, got "
            f"{type(times).__name__}, which has no shape to check. Pass a JAX "
            f"array — jnp.asarray() on a nested list."
        )
    if len(shape) != 1 or shape[0] < 1:
        raise ValueError(
            f"{cls_name}.times: must be a rank-1 array of at least one point, "
            f"got shape {tuple(shape)}"
        )
    value_check(
        times,
        lambda arr: bool(jnp.all(jnp.isfinite(arr))),
        f"{cls_name}.times: must be finite",
    )


def _decay_problem(
    *,
    n_times: int,
    t_max: float,
    noise_std: float,
    seed: int,
    where: str,
) -> tuple[Array, Gaussian, PSDLinOp, Array, Array]:
    """The pieces both decay factories share, so the two problems agree."""
    _check_dim(where, "n_times", n_times)
    _check_scale(where, "t_max", t_max)
    _check_scale(where, "noise_std", noise_std)
    times = jnp.linspace(t_max / n_times, t_max, n_times)
    u_true = jnp.array([2.0, 1.5])
    prior = Gaussian(jnp.array([1.0, 1.0]), PSDDiagonal(jnp.array([1.0, 1.0])))
    noise_cov = PSDDiagonal(jnp.full(n_times, float(noise_std) ** 2))
    error = noise_std * jax.random.normal(jax.random.key(seed), (n_times,))
    return times, prior, noise_cov, _decay(u_true, times) + error, u_true


def exponential_decay(
    *,
    n_times: int = 12,
    t_max: float = 3.0,
    noise_std: float = 0.02,
    seed: int = 0,
) -> ExponentialDecay:
    """An :class:`ExponentialDecay` problem on an evenly spaced set of points.

    The points are ``n_times`` values evenly spaced over ``(0, t_max]``, the
    true parameters are ``(2.0, 1.5)``, and the prior is
    :math:`\\mathcal{N}\\bigl((1, 1), I\\bigr)`. The same functional form as
    the user guide's "Writing a forward model" page, at a faster true rate, a
    wider prior over the rate, a five times narrower observation error and
    twelve observation points rather than three.

    Parameters
    ----------
    n_times
        The number of observation points :math:`N`. Keyword-only.
    t_max
        The last point. Keyword-only.
    noise_std
        The observation error standard deviation, the same at every point.
        Keyword-only.
    seed
        Seeds the observation error. Keyword-only.

    Returns
    -------
    ExponentialDecay

    Raises
    ------
    TypeError
        If ``n_times`` is not an ``int``.
    ValueError
        If ``n_times`` is below 1, or ``t_max`` or ``noise_std`` is not
        positive and finite.

    Notes
    -----
    The defaults are chosen so that assimilating the observation in one unit
    step and assimilating it gradually reach *reliably* different answers
    rather than coincidentally different ones. Measured over the eight
    observation seeds 0 to 7, against
    :class:`~pyeki.eki.AdaptiveESSSchedule` at 64 members: the two posterior
    means differ by between 0.10 and 0.25 in the rate, and the gradual answer
    is nearer ``u_true`` at every seed, by a factor between 2.7 and 41.
    ``tests/test_toy.py`` asserts both over all eight.
    """
    times, prior, noise_cov, y, u_true = _decay_problem(
        n_times=n_times,
        t_max=t_max,
        noise_std=noise_std,
        seed=seed,
        where="exponential_decay",
    )
    return ExponentialDecay(
        times=times, prior=prior, noise_cov=noise_cov, y=y, u_true=u_true
    )


def restricted_decay(
    *,
    n_times: int = 12,
    t_max: float = 3.0,
    noise_std: float = 0.02,
    rate_floor: float = 0.0,
    seed: int = 0,
) -> RestrictedDecay:
    """A :class:`RestrictedDecay` problem: :func:`exponential_decay`'s, with a domain.

    The same problem in every other respect, so a run against it can be
    compared with one against :func:`exponential_decay` directly. At the
    default floor the prior puts about 16% of its mass outside the valid
    domain, so a few members of any ensemble drawn from the prior fail, and
    none once the ensemble has concentrated above the floor.

    Parameters
    ----------
    n_times, t_max, noise_std, seed
        As :func:`exponential_decay`. Keyword-only.
    rate_floor
        The domain boundary; the model is defined where the rate exceeds it.
        Raise it toward the prior mean to fail more members. It must stay
        strictly below the true rate of 1.5, or the observation would have
        been generated where the model does not evaluate. Keyword-only.

    Returns
    -------
    RestrictedDecay

    Raises
    ------
    ValueError
        As :func:`exponential_decay`, and if ``rate_floor`` is at or above the
        true rate.
    """
    times, prior, noise_cov, y, u_true = _decay_problem(
        n_times=n_times,
        t_max=t_max,
        noise_std=noise_std,
        seed=seed,
        where="restricted_decay",
    )
    return RestrictedDecay(
        times=times,
        prior=prior,
        noise_cov=noise_cov,
        y=y,
        u_true=u_true,
        rate_floor=float(rate_floor),
    )
