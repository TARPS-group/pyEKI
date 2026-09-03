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

    from pyeki import toy
    from pyeki.eki import AdaptiveESSSchedule, EKIState, run

    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    result = run(state, problem.forward, problem.y, problem.noise_cov,
                 schedule=AdaptiveESSSchedule())

    result.mean, problem.truth

Conventions shared by all three:

- **A problem is a frozen dataclass of plain public values**, and it is
  deliberately *not* callable. ``forward``, ``y`` and ``noise_cov`` are passed
  to a run as three separate arguments, which is the real signature; a
  container accepted in their place is excluded from the EKI layer by design.
- **Every field is a value a caller could have written themselves**, so a
  problem can be modified by constructing the class directly rather than
  through its factory — with a different noise covariance, for instance.
- **A problem has no ensemble size.** :math:`J` is chosen where the initial
  ensemble is drawn, and a forward model is independent of it.
- **The synthetic observation is fixed at construction** from the factory's
  ``seed``, so a docs build or a test sees the same numbers every time.
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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from .gauss import Gaussian, GaussianJoint
from .linalg import Dense, LinOp, PSDDiagonal, PSDLinOp

__all__ = [
    "ExponentialDecay",
    "LinearGaussian",
    "RestrictedDecay",
    "exponential_decay",
    "linear_gaussian",
    "restricted_decay",
]

#: The largest posterior factor :meth:`LinearGaussian.posterior` will build,
#: in elements. The factor is ``(P, k)`` for a prior factor of width ``k``,
#: so at a full-rank prior this is a cap on ``P**2``.
_MAX_POSTERIOR_ELEMENTS = 20_000_000


# ---------------------------------------------------------------------------
# shared validation and per-member models
# ---------------------------------------------------------------------------


def _check_problem(
    cls_name: str,
    *,
    u_dim: int,
    v_dim: int,
    prior: Gaussian,
    noise_cov: PSDLinOp,
    y: Array,
    truth: Array,
) -> None:
    """Validate the four fields every problem carries against its sizes."""
    if not isinstance(prior, Gaussian):
        raise TypeError(
            f"{cls_name}.prior: must be a pyeki.gauss.Gaussian, got "
            f"{type(prior).__name__}"
        )
    if not isinstance(noise_cov, PSDLinOp):
        raise TypeError(
            f"{cls_name}.noise_cov: must be a pyeki.linalg.PSDLinOp, got "
            f"{type(noise_cov).__name__}"
        )
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
    if jnp.shape(y) != (v_dim,):
        raise ValueError(
            f"{cls_name}.y: must be ({v_dim},), the model's prediction shape, "
            f"got {jnp.shape(y)}"
        )
    if jnp.shape(truth) != (u_dim,):
        raise ValueError(
            f"{cls_name}.truth: must be ({u_dim},), the model's parameter "
            f"shape, got {jnp.shape(truth)}"
        )


def _decay(member: Array, times: Array) -> Array:
    """One member of the decay model: ``(2,) -> (N,)``.

    .. math::

        v_i = u_0 \\, e^{-u_1 t_i} .
    """
    return member[0] * jnp.exp(-member[1] * times)


def _restricted_decay(member: Array, times: Array, rate_floor: Array) -> Array:
    """One member of the decay model, outside its domain returning ``nan``."""
    return jnp.where(member[1] > rate_floor, _decay(member, times), jnp.nan)


# ---------------------------------------------------------------------------
# the linear-Gaussian problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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
    truth
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

    G: LinOp
    prior: Gaussian
    noise_cov: PSDLinOp
    y: Array
    truth: Array

    def __post_init__(self) -> None:
        if not isinstance(self.G, LinOp):
            raise TypeError(
                f"LinearGaussian.G: must be a pyeki.linalg.LinOp, got "
                f"{type(self.G).__name__}. Wrap a dense matrix as "
                f"pyeki.linalg.Dense(G); the closed-form posterior needs an "
                f"operator."
            )
        _check_problem(
            "LinearGaussian",
            u_dim=self.G.shape[1],
            v_dim=self.G.shape[0],
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            truth=self.truth,
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

        Notes
        -----
        ``G.matvec`` contracts the *trailing* axis, so the leading axis is
        carried through as a batch and the members stay in their rows. Written
        with a dense array instead, the same contraction is ``ensemble @ G.T``;
        ``ensemble @ G`` raises for a rectangular ``G`` and silently returns
        the transposed model's predictions for a square one.

        This model is pure JAX, and so ``jit``-able and ``vmap``-pable. That
        is a property of the toy models rather than a requirement on yours —
        see :mod:`pyeki.toy`.
        """
        return self.G.matvec(ensemble)

    def posterior(self, level: float = 1.0) -> Gaussian:
        """The exact posterior at tempering level :math:`\\beta`: a closed form.

        The posterior of the linear-Gaussian problem, conditioning on
        :math:`y` with the noise covariance :math:`R/\\beta` — so ``level=1.0``
        is the Bayesian posterior and a smaller level is the intermediate
        target a run passes through on the way to it.

        Two lines, both in :mod:`pyeki.gauss`::

            joint = GaussianJoint.from_linear_map(self.prior, self.G)
            return joint.condition(self.y, self.noise_cov / level)

        Copy them to reach the rest of that object: its ``v_marginal`` is the
        prior predictive distribution, and its ``pathwise`` map transports
        realizations.

        Parameters
        ----------
        level
            The tempering level :math:`\\beta`, a positive finite scalar.
            Defaults to 1.0.

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
            If ``level`` is not positive and finite, or if the posterior
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

        The cost is set by the width of the prior's factor, not by ``P``: at a
        full-rank prior in 2000 dimensions the factor is ``(2000, 2000)``, or
        32 MB. The size guard is on that product, and it raises rather than
        allocating; the run itself has no such limit, so a high-dimensional
        problem can be inverted where its closed form cannot be written down.
        """
        level = float(level)
        if not (level > 0.0) or level == float("inf"):
            raise ValueError(
                f"LinearGaussian.posterior: level must be positive and finite, "
                f"got {level}. The target at level 0 is the prior itself, which "
                f"is the `prior` field."
            )
        latent_dim = self.prior.cov.factor().shape[1]
        elements = self.u_dim * latent_dim
        if elements > _MAX_POSTERIOR_ELEMENTS:
            raise ValueError(
                f"LinearGaussian.posterior: the posterior factor would be "
                f"({self.u_dim}, {latent_dim}), {elements} elements, above this "
                f"module's budget of {_MAX_POSTERIOR_ELEMENTS}. The closed form "
                f"is not available at this size; the run is."
            )
        joint = GaussianJoint.from_linear_map(self.prior, self.G)
        return joint.condition(self.y, self.noise_cov / level)


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
    ``truth`` where the data can see it.

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
    if u_dim < 1 or v_dim < 1:
        raise ValueError(
            f"linear_gaussian: u_dim and v_dim must be at least 1, got "
            f"u_dim={u_dim}, v_dim={v_dim}"
        )
    if not (prior_std > 0.0 and noise_std > 0.0):
        raise ValueError(
            f"linear_gaussian: prior_std and noise_std must be positive, got "
            f"prior_std={prior_std}, noise_std={noise_std}"
        )
    key_map, key_truth, key_noise = jax.random.split(jax.random.key(seed), 3)
    G = Dense(jax.random.normal(key_map, (v_dim, u_dim)) / jnp.sqrt(u_dim))
    prior = Gaussian(
        jnp.zeros(u_dim), PSDDiagonal(jnp.full(u_dim, float(prior_std) ** 2))
    )
    truth = prior.sample(key_truth, 1)[0]
    error = noise_std * jax.random.normal(key_noise, (v_dim,))
    return LinearGaussian(
        G=G,
        prior=prior,
        noise_cov=PSDDiagonal(jnp.full(v_dim, float(noise_std) ** 2)),
        y=G.matvec(truth) + error,
        truth=truth,
    )


# ---------------------------------------------------------------------------
# the decay problem, and its restricted variant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExponentialDecay:
    """Two parameters, mildly nonlinear: an amplitude and a decay rate.

    .. math::

        v_i(u) = u_0 \\, e^{-u_1 t_i}, \\qquad i = 1, \\dots, N ,

    for a fixed set of points :math:`t_i`. Nonlinear enough that a single unit
    step and a tempering ladder reach visibly different answers, tame enough
    that both converge; two parameters, so results print on one line and plot
    in a plane. There is no closed-form posterior — use
    :class:`LinearGaussian` where an exact answer is the point.

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
    truth
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

    times: Array
    prior: Gaussian
    noise_cov: PSDLinOp
    y: Array
    truth: Array

    def __post_init__(self) -> None:
        _check_times("ExponentialDecay", self.times)
        _check_problem(
            "ExponentialDecay",
            u_dim=2,
            v_dim=int(jnp.shape(self.times)[0]),
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            truth=self.truth,
        )

    @property
    def u_dim(self) -> int:
        """The number of parameters, 2."""
        return 2

    @property
    def v_dim(self) -> int:
        """The number of predictions :math:`N`."""
        return int(jnp.shape(self.times)[0])

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

        Notes
        -----
        This is :func:`jax.vmap` of a function of one member, which is the
        wrapper a model written for a single parameter vector needs — and
        which cannot couple the rows even in principle.

        The mapped function is pure JAX, so this one is ``jit``-able and
        ``vmap``-pable. A model that is not, wrapped in an ordinary Python
        loop instead, is equally legal — see :mod:`pyeki.toy`.
        """
        return jax.vmap(_decay, in_axes=(0, None))(ensemble, self.times)


@dataclass(frozen=True)
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
    times, prior, noise_cov, y, truth
        As :class:`ExponentialDecay`.
    rate_floor
        The domain boundary: the model is defined where the rate, the second
        parameter, is strictly greater than this. A Python ``float``.

    Raises
    ------
    ValueError
        As :class:`ExponentialDecay`, and if ``rate_floor`` is not finite, or
        if ``truth`` is outside the valid domain — a problem whose own true
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
    prior. The realized fraction varies with the ensemble draw and, once a
    run is under way, falls as the ensemble concentrates.
    """

    times: Array
    prior: Gaussian
    noise_cov: PSDLinOp
    y: Array
    truth: Array
    rate_floor: float

    def __post_init__(self) -> None:
        _check_times("RestrictedDecay", self.times)
        _check_problem(
            "RestrictedDecay",
            u_dim=2,
            v_dim=int(jnp.shape(self.times)[0]),
            prior=self.prior,
            noise_cov=self.noise_cov,
            y=self.y,
            truth=self.truth,
        )
        floor = float(self.rate_floor)
        if floor != floor or floor in (float("inf"), float("-inf")):
            raise ValueError(
                f"RestrictedDecay.rate_floor: must be finite, got {floor}"
            )
        if not float(self.truth[1]) > floor:
            raise ValueError(
                f"RestrictedDecay: truth has rate {float(self.truth[1])}, which "
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
        return int(jnp.shape(self.times)[0])

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

        Notes
        -----
        The rows are built by :func:`jax.vmap` of a function of one member, as
        in :class:`ExponentialDecay`.

        Returning these same numbers as a NumPy array, or as a nested Python
        list, would give a bit-identical run: the return may be any array-like
        of the right shape. That is what lets a wrapper around an external
        code assemble its rows in Python without converting — and it is how a
        real failing model, which is not pure JAX, meets the same obligation
        this one meets with :func:`jax.numpy.where`.
        """
        return jax.vmap(_restricted_decay, in_axes=(0, None, None))(
            ensemble, self.times, self.rate_floor
        )


def _check_times(cls_name: str, times) -> None:
    if jnp.ndim(times) != 1 or jnp.shape(times)[0] < 1:
        raise ValueError(
            f"{cls_name}.times: must be a rank-1 array of at least one point, "
            f"got shape {jnp.shape(times)}"
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
    if n_times < 1:
        raise ValueError(f"{where}: n_times must be at least 1, got {n_times}")
    if not t_max > 0.0:
        raise ValueError(f"{where}: t_max must be positive, got {t_max}")
    if not noise_std > 0.0:
        raise ValueError(f"{where}: noise_std must be positive, got {noise_std}")
    times = jnp.linspace(t_max / n_times, t_max, n_times)
    truth = jnp.array([2.0, 1.5])
    prior = Gaussian(jnp.array([1.0, 1.0]), PSDDiagonal(jnp.array([1.0, 1.0])))
    noise_cov = PSDDiagonal(jnp.full(n_times, float(noise_std) ** 2))
    error = noise_std * jax.random.normal(jax.random.key(seed), (n_times,))
    return times, prior, noise_cov, _decay(truth, times) + error, truth


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
    :math:`\\mathcal{N}\\bigl((1, 1), I\\bigr)`. The same model as the user
    guide's "Writing a forward model" page uses, at a faster true rate, a
    wider prior and twelve observations rather than three — chosen so that a
    single unit step and an adaptive ladder reach reliably different answers
    rather than coincidentally different ones. Over eight observation seeds
    the two means differ by at least 0.10 in the rate, and the ladder's error
    against ``truth`` is around five times smaller.

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
    ValueError
        If ``n_times`` is below 1, or ``t_max`` or ``noise_std`` is not
        positive.
    """
    times, prior, noise_cov, y, truth = _decay_problem(
        n_times=n_times,
        t_max=t_max,
        noise_std=noise_std,
        seed=seed,
        where="exponential_decay",
    )
    return ExponentialDecay(
        times=times, prior=prior, noise_cov=noise_cov, y=y, truth=truth
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
    default floor the prior puts about 16% of its mass outside the valid domain,
    so several members of a moderate ensemble fail on the first step and fewer
    on each step after.

    Parameters
    ----------
    n_times, t_max, noise_std, seed
        As :func:`exponential_decay`. Keyword-only.
    rate_floor
        The domain boundary; the model is defined where the rate exceeds it.
        Raise it toward the prior mean, or above it, to fail more members.
        Keyword-only.

    Returns
    -------
    RestrictedDecay

    Raises
    ------
    ValueError
        As :func:`exponential_decay`, and if ``rate_floor`` is at or above the
        true rate.
    """
    times, prior, noise_cov, y, truth = _decay_problem(
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
        truth=truth,
        rate_floor=float(rate_floor),
    )
