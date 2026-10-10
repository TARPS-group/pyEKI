"""The divisor of covariances computed from particles (issue #91).

The default is unbiased: :math:`J - 1` unweighted, :math:`1 - \\sum_j w_j^2`
weighted. ``unbiased=False`` gives the empirical distribution's own moments,
divisor :math:`J` or :math:`1`. It is accepted by ``Ensemble.cov``,
``Ensemble.project`` and ``maps.statistical_linearization``; an unweighted
projection is always an ``EnsembleGaussian``, which carries its divisor, and
every reader of its factor rows honors it.

The file has two sections:

- **Exactness**: both divisors against closed forms computed here in NumPy,
  unweighted and weighted.
- **Regression**: one test per place an ``EnsembleGaussian``'s divisor is read
  or rebuilt. Reading particles out with :math:`\\sqrt{J-1}` from factor rows
  formed with :math:`J` returns them scaled by :math:`\\sqrt{(J-1)/J}` about
  their mean, without raising, so each test compares against a reference that
  does not read the divisor at all.
"""

from __future__ import annotations

import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import enskit  # noqa: F401  -- enables x64 before any array exists
from enskit import maps
from enskit.distribution import Ensemble, EnsembleGaussian, Gaussian
from enskit.kalman import (
    DomainLocalization,
    LocalizedUpdateRule,
    Matheron,
    SymmetricSquareRoot,
    gaussian_approximation,
)
from enskit.linalg import DensePSD, PSDDiagonal, debug_checks

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)
J = 7


@pytest.fixture(autouse=True)
def _reseed_rng(request):
    """Give every test its own deterministic stream, seeded from its id."""
    global RNG
    RNG = np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


def _close(got, want, factor=1e3):
    got, want = np.asarray(got), np.asarray(want)
    scale = max(1.0, np.abs(want).max())
    np.testing.assert_allclose(got, want, rtol=0, atol=factor * EPS * scale)


def _ensemble(log_weights=None, n=J):
    """Particles over ``x`` (3), ``y`` (2) and ``z`` (4), off-center and spread."""
    blocks = {
        "x": 5.0 + RNG.normal(size=(n, 3)),
        "y": RNG.normal(size=(n, 2)),
        "z": -2.0 + 3.0 * RNG.normal(size=(n, 4)),
    }
    return Ensemble(
        {k: jnp.asarray(v) for k, v in blocks.items()}, log_weights=log_weights
    )


def _log_weights(n=J):
    return jnp.asarray(RNG.normal(size=n))


def _reference_cov(ens, a, b, unbiased):
    """The covariance from the particles, in NumPy, without ``enskit``."""
    xa, xb = np.asarray(ens[a]), np.asarray(ens[b])
    n = xa.shape[0]
    if ens.is_weighted:
        w = np.exp(np.asarray(ens.log_weights) - np.asarray(ens.log_weights).max())
        w = w / w.sum()
    else:
        w = np.full(n, 1.0 / n)
    da, db = xa - w @ xa, xb - w @ xb
    raw = (w[:, None] * da).T @ db
    if not ens.is_weighted:
        return raw * n / (n - 1 if unbiased else n)
    return raw / (1.0 - np.sum(w**2)) if unbiased else raw


def _noise(d=2):
    M = RNG.normal(size=(d, d))
    return DensePSD(jnp.asarray(M @ M.T / d + np.eye(d)))


# ===========================================================================
# exactness
# ===========================================================================


@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("unbiased", (True, False))
def test_cov_against_its_closed_form(weighted, unbiased):
    ens = _ensemble(_log_weights() if weighted else None)
    for a, b in (("x", "x"), ("x", "z"), ("z", "y")):
        got = ens.cov(a, b, unbiased=unbiased).to_dense()
        _close(got, _reference_cov(ens, a, b, unbiased))


@pytest.mark.parametrize("weighted", (False, True))
def test_the_two_divisors_differ_by_their_ratio_exactly(weighted):
    ens = _ensemble(_log_weights() if weighted else None)
    if weighted:
        w = np.asarray(ens.weights)
        ratio = 1.0 - np.sum(w**2)
    else:
        ratio = (J - 1) / J
    biased = np.asarray(ens.cov("z", unbiased=False).to_dense())
    unbiased = np.asarray(ens.cov("z").to_dense())
    _close(biased, ratio * unbiased)


@pytest.mark.parametrize("unbiased", (True, False))
def test_uniform_weights_give_the_unweighted_divisor(unbiased):
    ens = _ensemble()
    uniform = Ensemble(
        {n: ens[n] for n in ens.names}, log_weights=jnp.zeros(J, jnp.float64)
    )
    _close(
        uniform.cov("x", "z", unbiased=unbiased).to_dense(),
        ens.cov("x", "z", unbiased=unbiased).to_dense(),
    )


@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("unbiased", (True, False))
def test_project_has_covs_moments_and_the_right_type(weighted, unbiased):
    ens = _ensemble(_log_weights() if weighted else None)
    g = ens.project(unbiased=unbiased)
    if weighted:
        assert type(g) is Gaussian
    else:
        assert type(g) is EnsembleGaussian
        assert g.unbiased is unbiased
        assert g.divisor == (J - 1 if unbiased else J)
    for a, b in (("x", "x"), ("y", "z")):
        _close(g.cov(a, b).to_dense(), _reference_cov(ens, a, b, unbiased))


def test_concentrated_weights_have_an_empirical_covariance():
    """Weight on one particle: the unbiased divisor is zero, the empirical one 1."""
    lw = jnp.asarray([0.0] + [-1e4] * (J - 1))
    ens = _ensemble(lw)
    with debug_checks(), pytest.raises(ValueError, match="unbiased=False"):
        ens.project()
    with debug_checks():
        g = ens.project(unbiased=False)
    _close(g.cov("x").to_dense(), np.zeros((3, 3)))


@pytest.mark.parametrize("weighted", (False, True))
def test_statistical_linearization_scales_only_omega(weighted):
    ens = _ensemble(_log_weights() if weighted else None)
    default = maps.statistical_linearization(ens, inputs="x", output="z")
    empirical = maps.statistical_linearization(
        ens, inputs="x", output="z", unbiased=False
    )
    ratio = 1.0 - float(np.sum(np.asarray(ens.weights) ** 2)) if weighted else (J - 1) / J
    _close(empirical.map.op.to_dense(), default.map.op.to_dense(), 1e4)
    _close(empirical.map.shift, default.map.shift, 1e4)
    _close(empirical.residuals, default.residuals, 1e4)
    _close(
        empirical.residual_cov.to_dense(), ratio * default.residual_cov.to_dense(), 1e4
    )


def test_statistical_linearization_min_norm_with_the_empirical_divisor():
    """An empirical projection is still aligned, so minimum norm still applies."""
    ens = _ensemble(n=4)  # x and z together: 7 input coordinates > J - 1 = 3
    default = maps.statistical_linearization(
        ens, inputs=("x", "z"), output="y", min_norm=True
    )
    empirical = maps.statistical_linearization(
        ens, inputs=("x", "z"), output="y", min_norm=True, unbiased=False
    )
    for n in ("x", "z"):
        _close(empirical.map.op[n].to_dense(), default.map.op[n].to_dense(), 1e4)


def test_argument_checks():
    ens = _ensemble()
    for call in (
        lambda: ens.cov("x", unbiased=1),
        lambda: ens.project(unbiased=None),
        lambda: maps.statistical_linearization(
            ens, inputs="x", output="y", unbiased="no"
        ),
        lambda: EnsembleGaussian(
            {"a": jnp.zeros(2)},
            factors={"a": jnp.zeros((2, 3))},
            n_particles=3,
            unbiased=0,
        ),
    ):
        with pytest.raises(TypeError, match="unbiased must be a Python bool"):
            call()
    with pytest.raises(ValueError, match="is a Gaussian"):
        maps.statistical_linearization(
            ens.project(), inputs="x", output="y", unbiased=False
        )


# ===========================================================================
# regression: no EnsembleGaussian path mixes divisors
# ===========================================================================


@pytest.mark.parametrize("unbiased", (True, False))
def test_realize_particles_returns_the_particles(unbiased):
    ens = _ensemble()
    back = ens.project(unbiased=unbiased).realize_particles()
    for n in ens.names:
        _close(back[n], ens[n])


@pytest.mark.parametrize("unbiased", (True, False))
def test_every_latent_space_keeping_method_keeps_the_divisor(unbiased):
    ens = _ensemble()
    g = ens.project(unbiased=unbiased)
    R = _noise()
    results = {
        "marginal": g.marginal("x", "y"),
        "drop": g.drop("z"),
        "rename": g.rename(z="w"),
        "add_noise": g.add_noise(y=R),
        "condition": g.add_noise(y=R).condition(y=jnp.zeros(2)),
        "condition, exact": g.condition(y=jnp.zeros(2)),
        "pushforward": maps.pushforward(
            g, maps.Linear(jnp.asarray(RNG.normal(size=(2, 3)))), inputs="x", output="v"
        ),
    }
    for what, h in results.items():
        assert type(h) is EnsembleGaussian, what
        assert h.unbiased is unbiased, what


def test_a_linear_pushforward_realizes_the_mapped_particles():
    """``maps`` rebuilds an aligned Gaussian through the public constructor.

    Dropping the divisor there gave particles shrunk by sqrt((J-1)/J) about
    their mean, in every block, with nothing raised.
    """
    ens = _ensemble()
    G = jnp.asarray(RNG.normal(size=(2, 3)))
    g = maps.pushforward(
        ens.project(unbiased=False), maps.Linear(G), inputs="x", output="v"
    )
    back = g.realize_particles()
    _close(back["x"], ens["x"])
    _close(back["v"], ens["x"] @ G.T, 1e4)


def test_the_square_root_update_is_invariant_to_a_consistent_rescaling():
    """Divisor J with noise cR is divisor J - 1 with noise R, c = (J-1)/J.

    The whitened factor is the same, so the transform and the mean
    increment are; reading back with sqrt(J) instead of sqrt(J - 1) gives
    the same particles. The default is checked against a dense reference in
    ``tests/test_kalman.py``.
    """
    ens = _ensemble()
    R = _noise()
    c = (J - 1) / J
    y = {"y": jnp.asarray(RNG.normal(size=2))}
    want = ens.project().add_noise(y=R).square_root_map("y")(y)
    approx = ens.project(unbiased=False).add_noise(y=c * R)
    got = approx.square_root_map("y")(y)
    via_condition = approx.condition(y).realize_particles()
    for n in ("x", "z"):
        _close(got[n], want[n], 1e4)
        _close(via_condition[n], want[n], 1e4)
    rule = SymmetricSquareRoot().build(ens, approx, "y")(y)
    for n in ("x", "z"):
        _close(rule[n], want[n], 1e4)


def test_particle_coefficients_read_the_particles_given_values():
    """The aligned shortcut against whitening each particle's actual value."""
    ens = _ensemble()
    g = ens.project(unbiased=False).add_noise(y=_noise())
    cmap = g.conditional_map("y")
    assert cmap.divisor == J
    key = jax.random.key(4)
    y = {"y": jnp.asarray(RNG.normal(size=2))}
    fast = cmap.particle_coefficients(y, key=key)
    slow = cmap.coefficients(ens, y, key=key)
    _close(fast, slow, 1e4)


def test_the_matheron_rules_aligned_path_agrees_with_its_general_path():
    """The general path whitens the particles' values and reads no divisor."""
    ens = _ensemble()
    approx = ens.project(unbiased=False).add_noise(y=_noise())
    plain = Gaussian(
        {n: approx.mean(n) for n in approx.names},
        factors={n: approx.factor(n).to_dense() for n in approx.names},
        block_covs={"y": approx.block_cov("y")},
    )
    key = jax.random.key(8)
    y = {"y": jnp.asarray(RNG.normal(size=2))}
    fast = Matheron().build(ens, approx, "y")
    slow = Matheron().build(ens, plain, "y")
    assert fast.aligned and not slow.aligned
    got, want = fast(y, key=key), slow(y, key=key)
    for n in ("x", "z"):
        _close(got[n], want[n], 1e4)


@pytest.mark.parametrize("rule", (SymmetricSquareRoot(), Matheron()), ids=repr)
def test_localization_without_tapering_is_the_global_update(rule):
    """Localization reads the particles' anomalies off the factor rows too."""
    ens = _ensemble()
    noise = PSDDiagonal(jnp.asarray(1.0 + RNG.uniform(size=2)))
    approx = ens.project(unbiased=False).add_noise(y=noise)
    loc = DomainLocalization(
        {"x": np.arange(3.0)[:, None], "z": np.arange(4.0)[:, None]},
        np.arange(2.0)[:, None],
        radius=100.0,
        max_neighbors=2,
        taper=jnp.ones_like,
    )
    key = jax.random.key(2)
    y = {"y": jnp.asarray(RNG.normal(size=2))}
    want = rule.build(ens, approx, "y")(y, key=key)
    got = LocalizedUpdateRule(rule, loc).build(ens, approx, "y")(y, key=key)
    for n in ("x", "z"):
        _close(got[n], want[n], 1e4)


def test_the_default_approximation_is_unbiased():
    """``kalman`` does not take the flag: its approximation is always J - 1."""
    g = gaussian_approximation(_ensemble(), {"y": _noise()})
    assert type(g) is EnsembleGaussian and g.unbiased and g.divisor == J - 1


def test_the_divisor_is_static_and_survives_flatten():
    g = _ensemble().project(unbiased=False)
    leaves, tree = jax.tree.flatten(g)
    rebuilt = jax.tree.unflatten(tree, leaves)
    assert rebuilt.unbiased is False and rebuilt.divisor == J
    assert tree != jax.tree.structure(_ensemble().project())
    assert "unbiased=False" in repr(g) and "unbiased" not in repr(_ensemble().project())


def test_the_divisor_survives_jit_and_vmap():
    ens = _ensemble()
    R = _noise()

    def realize(e):
        g = e.project(unbiased=False).add_noise(y=R)
        assert g.unbiased is False
        return g.realize_particles(exclude_block_covs="y")["x"]

    _close(jax.jit(realize)(ens), ens["x"])
    family = jax.tree.map(lambda a: jnp.stack([a, 2.0 * a]), ens)
    got = jax.vmap(realize)(family)
    _close(got[0], ens["x"])
    _close(got[1], 2.0 * ens["x"])
