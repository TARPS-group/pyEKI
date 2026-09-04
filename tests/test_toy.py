"""The toy problems: their models, their closed form, and their failures.

Four kinds of test, in the order the module's claims are made:

1. **Contract conformance** — every model returns the documented shape at the
   documented input shape, in the run's dtype, is ``jit``-able and
   ``vmap``-pable, is row-independent and deterministic, and passes
   :func:`pyeki.eki.testing.check_forward_model`. The checker gets its own
   negative tests, since a checker with no failing case is worthless.
2. **Exactness** — the linear problem's closed form against a dense reference
   written here, at three tempering levels; and a full run from an
   exact-moment ensemble reaching that closed form to floating point.
3. **The properties the documentation depends on** — the ladder's advantage
   over a single unit step, the failure fraction's determinism, and the
   subspace confinement at :math:`P \\gg J`.
4. **The user guide's runnable blocks**, with their printed numbers pinned.

These tests do not replace the local forward models in ``test_eki.py``: those
are instrumented — one records every argument it is given — and their
references are written locally on purpose, which is what makes them regression
tests for the layer rather than for this module.
"""
from __future__ import annotations

import dataclasses
import math
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import prints_as

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki import toy
from pyeki.eki import (
    AdaptiveESSSchedule,
    EKIError,
    EKIState,
    FixedSchedule,
    TransformUpdate,
    run,
)
from pyeki.eki.testing import check_forward_model
from pyeki.gauss import Gaussian
from pyeki.linalg import DensePSD, PSDDiagonal, PSDLowRank, UnsupportedOpError

EPS = float(np.finfo(np.float64).eps)

#: One instance of each model, with the sizes it should answer at.
PROBLEMS = [
    ("linear_gaussian", toy.linear_gaussian(), 4, 8),
    ("exponential_decay", toy.exponential_decay(), 2, 12),
    ("restricted_decay", toy.restricted_decay(), 2, 12),
]
IDS = [name for name, *_ in PROBLEMS]


def _ensemble(n_members: int, u_dim: int, seed: int = 0):
    """A pseudo-random ensemble, drawn outside the models under test."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(n_members, u_dim)))


def _identical(got, want) -> bool:
    """Bit-identity, counting two ``nan`` s as equal: failed rows are legal."""
    return np.array_equal(np.asarray(got), np.asarray(want), equal_nan=True)


def _exact_moment_ensemble(J: int, mu: np.ndarray, F: np.ndarray) -> np.ndarray:
    """An ensemble whose empirical moments are exactly ``mu`` and ``F @ F.T``.

    The QR-of-ones construction the joint Gaussian contract specifies: the
    complete QR of the all-ones vector in R^J gives columns that are
    orthonormal and orthogonal to it, and ``mu + sqrt(J - 1) E F.T`` then has
    mean ``mu`` and empirical covariance ``F F.T`` under the package's J - 1
    divisor. Only J >= k + 1 binds. Written out here rather than imported,
    as the package's conformance rules require of a reference.
    """
    k = F.shape[1]
    assert J >= k + 1, "the construction needs J >= k + 1"
    Q, _ = np.linalg.qr(np.ones((J, 1)), mode="complete")
    return mu + np.sqrt(J - 1) * Q[:, 1 : k + 1] @ F.T


# ===========================================================================
# 1. every model against the forward-model contract
# ===========================================================================


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_1_every_model_returns_the_documented_shape_and_dtype(
    name, problem, u_dim, v_dim
):
    """(J, P) in, (J, N) out, in the run's working dtype, at three sizes of J.

    A model is independent of the ensemble size, so one instance answers at
    every J. The dtype matters because a narrower return is promoted with a
    warning rather than rejected, so no other test would catch a float32
    model here.
    """
    assert (problem.u_dim, problem.v_dim) == (u_dim, v_dim)
    for n_members in (2, 5, 64):
        predictions = problem.forward(_ensemble(n_members, u_dim))
        assert predictions.shape == (n_members, v_dim)
        assert predictions.dtype == jnp.float64
    assert jnp.shape(problem.y) == (v_dim,)
    assert jnp.shape(problem.u_true) == (u_dim,)
    assert problem.prior.dim == u_dim
    assert problem.noise_cov.shape == (v_dim, v_dim)


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_1_no_problem_is_callable(name, problem, u_dim, v_dim):
    """A problem is not a forward model, and must not be mistakable for one.

    ``run`` takes the callable, the observation and the noise covariance as
    three arguments, and the EKI contract excludes a container accepted in
    their place. A ``__call__`` here would make the container look like the
    interface, and a reader's own model would then be a class they pass to
    ``run``, which does not work.
    """
    assert not callable(problem), (
        f"{name} became callable; pass problem.forward, problem.y and "
        f"problem.noise_cov as three arguments"
    )


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_2_every_model_is_jittable_and_vmappable(name, problem, u_dim, v_dim):
    """A convenience of the toy models, so it is tested as one.

    Not a property of forward models in general — the contract requires
    neither — but these are used across the suite and the documentation, so a
    change that broke tracing would be one to notice.
    """
    ensemble = _ensemble(6, u_dim)
    eager = problem.forward(ensemble)
    assert _identical(jax.jit(problem.forward)(ensemble), eager)

    stacked = jnp.stack([ensemble, ensemble + 0.25])
    mapped = jax.vmap(problem.forward)(stacked)
    assert mapped.shape == (2, 6, v_dim)
    assert _identical(mapped[0], eager)
    assert _identical(mapped[1], problem.forward(ensemble + 0.25))


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_3_every_model_is_row_independent_and_deterministic(
    name, problem, u_dim, v_dim
):
    """Row j of the return depends only on row j of the argument.

    The one requirement beyond the shapes that nothing inside a run detects.
    Checked by permuting the members — for a row-independent model the
    predictions permute with them, bit-exactly, since the rows are the same
    set of members either way — and by re-evaluating a subset of them, which
    catches the symmetric couplings a permutation cannot. The subset
    comparison is to a tolerance: a differently shaped batch legitimately
    takes a different matmul kernel and rounds differently in the last bits.
    """
    ensemble = _ensemble(7, u_dim)
    predictions = problem.forward(ensemble)
    permutation = jnp.array([3, 1, 0, 6, 5, 4, 2])
    assert _identical(
        problem.forward(ensemble[permutation, :]), predictions[permutation, :]
    )
    for subset in (jnp.array([0, 4]), jnp.array([2, 3, 6])):
        np.testing.assert_allclose(
            problem.forward(ensemble[subset, :]),
            predictions[subset, :],
            rtol=0,
            atol=1e-12,
        )
    assert _identical(problem.forward(ensemble), predictions)


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_4_every_model_passes_check_forward_model(name, problem, u_dim, v_dim):
    """The harness a user runs their own model through, run on ours."""
    check_forward_model(problem.forward, u_dim=u_dim, v_dim=v_dim)


def test_4_check_forward_model_rejects_the_defects_it_claims_to_catch():
    """A checker with no failing case is worthless, so each check gets one.

    The two row-coupling cases are the valuable ones: that defect is the one
    the contract calls undetectable, and it is undetectable *from inside a
    run* rather than absolutely. They are separate cases because a symmetric
    coupling survives a permutation of the members and only the subset
    comparison catches it.
    """
    times = jnp.linspace(0.25, 1.0, 4)
    decay = jax.vmap(lambda u: u[0] * jnp.exp(-u[1] * times))

    def normalized(ensemble):
        # Symmetric coupling: normalizes across the ensemble. Survives a
        # permutation, so only the subset check sees it.
        return decay(ensemble - jnp.mean(ensemble, axis=0))

    def ordered(ensemble):
        # Order-dependent coupling: a running total down the rows.
        return decay(jnp.cumsum(ensemble, axis=0))

    def per_member(ensemble):
        # Written for one member, and handed the whole ensemble: returns one
        # prediction vector. The other failure mode of the same mistake is a
        # raise from inside JAX, which needs no check to be noticed.
        member = ensemble[0]
        return member[0] * jnp.exp(-member[1] * times)

    def narrow(ensemble):
        return decay(ensemble).astype(jnp.float32)

    def integer(ensemble):
        return jnp.zeros((ensemble.shape[0], times.size), dtype=jnp.int64)

    def stochastic(ensemble):
        key = jax.random.key(int(np.random.default_rng().integers(1 << 30)))
        return decay(ensemble) + jax.random.normal(key, (ensemble.shape[0], 4))

    for model, fragment in [
        (ordered, "permuting the members changed more than the order"),
        (normalized, "alongside different members"),
        (per_member, "returned shape"),
        (narrow, "float32"),
        (integer, "not a real floating type"),
        (stochastic, "not deterministic"),
    ]:
        with pytest.raises(AssertionError, match=fragment):
            check_forward_model(model, u_dim=2, v_dim=4)

    # The declaration is the escape hatch, and it suppresses exactly the two
    # checks a stochastic model cannot satisfy -- and nothing else. Every
    # other check is re-run under it, since asserting that for one of them
    # would pass an implementation that returned early.
    check_forward_model(stochastic, u_dim=2, v_dim=4, stochastic=True)
    for model, fragment in [
        (per_member, "returned shape"),
        (narrow, "float32"),
        (integer, "not a real floating type"),
    ]:
        with pytest.raises(AssertionError, match=fragment):
            check_forward_model(model, u_dim=2, v_dim=4, stochastic=True)


def test_4_check_forward_model_checks_the_second_ensemble_size_and_the_argument():
    """Two claims of its own that no defect above exercises.

    A model that answers at one ensemble size and not another passes every
    other check, and the Notes' argument for having no mutation check — that
    a `jax.Array` cannot be written into — depends on the argument being one.
    """
    times = jnp.linspace(0.25, 1.0, 4)
    decay = jax.vmap(lambda u: u[0] * jnp.exp(-u[1] * times))

    def fixed_size(ensemble):
        # A wrapper that preallocated for one ensemble size, as a subprocess
        # wrapper naturally does, and was then handed another.
        predictions = np.full((6, 4), np.nan)
        rows = min(ensemble.shape[0], 6)
        predictions[:rows] = np.asarray(decay(ensemble))[:rows]
        return predictions[: ensemble.shape[0]]

    with pytest.raises(AssertionError, match="returned shape"):
        check_forward_model(fixed_size, u_dim=2, v_dim=4)

    seen = []

    def recording(ensemble):
        seen.append(ensemble)
        return decay(ensemble)

    check_forward_model(recording, u_dim=2, v_dim=4)
    assert len(seen) == 5, "the docstring promises five calls"
    assert {tuple(a.shape) for a in seen} == {(6, 2), (7, 2), (2, 2)}
    for argument in seen:
        assert isinstance(argument, jax.Array), (
            "the Notes argue no mutation check is needed because the argument "
            "is a jax.Array, which cannot be written into"
        )
        assert not np.asarray(argument).flags.writeable


def test_4_check_forward_model_compares_where_the_failures_are():
    """A model whose failing rows move between the full ensemble and a subset.

    The non-finite pattern is compared, not only the finite values: a domain
    that depends on the other members is a coupling like any other, and the
    finite entries alone may agree.
    """
    times = jnp.linspace(0.25, 1.0, 4)

    def moving_domain(ensemble):
        # Fails whichever member has the smallest first parameter -- which
        # depends on the company it is in.
        predictions = jax.vmap(lambda u: u[0] * jnp.exp(-u[1] * times))(ensemble)
        worst = jnp.argmin(ensemble[:, 0])
        return predictions.at[worst].set(jnp.nan)

    with pytest.raises(AssertionError, match="non-finite entries"):
        check_forward_model(moving_domain, u_dim=2, v_dim=4)


def test_4_check_forward_model_accepts_a_failing_model_and_a_numpy_one():
    """Non-finite rows and non-JAX returns are legal, and must not be flagged.

    The failing model is the case the nan-aware comparison exists for: two
    permuted ``nan`` rows are not equal under the ordinary comparison, so a
    naive checker would reject every model that can fail.
    """
    failing = toy.restricted_decay()
    predictions = np.asarray(failing.forward(_ensemble(6, 2)))
    assert not np.isfinite(predictions).all(), "no member failed, so this is vacuous"
    assert np.isfinite(predictions).any(), "every member failed, so this is vacuous"
    check_forward_model(failing.forward, u_dim=2, v_dim=12)

    times = np.linspace(0.25, 1.0, 4)

    def numpy_model(ensemble):
        members = np.asarray(ensemble)  # a read-only view; only read
        return [[float(u[0] * np.exp(-u[1] * t)) for t in times] for u in members]

    check_forward_model(numpy_model, u_dim=2, v_dim=4)


# ===========================================================================
# 2. exactness, for the linear problem
# ===========================================================================


def _dense_posterior(problem, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """The precision-form posterior, in plain dense NumPy.

    Written here rather than routed through any package code, so that the
    comparison is between two independent paths. Valid only for an invertible
    prior covariance, which is what these tests build.
    """
    C0 = np.asarray(problem.prior.cov.to_dense())
    G = np.asarray(problem.G.to_dense())
    R = np.asarray(problem.noise_cov.to_dense())
    prior_precision = np.linalg.inv(C0)
    data_precision = G.T @ np.linalg.solve(R, G)
    cov = np.linalg.inv(prior_precision + beta * data_precision)
    mean = cov @ (
        prior_precision @ np.asarray(problem.prior.mean)
        + beta * G.T @ np.linalg.solve(R, np.asarray(problem.y))
    )
    return mean, cov


def _general_linear_problem():
    """A `LinearGaussian` with a non-zero prior mean and correlated noise.

    Every problem `linear_gaussian` builds has prior mean exactly zero and
    diagonal noise, which makes two of the three terms of the conditioning
    mean invisible: dropping the prior-mean term, or the ``G m_0`` residual
    term, changes nothing that the shipped fixture can see. This fixture
    restores both, so the exactness test below covers the whole formula.
    """
    base = toy.linear_gaussian(u_dim=4, v_dim=8, seed=5)
    rng = np.random.default_rng(11)
    factor = np.tril(rng.normal(size=(4, 4))) + 4.0 * np.eye(4)
    noise = rng.normal(size=(8, 8))
    return dataclasses.replace(
        base,
        prior=Gaussian(
            jnp.asarray(rng.normal(size=4)),
            DensePSD.from_matrix(jnp.asarray(factor @ factor.T)),
        ),
        noise_cov=DensePSD.from_matrix(
            jnp.asarray(noise @ noise.T / 8.0 + np.eye(8))
        ),
    )


@pytest.mark.parametrize("beta", [0.25, 1.0, 2.0])
def test_5_the_closed_form_posterior_covers_the_whole_conditioning_formula(beta):
    """The same check on a problem whose prior mean and noise are general.

    The tolerance is `1e3 * EPS * scale` as in the sibling test, and it is
    not portable to arbitrary sizes: at `u_dim > v_dim` and level 2 the
    *reference* loses accuracy, since it inverts the precision matrix. Widen
    the parametrization and this tolerance must be revisited.
    """
    problem = _general_linear_problem()
    posterior = problem.posterior(beta)
    mean_ref, cov_ref = _dense_posterior(problem, beta)
    scale = max(np.abs(mean_ref).max(), np.abs(cov_ref).max())

    assert np.abs(np.asarray(problem.prior.mean)).min() > 0.1, "fixture is vacuous"
    np.testing.assert_allclose(
        posterior.mean, mean_ref, rtol=0, atol=1e3 * EPS * scale
    )
    np.testing.assert_allclose(
        posterior.cov.to_dense(), cov_ref, rtol=0, atol=1e3 * EPS * scale
    )


def test_5_the_prior_mean_and_the_residual_term_are_both_load_bearing():
    """Both terms the shipped fixture cannot see, shown to matter.

    Without this, `test_5` would pass an implementation that conditioned on
    `y` alone, ignoring the prior mean and the `G m_0` residual — because
    every `linear_gaussian` problem has prior mean zero.
    """
    problem = _general_linear_problem()
    zero_mean = dataclasses.replace(
        problem, prior=Gaussian(jnp.zeros(problem.u_dim), problem.prior.cov)
    )
    difference = np.abs(
        np.asarray(problem.posterior().mean) - np.asarray(zero_mean.posterior().mean)
    ).max()
    assert difference > 0.05, difference


@pytest.mark.parametrize("beta", [0.25, 1.0, 2.0])
def test_5_the_closed_form_posterior_matches_a_dense_reference(beta):
    """`posterior(beta)` is the posterior at noise R / beta, exactly.

    The layer's own conditioning is already checked against a dense reference
    by the joint Gaussian contract's obligation 14. What this pins is the toy
    model's composition: that it divides the *noise* by beta, and that it
    divides it at all. The mis-scaling this guards is the layer's signature
    silent bug; the next test measures how far off it would be.
    """
    problem = toy.linear_gaussian()
    posterior = problem.posterior(beta)
    mean_ref, cov_ref = _dense_posterior(problem, beta)
    scale = max(np.abs(mean_ref).max(), np.abs(cov_ref).max())

    assert isinstance(posterior, Gaussian)
    assert isinstance(posterior.cov, PSDLowRank)
    assert posterior.cov.F.shape == (problem.u_dim, problem.u_dim)
    np.testing.assert_allclose(
        posterior.mean, mean_ref, rtol=0, atol=1e3 * EPS * scale
    )
    np.testing.assert_allclose(
        posterior.cov.to_dense(), cov_ref, rtol=0, atol=1e3 * EPS * scale
    )


def test_5_beta_is_load_bearing_and_the_tolerance_would_catch_it():
    """The two betas must be far apart at the tolerance test 5 uses.

    Without this, test 5 would pass an implementation that ignored ``level``
    altogether, since its default agrees.
    """
    problem = toy.linear_gaussian()
    gap = np.abs(
        np.asarray(problem.posterior(0.5).mean)
        - np.asarray(problem.posterior(1.0).mean)
    ).max()
    # Test 5 compares at roughly 1e3 * EPS * scale, about 2e-13 here, so a
    # gap of 0.01 is ten orders of magnitude above what it would accept.
    assert gap > 0.01, "the two betas are indistinguishable, so test 5 has no teeth"


@pytest.mark.parametrize(
    "increments", [(1.0,), (0.5, 0.5), (0.25,) * 4, (0.5, 0.25, 0.25)]
)
def test_6_a_ladder_from_an_exact_moment_ensemble_reaches_the_closed_form(
    increments,
):
    """A run converges on the toy model's own closed form, to floating point.

    ``test_eki.py``'s telescoping test pins the *layer* against a reference
    written there. This pins that ``LinearGaussian.posterior`` is the same
    posterior a run of ``LinearGaussian.forward`` reaches — the claim every
    comparison in the documentation rests on, and the one that would break if
    the closed form were built from a different problem than the model.
    Not redundant with that test; do not delete it as such.
    """
    problem = toy.linear_gaussian()
    J = 12
    factor = np.asarray(problem.prior.cov.factor().to_dense())
    members = jnp.asarray(
        _exact_moment_ensemble(J, np.asarray(problem.prior.mean), factor)
    )
    state = EKIState(members, 0.0, 0, jax.random.key(0))

    result = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=FixedSchedule(increments),
        update=TransformUpdate(),
        max_steps=len(increments),
    )

    closed = problem.posterior()
    ensemble = np.asarray(result.ensemble)
    anomalies = ensemble - ensemble.mean(axis=0)
    closed_cov = np.asarray(closed.cov.to_dense())
    scale = max(np.abs(np.asarray(closed.mean)).max(), np.abs(closed_cov).max())
    np.testing.assert_allclose(
        ensemble.mean(axis=0),
        np.asarray(closed.mean),
        rtol=0,
        atol=1e3 * EPS * scale,
    )
    np.testing.assert_allclose(
        anomalies.T @ anomalies / (J - 1),
        closed_cov,
        rtol=0,
        atol=1e3 * EPS * scale,
    )


def test_6_the_posterior_size_guard_raises_before_allocating():
    """A P-by-k factor above the budget is refused, naming both sizes.

    The run has no such limit, so the guard must not read as one on the
    problem: a high-dimensional problem is invertible where its closed form
    cannot be written down.
    """
    big = toy.linear_gaussian(u_dim=5000, v_dim=10)
    with pytest.raises(ValueError, match=r"5000-by-5000"):
        big.posterior()
    assert big.forward(_ensemble(3, 5000)).shape == (3, 10)


@pytest.mark.parametrize("beta", [0.0, -1.0, float("inf"), float("nan")])
def test_6_a_non_positive_beta_raises(beta):
    with pytest.raises(ValueError, match="positive and finite"):
        toy.linear_gaussian().posterior(beta)


# ===========================================================================
# 3. the properties the documentation depends on
# ===========================================================================


@pytest.mark.parametrize("seed", range(8))
def test_7_the_ladder_beats_a_single_unit_step_on_the_decay_problem(seed):
    """The property tutorials 2 and 3 and notebook 02 are built on.

    Asserted over **every** observation seed the factory's docstring claims,
    not only the default: the claim is that the two answers differ reliably
    rather than coincidentally, and a single-seed test cannot distinguish
    those. Measured over seeds 0 to 7: the rate gap is 0.104 to 0.228 and the
    ladder is nearer the u_true by a factor of 2.70 to 41.1, so the thresholds
    below carry margins of 1.3x, 1.35x and 1.4x on the worst seed.

    An earlier version of this test asserted `ladder_error < one_step_error /
    3` and `< 0.05` at seed 0 alone, where they hold with room to spare; seeds
    4 and 7 respectively break both. The docstring claiming eight-seed
    robustness was therefore false, and this is what makes it true.
    """
    problem = toy.exponential_decay(seed=seed)
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    common = (problem.forward, problem.y, problem.noise_cov)

    one_step = run(state, *common, schedule=FixedSchedule((1.0,)))
    ladder = run(state, *common, schedule=AdaptiveESSSchedule())

    u_true = np.asarray(problem.u_true)
    one_step_error = np.abs(np.asarray(one_step.mean) - u_true).max()
    ladder_error = np.abs(np.asarray(ladder.mean) - u_true).max()
    gap = np.abs(np.asarray(one_step.mean) - np.asarray(ladder.mean)).max()

    assert ladder.n_completed_steps > 1
    assert gap > 0.08, gap
    assert ladder_error < one_step_error / 2, (ladder_error, one_step_error)
    assert ladder_error < 0.08, ladder_error


def test_8_the_restricted_model_fails_exactly_its_out_of_domain_members():
    """The failure is a deterministic function of the parameters, not a rate.

    Which members fail is decided by the rate alone, and the whole row goes
    non-finite when it does — a partially finite row would be read as a valid
    member with a huge misfit, which is the failure mode that stalls an
    adaptive ladder instead of flagging itself.
    """
    problem = toy.restricted_decay()
    ensemble = _ensemble(64, 2)
    predictions = np.asarray(problem.forward(ensemble))
    finite_rows = np.isfinite(predictions).all(axis=1)
    expected = np.asarray(ensemble)[:, 1] > 0.0

    assert np.array_equal(finite_rows, expected)
    assert np.array_equal(np.isfinite(predictions).any(axis=1), expected), (
        "a partially finite row would count as a valid member"
    )
    assert 0 < (~finite_rows).sum() < 64, "the fixture must have both kinds"


def test_8_the_failure_fraction_is_monotone_in_the_rate_floor():
    """The knob a notebook sweeps, and the direction it moves in."""
    ensemble = _ensemble(64, 2)
    counts = []
    for rate_floor in (-1.0, -0.5, 0.0, 0.5, 1.0):
        problem = toy.restricted_decay(rate_floor=rate_floor)
        predictions = np.asarray(problem.forward(ensemble))
        counts.append(int((~np.isfinite(predictions).all(axis=1)).sum()))
    assert counts == sorted(counts) and counts[0] < counts[-1], counts
    # Only the floor moves: everything else is the exponential_decay problem.
    plain = toy.exponential_decay()
    restricted = toy.restricted_decay()
    for field in ("times", "y", "u_true"):
        assert _identical(getattr(plain, field), getattr(restricted, field))
    # "the same problem in every other respect" includes these two.
    assert _identical(plain.prior.mean, restricted.prior.mean)
    assert _identical(plain.prior.cov.diag(), restricted.prior.cov.diag())
    assert _identical(plain.noise_cov.diag(), restricted.noise_cov.diag())


def test_8_a_run_against_the_restricted_model_repairs_and_reports():
    """Repair is the default, and a failed run is still a completed one.

    The numbers are pinned because the user guide prints them.
    """
    problem = toy.restricted_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    common = (problem.forward, problem.y, problem.noise_cov)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run(state, *common, schedule=AdaptiveESSSchedule())

    assert result.min_n_valid == 55
    assert np.array_equal(np.asarray(result.stacked.n_valid), [55, 64, 64, 64, 64])
    assert [w for w in caught if "evaluations failed" in str(w.message)]
    assert np.abs(np.asarray(result.mean) - np.asarray(problem.u_true)).max() < 0.05

    prints_as(result.mean, [1.9801, 1.474])  # the page prints this too
    with pytest.raises(EKIError, match="finite"):
        run(state, *common, schedule=AdaptiveESSSchedule(), on_failure="raise")


def test_9_the_high_dimensional_problem_is_confined_and_over_confident():
    """The subspace bound, against the closed form rather than against nothing.

    Every iterate lies in the affine span of the initial ensemble, of
    dimension at most J - 1 — so at P = 2000 with J = 40 the run reports a
    spread that the exact posterior contradicts by a factor of seventy, with
    nothing raised and no history field flagging it. Both halves are the
    lesson; the closed form is what makes the second half sayable.
    """
    problem = toy.linear_gaussian(u_dim=2000, v_dim=40)
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=40)
    result = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )

    initial = np.asarray(state.ensemble)
    displacement = np.asarray(result.ensemble) - initial.mean(axis=0)
    # The bound, not the realized value. The rank is 39 and the run takes 5
    # steps today, but pinning either makes an intentional schedule change
    # look like a regression, and tutorial 5 hedges them for that reason.
    assert np.linalg.matrix_rank(displacement) <= state.n_members - 1
    assert result.status == "schedule_exhausted"

    ensemble_sd = float(result.ensemble.std(axis=0, ddof=1).mean())
    exact_sd = float((problem.posterior().cov.diag() ** 0.5).mean())
    assert exact_sd / ensemble_sd > 20.0, (ensemble_sd, exact_sd)
    # The numbers the user guide prints, to the precision it prints them.
    prints_as(ensemble_sd, 0.0140)
    prints_as(exact_sd, 0.9900)


# ===========================================================================
# 4. construction, reproducibility, and the guide's blocks
# ===========================================================================


def test_10_a_problem_is_reproducible_from_its_seed_and_varies_with_it():
    """A docs build must see the same numbers every time it runs."""
    for factory in (toy.linear_gaussian, toy.exponential_decay, toy.restricted_decay):
        first, again = factory(seed=3), factory(seed=3)
        different = factory(seed=4)
        assert _identical(first.y, again.y)
        assert _identical(first.u_true, again.u_true)
        assert not _identical(first.y, different.y)


def test_10_the_factories_and_classes_validate_as_documented():
    problem = toy.linear_gaussian()
    with pytest.raises(ValueError, match="at least 1"):
        toy.linear_gaussian(u_dim=0)
    with pytest.raises(ValueError, match="must be positive"):
        toy.linear_gaussian(noise_std=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        toy.exponential_decay(t_max=-1.0)
    with pytest.raises(ValueError, match="at least 1"):
        toy.restricted_decay(n_times=0)
    with pytest.raises(ValueError, match="outside the valid domain"):
        toy.restricted_decay(rate_floor=2.0)  # above the true rate of 1.5

    with pytest.raises(TypeError, match="must be a pyeki.linalg.LinOp"):
        dataclasses.replace(problem, G=np.zeros((8, 4)))
    with pytest.raises(ValueError, match=r"LinearGaussian.y: expected an array"):
        dataclasses.replace(problem, y=jnp.zeros(3))
    with pytest.raises(ValueError, match="the prior has dimension"):
        dataclasses.replace(problem, prior=toy.linear_gaussian(u_dim=5).prior)


def test_11_the_forward_model_pages_checker_block_runs():
    """The `check_forward_model` block added to the forward-model guide.

    Its ``forward`` is that page's own opening example, so the call has to
    hold at the sizes the page states.
    """
    times = jnp.array([0.5, 1.0, 2.0])

    def forward(ensemble):  # (J, 2) in
        return ensemble[:, 0:1] * jnp.exp(-ensemble[:, 1:2] * times)  # (J, 3) out

    check_forward_model(forward, u_dim=2, v_dim=3)


def test_11_the_toy_models_page_blocks_run():
    """Every runnable block of the user guide's toy-models page, in order.

    Its printed values are pinned here, so the page cannot rot unnoticed.
    Conformance obligation 26 of the EKI contract is the rule; the
    forward-model page's own example is the pattern.
    """
    problem = toy.linear_gaussian(u_dim=4, v_dim=8)
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=32)
    result = run(
        state,
        problem.forward,
        problem.y,
        problem.noise_cov,
        schedule=AdaptiveESSSchedule(),
    )
    exact = problem.posterior()
    fitted = Gaussian.from_samples(result.ensemble)

    prints_as(result.mean, [-1.3289, 1.362, 0.6229, 0.1361])
    prints_as(exact.mean, [-1.3303, 1.3749, 0.6281, 0.1409])
    prints_as(problem.u_true, [-1.4009, 1.4321, 0.6248, 0.2005])
    # The page says "within 0.013"; measured 0.01292, so the threshold is
    # stated loosely enough that a change of a few percent does not fail it.
    assert np.abs(np.asarray(result.mean) - np.asarray(exact.mean)).max() < 0.015
    prints_as(fitted.cov.diag() ** 0.5, [0.0686, 0.1272, 0.1167, 0.0877])
    prints_as(exact.cov.diag() ** 0.5, [0.0687, 0.1276, 0.1165, 0.0876])

    # The correlated-noise variant, whose R the page names.
    rng = np.random.default_rng(1)
    M = rng.normal(size=(8, 8))
    R = jnp.asarray(M @ M.T / 8 + 0.01 * np.eye(8))
    correlated = dataclasses.replace(problem, noise_cov=DensePSD.from_matrix(R))
    prints_as(correlated.posterior().mean, [-1.4093, 0.8248, 0.4646, 0.0692])


def _dense_posterior_from_blocks(problem, beta: float):
    """The block form of the posterior, which never inverts the prior.

    ``_dense_posterior`` inverts ``C0``, so it cannot check the claim that a
    *singular* prior covariance is fine — the case a precision-form posterior
    cannot express at all. This reference uses only the observation-side
    solve, so it holds at any prior rank.
    """
    C0 = np.asarray(problem.prior.cov.to_dense())
    G = np.asarray(problem.G.to_dense())
    R = np.asarray(problem.noise_cov.to_dense()) / beta
    m0 = np.asarray(problem.prior.mean)
    cross = C0 @ G.T
    solved = np.linalg.solve(
        G @ cross + R, np.column_stack([np.asarray(problem.y) - G @ m0, cross.T])
    )
    return m0 + cross @ solved[:, 0], C0 - cross @ solved[:, 1:]


@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_5_a_singular_prior_covariance_works_and_narrows_the_factor(beta):
    """The one claim the closed form makes that the test file's own reference
    cannot check, so it gets a reference that can.

    A rank-2 prior in four dimensions: the posterior factor is ``(4, 2)``, not
    ``(4, 4)``, and the answer matches the block form of the conditioning
    identity. Every problem `linear_gaussian` builds has a full-rank diagonal
    prior, so without this `k` and `P` are the same number everywhere and
    `latent_dim = self.u_dim` would pass the whole suite.
    """
    base = toy.linear_gaussian(u_dim=4, v_dim=6, seed=2)
    factor = jnp.asarray(np.random.default_rng(4).normal(size=(4, 2)))
    problem = dataclasses.replace(
        base, prior=Gaussian(jnp.asarray([0.5, -1.0, 0.25, 2.0]), PSDLowRank(factor))
    )

    posterior = problem.posterior(beta)
    assert posterior.cov.F.shape == (4, 2), "the factor must narrow with the prior"
    mean_ref, cov_ref = _dense_posterior_from_blocks(problem, beta)
    scale = max(np.abs(mean_ref).max(), np.abs(cov_ref).max())
    np.testing.assert_allclose(
        posterior.mean, mean_ref, rtol=0, atol=1e3 * EPS * scale
    )
    np.testing.assert_allclose(
        posterior.cov.to_dense(), cov_ref, rtol=0, atol=1e3 * EPS * scale
    )


def test_5_the_two_standard_deviations_reach_the_arrays_they_document():
    """`prior_std` and `noise_std` have to do what their names say.

    Nothing else in the suite calls either factory at a non-default scale, so
    hardcoding the defaults inside `linear_gaussian` passed every test.
    """
    problem = toy.linear_gaussian(u_dim=3, v_dim=5, prior_std=4.0, noise_std=0.25)
    np.testing.assert_allclose(problem.prior.cov.diag(), np.full(3, 16.0))
    np.testing.assert_allclose(problem.noise_cov.diag(), np.full(5, 0.0625))
    decay = toy.exponential_decay(noise_std=0.5)
    np.testing.assert_allclose(decay.noise_cov.diag(), np.full(12, 0.25))


def test_5_the_prior_and_the_noise_must_support_what_the_posterior_needs():
    """The two documented `UnsupportedOpError` paths."""
    problem = toy.linear_gaussian(u_dim=4, v_dim=6)
    # PSDLowRank withholds `whiten`, which conditioning needs of the noise.
    low_rank = PSDLowRank(jnp.asarray(np.random.default_rng(0).normal(size=(6, 2))))
    with pytest.raises(UnsupportedOpError):
        dataclasses.replace(problem, noise_cov=low_rank).posterior()


def test_6_the_decay_problems_answer_at_a_size_they_were_not_built_at():
    """`n_times` and `t_max` reach `times`, and the grid is half-open.

    A constant `v_dim`, and a grid starting at 0 rather than at
    `t_max / n_times`, both passed the suite: the second is a materially
    different problem, since at `t = 0` the prediction is the amplitude
    exactly and carries no information about the rate.
    """
    problem = toy.exponential_decay(n_times=5, t_max=10.0)
    assert problem.v_dim == 5
    np.testing.assert_allclose(problem.times, [2.0, 4.0, 6.0, 8.0, 10.0])
    assert problem.forward(_ensemble(3, 2)).shape == (3, 5)
    assert float(problem.times[0]) > 0.0, "t = 0 carries no rate information"
    assert toy.restricted_decay(n_times=7).v_dim == 7


def test_6_the_documented_true_parameters_are_what_the_factories_build():
    """Every accuracy assertion is relative to `problem.u_true` itself, so the
    documented values `(2.0, 1.5)` were unpinned and free to drift."""
    for factory in (toy.exponential_decay, toy.restricted_decay):
        np.testing.assert_array_equal(np.asarray(factory().u_true), [2.0, 1.5])
    problem = toy.linear_gaussian()
    # The linear problem's u_true is a prior draw, so it is pinned by digits.
    prints_as(problem.u_true, [-1.4009, 1.4321, 0.6248, 0.2005])


def test_8_the_failure_fraction_matches_its_closed_form_under_the_prior():
    """The Notes give Phi((rate_floor - m1) / sigma1); check against it.

    A closed form exists, so the package's rules say compare against it rather
    than against a tolerance. The prior is N((1, 1), I) and the default floor
    is 0, so the expected fraction is Phi(-1) = 0.1587.
    """
    problem = toy.restricted_decay()
    members = problem.prior.sample(jax.random.key(7), 20_000)
    predictions = np.asarray(problem.forward(members))
    failed = (~np.isfinite(predictions).all(axis=1)).mean()

    mean, sd = float(problem.prior.mean[1]), float(problem.prior.cov.diag()[1]) ** 0.5
    expected = 0.5 * math.erfc(-(problem.rate_floor - mean) / (sd * math.sqrt(2.0)))
    assert expected == pytest.approx(0.15866, abs=1e-5)
    # Three standard errors of a 20,000-sample binomial is about 0.008.
    assert failed == pytest.approx(expected, abs=0.01), (failed, expected)


def test_8_the_domain_boundary_is_strict_in_both_places():
    """`>` not `>=`, in the model and in the u_true guard.

    Both were untested: the model's mask is computed from a continuous draw
    where exact equality has probability zero, and the u_true guard was only
    exercised at a floor strictly above the true rate.
    """
    problem = toy.restricted_decay(rate_floor=0.5)
    on_boundary = jnp.array([[2.0, 0.5], [2.0, 0.5 + 1e-12]])
    finite = np.isfinite(np.asarray(problem.forward(on_boundary))).all(axis=1)
    assert not finite[0], "a rate exactly at the floor is outside the domain"
    assert finite[1]
    # And a floor exactly at the true rate is refused, not only one above it.
    with pytest.raises(ValueError, match="outside the valid domain"):
        toy.restricted_decay(rate_floor=1.5)


# ===========================================================================
# 5. regressions for the defects an adversarial review found
# ===========================================================================


def test_12_regression_the_checker_rejects_a_coupling_in_a_small_observable():
    """A per-element tolerance, not one global scale set by the largest value.

    With a single global scale, a model whose observables span orders of
    magnitude could couple its small ones freely: a 50% coupling in an O(1)
    component was invisible beside a component of size 1e8, while the same
    coupling alone was caught with a margin of 6e5. That is the mixed-units
    case every real forward model presents.
    """

    def mixed(ensemble):
        big = 1e8 * ensemble[:, 0]
        small = ensemble[:, 1] + 0.5 * jnp.mean(ensemble[:, 1])
        return jnp.stack([big, small], axis=-1)

    with pytest.raises(AssertionError, match="alongside different members"):
        check_forward_model(mixed, u_dim=2, v_dim=2)


def test_12_regression_the_checker_refuses_an_ensemble_too_small_to_check():
    """At J < 3 both row-independence comparisons are vacuous, so it raises.

    At J = 1 the out-of-bounds subset index is silently clamped by JAX, making
    the comparison a tautology, and a definitively coupled model passed. At
    J = 2 the subset is one member twice and a fair permutation is the
    identity for most seeds.
    """

    def coupled(ensemble):
        return ensemble - jnp.mean(ensemble, axis=0)

    for n_members in (1, 2):
        with pytest.raises(ValueError, match="n_members must be at least 3"):
            check_forward_model(coupled, u_dim=2, v_dim=2, n_members=n_members)
    with pytest.raises(AssertionError):
        check_forward_model(coupled, u_dim=2, v_dim=2, n_members=3)


@pytest.mark.parametrize("seed", range(12))
def test_12_regression_the_permutation_is_never_the_identity(seed):
    """A fair draw returns the identity often at small J, asserting nothing."""

    def ordered(ensemble):
        return jnp.cumsum(ensemble, axis=0)

    with pytest.raises(AssertionError, match="permuting the members"):
        check_forward_model(ordered, u_dim=2, v_dim=2, n_members=3, seed=seed)


def test_12_regression_the_restricted_model_differentiates():
    """`jnp.where` evaluates both branches, so the discarded one must be safe.

    Without clamping the rate inside the valid branch, a member below the
    floor computes `exp(-rate * t)` in the discarded branch, overflows to
    `inf`, and the derivative returns `nan` from `0 * inf`. The threshold is
    a rate of about -236 at the default `t_max` — but only about -29 in
    float32, and it scales with `t_max`, so it is not a remote corner.
    """
    problem = toy.restricted_decay()
    total = lambda ensemble: jnp.nansum(problem.forward(ensemble))  # noqa: E731
    for rate in (-10.0, -300.0, -1e5):
        ensemble = jnp.array([[2.0, rate], [2.0, 1.5]])
        gradient = np.asarray(jax.grad(total)(ensemble))
        assert np.isfinite(gradient).all(), (rate, gradient)
    # And the failing row is still wholly non-finite, which is the signal.
    predictions = np.asarray(problem.forward(jnp.array([[2.0, -300.0]])))
    assert not np.isfinite(predictions).any()


def test_12_regression_the_size_guard_bounds_the_transform_too():
    """The conditioning forms a (k, k) array, so a wide prior factor is bound.

    A guard on `P * k` alone let a `PSDLowRank` prior of width 9000 in four
    dimensions through — 36,000 guarded elements while building a 9000-by-9000
    transform.
    """
    problem = toy.linear_gaussian(u_dim=4, v_dim=3)
    wide = Gaussian(
        jnp.zeros(4),
        PSDLowRank(jnp.asarray(np.random.default_rng(0).normal(size=(4, 9000)))),
    )
    with pytest.raises(ValueError, match="9000-by-9000"):
        dataclasses.replace(problem, prior=wide).posterior()


def test_12_regression_every_problem_field_is_keyword_only():
    """`times` and `y` are both (N,) arrays, so a positional swap is silent.

    The same hazard `pyeki.gauss` makes its sample and factor fields
    keyword-only for — and worse here, since `times` and `y` collide at every
    N rather than only when P == N.
    """
    problem = toy.exponential_decay()
    with pytest.raises(TypeError, match="positional"):
        toy.ExponentialDecay(
            problem.y,  # would be `times`
            problem.times,
            problem.prior,
            problem.noise_cov,
            problem.u_true,
        )
    with pytest.raises(TypeError, match="positional"):
        toy.LinearGaussian(
            problem.prior, problem.noise_cov, problem.y, problem.u_true, problem.times
        )


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_12_regression_a_problem_hashes_and_compares_without_raising(
    name, problem, u_dim, v_dim
):
    """Array fields make a synthesized `__eq__`/`__hash__` raise, not answer.

    Every other value class in the package is `eq=False, repr=False`, and its
    `repr` shows type and static sizes rather than array data.
    """
    assert isinstance(hash(problem), int)
    assert problem == problem
    assert (problem == dataclasses.replace(problem)) is False
    text = repr(problem)
    assert text.startswith(type(problem).__name__)
    assert "Array" not in text and len(text) < 80, text


def test_12_regression_a_field_that_cannot_be_inspected_is_rejected():
    """A Python list passes a shape check and then fails inside the model.

    `jnp.shape` accepts anything shape-like — and is deprecated for
    non-arrays — so validation waved lists through and the model raised an
    error naming a tracer instead of the field.
    """
    problem = toy.exponential_decay()
    for field, value in [
        ("y", [0.0] * 12),
        ("u_true", [2.0, 1.5]),
        ("times", [0.5, 1.0]),
    ]:
        with pytest.raises(TypeError, match="no shape to check"):
            dataclasses.replace(problem, **{field: value})


def test_12_regression_a_vmapped_family_field_is_rejected_at_construction():
    """Otherwise it is diagnosed later, by the operator, not by the problem."""
    problem = toy.linear_gaussian()
    family = jax.vmap(PSDDiagonal)(jnp.ones((3, 8)))
    with pytest.raises(ValueError, match="vmapped family"):
        dataclasses.replace(problem, noise_cov=family)


def test_12_regression_rate_floor_is_type_checked_not_coerced():
    """`float()` on a bool moves the domain boundary; on a str it detonates.

    `True` silently became a floor of 1.0 — failing about 84% of the shipped
    prior rather than 16% — while the validator's message reported the
    coerced value. A string passed validation and raised from inside `vmap`.
    """
    problem = toy.restricted_decay()
    for value in ("0.0", True, jnp.float64(0.0)):
        with pytest.raises(TypeError, match="must be a Python float"):
            dataclasses.replace(problem, rate_floor=value)
    with pytest.raises(ValueError, match="must be finite"):
        dataclasses.replace(problem, rate_floor=float("inf"))
    assert isinstance(problem.rate_floor, float)


def test_12_regression_the_factories_reject_infinite_scales():
    """`not (x > 0)` catches nan and lets inf through, building a nan problem."""
    with pytest.raises(ValueError, match="positive and finite"):
        toy.linear_gaussian(prior_std=float("inf"))
    with pytest.raises(ValueError, match="positive and finite"):
        toy.linear_gaussian(noise_std=float("inf"))
    with pytest.raises(ValueError, match="positive and finite"):
        toy.exponential_decay(t_max=float("inf"))
    with pytest.raises(TypeError, match="must be an int"):
        toy.linear_gaussian(u_dim=True)


def test_12_regression_no_layer_imports_the_toy_module():
    """The architectural rule `CLAUDE.md` calls permanent, in one line.

    `pyeki.toy` depends on two layers, so an import in the other direction
    would make toy problems load-bearing for the library. Checked in a fresh
    interpreter, since this one has already imported the module.
    """
    program = (
        "import sys; import pyeki, pyeki.linalg, pyeki.gauss, pyeki.eki, "
        "pyeki.eki.testing, pyeki.linalg.testing; "
        "print('pyeki.toy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", result.stdout


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_12_regression_a_problem_is_not_a_pytree(name, problem, u_dim, v_dim):
    """Deliberately a plain frozen dataclass, so `jit` sees it as one leaf.

    Registering one would change what crosses a trace boundary, silently, and
    would import every question the operator layer had to answer about
    vmapped families and constructor-bypassing unflatten.
    """
    leaves = jax.tree.leaves(problem)
    assert len(leaves) == 1 and leaves[0] is problem, (
        "a problem became a pytree; jit and vmap semantics change silently"
    )


@pytest.mark.parametrize("name, problem, u_dim, v_dim", PROBLEMS, ids=IDS)
def test_12_regression_forward_refuses_anything_but_one_ensemble(
    name, problem, u_dim, v_dim
):
    """A single parameter vector returned a plausible ``(N,)``, silently.

    The generalized-ufunc convention carried any leading rank through, so
    ``problem.forward(problem.u_true)`` answered — and passing one member
    instead of the ensemble is the mistake the forward-model guide calls the
    most common one. The two decay models raised an ``IndexError`` from
    inside JAX, naming neither the ensemble nor the model. All three now
    agree, and ``vmap`` over the method still works.
    """
    with pytest.raises(ValueError, match=f"expected a .J, {u_dim}. ensemble"):
        problem.forward(problem.u_true)
    with pytest.raises(ValueError, match="never with a further leading axis"):
        problem.forward(jnp.zeros((2, 3, u_dim)))
    with pytest.raises(ValueError, match="expected a"):
        problem.forward(jnp.zeros((3, u_dim + 1)))
    assert jax.vmap(problem.forward)(jnp.zeros((2, 3, u_dim))).shape == (2, 3, v_dim)
