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
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
from pyeki.linalg import DensePSD, PSDLowRank

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
    assert jnp.shape(problem.truth) == (u_dim,)
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
    # checks a stochastic model cannot satisfy -- and nothing else.
    check_forward_model(stochastic, u_dim=2, v_dim=4, stochastic=True)
    with pytest.raises(AssertionError, match="returned shape"):
        check_forward_model(per_member, u_dim=2, v_dim=4, stochastic=True)


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


def _dense_posterior(problem, level: float) -> tuple[np.ndarray, np.ndarray]:
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
    cov = np.linalg.inv(prior_precision + level * data_precision)
    mean = cov @ (
        prior_precision @ np.asarray(problem.prior.mean)
        + level * G.T @ np.linalg.solve(R, np.asarray(problem.y))
    )
    return mean, cov


@pytest.mark.parametrize("level", [0.25, 1.0, 2.0])
def test_5_the_closed_form_posterior_matches_a_dense_reference(level):
    """`posterior(level)` is the posterior at noise R / level, exactly.

    The layer's own conditioning is already checked against a dense reference
    by the joint Gaussian contract's obligation 14. What this pins is the toy
    model's composition: that it divides the *noise* by the level, and that it
    divides it at all. The mis-scaling this guards is the layer's signature
    silent bug; the next test measures how far off it would be.
    """
    problem = toy.linear_gaussian()
    posterior = problem.posterior(level)
    mean_ref, cov_ref = _dense_posterior(problem, level)
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


def test_5_the_level_is_load_bearing_and_the_tolerance_would_catch_it():
    """The two levels must be far apart at the tolerance test 5 uses.

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
    assert gap > 0.01, "the two levels are indistinguishable, so test 5 has no teeth"


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
    with pytest.raises(ValueError, match=r"\(5000, 5000\)"):
        big.posterior()
    assert big.forward(_ensemble(3, 5000)).shape == (3, 10)


@pytest.mark.parametrize("level", [0.0, -1.0, float("inf"), float("nan")])
def test_6_a_non_positive_level_raises(level):
    with pytest.raises(ValueError, match="positive and finite"):
        toy.linear_gaussian().posterior(level)


# ===========================================================================
# 3. the properties the documentation depends on
# ===========================================================================


def test_7_the_ladder_beats_a_single_unit_step_on_the_decay_problem():
    """The property tutorials 2 and 3 and notebook 02 are built on.

    Asserted as a separation the two answers must show and an error the
    ladder must beat, rather than as a tolerance chosen to pass: the problem
    was tuned until this held across eight observation seeds, so it should
    fail if a schedule change ever removes it.
    """
    problem = toy.exponential_decay()
    state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
    common = (problem.forward, problem.y, problem.noise_cov)

    one_step = run(state, *common, schedule=FixedSchedule((1.0,)))
    ladder = run(state, *common, schedule=AdaptiveESSSchedule())

    truth = np.asarray(problem.truth)
    one_step_error = np.abs(np.asarray(one_step.mean) - truth).max()
    ladder_error = np.abs(np.asarray(ladder.mean) - truth).max()

    assert ladder.n_completed_steps > 1
    assert np.abs(np.asarray(one_step.mean) - np.asarray(ladder.mean)).max() > 0.1
    assert ladder_error < one_step_error / 3
    assert ladder_error < 0.05


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
    for field in ("times", "y", "truth"):
        assert _identical(getattr(plain, field), getattr(restricted, field))


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
    assert np.abs(np.asarray(result.mean) - np.asarray(problem.truth)).max() < 0.05

    with pytest.raises(EKIError):
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
    assert np.linalg.matrix_rank(displacement) <= state.n_members - 1

    ensemble_sd = float(result.ensemble.std(axis=0, ddof=1).mean())
    exact_sd = float((problem.posterior().cov.diag() ** 0.5).mean())
    assert exact_sd / ensemble_sd > 20.0, (ensemble_sd, exact_sd)
    # The numbers the user guide prints.
    assert ensemble_sd == pytest.approx(0.0140, abs=5e-5)
    assert exact_sd == pytest.approx(0.9900, abs=5e-5)


# ===========================================================================
# 4. construction, reproducibility, and the guide's blocks
# ===========================================================================


def test_10_a_problem_is_reproducible_from_its_seed_and_varies_with_it():
    """A docs build must see the same numbers every time it runs."""
    for factory in (toy.linear_gaussian, toy.exponential_decay, toy.restricted_decay):
        first, again = factory(seed=3), factory(seed=3)
        different = factory(seed=4)
        assert _identical(first.y, again.y)
        assert _identical(first.truth, again.truth)
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
    with pytest.raises(ValueError, match=r"LinearGaussian.y: must be \(8,\)"):
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

    np.testing.assert_allclose(result.mean, [-1.3289, 1.362, 0.6229, 0.1361], atol=5e-5)
    np.testing.assert_allclose(exact.mean, [-1.3303, 1.3749, 0.6281, 0.1409], atol=5e-5)
    np.testing.assert_allclose(
        problem.truth, [-1.4009, 1.4321, 0.6248, 0.2005], atol=5e-5
    )
    assert np.abs(np.asarray(result.mean) - np.asarray(exact.mean)).max() < 0.013
    np.testing.assert_allclose(
        fitted.cov.diag() ** 0.5, [0.0686, 0.1272, 0.1167, 0.0877], atol=5e-5
    )
    np.testing.assert_allclose(
        exact.cov.diag() ** 0.5, [0.0687, 0.1276, 0.1165, 0.0876], atol=5e-5
    )

    # The correlated-noise variant, whose R the page names.
    rng = np.random.default_rng(1)
    M = rng.normal(size=(8, 8))
    R = jnp.asarray(M @ M.T / 8 + 0.01 * np.eye(8))
    correlated = dataclasses.replace(problem, noise_cov=DensePSD.from_matrix(R))
    np.testing.assert_allclose(
        correlated.posterior().mean, [-1.4093, 0.8248, 0.4646, 0.0692], atol=5e-5
    )
