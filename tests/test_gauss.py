"""Conformance and regression tests for the joint Gaussian layer.

The file has two sections. The first works through the thirteen numbered
conformance obligations of the "Joint Gaussian contract"; the second holds one
targeted regression test per class of silent failure the contract names, under
the same do-not-delete rule as the operator layer's — each documents why a rule
of the contract exists, and deleting it as redundant loses that.

Two rules govern the reference throughout:

- **The dense reference is hand-written here.** Plain dense linear algebra over
  means, anomalies and materialized operators, never routed through
  ``gain_weights``, ``sqrt_transform`` or any other code of ``pyeki.gauss``, so
  every comparison is between two genuinely independent paths.
- **Exactness tests compare against closed forms**, at a tolerance of a few
  machine epsilons times the natural scale of the quantity, never a tolerance
  chosen to make the test pass.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.gauss import EnsembleJoint, Gaussian, gain_weights, sqrt_transform
from pyeki.linalg import (
    Dense,
    DensePSD,
    PSDDiagonal,
    PSDLinOp,
    PSDLowRank,
    Triangular,
    UnsupportedOpError,
    debug_checks,
    dense_matvec,
    linop,
    tri_solve,
)
from pyeki.linalg.testing import check_operator

RNG = np.random.default_rng(0)
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# fixtures: problems, and two test-local operators
# ---------------------------------------------------------------------------


def _psd(n: int) -> np.ndarray:
    """A well-conditioned dense PSD matrix, as a NumPy array."""
    M = RNG.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)


def _problem(J: int, P: int, N: int):
    """An ensemble, a noise covariance and an observation, all NumPy."""
    return (
        RNG.normal(size=(J, P)),
        RNG.normal(size=(J, N)),
        _psd(N),
        RNG.normal(size=N),
    )


@linop
class RotatedWhitenPSD(PSDLinOp):
    """A dense PSD operator whose whitener carries an arbitrary rotation.

    ``Q L^-1`` whitens exactly when ``Q`` is orthogonal and ``L L^T`` is the
    matrix, since ``Q L^-1 (L L^T) L^-T Q^T = Q Q^T = I``. The operator
    contract leaves the choice of whitener to the implementation, and no
    shipped operator exercises that freedom — Cholesky uniqueness makes every
    in-package whitener identical — so this class exists to give the layer two
    genuinely different whiteners for the same matrix.
    """

    L: Array
    Q: Array

    @classmethod
    def from_matrix(cls, A, Q) -> RotatedWhitenPSD:
        return cls(jnp.linalg.cholesky(jnp.asarray(A)), jnp.asarray(Q))

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def _to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)

    def _factor(self):
        return Triangular(self.L, lower=True)

    def _whiten(self, x: Array) -> Array:
        return dense_matvec(self.Q, tri_solve(self.L, x, lower=True))

    def _solve(self, b: Array) -> Array:
        return tri_solve(self.L, tri_solve(self.L, b, lower=True), lower=True, trans=1)

    def _logdet(self) -> Array:
        d = jnp.diagonal(self.L, axis1=-2, axis2=-1)
        return 2.0 * jnp.sum(jnp.log(d), axis=-1)

    def _diag(self) -> Array:
        return jnp.sum(self.L * self.L, axis=-1)


@linop
class WhitenOnlyPSD(PSDLinOp):
    """A PSD operator that whitens, but implements no ``factor`` and no ``logdet``.

    ``PSDLowRank`` is the one shipped ``PSDLinOp`` that disclaims operations,
    and it covers the ``whiten`` and ``logdet`` cases. It cannot cover the
    ``factor`` case, nor the ``logdet`` case *reached through* a working
    ``whiten``, which is what this class supplies.
    """

    L: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def _to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)

    def _whiten(self, x: Array) -> Array:
        return tri_solve(self.L, x, lower=True)


@linop
class CountingWhitenPSD(PSDLinOp):
    """A whitening operator that records how many vectors it has whitened.

    The count is a plain Python list on the instance rather than a field, so
    it stays invisible to the pytree machinery. Eager use only.
    """

    L: Array

    @property
    def shape(self) -> tuple[int, int]:
        n = self.L.shape[-1]
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape[:-2])

    def _matvec(self, x: Array) -> Array:
        return dense_matvec(self.L, dense_matvec(self.L.swapaxes(-1, -2), x))

    def _to_dense(self) -> Array:
        return self.L @ self.L.swapaxes(-1, -2)

    def _whiten(self, x: Array) -> Array:
        object.__getattribute__(self, "log").append(
            1 if x.ndim == 1 else int(np.prod(x.shape[:-1]))
        )
        return tri_solve(self.L, x, lower=True)

    @classmethod
    def counting(cls, A) -> CountingWhitenPSD:
        op = cls(jnp.linalg.cholesky(jnp.asarray(A)))
        object.__setattr__(op, "log", [])
        return op


def test_local_operators_satisfy_the_operator_contract():
    """The two test-local operators are contract-valid, so the layer's failures
    cannot be blamed on the fixtures."""
    R = _psd(4)
    Q, _ = np.linalg.qr(RNG.normal(size=(4, 4)))
    rotated = RotatedWhitenPSD.from_matrix(R, Q)
    check_operator(rotated)
    check_operator(WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(4)))))
    assert rotated.capabilities() == frozenset(
        {"solve", "solve_mat", "logdet", "diag", "factor", "whiten", "whiten_mat"}
    )
    assert WhitenOnlyPSD(
        jnp.linalg.cholesky(jnp.asarray(_psd(4)))
    ).capabilities() == frozenset({"whiten", "whiten_mat"})


# ---------------------------------------------------------------------------
# the hand-written dense reference
#
# Plain dense linear algebra over means, anomalies and materialized
# operators. Nothing below calls into pyeki.gauss.
# ---------------------------------------------------------------------------


def _dense_moments(U: np.ndarray, V: np.ndarray):
    """Empirical means and covariances of a row-wise ensemble, divisor J - 1."""
    J = U.shape[0]
    u_mean, v_mean = U.mean(axis=0), V.mean(axis=0)
    Au, Av = U - u_mean, V - v_mean
    return {
        "u_mean": u_mean,
        "v_mean": v_mean,
        "Au": Au,
        "Av": Av,
        "Cuu": Au.T @ Au / (J - 1),
        "Cuv": Au.T @ Av / (J - 1),
        "Cvv": Av.T @ Av / (J - 1),
    }


def _dense_gain(U: np.ndarray, V: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The Kalman gain K = C_uv (C_vv + R)^-1, formed densely."""
    m = _dense_moments(U, V)
    return m["Cuv"] @ np.linalg.inv(m["Cvv"] + R)


def _dense_posterior(U: np.ndarray, V: np.ndarray, R: np.ndarray, y: np.ndarray):
    """Posterior moments of the Gaussian fitted to the ensemble."""
    m = _dense_moments(U, V)
    K = _dense_gain(U, V, R)
    return (
        m["u_mean"] + K @ (y - m["v_mean"]),
        m["Cuu"] - K @ m["Cuv"].T,
    )


def _dense_conditional(mu_u, mu_v, Cuu, Cuv, Cvv, R, y):
    """Closed-form Gaussian conditioning for an analytic joint."""
    K = Cuv @ np.linalg.inv(Cvv + R)
    return mu_u + K @ (y - mu_v), Cuu - K @ Cuv.T


def _recovered_whitener(noise_cov, N: int) -> np.ndarray:
    """The whitener W a noise operator applies, as a dense matrix.

    The row-wise batch contract makes ``whiten(I)`` the batch of ``W e_i``,
    that is ``W`` transposed.
    """
    return np.asarray(noise_cov.whiten(jnp.eye(N))).T


def _sample_moments(members: np.ndarray):
    """Sample mean and covariance (divisor J - 1) of a row-wise ensemble."""
    J = members.shape[0]
    A = members - members.mean(axis=0)
    return members.mean(axis=0), A.T @ A / (J - 1)


def _scaled_whitened_anomalies(V: np.ndarray, noise_cov, J: int) -> np.ndarray:
    """S = A_v W^T / sqrt(J - 1), formed densely from the recovered whitener."""
    Av = V - V.mean(axis=0)
    W = _recovered_whitener(noise_cov, V.shape[1])
    return Av @ W.T / np.sqrt(J - 1)


def _s_with_spectrum(J: int, N: int, sigmas, seed: int = 11) -> Array:
    """An ``(J, N)`` matrix with exactly the prescribed singular values."""
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.normal(size=(J, J)))
    Q2, _ = np.linalg.qr(rng.normal(size=(N, N)))
    D = np.zeros((J, N))
    for i, value in enumerate(sigmas):
        D[i, i] = value
    return jnp.asarray(Q1 @ D @ Q2.T)


def _exact_moment_ensemble(J: int, mu: np.ndarray, G: np.ndarray) -> np.ndarray:
    """An ensemble whose empirical moments equal ``mu`` and ``G @ G.T`` exactly.

    Take the complete QR of the all-ones vector in R^J, let E be its last k
    columns — orthonormal and each orthogonal to the ones vector — and set the
    members to ``mu + sqrt(J - 1) E G^T``. Then the member rows have mean
    ``mu`` (because ``1^T E = 0``) and empirical covariance ``G E^T E G^T``,
    which is ``G G^T`` (because ``E^T E = I_k``). Only ``J >= k + 1`` binds;
    at ``J = k`` the construction silently fails.
    """
    k = G.shape[1]
    assert J >= k + 1, "the construction needs J >= k + 1"
    Q, _ = np.linalg.qr(np.ones((J, 1)), mode="complete")
    E = Q[:, 1 : k + 1]
    return mu + np.sqrt(J - 1) * E @ G.T


def _count_svd(jaxpr) -> int:
    """Count ``svd`` primitives in a jaxpr, recursing into sub-jaxprs."""
    total = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "svd":
            total += 1
        for param in eqn.params.values():
            candidates = param if isinstance(param, (tuple, list)) else [param]
            for candidate in candidates:
                inner = getattr(candidate, "jaxpr", candidate)
                if hasattr(inner, "eqns"):
                    total += _count_svd(inner)
    return total


# ===========================================================================
# Section 1 -- the thirteen conformance obligations
# ===========================================================================

# --- 1. gain against dense ---------------------------------------------------

SHAPE_REGIMES = [
    pytest.param(6, 4, 9, id="N>J"),
    pytest.param(6, 4, 6, id="N=J"),
    pytest.param(8, 4, 5, id="N<J"),
]


@pytest.mark.parametrize(("J", "P", "N"), SHAPE_REGIMES)
@pytest.mark.parametrize("batch_rank", [0, 1, 2])
def test_1_gain_weights_reproduces_the_dense_gain(J, P, N, batch_rank):
    """gain_weights composed with whiten reproduces K r elementwise, in all
    three shape regimes and at batch ranks 0, 1 and 2."""
    U, V, R, _ = _problem(J, P, N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    residual_shape = (2, 3, N)[2 - batch_rank :]
    r = RNG.normal(size=residual_shape)

    s = _scaled_whitened_anomalies(V, noise_cov, J)
    w = gain_weights(jnp.asarray(s), noise_cov.whiten(jnp.asarray(r)))
    Au = U - U.mean(axis=0)
    got = np.einsum("pj,...j->...p", Au.T, np.asarray(w)) / np.sqrt(J - 1)

    want = np.einsum("pn,...n->...p", _dense_gain(U, V, R), r)
    assert got.shape == residual_shape[:-1] + (P,)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e3 * EPS * np.abs(want).max())


# --- 2. whitener invariance --------------------------------------------------


def test_2_weights_are_invariant_to_the_choice_of_whitener():
    """Two operators representing the same R with different whiteners give the
    same weights, and the same updates."""
    J, P, N = 7, 4, 5
    U, V, R, y = _problem(J, P, N)
    Q, _ = np.linalg.qr(RNG.normal(size=(N, N)))
    plain = DensePSD.from_matrix(jnp.asarray(R))
    rotated = RotatedWhitenPSD.from_matrix(R, Q)

    # the two whiteners really do differ, while both whiten R
    W_plain, W_rot = _recovered_whitener(plain, N), _recovered_whitener(rotated, N)
    assert np.abs(W_plain - W_rot).max() > 0.1
    for W in (W_plain, W_rot):
        np.testing.assert_allclose(W @ R @ W.T, np.eye(N), rtol=0, atol=1e3 * EPS)

    r = jnp.asarray(y)
    weights = [
        gain_weights(
            jnp.asarray(_scaled_whitened_anomalies(V, cov, J)), cov.whiten(r)
        )
        for cov in (plain, rotated)
    ]
    np.testing.assert_allclose(weights[0], weights[1], rtol=0, atol=1e4 * EPS)

    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    np.testing.assert_allclose(
        joint.transform_update(jnp.asarray(y), plain),
        joint.transform_update(jnp.asarray(y), rotated),
        rtol=0,
        atol=1e4 * EPS,
    )
    np.testing.assert_allclose(
        joint.condition(jnp.asarray(y), plain).mean,
        joint.condition(jnp.asarray(y), rotated).mean,
        rtol=0,
        atol=1e4 * EPS,
    )


# --- 3. transform against dense ---------------------------------------------


@pytest.mark.parametrize(
    ("J", "N"), [(6, 9), (6, 6), (7, 3)], ids=["N>J", "N=J", "N<J"]
)
def test_3_sqrt_transform_matches_a_dense_eigh_reference(J, N):
    """At moderate scale, where forming I + s s^T is trustworthy, the transform
    matches an eigh-based (I + s s^T)^-1/2 — including rho < J, the case the
    naive thin-SVD formula gets wrong."""
    rho = min(J, N)
    s = _s_with_spectrum(J, N, np.geomspace(1e2, 1e-2, rho))
    T = np.asarray(sqrt_transform(s))

    gram = np.eye(J) + np.asarray(s) @ np.asarray(s).T
    w, Vec = np.linalg.eigh(gram)
    reference = (Vec * w**-0.5) @ Vec.T

    scale = max(1.0, float(np.linalg.svd(np.asarray(s), compute_uv=False)[0]))
    np.testing.assert_allclose(T, reference, rtol=0, atol=1e3 * EPS * scale)


@pytest.mark.parametrize(
    ("J", "N"), [(5, 7), (6, 3)], ids=["N>J", "N<J"]
)
@pytest.mark.parametrize("sigma_max", [1e4, 1e8, 1e10])
def test_3_sqrt_transform_satisfies_the_stably_formed_invariant(J, N, sigma_max):
    """At large sigma_max the dense reference is the inaccurate side, so the
    transform is checked against T T^T + (T s)(T s)^T = I instead.

    The spectrum is deliberately *mixed*. With every singular value at one
    large scale, T T^T is itself of order sigma_max^-2 and the identity's loss
    from forming I + s s^T is invisible: both this invariant and the
    algebraically equivalent T (I + s s^T) T^T pass. A spectrum spanning
    decades is what separates them, and the second test below pins that.
    """
    rho = min(J, N)
    sigmas = np.geomspace(sigma_max, 1.0, rho)
    s = _s_with_spectrum(J, N, sigmas)
    T = np.asarray(sqrt_transform(s))
    Ts = T @ np.asarray(s)

    residual = np.abs(T @ T.T + Ts @ Ts.T - np.eye(J)).max()
    assert residual <= 128 * EPS * max(1.0, sigma_max)


def test_3_sqrt_transform_is_symmetric_and_preserves_mean_centring():
    """T = T^T for every s; T 1 = 1 for mean-centred s only."""
    J, N = 6, 9
    for scale in (1.0, 1e3, 1e8):
        A = RNG.normal(size=(J, N))
        s = jnp.asarray((A - A.mean(axis=0)) * scale)
        T = np.asarray(sqrt_transform(s))
        sigma_max = float(np.linalg.svd(np.asarray(s), compute_uv=False)[0])

        np.testing.assert_allclose(T, T.T, rtol=0, atol=1e3 * EPS)

        # The modifier-induced mean shift is O((eps sigma_max)^2); the computed
        # matrix-vector product carries ordinary O(eps) round-off on top, which
        # dominates until sigma_max reaches 1/sqrt(eps).
        shift = np.abs(T @ np.ones(J) - 1.0).max()
        assert shift <= 64 * EPS + (EPS * sigma_max) ** 2

    # on general (uncentred) s no such identity holds
    general = jnp.asarray(RNG.normal(size=(J, N)) + 5.0)
    assert np.abs(np.asarray(sqrt_transform(general)) @ np.ones(J) - 1.0).max() > 0.1


# --- 4. moment exactness of the posterior ------------------------------------


@pytest.mark.parametrize(("J", "P", "N"), SHAPE_REGIMES)
def test_4_posterior_moments_match_the_dense_posterior(J, P, N):
    """transform_update's sample moments, and condition's mean and densified
    covariance, all equal the hand-written dense posterior; and the two methods
    satisfy the elementwise identity u_j' = m_post + sqrt(J-1) F_j."""
    U, V, R, y = _problem(J, P, N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    m_post, C_post = _dense_posterior(U, V, R, y)
    scale = max(np.abs(m_post).max(), np.abs(C_post).max())

    members = np.asarray(joint.transform_update(jnp.asarray(y), noise_cov))
    sample_mean, sample_cov = _sample_moments(members)
    np.testing.assert_allclose(sample_mean, m_post, rtol=0, atol=1e3 * EPS * scale)
    np.testing.assert_allclose(sample_cov, C_post, rtol=0, atol=1e3 * EPS * scale)

    posterior = joint.condition(jnp.asarray(y), noise_cov)
    assert isinstance(posterior, Gaussian)
    assert isinstance(posterior.cov, PSDLowRank)
    assert posterior.cov.F.shape == (P, J)
    np.testing.assert_allclose(posterior.mean, m_post, rtol=0, atol=1e3 * EPS * scale)
    np.testing.assert_allclose(
        posterior.cov.to_dense(), C_post, rtol=0, atol=1e3 * EPS * scale
    )

    F = np.asarray(posterior.cov.F)
    np.testing.assert_allclose(
        members,
        np.asarray(posterior.mean) + np.sqrt(J - 1) * F.T,
        rtol=0,
        atol=1e3 * EPS * scale,
    )


def test_4_posterior_covariance_rank_is_bounded_by_the_ensemble():
    """rank(C_post) <= J - 1, so the posterior is singular in the usual regime."""
    J, P, N = 5, 8, 6
    U, V, R, y = _problem(J, P, N)
    posterior = EnsembleJoint(jnp.asarray(U), jnp.asarray(V)).condition(
        jnp.asarray(y), DensePSD.from_matrix(jnp.asarray(R))
    )
    singular = np.linalg.svd(np.asarray(posterior.cov.to_dense()), compute_uv=False)
    assert np.sum(singular > 1e-10 * singular[0]) <= J - 1


# --- 5. pathwise against dense, elementwise ---------------------------------


@pytest.mark.parametrize(("J", "P", "N"), SHAPE_REGIMES)
def test_5_pathwise_update_matches_the_dense_perturbed_observation_update(J, P, N):
    """With eps recomputed from the same key, pathwise_update equals
    u_j + K(y - v_j - W^-1 eps_j) computed densely."""
    U, V, R, y = _problem(J, P, N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    key = jax.random.key(4)

    got = np.asarray(
        EnsembleJoint(jnp.asarray(U), jnp.asarray(V)).pathwise_update(
            key, jnp.asarray(y), noise_cov
        )
    )

    eps = np.asarray(jax.random.normal(key, (J, N)))
    W = _recovered_whitener(noise_cov, N)
    eta = np.linalg.solve(W, eps.T).T  # W^-1 eps_j, row-wise
    K = _dense_gain(U, V, R)
    want = U + (K @ (y - V - eta).T).T

    np.testing.assert_allclose(
        got, want, rtol=0, atol=1e3 * EPS * max(1.0, np.abs(want).max())
    )


# --- 6. exact-moment fixtures ------------------------------------------------


def test_6_exact_moment_ensemble_reaches_the_analytic_posterior():
    """On an ensemble whose empirical moments equal an analytic linear-Gaussian
    joint, condition and transform_update reproduce that joint's closed-form
    posterior moments."""
    P, N, k = 3, 4, 5
    J = k + 1
    G = RNG.normal(size=(P + N, k))
    mu = RNG.normal(size=P + N)
    R = _psd(N)
    y = RNG.normal(size=N)

    members = _exact_moment_ensemble(J, mu, G)
    U, V = members[:, :P], members[:, P:]

    # the fixture is exact: check it before trusting what it implies
    C = G @ G.T
    moments = _dense_moments(U, V)
    np.testing.assert_allclose(
        np.concatenate([moments["u_mean"], moments["v_mean"]]),
        mu,
        rtol=0,
        atol=1e3 * EPS * np.abs(mu).max(),
    )
    for block, want in (
        ("Cuu", C[:P, :P]),
        ("Cuv", C[:P, P:]),
        ("Cvv", C[P:, P:]),
    ):
        np.testing.assert_allclose(
            moments[block], want, rtol=0, atol=1e3 * EPS * np.abs(C).max()
        )

    m_post, C_post = _dense_conditional(
        mu[:P], mu[P:], C[:P, :P], C[:P, P:], C[P:, P:], R, y
    )
    scale = max(np.abs(m_post).max(), np.abs(C_post).max())

    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))

    posterior = joint.condition(jnp.asarray(y), noise_cov)
    np.testing.assert_allclose(posterior.mean, m_post, rtol=0, atol=1e4 * EPS * scale)
    np.testing.assert_allclose(
        posterior.cov.to_dense(), C_post, rtol=0, atol=1e4 * EPS * scale
    )

    sample_mean, sample_cov = _sample_moments(
        np.asarray(joint.transform_update(jnp.asarray(y), noise_cov))
    )
    np.testing.assert_allclose(sample_mean, m_post, rtol=0, atol=1e4 * EPS * scale)
    np.testing.assert_allclose(sample_cov, C_post, rtol=0, atol=1e4 * EPS * scale)


# --- 7. marginal formulas ----------------------------------------------------


def test_7_sample_matches_its_pinned_elementwise_definition():
    """sample is exactly mean + L.matvec(normal(key, (n_samples, k)))."""
    n = 4
    mean = jnp.asarray(RNG.normal(size=n))
    cov = DensePSD.from_matrix(jnp.asarray(_psd(n)))
    key = jax.random.key(7)

    got = Gaussian(mean, cov).sample(key, 5)

    L = cov.factor()
    want = mean + L.matvec(jax.random.normal(key, (5, L.shape[1])))
    assert got.shape == (5, n)
    np.testing.assert_array_equal(got, want)


def test_7_sample_from_a_wide_factor_uses_the_factor_width():
    """The draw's width is the factor's k, not n."""
    F = jnp.asarray(RNG.normal(size=(3, 6)))
    got = Gaussian(jnp.zeros(3), PSDLowRank(F)).sample(jax.random.key(8), 2)
    want = Dense(F).matvec(jax.random.normal(jax.random.key(8), (2, 6)))
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("batch_rank", [0, 1, 2])
def test_7_log_density_matches_the_dense_closed_form(batch_rank):
    """log_density equals -1/2 (n log 2pi + logdet C + (x-m)^T C^-1 (x-m))."""
    n = 4
    C = _psd(n)
    mean = RNG.normal(size=n)
    gaussian = Gaussian(jnp.asarray(mean), DensePSD.from_matrix(jnp.asarray(C)))

    shape = (2, 3, n)[2 - batch_rank :]
    x = RNG.normal(size=shape)

    got = gaussian.log_density(jnp.asarray(x))

    d = x - mean
    quadratic = np.einsum("...i,...i->...", d, np.linalg.solve(C, d[..., None])[..., 0])
    want = -0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(C)[1] + quadratic)

    assert got.shape == shape[:-1]
    assert isinstance(got, jax.Array)
    np.testing.assert_allclose(
        got, want, rtol=0, atol=1e3 * EPS * max(1.0, np.abs(want).max())
    )


def test_7_log_density_differentiates():
    """Differentiable with respect to x and to the array leaves, which
    hyperparameter estimation needs."""
    n = 3
    gaussian = Gaussian(
        jnp.asarray(RNG.normal(size=n)), DensePSD.from_matrix(jnp.asarray(_psd(n)))
    )
    x = jnp.asarray(RNG.normal(size=n))

    grad_x = jax.grad(gaussian.log_density)(x)
    assert grad_x.shape == (n,)
    assert bool(jnp.all(jnp.isfinite(grad_x)))

    grads = jax.grad(lambda g: g.log_density(x))(gaussian)
    assert bool(jnp.all(jnp.isfinite(grads.mean)))
    assert bool(jnp.all(jnp.isfinite(grads.cov.L)))


# --- 8. degeneracy -----------------------------------------------------------


@pytest.mark.parametrize(("J", "P", "N"), [(6, 4, 5), (2, 3, 1), (4, 2, 7)])
def test_8_zero_prediction_anomalies_make_both_updates_the_identity(J, P, N):
    """A collapsed ensemble is a no-op, not nan.

    Every row of v_samples takes the *same, exactly representable* value: a
    collapsed ensemble of arbitrary values leaves anomalies at O(eps) rather
    than bit-zero, and would not exercise the exactly-zero path.
    """
    U = RNG.normal(size=(J, P))
    V = np.tile(np.full(N, 0.5), (J, 1))  # 0.5 is exact in binary
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(N)))
    y = jnp.asarray(RNG.normal(size=N))
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))

    pathwise = joint.pathwise_update(jax.random.key(0), y, noise_cov)
    np.testing.assert_array_equal(pathwise, jnp.asarray(U))

    transform = joint.transform_update(y, noise_cov)
    # reconstructs u_bar + a_j, so round-off rather than bit-exact
    np.testing.assert_allclose(
        transform, U, rtol=0, atol=1e3 * EPS * max(1.0, np.abs(U).max())
    )

    posterior = joint.condition(y, noise_cov)
    moments = _dense_moments(U, V)
    np.testing.assert_allclose(
        posterior.mean, moments["u_mean"], rtol=0, atol=1e3 * EPS
    )
    np.testing.assert_allclose(
        posterior.cov.to_dense(),
        moments["Cuu"],
        rtol=0,
        atol=1e3 * EPS * max(1.0, np.abs(moments["Cuu"]).max()),
    )

    for array in (pathwise, transform, posterior.mean, posterior.cov.to_dense()):
        assert bool(jnp.all(jnp.isfinite(array)))


def test_8_primitives_are_finite_on_an_exactly_collapsed_operand():
    """The kernel's formulas are continuous at sigma = 0 and need no
    special-casing; the value at an exactly zero s is exact."""
    s = jnp.zeros((4, 6))
    np.testing.assert_array_equal(gain_weights(s, jnp.ones(6)), jnp.zeros(4))
    np.testing.assert_array_equal(sqrt_transform(s), jnp.eye(4))


# --- 9. capability propagation -----------------------------------------------


def test_9_noise_covariance_without_whiten_raises_from_the_conditioning_methods():
    """PSDLowRank is the shipped PSDLinOp that disclaims whiten."""
    J, P, N = 5, 3, 4
    joint = EnsembleJoint(
        jnp.asarray(RNG.normal(size=(J, P))), jnp.asarray(RNG.normal(size=(J, N)))
    )
    noise_cov = PSDLowRank(jnp.asarray(RNG.normal(size=(N, N))))
    y = jnp.asarray(RNG.normal(size=N))
    assert not noise_cov.supports("whiten")

    for call in (
        lambda: joint.pathwise_update(jax.random.key(0), y, noise_cov),
        lambda: joint.transform_update(y, noise_cov),
        lambda: joint.condition(y, noise_cov),
    ):
        with pytest.raises(UnsupportedOpError, match="whiten"):
            call()


def test_9_covariance_without_factor_raises_from_sample():
    n = 4
    cov = WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(n))))
    assert not cov.supports("factor")
    with pytest.raises(UnsupportedOpError, match="factor"):
        Gaussian(jnp.zeros(n), cov).sample(jax.random.key(0), 3)


def test_9_covariance_without_logdet_raises_from_log_density():
    """Reached through a working whiten, so it is the logdet check that fires —
    the capability checks run in the order log_density names them."""
    n = 4
    cov = WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(n))))
    assert cov.supports("whiten") and not cov.supports("logdet")
    with pytest.raises(UnsupportedOpError, match="logdet"):
        Gaussian(jnp.zeros(n), cov).log_density(jnp.zeros(n))

    # a covariance lacking whiten fails on whiten first
    with pytest.raises(UnsupportedOpError, match="whiten"):
        Gaussian(jnp.zeros(n), PSDLowRank(jnp.eye(n))).log_density(jnp.zeros(n))


@pytest.mark.parametrize(("J", "P"), [(4, 9), (12, 3)], ids=["J-1<P", "J-1>=P"])
def test_9_posterior_log_density_raises_at_any_ensemble_size(J, P):
    """The posterior's covariance is a PSDLowRank, whose static capability
    choice withholds whiten at every width."""
    N = 5
    U, V, R, y = _problem(J, P, N)
    posterior = EnsembleJoint(jnp.asarray(U), jnp.asarray(V)).condition(
        jnp.asarray(y), DensePSD.from_matrix(jnp.asarray(R))
    )
    assert posterior.cov.supports("factor")
    with pytest.raises(UnsupportedOpError, match="whiten"):
        posterior.log_density(jnp.zeros(P))
    # sampling the posterior does work: the factor is the stored representation
    assert posterior.sample(jax.random.key(0), 3).shape == (3, P)


# --- 10. validation ----------------------------------------------------------


def test_10_gaussian_construction_validation():
    cov = DensePSD.from_matrix(jnp.asarray(_psd(3)))
    with pytest.raises(ValueError, match="rank 1"):
        Gaussian(jnp.zeros((2, 3)), cov)
    with pytest.raises(ValueError, match="rank 1"):
        Gaussian(jnp.asarray(1.0), cov)
    with pytest.raises(TypeError, match="PSDLinOp"):
        Gaussian(jnp.zeros(3), jnp.eye(3))
    with pytest.raises(TypeError, match="PSDLinOp"):
        Gaussian(jnp.zeros(3), Dense(jnp.eye(3)))
    with pytest.raises(ValueError, match="disagrees"):
        Gaussian(jnp.zeros(4), cov)
    with pytest.raises(ValueError, match="core sizes must be positive"):
        Gaussian(jnp.zeros(0), cov)


def test_10_ensemble_joint_construction_validation():
    with pytest.raises(ValueError, match="rank 2"):
        EnsembleJoint(jnp.zeros(6), jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="rank 2"):
        EnsembleJoint(jnp.zeros((3, 2)), jnp.zeros((3, 2, 1)))
    with pytest.raises(ValueError, match="same number of members"):
        EnsembleJoint(jnp.zeros((3, 2)), jnp.zeros((4, 2)))
    with pytest.raises(ValueError, match="at least 2 members"):
        EnsembleJoint(jnp.zeros((1, 2)), jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match="core sizes must be positive"):
        EnsembleJoint(jnp.zeros((3, 0)), jnp.zeros((3, 2)))


def test_10_conditioning_call_validation():
    J, P, N = 5, 3, 4
    joint = EnsembleJoint(
        jnp.asarray(RNG.normal(size=(J, P))), jnp.asarray(RNG.normal(size=(J, N)))
    )
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(N)))
    y = jnp.asarray(RNG.normal(size=N))

    def calls(y_arg, cov_arg):
        return (
            lambda: joint.pathwise_update(jax.random.key(0), y_arg, cov_arg),
            lambda: joint.transform_update(y_arg, cov_arg),
            lambda: joint.condition(y_arg, cov_arg),
        )

    for bad_y in (jnp.zeros(N + 1), jnp.zeros((1, N)), jnp.asarray(0.0)):
        for call in calls(bad_y, noise_cov):
            with pytest.raises(ValueError, match="expected y of shape"):
                call()

    for call in calls(y, jnp.eye(N)):
        with pytest.raises(TypeError, match="PSDLinOp"):
            call()

    wrong_side = DensePSD.from_matrix(jnp.asarray(_psd(N + 1)))
    for call in calls(y, wrong_side):
        with pytest.raises(ValueError, match="side 4"):
            call()


def test_10_sample_and_log_density_call_validation():
    n = 3
    gaussian = Gaussian(jnp.zeros(n), DensePSD.from_matrix(jnp.asarray(_psd(n))))

    for bad in (3.0, True, np.int64(3), "3"):
        with pytest.raises(TypeError, match="Python int"):
            gaussian.sample(jax.random.key(0), bad)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            gaussian.sample(jax.random.key(0), bad)

    for bad_x in (jnp.asarray(0.0), jnp.zeros(n + 1), jnp.zeros((2, n - 1))):
        with pytest.raises(ValueError, match="expected x of core shape"):
            gaussian.log_density(bad_x)


def test_10_primitive_operand_validation():
    s = jnp.asarray(RNG.normal(size=(4, 6)))
    for bad_s in (jnp.zeros(6), jnp.zeros((2, 4, 6)), jnp.zeros((0, 6))):
        with pytest.raises(ValueError, match="expected s of shape"):
            gain_weights(bad_s, jnp.zeros(6))
        with pytest.raises(ValueError, match="expected s of shape"):
            sqrt_transform(bad_s)
    for bad_b in (jnp.asarray(0.0), jnp.zeros(5), jnp.zeros((3, 7))):
        with pytest.raises(ValueError, match="expected b of core shape"):
            gain_weights(s, bad_b)


def test_10_value_checks_are_debug_only():
    """Tier 4 runs at construction and at call, and only in debug mode."""
    n = 3
    cov = DensePSD.from_matrix(jnp.asarray(_psd(n)))
    joint_args = (jnp.zeros((3, 2)), jnp.asarray([[0.0], [jnp.nan], [0.0]]))

    # off by default: no exception, just a nan-bearing object
    Gaussian(jnp.asarray([0.0, jnp.nan, 0.0]), cov)
    EnsembleJoint(*joint_args)
    gain_weights(jnp.asarray([[jnp.nan, 0.0]]), jnp.zeros(2))

    with debug_checks(True):
        with pytest.raises(ValueError, match="mean must be finite"):
            Gaussian(jnp.asarray([0.0, jnp.nan, 0.0]), cov)
        with pytest.raises(ValueError, match="v_samples must be finite"):
            EnsembleJoint(*joint_args)
        with pytest.raises(ValueError, match="s must be finite"):
            gain_weights(jnp.asarray([[jnp.nan, 0.0]]), jnp.zeros(2))
        with pytest.raises(ValueError, match="b must be finite"):
            gain_weights(jnp.asarray([[1.0, 0.0]]), jnp.asarray([jnp.nan, 0.0]))
        with pytest.raises(ValueError, match="s must be finite"):
            sqrt_transform(jnp.asarray([[jnp.nan, 0.0]]))

        good = EnsembleJoint(
            jnp.asarray(RNG.normal(size=(3, 2))), jnp.asarray(RNG.normal(size=(3, 2)))
        )
        noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(2)))
        with pytest.raises(ValueError, match="y must be finite"):
            good.transform_update(jnp.asarray([jnp.nan, 0.0]), noise_cov)


# --- 11. JAX round trips -----------------------------------------------------


def _reference_problem():
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    return (
        EnsembleJoint(jnp.asarray(U), jnp.asarray(V)),
        DensePSD.from_matrix(jnp.asarray(R)),
        jnp.asarray(y),
    )


@pytest.mark.parametrize("factory", ["gaussian", "joint"])
def test_11_flatten_unflatten_preserves_type_and_behaviour(factory):
    if factory == "gaussian":
        obj = Gaussian(
            jnp.asarray(RNG.normal(size=3)),
            DensePSD.from_matrix(jnp.asarray(_psd(3))),
        )
        probe = lambda o: o.log_density(jnp.zeros(3))  # noqa: E731
    else:
        obj, noise_cov, y = _reference_problem()
        probe = lambda o: o.transform_update(y, noise_cov)  # noqa: E731

    leaves, treedef = jax.tree_util.tree_flatten(obj)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(obj)
    assert rebuilt.batch_shape == ()
    np.testing.assert_array_equal(probe(rebuilt), probe(obj))

    # JAX internals occasionally unflatten with bare sentinels
    sentinels = jax.tree_util.tree_unflatten(treedef, [object() for _ in leaves])
    assert type(sentinels) is type(obj)


def test_11_conditioning_methods_run_under_jit():
    joint, noise_cov, y = _reference_problem()
    key = jax.random.key(2)
    for eager, jitted in (
        (
            joint.pathwise_update(key, y, noise_cov),
            jax.jit(lambda j, y: j.pathwise_update(key, y, noise_cov))(joint, y),
        ),
        (
            joint.transform_update(y, noise_cov),
            jax.jit(lambda j, y: j.transform_update(y, noise_cov))(joint, y),
        ),
        (
            joint.condition(y, noise_cov).mean,
            jax.jit(lambda j, y: j.condition(y, noise_cov).mean)(joint, y),
        ),
    ):
        np.testing.assert_allclose(eager, jitted, rtol=0, atol=1e3 * EPS)

    logp = Gaussian(jnp.zeros(3), DensePSD.from_matrix(jnp.asarray(_psd(3))))
    np.testing.assert_allclose(
        jax.jit(logp.log_density)(jnp.ones(3)), logp.log_density(jnp.ones(3))
    )


def test_11_constructing_a_joint_inside_vmap_round_trips():
    """The vmap exit boundary rebuilds a family, bypassing the constructor."""
    U = jnp.asarray(RNG.normal(size=(5, 6, 3)))
    V = jnp.asarray(RNG.normal(size=(5, 6, 4)))
    family = jax.vmap(EnsembleJoint)(U, V)
    assert type(family) is EnsembleJoint
    assert family.batch_shape == (5,)
    assert (family.n_members, family.u_dim, family.v_dim) == (6, 3, 4)


def test_11_a_vmapped_family_agrees_with_a_python_loop():
    members, J, P, N = 5, 6, 3, 4
    U = RNG.normal(size=(members, J, P))
    V = RNG.normal(size=(members, J, N))
    ys = RNG.normal(size=(members, N))
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(N)))

    def update(u, v, y):
        return EnsembleJoint(u, v).transform_update(y, noise_cov)

    batched = jax.vmap(update)(jnp.asarray(U), jnp.asarray(V), jnp.asarray(ys))
    looped = jnp.stack(
        [
            update(jnp.asarray(U[i]), jnp.asarray(V[i]), jnp.asarray(ys[i]))
            for i in range(members)
        ]
    )
    np.testing.assert_allclose(batched, looped, rtol=0, atol=1e3 * EPS)


def test_11_vmap_over_the_noise_operator():
    """A family of noise operators is applied with vmap, not passed directly."""
    joint, _, y = _reference_problem()
    N = joint.v_dim
    matrices = jnp.stack([jnp.asarray(_psd(N)) for _ in range(3)])
    covs = jax.vmap(DensePSD.from_matrix)(matrices)
    assert covs.batch_shape == (3,)

    batched = jax.vmap(lambda cov: joint.transform_update(y, cov))(covs)
    looped = jnp.stack(
        [
            joint.transform_update(y, DensePSD.from_matrix(matrices[i]))
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(batched, looped, rtol=0, atol=1e3 * EPS)


# --- 12. reproducibility and repr -------------------------------------------


def test_12_stochastic_calls_are_reproducible():
    joint, noise_cov, y = _reference_problem()
    key = jax.random.key(3)
    np.testing.assert_array_equal(
        joint.pathwise_update(key, y, noise_cov),
        joint.pathwise_update(key, y, noise_cov),
    )
    assert not np.allclose(
        joint.pathwise_update(key, y, noise_cov),
        joint.pathwise_update(jax.random.key(4), y, noise_cov),
    )

    gaussian = Gaussian(jnp.zeros(3), DensePSD.from_matrix(jnp.asarray(_psd(3))))
    np.testing.assert_array_equal(
        gaussian.sample(key, 4), gaussian.sample(key, 4)
    )
    assert not np.allclose(
        gaussian.sample(key, 4), gaussian.sample(jax.random.key(5), 4)
    )


def test_12_repr_names_static_sizes_and_no_array_data():
    gaussian = Gaussian(jnp.zeros(12), DensePSD.from_matrix(jnp.asarray(_psd(12))))
    assert repr(gaussian) == "Gaussian(n=12)"

    joint = EnsembleJoint(jnp.zeros((100, 12)), jnp.zeros((100, 40)))
    assert repr(joint) == "EnsembleJoint(n_members=100, u_dim=12, v_dim=40)"

    for text in (repr(gaussian), repr(joint)):
        assert "0." not in text and "Array" not in text


def test_12_repr_never_raises_on_unreadable_leaves():
    treedef = jax.tree_util.tree_structure(
        EnsembleJoint(jnp.zeros((2, 2)), jnp.zeros((2, 2)))
    )
    broken = jax.tree_util.tree_unflatten(treedef, [object(), object()])
    assert repr(broken) == "<EnsembleJoint (unprintable leaves)>"


def test_12_the_pinned_prng_draws_are_snapshotted():
    """Both draws are pinned by the contract, and the bit stream belongs to
    JAX. Snapshotting them makes a JAX-side stream change visible here rather
    than silently absorbed into every downstream result.

    Captured under JAX 0.10.2 with x64 enabled and default PRNG settings. A
    failure here is not necessarily a pyEKI bug — check JAX's version and the
    jax_threefry_partitionable / x64 flags first — but it does mean every
    stochastic output of this layer changed.
    """
    key = jax.random.key(0)
    # Gaussian.sample draws normal(key, (n_samples, k))
    np.testing.assert_allclose(
        jax.random.normal(key, (2, 3)),
        np.array(
            [
                [-0.2058421394796434, -0.7847657764467411, 1.8160866726679836],
                [0.1878440128937887, 0.08086788103743275, -0.3721107933346203],
            ]
        ),
        rtol=0,
        atol=0,
    )
    # pathwise_update draws normal(key, (J, N))
    np.testing.assert_allclose(
        jax.random.normal(key, (3, 2)),
        np.array(
            [
                [-0.2058421394796434, -0.7847657764467411],
                [1.8160866726679836, 0.1878440128937887],
                [0.08086788103743275, -0.3721107933346203],
            ]
        ),
        rtol=0,
        atol=0,
    )


# --- 13. families ------------------------------------------------------------


def _stacked(obj, reps: int = 8):
    """A vmapped family, built the way a pytree reconstruction builds one."""
    return jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * reps), obj)


def test_13_gaussian_family_is_legible_and_inert():
    family = _stacked(
        Gaussian(jnp.zeros(12), DensePSD.from_matrix(jnp.asarray(_psd(12))))
    )
    assert family.batch_shape == (8,)
    assert family.n == 12  # core sizes, never batch sizes
    assert repr(family) == "vmapped(Gaussian(n=12), batch=(8,))"

    with pytest.raises(ValueError, match="vmapped family"):
        family.sample(jax.random.key(0), 2)
    with pytest.raises(ValueError, match="vmapped family"):
        family.log_density(jnp.zeros(12))


def test_13_ensemble_joint_family_is_legible_and_inert():
    joint, noise_cov, y = _reference_problem()
    family = _stacked(joint)
    assert family.batch_shape == (8,)
    assert (family.n_members, family.u_dim, family.v_dim) == (6, 3, 4)
    assert repr(family) == (
        "vmapped(EnsembleJoint(n_members=6, u_dim=3, v_dim=4), batch=(8,))"
    )

    with pytest.raises(ValueError, match="vmapped family"):
        family.pathwise_update(jax.random.key(0), y, noise_cov)
    with pytest.raises(ValueError, match="vmapped family"):
        family.transform_update(y, noise_cov)
    with pytest.raises(ValueError, match="vmapped family"):
        family.condition(y, noise_cov)
    for name in ("u_mean", "v_mean", "u_anomalies", "v_anomalies"):
        with pytest.raises(ValueError, match="vmapped family"):
            getattr(family, name)


def test_13_genuine_construction_rejects_a_family_covariance():
    cov_family = _stacked(DensePSD.from_matrix(jnp.asarray(_psd(3))), reps=4)
    assert cov_family.batch_shape == (4,)
    with pytest.raises(ValueError, match="vmapped family"):
        Gaussian(jnp.zeros(3), cov_family)


def test_13_a_family_noise_covariance_is_rejected_at_the_call():
    joint, noise_cov, y = _reference_problem()
    cov_family = _stacked(noise_cov, reps=4)
    with pytest.raises(ValueError, match="vmapped family"):
        joint.transform_update(y, cov_family)
    with pytest.raises(ValueError, match="vmapped family"):
        joint.condition(y, cov_family)
    with pytest.raises(ValueError, match="vmapped family"):
        joint.pathwise_update(jax.random.key(0), y, cov_family)


def test_13_inconsistently_stacked_leaves_are_diagnosed_at_batch_shape():
    treedef = jax.tree_util.tree_structure(
        EnsembleJoint(jnp.zeros((3, 2)), jnp.zeros((3, 4)))
    )
    mismatched = jax.tree_util.tree_unflatten(
        treedef, [jnp.zeros((2, 3, 2)), jnp.zeros((5, 3, 4))]
    )
    with pytest.raises(ValueError, match="do not broadcast"):
        _ = mismatched.batch_shape


# --- beyond the numbered obligations: one method call, one SVD ---------------


def test_class_methods_compute_exactly_one_svd():
    """The class methods share a single SVD internally; the public primitives
    each recompute their own. Counted in the jaxpr, since the sharing is
    invisible to inspection and a stray second SVD is a silent cost."""
    joint, noise_cov, y = _reference_problem()
    key = jax.random.key(0)

    for name, fn in (
        ("pathwise_update", lambda j, y: j.pathwise_update(key, y, noise_cov)),
        ("transform_update", lambda j, y: j.transform_update(y, noise_cov)),
        ("condition", lambda j, y: j.condition(y, noise_cov).mean),
        ("condition-cov", lambda j, y: j.condition(y, noise_cov).cov.F),
    ):
        count = _count_svd(jax.make_jaxpr(fn)(joint, y).jaxpr)
        assert count == 1, f"{name} computed {count} SVDs"

    s = jnp.asarray(RNG.normal(size=(4, 6)))
    assert _count_svd(jax.make_jaxpr(gain_weights)(s, jnp.zeros(6)).jaxpr) == 1
    assert _count_svd(jax.make_jaxpr(sqrt_transform)(s).jaxpr) == 1
    both = jax.make_jaxpr(lambda s: (gain_weights(s, jnp.zeros(6)), sqrt_transform(s)))
    assert _count_svd(both(s).jaxpr) == 2


# ===========================================================================
# Section 2 -- targeted regression tests. DO NOT DELETE AS REDUNDANT.
#
# One test per class of silent failure: each produces wrong numbers, a wrong
# cost, or a corrupted joint law without raising, and each documents why a
# rule of the contract exists.
# ===========================================================================


def test_regression_thin_svd_needs_the_identity_completion():
    """The naive U (I + Sigma^2)^-1/2 U^T omits the identity on the orthogonal
    complement of U's columns, and is wrong whenever rho < J.

    Wrong by O(1), with no exception and no shape error: the naive form is a
    (J, J) matrix like the right one.
    """
    J, N = 7, 3  # rho = 3 < J
    s = jnp.asarray(RNG.normal(size=(J, N)))

    U, sigma, _ = jnp.linalg.svd(s, full_matrices=False)
    naive = (U * (1.0 / jnp.sqrt(1.0 + sigma**2))) @ U.T

    w, Vec = np.linalg.eigh(np.eye(J) + np.asarray(s) @ np.asarray(s).T)
    reference = (Vec * w**-0.5) @ Vec.T

    np.testing.assert_allclose(
        sqrt_transform(s), reference, rtol=0, atol=1e3 * EPS
    )
    assert np.abs(np.asarray(naive) - reference).max() > 0.5

    # the two agree only when rho == J, which is why N < J is the case to test
    square = jnp.asarray(RNG.normal(size=(4, 4)))
    U2, sigma2, _ = jnp.linalg.svd(square, full_matrices=False)
    np.testing.assert_allclose(
        sqrt_transform(square),
        (U2 * (1.0 / jnp.sqrt(1.0 + sigma2**2))) @ U2.T,
        rtol=0,
        atol=1e3 * EPS,
    )


def test_regression_stably_formed_invariant_beats_the_re_formed_one():
    """Check 3's invariant must be formed as T T^T + (T s)(T s)^T = I.

    The algebraically equivalent T (I + s s^T) T^T = I re-forms the
    sigma_max^2-sized intermediate whose rounding the check exists to detect,
    and no float64 implementation can meet the tolerance. The failure needs a
    *mixed* spectrum: with every singular value at one large scale, T T^T is
    itself of order sigma_max^-2 and hides the loss.
    """
    J, N = 5, 7
    sigma_max = 1e10
    s = _s_with_spectrum(J, N, [sigma_max, 1e5, 1.0, 1.0, 1.0])
    T = np.asarray(sqrt_transform(s))
    S = np.asarray(s)
    Ts = T @ S

    stable = np.abs(T @ T.T + Ts @ Ts.T - np.eye(J)).max()
    re_formed = np.abs(T @ (np.eye(J) + S @ S.T) @ T.T - np.eye(J)).max()

    assert stable <= 128 * EPS * sigma_max
    assert re_formed > 1.0
    assert re_formed > 1e6 * stable


def test_regression_mixing_perturbation_representations_corrupts_the_update():
    """The perturbation must live in exactly one representation.

    pathwise_update writes the whitened perturbed residual as W(y - v_j) - eps_j
    and never exposes eps, because pushing the *same* eps through factor() to
    materialize y_j = v_j + L eps_j gives a different update: W L has
    orthonormal rows but is not the identity. Marginal statistics still look
    right, which is what makes it silent.
    """
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    Q, _ = np.linalg.qr(RNG.normal(size=(N, N)))
    noise_cov = RotatedWhitenPSD.from_matrix(R, Q)
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    key = jax.random.key(6)

    whitened_route = np.asarray(joint.pathwise_update(key, jnp.asarray(y), noise_cov))

    # the same eps, pushed through factor() instead
    eps = jax.random.normal(key, (J, N))
    L = noise_cov.factor()
    perturbed = jnp.asarray(V) + L.matvec(eps)
    s = jnp.asarray(_scaled_whitened_anomalies(V, noise_cov, J))
    w = gain_weights(s, noise_cov.whiten(jnp.asarray(y) - perturbed))
    Au = U - U.mean(axis=0)
    factor_route = U + np.einsum("pj,ij->ip", Au.T, np.asarray(w)) / np.sqrt(J - 1)

    assert np.abs(whitened_route - factor_route).max() > 1e-3

    # W L is orthonormal-rowed, which is why both look right marginally
    W = _recovered_whitener(noise_cov, N)
    WL = W @ np.asarray(L.to_dense())
    np.testing.assert_allclose(WL @ WL.T, np.eye(N), rtol=0, atol=1e3 * EPS)
    assert np.abs(WL - np.eye(N)).max() > 0.1


def test_regression_uncentred_transform_shifts_the_ensemble_mean():
    """T 1 = 1 holds only for mean-centred s, and the update depends on it.

    Building s from raw samples rather than anomalies still produces a
    perfectly valid (I + s s^T)^-1/2 — no exception, no shape error — but it no
    longer fixes the ones vector, so the transformed anomalies stop summing to
    zero and the posterior ensemble's mean is silently shifted away from the
    posterior mean.
    """
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    V = V + 20.0  # a large mean makes the uncentred error unmistakable
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    m_post, _ = _dense_posterior(U, V, R, y)

    # the layer centres, so its transform fixes the ones vector and the
    # updated ensemble's mean is the posterior mean
    centred = jnp.asarray(_scaled_whitened_anomalies(V, noise_cov, J))
    np.testing.assert_allclose(
        sqrt_transform(centred) @ jnp.ones(J), jnp.ones(J), rtol=0, atol=1e3 * EPS
    )
    members = np.asarray(joint.transform_update(jnp.asarray(y), noise_cov))
    np.testing.assert_allclose(
        members.mean(axis=0), m_post, rtol=0, atol=1e3 * EPS * np.abs(m_post).max()
    )

    # the uncentred version does not, and shifts the mean
    W = _recovered_whitener(noise_cov, N)
    uncentred = jnp.asarray(V @ W.T / np.sqrt(J - 1))
    T_bad = sqrt_transform(uncentred)
    assert np.abs(np.asarray(T_bad) @ np.ones(J) - 1.0).max() > 0.1
    Au = U - U.mean(axis=0)
    shifted = m_post + np.asarray(T_bad @ jnp.asarray(Au))
    assert np.abs(shifted.mean(axis=0) - m_post).max() > 1e-3


def test_regression_svd_gradient_is_nan_at_an_exactly_collapsed_operand():
    """The primitives are smooth at an exactly collapsed s, but the SVD's
    gradient there is nan. The contract scopes the differentiability claim to
    distinct, nonzero singular values; this asserts the nan so that the day JAX
    changes it — or the day a custom JVP is added — is visible.
    """
    collapsed = jnp.zeros((3, 4))
    assert bool(
        jnp.isnan(jax.grad(lambda s: jnp.sum(sqrt_transform(s)))(collapsed)).any()
    )
    assert bool(
        jnp.isnan(
            jax.grad(lambda s: jnp.sum(gain_weights(s, jnp.ones(4))))(collapsed)
        ).any()
    )

    # distinct, nonzero singular values differentiate finitely
    generic = jnp.asarray(RNG.normal(size=(3, 4)))
    assert bool(
        jnp.all(jnp.isfinite(jax.grad(lambda s: jnp.sum(sqrt_transform(s)))(generic)))
    )

    # and so does the float-generic degeneracy of mean-centring: at N >= J the
    # smallest singular value is ~1e-16, which is not an exact tie
    A = RNG.normal(size=(5, 7))
    centred = jnp.asarray((A - A.mean(axis=0)) / np.sqrt(4))
    sigma = np.linalg.svd(np.asarray(centred), compute_uv=False)
    assert sigma[-1] < 1e-14 and sigma[-1] != 0.0
    for fn in (
        lambda s: jnp.sum(sqrt_transform(s)),
        lambda s: jnp.sum(gain_weights(s, jnp.ones(7))),
    ):
        assert bool(jnp.all(jnp.isfinite(jax.grad(fn)(centred))))


def test_regression_singular_noise_covariance_yields_nan_without_raising():
    """A singular noise covariance is not detectable cheaply, so it surfaces as
    nan rather than an exception.

    Asserted deliberately, so the day it starts raising is visible here rather
    than in a caller's results. Debug checks stay off for that half: in debug
    mode the result checks turn the nan into an exception, which the second
    half of this test pins separately.
    """
    J, P, N = 5, 3, 4
    U, V, _, y = _problem(J, P, N)
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    diagonal = jnp.asarray([1.0, 0.0, 2.0, 3.0])  # a zero variance
    singular = PSDDiagonal(diagonal)

    assert singular.supports("whiten")  # support is static, and says nothing
    assert bool(jnp.isinf(singular.whiten(jnp.ones(N))).any())

    for result in (
        joint.pathwise_update(jax.random.key(0), jnp.asarray(y), singular),
        joint.transform_update(jnp.asarray(y), singular),
        joint.condition(jnp.asarray(y), singular).mean,
    ):
        assert bool(jnp.isnan(result).all())


def test_regression_all_three_methods_report_a_nan_result_in_debug_mode():
    """The result check is a postcondition, and it applies uniformly.

    Before it existed, `condition` alone raised — because it happened to route
    its mean through a constructor — and even then the posterior covariance
    factor went unchecked. The uniformity is the point: a caller who wraps a
    tempering loop in debug_checks must not get an exception from one
    conditioning path and a silent nan from the others on identical inputs.

    One asymmetry is deliberate and worth recording, because it limits what
    this test can guard. `condition` checks both halves of its result, but
    nothing distinguishes them: every input that makes the posterior factor
    non-finite makes the mean non-finite too — both flow from the same SVD, and
    a zero weight vector against a non-finite anomaly is nan rather than zero.
    So the mean check is what fires here, and deleting the factor check would
    not fail any test. It is kept for completeness, against a future change
    that computes the mean by a route the factor does not share.
    """
    J, P, N = 5, 3, 4
    U, V, _, y = _problem(J, P, N)
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    singular = PSDDiagonal(jnp.asarray([1.0, 0.0, 2.0, 3.0]))

    with debug_checks(True):
        for call in (
            lambda: joint.pathwise_update(jax.random.key(0), jnp.asarray(y), singular),
            lambda: joint.transform_update(jnp.asarray(y), singular),
            lambda: joint.condition(jnp.asarray(y), singular),
        ):
            with pytest.raises(ValueError, match="is not finite"):
                call()
        # the message points at the cause the check cannot itself detect
        with pytest.raises(ValueError, match="singular noise_cov"):
            joint.transform_update(jnp.asarray(y), singular)


def test_regression_result_checks_are_skipped_under_jit():
    """Tier 4 reads array values, so it is skipped on tracers.

    Asserted so the limit of the previous test is on the record: the result
    check is an eager-mode debugging aid, and a jitted driver loop gets the nan
    regardless of the debug flag.
    """
    J, P, N = 5, 3, 4
    U, V, _, y = _problem(J, P, N)
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))
    singular = PSDDiagonal(jnp.asarray([1.0, 0.0, 2.0, 3.0]))

    with debug_checks(True):
        jitted = jax.jit(lambda j, y: j.transform_update(y, singular))
        assert bool(jnp.isnan(jitted(joint, jnp.asarray(y))).all())
        mapped = jax.vmap(lambda y: joint.transform_update(y, singular))
        assert bool(jnp.isnan(mapped(jnp.stack([jnp.asarray(y)] * 2))).all())


def test_regression_sample_is_outside_the_result_check_by_design():
    """`sample` is excluded because its output is non-finite only if a field
    is, and fields are validated at construction — which is what makes the
    exclusion safe rather than an oversight."""
    with debug_checks(True):
        with pytest.raises(ValueError, match="PSDLowRank.F must be finite"):
            PSDLowRank(jnp.full((3, 2), jnp.nan))
        with pytest.raises(ValueError, match="mean must be finite"):
            Gaussian(jnp.full(3, jnp.nan), PSDLowRank(jnp.ones((3, 2))))

    # with the checks off, both construct and sampling is silently nan
    silent = Gaussian(jnp.zeros(3), PSDLowRank(jnp.full((3, 2), jnp.nan)))
    assert bool(jnp.isnan(silent.sample(jax.random.key(0), 2)).all())


def test_10_log_density_value_check_is_debug_only():
    """The evaluation point is a call-time operand like `y`, and covered by
    tier 4 for the same reason: a nan `x` returns nan without an exception."""
    n = 3
    gaussian = Gaussian(jnp.zeros(n), DensePSD.from_matrix(jnp.asarray(_psd(n))))
    bad_x = jnp.asarray([0.0, jnp.nan, 0.0])

    assert bool(jnp.isnan(gaussian.log_density(bad_x)))
    with debug_checks(True):
        with pytest.raises(ValueError, match="x must be finite"):
            gaussian.log_density(bad_x)


def test_regression_each_update_applies_the_whitener_j_plus_one_times():
    """Whitening the anomalies and the residuals in two calls costs 2J
    applications of W in the stochastic update, twice what is needed.

    Whitening is linear, so it commutes with centring and with subtracting y:
    one call on the J + 1 stacked rows [V; y] yields every whitened quantity
    the kernel needs. For a dense whitener those applications are the dominant
    O(J N^2) term, so a second call is a silent 2x — it produces correct
    numbers, which is why only a count catches it.
    """
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    joint = EnsembleJoint(jnp.asarray(U), jnp.asarray(V))

    for name, call in (
        ("pathwise_update", lambda cov: joint.pathwise_update(
            jax.random.key(0), jnp.asarray(y), cov)),
        ("transform_update", lambda cov: joint.transform_update(jnp.asarray(y), cov)),
        ("condition", lambda cov: joint.condition(jnp.asarray(y), cov)),
    ):
        noise_cov = CountingWhitenPSD.counting(R)
        call(noise_cov)
        applied = sum(object.__getattribute__(noise_cov, "log"))
        assert applied == J + 1, f"{name} whitened {applied} vectors, expected {J + 1}"

    # and the numbers are unchanged by the grouping: the dense reference
    # centres before whitening, the implementation whitens before centring
    plain = DensePSD.from_matrix(jnp.asarray(R))
    m_post, _ = _dense_posterior(U, V, R, y)
    np.testing.assert_allclose(
        joint.transform_update(jnp.asarray(y), plain).mean(axis=0),
        m_post,
        rtol=0,
        atol=1e3 * EPS * np.abs(m_post).max(),
    )
