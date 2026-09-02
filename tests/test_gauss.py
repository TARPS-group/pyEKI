"""Conformance and regression tests for the joint Gaussian layer.

The file has two sections. The first works through the fifteen numbered
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

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.gauss import (
    EmpiricalJoint,
    Gaussian,
    GaussianJoint,
    gain_weights,
    sqrt_transform,
)
from pyeki.linalg import (
    Dense,
    DensePSD,
    LinOp,
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
    """The three test-local operators are contract-valid, so the layer's
    failures cannot be blamed on the fixtures."""
    R = _psd(4)
    Q, _ = np.linalg.qr(RNG.normal(size=(4, 4)))
    rotated = RotatedWhitenPSD.from_matrix(R, Q)
    check_operator(rotated)
    check_operator(WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(4)))))
    check_operator(CountingWhitenPSD.counting(_psd(4)))
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


def _exact_posterior_mean(U, V, R, y) -> np.ndarray:
    """The posterior mean in exact rational arithmetic over the stored floats.

    A float64 reference is not good enough for the ill-conditioned,
    large-prediction-mean regime: it is itself the inaccurate side there. This
    forms the empirical moments and solves (C_vv + R) x = y - v_bar exactly,
    by Gaussian elimination over ``Fraction``, so the comparison has a true
    answer to compare against. Only for tiny problems.
    """
    J, P, N = U.shape[0], U.shape[1], V.shape[1]
    Uf = [[Fraction(float(x)) for x in row] for row in U]
    Vf = [[Fraction(float(x)) for x in row] for row in V]
    Rf = [[Fraction(float(x)) for x in row] for row in R]
    um = [sum(Uf[i][k] for i in range(J)) / J for k in range(P)]
    vm = [sum(Vf[i][k] for i in range(J)) / J for k in range(N)]
    Au = [[Uf[i][k] - um[k] for k in range(P)] for i in range(J)]
    Av = [[Vf[i][k] - vm[k] for k in range(N)] for i in range(J)]
    Cuv = [[sum(Au[i][a] * Av[i][b] for i in range(J)) / (J - 1)
            for b in range(N)] for a in range(P)]
    M = [[sum(Av[i][a] * Av[i][b] for i in range(J)) / (J - 1) + Rf[a][b]
          for b in range(N)] for a in range(N)]
    rhs = [Fraction(float(y[k])) - vm[k] for k in range(N)]
    for c in range(N):                                   # exact elimination
        pivot = max(range(c, N), key=lambda r: abs(M[r][c]))
        M[c], M[pivot] = M[pivot], M[c]
        rhs[c], rhs[pivot] = rhs[pivot], rhs[c]
        for r in range(c + 1, N):
            f = M[r][c] / M[c][c]
            for k in range(c, N):
                M[r][k] -= f * M[c][k]
            rhs[r] -= f * rhs[c]
    x = [Fraction(0)] * N
    for r in reversed(range(N)):
        x[r] = (rhs[r] - sum(M[r][k] * x[k] for k in range(r + 1, N))) / M[r][r]
    return np.array(
        [float(um[a] + sum(Cuv[a][k] * x[k] for k in range(N))) for a in range(P)]
    )


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


def _exact_moment_ensemble(J: int, mu: np.ndarray, F: np.ndarray) -> np.ndarray:
    """An ensemble whose empirical moments equal ``mu`` and ``F @ F.T`` exactly.

    Take the complete QR of the all-ones vector in R^J, let E be its last k
    columns — orthonormal and each orthogonal to the ones vector — and set the
    members to ``mu + sqrt(J - 1) E F^T``. Then the member rows have mean
    ``mu`` (because ``1^T E = 0``) and empirical covariance ``F E^T E F^T``,
    which is ``F F^T`` (because ``E^T E = I_k``). Only ``J >= k + 1`` binds;
    at ``J = k`` the construction silently fails.

    ``F`` is a factor of the target covariance, in the column convention the
    operator layer uses; ``G`` is reserved for a linear map.
    """
    k = F.shape[1]
    assert J >= k + 1, "the construction needs J >= k + 1"
    Q, _ = np.linalg.qr(np.ones((J, 1)), mode="complete")
    E = Q[:, 1 : k + 1]
    return mu + np.sqrt(J - 1) * E @ F.T


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
# Section 1 -- the fifteen conformance obligations
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

    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    np.testing.assert_allclose(
        joint.transform_update(jnp.asarray(y), plain),
        joint.transform_update(jnp.asarray(y), rotated),
        rtol=0,
        atol=1e4 * EPS,
    )
    np.testing.assert_allclose(
        joint.to_gaussian_joint().condition(jnp.asarray(y), plain).mean,
        joint.to_gaussian_joint().condition(jnp.asarray(y), rotated).mean,
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


@pytest.mark.parametrize(("J", "N"), [(6, 9), (50, 60)], ids=["small", "large-J"])
def test_3_sqrt_transform_is_symmetric_and_preserves_mean_centering(J, N):
    """T = T^T for every s; T 1 = 1 for mean-centered s only.

    Both terms of the tolerance are needed and both scale. The mean shift the
    modifier induces is O((eps sigma_max)^2), but the *computed* T @ 1 carries
    ordinary round-off from its J-term dot products on top, which dominates
    until sigma_max reaches 1/sqrt(eps) — and that floor grows with J, so a
    constant floor calibrated at one shape expires at another. Measured worst
    ratios over J up to 400 and sigma_max up to 1e13: 0.83 against J*EPS in
    the floor regime, 2.0 against (EPS*sigma_max)^2 at large sigma_max. The
    large-J case is parametrized in so neither constant can be fitted to a
    single shape again.
    """
    for scale in (1.0, 1e3, 1e8, 1e11):
        A = RNG.normal(size=(J, N))
        s = jnp.asarray((A - A.mean(axis=0)) * scale)
        T = np.asarray(sqrt_transform(s))
        sigma_max = float(np.linalg.svd(np.asarray(s), compute_uv=False)[0])

        np.testing.assert_allclose(T, T.T, rtol=0, atol=1e3 * EPS)

        shift = np.abs(T @ np.ones(J) - 1.0).max()
        assert shift <= 8 * J * EPS + 16 * (EPS * sigma_max) ** 2

    # on general (uncentered) s no such identity holds
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
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    m_post, C_post = _dense_posterior(U, V, R, y)
    scale = max(np.abs(m_post).max(), np.abs(C_post).max())

    members = np.asarray(joint.transform_update(jnp.asarray(y), noise_cov))
    sample_mean, sample_cov = _sample_moments(members)
    np.testing.assert_allclose(sample_mean, m_post, rtol=0, atol=1e3 * EPS * scale)
    np.testing.assert_allclose(sample_cov, C_post, rtol=0, atol=1e3 * EPS * scale)

    posterior = joint.to_gaussian_joint().condition(jnp.asarray(y), noise_cov)
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
    posterior = EmpiricalJoint(
        u_samples=jnp.asarray(U), v_samples=jnp.asarray(V)
    ).to_gaussian_joint().condition(jnp.asarray(y), DensePSD.from_matrix(jnp.asarray(R)))
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

    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    got = np.asarray(joint.pathwise_update(key, jnp.asarray(y), noise_cov))

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
    F = RNG.normal(size=(P + N, k))
    mu = RNG.normal(size=P + N)
    R = _psd(N)
    y = RNG.normal(size=N)

    members = _exact_moment_ensemble(J, mu, F)
    U, V = members[:, :P], members[:, P:]

    # the fixture is exact: check it before trusting what it implies
    C = F @ F.T
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

    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))

    posterior = joint.to_gaussian_joint().condition(jnp.asarray(y), noise_cov)
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


def test_7_from_samples_matches_the_empirical_moments():
    """from_samples reproduces the sample mean and the J-1 covariance exactly."""
    J, n = 6, 3
    samples = jnp.asarray(RNG.normal(size=(J, n)))

    got = Gaussian.from_samples(samples)

    ref_mean = np.mean(np.asarray(samples), axis=0)
    ref_cov = np.cov(np.asarray(samples).T, ddof=1)
    np.testing.assert_allclose(
        got.mean, ref_mean, rtol=0, atol=1e3 * EPS * np.abs(ref_mean).max()
    )
    np.testing.assert_allclose(
        got.cov.to_dense(), ref_cov, rtol=0, atol=1e3 * EPS * np.abs(ref_cov).max()
    )
    np.testing.assert_allclose(
        got.cov.diag(), np.diag(ref_cov), rtol=0, atol=1e3 * EPS * np.abs(ref_cov).max()
    )


def test_7_from_samples_holds_a_low_rank_factor_and_never_forms_the_matrix():
    """The covariance is PSDLowRank of width J: diag and factor, nothing else."""
    J, n = 4, 9
    samples = jnp.asarray(RNG.normal(size=(J, n)))

    cov = Gaussian.from_samples(samples).cov

    assert isinstance(cov, PSDLowRank)
    assert cov.shape == (n, n)
    assert cov.F.shape == (n, J)
    assert cov.supports("diag") and cov.supports("factor")
    for op in ("solve", "whiten", "logdet"):
        assert not cov.supports(op)
    # Singular at J - 1 < n, so a density is undefined and must raise.
    with pytest.raises(UnsupportedOpError):
        Gaussian.from_samples(samples).log_density(jnp.zeros(n))


def test_7_from_samples_agrees_with_the_joint_it_is_the_one_block_case_of():
    """Its moments equal EmpiricalJoint's u-block moments, from the same members."""
    J, P, N = 7, 4, 3
    u = jnp.asarray(RNG.normal(size=(J, P)))
    v = jnp.asarray(RNG.normal(size=(J, N)))

    fit = Gaussian.from_samples(u)
    joint = EmpiricalJoint(u_samples=u, v_samples=v)

    np.testing.assert_array_equal(fit.mean, joint.u_mean)
    want = np.asarray(joint.u_anomalies).T @ np.asarray(joint.u_anomalies) / (J - 1)
    np.testing.assert_allclose(
        fit.cov.to_dense(), want, rtol=0, atol=1e3 * EPS * np.abs(want).max()
    )


def test_7_from_samples_gives_identical_members_exactly_zero_spread():
    """The stable centring, not jnp.mean: a collapsed sample has no anomalies."""
    collapsed = jnp.tile(jnp.asarray([1e23, 2e23, 3e23]), (5, 1))

    cov = Gaussian.from_samples(collapsed).cov

    np.testing.assert_array_equal(cov.F, np.zeros_like(cov.F))
    np.testing.assert_array_equal(cov.to_dense(), np.zeros((3, 3)))


def test_7_from_samples_validates_rank_and_member_count():
    with pytest.raises(ValueError, match="rank 2"):
        Gaussian.from_samples(jnp.zeros(5))
    with pytest.raises(ValueError, match="at least 2 samples"):
        Gaussian.from_samples(jnp.zeros((1, 3)))



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
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))

    pathwise = joint.pathwise_update(jax.random.key(0), y, noise_cov)
    np.testing.assert_array_equal(pathwise, jnp.asarray(U))

    transform = joint.transform_update(y, noise_cov)
    # reconstructs u_bar + a_j, so round-off rather than bit-exact
    np.testing.assert_allclose(
        transform, U, rtol=0, atol=1e3 * EPS * max(1.0, np.abs(U).max())
    )

    posterior = joint.to_gaussian_joint().condition(y, noise_cov)
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
    joint = EmpiricalJoint(
        u_samples=jnp.asarray(RNG.normal(size=(J, P))),
        v_samples=jnp.asarray(RNG.normal(size=(J, N))),
    )
    noise_cov = PSDLowRank(jnp.asarray(RNG.normal(size=(N, N))))
    y = jnp.asarray(RNG.normal(size=N))
    assert not noise_cov.supports("whiten")

    for call in (
        lambda: joint.pathwise_update(jax.random.key(0), y, noise_cov),
        lambda: joint.transform_update(y, noise_cov),
        lambda: joint.to_gaussian_joint().condition(y, noise_cov),
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
    posterior = EmpiricalJoint(
        u_samples=jnp.asarray(U), v_samples=jnp.asarray(V)
    ).to_gaussian_joint().condition(jnp.asarray(y), DensePSD.from_matrix(jnp.asarray(R)))
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


def test_10_empirical_joint_construction_validation():
    with pytest.raises(ValueError, match="rank 2"):
        EmpiricalJoint(u_samples=jnp.zeros(6), v_samples=jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="rank 2"):
        EmpiricalJoint(u_samples=jnp.zeros((3, 2)), v_samples=jnp.zeros((3, 2, 1)))
    with pytest.raises(ValueError, match="same number of samples"):
        EmpiricalJoint(u_samples=jnp.zeros((3, 2)), v_samples=jnp.zeros((4, 2)))
    with pytest.raises(ValueError, match="at least 2 samples"):
        EmpiricalJoint(u_samples=jnp.zeros((1, 2)), v_samples=jnp.zeros((1, 2)))
    with pytest.raises(ValueError, match="core sizes must be positive"):
        EmpiricalJoint(u_samples=jnp.zeros((3, 0)), v_samples=jnp.zeros((3, 2)))


def test_10_conditioning_call_validation():
    J, P, N = 5, 3, 4
    joint = EmpiricalJoint(
        u_samples=jnp.asarray(RNG.normal(size=(J, P))),
        v_samples=jnp.asarray(RNG.normal(size=(J, N))),
    )
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(N)))
    y = jnp.asarray(RNG.normal(size=N))

    def calls(y_arg, cov_arg):
        return (
            lambda: joint.pathwise_update(jax.random.key(0), y_arg, cov_arg),
            lambda: joint.transform_update(y_arg, cov_arg),
            lambda: joint.to_gaussian_joint().condition(y_arg, cov_arg),
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
        with pytest.raises(ValueError, match="dimension 4"):
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
    joint_args = {
        "u_samples": jnp.zeros((3, 2)),
        "v_samples": jnp.asarray([[0.0], [jnp.nan], [0.0]]),
    }

    # off by default: no exception, just a nan-bearing object
    Gaussian(jnp.asarray([0.0, jnp.nan, 0.0]), cov)
    EmpiricalJoint(**joint_args)
    gain_weights(jnp.asarray([[jnp.nan, 0.0]]), jnp.zeros(2))

    with debug_checks(True):
        with pytest.raises(ValueError, match="mean must be finite"):
            Gaussian(jnp.asarray([0.0, jnp.nan, 0.0]), cov)
        with pytest.raises(ValueError, match="v_samples must be finite"):
            EmpiricalJoint(**joint_args)
        with pytest.raises(ValueError, match="s must be finite"):
            gain_weights(jnp.asarray([[jnp.nan, 0.0]]), jnp.zeros(2))
        with pytest.raises(ValueError, match="b must be finite"):
            gain_weights(jnp.asarray([[1.0, 0.0]]), jnp.asarray([jnp.nan, 0.0]))
        with pytest.raises(ValueError, match="s must be finite"):
            sqrt_transform(jnp.asarray([[jnp.nan, 0.0]]))

        good = EmpiricalJoint(
            u_samples=jnp.asarray(RNG.normal(size=(3, 2))),
        v_samples=jnp.asarray(RNG.normal(size=(3, 2))),
        )
        noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(2)))
        with pytest.raises(ValueError, match="y must be finite"):
            good.transform_update(jnp.asarray([jnp.nan, 0.0]), noise_cov)


# --- 11. JAX round trips -----------------------------------------------------


def _reference_problem():
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    return (
        EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V)),
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
            joint.to_gaussian_joint().condition(y, noise_cov).mean,
            jax.jit(lambda j, y: j.to_gaussian_joint().condition(y, noise_cov).mean)(
                joint, y
            ),
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
    family = jax.vmap(lambda u, v: EmpiricalJoint(u_samples=u, v_samples=v))(U, V)
    assert type(family) is EmpiricalJoint
    assert family.batch_shape == (5,)
    assert (family.n_samples, family.u_dim, family.v_dim) == (6, 3, 4)


def test_11_a_vmapped_family_agrees_with_a_python_loop():
    members, J, P, N = 5, 6, 3, 4
    U = RNG.normal(size=(members, J, P))
    V = RNG.normal(size=(members, J, N))
    ys = RNG.normal(size=(members, N))
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(N)))

    def update(u, v, y):
        return EmpiricalJoint(u_samples=u, v_samples=v).transform_update(y, noise_cov)

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

    joint = EmpiricalJoint(u_samples=jnp.zeros((100, 12)), v_samples=jnp.zeros((100, 40)))
    assert repr(joint) == "EmpiricalJoint(n_samples=100, u_dim=12, v_dim=40)"

    for text in (repr(gaussian), repr(joint)):
        assert "0." not in text and "Array" not in text


def test_12_repr_never_raises_on_unreadable_leaves():
    treedef = jax.tree_util.tree_structure(
        EmpiricalJoint(u_samples=jnp.zeros((2, 2)), v_samples=jnp.zeros((2, 2)))
    )
    broken = jax.tree_util.tree_unflatten(treedef, [object(), object()])
    assert repr(broken) == "<EmpiricalJoint (unprintable leaves)>"


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


def test_13_empirical_joint_family_is_legible_and_inert():
    joint, noise_cov, y = _reference_problem()
    family = _stacked(joint)
    assert family.batch_shape == (8,)
    assert (family.n_samples, family.u_dim, family.v_dim) == (6, 3, 4)
    assert repr(family) == (
        "vmapped(EmpiricalJoint(n_samples=6, u_dim=3, v_dim=4), batch=(8,))"
    )

    with pytest.raises(ValueError, match="vmapped family"):
        family.pathwise_update(jax.random.key(0), y, noise_cov)
    with pytest.raises(ValueError, match="vmapped family"):
        family.transform_update(y, noise_cov)
    with pytest.raises(ValueError, match="vmapped family"):
        family.to_gaussian_joint()
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
        joint.to_gaussian_joint().condition(y, cov_family)
    with pytest.raises(ValueError, match="vmapped family"):
        joint.pathwise_update(jax.random.key(0), y, cov_family)


def test_13_inconsistently_stacked_leaves_are_diagnosed_at_batch_shape():
    treedef = jax.tree_util.tree_structure(
        EmpiricalJoint(u_samples=jnp.zeros((3, 2)), v_samples=jnp.zeros((3, 4)))
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
        ("condition", lambda j, y: j.to_gaussian_joint().condition(y, noise_cov).mean),
        (
            "condition-cov",
            lambda j, y: j.to_gaussian_joint().condition(y, noise_cov).cov.F,
        ),
        (
            "pathwise",
            lambda j, y: j.to_gaussian_joint().pathwise(
                u=j.u_samples,
                v=j.v_samples,
                whitened_noise=jnp.zeros_like(j.v_samples),
                y=y,
                noise_cov=noise_cov,
            ),
        ),
    ):
        count = _count_svd(jax.make_jaxpr(fn)(joint, y).jaxpr)
        assert count == 1, f"{name} computed {count} SVDs"

    s = jnp.asarray(RNG.normal(size=(4, 6)))
    assert _count_svd(jax.make_jaxpr(gain_weights)(s, jnp.zeros(6)).jaxpr) == 1
    assert _count_svd(jax.make_jaxpr(sqrt_transform)(s).jaxpr) == 1
    both = jax.make_jaxpr(lambda s: (gain_weights(s, jnp.zeros(6)), sqrt_transform(s)))
    assert _count_svd(both(s).jaxpr) == 2


# --- 14. the joint factor and its constructors -------------------------------


def _linear_gaussian(P: int, N: int, k: int | None = None, seed: int = 23):
    """A linear-Gaussian problem, and everything needed to check it densely.

    Returns the prior as a :class:`Gaussian` with a structured covariance, the
    map as an operator, the noise as an operator, an observation, and the
    NumPy originals. ``k`` below ``P`` gives a rank-deficient prior, for which
    no precision-form posterior exists.
    """
    rng = np.random.default_rng(seed)
    if k is None:
        d = rng.uniform(0.5, 3.0, size=P)
        prior_cov, L = PSDDiagonal(jnp.asarray(d)), np.diag(np.sqrt(d))
    else:
        L = rng.normal(size=(P, k))
        prior_cov = PSDLowRank(jnp.asarray(L))
    m0 = rng.normal(size=P)
    G = rng.normal(size=(N, P))
    R = _psd(N)
    y = rng.normal(size=N)
    return (
        Gaussian(jnp.asarray(m0), prior_cov),
        Dense(jnp.asarray(G)),
        DensePSD.from_matrix(jnp.asarray(R)),
        jnp.asarray(y),
        {"m0": m0, "L": L, "G": G, "R": R, "y": y, "C0": L @ L.T},
    )


@pytest.mark.parametrize("k", [None, 2], ids=["full-rank prior", "k<P"])
def test_14_from_linear_map_reaches_the_closed_form_posterior(k):
    """The joint of u and Gu, conditioned, is the closed-form linear-Gaussian
    posterior — the entry point the layer previously had no way to reach.

    Checked against two independent dense references: block conditioning of
    the analytic joint, and, where the prior is invertible, the precision form
    (C_0^-1 + G^T R^-1 G)^-1. At k < P the prior is singular and the precision
    form does not exist, which is precisely the case the factor route handles
    and an inverse-based one cannot.
    """
    P, N = 4, 3
    prior, linear_map, noise_cov, y, ref = _linear_gaussian(P, N, k)
    C0, G, R, y_np = ref["C0"], ref["G"], ref["R"], ref["y"]

    joint = GaussianJoint.from_linear_map(prior, linear_map)
    assert joint.latent_dim == (P if k is None else k)
    assert joint.u_dim == P and joint.v_dim == N

    m_ref, C_ref = _dense_conditional(
        ref["m0"], G @ ref["m0"], C0, C0 @ G.T, G @ C0 @ G.T, R, y_np
    )
    scale = max(np.abs(m_ref).max(), np.abs(C_ref).max())
    posterior = joint.condition(y, noise_cov)
    np.testing.assert_allclose(posterior.mean, m_ref, rtol=0, atol=1e3 * EPS * scale)
    np.testing.assert_allclose(
        posterior.cov.to_dense(), C_ref, rtol=0, atol=1e3 * EPS * scale
    )
    assert isinstance(posterior.cov, PSDLowRank)
    assert posterior.cov.F.shape == (P, joint.latent_dim)

    if k is None:
        precision = np.linalg.inv(C0) + G.T @ np.linalg.solve(R, G)
        C_txt = np.linalg.inv(precision)
        m_txt = C_txt @ (
            np.linalg.solve(C0, ref["m0"]) + G.T @ np.linalg.solve(R, y_np)
        )
        np.testing.assert_allclose(posterior.mean, m_txt, rtol=0, atol=1e3 * EPS * scale)
        np.testing.assert_allclose(
            posterior.cov.to_dense(), C_txt, rtol=0, atol=1e3 * EPS * scale
        )


def test_14_from_linear_map_keeps_the_prior_factor_an_operator():
    """The u block stays whatever operator the prior's ``factor()`` returned,
    and only the v block is materialized. Densifying the u factor would throw
    away the structure the operator layer exists to carry."""
    prior, linear_map, _, _, _ = _linear_gaussian(4, 3)
    joint = GaussianJoint.from_linear_map(prior, linear_map)
    assert isinstance(joint.u_factor, LinOp)
    assert type(joint.u_factor) is type(prior.cov.factor())
    assert isinstance(joint.v_factor, Dense)
    assert joint.v_factor.shape == (3, 4)


def test_14_conditioning_a_linear_map_joint_whitens_k_plus_one_vectors():
    """The cost claim generalizes off the sample case: whitening is k + 1
    applications, from one call on the stacked columns [F_v | y - v_bar],
    whatever the latent width means."""
    P, N, k = 5, 4, 2
    prior, linear_map, _, y, ref = _linear_gaussian(P, N, k)
    joint = GaussianJoint.from_linear_map(prior, linear_map)
    noise_cov = CountingWhitenPSD.counting(ref["R"])
    joint.condition(y, noise_cov)
    applied = sum(object.__getattribute__(noise_cov, "log"))
    assert applied == k + 1, f"whitened {applied} vectors, expected {k + 1}"


def test_14_from_samples_reproduces_the_empirical_moments():
    """``from_samples`` matches moments: the two marginals and the implied
    cross-covariance equal the hand-written empirical ones."""
    J, P, N = 7, 3, 4
    U, V, _, _ = _problem(J, P, N)
    joint = GaussianJoint.from_samples(
        u_samples=jnp.asarray(U), v_samples=jnp.asarray(V)
    )
    assert joint.latent_dim == J
    ref = _dense_moments(U, V)
    scale = max(np.abs(ref["Cuu"]).max(), np.abs(ref["Cvv"]).max())

    np.testing.assert_allclose(joint.u_mean, ref["u_mean"], rtol=0, atol=1e3 * EPS)
    np.testing.assert_allclose(joint.v_mean, ref["v_mean"], rtol=0, atol=1e3 * EPS)
    np.testing.assert_allclose(
        joint.u_marginal.cov.to_dense(), ref["Cuu"], rtol=0, atol=1e3 * EPS * scale
    )
    np.testing.assert_allclose(
        joint.v_marginal.cov.to_dense(), ref["Cvv"], rtol=0, atol=1e3 * EPS * scale
    )
    # the cross block, which no accessor exposes: F_u F_v^T
    Fu = np.asarray(joint.u_factor.to_dense())
    Fv = np.asarray(joint.v_factor.to_dense())
    np.testing.assert_allclose(Fu @ Fv.T, ref["Cuv"], rtol=0, atol=1e3 * EPS * scale)


def test_14_the_projection_to_a_joint_is_lossless():
    """A mean and a centred factor of width J *are* the samples: u_j = u_bar +
    sqrt(J - 1) (F_u)_j. Nothing needs the samples supplied a second time, and
    both updates read their realizations off the factor."""
    J, P, N = 9, 4, 3
    U, V, _, _ = _problem(J, P, N)
    empirical = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    joint = empirical.to_gaussian_joint()

    for factor, samples in (
        (joint.u_factor, U),
        (joint.v_factor, V),
    ):
        F = np.asarray(factor.to_dense())
        recovered = np.asarray(
            samples.mean(axis=0) + np.sqrt(J - 1) * F.T
        )
        np.testing.assert_allclose(
            recovered, samples, rtol=0, atol=1e3 * EPS * np.abs(samples).max()
        )
        # centred, which is what makes the latent index a sample index
        assert np.abs(F @ np.ones(J)).max() < 1e3 * EPS * np.abs(F).max()


def test_14_from_factors_accepts_arrays_and_operators_alike():
    """The escape hatch wraps a bare array as a ``Dense`` and leaves an
    operator alone, so a caller mixing the two gets one representation."""
    P, N, k = 3, 4, 2
    Fu = jnp.asarray(RNG.normal(size=(P, k)))
    Fv = Dense(jnp.asarray(RNG.normal(size=(N, k))))
    joint = GaussianJoint.from_factors(
        u_mean=jnp.zeros(P), v_mean=jnp.zeros(N), u_factor=Fu, v_factor=Fv
    )
    assert isinstance(joint.u_factor, Dense)
    assert joint.v_factor is Fv
    assert joint.latent_dim == k
    np.testing.assert_allclose(joint.u_marginal.cov.to_dense(), np.asarray(Fu @ Fu.T))


def test_14_construction_and_constructor_validation():
    P, N, k = 3, 4, 2
    Fu = Dense(jnp.zeros((P, k)))
    Fv = Dense(jnp.zeros((N, k)))
    fields = {"u_mean": jnp.zeros(P), "v_mean": jnp.zeros(N)}

    with pytest.raises(TypeError, match="pyeki.linalg.LinOp"):
        GaussianJoint(**fields, u_factor=jnp.zeros((P, k)), v_factor=Fv)
    with pytest.raises(ValueError, match="disagrees with u_mean"):
        GaussianJoint(**fields, u_factor=Dense(jnp.zeros((P + 1, k))), v_factor=Fv)
    with pytest.raises(ValueError, match="share a latent width"):
        GaussianJoint(**fields, u_factor=Fu, v_factor=Dense(jnp.zeros((N, k + 1))))
    with pytest.raises(ValueError, match="rank 1"):
        GaussianJoint(
            u_mean=jnp.zeros((1, P)), v_mean=jnp.zeros(N), u_factor=Fu, v_factor=Fv
        )

    prior, linear_map, _, _, _ = _linear_gaussian(P, N)
    with pytest.raises(TypeError, match="pyeki.gauss.Gaussian"):
        GaussianJoint.from_linear_map(jnp.zeros(P), linear_map)
    with pytest.raises(TypeError, match="pyeki.linalg.LinOp"):
        GaussianJoint.from_linear_map(prior, jnp.zeros((N, P)))
    with pytest.raises(ValueError, match="takes vectors of length"):
        GaussianJoint.from_linear_map(prior, Dense(jnp.zeros((N, P + 1))))
    no_factor = WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(P))))
    assert not no_factor.supports("factor")
    with pytest.raises(UnsupportedOpError, match="factor"):
        GaussianJoint.from_linear_map(
            Gaussian(jnp.zeros(P), no_factor), Dense(jnp.zeros((N, P)))
        )
    with pytest.raises(ValueError, match="at least 2 samples"):
        GaussianJoint.from_samples(
            u_samples=jnp.zeros((1, P)), v_samples=jnp.zeros((1, N))
        )


def test_14_a_joint_family_is_legible_and_inert():
    joint = GaussianJoint.from_samples(
        u_samples=jnp.zeros((6, 3)), v_samples=jnp.zeros((6, 4))
    )
    family = _stacked(joint, reps=5)
    assert family.batch_shape == (5,)
    assert (family.u_dim, family.v_dim, family.latent_dim) == (3, 4, 6)
    assert repr(family) == (
        "vmapped(GaussianJoint(u_dim=3, v_dim=4, latent_dim=6), batch=(5,))"
    )
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(4)))
    y = jnp.zeros(4)
    with pytest.raises(ValueError, match="vmapped family"):
        family.condition(y, noise_cov)
    with pytest.raises(ValueError, match="vmapped family"):
        family.pathwise(
            u=jnp.zeros(3),
            v=jnp.zeros(4),
            whitened_noise=jnp.zeros(4),
            y=y,
            noise_cov=noise_cov,
        )
    for name in ("u_marginal", "v_marginal"):
        with pytest.raises(ValueError, match="vmapped family"):
            getattr(family, name)


def test_14_a_joint_round_trips_through_a_pytree_and_runs_under_jit():
    joint, noise_cov, y = _reference_problem()
    gaussian_joint = joint.to_gaussian_joint()
    leaves, treedef = jax.tree_util.tree_flatten(gaussian_joint)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is GaussianJoint
    np.testing.assert_array_equal(
        rebuilt.condition(y, noise_cov).mean,
        gaussian_joint.condition(y, noise_cov).mean,
    )
    np.testing.assert_allclose(
        jax.jit(lambda j, y: j.condition(y, noise_cov).mean)(gaussian_joint, y),
        gaussian_joint.condition(y, noise_cov).mean,
        rtol=0,
        atol=1e2 * EPS,
    )


# --- 15. the pathwise map ----------------------------------------------------


def test_15_pathwise_matches_the_dense_matheron_map_elementwise():
    """u + K(y - v - eta) with eta = W^-1 eps, formed densely and compared
    element by element. The map is affine, so this is the whole obligation;
    the distributional claim follows and is checked exactly below."""
    J, P, N = 8, 3, 4
    U, V, R, y = _problem(J, P, N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    joint = GaussianJoint.from_samples(
        u_samples=jnp.asarray(U), v_samples=jnp.asarray(V)
    )
    eps = RNG.normal(size=(J, N))

    got = joint.pathwise(
        u=jnp.asarray(U),
        v=jnp.asarray(V),
        whitened_noise=jnp.asarray(eps),
        y=jnp.asarray(y),
        noise_cov=noise_cov,
    )
    W = _recovered_whitener(noise_cov, N)
    eta = np.linalg.solve(W, eps.T).T
    want = U + (y - V - eta) @ _dense_gain(U, V, R).T
    np.testing.assert_allclose(
        got, want, rtol=0, atol=1e3 * EPS * max(1.0, np.abs(want).max())
    )


def test_15_pathwise_carries_an_exact_moment_sample_set_to_the_posterior():
    """Matheron's rule pushes the joint law to the posterior law, exactly.

    Rather than a Monte Carlo check, build realizations whose *empirical*
    moments equal the target joint's exactly — u, v and eta together, so that
    eta has covariance R and zero cross-covariance with the pair. The map is
    affine, so exact input moments give exact output moments: the transported
    sample mean and covariance must equal the closed-form posterior's.
    """
    P, N, k = 3, 2, 2
    rng = np.random.default_rng(31)
    Fu, Fv = rng.normal(size=(P, k)), rng.normal(size=(N, k))
    m_u, m_v = rng.normal(size=P), rng.normal(size=N)
    R = _psd(N)
    L_R = np.linalg.cholesky(R)
    y = rng.normal(size=N)

    # the joint factor of (u, v, eta): one shared latent for the pair, an
    # independent one for the noise, so cross-covariance is exactly zero
    stacked = np.block(
        [[Fu, np.zeros((P, N))], [Fv, np.zeros((N, N))], [np.zeros((N, k)), L_R]]
    )
    J = k + N + 1
    rows = _exact_moment_ensemble(J, np.concatenate([m_u, m_v, np.zeros(N)]), stacked)
    U, V, eta = rows[:, :P], rows[:, P : P + N], rows[:, P + N :]

    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    W = _recovered_whitener(noise_cov, N)
    joint = GaussianJoint.from_factors(
        u_mean=jnp.asarray(m_u),
        v_mean=jnp.asarray(m_v),
        u_factor=jnp.asarray(Fu),
        v_factor=jnp.asarray(Fv),
    )
    transported = np.asarray(
        joint.pathwise(
            u=jnp.asarray(U),
            v=jnp.asarray(V),
            whitened_noise=jnp.asarray(eta @ W.T),
            y=jnp.asarray(y),
            noise_cov=noise_cov,
        )
    )

    m_ref, C_ref = _dense_conditional(
        m_u, m_v, Fu @ Fu.T, Fu @ Fv.T, Fv @ Fv.T, R, y
    )
    scale = max(np.abs(m_ref).max(), np.abs(C_ref).max())
    sample_mean, sample_cov = _sample_moments(transported)
    np.testing.assert_allclose(sample_mean, m_ref, rtol=0, atol=1e4 * EPS * scale)
    np.testing.assert_allclose(sample_cov, C_ref, rtol=0, atol=1e4 * EPS * scale)


def test_15_pathwise_agrees_with_the_sample_update_on_the_joints_own_samples():
    """``EmpiricalJoint.pathwise_update`` is the same map on the samples the
    joint was fitted to. It takes the cheaper route — the residuals come from
    the whitened factor rather than a second whitening — so the two agree to
    round-off rather than bit for bit."""
    J, P, N = 7, 3, 4
    U, V, R, y = _problem(J, P, N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    empirical = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    key = jax.random.key(4)

    shipped = np.asarray(empirical.pathwise_update(key, jnp.asarray(y), noise_cov))
    general = np.asarray(
        empirical.to_gaussian_joint().pathwise(
            u=empirical.u_samples,
            v=empirical.v_samples,
            whitened_noise=jax.random.normal(key, (J, N)),
            y=jnp.asarray(y),
            noise_cov=noise_cov,
        )
    )
    np.testing.assert_allclose(
        shipped, general, rtol=0, atol=1e3 * EPS * np.abs(shipped).max()
    )


def test_15_pathwise_batches_over_realizations_and_validates_them():
    J, P, N = 5, 3, 4
    U, V, R, y = _problem(J, P, N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    joint = GaussianJoint.from_samples(
        u_samples=jnp.asarray(U), v_samples=jnp.asarray(V)
    )
    args = {
        "u": jnp.asarray(U),
        "v": jnp.asarray(V),
        "whitened_noise": jnp.zeros((J, N)),
        "y": jnp.asarray(y),
        "noise_cov": noise_cov,
    }
    assert joint.pathwise(**args).shape == (J, P)
    # a single realization carries no batch axis at all
    single = {**args, "u": args["u"][0], "v": args["v"][0],
              "whitened_noise": jnp.zeros(N)}
    assert joint.pathwise(**single).shape == (P,)

    for name, bad in (
        ("u", jnp.zeros((J, P + 1))),
        ("v", jnp.zeros((J, N + 1))),
        ("whitened_noise", jnp.zeros((J, N + 1))),
    ):
        with pytest.raises(ValueError, match=f"expected {name} of core shape"):
            joint.pathwise(**{**args, name: bad})
    with pytest.raises(TypeError):
        joint.pathwise(args["u"], args["v"], args["whitened_noise"], args["y"],
                       noise_cov)


# ===========================================================================
# Section 2 -- targeted regression tests. DO NOT DELETE AS REDUNDANT.
#
# One test per class of silent failure: each produces wrong numbers, a wrong
# cost, or a corrupted joint law without raising, and each documents why a
# rule of the contract exists.
# ===========================================================================



def test_regression_the_square_root_reading_needs_a_centred_factor():
    """``transform_update`` reads updated samples off the conditioned factor's
    columns, which is why it lives on the class that holds samples and takes
    no sample argument.

    Suppose it took one. T acts on the *latent* index, so applying it to any
    other sample set of the same shape leaves the posterior mean intact —
    1^T A' = 0 and T 1 = 1 both still hold — while the covariance becomes
    A'^T (I + S S^T)^-1 A' / (J - 1), which is not the posterior's. Right
    first moment, wrong second, no nan, nothing raised: the failure a value
    check could only catch in debug mode, and which holding the reading here
    makes unrepresentable.
    """
    J, P, N = 8, 3, 4
    U, V, R, y = _problem(J, P, N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    empirical = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    posterior = empirical.to_gaussian_joint().condition(jnp.asarray(y), noise_cov)
    m_post = np.asarray(posterior.mean)
    C_post = np.asarray(posterior.cov.to_dense())

    T = np.asarray(
        sqrt_transform(jnp.asarray(_scaled_whitened_anomalies(V, noise_cov, J)))
    )
    # the shipped reading, for comparison: exact in both moments
    shipped_mean, shipped_cov = _sample_moments(
        np.asarray(empirical.transform_update(jnp.asarray(y), noise_cov))
    )
    scale = max(np.abs(m_post).max(), np.abs(C_post).max())
    np.testing.assert_allclose(shipped_mean, m_post, rtol=0, atol=1e3 * EPS * scale)
    np.testing.assert_allclose(shipped_cov, C_post, rtol=0, atol=1e3 * EPS * scale)

    # the same transform against an unrelated sample set of the same shape
    other = RNG.normal(size=(J, P))
    wrong = m_post + T @ (other - other.mean(axis=0))
    wrong_mean, wrong_cov = _sample_moments(wrong)

    assert np.all(np.isfinite(wrong)), "the failure is a wrong number, not a nan"
    np.testing.assert_allclose(wrong_mean, m_post, rtol=0, atol=1e3 * EPS * scale)
    cov_error = np.abs(wrong_cov - C_post).max()
    assert cov_error > 0.1 * np.abs(C_post).max(), (
        f"covariance error {cov_error:.2e} against a covariance of scale "
        f"{np.abs(C_post).max():.2e} -- the regression needs the two to differ"
    )


def test_regression_independently_chosen_factors_lose_the_cross_covariance():
    """``from_factors`` cannot check that its two row blocks come from one
    factorization of the joint covariance, and the failure is silent.

    Factorizing C_uu and C_vv separately gives a pair that still defines a
    valid joint Gaussian — any [F_u; F_v] does, since F F^T is PSD — with the
    intended marginals and a cross-covariance that is whatever those two
    factorizations happen to imply. Conditioning then returns a finite,
    plausible posterior for a different joint. ``from_linear_map`` and
    ``from_samples`` build coherent pairs by their arithmetic, which is why
    they are the documented routes.

    The latent-width check catches this whenever P != N, since two separate
    square factorizations have widths P and N. It cannot at P == N, which is
    the case exercised here: the hazard is real but narrow, and the width
    check is what narrows it.
    """
    P = N = 4
    rng = np.random.default_rng(41)
    F = rng.normal(size=(P + N, P))
    C = F @ F.T
    Cuu, Cuv, Cvv = C[:P, :P], C[:P, P:], C[P:, P:]
    R = _psd(N)
    y = rng.normal(size=N)
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    zeros_u, zeros_v = np.zeros(P), np.zeros(N)

    coherent = GaussianJoint.from_factors(
        u_mean=jnp.asarray(zeros_u),
        v_mean=jnp.asarray(zeros_v),
        u_factor=jnp.asarray(F[:P]),
        v_factor=jnp.asarray(F[P:]),
    )
    independent = GaussianJoint.from_factors(
        u_mean=jnp.asarray(zeros_u),
        v_mean=jnp.asarray(zeros_v),
        u_factor=jnp.asarray(np.linalg.cholesky(Cuu)),
        v_factor=jnp.asarray(np.linalg.cholesky(Cvv)),
    )

    # both are accepted, and both have the intended marginals
    for joint in (coherent, independent):
        np.testing.assert_allclose(
            joint.u_marginal.cov.to_dense(), Cuu, rtol=0, atol=1e3 * EPS * np.abs(C).max()
        )
        np.testing.assert_allclose(
            joint.v_marginal.cov.to_dense(), Cvv, rtol=0, atol=1e3 * EPS * np.abs(C).max()
        )

    assert coherent.latent_dim == P and independent.latent_dim == P

    # they disagree on the cross block, and so on the posterior
    Fu_ind = np.asarray(independent.u_factor.to_dense())
    Fv_ind = np.asarray(independent.v_factor.to_dense())
    assert np.abs(Fu_ind @ Fv_ind.T - Cuv).max() > 0.1 * np.abs(Cuv).max()

    m_ref, _ = _dense_conditional(zeros_u, zeros_v, Cuu, Cuv, Cvv, R, y)
    got_coherent = np.asarray(coherent.condition(jnp.asarray(y), noise_cov).mean)
    got_independent = np.asarray(independent.condition(jnp.asarray(y), noise_cov).mean)
    np.testing.assert_allclose(
        got_coherent, m_ref, rtol=0, atol=1e3 * EPS * np.abs(m_ref).max()
    )
    assert np.all(np.isfinite(got_independent))
    assert np.abs(got_independent - m_ref).max() > 0.1 * np.abs(m_ref).max()


def test_regression_transporting_arbitrary_realizations_costs_one_whitening_each():
    """The two pathwise routes have genuinely different costs, and the cheaper
    one is only available where the realizations are the joint's own factor.

    ``pathwise_update`` gets its per-sample residuals from the whitened factor,
    W(y - v_j) = W(y - v_bar) - sqrt(J-1) S_j, for J + 1 applications of W.
    ``pathwise`` cannot: its ``v`` is arbitrary data, so it whitens k columns
    for S and one residual per realization. Both are correct; only a count
    tells them apart, and for a dense whitener the difference is the dominant
    O(J N^2) term.
    """
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    empirical = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))

    counter = CountingWhitenPSD.counting(R)
    empirical.pathwise_update(jax.random.key(0), jnp.asarray(y), counter)
    assert sum(object.__getattribute__(counter, "log")) == J + 1

    counter = CountingWhitenPSD.counting(R)
    empirical.to_gaussian_joint().pathwise(
        u=empirical.u_samples,
        v=empirical.v_samples,
        whitened_noise=jnp.zeros((J, N)),
        y=jnp.asarray(y),
        noise_cov=counter,
    )
    # k = J columns for S, plus one residual per realization
    assert sum(object.__getattribute__(counter, "log")) == 2 * J


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
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
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


def test_regression_uncentered_transform_shifts_the_ensemble_mean():
    """T 1 = 1 holds only for mean-centered s, and the update depends on it.

    Building s from raw samples rather than anomalies still produces a
    perfectly valid (I + s s^T)^-1/2 — no exception, no shape error — but it no
    longer fixes the ones vector, so the transformed anomalies stop summing to
    zero and the posterior ensemble's mean is silently shifted away from the
    posterior mean.
    """
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    V = V + 20.0  # a large mean makes the uncentered error unmistakable
    noise_cov = DensePSD.from_matrix(jnp.asarray(R))
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    m_post, _ = _dense_posterior(U, V, R, y)

    # the layer centers, so its transform fixes the ones vector and the
    # updated ensemble's mean is the posterior mean
    centered = jnp.asarray(_scaled_whitened_anomalies(V, noise_cov, J))
    np.testing.assert_allclose(
        sqrt_transform(centered) @ jnp.ones(J), jnp.ones(J), rtol=0, atol=1e3 * EPS
    )
    members = np.asarray(joint.transform_update(jnp.asarray(y), noise_cov))
    np.testing.assert_allclose(
        members.mean(axis=0), m_post, rtol=0, atol=1e3 * EPS * np.abs(m_post).max()
    )

    # the uncentered version does not, and shifts the mean
    W = _recovered_whitener(noise_cov, N)
    uncentered = jnp.asarray(V @ W.T / np.sqrt(J - 1))
    T_bad = sqrt_transform(uncentered)
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

    # and so does the float-generic degeneracy of mean-centering: at N >= J the
    # smallest singular value is ~1e-16, which is not an exact tie
    A = RNG.normal(size=(5, 7))
    centered = jnp.asarray((A - A.mean(axis=0)) / np.sqrt(4))
    sigma = np.linalg.svd(np.asarray(centered), compute_uv=False)
    assert sigma[-1] < 1e-14 and sigma[-1] != 0.0
    for fn in (
        lambda s: jnp.sum(sqrt_transform(s)),
        lambda s: jnp.sum(gain_weights(s, jnp.ones(7))),
    ):
        assert bool(jnp.all(jnp.isfinite(jax.grad(fn)(centered))))


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
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    diagonal = jnp.asarray([1.0, 0.0, 2.0, 3.0])  # a zero variance
    singular = PSDDiagonal(diagonal)

    assert singular.supports("whiten")  # support is static, and says nothing
    assert bool(jnp.isinf(singular.whiten(jnp.ones(N))).any())

    for result in (
        joint.pathwise_update(jax.random.key(0), jnp.asarray(y), singular),
        joint.transform_update(jnp.asarray(y), singular),
        joint.to_gaussian_joint().condition(jnp.asarray(y), singular).mean,
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
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    singular = PSDDiagonal(jnp.asarray([1.0, 0.0, 2.0, 3.0]))

    with debug_checks(True):
        for call in (
            lambda: joint.pathwise_update(jax.random.key(0), jnp.asarray(y), singular),
            lambda: joint.transform_update(jnp.asarray(y), singular),
            lambda: joint.to_gaussian_joint().condition(jnp.asarray(y), singular),
        ):
            with pytest.raises(ValueError, match="must be finite"):
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
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
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

    Whitening is linear, so it commutes with centering and with subtracting y:
    one call on the k + 1 stacked columns [F_v | y - v_bar] yields every
    whitened quantity the kernel needs, and at k = J that is J + 1. For a
    dense whitener those applications are the dominant
    O(J N^2) term, so a second call is a silent 2x — it produces correct
    numbers, which is why only a count catches it.
    """
    J, P, N = 6, 3, 4
    U, V, R, y = _problem(J, P, N)
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))

    for name, call in (
        ("pathwise_update", lambda cov: joint.pathwise_update(
            jax.random.key(0), jnp.asarray(y), cov)),
        ("transform_update", lambda cov: joint.transform_update(jnp.asarray(y), cov)),
        (
            "condition",
            lambda cov: joint.to_gaussian_joint().condition(jnp.asarray(y), cov),
        ),
    ):
        noise_cov = CountingWhitenPSD.counting(R)
        call(noise_cov)
        applied = sum(object.__getattribute__(noise_cov, "log"))
        assert applied == J + 1, f"{name} whitened {applied} vectors, expected {J + 1}"

    # and the numbers are unchanged by the grouping: both the dense reference
    # and the implementation center before whitening -- the implementation
    # structurally, since the joint factor is centered at construction
    plain = DensePSD.from_matrix(jnp.asarray(R))
    m_post, _ = _dense_posterior(U, V, R, y)
    np.testing.assert_allclose(
        joint.transform_update(jnp.asarray(y), plain).mean(axis=0),
        m_post,
        rtol=0,
        atol=1e3 * EPS * np.abs(m_post).max(),
    )


def test_regression_anomalies_are_centered_before_whitening():
    """The kernel must center before whitening, and only accuracy shows it.

    Whitening is linear, so the two groupings give the same S in exact
    arithmetic — but not the same accuracy. Centering *whitened* predictions
    makes the cancellation ratio ||W v_bar|| / ||W a_j|| rather than
    ||v_bar|| / ||a_j||, so the error grows with kappa(W) = sqrt(kappa(R))
    when the prediction mean lies along a precise direction of the noise.

    Both orders whiten J + 1 vectors, so the cost regression test below passes
    either way, and every moment-exactness obligation passes either way too
    because their fixtures are well scaled. This asserts against an exact
    rational reference in the regime that separates them.
    """
    J, P, N = 5, 3, 4
    kappa = 1e10
    rng = np.random.default_rng(17)
    Q, _ = np.linalg.qr(rng.normal(size=(N, N)))
    Rm = Q @ np.diag(np.geomspace(1.0, 1.0 / kappa, N)) @ Q.T
    Rm = (Rm + Rm.T) / 2
    A = rng.normal(size=(J, N))
    V = A - A.mean(axis=0) + 1e10 * Q[:, -1]  # mean along R's most precise direction
    U = rng.normal(size=(J, P))
    noise_cov = DensePSD.from_matrix(jnp.asarray(Rm))
    y = jnp.asarray(rng.normal(size=N))

    exact = _exact_posterior_mean(U, V, Rm, np.asarray(y))
    scale = max(1.0, np.abs(exact).max())

    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    shipped = np.asarray(joint.to_gaussian_joint().condition(y, noise_cov).mean)

    # the same kernel, whitening before centering -- the reverted grouping
    whitened_v = np.asarray(noise_cov.whiten(jnp.asarray(V)))
    s_bad = (whitened_v - whitened_v.mean(axis=0)) / np.sqrt(J - 1)
    r_bad = np.asarray(noise_cov.whiten(y)) - whitened_v.mean(axis=0)
    Au = U - U.mean(axis=0)
    reverted = U.mean(axis=0) + Au.T @ np.asarray(
        gain_weights(jnp.asarray(s_bad), jnp.asarray(r_bad))
    ) / np.sqrt(J - 1)

    shipped_err = np.abs(shipped - exact).max() / scale
    reverted_err = np.abs(reverted - exact).max() / scale

    # measured: shipped 1.0e-12, reverted 9.0e-06 -- nine orders apart
    assert shipped_err < 1e-9, f"shipped relative error {shipped_err:.2e}"
    assert reverted_err > 1e-7, f"reverted relative error only {reverted_err:.2e}"
    assert reverted_err > 1e3 * shipped_err


def test_regression_a_collapsed_ensemble_is_exact_at_any_magnitude():
    """Anomalies of identical members must be *exactly* zero.

    jnp.mean of J bit-identical rows sums and divides, which does not in
    general return the value it was given, so a plain subtraction leaves
    spurious anomalies of about eps*|v_bar|. The gain amplifies those into a
    wrong, finite, nan-free update once the members are large — the failure
    obligation 8 cannot see, because it prescribes an exactly representable
    value whose mean happens to round back to itself.
    """
    J, P, N = 10, 3, 2
    U = np.random.default_rng(0).normal(size=(J, P))
    noise_cov = DensePSD.from_matrix(jnp.eye(N))
    y = jnp.zeros(N)

    for magnitude in (1.0, 0.1, 6.02e23, 1e150):
        V = np.full((J, N), magnitude)
        joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))

        np.testing.assert_array_equal(joint.v_anomalies, jnp.zeros((J, N)))
        np.testing.assert_array_equal(
            joint.pathwise_update(jax.random.key(0), y, noise_cov), jnp.asarray(U)
        )
        np.testing.assert_allclose(
            joint.transform_update(y, noise_cov), U, rtol=0, atol=1e3 * EPS
        )


def test_regression_debug_checks_survive_a_trace_over_closed_over_arrays():
    """Tier-4 checks must be skipped inside a trace, not crash in it.

    jnp.isfinite applied to a *concrete* array while a trace is live is staged
    into that trace, so the check's bool conversion sees a tracer. Guarding on
    whether the operand is a tracer does not catch it: the operand is
    concrete. This is the documented driver shape — y and noise_cov closed
    over, only the ensemble traced — and it used to raise
    TracerBoolConversionError from inside a debug check, cache-dependently.
    """
    J, P, N = 4, 3, 2
    U = np.random.default_rng(0).normal(size=(J, P))
    V = np.random.default_rng(1).normal(size=(J, N))
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))
    y = jnp.zeros(N)
    noise_cov = PSDDiagonal(jnp.ones(N))
    eager = np.asarray(joint.transform_update(y, noise_cov))

    with debug_checks(True):
        # every array closed over, nothing traced
        np.testing.assert_allclose(
            jax.jit(lambda: joint.transform_update(y, noise_cov))(), eager
        )
        np.testing.assert_allclose(
            jax.jit(lambda: joint.to_gaussian_joint().condition(y, noise_cov).mean)(),
            np.asarray(joint.to_gaussian_joint().condition(y, noise_cov).mean),
        )
        jax.jit(lambda: joint.pathwise_update(jax.random.key(0), y, noise_cov))()
        # and a scan, which is how a tempering driver runs
        def step(carry, _):
            joint_t = EmpiricalJoint(u_samples=carry, v_samples=jnp.asarray(V))
            return joint_t.transform_update(y, noise_cov), None

        out, _ = jax.lax.scan(step, jnp.asarray(U), None, length=3)
        assert bool(jnp.all(jnp.isfinite(out)))
        # constructing an operator from a closed-over concrete array, too
        A = jnp.ones(N)
        jax.jit(lambda: PSDDiagonal(A).diag())()


def test_regression_check_order_with_two_simultaneous_violations():
    """The specified check order is only observable when two things are wrong.

    Supplying one bad argument at a time, as the validation tests do, pins no
    ordering at all: three independent reorderings of the guard sequence pass
    such tests. Each case below has two violations, so exactly one error can
    win, and which one is the contract's ordering rule.
    """
    J, P, N = 5, 3, 4
    joint = EmpiricalJoint(
        u_samples=jnp.asarray(RNG.normal(size=(J, P))),
        v_samples=jnp.asarray(RNG.normal(size=(J, N))),
    )
    bad_y = jnp.zeros(N + 3)
    good_y = jnp.zeros(N)

    # capability before tier-3 operand and side checks
    no_whiten = PSDLowRank(jnp.asarray(RNG.normal(size=(N, N))))
    with pytest.raises(UnsupportedOpError):
        joint.transform_update(bad_y, no_whiten)
    wrong_side_no_whiten = PSDLowRank(jnp.asarray(RNG.normal(size=(N + 1, N + 1))))
    with pytest.raises(UnsupportedOpError):
        joint.transform_update(good_y, wrong_side_no_whiten)

    # noise_cov's type and family checks come *ahead* of the capability check,
    # the one forced exception: supports() cannot be asked of a non-operator
    with pytest.raises(TypeError, match="PSDLinOp"):
        joint.transform_update(bad_y, jnp.eye(N))
    family = _stacked(DensePSD.from_matrix(jnp.asarray(_psd(N))), reps=2)
    with pytest.raises(ValueError, match="vmapped family"):
        joint.transform_update(bad_y, family)

    # the side check stays behind the capability check but ahead of y
    with pytest.raises(ValueError, match="dimension"):
        joint.transform_update(bad_y, DensePSD.from_matrix(jnp.asarray(_psd(N + 1))))

    # and the family guard on the joint precedes everything
    joint_family = _stacked(joint, reps=2)
    with pytest.raises(ValueError, match="vmapped family"):
        joint_family.transform_update(bad_y, jnp.eye(N))

    # log_density's two capability checks, in the order it names them
    whiten_only = WhitenOnlyPSD(jnp.linalg.cholesky(jnp.asarray(_psd(3))))
    with pytest.raises(UnsupportedOpError, match="logdet"):
        Gaussian(jnp.zeros(3), whiten_only).log_density(jnp.zeros(99))
    with pytest.raises(UnsupportedOpError, match="whiten"):
        Gaussian(jnp.zeros(3), PSDLowRank(jnp.eye(3))).log_density(jnp.zeros(99))


def test_error_messages_name_the_object_the_method_and_the_offending_value():
    """The message obligations are normative, and a fragment match does not
    pin them: a terse message losing the repr, the method and the shape passes
    every other validation test."""
    J, P, N = 5, 3, 4
    joint = EmpiricalJoint(
        u_samples=jnp.asarray(RNG.normal(size=(J, P))),
        v_samples=jnp.asarray(RNG.normal(size=(J, N))),
    )
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(N)))

    with pytest.raises(ValueError) as excinfo:
        joint.transform_update(jnp.zeros((2, N)), noise_cov)
    message = str(excinfo.value)
    assert repr(joint) in message          # the object
    assert "transform_update" in message   # the method
    assert f"({N},)" in message            # the expectation
    assert "(2, 4)" in message             # the offending shape

    gaussian = Gaussian(jnp.zeros(3), DensePSD.from_matrix(jnp.asarray(_psd(3))))
    with pytest.raises(ValueError) as excinfo:
        gaussian.log_density(jnp.zeros((2, 7)))
    message = str(excinfo.value)
    assert repr(gaussian) in message
    assert "log_density" in message
    assert "(2, 7)" in message

    with pytest.raises(ValueError) as excinfo:
        gain_weights(jnp.zeros((3, 4)), jnp.zeros(9))
    assert "gain_weights" in str(excinfo.value) and "(9,)" in str(excinfo.value)


def test_derived_moment_properties_have_the_documented_values():
    """v_mean and v_anomalies are public and nothing in the kernel reads them,
    so without this they have no value coverage at all: replacing v_mean with
    zeros passes the whole suite."""
    J, P, N = 6, 3, 4
    U, V, _, _ = _problem(J, P, N)
    joint = EmpiricalJoint(u_samples=jnp.asarray(U), v_samples=jnp.asarray(V))

    for got, want in (
        (joint.u_mean, U.mean(axis=0)),
        (joint.v_mean, V.mean(axis=0)),
        (joint.u_anomalies, U - U.mean(axis=0)),
        (joint.v_anomalies, V - V.mean(axis=0)),
    ):
        np.testing.assert_allclose(
            got, want, rtol=0, atol=1e3 * EPS * max(1.0, np.abs(want).max())
        )
    # anomalies sum to zero, and mean + anomalies recovers the samples
    np.testing.assert_allclose(joint.v_anomalies.sum(axis=0), 0.0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        joint.u_mean + joint.u_anomalies, U, rtol=0, atol=1e3 * EPS
    )


def test_gaussian_batch_shape_includes_its_covariance_contribution():
    """A reconstruction that batches only the covariance is still a family.

    Stacking every leaf cannot see this: `mean` alone already reports the
    batch, so ignoring the covariance's contribution passes.
    """
    n = 3
    gaussian = Gaussian(jnp.zeros(n), DensePSD.from_matrix(jnp.asarray(_psd(n))))
    leaves, treedef = jax.tree_util.tree_flatten(gaussian)
    mean_leaf, cov_leaf = leaves

    cov_only = jax.tree_util.tree_unflatten(
        treedef, [mean_leaf, jnp.stack([cov_leaf] * 4)]
    )
    assert cov_only.batch_shape == (4,)
    assert cov_only.n == n
    with pytest.raises(ValueError, match="vmapped family"):
        cov_only.log_density(jnp.zeros(n))

    # incompatible contributions are diagnosed at the property
    mismatched = jax.tree_util.tree_unflatten(
        treedef, [jnp.stack([mean_leaf] * 2), jnp.stack([cov_leaf] * 5)]
    )
    with pytest.raises(ValueError, match="do not broadcast"):
        _ = mismatched.batch_shape


def test_gaussian_repr_never_raises_on_unreadable_leaves():
    """The never-raises rule is per class, and only EmpiricalJoint was covered."""
    treedef = jax.tree_util.tree_structure(
        Gaussian(jnp.zeros(2), DensePSD.from_matrix(jnp.eye(2)))
    )
    broken = jax.tree_util.tree_unflatten(treedef, [object(), object()])
    assert repr(broken) == "<Gaussian (unprintable leaves)>"


def test_non_array_fields_are_rejected_at_construction():
    """Tier 2 is unconditional, so a field with no shape is an error.

    Waving it through silences every check that follows — including those on
    the other field — and yields an object whose every accessor raises
    AttributeError instead of a constructor ValueError.
    """
    cov = DensePSD.from_matrix(jnp.asarray(_psd(3)))
    with pytest.raises(TypeError, match="no shape to check"):
        Gaussian([0.0, 0.0, 0.0], cov)
    with pytest.raises(TypeError, match="no shape to check"):
        EmpiricalJoint(u_samples=[[1.0, 2.0], [3.0, 4.0]], v_samples=jnp.zeros((2, 2)))
    with pytest.raises(TypeError, match="no shape to check"):
        EmpiricalJoint(u_samples=jnp.zeros((2, 2)), v_samples=[[1.0], [2.0]])

    # NumPy arrays have a shape and are accepted; the checks then apply
    Gaussian(np.zeros(3), cov)
    with pytest.raises(ValueError, match="disagrees"):
        Gaussian(np.zeros(4), cov)
    with pytest.raises(ValueError, match="at least 2 samples"):
        EmpiricalJoint(u_samples=np.zeros((1, 2)), v_samples=np.zeros((1, 2)))


def test_primitives_coerce_their_matrix_operand():
    """`s` is coerced like `b`, so a nested list raises the contract's
    ValueError on a shape violation rather than AttributeError on ndim."""
    np.testing.assert_allclose(
        gain_weights([[1.0, 2.0], [3.0, 4.0]], jnp.ones(2)),
        gain_weights(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), jnp.ones(2)),
    )
    assert sqrt_transform([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
    with pytest.raises(ValueError, match="expected s of shape"):
        gain_weights([1.0, 2.0], jnp.ones(2))
    with pytest.raises(ValueError, match="expected s of shape"):
        sqrt_transform([1.0, 2.0])


def test_the_transform_does_not_promote_the_dtype():
    """jnp.eye defaults to float64, which silently upcast an otherwise-float32
    pipeline and made `condition` return a Gaussian whose mean and covariance
    factor had different dtypes — enough to break a lax.scan carry."""
    J, P, N = 5, 3, 4
    U = jnp.asarray(RNG.normal(size=(J, P)), dtype=jnp.float32)
    V = jnp.asarray(RNG.normal(size=(J, N)), dtype=jnp.float32)
    s = jnp.asarray(RNG.normal(size=(J, N)), dtype=jnp.float32)
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(N), dtype=jnp.float32))
    y = jnp.asarray(RNG.normal(size=N), dtype=jnp.float32)

    assert sqrt_transform(s).dtype == jnp.float32
    joint = EmpiricalJoint(u_samples=U, v_samples=V)
    assert joint.transform_update(y, noise_cov).dtype == jnp.float32
    posterior = joint.to_gaussian_joint().condition(y, noise_cov)
    assert posterior.mean.dtype == posterior.cov.F.dtype == jnp.float32

    # float64 stays float64, which is what the package actually runs on
    joint64 = EmpiricalJoint(
        u_samples=jnp.asarray(RNG.normal(size=(J, P))),
        v_samples=jnp.asarray(RNG.normal(size=(J, N))),
    )
    cov64 = DensePSD.from_matrix(jnp.asarray(_psd(N)))
    assert joint64.transform_update(jnp.zeros(N), cov64).dtype == jnp.float64


def test_empirical_joint_fields_are_keyword_only():
    """The two fields are same-rank arrays agreeing on the member axis, so a
    swap is shape-valid whenever P == N and no check can catch it.

    Verified below: with P == N both orders would be accepted and would give
    different, finite, plausible answers. Naming them is the only defence, so
    positional construction is rejected outright.
    """
    U = jnp.asarray(RNG.normal(size=(5, 3)))
    V = jnp.asarray(RNG.normal(size=(5, 3)))  # P == N: a swap is undetectable

    with pytest.raises(TypeError, match="positional"):
        EmpiricalJoint(U, V)

    joint = EmpiricalJoint(u_samples=U, v_samples=V)
    swapped = EmpiricalJoint(u_samples=V, v_samples=U)
    noise_cov = DensePSD.from_matrix(jnp.asarray(_psd(3)))
    y = jnp.asarray(RNG.normal(size=3))

    # both are computable and neither is nan -- which is why the swap is a
    # silent failure rather than an error, and why naming is the fix
    a = joint.transform_update(y, noise_cov)
    b = swapped.transform_update(y, noise_cov)
    assert bool(jnp.all(jnp.isfinite(a))) and bool(jnp.all(jnp.isfinite(b)))
    assert not np.allclose(a, b)

    # the pytree path bypasses __init__, so families are unaffected
    family = jax.vmap(lambda u, v: EmpiricalJoint(u_samples=u, v_samples=v))(
        jnp.zeros((4, 5, 3)), jnp.zeros((4, 5, 3))
    )
    assert family.batch_shape == (4,)

    # Gaussian stays positional: its two fields cannot be swapped, being an
    # array and an operator
    Gaussian(jnp.zeros(3), DensePSD.from_matrix(jnp.eye(3)))


def test_regression_anomalies_are_formed_over_the_member_axis_when_batched():
    """The centring subtracts per-member means, not per-batch ones.

    ``jnp.mean(..., axis=-2)`` without ``keepdims`` drops the member axis, so
    the subtraction right-aligns against the *batch* axis. For an operand
    whose leading axis happens to equal ``J`` that broadcasts happily and
    returns finite, plausible, wrong anomalies rather than raising.
    """
    from pyeki.gauss import _centered

    for shape in [(3, 3, 4), (2, 5, 5), (4, 6, 2)]:
        x = jnp.asarray(RNG.normal(size=shape))
        got = np.asarray(_centered(x))
        want = np.asarray(x) - np.asarray(x).mean(axis=-2, keepdims=True)
        assert got.shape == want.shape
        assert np.abs(got - want).max() < 64 * EPS * np.abs(want).max()

    # The unbatched path is unchanged, and identical members still give
    # exactly zero rather than round-off.
    collapsed = jnp.full((7, 3), 6e23)
    assert np.array_equal(np.asarray(_centered(collapsed)), np.zeros((7, 3)))
