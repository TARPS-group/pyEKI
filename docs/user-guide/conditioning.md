# Gaussian conditioning

`pyeki.gauss` is the layer that turns an ensemble into an updated ensemble. It
provides a Gaussian distribution, the joint Gaussian an ensemble determines,
and the three ways to condition that joint on an observation.

This page is about *when and why* to reach for each piece. The
{doc}`../gaussian-contract` reference page specifies exactly *what* each one
does, and is the place to look for precise shapes, error behaviour and the
conditioning mathematics.

## The two objects

```python
import pyeki  # enables float64; import this before creating arrays
import jax.numpy as jnp
from pyeki.gauss import Gaussian, EnsembleJoint
from pyeki.linalg import PSDDiagonal, DensePSD

prior = Gaussian(jnp.zeros(12), DensePSD.from_matrix(C0))   # mean + covariance
u = prior.sample(key, 40)                                   # (40, 12) ensemble
v = forward(u)                                              # (40, N) predictions
joint = EnsembleJoint(u, v)                                 # the joint Gaussian
```

A `Gaussian` is a mean vector and a covariance operator. It is what you build a
prior with, and what conditioning hands back.

An `EnsembleJoint` holds two blocks of paired samples: `u_samples`, the block
being updated — the parameters — and `v_samples`, the predicted observations
from the same members. It *acts as* the joint Gaussian whose mean and
covariance match the ensemble's, so conditioning it is exact Gaussian
conditioning of that fitted Gaussian. Nothing is stored but the samples: the
means and anomalies are computed on access, and no covariance matrix of
parameter or observation dimension is ever formed.

**Samples are rows.** An ensemble is `(J, dim)`, one member per row, which is
what a `vmap`-ed forward model produces and what the operator layer treats as a
batch of vectors. Ensembles move between the two layers with no transposes.

## Which conditioning operation

Three methods, all conditioning on the same model $y = v + \eta$ with
$\eta \sim \mathcal{N}(0, R)$:

| call | returns | use it when |
| --- | --- | --- |
| `joint.pathwise_update(key, y, noise_cov)` | `(J, P)` updated members | you want the standard stochastic EKI update |
| `joint.transform_update(y, noise_cov)` | `(J, P)` updated members | you want the deterministic update, with no sampling noise |
| `joint.condition(y, noise_cov)` | a posterior `Gaussian` | you want the posterior itself — to sample it at a different size, or for diagnostics |

The two updates return parameters only. EKI re-evaluates the forward model to
get the next step's predictions, so there is nothing to be gained from updating
`v`.

**The difference between the updates.** `transform_update` returns an ensemble
whose sample mean *is* the posterior mean and whose sample covariance *is* the
posterior covariance — exactly, not asymptotically in $J$. It perturbs nothing
and needs no key. `pathwise_update` draws one perturbation per member; its
sample moments are unbiased estimators of the same posterior moments, but they
fluctuate, the covariance at the usual $O(J^{-1/2})$ rate. In exchange it keeps
members statistically independent given the ensemble, which the deterministic
transform does not.

Neither is uniformly better, and choosing between them belongs to the algorithm
you are building rather than to this layer.

:::{note}
`pathwise_update` neither takes nor returns the perturbations, deliberately. A
perturbation is used through the whitened shortcut $W(y - v_j) - \varepsilon_j$,
and pushing that *same* $\varepsilon_j$ through `factor()` as well — to
materialize a perturbed observation for storage, say — corrupts the joint law of
the update while every marginal statistic still looks correct. Owning both the
draw and its single use is how the method rules that out. If you need
perturbed observations materialized, you are building a different algorithm:
use the primitives below, draw your own $\varepsilon$, and pick one
representation for it.
:::

## Noise covariances only need whitening

Every conditioning method uses `noise_cov` through exactly one operation,
`whiten`. That is not an accident of the implementation — it is the reason the
noise interface looks the way it does. A noise operator with no cheap
factorization, no solve and no log-determinant still drives every update:

```python
noise_cov = block_diag(
    PSDDiagonal(instrument_variances),   # independent errors
    DensePSD.from_matrix(correlated),    # a correlated block
)
u_next = joint.pathwise_update(key, y, noise_cov)
```

This also makes tempering cheap. A tempered step wants $R/\delta\beta$, which
is the operator layer's scalar scaling:

```python
for key_t, dbeta in schedule:
    v = forward(u)
    u = EnsembleJoint(u, v).pathwise_update(key_t, y, noise_cov / dbeta)
```

The increment flows through a scalar field, so an adaptive schedule never
re-factorizes the noise operator, however many candidate increments it tries.

## The posterior is low rank, and says so

`condition` returns a `Gaussian` whose covariance is a
{class}`~pyeki.linalg.PSDLowRank` holding a factor of width $J$ — never a dense
$P \times P$ matrix. That is the honest representation: a posterior read off
$J$ members has rank at most $J - 1$, so it is genuinely singular whenever
$J - 1 < P$, which is the normal EKI regime.

The consequence is visible in the interface. You can sample the posterior,
because the factor is what sampling needs:

```python
posterior = joint.condition(y, noise_cov)
draws = posterior.sample(key, 500)          # works: rank-deficient or not
posterior.log_density(x)                    # raises UnsupportedOpError
```

`log_density` raises rather than returning `-inf` or a plausible wrong number,
because a rank-deficient Gaussian has no density on $\mathbb{R}^P$. If your
problem really is in the $J - 1 \ge P$ regime and you want that density, say so
at the call site by densifying the covariance deliberately — see
{doc}`operators` on `densify`.

## Degenerate ensembles are no-ops, not `nan`

If the prediction anomalies collapse to zero — every member predicting the same
thing — the observation carries no information about the parameters through this
ensemble, and the methods return exactly that: both updates return `u_samples`
unchanged, and `condition` returns the prior marginal's moments. There is no
`nan`, and no regularization parameter to tune. The gain multipliers are bounded
by $1/2$ however collapsed or ill-conditioned the ensemble becomes.

The direction that *does* degrade is small noise: a very large whitener, giving
a large $\sigma_{\max}$. Collapse is the numerically pristine end.

The one precondition you own is that `noise_cov` be nonsingular, which is
`whiten`'s own. A singular noise covariance cannot be detected cheaply, so it
surfaces as `nan` rather than an exception.

## Families are `vmap`, never stored batch axes

Both classes are unbatched frozen pytrees, exactly like operators. A family —
several joints, several priors, several noise levels — is a `jax.vmap` over the
pytree, not an object with extra leading axes:

```python
joints = jax.vmap(EnsembleJoint)(u_batch, v_batch)   # (m, J, P), (m, J, N)
joints.batch_shape                                    # (8,)
joints.transform_update(y, noise_cov)                 # ValueError: apply under vmap

updates = jax.vmap(lambda j, y: j.transform_update(y, noise_cov))(joints, ys)
```

A family identifies itself — `batch_shape`, and a
`vmapped(EnsembleJoint(...), batch=(8,))` repr — and refuses every operation
with a message telling you to map it. The same holds for a family of noise
operators: map it, do not pass it in directly.

The vectors these classes take are correspondingly unbatched: `y` is one
observation of shape `(N,)`. A family of updates over several observations is
`vmap` over the method. The exceptions are `Gaussian.log_density`, whose
evaluation point follows the operator layer's batch contract, and the two
primitives below.

## Dropping to the primitives

Everything above routes through one computation, and the two pieces of it are
public:

```python
from pyeki.gauss import gain_weights, sqrt_transform

w = gain_weights(s, b)      # (J, N), (..., N) -> (..., J)
T = sqrt_transform(s)       # (J, N) -> (J, J)
```

These are pure array functions with no divisor, no whitening and no randomness
folded in: `s` must already be the scaled whitened anomaly matrix and `b` an
already-whitened residual. They are the advanced tier, for building analyses
the classes do not cover — domain localization is the planned consumer, calling
`gain_weights` once per local block under `vmap`. Because they cannot check
what you pass them, the class methods are the default interface and these are
the escape hatch.

Two properties matter if you build on them. Each computes one SVD per call, so
batch your residuals into a single call rather than looping — the $J$ per-member
residuals of an update are one `(J, N)` operand. And they are differentiable
wherever the singular values of `s` are distinct and nonzero; at exactly
repeated or exactly zero singular values the SVD's gradient is `nan` even though
the functions themselves are smooth there.
