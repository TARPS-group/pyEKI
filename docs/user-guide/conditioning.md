# Gaussian conditioning

`pyeki.gauss` conditions a joint Gaussian over two blocks on a noisy
observation of the second. It provides a Gaussian distribution, a joint
Gaussian, a container for paired samples, and the operations that connect
them.

This page is about *when and why* to reach for each piece. The
{doc}`../gaussian-contract` reference page specifies exactly *what* each one
does — precise shapes, error behaviour, the conditioning mathematics — and
{doc}`../joint-factor` derives the representation they all share.

The layer knows nothing about inversion. {doc}`running-an-inversion` is where
samples become an ensemble and the two blocks become parameters and predicted
observations.

## The three objects

```python
import pyeki  # enables float64; import this before creating arrays
import jax
import jax.numpy as jnp
from pyeki.gauss import Gaussian, GaussianJoint, EmpiricalJoint
from pyeki.linalg import PSDDiagonal, DensePSD, Dense, block_diag

prior = Gaussian(jnp.zeros(12), DensePSD.from_matrix(C0))   # mean + covariance
u = prior.sample(key, 40)                       # (40, 12) samples
v = g(u)                                        # (40, N) paired values
samples = EmpiricalJoint(u_samples=u, v_samples=v)
joint = samples.to_gaussian_joint()             # the Gaussian fitted to them
```

A **`Gaussian`** is a mean vector and a covariance operator. It is what you
build a prior with, and what conditioning hands back.

A **`GaussianJoint`** is a joint Gaussian over the two blocks. It owns every
conditioning identity: the posterior, and the pathwise map. It holds the two
means and a *joint factor* — one factor of the joint covariance, split into
the row blocks that drive $u$ and $v$ from a shared latent vector — so no
covariance block of either dimension is ever formed, and a structured
covariance stays structured.

An **`EmpiricalJoint`** holds two blocks of paired samples: `u_samples`, the
block being updated, and `v_samples`, the block that is observed — row $j$ of
each belongs to the same draw. It holds the samples and nothing else, and
offers the two updates that return updated samples.

**Samples are rows.** A block is `(J, dim)`, one draw per row, which is what
`jax.vmap` produces and what the operator layer treats as a batch of vectors.
Blocks move between the two layers with no transposes. Factors are the other
way round: `(dim, k)`, as the operator layer's `factor()` returns them.

## Building a joint

Three constructors, for three situations.

| call | builds |
| --- | --- |
| `GaussianJoint.from_linear_map(prior, linear_map)` | the joint of $u$ and $Gu$ — a linear-Gaussian model |
| `GaussianJoint.from_samples(u_samples=…, v_samples=…)` | the joint fitted to paired samples' moments |
| `GaussianJoint.from_factors(…)` | a joint from row blocks you already hold |

**`from_linear_map` is the closed-form case.** Given a prior and a linear map,
conditioning the joint it builds gives the exact linear-Gaussian posterior —
useful for checking an algorithm against a problem whose answer you know:

```python
prior = Gaussian(m0, PSDDiagonal(variances))
joint = GaussianJoint.from_linear_map(prior, Dense(G))
posterior = joint.condition(y, noise_cov)          # exact, closed form
```

Both arguments are operators, so a structured prior stays structured and a
structured map is applied as one. Nothing is inverted, which is why a
*singular* prior covariance is fine here — the case a precision-form posterior
$(C_0^{-1} + G^\top R^{-1} G)^{-1}$ cannot express at all.

:::{note}
**The observation noise is not part of the joint.** `from_linear_map` builds
the joint of $u$ and $Gu$; the model $y = v + \eta$ arrives at the
`condition` call. That is what lets you condition the same joint at a
sequence of noise levels — `noise_cov / c` for a traced `c` — without
rebuilding anything.
:::

**`from_samples` fits moments.** The means are the sample means and the
covariance blocks are the empirical covariances with the $J-1$ divisor. The
result is a Gaussian *fit to* the samples, not the equal-weight point-mass
distribution of the samples themselves. `EmpiricalJoint.to_gaussian_joint()`
is the same thing, spelled from the sample side.

**`from_factors` is the escape hatch.** It takes the two row blocks directly,
and it cannot check that they came from one factorization of the joint
covariance. Factorizing $C_{uu}$ and $C_{vv}$ separately gives the marginals
you meant and a cross-covariance you did not; conditioning then answers
correctly for a different joint. Prefer the other two.

## Which operation

Four, all conditioning on the same model $y = v + \eta$ with $\eta \sim
\mathcal{N}(0, R)$:

| call | returns | use it when |
| --- | --- | --- |
| `joint.condition(y, noise_cov)` | a posterior `Gaussian` | you want the posterior itself — to sample it at any size, or for diagnostics |
| `joint.pathwise(u=…, v=…, whitened_noise=…, y=…, noise_cov=…)` | `(…, P)` transported realizations | you have realizations of the joint and want posterior draws from them |
| `samples.transform_update(y, noise_cov)` | `(J, P)` updated samples | you want the deterministic update of your samples, with no sampling noise |
| `samples.pathwise_update(key, y, noise_cov)` | `(J, P)` updated samples | you want the stochastic, perturbed-observation update of your samples |

The top two are `GaussianJoint` methods; the bottom two are `EmpiricalJoint`
methods, and both return the `u` block only. A caller that needs a matching
`v` recomputes it, so there is nothing to be gained from updating `v` here.

**Why `condition` is not a method on `EmpiricalJoint`.** Conditioning a set of
samples means conditioning a Gaussian fitted to their moments. That fit is a
modelling step, so it is written out — `samples.to_gaussian_joint().condition(...)`
— rather than hidden inside a method whose name would suggest you had
conditioned the samples themselves.

**The difference between the two updates.** `transform_update` returns a block
whose sample mean *is* the posterior mean and whose sample covariance *is* the
posterior covariance — exactly, not asymptotically in $J$. It perturbs nothing
and needs no key. `pathwise_update` draws one perturbation per sample; its
moments are unbiased estimators of the same posterior moments, but they
fluctuate, the covariance at the usual $O(J^{-1/2})$ rate. In exchange it
keeps the rows statistically independent given the samples, which the
deterministic transform does not.

Neither is uniformly better, and choosing between them belongs to the
algorithm you are building rather than to this layer.

**`pathwise` versus `pathwise_update`.** They are the same map. `pathwise` is
the general form: it transports whatever realizations you hand it, including
ones the joint never saw, and takes the perturbation explicitly.
`pathwise_update` applies it to the samples the joint was fitted to, owns the
draw, and takes a shortcut those samples make available — so it whitens $J+1$
vectors where the general form whitens one per realization. Use
`pathwise_update` on your own samples; reach for `pathwise` when the
realizations come from somewhere else.

:::{note}
`pathwise_update` neither takes nor returns the perturbations, deliberately. A
perturbation is used through the whitened shortcut $W(y - v_j) -
\varepsilon_j$, and pushing that *same* $\varepsilon_j$ through `factor()` as
well — to materialize a perturbed observation for storage, say — corrupts the
joint law of the update while every marginal statistic still looks correct.
Owning both the draw and its single use is how the method rules that out.

`pathwise` does take the perturbation, and pins the representation instead:
`whitened_noise` is $\varepsilon$, a standard normal draw in whitened
coordinates, never a draw from $R$. The same rule applies — use it once, and
only there.
:::

## Noise covariances only need whitening

Every conditioning method uses `noise_cov` through exactly one operation,
`whiten`. That is not an accident of the implementation — it is the reason the
noise interface looks the way it does, and it is a promise about the
*interface*: a custom operator that implements `_whiten` and nothing else
drives every update in this layer, with no `factor`, no `solve` and no
`logdet`. Structured noise is used the same way, block by block:

```python
noise_cov = block_diag(
    PSDDiagonal(instrument_variances),   # independent errors
    DensePSD.from_matrix(correlated),    # a correlated block
)
u_next = samples.pathwise_update(key, y, noise_cov)   # whiten only
```

(The shipped operators happen to support more than `whiten` — this one solves
and has a log-determinant too. Nothing here calls them.)

It also makes a rescaled noise covariance cheap. Conditioning with $R/c$ is
the operator layer's scalar scaling:

```python
for key_t, c in scales:
    v = g(u)
    joint = EmpiricalJoint(u_samples=u, v_samples=v)
    u = joint.pathwise_update(key_t, y, noise_cov / c)
```

The scale flows through a scalar field, so a caller choosing $c$ adaptively
never re-factorizes the noise operator, however many candidates it tries.

## The posterior is low rank, and says so

`condition` returns a `Gaussian` whose covariance is a
{class}`~pyeki.linalg.PSDLowRank` holding a factor of width $k$ — never a
dense $P \times P$ matrix. That is the honest representation: a posterior
whose joint came from $J$ samples has rank at most $J - 1$, so it is genuinely
singular whenever $J - 1 < P$, which is the usual regime here.

The consequence is visible in the interface. You can sample the posterior,
because the factor is what sampling needs:

```python
posterior = joint.condition(y, noise_cov)
draws = posterior.sample(key, 500)          # works: rank-deficient or not
posterior.log_density(x)                    # raises UnsupportedOpError
```

`log_density` raises rather than returning `-inf` or a plausible wrong number,
because a rank-deficient Gaussian has no density on $\mathbb{R}^P$. If your
problem really is full rank and you want that density, say so at the call site
by densifying the covariance deliberately — see {doc}`operators` on `densify`.

The two marginals are available the same way: `joint.u_marginal` and
`joint.v_marginal` are `Gaussian`s over the two blocks, with the same low-rank
covariances. `v_marginal` is the noise-free marginal of $v$ — the observation
noise is not in it.

## Collapsed samples are no-ops, not `nan`

If the `v` anomalies collapse to zero — every row taking the same value — the
observation carries no information about `u` through these samples, and the
methods return exactly that: both updates return `u_samples` unchanged, and
`condition` returns the `u` marginal's own moments. There is no `nan`, and no
regularization parameter to tune. The gain multipliers are bounded by $1/2$
however collapsed or ill-conditioned $s$ becomes.

The direction that *does* degrade is small noise: a very large whitener, giving
a large $\sigma_{\max}$. Collapse is the numerically pristine end.

The one precondition you own is that `noise_cov` be nonsingular, which is
`whiten`'s own. A singular noise covariance cannot be detected *before the
fact*, so by default it surfaces as `nan` rather than an exception. Under
`debug_checks` every conditioning method checks what it returns, which turns
that `nan` into a `ValueError` naming the likely cause:

```python
from pyeki.linalg import debug_checks

with debug_checks(True):
    samples.transform_update(y, singular_noise)   # ValueError, not a nan block
```

Like every value-level check in pyEKI this reads array contents, so it is
skipped on tracers: it fires in eager code and in tests, and not inside a
`jit`-compiled loop. It is a debugging aid, not a guard on production runs.

## Families are `vmap`, never stored batch axes

All three classes are unbatched frozen pytrees, exactly like operators. A
family — several joints, several priors, several noise levels — is a
`jax.vmap` over the pytree, not an object with extra leading axes:

```python
build = lambda u, v: EmpiricalJoint(u_samples=u, v_samples=v)
families = jax.vmap(build)(u_batch, v_batch)          # (8, J, P), (8, J, N)
families.batch_shape                                  # (8,)
families.transform_update(y, noise_cov)               # ValueError: apply under vmap

updates = jax.vmap(lambda j, y: j.transform_update(y, noise_cov))(families, ys)
```

A family identifies itself — `batch_shape`, and a
`vmapped(EmpiricalJoint(...), batch=(8,))` repr — and refuses every operation
with a message telling you to map it. The same holds for a family of noise
operators: map it, do not pass it in directly.

The vectors these classes take are correspondingly unbatched: `y` is one
observation of shape `(N,)`. A family of updates over several observations is
`vmap` over the method. Three arguments are exceptions, following the operator
layer's batch contract instead: `Gaussian.log_density`'s evaluation point,
`pathwise`'s three realization arguments, and the primitives' operands.

## Dropping to the primitives

Everything above routes through one computation, and the two pieces of it are
public:

```python
from pyeki.gauss import gain_weights, sqrt_transform

w = gain_weights(s, b)      # (k, N), (..., N) -> (..., k)
T = sqrt_transform(s)       # (k, N) -> (k, k)
```

These are pure array functions with no divisor, no whitening and no randomness
folded in: `s` must already be the whitened factor $(W F_v)^\top$ of the
observed block, and `b` an already-whitened residual. They are the advanced
tier, for building analyses the classes do not cover — domain localization is
the planned consumer, calling `gain_weights` once per local block under
`vmap`. Because they cannot check what you pass them, the class methods are
the default interface and these are the escape hatch.

Two properties matter if you build on them. Each computes one SVD per call, so
batch your residuals into a single call rather than looping — the $J$
per-sample residuals of a stochastic update are one `(J, N)` operand. And they
are differentiable wherever the singular values of `s` are distinct and
nonzero; at exactly repeated or exactly zero singular values the SVD's
gradient is `nan` even though the functions themselves are smooth there.
