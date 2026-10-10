# Distributions over named blocks

`enskit.distribution` holds the probabilistic operations every ensemble Kalman
method is built from: two distributions, the crossings between them, exact
Gaussian algebra, and two maps that move particles to a conditional. Nothing
here knows about forward models, observations or time; this page is about the
operations themselves, and when to reach for each.

The {doc}`../distribution-contract` specifies exactly what each call does:
shapes, errors, the conditioning mathematics, and the accuracy you can expect.
{doc}`../joint-factor` derives the factor representation the `Gaussian` uses.

## Two distributions over named blocks

```python
import enskit  # enables float64; import this before creating arrays
import jax
import jax.numpy as jnp
from enskit.distribution import Ensemble, Gaussian
from enskit.linalg import DensePSD, PSDDiagonal

ens = Ensemble(x=x, g=g)          # particles: (J, 40) and (J, 6) arrays
prior = Gaussian.independent(x=(m0, DensePSD(C0)))
```

An **`Ensemble`** is $J$ *particles*: an empirical distribution, optionally
weighted. Each block is a `(J, d)` array whose row $j$ is particle $j$, so
`op.matvec(ens["x"])` applies an operator to every particle.

A **`Gaussian`** is a joint Gaussian over blocks, held as a mean, a row of one
*shared factor*, and an *independent term* per block:

$$
x_b = m_b + F_b\,\xi + e_b, \qquad \xi \sim \mathcal N(0, I_k), \qquad
e_b \sim \mathcal N(0, D_b).
$$

The shared latent vector $\xi$ is what correlates the blocks, so every
cross-covariance $F_aF_b^\top$ is consistent by construction. The independent
terms keep their own structure: a noise covariance that is only ever whitened,
a diagonal or Kronecker prior that is never densified.

**Blocks have names**, so no code indexes into a concatenated vector. Every
argument that is a set of block values accepts keywords, a mapping, or both:
`g.condition(y=y_obs)` and `g.condition({"y": y_obs})` are the same call. Use
the mapping for a name that is not a Python identifier, or that collides with
a parameter such as `key`.

## From particles to a Gaussian: `project`

```python
approx = ens.project()            # an EnsembleGaussian: k = J, no terms
```

Conditioning particles means conditioning a Gaussian fitted to them, and
`project()` is where that fit happens. It is the moment-matching Gaussian, with
the ensemble's mean and covariance jointly over all blocks (with the divisor
below), and it is always written by you, at the call site: there is no
`condition` on an `Ensemble`.

For an unweighted ensemble the result is an **`EnsembleGaussian`**: its factor
row is $F_b = A_b^\top/\sqrt{\delta}$ for the anomalies $A_b$, so latent
coordinate $j$ belongs to particle $j$. That correspondence is what lets it do
two things no other Gaussian can: read its particles back out
(`realize_particles()`) and move them as a set (`square_root_map`). Every
operation that keeps the latent space keeps the type. A weighted ensemble
projects to a plain `Gaussian`: dividing out $\sqrt{w_j}$ to read particles
back would be meaningless at a zero weight.

(guide-divisor)=
### The divisor

Every covariance computed from particles, by `cov`, `project` or
`maps.statistical_linearization`, divides by the same thing:

| | unweighted | weighted |
| --- | --- | --- |
| `unbiased=True`, the default | $J - 1$ | $1 - \sum_j w_j^2$ |
| `unbiased=False` | $J$ | $1$ |

The default treats the particles as samples, and gives an unbiased estimate
of their distribution's covariance; it is what every update and algorithm in
EnsKit uses. `unbiased=False` gives the covariance of the empirical
distribution $\hat p = \sum_j w_j\delta_{x_j}$ itself. Use it when the
particles are not samples but the points of a deterministic rule, such as
a quadrature or sigma-point rule with nonnegative weights, whose weights
reproduce a mean and covariance exactly:

```python
approx = ens.project(unbiased=False)   # an EnsembleGaussian, divisor J
assert approx.divisor == ens.n_particles
```

An `EnsembleGaussian` remembers its divisor $\delta$, and every operation
that reads particles out of it uses it. So `realize_particles()` returns the
particles whichever divisor projected them, and conditioning moves them
consistently.

## Adding known noise: `add_noise` and `absorb`

```python
approx = ens.project().add_noise(g=R)   # g now has the independent term R
```

`add_noise` adds independent noise to a block in place. On a block with no
independent term it just sets one, and the latent space is unchanged: the
result is still an `EnsembleGaussian`, and `R` is used afterward only through
what later operations ask of it: conditioning only whitens it. This is the cheap, exact way to account for known noise: adding the
covariance after the fit, rather than sampling noise into the particles first,
gives a lower-variance estimate of the same joint.

On a block that already has a term, `add_noise` first moves the old term into
the shared factor, `absorb`, which needs that term's `factor()` and grows the
latent space. `absorb` is also what to call before a map of block $b$ that
must stay correlated with $b$. When repeated absorbing makes the latent width
grow without bound, `compress()` re-factors it to at most the dimension of the
blocks with factor rows.

## Conditioning

```python
post = approx.condition(g=y_obs)          # a Gaussian over the other blocks
post.mean("x"), post.cov("x")             # a vector, an operator
```

`condition` returns the exact conditional distribution, of the same kind and
on the same latent space. Two cases, decided by the given blocks:

- **Noisy values**, every given block has an independent term: the usual case.
  The result's factor row is $F_xT$ with $T = (I + SS^\top)^{-1/2}$ held as an
  operator, so a structured row stays structured. One whitened factor
  $S = (WF_c)^\top$ and one singular value decomposition serve the mean and
  the factor.
- **Exact values**, no given block has one: a QR route that materializes the
  targets' rows, meant for factors already held as arrays. It needs the given
  rows to have full rank, so no more given coordinates than $J - 1$ on an
  `EnsembleGaussian`.

`log_density(...)` is the density of the marginal over the named blocks, at a
batch of values. It is the evidence of values under a joint, and the objective
for tuning hyperparameters: its first derivatives are finite even at a
collapsed ensemble.

## Regression: the conditional as an affine function

```python
reg = approx.regression("g", given="x")
reg.coefficients["x"], reg.intercept, reg.residual_cov   # A, c, Omega
```

`condition` answers "what is $g$ given *this* value of $x$". `regression`
answers it for every value at once: the conditional of a Gaussian is

$$
g \mid x \sim \mathcal N(Ax + c,\ \Omega), \qquad
A = C_{gx}C_{xx}^{-1}, \qquad c = m_g - Am_x, \qquad
\Omega = C_{gg} - AC_{xg},
$$

and `regression` returns $A$ (one operator per given block), $c$ and
$\Omega$. On an ensemble's projection this is the least-squares regression of
the particles' $g$ on their $x$. Reach for it when you want the relationship
itself rather than a conditional at one value: how sensitive one block is
to another across the particles. The maps layer packages it as a map, {func}`~enskit.maps.statistical_linearization`
({doc}`maps`).

The given blocks' structure picks the computation, as it does for
`condition`:

- **With independent terms** on the given blocks it is ridge regression:
  `ens.project().add_noise(x=Lam).regression("g", given="x")` gives
  $A = \hat C_{gx}(\hat C_{xx} + \Lambda)^{-1}$.
- **Without them**, and with at most $J - 1$ given coordinates, it is the
  unique least-squares fit.
- **With more given coordinates than $J - 1$**, every $A$ that maps the
  given blocks' anomalies to the target's fits exactly, and `regression`
  raises unless you pass `min_norm=True`, which picks the one of least
  Frobenius norm. That choice depends on your coordinates: rescale a block
  and a different $A$ has least norm. A ridge term $\varepsilon M$ with
  small $\varepsilon$ picks the least $\operatorname{tr}(AMA^\top)$ instead.

The given rows' rank is assumed, not measured, outside
`enskit.linalg.debug_checks()`: duplicated particles (after resampling), a
coordinate that is constant across particles, or an ensemble already
conditioned exactly on other blocks give wrong coefficients silently. Check
in debug mode, or add a ridge term, when that can happen.

## Moving particles: the two conditional maps

`condition` returns a distribution. To get *particles* of the conditional,
there are two maps, built once from the block names and called with values:

| | `conditional_map(given)` | `square_root_map(given)` |
| --- | --- | --- |
| returns | `MatheronMap` | `SquareRootMap` |
| moves | any samples of the joint, one at a time | the particles the `EnsembleGaussian` was built from, as a set |
| randomness | a perturbation per sample, when given a key | none, unless a target has an independent term |
| output moments, on the joint's own particles | unbiased for the conditional's | equal to the conditional's |
| update it implements | the stochastic ensemble Kalman update (Burgers et al., 1998) | the symmetric square-root update (Bishop et al., 2001; Hunt et al., 2007) |

**`MatheronMap` is pointwise**: $x \mapsto x + K(y^* - y)$, Matheron's rule
(Journel & Huijbregts, 1978), called pathwise conditioning in machine learning
(Wilson et al., 2021). Applied to joint samples that include the given blocks'
noise, it returns exact samples of the conditional. Its `key` says what the
samples hold: without one they are taken to carry their noise; with one they
are taken to be noise-free, and the map draws that noise itself, in whitened
coordinates, so the noise covariance is only ever whitened.

```python
cmap = approx.conditional_map("g")
moved = cmap(ens, g=y_obs, key=key)                    # 2J whitened vectors
w = cmap.particle_coefficients(g=y_obs, key=key)       # J + 1 whitened vectors
x_new = ens["x"] + approx.factor("x").matvec(w)
```

On the particles the Gaussian was projected from, `particle_coefficients` reads
their whitened residuals off $S$ instead of whitening each one, which halves
the dominant cost for a dense noise covariance. It returns latent coefficients
rather than particles, so particles that should not move come back
bit-identical.

**`SquareRootMap` moves the particle set** through each particle's latent
coordinate, so it has no samples argument. It is
`condition(...).realize_particles()` with everything that does not depend on
the value computed once.

Which to use is a modeling choice that belongs to the update rule you are
building (`enskit.kalman` makes it). The square-root map reproduces the
conditional's moments exactly but recombines existing anomalies, so on a
nonlinear problem it can leave the particles strung along a few directions;
the Matheron map refills the cloud with fresh perturbations at the cost of
sampling noise.

## Weights

```python
from enskit.distribution import reweight, effective_sample_size, resample

ens = reweight(ens, log_likelihoods)           # in log space, composes
if effective_sample_size(ens) < J / 2:
    ens = resample(key, ens)                   # systematic, by default
```

Weights live in log space, so factors spanning hundreds of orders of magnitude
compose without underflow, and the effective sample size $1/\sum_j w_j^2$
(Kong et al., 1994) is computed without leaving it. `resample` draws an
unweighted ensemble, systematically (Kitagawa, 1996) or multinomially (Gordon
et al., 1993). The Kalman update rules refuse weighted ensembles, whose
projection is not aligned; resample first.

**Failed particles.** A particle whose forward-model evaluation failed comes
back as a row of `nan`, and `ens.all_finite` marks which particles are
intact. An ensemble holding failed particles is valid, but its mean and
covariance are `nan` until something is done about them. Giving them weight
zero drops them from the mean, the covariance, `project`, and any later
`resample`, without drawing anything:

```python
ens = reweight(ens, jnp.where(ens.all_finite, 0.0, -jnp.inf))
```

With debug checks on, `project` and `cov` raise if a failed particle still
has positive weight.

## Exact moments for tests and examples

`exact_moment_ensemble(key, gaussian, J)` returns particles whose sample mean
and covariance equal the Gaussian's *exactly*, jointly over every block
(Pham, 2001). Use it where a number must not depend on sampling error: a test
that a square-root update reproduces a posterior, or a figure in a tutorial.
It needs more particles than the Gaussian has sources, $k$ plus the widths of
the independent terms' factors.

## Things to know

**Families come from `jax.vmap`.** Distributions are unbatched. To condition a
family of Gaussians, or one Gaussian at a family of values, `vmap` the call. A
pytree rebuilt with stacked leaves reports its `batch_shape` and refuses every
method until it is applied under `vmap`.

**Structure is part of the pytree.** Whether a block has a factor row or an
independent term, and whether an ensemble is weighted, is tree structure. A
`lax.scan` carry must leave each iteration with the structure it entered with:
`reweight` then `resample` inside the body is fine, since the ensemble comes
back unweighted, but `reweight` alone, `add_noise` on a carried Gaussian's
block that had no term, or a resample under `lax.cond` is not.

**Value checks are opt-in.** Inside `enskit.linalg.debug_checks()`, every
conditioning path checks that its whitened quantities are finite, naming the
given block whose independent term is the likely cause (a singular one, most
often), and that its result is finite. Outside debug mode, and always under
`jit`, a singular noise covariance gives `nan` without raising.

**Accuracy at extreme ratios.** The computation is backward stable: when the
given blocks' factor is $10^{8}$ times larger than their noise in some
direction, expect the conditional to be accurate to about
$10^{8}\varepsilon \approx 10^{-8}$ relative, which is also how far the exact
conditional moves when the factor is perturbed by one rounding. The contract's
*Accuracy* section has the measurements.

## References

- Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
  with the ensemble transform Kalman filter. Part I: Theoretical aspects.
  *Monthly Weather Review*, 129(3), 420–436.
- Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993). Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F (Radar
  and Signal Processing)*, 140(2), 107–113.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Journel, A. G. & Huijbregts, C. J. (1978). *Mining Geostatistics*. Academic
  Press.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian
  nonlinear state space models. *Journal of Computational and Graphical
  Statistics*, 5(1), 1–25.
- Kong, A., Liu, J. S. & Wong, W. H. (1994). Sequential imputations and
  Bayesian missing data problems. *Journal of the American Statistical
  Association*, 89(425), 278–288.
- Pham, D. T. (2001). Stochastic methods for sequential data assimilation in
  strongly nonlinear systems. *Monthly Weather Review*, 129(5), 1194–1207.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
  M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
  Machine Learning Research*, 22(105), 1–47.
