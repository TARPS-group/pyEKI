# Pushing distributions through maps

`enskit.maps` adds a block to a distribution: the distribution of
$f(\text{inputs})$, kept jointly with the inputs it came from. That is how a
simulator's outputs enter an ensemble, how a prior becomes the joint of
parameters and predictions, and how known noise is added either by sampling
it or exactly. This page says when to use each route; the
{doc}`../maps-contract` states exactly what each call does, including the
**simulator contract** every callable used as a map must satisfy.

## Through a simulator: particles in, particles out

```python
import enskit  # enables float64; import this before creating arrays
import jax
import jax.numpy as jnp
import numpy as np
from enskit import maps
from enskit.distribution import Ensemble, Gaussian
from enskit.linalg import DensePSD, PSDDiagonal

grid = jnp.array([0.5, 1.0, 2.0])

def simulate(u):                                    # (J, 2) in, (J, 3) out
    return u[:, :1] * jnp.exp(-u[:, 1:2] * grid)

ens = Ensemble(u=jax.random.normal(jax.random.key(0), (64, 2)))
ens = maps.pushforward(ens, simulate, inputs="u", output="g")   # blocks: u, g
```

A *simulator* is any callable. `pushforward` calls it **once, with every
particle**: one `(J, d)` array per input block, in the order of `inputs`.
It returns `(J, d_out)`, and row $j$ of the output becomes particle $j$'s
value of the new block, so each particle keeps its own input and output
together. Nothing needs subclassing or registering. A function written for
one particle is wrapped with `jax.vmap`, or with a loop when it is not JAX.

`inputs` names the blocks passed, positionally; by default it is every
block. `output` either names a new block, which is appended, or one of the
inputs, which is replaced in place:

```python
def advance(u):
    return u + 0.1 * jnp.tanh(u)

moved = maps.pushforward(ens, advance, inputs="u", output="u")     # u replaced
```

A simulator that computes several quantities in one expensive run returns
them together, as a tuple in the order of `output` or as a mapping (or a
`NamedTuple`) keyed by the names:

```python
def both(u):
    return {"g": simulate(u), "rate": u[:, 1:2] ** 2}

ens2 = maps.pushforward(ens.marginal("u"), both, inputs="u", output=("g", "rate"))
```

## Writing a simulator

The obligations, in full:

| what | requirement |
| --- | --- |
| **arguments** | one `(J, d)` `jax.Array` per input block, in the order of `inputs`; concrete outside `jax.jit` |
| **return** | `(J, d_out)` as a `jax.Array`, a NumPy array or a nested list (read in the ensemble's dtype); several outputs as a tuple, mapping or `NamedTuple` |
| **dtype** | the ensemble's; a narrower float is promoted with a warning, a wider one or an integer raises |
| **rows** | row $j$ of every output depends only on row $j$ of the inputs |
| **failure** | a **non-finite row**, never an exception |
| **randomness** | allowed; a simulator that wants a JAX key sets `needs_key = True` and receives it first |

**Rows must be independent.** A simulator that normalizes across the batch,
or writes into an accumulator shared by the rows, returns finite numbers of
the right shape that are not a sample of the joint of inputs and outputs, and
nothing inside `pushforward` can tell. `enskit.testing.check_simulator`
can, from outside, by permuting the particles and by calling on a subset of
them:

```python
from enskit.testing import check_simulator

check_simulator(simulate, 2, 3)                      # input dims, output dims
check_simulator(both, 2, {"g": 3, "rate": 1})        # several outputs
```

It calls the simulator five times, so give it a cheap configuration of an
expensive code.

**A failed particle is a non-finite row.** The callable owns its own
exceptions: a crash, a timeout or a lost worker is caught there and returned
as `nan` for the particles affected. An exception that escapes stops the
`pushforward`, and every particle's result is lost with it. The failed rows
are written into the ensemble as returned; `ens.all_finite` finds them, and
what to do about them is decided above this layer. To give them weight zero:

```python
from enskit.distribution import reweight

def fragile(u):
    out = simulate(u)
    return jnp.where(u[:, 1:2] > -1.5, out, jnp.nan)   # fails outside a domain

tried = maps.pushforward(ens.marginal("u"), fragile, inputs="u", output="g")
kept = reweight(tried, jnp.where(tried.all_finite, 0.0, -jnp.inf))
```

**Wider is an error, narrower a warning.** A simulator returning `float64`
for a `float32` ensemble raises, naming the simulator: writing it would throw
away digits it computed. One returning `float32` for a `float64` ensemble is
promoted, with a warning, because promotion cannot restore the digits the
simulator never had.

## Host-side simulators under `jit`: `BlackBox`

Outside any trace a simulator receives concrete arrays, so it may call NumPy,
run a subprocess or wait on a scheduler. Inside `jax.jit` or `jax.vmap` it
receives tracers, which such code cannot read. `BlackBox` makes it callable
there:

```python
def host_simulate(u):                     # NumPy in, NumPy out
    return u[:, :1] * np.exp(-u[:, 1:2] * np.asarray(grid))

box = maps.BlackBox(host_simulate, output_dim=3)
push = jax.jit(lambda e: maps.pushforward(e, box, inputs="u", output="g"))
traced = push(ens.marginal("u"))
```

`output_dim` is needed because JAX must know the result's shape before the
simulator runs. Under `jax.vmap`, the simulator is called once per member of
the family, each time with ordinary 2-D arrays. Differentiating through a
`BlackBox` gives **zero**: its outputs are treated as constants. That is the
right derivative when the parameter being differentiated acts elsewhere (a
noise scale, a prior's hyperparameter) and the simulator's outputs are data.

## Through structure: exact operations on a Gaussian

A Gaussian pushed through a nonlinear map is not Gaussian, so `pushforward`
on a `Gaussian` accepts only maps with structure. Two ship:

- **`Linear(A, shift=c)`**: $x \mapsto Ax + c$ for any operator $A$, or
  $\sum_b A_b x_b + c$ with a mapping of operators for several inputs. The
  new block's factor row is the product operator $AF_x$, so the output stays
  correlated with the input and a structured $A$ is never densified.
- **`AdditiveNoise(R)`**: $x \mapsto x + e$, $e \sim \mathcal N(0, R)$
  independent of everything. On a Gaussian it adds $R$ as the new block's
  independent term, exactly, without sampling.

Two pushforwards of a prior give the linear-Gaussian joint of parameters,
noise-free outputs and noisy outputs, and conditioning it gives the exact
posterior:

```python
H = jnp.array([[1.0, 0.5], [0.0, 2.0], [1.0, -1.0]])
R = PSDDiagonal(jnp.full(3, 0.1))
prior = Gaussian.independent(x=(jnp.zeros(2), DensePSD(jnp.eye(2))))

joint = (prior.pipe(maps.pushforward, maps.Linear(H), inputs="x", output="g")
              .pipe(maps.pushforward, maps.AdditiveNoise(R), inputs="g", output="y"))
posterior = joint.condition(y=jnp.array([0.3, -0.2, 0.5])).marginal("x")
```

`pipe` is how an operation from this layer chains onto a distribution: the
distribution layer cannot have a `pushforward` method without importing a
layer above it.

**On an ensemble, the order of operations chooses the estimator.** Both
lines below estimate the joint of $g$ and $y = g + e$:

```python
sampled = maps.pushforward(ens, maps.AdditiveNoise(R), inputs="g", output="y",
                           key=jax.random.key(1)).project()
exact = maps.pushforward(ens.project(), maps.AdditiveNoise(R), inputs="g", output="y")
```

The first draws the noise for every particle, so the covariance of $y$
carries the sampling error of $J$ noise draws. The second adds $R$ to the
fitted covariance exactly. It has lower variance, it needs only `whiten` of
$R$ when $y$ is later conditioned on (where sampling needs `factor`), and it
keeps the result an `EnsembleGaussian`.

`Linear` on an ensemble's projection, by contrast, gains nothing: pushing the
particles through $A$ and projecting gives the same moments, since sample
covariances are linear in the particles. Linear structure pays off once the
Gaussian is not a sample covariance: a prior, a static covariance, or the
result of earlier exact operations.

## Linearizing a simulator: `statistical_linearization`

```python
# ens holds u and g = simulate(u), from the pushforward at the top of the page
fit = maps.statistical_linearization(ens, inputs="u", output="g")
predicted = fit.map(ens["u"])   # (64, 3): A u_j + b, a maps.Linear applied
fit.residuals                   # (64, 3): g_j - A u_j - b
fit.residual_cov                # Omega, their covariance: a (3, 3) operator
```

Below, $x$ stands for the inputs (`u` here) and $y = f(x)$ for the output
(`g`). The function reads both from the distribution; to linearize a
simulator, push the particles through it first, as at the top of the page.

An ensemble Kalman update never differentiates $f$; what it uses instead is
the affine map that best predicts $f(x)$ from $x$ over the particles, the
*statistical linear regression* of $f$ (Lefebvre et al., 2002):

$$
A = C_{yx}C_{xx}^{-1}, \qquad b = m_y - Am_x, \qquad
\Omega = \operatorname{cov}\big(f(x) - Ax - b\big).
$$

`statistical_linearization` returns it, as a `Linear` map you can call on
particles or push a Gaussian through exactly, plus the residuals. It reads
the output block from the distribution and never calls $f$, so an ensemble
whose outputs a run already computed costs nothing more to linearize.

**When $x$ is Gaussian, $A$ is the average Jacobian.** By Stein's lemma
(Stein, 1981),
$C_{yx} = \mathbb E[Df(x)]\,C_{xx}$ for $x \sim \mathcal N(m, C)$, so
$A = \mathbb E[Df(x)]$: a derivative averaged over the spread of the
particles rather than taken at a point. For any other distribution of $x$,
such as an ensemble after a nonlinear update, $A$ is the least-squares slope,
which is not an average Jacobian in general.

**The update uses this linear model.** A deterministic square-root update
of the ensemble on an observed $y$ with noise $R$
(`kalman.SymmetricSquareRoot`) gives the same mean and covariance as the
exact linear-Gaussian update with model $A$, intercept $b$ and noise
$R + \Omega$: the residual is treated as extra noise. The stochastic
`kalman.Matheron` update agrees with it in expectation over its noise
draws. To add $\Omega$ to a noise covariance
yourself, use `LowRankUpdate(R, fit.residual_cov.factor())`; $\Omega$ alone
is singular, so it cannot be conditioned on.

**What the residuals can tell you depends on $J$.** With $d_x$ input
coordinates:

- if $d_x \ge J - 1$, the fit passes through every particle. The residuals
  are zero to round-off whatever $f$ is, and every $A$ that fits agrees on
  the span of the input anomalies. `min_norm=True` is required here, and
  returns the $A$ that is zero off that span: the linearization an EKI run
  with fewer particles than parameters implicitly uses;
- if $d_x < J - 1$, the residuals measure how far $f$ is from affine over
  the particles. $\Omega$ uses the divisor $J - 1$ by default; for a linear model with
  independent errors it underestimates their covariance by the factor
  $(J - 1 - d_x)/(J - 1)$. When the particles are the points of a
  deterministic rule rather than samples, pass `unbiased=False` for the
  divisor $J$ ({ref}`guide-divisor`); $A$ and $b$ are the same either way.

To shrink the fit toward zero, regress a Gaussian with a term on the inputs,
`maps.statistical_linearization(ens.project().add_noise(x=Lam), ...)`, which
is ridge regression; a Gaussian has no particles, so `residuals` is then
`None`. {doc}`distributions` describes the regression itself, and example
{doc}`../examples/ex16_statistical_linearization` works one problem through.

## Your own structured map

A map with structure of its own, such as a linearization or a sigma-point
rule, implements the two methods of the `StructuredMap` protocol,
`push_ensemble(ensemble, inputs, output, key)` and
`push_gaussian(gaussian, inputs, output)`, and `pushforward` dispatches to
them. The contract's {ref}`maps-structured` lists what an implementation
must do; `pushforward` checks that the result keeps every other block in its
position.

## References

- Lefebvre, T., Bruyninckx, H. & De Schutter, J. (2002). Comment on "A new
  method for the nonlinear transformation of means and covariances in
  filters and estimators". *IEEE Transactions on Automatic Control*, 47(8),
  1406–1409.
- Stein, C. M. (1981). Estimation of the mean of a multivariate normal
  distribution. *The Annals of Statistics*, 9(6), 1135–1151.
