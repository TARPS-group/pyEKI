# Toy problems

`pyeki.toy` ships three small calibration problems, each bundling a forward
model with a prior, an observation error covariance, a synthetic observation
and the parameters that generated it. They exist so that the package's own
tests, this documentation and the tutorials all work the same problems, and so
that trying pyEKI needs no data and no model of your own.

:::{important}
**These are not production models, and they are not an interface.** pyEKI
ships no forward models for real use and defines no base class, protocol or
registry for one — a forward model is any callable from a `(J, P)` ensemble to
`(J, N)` predictions, and {doc}`writing-a-forward-model` is the whole
obligation. What these problems exemplify is that callable and the setup
around it. Nothing in the library imports this module.
:::

## The three problems

| factory | model | for |
| --- | --- | --- |
| `linear_gaussian(u_dim=…, v_dim=…)` | $v = Gu$, at any pair of dimensions | a problem whose posterior is known exactly, and — at $P \gg J$ — the ensemble-size limit |
| `exponential_decay()` | $v_i = u_0 e^{-u_1 t_i}$, two parameters | a mildly nonlinear problem, where the tempering ladder earns its cost |
| `restricted_decay()` | the same, defined only for rates above a floor | a model that fails, reproducibly |

Each factory takes a `seed`, and every array it builds is a deterministic
function of its arguments — so a run over a toy problem gives the same numbers
in a docs build, in a test, and on your machine.

## A complete run, against a known answer

The linear problem is the one to reach for first, because its posterior is
available in closed form and the run can be compared against it rather than
against a tolerance:

```python
import jax
from pyeki import toy
from pyeki.eki import AdaptiveESSSchedule, EKIState, run
from pyeki.gauss import Gaussian

problem = toy.linear_gaussian(u_dim=4, v_dim=8)

state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=32)
result = run(state, problem.forward, problem.y, problem.noise_cov,
             schedule=AdaptiveESSSchedule())

exact = problem.posterior()               # the closed form, as a Gaussian
fitted = Gaussian.from_samples(result.ensemble)

result.mean      # [-1.3289  1.362   0.6229  0.1361]
exact.mean       # [-1.3303  1.3749  0.6281  0.1409]
problem.truth    # [-1.4009  1.4321  0.6248  0.2005]
```

The run's mean is within `0.013` of the exact posterior mean, and the two sit
about the same distance from `truth` — the remaining gap is what eight noisy
observations can support, not error in the run. The spreads agree to three
digits as well:

```python
fitted.cov.diag() ** 0.5    # [0.0686  0.1272  0.1167  0.0877]
exact.cov.diag() ** 0.5     # [0.0687  0.1276  0.1165  0.0876]
```

Three details of that call are worth naming, because they are the ones a
reader will carry to their own problem:

- **`forward`, `y` and `noise_cov` are three arguments**, and a problem is not
  callable. That is the real signature of a run; the container is a
  convenience for setting one up, and the EKI contract deliberately excludes
  one being accepted in their place.
- **`posterior()` is not new mathematics.** It is
  `GaussianJoint.from_linear_map(prior, G).condition(y, noise_cov / level)`,
  two lines of {doc}`conditioning`. Copy them to reach the rest of that
  object — `v_marginal` is the prior predictive distribution.
- **`posterior(level=β)` is the tempered target** a run passes through on the
  way to `level=1.0`, which is what makes an intermediate step checkable too.

## A model that fails

`restricted_decay()` is the decay model restricted to positive rates, so a
member whose rate is not positive gets a wholly non-finite prediction row —
which is how a forward model signals a failed member. The prior puts about 16%
of its mass below the floor, so the first step loses nine of the sixty-four
members and no later step loses any:

```python
problem = toy.restricted_decay()
state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
result = run(state, problem.forward, problem.y, problem.noise_cov,
             schedule=AdaptiveESSSchedule())

result.min_n_valid            # 55
result.stacked.n_valid        # [55 64 64 64 64]
result.mean                   # [1.9801  1.474 ]  against a truth of [2. 1.5]
```

Which members fail is a deterministic function of the ensemble, so this is
reproducible; the *fraction* is a property of the problem rather than an
injected rate, and `rate_floor=` moves it. Raise the floor toward the prior
mean to fail more members and see where repair stops being adequate; it must
stay below the true rate of 1.5, or the observation would have been generated
where the model does not evaluate.

The failure here is signalled with `jnp.where`, which is the cheap version.
The realistic one is a wrapper that catches its own subprocess failures and
returns non-finite rows: {doc}`writing-a-forward-model` works one through.

## When the parameters outnumber the ensemble

`linear_gaussian` takes its dimensions, so the high-dimensional case is the
same problem at a different size — and the closed form is what makes its
lesson visible:

```python
problem = toy.linear_gaussian(u_dim=2000, v_dim=40)
state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=40)
result = run(state, problem.forward, problem.y, problem.noise_cov,
             schedule=AdaptiveESSSchedule())

result.ensemble.std(axis=0, ddof=1).mean()          # 0.0140
(problem.posterior().cov.diag() ** 0.5).mean()      # 0.9900
```

The ensemble reports an average posterior standard deviation seventy times
smaller than the exact posterior's. Forty observations cannot constrain two thousand
parameters, and the exact posterior says so; the run fits them with 39 degrees
of freedom and collapses. Nothing raises, and no field of `HistoryRecord`
flags it.

`posterior()` builds a `(P, k)` factor for a prior covariance factor of width
`k`, and a `(k, k)` transform on the way, so it raises above a budget of 20
million elements rather than allocating. At a full-rank prior in 2000
dimensions the returned factor is 32 MB and the peak is nearer 180 MB; above
about four thousand dimensions the closed form is simply unavailable, while
the run is not.

## What these models are, and are not, an example of

Every model here is pure JAX, so each is `jit`-able, `vmap`-pable and
differentiable. **That is a convenience of these particular models** — chosen
to keep a test suite that uses them heavily cheap — **and not a requirement on
yours.** A run's driver loop is ordinary Python precisely so that a
subprocess, a job-scheduler submission or a legacy binary is a legal forward
model, and none of those is traceable.

What they *are* an example of is the batched convention and row independence:

- **A forward model is called once per step with the whole ensemble.** The
  linear model applies its operator to the trailing axis; both decay models
  are `jax.vmap` of a function of one member, which is the wrapper a
  per-member model needs.
- **Row `j` of the return depends only on row `j` of the argument.** That is
  the one requirement nothing inside a run detects, and both idioms above make
  it structural rather than a claim. From outside a run it *is* detectable:
  `pyeki.eki.testing.check_forward_model` permutes the ensemble and
  re-evaluates a subset of it, which between them catch an order-dependent
  coupling and a symmetric one. Neither is sufficient alone.

```python
from pyeki.eki.testing import check_forward_model

check_forward_model(my_forward, u_dim=12, v_dim=40)
```

It calls the model five times, so check a cheap configuration of an expensive
one. Pass `stochastic=True` for a model that is legitimately not
deterministic; a run permits one.

## Modifying a problem

Every field is a plain public value, so a variant is a direct construction
rather than a new factory. Giving the linear problem correlated observation
error, for instance, changes nothing else about the run:

```python
import dataclasses
import numpy as np
import jax.numpy as jnp
from pyeki.linalg import DensePSD

rng = np.random.default_rng(1)
M = rng.normal(size=(8, 8))
R = jnp.asarray(M @ M.T / 8 + 0.01 * np.eye(8))

problem = toy.linear_gaussian(u_dim=4, v_dim=8)
correlated = dataclasses.replace(problem, noise_cov=DensePSD.from_matrix(R))

correlated.posterior().mean       # [-1.4093  0.8248  0.4646  0.0692]
```

Replacing `noise_cov` is the safe case. Replacing `G` or `prior` leaves `y`
as the old map generated it, so `truth` is no longer the parameters behind the
observation and `posterior()` answers a problem nobody posed — build a fresh
one from the factory instead.

The closed form follows the new covariance, since `posterior()` reads the
field.
