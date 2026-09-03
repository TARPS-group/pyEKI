# pyEKI

Ensemble Kalman Inversion for derivative-free Bayesian calibration.

:::{admonition} Pre-alpha
:class: warning

The linear operator, Gaussian conditioning and EKI layers are implemented and
tested — you can run an inversion today. The localization layer, needed when
the parameter dimension far exceeds the ensemble size, is not yet built.
:::

## What problem does this solve?

You have a forward model $\mathcal{G}$ that maps parameters to predictions, and
observations $y$ of those predictions corrupted by noise:

$$y = \mathcal{G}(\theta) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Sigma).$$

You want to estimate $\theta$ and quantify how uncertain that estimate is. The
difficulty is that $\mathcal{G}$ is expensive, and often you cannot
differentiate it — it may be a legacy simulator, a coupled code, or a binary
invoked as a subprocess.

Ensemble Kalman Inversion (EKI) handles exactly this case. It advances an
ensemble of parameter vectors toward the posterior using only forward
evaluations, requiring no gradients, no adjoint, and no access to the model's
internals.

## What pyEKI provides

::::{grid} 2
:gutter: 3

:::{grid-item-card} Structured operators
`pyeki.linalg` represents covariance matrices by how they act on vectors, so
that structure — block, diagonal, triangular and low-rank today, Kronecker
planned — is exploited rather than materialized.
:::

:::{grid-item-card} Gaussian conditioning
`pyeki.gauss` provides the Gaussian machinery of the ensemble update —
sampling, whitened-SVD conditioning, and the square-root transform, in both
the stochastic and deterministic forms.
:::

:::{grid-item-card} Localization
`pyeki.localize` supports problems where the parameter dimension far exceeds
the ensemble size. *(planned)*
:::

:::{grid-item-card} The EKI algorithms
`pyeki.eki` provides tempering schedules, ensemble updates, inflation, and the
driver loop, in both the approximate-sampling and the optimization form.
:::
::::

## What pyEKI is not

pyEKI does not implement production forward models, priors, or Gaussian
process kernels. It ships three toy problems, in `pyeki.toy`, for its own
tests and this documentation.
The forward model is any callable from parameters to predicted observations,
and a prior is any operator meeting the covariance interface. Building those
belongs to the caller, which keeps pyEKI independent of the domain being
calibrated. {doc}`user-guide/writing-a-forward-model` states everything a
forward model must satisfy, and works through wrapping an external
executable.

## Quick example

Calibrating a two-parameter decay model against three noisy observations:

```python
import pyeki                      # enables float64; import before creating arrays
import jax, jax.numpy as jnp
from pyeki.linalg import PSDDiagonal
from pyeki.gauss import Gaussian
from pyeki.eki import EKIState, AdaptiveESSSchedule, run

# The forward model: any callable from a (J, P) ensemble to (J, N) predictions.
times = jnp.array([0.5, 1.0, 2.0])
def forward(u):
    return u[:, :1] * jnp.exp(-u[:, 1:2] * times)

y = jnp.array([1.75, 1.38, 0.82])                             # observations
noise = PSDDiagonal(jnp.full(3, 0.01))                        # error covariance
prior = Gaussian(mean=jnp.zeros(2), cov=PSDDiagonal(jnp.array([4.0, 1.0])))

state = EKIState.from_prior(jax.random.key(0), prior, n_members=64)
result = run(state, forward, y, noise, schedule=AdaptiveESSSchedule())

result.mean            # posterior mean estimate
result.stacked.ess     # effective sample size at each step of the ladder
```

{doc}`tutorials/01-first-inversion` builds this up from the beginning and explains
every choice in it.

## How this documentation is organized

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tutorials
Read in order, each building on the last. Start here if you are new — the
first one runs an inversion in twenty lines and assumes nothing.
+++
{doc}`tutorials/index`
:::

:::{grid-item-card} User guide
Answers "when and why" for a specific choice, once you know the shape of the
problem. Read out of order, as needed.
+++
{doc}`user-guide/running-an-inversion`
:::

:::{grid-item-card} Examples
Runnable notebooks working through a problem end to end, with plots and
diagnostics.
+++
{doc}`examples/index`
:::

:::{grid-item-card} Reference
The normative contracts specifying each layer's behaviour exactly, the design
notes, and the API.
+++
{doc}`api/index`
:::
::::

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
```

```{toctree}
:maxdepth: 2
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

user-guide/quickstart
user-guide/operators
user-guide/conditioning
user-guide/running-an-inversion
user-guide/writing-a-forward-model
user-guide/toy-models
user-guide/writing-an-operator
```

```{toctree}
:maxdepth: 2
:caption: Examples
:hidden:

examples/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

linop-contract
gaussian-contract
eki-contract
joint-factor
design
api/index
```
