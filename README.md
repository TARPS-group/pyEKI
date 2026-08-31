# pyEKI

Ensemble Kalman Inversion for derivative-free Bayesian calibration.

**Status: pre-alpha.** The linear operator, Gaussian conditioning and EKI layers
are implemented, tested and documented — you can run an inversion today. The
localization layer, needed when the parameter dimension far exceeds the ensemble
size, is not yet built. See [HANDOFF.md](HANDOFF.md) for state and next steps.

## What it is

Ensemble Kalman Inversion (EKI) estimates the parameters of an expensive,
possibly non-differentiable forward model from noisy observations. It needs
only forward evaluations: no gradients, no adjoint, no access to the model's
internals. That makes it well suited to legacy simulators, coupled codes, and
anything that runs as a subprocess.

pyEKI aims to be a small, robust, efficient implementation of EKI and its
common variants, built on:

- **Structured linear operators** so covariance structure is exploited rather
  than materialized as dense arrays.
- **A joint Gaussian abstraction** providing sampling, the whitened-SVD
  ensemble update, and the posterior as a distribution.
- **The EKI algorithms themselves** — tempering schedules, ensemble updates,
  inflation, and the driver loop, in both the approximate-sampling and the
  optimization form.
- **Localization** for problems where the parameter dimension far exceeds the
  ensemble size. *(planned)*

## What it is not

pyEKI does not implement forward models, priors, or Gaussian process kernels.
The forward model is any callable mapping parameters to predicted observations,
and a prior is any operator satisfying the covariance interface. Building those
is the caller's job, which keeps pyEKI independent of the domain being
calibrated. What a forward model must satisfy is stated in one place, in the
"Writing a forward model" page of the user guide.

## Installation

```bash
uv sync
```

For development, including docs and test tooling:

```bash
uv sync --group dev
```

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
result.stacked.ess     # effective sample size at each iteration of the ladder
```

The same driver runs the optimization form — it is a different schedule, not a
different function or a flag:

```python
from pyeki.eki import DiscrepancyStop, FixedSchedule

result = run(state, forward, y, noise,
             schedule=FixedSchedule.constant(1.0, n_steps=200),
             stop=DiscrepancyStop(tau=1.0))
```

Covariances are structured operators, so a mix of independent and correlated
observation error costs the sum over blocks rather than the cube of the total
size:

```python
from pyeki.linalg import DensePSD, block_diag

noise = block_diag(
    PSDDiagonal(jnp.array([0.5, 0.5, 2.0])),   # independent errors
    DensePSD.from_matrix(jnp.eye(2) + 0.3),    # correlated block
)

noise.shape          # (5, 5)
noise.logdet()       # summed over blocks, never forms a 5x5 matrix
noise.whiten(y)      # applied block by block
```

## Documentation

<https://tarps-group.github.io/pyEKI/>

Start with the tutorials, which build up from a first inversion; the user guide
answers "when and why" for each choice; the contracts specify behaviour
normatively.

## License

MIT
