# pyEKI

Ensemble Kalman Inversion for derivative-free Bayesian calibration.

**Status: pre-alpha.** The linear operator layer is implemented and tested. The
conditioning and EKI layers are in progress — see [HANDOFF.md](HANDOFF.md).

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
- **A joint Gaussian abstraction** with conditioning methods that pick an
  algorithm appropriate to the problem's dimensions.
- **Localization** for problems where the parameter dimension far exceeds the
  ensemble size.
- **The EKI algorithms themselves** — tempering schedules, ensemble updates,
  inflation, and the driver loop.

## What it is not

pyEKI does not implement forward models, priors, or Gaussian process kernels.
The forward model is any callable mapping parameters to predicted observations,
and a prior is any operator satisfying the covariance interface. Building those
is the caller's job, which keeps pyEKI independent of the domain being
calibrated.

## Installation

```bash
uv sync
```

For development, including docs and test tooling:

```bash
uv sync --group dev
```

## Quick example

Structured operators compose, and expose only the operations they can perform
cheaply:

```python
import pyeki  # enables float64; import before creating arrays
import jax.numpy as jnp
from pyeki.linalg import DensePSD, PSDDiagonal, block_diag

noise = BlockDiag((
    PSDDiagonal(jnp.array([0.5, 0.5, 2.0])),   # independent errors
    DensePSD.from_matrix(jnp.eye(2) + 0.3),    # correlated block
))

noise.shape          # (5, 5)
noise.logdet()       # summed over blocks, never forms a 5x5 matrix
noise.whiten(y)      # applied block by block
```

## Documentation

<https://tarps-group.github.io/pyEKI/>

## License

MIT
