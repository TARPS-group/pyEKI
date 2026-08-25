# pyEKI

Ensemble Kalman Inversion for derivative-free Bayesian calibration.

:::{admonition} Pre-alpha
:class: warning

The linear operator layer is implemented and tested. The conditioning and EKI
layers are in progress.
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
that structure — block, diagonal, triangular today; Kronecker and low-rank
planned — is exploited rather than materialized.
:::

:::{grid-item-card} Gaussian conditioning
`pyeki.gauss` provides joint Gaussian distributions and conditioning methods
that select an algorithm appropriate to the problem's dimensions. *(planned)*
:::

:::{grid-item-card} Localization
`pyeki.localize` supports problems where the parameter dimension far exceeds
the ensemble size. *(planned)*
:::

:::{grid-item-card} The EKI algorithms
`pyeki.eki` provides tempering schedules, ensemble updates, inflation, and the
driver loop, plus common variants. *(planned)*
:::
::::

## What pyEKI is not

pyEKI does not implement forward models, priors, or Gaussian process kernels.
The forward model is any callable from parameters to predicted observations,
and a prior is any operator meeting the covariance interface. Building those
belongs to the caller, which keeps pyEKI independent of the domain being
calibrated.

## Quick example

```python
import pyeki  # enables float64; import before creating arrays
import jax.numpy as jnp
from pyeki.linalg import PSDDiagonal, DensePSD, block_diag

noise = block_diag(
    PSDDiagonal(jnp.array([0.5, 0.5, 2.0])),      # independent errors
    DensePSD.from_matrix(jnp.eye(2) + 0.3),    # correlated block
)

noise.shape          # (5, 5)
noise.logdet()       # summed over blocks, never forms a 5x5 matrix
noise.whiten(y)      # applied block by block
```

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
user-guide/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

user-guide/operators
user-guide/writing-an-operator
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

linop-contract
design
api/index
```
