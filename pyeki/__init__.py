"""pyEKI — Ensemble Kalman Inversion for derivative-free Bayesian calibration.

Ensemble Kalman Inversion (EKI) estimates the parameters of an expensive,
possibly non-differentiable forward model from noisy observations, using only
forward evaluations. pyEKI provides the pieces that requires:

- :mod:`pyeki.linalg` — structured linear operators, so covariance structure is
  exploited rather than materialized as dense arrays.
- :mod:`pyeki.gauss` — joint Gaussian distributions and the conditioning
  operations EKI is built from.
- :mod:`pyeki.eki` — the algorithms themselves: tempering schedules, ensemble
  updates, inflation, and the driver loop.
- :mod:`pyeki.localize` — distance-based localization for high-dimensional
  problems. *(planned)*

The forward model is any callable mapping parameters to predicted observations,
so pyEKI is independent of the model being calibrated.

Importing this package enables JAX float64. JAX defaults to float32, which is
not accurate enough for the conditioning arithmetic: ensemble anomalies are
formed by subtraction, and the resulting cancellation loses several digits.
Import pyEKI before creating any array, since arrays made beforehand stay
float32 and are not promoted afterwards.

Notes
-----
The float64 setting applies to the current process only. Worker processes
created by a process pool do not inherit it; set the environment variable
``JAX_ENABLE_X64=1`` when forward-model evaluations run in separate processes.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
