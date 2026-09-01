# 1. Your first inversion

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader ends with a complete, working inversion of about twenty lines,
against a forward model they could swap for their own, and an answer they know
how to read off.

## Prerequisites

None. Python and NumPy familiarity only. Do **not** assume the reader knows
what EKI is, what tempering is, or that `pyeki.linalg` exists.

## What this page covers

- The problem shape: an expensive, non-differentiable $\mathcal{G}$, noisy
  observations $y$, and the goal of estimating parameters with uncertainty.
  Two paragraphs, not a derivation.
- `import pyeki` first, and why — it enables float64, and arrays created
  beforehand stay float32 and are not promoted.
- Defining a forward model. **The single most important mechanical point on
  this page: `forward` receives the whole `(J, P)` ensemble and returns
  `(J, N)`, not one member at a time.** A reader who writes a per-member
  function gets a shape error from inside JAX with no indication of the cause.
  Show the ensemble-shaped version, and show the `jax.vmap` one-liner for
  wrapping a per-member model. {doc}`../user-guide/writing-a-forward-model`
  states the full obligation; link to it rather than restating it.
- A prior: `Gaussian(mean=..., cov=PSDDiagonal(...))`. Introduce `PSDDiagonal`
  as "the diagonal covariance" with no discussion of the operator layer.
- An observation error covariance, also `PSDDiagonal`.
- `EKIState.from_prior(key, prior, n_members=...)`, and that keys are typed
  keys from `jax.random.key`.
- `run(state, forward, y, noise_cov, schedule=AdaptiveESSSchedule())`.
- Reading the answer: `result.mean`, `result.ensemble`, and one sentence that
  the ensemble *is* the uncertainty estimate.

## Deliberately not covered

Defer, and say so with a forward link:

- what the schedule is doing, or that a ladder exists → tutorial 3
- diagnostics, and whether to trust the answer → tutorial 2
- any operator other than `PSDDiagonal` → tutorial 4
- `TransformUpdate` vs `PathwiseUpdate`, inflation, stopping rules
- the `iterate` generator

## API exercised

`pyeki`, `pyeki.linalg.PSDDiagonal`, `pyeki.gauss.Gaussian`,
`pyeki.eki.EKIState.from_prior`, `pyeki.eki.run`,
`pyeki.eki.AdaptiveESSSchedule`, `EKIResult.mean`, `EKIResult.ensemble`.

## Notes for the writer

Pick a forward model with two parameters so results print on one line and can
be plotted in a plane. Use the shared toy models rather than defining one
inline, so tutorials 2 and 3 can carry the same problem forward without
redefining it.
