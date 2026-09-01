# 3. Sampling or optimizing

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader understands that the two well-known modes of EKI are two schedules
through one driver, and can choose between them for their own problem.

## Prerequisites

Tutorials 1 and 2.

## What this page covers

- The tempering family
  $\pi_\beta(u) \propto \pi_0(u) e^{-\beta \Phi(\mathcal{G}(u))}$: the prior at
  $\beta = 0$, the posterior at $\beta = 1$, and concentration on the minimizers
  of the misfit as $\beta \to \infty$.
- **The identity that makes the ladder exact rather than approximate**: moving
  one increment up is conditioning on the same observation with the noise
  covariance divided by that increment. This is the conceptual core of the
  page.
- **Per-step noise is $\Sigma/\Delta\beta$, never $\Sigma/\beta$.** Getting it
  wrong yields a plausible posterior that is wrong by $(T+1)/2$ times the data
  precision on a uniform $T$-step ladder, growing with ladder length. Worth
  stating even though the library gets it right, because readers reimplementing
  or reading other code will meet it.
- Sampling form: `AdaptiveESSSchedule`, budget $\beta = 1$, no stopping rule.
  What the ESS criterion is choosing and why it is monotone in the increment.
- Optimization form: `FixedSchedule.constant(1.0, n_steps=...)` with
  `DiscrepancyStop(tau=1.0)`. What the discrepancy principle is, and that
  overfitting is what the stopping rule prevents.
- `AdaptiveMisfitSchedule` as the third option, and when it is preferable.
- `TransformUpdate` vs `PathwiseUpdate`: deterministic square-root transform
  against the stochastic form, and what each costs. Keep it to a comparison
  the reader can act on.
- Same problem, both forms, side by side, with the difference in the answers
  shown rather than asserted.

## Deliberately not covered

- inflation → tutorial 5
- writing a schedule → tutorial 7
- the exactness claim's full list of preconditions — link to the EKI contract

## API exercised

`FixedSchedule`, `AdaptiveESSSchedule`, `AdaptiveMisfitSchedule`,
`DiscrepancyStop`, `TransformUpdate`, `PathwiseUpdate`.

## Notes for the writer

`docs/design.md` has the monotonicity argument for the ESS criterion and the
two-bound structure of the misfit criterion. Reference the derivations; do not
reproduce them here.
