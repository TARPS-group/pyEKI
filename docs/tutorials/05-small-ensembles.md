# 5. When the ensemble is too small

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can recognize the ensemble-size limit in their own results, and knows
which mitigations exist and what each costs.

## Prerequisites

Tutorials 1 to 4.

## What this page covers

- **The subspace bound, stated plainly and early.** Every iterate of a run lies
  in the affine subspace spanned by the initial ensemble, of dimension at most
  $J - 1$ — however many steps are run and however the schedule is chosen.
  $J$ bounds what a run can *represent*, not merely how accurately it
  estimates moments.
- A demonstration rather than an assertion. A linear model at $P = 2000$,
  $N = 40$, $J = 40$, `AdaptiveESSSchedule` to $\beta = 1$: the run reports
  `schedule_exhausted` in 27 steps, average posterior standard deviation falls
  from 1.0 to 0.29, and the answer occupies 39 of 2000 directions. Nothing
  raises, and no field of `HistoryRecord` flags it. **The failure is silent, and
  that is the lesson.**
- How the symptom appears in diagnostics: collapsing `spread`, and `ess`
  behaviour that looks healthy while the answer is degenerate.
- Mitigation 1, ensemble size. The honest first answer, bounded by the cost of
  the forward model.
- Mitigation 2, `MultiplicativeInflation`. What it does to the anomalies, and
  that it **cannot leave the subspace** — so it treats collapse, not the bound.
- Mitigation 3, `AdditiveInflation`. The only shipped mechanism that leaves the
  subspace, and what that costs in bias.
- Mitigation 4, localization: the mechanism that actually lifts the bound, and
  is **not yet implemented**. Explain domain localization in a paragraph, say
  what it will look like as an `EnsembleUpdate`, and link the issue. Do not
  imply it is available.
- That inflation is a departure from the tempering ladder, and is opt-in.

## Deliberately not covered

- degenerate ensembles as a numerical edge case — that is handled, and belongs
  in {doc}`../user-guide/conditioning`
- implementing localization

## API exercised

`MultiplicativeInflation`, `AdditiveInflation`, `HistoryRecord.spread`,
`HistoryRecord.ess`, `EKIState.n_members`.

## Notes for the writer

The measured numbers above are from the shipped driver and can be reproduced
directly; do not re-derive them, but do re-run them, since a schedule change
would move the step count.
