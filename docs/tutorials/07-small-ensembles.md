# 7. When the ensemble is too small

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can recognize the ensemble-size limit in their own results, and knows
which mitigations exist and what each costs.

## Prerequisites

Tutorials 1 to 3, and {doc}`06-covariances-as-operators`
for the operator `AdditiveInflation` takes.

## What this page covers

- **The subspace bound, stated plainly and early.** Every iterate of a run lies
  in the affine subspace spanned by the initial ensemble, of dimension at most
  $J - 1$ — however many steps are run and however the schedule is chosen.
  $J$ bounds what a run can *represent*, not merely how accurately it
  estimates moments.
- A demonstration rather than an assertion, and one that can be checked
  against a closed form. `toy.linear_gaussian(u_dim=2000, v_dim=40)` with
  $J = 40$ and `AdaptiveESSSchedule` to $\beta = 1$: the run reports
  `schedule_exhausted` in a handful of steps, the answer occupies at most
  $J - 1 = 39$ of 2000 directions — and does occupy 39 — and the ensemble's
  average posterior standard deviation is **0.014** where
  the exact posterior's — `problem.posterior()`, available because the model
  is linear — is **0.990**. Forty observations cannot constrain two thousand
  parameters; the run fits them with 39 degrees of freedom and collapses,
  reporting a spread seventy times too small. Nothing raises, and no field of
  `HistoryRecord` flags it. **The failure is silent, and that is the lesson.**
  Comparing against the closed form is what makes it sayable rather than
  merely assertable, and it is worth showing that the same comparison is
  unavailable for a model that is not linear.
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
`HistoryRecord.ess`, `EKIState.n_members`, `pyeki.toy.linear_gaussian`,
`pyeki.toy.LinearGaussian.posterior`.

## Notes for the writer

The two standard deviations and the terminating status are pinned by
`tests/test_toy.py`, which also asserts the rank *bound*. The step count and
the realized rank are deliberately **not** pinned: a schedule change moves the
step count legitimately, and a test that failed for it would read as a
regression. Re-run them when writing this page rather than trusting the
numbers above, and state them as measurements of one configuration rather than
as properties.
