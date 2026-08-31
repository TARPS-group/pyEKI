# 2. Reading a run

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can look at a finished run and say whether to trust it, and if not,
which knob the symptom points at.

## Prerequisites

Tutorial 1. Same forward model, carried forward.

## What this page covers

- Why the question needs asking: `run` returning `status='schedule_exhausted'`
  means the ladder finished, **not** that the answer is good. This framing is
  the point of the page.
- `EKIResult`: `status`, `n_evaluations`, `n_updates`, `beta`,
  `budget_complete`, `stop_fired`, `min_n_valid`.
- `result.stacked` and the eleven fields of `HistoryRecord` — `step`,
  `n_valid`, `beta`, `increment`, `beta_next`, `misfit_mean`, `misfit_min`,
  `misfit_max`, `centre_misfit`, `spread`, `ess`. Group them into what they
  diagnose rather than listing them flat.
- The three plots worth making: misfit against step, `ess` against step, and
  `spread` against step. Say what healthy and unhealthy look like in each.
- **`centre_misfit` is not the mean of the per-member misfits.** They differ by
  exactly $\tfrac{J-1}{2J}\operatorname{tr}(W \widehat C_{vv} W^\top)$. A reader
  comparing the two and finding a gap will assume a bug.
- `misfits` as the public array-level helper, for a run driven by hand.
- One paragraph on `Gaussian.from_samples(result.ensemble)` to summarize the
  posterior as a distribution.

## Deliberately not covered

- fixing anything the diagnostics reveal — tutorials 3, 5 and 6 each own a fix
- writing a custom stopping rule → tutorial 7

## API exercised

`EKIResult` (all properties), `HistoryRecord`, `pyeki.eki.misfits`,
`pyeki.eki.effective_sample_size`, `pyeki.gauss.Gaussian.from_samples`.

## Notes for the writer

The user guide's "Reading the result" section is the reference version of this
material. This page should be narrative and plot-led, and link there rather
than duplicating the field-by-field table.

`jax.tree.map(lambda *xs: jnp.stack(xs), *result.history)` is the manual form
of `result.stacked`; prefer the property. Worth one sentence that a static
field on a record would break the stacking, since it explains why the record
carries no metadata.
