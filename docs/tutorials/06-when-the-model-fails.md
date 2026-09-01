# 6. When the forward model fails

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can run an inversion against a forward model that sometimes crashes,
returns `nan`, or takes long enough that the run must be resumable — which
describes most real simulators.

## Prerequisites

Tutorials 1 to 3. Tutorial 4 is helpful but not required.

## What this page covers

- Why this is the normal case, not the exceptional one: parameter draws that
  leave a model's valid domain are routine early in a ladder, when the ensemble
  still has prior spread.
- What the library does by default. `on_failure='repair'` replaces a
  non-finite member's prediction with the valid centre, and the run continues.
  Show the actual per-step message and the summary `UserWarning`, so the reader
  recognizes them:

  ```
  step 0: 63 of 64 members' predictions were finite; the rest were repaired
          to the valid centre
  UserWarning: pyeki.eki.run: some forward-model evaluations failed; the worst
  step had 63 of 64 members valid. Each such step conditioned on a covariance
  damped by (n_valid - 1) / (J - 1). Inspect result.stacked.n_valid.
  ```

- The cost of repair, stated honestly: the step conditions on a covariance
  damped by $(n_{\text{valid}} - 1)/(J - 1)$. Repair is a departure from the
  ladder, and it is opt-in by being the default.
- The `on_failure` alternatives, and when to prefer raising over repairing.
- `result.min_n_valid` and `result.stacked.n_valid` as the audit trail; a run
  with a persistently low `n_valid` is not a run to trust.
- Making a wrapper that catches subprocess failures and returns `nan` rows,
  which is the adapter most real models need.
- `EKIError`, and that it carries the state and the history — so a failed run is
  still inspectable and still resumable.
- Checkpointing and resumption: `EKIState` is a pytree, and **`step` is
  cumulative across runs**, so chaining a fresh ladder onto a finished state
  returns unchanged with nothing raised. Use `EKIState.restart()`. This trap
  belongs on this page because resumption is where it bites.
- `iterate` as the generator form, for progress reporting and for checkpointing
  between steps.

## Deliberately not covered

- the validity mask deferral, i.e. that only `n_valid` is carried and not the
  per-member boolean mask — a design note, not a user concern
- retry and caching strategies for expensive models, which are caller concerns

## API exercised

`run(..., on_failure=...)`, `iterate`, `EKIError`, `EKIState.restart`,
`EKIResult.min_n_valid`, `HistoryRecord.n_valid`,
`pyeki.eki.repair_failed_members`.

The wrapper obligation itself — catch your own exceptions, return a non-finite
row — is specified in {doc}`../user-guide/writing-a-forward-model`, whose
worked example is an external executable that does exactly that. This page
should exercise it, not restate it.

## Notes for the writer

The user guide's "Failed members", "Driving the loop yourself" and
"Checkpointing, resumption, and errors" sections are the reference versions.
This page should be a narrative built on one flaky model, linking there for the
full rules.
