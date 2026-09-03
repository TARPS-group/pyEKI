# 5. Transform or pathwise

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can choose between the two ways pyEKI moves an ensemble through a
step, and knows what each one costs.

## Prerequisites

Tutorials 1 to 4.

## What this page covers

Both rules answer the same question — the ensemble and its predictions are in
hand, and the increment is chosen, so where do the members go? They differ in
how they produce the spread.

- **`PathwiseUpdate`**, the classical perturbed-observation form. Each member
  is conditioned on the observation plus its own draw from the observation
  error, so the spread comes from those draws. It consumes the run's key, so
  two runs from different keys differ.
- **`TransformUpdate`**, the deterministic square-root form and the default.
  The ensemble's anomalies are transformed so that their empirical covariance
  equals the conditioned one directly, with nothing drawn.
- **What the choice costs.** `TransformUpdate` is deterministic, so a
  difference between two runs is a bug rather than a seed — which is why it is
  the default. The exactness property holds *exactly* under it and only *in
  expectation* under the stochastic form. `PathwiseUpdate` is the form most of
  the literature is written in, and the one for which the pathwise-sampling
  reading of a run is available.
- **Same problem, both rules, measured.** Not just the two answers, but the
  variability of each across keys: the honest comparison is one run of
  `TransformUpdate` against the spread of many `PathwiseUpdate` runs, because
  a single pathwise run tells the reader nothing about whether the difference
  they see is the rule or the seed.
- **Where the stochastic form is preferable**, stated as plainly as the case
  for the default. Its errors are unbiased across keys rather than systematic,
  and averaging over keys is available to it.
- Both rules are two lines over `pyeki.gauss`, and the reason the square-root
  reading is valid only for a centred ensemble is in
  {doc}`../user-guide/conditioning` and {doc}`../joint-factor`. Link; do not
  reproduce.

## Deliberately not covered

- the derivation of either update → {doc}`../gaussian-contract`
- why the square-root update lives on the sample container rather than the
  Gaussian → {doc}`../joint-factor`
- writing an update rule of your own → {doc}`09-your-own-policy`

## API exercised

`TransformUpdate`, `PathwiseUpdate`, `EKIState.key`,
`pyeki.gauss.EmpiricalJoint`.

## Notes for the writer

Measured on `toy.exponential_decay()` at 64 members, `AdaptiveESSSchedule`:
the pathwise run took 8 forward evaluations against the transform run's 6, and
the two posterior means differ in the third digit. That is one key, and one
key is not a comparison — the page's own point above is that this needs
replication across keys before anything is claimed from it.

The temptation to resist is ranking the two rules. The library has a default
and states why; this page should leave a reader able to defend either choice
on their own problem.
