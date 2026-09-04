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

**One finding this page should carry, because it is what moved
{doc}`01-first-inversion` off the default.** On the same problem, over eight
keys, `TransformUpdate` reproduces the target's covariance but not its shape:
the ensemble's least-varying principal direction comes out with a kurtosis
between 6 and 34, against 3 for a Gaussian, and at 64 members two members can
hold 72% of that direction's variance. `PathwiseUpdate` gives 2.3 to 3.6.

It is the nonlinearity rather than sampling error, and the two checks that
establish that are worth repeating on the page. Ensemble size does not help —
the kurtosis is 19 at 64 members and 186 at 4096. And on
`toy.linear_gaussian` the same rule leaves the ensemble Gaussian
(kurtosis 3.2) *identically* at 1, 6 and 20 steps, so the rule is not
intrinsically spiky: `transform_update` is a linear recombination of the
anomalies the ensemble already has, and a curved forward model gives it
anomalies whose off-ridge mass sits in a few members.

This is one problem at one size, which is exactly the caveat below. Do not
promote it to a ranking of the two rules; issue #29 is where the question of
the package default is being decided, and this page should read the same
whichever way that goes.

The temptation to resist is ranking the two rules. The library has a default
and states why; this page should leave a reader able to defend either choice
on their own problem.
