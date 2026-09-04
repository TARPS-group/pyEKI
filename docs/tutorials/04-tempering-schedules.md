# 4. Tempering schedules

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can choose how gradually to assimilate an observation, and can see
from a finished run whether the choice was gradual enough.

## Prerequisites

Tutorials 1 to 3. The tempered family and the two destinations are defined
there; this page is about the path between them.

## What this page covers

The page is built up one ladder at a time, on the same problem, each ladder
answering a defect in the one before it. The figure carries the argument: at
each level, contours of the tempered distribution with that level's ensemble
overlaid, so how well the ensemble tracks its target is visible rather than
asserted.

1. **One step.** `FixedSchedule.constant(1.0, n_steps=1)` — a single Kalman
   update, the thing {doc}`01-first-inversion` illustrates. The ensemble lands
   near the right place and is far too wide, because one moment-matched
   Gaussian has to describe the whole distance from prior to posterior.
2. **A uniform ladder.** `FixedSchedule.uniform(T)`. Better, and the reader
   should be shown *where* the remaining error is: a uniform ladder spends
   most of its steps where the target has stopped moving, and takes its
   largest relative jump at the very first step, where the target moves most.
3. **A ladder that starts small and grows.** `FixedSchedule` takes an
   arbitrary tuple of increments, so this is a direct construction — a
   geometric sequence normalized to sum to one. Same number of forward
   evaluations as the uniform ladder, spent where the target is moving.
4. **An adaptive ladder.** `AdaptiveESSSchedule`, which chooses each increment
   from the ensemble it actually has rather than from a shape decided in
   advance. Then `AdaptiveMisfitSchedule`, which measures the same question
   against the observation dimension instead of the ensemble size, and takes
   markedly longer steps as a result.

Then, having earned it:

- **Why the early increments are tiny.** The adaptive ladder's first
  increments are far smaller than its last. That is the shape of the problem,
  not a defect: the first increments are where the likelihood is most
  informative relative to the prior.
- **Two ladders give two answers, and neither bounds the other.** This is the
  page that must demonstrate it, since it is the page with four ladders on one
  problem. A finer ladder makes each step's Gaussian approximation more
  accurate but accumulates more sampling error and collapses the ensemble
  further, so refinement is not monotone improvement.
- **Cost.** Every extra step is one more evaluation of the forward model, at
  `n_members` member evaluations each. Neither adaptive schedule ever
  evaluates the model to choose an increment, so adaptivity is free in the
  resource that matters.
- **The increment floor.** At the shipped defaults the worst case is exactly
  `run`'s default `max_steps`, and the driver checks that arithmetic at entry.
  One paragraph, linking to the user guide's note.

## Deliberately not covered

- the two update rules → {doc}`05-transform-or-pathwise`
- inflation → {doc}`07-small-ensembles`
- writing a schedule of your own → {doc}`09-your-own-policy`
- the monotonicity argument for the ESS criterion, and the two-bound structure
  of the misfit criterion → {doc}`../design`

## API exercised

`FixedSchedule` (`uniform`, `constant`, and an explicit increment tuple),
`AdaptiveESSSchedule`, `AdaptiveMisfitSchedule`, `pyeki.eki.iterate`,
`HistoryRecord.beta`, `HistoryRecord.increment`, `Evaluation.beta`.

## Notes for the writer

**The figure is the page.** `toy.exponential_decay()` has two parameters, so
the tempered distribution can be drawn. It has no closed form, but in two
dimensions the unnormalized density on a grid is all a contour plot needs:

```python
log_prior = problem.prior.log_density(grid)                          # (n*n,)
phi = misfits(problem.y, problem.forward(grid), problem.noise_cov)   # (n*n,)
log_pi = log_prior - beta * phi          # the tempered density, up to a constant
```

A 160x160 grid costs about 0.4 s, and both terms are reusable at every level —
only the multiplier on `phi` changes.

Four things that produce a plausible wrong picture:

- **Pair each ensemble with its own level.** `Evaluation` carries `beta`, and
  it is bit-identical to the `beta` of the record from the same step. Pair on
  `evaluation.beta` and the question cannot arise. Pairing an ensemble with
  the *next* level instead — `record.beta_next` — puts every cloud one contour
  set out of step, which looks like the method tracking badly rather than like
  a plotting bug.
- **The final ensemble is not in the loop.** The last record's `beta` is the
  level *entering* the last step, so the ensemble at $\beta = 1$ comes from
  `run(...)`, or from the state the generator yields last.
- **Do not compute moments from the grid and call them a distribution's
  moments.** A box that comfortably holds the posterior truncates the prior:
  on `[0.5, 3.5] x [0.2, 3.0]` the grid reports a prior mean of
  `[1.48, 1.31]` where the prior's is `[1, 1]`. At $\beta = 1$ a converged
  grid is trustworthy — verified stable to five digits across two boxes and
  three resolutions — but say that it is quadrature rather than a closed form,
  and use `toy.linear_gaussian` wherever exactness is the point.
- **Each panel needs its own limits.** The distribution's spread falls by a
  factor of about thirty from $\beta = 0$ to $\beta = 1$, so fixed limits make
  the late panels a dot. Drawing the previous panel's box into each panel
  keeps the shrinkage visible.

Measured, so the page has something to be checked against — re-measure when
writing rather than trusting these. These are the **default**
`TransformUpdate`, at 256 members, where the adaptive ladder's levels are
`0, 0.001, 0.003, 0.011, 0.055, 0.419, 1.0`; against a converged grid
reference whose standard deviations are `[0.037, 0.032]`:

| | mean error | ensemble sd | sd relative to the reference |
| --- | --- | --- | --- |
| adaptive ladder | 0.004 | `[0.039, 0.041]` | `[1.07, 1.30]` |
| one unit step | 0.035 | `[0.075, 0.737]` | `[2.04, 23.2]` |

Note that {doc}`01-first-inversion` runs `PathwiseUpdate` rather than the
default, so its level sequence and its numbers differ from the table above.
Decide which rule this page uses before measuring anything, and say which.

Stay true to what the problem shows. The one-step and adaptive ends of the
sequence differ starkly on this problem; the ladders in between may not
separate cleanly, and if they do not, say so rather than tuning the problem
until they do.
