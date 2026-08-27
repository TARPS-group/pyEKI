# Running an inversion

`pyeki.eki` is the layer that turns Gaussian conditioning into a *run*: an
initial ensemble, a ladder of intermediate targets, one ensemble update per
rung, and a record of what happened.

This page is about *when and why* to reach for each piece. The
{doc}`../eki-contract` reference page specifies exactly *what* each one does,
and is the place to look for precise shapes, error behaviour, and the
mathematics of the two adaptive criteria.

## The shortest complete run

```python
import pyeki  # enables float64; import this before creating arrays
import jax
import jax.numpy as jnp
from pyeki.eki import AdaptiveESSSchedule, EKIState, run
from pyeki.gauss import Gaussian
from pyeki.linalg import PSDDiagonal, DensePSD

prior = Gaussian(jnp.zeros(12), DensePSD.from_matrix(C0))
noise_cov = PSDDiagonal(instrument_variances)          # side N

state = EKIState.from_prior(jax.random.key(0), prior, n_members=64)
result = run(state, forward, y, noise_cov, schedule=AdaptiveESSSchedule())

result.ensemble       # (64, 12) approximate posterior ensemble
result.mean           # (12,) its mean
result.beta           # 1.0 — the ladder finished
```

`forward` is any callable taking a `(J, P)` array of members and returning a
`(J, N)` array of predictions. That is the whole forward-model interface: pyEKI
ships no models and defines no base class, and the callable may be `jit`-ed,
may fan out over processes, or may block on a job scheduler.

## Three choices, and the same driver

A variant of Ensemble Kalman Inversion is a choice on three independent axes,
and `run` is the same function in every case:

| axis | argument | question it answers |
| --- | --- | --- |
| how far, and when to stop | `schedule=`, `stop=` | what is the next increment, and when does the run end? |
| how the ensemble moves | `update=` | given the increment, what is the new ensemble? |
| how spread is maintained | `inflation=` | what happens before each forward evaluation? |

The axes compose without restriction, and the driver never inspects one to
decide another. That is mechanical, and it is not a claim that every
combination is *meaningful* — see [Two traps](#two-traps) below.

## The two forms are two schedules

The sampling form and the optimization form are not two drivers and not a
flag. They differ in whether the ladder has a temperature budget.

```python
from pyeki.eki import DiscrepancyStop, FixedSchedule

# Sampling: a budget of beta = 1, an adaptive ladder, no stopping rule.
sampled = run(state, forward, y, noise_cov, schedule=AdaptiveESSSchedule())

# Optimization: unit steps, no budget, stop when the data are fit.
fit = run(state, forward, y, noise_cov,
          schedule=FixedSchedule.constant(1.0, n_steps=200),
          stop=DiscrepancyStop(tau=1.0))
assert fit.stop_fired            # False means the ladder ran out first
```

`sampled.ensemble` is an approximate posterior ensemble, under the caveats in
[What a run does not promise](#what-a-run-does-not-promise). `fit.ensemble` is
a *collapsing* ensemble around a regularized fit, which is not a posterior at
all — which is why neither result is named `posterior`.

## The tempering ladder, in one identity

The layer's targets are

$$\pi_\beta(u) \;\propto\; \pi_0(u)\, e^{-\beta \Phi(\mathcal{G}(u))},
\qquad \Phi(v) = \tfrac12 \lVert W(y - v)\rVert^2 ,$$

the prior at $\beta = 0$ and the posterior at $\beta = 1$. Moving one increment
up that ladder is an identity, not an approximation: it is conditioning on the
same observation with the noise covariance divided by that increment. So a
step is

```python
EnsembleJoint(u_samples=u, v_samples=v).transform_update(y, noise_cov / dbeta)
```

and nothing in `pyeki.gauss` had to change to serve tempering.

Two consequences are worth carrying around. Per-step precisions **add**, so a
ladder whose increments sum to 1 composes to one-shot conditioning at
$\beta = 1$ — which is the one property this layer proves about itself. And the
divisor is the *increment*, never the accumulated level: dividing by $\beta$
raises nothing and produces a plausible-looking posterior that is wrong by a
factor growing with the ladder's length.

## Choosing a schedule

| schedule | ladder | reach for it when |
| --- | --- | --- |
| `FixedSchedule.uniform(T)` | `T` rungs of `1/T`, to $\beta = 1$ | you must know the evaluation budget in advance |
| `FixedSchedule.constant(c, T)` | `T` rungs of `c` | the optimization form, or `constant(1.0, 1)` for a single Kalman update |
| `AdaptiveESSSchedule()` | adaptive, budget 1 | the posterior ensemble is the deliverable |
| `AdaptiveMisfitSchedule()` | adaptive, budget 1 | the *fit* is the deliverable, or evaluations are scarce |

The two adaptive schedules answer the same question — how far can the target
move before this ensemble stops describing it? — and measure it differently.
`AdaptiveESSSchedule` keeps the effective sample size of the tempering weights
above a fraction of $J$, so each intermediate target stays representable by the
ensemble that must describe it. `AdaptiveMisfitSchedule` instead absorbs as
much data per rung as the noise at that rung can explain, calibrated to the
observation dimension rather than to the ensemble size.

They do not agree, and they are not variants of one idea: the misfit schedule
takes **far longer steps**, driving the effective sample size to within a
member or two of its floor of 1. That is its logic, not a defect. Its
`divergence_budget` defaults to $N/2$, at which the schedule has no tuning
parameter at all — the observation dimension supplies it.

Neither adaptive schedule ever evaluates the forward model to choose an
increment. Both read only the misfits the driver already computed, so
adaptivity is free in the resource that matters.

:::{note}
A budgeted adaptive schedule cannot overshoot its budget: the increment is
clamped by the remaining budget last, after the floor and the ceiling. At the
shipped defaults — `beta_target=1.0`, `min_increment=1e-3` — the worst case is
1000 rungs, which is exactly `run`'s default `max_steps`. Lowering the floor or
*raising* the budget breaks that relation, and the driver checks the arithmetic
at entry and raises `ValueError` before spending an evaluation rather than
letting you discover it a thousand model calls later.
:::

## Choosing an update

```python
from pyeki.eki import PathwiseUpdate, TransformUpdate

run(..., update=TransformUpdate())   # the default: deterministic, no key
run(..., update=PathwiseUpdate())    # perturbed observations, consumes the key
```

Both are two lines over `pyeki.gauss`, and the choice is the one
{doc}`conditioning` describes. `TransformUpdate` is the default because the
exactness property above holds *exactly* under it and only *in expectation*
under the stochastic update; it is also deterministic, so your first two runs
agree and a difference between them is a bug rather than a seed.
`PathwiseUpdate` is the classical perturbed-observation form most of the
literature is written in, and the one for which the pathwise-sampling reading
of a run is available.

## Reading the result

```python
result.status              # "schedule_exhausted" or "stopping_rule"
result.budget_complete     # did the ladder finish?
result.stop_fired          # did the stopping rule fire?
result.n_steps             # rungs; result.n_evaluations is the forward calls
result.min_n_valid         # the worst step's valid-member count
result.stacked             # the history, every field (T,)-shaped
result.last_evaluation     # the final forward evaluation
```

`stacked` is the whole history as one record, which is what you plot:

```python
plt.plot(result.stacked.step, result.stacked.misfit_mean)
```

It works on an empty history too, returning `(0,)`-shaped fields rather than
raising, which the obvious `jax.tree.map` one-liner does not.

There are deliberately **two** termination booleans and no single `converged`.
A name like that has to pick one of two questions and then answers the other
one wrongly: defined as "a stopping rule fired" it is `False` on a perfectly
completed $\beta = 1$ ladder and `True` on an early exit at $\beta = 0.4$.

**`last_evaluation` is not `result.ensemble`.** On a schedule-exhausted run the
last update produces the returned ensemble and the loop then ends, so the
returned ensemble has never been evaluated and the last record's misfits
describe the ensemble *before* it. `last_evaluation` is the run's final
forward evaluation — its members, its predictions, and its whitened residuals —
so the two questions everyone asks first, *what is my final misfit* and *what
does the posterior predictive look like*, cost nothing more.

Moments beyond the mean are one line through the layer below:

```python
from pyeki.gauss import Gaussian

fit = Gaussian.from_samples(result.ensemble)
fit.cov.diag()                       # (P,) per-coordinate variances
fit.sample(key, 1000)                # draws from the fitted moments
```

That line is a *fit to the terminal ensemble*, not a further conditioning step
and not a posterior.

(two-traps)=
## Two traps

**A stopping rule on a budgeted ladder.** A stopping rule fires on the misfits
regardless of how much budget remains, so pairing `DiscrepancyStop` with
`AdaptiveESSSchedule()` can end a sampling run at $\beta = 0.4$, whose ensemble
is neither a posterior nor a fit. Nothing raises; `stop_fired=True` with
`budget_complete=False` is what says so.

**Chaining a new ladder onto a finished state.** `state.step` is cumulative
across runs — which is exactly what makes resumption work — so handing a
*finished* state to a fresh schedule of the same length finds the ladder
already exhausted and returns immediately, with an empty history and the
ensemble unchanged. Nothing raises, because a finished ladder legitimately
returns. Use `restart()`:

```python
phase2 = result.state.restart()      # step = 0, beta = 0.0, same ensemble and key
```

A run that does no work at all logs at `WARNING` on the `pyeki.eki` logger, so
the silent case is at least not silent.

## Failed members

A member is *failed* when its prediction row contains any non-finite entry.
That is the whole failure signal, and it puts one real obligation on you: **a
forward model that may crash, time out, or lose a worker must catch that itself
and return a non-finite row** for the affected members. An exception that
escapes the callable propagates out of the driver and stops the run.

By default failed members are *repaired* — moved to the valid members' centre,
with the valid members left exactly where they were. The ensemble size never
changes, so no downstream shape becomes dynamic. The cost is that the step
conditions on a covariance damped by $(J_v - 1)/(J - 1)$, which is exactly the
gain at a slightly shorter increment: the safe direction, bounded by the
failure fraction.

Failures are surfaced three ways, because any one of them is easy to miss:
every record carries `n_valid`, the driver logs at `WARNING`, and `run` issues
one `warnings.warn` per run in which any member ever failed. `on_failure="raise"`
turns any failure into an `EKIError` instead; fewer than two valid members
raises under either setting, since a single member has no anomalies.

What the signal *cannot* see is finite nonsense — a solver returning zeros, its
initial condition, or a sentinel such as `-9999`. A fill value is the dangerous
case: it produces an enormous misfit, which an adaptive schedule reads as
genuine ensemble disagreement and answers by shrinking the increment, so the
run stalls on a broken member instead of flagging it. Map those to non-finite
rows in your own wrapper, where the information exists.

## Inflation

```python
from pyeki.eki import AdditiveInflation, MultiplicativeInflation

run(..., inflation=MultiplicativeInflation(anomaly_factor=1.02))
run(..., inflation=AdditiveInflation.from_cov(0.01 * prior.cov))
```

Inflation maintains ensemble spread, and it runs at the *top* of each step, on
the ensemble that is about to be evaluated. That placement means the ensemble a
run returns is never an inflated one and predictions always match the members
they update; its cost is that the initial ensemble is inflated before it is
ever evaluated.

The field is `anomaly_factor`, not `factor`, because it multiplies the
anomalies and therefore scales the covariance by its **square**. The literature
uses both conventions with the same symbols, and a caller passing an intended
variance inflation of 1.2 would silently get 1.44.

Prefer `AdditiveInflation.from_cov(cov)` to `AdditiveInflation(cov)`: it
factorizes once at construction, where the plain constructor re-runs
`cov.factor()` on every rung — an $O(P^3)$ Cholesky per step for a dense
covariance, on a knob you turned on to *delay* collapse.

:::{warning}
**Inflation breaks the ladder, by design.** A run with inflation on is not a
tempering ladder for the target family; it is a deliberately widened variant,
and the exactness property above no longer holds. Sampling-form runs should
leave it off, which is the default.

"Inflation" is also an overloaded word. Here it means *ensemble* inflation. In
much of the ensemble-Kalman literature the same word names inflating the
*observation noise* by a factor $\alpha$, which in pyEKI is the tempering
increment $\Delta\beta = 1/\alpha$ and is the schedule's business. Read any
external formula's definition before transcribing it.
:::

## Driving the loop yourself

`iterate` is the same run as a generator, yielding
`(state, record, evaluation)` after every rung. It is the extension point for
anything that needs to *observe* or *interrupt*: per-step checkpointing,
custom logging, a wall-clock budget, an early `break`.

```python
from pyeki.eki import INTERRUPTED, EKIResult, iterate

records = []
for state, record, evaluation in iterate(state, forward, y, noise_cov,
                                         schedule=sched):
    records.append(record)
    if time.monotonic() > deadline:
        break

result = EKIResult(state=state, history=tuple(records),
                   status=INTERRUPTED, last_evaluation=evaluation)
```

Anything that needs to *revisit* a rung — backtracking, damping, trial
increments — uses the two phases directly instead. `evaluate` costs the forward
evaluation and moves nothing; `apply` moves the state by a given increment,
using an evaluation you already have. One evaluation therefore serves any
number of trial increments:

```python
from pyeki.eki import apply, evaluate

s, delta = state, 1.0
current = evaluate(s, forward, y, noise_cov)
while not done(current):
    trial, record = apply(s, current, increment=delta, y=y, noise_cov=noise_cov)
    probe = evaluate(trial, forward, y, noise_cov)
    if probe.centre_misfit < current.centre_misfit:
        s, current, delta = trial, probe, delta * 1.5   # accept, lengthen
    else:
        delta = delta / 2                               # reject, reuse `current`
```

The accepted branch reuses `probe` as the next rung's evaluation, so this costs
one forward evaluation per rung plus one per rejection — which is what
backtracking costs in any implementation. A loop written against `advance`
alone would re-evaluate the current state on every trial, doubling that.

The same pair is how you vary the *data* between rungs — a subsampled or
randomized observation vector — which `run` and `iterate` deliberately fix for
a whole run.

## Checkpointing, resumption, and errors

`run` on a state returned by a previous run **continues** it, and the tail is
bit-identical to an uninterrupted run. That is the sole checkpointing
mechanism, and it is why policies may not hold iteration state: a schedule that
counted its own calls would be unresumable. `EKIState` is a pytree of arrays
and one small static, so serializing it is your choice of format.

`EKIError` carries the run, on every raise path:

```python
from pyeki.eki import EKIError

try:
    result = run(state, forward, y, noise_cov, schedule=sched)
except EKIError as exc:
    checkpoint(exc.state)          # resume from here after investigating
    diagnose(exc.history)
```

There is no `"max_steps"` status, because exceeding `max_steps` **raises**: a
sampling run that silently returned an ensemble at $\beta = 0.7$ labelled as a
posterior is the failure the two termination booleans exist to expose.
`max_steps` bounds the iterations of *this call*, not `state.step`, so a
resumed run gets the allowance you asked for.

## Progress reporting

The driver emits one `logging` record per step at `INFO` on the logger named
`pyeki.eki`, carrying the step, the level, the increment and the mean misfit,
and one at `WARNING` when any member fails. No handler is installed and no
configuration is read, so by default you see nothing:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Timings, profiles and progress bars are yours to add around an `iterate` loop.

## Writing your own policy

Each axis is a **protocol**, not a base class: an implementation is anything
with the right call signature — a frozen dataclass, or a plain function where
the protocol has a single method. This is the one place where pyEKI is
deliberately open to extension at the algorithm level.

Two rules bind every policy. Everything after the key is **keyword-only**,
because an update's `ensemble` and `predictions` have the same shape whenever
$P = N$ and a positional protocol would let them be transposed with no error at
all. And a policy must be **pure**: no iteration state, no counters, which is
what keeps a run resumable.

`pyeki.eki.testing` is the harness for one, and purity is the reason it exists:

```python
from pyeki.eki.testing import check_schedule, check_update, synthetic_evaluation

check_schedule(MySchedule())
check_update(MyUpdate())
```

Each check takes a policy and a small synthetic `Evaluation`, which the module
also constructs, so testing a schedule never means running a forward model.

(what-a-run-does-not-promise)=
## What a run does not promise

- **Exactness is claimed for the affine-Gaussian case only.** An affine
  forward model, a Gaussian prior, an ensemble whose empirical moments equal
  the prior's, `TransformUpdate`, no inflation, no failed members, and
  increments summing *exactly* to 1. Every clause is load-bearing, and the
  last is easy to miss: `FixedSchedule.uniform(T)` sums to 1 only to
  round-off.
- **For a nonlinear model the output is an approximation with no consistency
  guarantee.** It depends on the schedule, the ensemble size and the update
  rule; two ladders give two answers and neither bounds the other. A finer
  ladder makes each step's Gaussian approximation more accurate but
  accumulates more sampling error and collapses the ensemble further, so
  refinement is not monotone improvement.
- **Members are not independent posterior draws.** The empirical gain couples
  them, and that coupling is precisely the sampling error $J$ controls.
  Uncertainty lives in the ensemble's spread, not in any member.
- **$J$ bounds what a run can represent.** Every iterate lies in the affine
  subspace spanned by the initial ensemble, of dimension at most $J - 1$,
  however many steps you run. Directions absent from the initial ensemble are
  unreachable, and `AdditiveInflation` is the only shipped mechanism that
  leaves that subspace.
- **The optimization form deliberately collapses the ensemble.** Its terminal
  spread measures numerical convergence, not posterior uncertainty.
