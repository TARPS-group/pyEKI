# 2. Reading a run

A run almost always returns something. It returns something when the ladder
finished cleanly, and it returns something when the ensemble collapsed after
two steps and spent the rest of the run agreeing with itself. Nothing in the
result is named `converged`, because no single field can be.

This page is about what to look at instead. It carries on with the problem and
the initial ensemble of {doc}`01-first-inversion`, and with the library's
default update rule rather than the pathwise one that page selects:

```python
import pyeki
import jax
from pyeki import toy
from pyeki.eki import AdaptiveESSSchedule, EKIState, run

problem = toy.exponential_decay()
state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
result = run(state, problem.forward, problem.y, problem.noise_cov,
             schedule=AdaptiveESSSchedule())
```

## How the run ended

```python
result.status             # 'schedule_exhausted'
result.budget_complete    # True  -- the ladder reached beta = 1
result.stop_fired         # False -- no stopping rule ended it
result.beta               # 1.0
result.min_n_valid        # 64    -- every member's predictions were usable
result.n_evaluations      # 6     -- forward model calls
result.n_completed_steps  # 6     -- times the ensemble moved
```

`status` says which of two things ended the run: the ladder ran out
(`'schedule_exhausted'`) or a stopping rule fired (`'stopping_rule'`). It does
not say whether the answer is good. A run whose ensemble collapsed on the
first step also reports `'schedule_exhausted'`.

There are two termination booleans rather than one because the two questions
are different, and either answer alone is misleading. `budget_complete` asks
whether the ladder got all the way to $\beta = 1$. `stop_fired` asks whether
something ended the run early. For a run that is meant to reach the target
distribution, you want `budget_complete=True` and `stop_fired=False`, which is
what this run reports.

`n_evaluations` and `n_completed_steps` are equal here, and differ by one when
a run ends on a stopping rule: reaching that decision cost an evaluation, and
the update it would have driven was discarded. Your cost in model calls is
always `n_evaluations`, and in member evaluations
`n_evaluations * n_members` — 384 here.

`min_n_valid` is the worst step's count of members whose predictions were
finite. Below `n_members` means the forward model failed on some members, which
is {doc}`08-when-the-model-fails`.

## The history

Every step leaves a record, and `result.stacked` returns the whole history as
one object with each field stacked across steps — which is what you plot.

```python
history = result.stacked
history.beta       # [0.      0.001   0.0028  0.0115  0.0677  0.5514]
history.increment  # [0.001   0.0018  0.0087  0.0562  0.4837  0.4486]
```

The eleven fields answer four different questions.

| field | what it tells you |
| --- | --- |
| `step`, `beta`, `increment`, `beta_next` | **where on the ladder** this step was, and how far it went |
| `misfit_mean`, `misfit_min`, `misfit_max`, `centre_misfit` | **how well the members fit the observations** |
| `spread`, `ess` | **whether the ensemble can still describe its target** |
| `n_valid` | **whether the forward model worked** |

Two things about `beta` are worth fixing in mind now, because they are easy to
misread later. Each record's `beta` is the level the step *started* from, and
its `beta_next` the level it moved to — so the last record's `beta` is 0.5514,
not 1.0, and the ensemble at $\beta = 1$ is `result.ensemble` rather than
anything in the history. And `increment` is what that step added, so the
increments sum to the level reached.

The records carry no other information — no timestamps, no schedule name.
That is deliberate: a field of that kind would make each record a different
shape to JAX, and `result.stacked` could not stack them.

## Three things worth plotting

**The mean misfit**, `misfit_mean`. The misfit of one member is half the sum
of its squared prediction errors, each divided by that observation's error
standard deviation. It measures fit in units of observation noise, so it has a
reference value: if a member fits the data as well as the noise allows, each of
the $N$ observations contributes about $\tfrac12$, and the misfit is about
$N/2$. Here $N = 12$, so the reference is 6.

**The effective sample size**, `ess`. Going up the ladder reweights the
members, and this is how many of the 64 members that reweighting effectively
leaves. It is at most `n_members` and at least 1. A value of 1 means one
member carries essentially all of the weight, and the other 63 are not
contributing.

**The spread**, `spread`. The root-mean-square standard deviation of the
members' parameters. It should decrease — the observations narrow the answer —
but *how fast* is the diagnostic.

```python
import matplotlib.pyplot as plt

plt.plot(history.step, history.misfit_mean)
plt.yscale("log")           # the first step's misfit is 1.9e5
```

```{figure} ../_generated/figures/02-trajectories.png
:alt: Three panels showing mean misfit, effective sample size and ensemble spread against step, for an adaptive ladder and for three equal steps.
:width: 100%

The same problem and the same initial ensemble, under an adaptive ladder (six
steps) and under three equal steps of $1/3$. Both reach $\beta = 1$ and both
report `'schedule_exhausted'`.
```

The adaptive run is the healthy one, and each panel says so differently.

Its misfit falls from $1.9 \times 10^5$ to 6.80, which is just above the
reference of 6 — the members fit the data about as well as the noise allows,
and no better. Its effective sample size sits on 32, which is half of 64 and
is the floor this schedule holds: each increment is chosen to keep it there.
The last step is the exception, at 57.4, because by then only 0.4486 of budget
remained and the step could not be as long as the floor would have permitted.
Its spread falls smoothly, by a factor between 1.3 and 2.7 per step.

The three-step run reaches the same level, and each panel gives a reason not
to trust it. Its effective sample size at the first step is **1.0**: at an
increment of $1/3$, a single member out of 64 carries essentially the entire
weight of the reweighting, and the step conditions on that member's opinion.
Its spread then falls by a factor of 5.2 in one step, which is the same event
seen from another angle. Its last recorded misfit is 25.5, four times the
reference of 6, so the members it was still working with did not fit the data
well. The two runs end up with different answers — the three-step run's spread
is 35% larger in the amplitude and 47% larger in the rate — and nothing in
`status` distinguishes them.

Which of those answers is nearer the truth is not a question the diagnostics
can settle, and {doc}`04-tempering-schedules` settles it by measuring both
against the target distribution. What the diagnostics do say, on their own, is
that a step which leaves one member carrying all the weight has not done the
calculation the method intends.

:::{note}
The adaptive run's own first step also sits below the floor, at 24.6 rather
than 32, and this is not a violation. At that step the schedule wanted an
increment smaller than 0.001 — its minimum — and had to take 0.001 anyway. At
0.0001 the effective sample size would have been 53.3. A first step at or
below the floor is normal, because the prior ensemble is as far from the
target as it ever gets.
:::

## A checklist

For a run meant to reach the target distribution, five things:

1. **`budget_complete` is `True` and `stop_fired` is `False`.** Otherwise the
   ensemble is somewhere partway up the ladder, and is neither a target
   distribution nor a fit.
2. **`min_n_valid` equals `n_members`.** If not, see
   {doc}`08-when-the-model-fails`.
3. **`misfit_mean` settles near $N/2$.** Far above means the model never fit
   the data. Far below means the residuals are smaller than the observation
   error you declared, which usually means that error is overstated rather
   than that the fit is unusually good.
4. **`ess` stays well above 1 after the first step** — and, for an adaptive
   schedule, at or above its floor. A step at 1 or 2 has conditioned on a
   handful of members.
5. **`spread` decreases gradually.** A single step that cuts it by a factor of
   several is a step that was too long.

If a check fails, the fix is usually the ladder — a finer one, which is
{doc}`04-tempering-schedules` — or the ensemble size, which is
{doc}`07-small-ensembles`.

Two further things this checklist cannot see. It cannot tell you whether the
answer is *right*: this problem is nonlinear, so even a run passing every
check returns an approximation with no guarantee attached. And a spread can be
healthy-looking and still much too small if the parameters far outnumber the
members; that failure has no symptom in the history at all, and
{doc}`07-small-ensembles` is about it.

## Look at the whole ensemble, not just the mean

`misfit_mean` hides a lot. At the last step of this run:

```python
history.misfit_min[-1]    # 4.5927
history.misfit_mean[-1]   # 6.7961
history.misfit_max[-1]    # 43.2523
```

The worst member fits six times worse than the average one. That is not a
problem in itself — 64 members drawn from a wide prior will not all end up in
the same place — but it is the kind of thing to look at before averaging
anything.

`result.last_evaluation` holds the run's final forward evaluation: its
members, their predictions, and each member's misfit. It answers *what is my
final misfit* and *what does the answer predict* without evaluating the model
again.

```python
evaluation = result.last_evaluation
evaluation.misfits[:4]    # [ 5.0189  29.4126   4.9678   4.7952]
```

Note that `last_evaluation` is not an evaluation of `result.ensemble`. On a run
that ended because the ladder finished, the last step produced the returned
ensemble and the loop then ended, so the returned ensemble has never been
through the model.

## The centre's misfit is not the average misfit

`centre_misfit` is the misfit of the ensemble's mean prediction. It is a
different number from the average of the members' misfits, and the difference
is large:

```python
evaluation.centre_misfit      # 4.6065
evaluation.misfits.mean()     # 6.7961
```

Both are correct. A misfit is a squared quantity, so averaging the members'
misfits includes their disagreement with each other, while the misfit of the
average does not. The gap is exactly

$$\frac{J-1}{2J}\operatorname{tr}\!\bigl(W \widehat C_{vv} W^{\top}\bigr),$$

where $J$ is the number of members, $\widehat C_{vv}$ the ensemble's
prediction covariance and $W$ divides by the observation error — so the gap is
the ensemble's own prediction spread, measured in units of observation noise.
It is 2.1896 here, and accounts for the difference to the last digit. It
shrinks as the ensemble collapses, so the two numbers converge in the
optimization form and stay apart in the sampling form.

Use `centre_misfit` when you want to know how well the answer's centre fits,
and `misfit_mean` when you want to know how well a typical member fits. They
are not interchangeable, and a reader who compares them expecting agreement
will go looking for a bug.

## Summarizing the answer as a distribution

The ensemble is the answer, but a two-moment summary of it is one line:

```python
from pyeki.gauss import Gaussian

fitted = Gaussian.from_samples(result.ensemble)
fitted.mean                    # [1.9802  1.4741]
fitted.cov.diag() ** 0.5       # [0.0396  0.0363]
```

That is a fit to the final ensemble, not a further conditioning step. It gives
you the covariance — including the correlation between the amplitude and the
rate, which is 0.83 here and is the part of the answer that a pair of standard
deviations does not carry — and it can be drawn from, which gives new points
rather than the 64 you already have.

## Driving the loop yourself

The misfit and the effective sample size are both public functions, so a loop
you drive yourself can compute them:

```python
from pyeki.eki import effective_sample_size, misfits

phi = misfits(problem.y, problem.forward(result.ensemble), problem.noise_cov)
phi.mean()                         # 5.7186
effective_sample_size(phi, 0.1)    # 61.7289 -- were the next increment 0.1
effective_sample_size(phi, 1.0)    # 57.1571
```

`effective_sample_size` is what an adaptive schedule evaluates, and it needs
no forward evaluation — only misfits that have already been computed. That is
why adaptivity is free in the resource that matters.
{doc}`../user-guide/running-an-inversion` covers `iterate`, the generator form
of a run, for anything that needs to watch or interrupt a run step by step.

## Next

The diagnostics above will tell you a run went badly. They will not tell you
what the run was *for*, and the two things a run can be aiming at want
different diagnostics: {doc}`03-sampling-or-optimizing`.
