# 3. Sampling or optimizing

There are two different things you might want from a calibration, and they are
not the same calculation.

You might want the range of parameter values the observations are consistent
with — an answer with uncertainty attached. Or you might want the single
parameter vector that fits the observations best, with the uncertainty being
somebody else's problem. Ensemble Kalman Inversion does both, and it is used
in the literature for both, often without saying which.

In pyEKI they are the same driver with a different destination on the same
ladder. This page is about choosing, and about what the ensemble means in each
case — because in one of them the ensemble's spread is the answer, and in the
other it means almost nothing.

The problem and the initial ensemble are the ones the previous two pages used:

```python
import pyeki
import jax
from pyeki import toy
from pyeki.eki import EKIState

problem = toy.exponential_decay()
state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
```

## Two destinations on one ladder

{doc}`01-first-inversion` introduced the sequence of distributions a run walks
up,

$$\pi_\beta(u) \;\propto\; \pi_0(u)\, e^{-\beta \Phi(u)},$$

with the prior at $\beta = 0$ and the distribution the observations imply at
$\beta = 1$. The exponent is what makes the two forms one family. Nothing
stops a run at $\beta = 1$ unless you tell it to, and $\beta$ larger than 1
counts the observations more than once — a distribution more concentrated than
the observations warrant. As $\beta$ grows the family concentrates on the
parameter values that minimize the misfit, and in the limit on the best fit
alone.

So:

- **stop at $\beta = 1$** and the ensemble describes the distribution the
  observations imply. This is the *sampling* form.
- **keep going** and the ensemble collapses onto the best fit. This is the
  *optimization* form.

Both are `run`. The difference is whether the ladder has a budget.

## The sampling form

A budget of $\beta = 1$, an adaptive ladder, and no stopping rule — which is
what {doc}`01-first-inversion` and {doc}`02-reading-a-run` used.

```python
from pyeki.eki import AdaptiveESSSchedule, run

sampled = run(state, problem.forward, problem.y, problem.noise_cov,
              schedule=AdaptiveESSSchedule())

sampled.beta                             # 1.0
sampled.budget_complete                  # True
sampled.mean                             # [1.9802  1.4741]
sampled.ensemble.std(axis=0, ddof=1)     # [0.0396  0.0363]
```

`AdaptiveESSSchedule` has `beta_target=1.0` by default, and the driver
guarantees it cannot overshoot: the increment is clipped by whatever budget
remains, so the last step lands exactly on 1. Here it took six forward
evaluations.

## The optimization form

Unit steps, no budget, and a rule that decides when to stop.

```python
from pyeki.eki import DiscrepancyStop, FixedSchedule

fit = run(state, problem.forward, problem.y, problem.noise_cov,
          schedule=FixedSchedule.constant(1.0, n_steps=200),
          stop=DiscrepancyStop(tau=1.0))

fit.status                # 'stopping_rule'
fit.beta                  # 3.0
fit.n_evaluations         # 4
fit.n_completed_steps     # 3
fit.mean                  # [1.9819  1.4758]
```

`FixedSchedule.constant(1.0, n_steps=200)` is a ladder of up to 200 unit
steps. It has no budget, so it does not stop at $\beta = 1$; the 200 is an
upper bound on the run's length, and the stopping rule is expected to end it
first. Here it ended after three steps, at $\beta = 3$.

**Why a stopping rule is not optional.** Left to run, the optimization form
keeps fitting. It cannot distinguish signal from noise, so past a point it is
fitting the noise in the particular observations you have — which is
overfitting, and produces a parameter vector that reproduces this dataset and
generalizes worse than a less exact one. The ensemble's spread collapses at
the same time, which makes the answer look increasingly confident as it gets
worse.

**The discrepancy principle** is the standard way to decide when to stop, and
it is simpler than its name. Your observations have known error. So a model
that fits them *perfectly* is fitting the errors, and there is a level of
misfit below which you should not go: the level the noise alone accounts for.
Stop when you reach it.

In numbers: the misfit of a member that fits as well as the noise allows is
about $N/2$, which is 6 for this problem's twelve observations.
`DiscrepancyStop(tau=1.0)` fires when the misfit of the ensemble's mean
prediction — `centre_misfit` — drops to that level. In this run:

```python
fit.stacked.centre_misfit    # [9654.0285  577.3526  6.5196  4.5978]
```

6.52 is above the threshold of 6, so the run continued; 4.60 is below it, so
it stopped. `tau` is a tolerance on that threshold, which becomes
$\tau^2 N / 2$: `tau=2.0` stops at 24 instead of 6, which is a more
conservative choice and, on this problem, stops one step earlier.

## Side by side

The two forms, and two variants of the optimization form, on the same problem
from the same initial ensemble.

To say which is closer to the truth, there has to be a truth to compare
against. This problem has only two parameters, so the density of the target
distribution can be evaluated on a grid and its mean and spread computed
numerically. That is a reference this problem is small enough to have. It is
numerical quadrature rather than a formula, and it is converged here — the
values below are stable to six digits across three grid resolutions. For a
problem where an exact formula is the point, use
`toy.linear_gaussian(...).posterior()`, which is a closed form.

The target's mean is `[1.9769, 1.4719]` and its standard deviations are
`[0.0366, 0.0317]`.

| run | reached | forward calls | error in the mean | spread, relative to the target |
| --- | --- | --- | --- | --- |
| sampling, adaptive ladder | $\beta = 1$ | 6 | 0.0033 | `[1.08, 1.14]` |
| optimization, `tau=2` | $\beta = 2$ | 3 | 0.0528 | `[1.40, 2.94]` |
| optimization, `tau=1` | $\beta = 3$ | 4 | 0.0049 | `[0.87, 0.98]` |
| optimization, 30 unit steps | $\beta = 30$ | 30 | 0.0005 | `[0.19, 0.19]` |

```{figure} ../_generated/figures/03-two-forms.png
:alt: Left, three ensembles in the amplitude-rate plane against contours of the target distribution. Right, ensemble spread against beta for the optimization form, on a log scale, with the stopping point marked.
:width: 100%

Left, the sampling form at $\beta = 1$, the optimization form stopped at
$\beta = 3$, and the same run continued to $\beta = 30$, against contours of
the target distribution. Right, the optimization form's spread as $\beta$
grows, with the level at which `DiscrepancyStop(tau=1.0)` fires and the
sampling form's spread for comparison.
```

Read the last two rows together. The optimization form run to $\beta = 30$
finds the best-fitting parameters almost exactly — the smallest error in the
mean of the four rows, six times smaller than the sampling form's — and
reports a spread five times too small. That combination is the whole character of the
optimization form: an excellent point estimate, and an uncertainty that is not
one. Its terminal spread measures how far the ensemble has converged
numerically, not how much the observations leave open.

The `tau=1` row is where the honest warning goes. Its spread happens to land
within 13% of the target's, which looks like it got the uncertainty right too.
It did not. Stopping one step earlier, at `tau=2`, gives a spread nearly three
times too large in the rate; stopping later gives one five times too small.
The spread is set by *where the run happened to stop*, so its agreement here
is an accident of this problem and this threshold, and nothing to rely on.

## Which one you want

**The sampling form**, if the deliverable includes uncertainty — an interval, a
predictive distribution, an input to something downstream that needs to know
how well the parameters are known. Use `AdaptiveESSSchedule()`, leave
`stop=None`, and check that `budget_complete` is `True`.

**The optimization form**, if the deliverable is one parameter vector — the
best-fitting configuration, a starting point for something else, or a
calibration to be validated separately. Use
`FixedSchedule.constant(1.0, n_steps=…)` with a `DiscrepancyStop`. Read
`result.mean`, and do not read the spread as uncertainty.

On this problem the optimization form is also the cheaper of the two, at three
or four forward evaluations against six. That ordering is not guaranteed — it
depends on how fine a ladder the sampling form needs, which is the subject of
{doc}`04-tempering-schedules` — but the optimization form asks less of the
ladder, so it is the usual direction.

## The one combination to avoid

A stopping rule fires on the misfits, and does not know whether the schedule
has a budget. So it can end a sampling run partway up the ladder:

```python
trap = run(state, problem.forward, problem.y, problem.noise_cov,
           schedule=AdaptiveESSSchedule(),          # budget: beta = 1
           stop=DiscrepancyStop(tau=1.0))           # fires on the misfit

trap.beta                                # 0.0677
trap.stop_fired                          # True
trap.budget_complete                     # False
trap.ensemble.std(axis=0, ddof=1)        # [0.1504  0.1431]
```

The run ended at $\beta = 0.068$, which is neither destination. Its ensemble
describes a distribution a small fraction of the way up the ladder, and its
spread is about four times the target's. Nothing raises, because both
arguments are legal. The pair `stop_fired=True` with `budget_complete=False`
is what says it happened — which is why there is no single `converged` field.

## What "exact" means here

The conditioning inside each step is exact. The method as a whole is not, and
it is worth knowing precisely where the line is.

A run reproduces the target distribution exactly only when all of the
following hold: the forward model is affine, the prior is Gaussian, the initial
ensemble's mean and covariance equal the prior's exactly, the update is
`TransformUpdate`, there is no inflation, no member's evaluation failed, and
the increments sum to exactly 1. Every clause is load-bearing, and the last is
easy to miss — `FixedSchedule.uniform(T)` sums to 1 only to round-off.
{doc}`../eki-contract` states the list normatively.

The decay model is not affine, so this run's answer is an approximation with
no guarantee attached. Two consequences to carry:

- **Two ladders give two answers, and neither bounds the other.** A finer
  ladder makes each step's Gaussian fit more accurate, and also accumulates
  more sampling error and collapses the ensemble further. Refinement is not
  monotone improvement. {doc}`04-tempering-schedules` measures this.
- **Members are not independent draws.** The update couples them. Uncertainty
  lives in the ensemble's spread, not in any individual member.

If you want to check the machinery rather than illustrate it, use a problem
with an exact answer. `toy.linear_gaussian(u_dim=…, v_dim=…)` is affine with a
Gaussian prior, and its `posterior()` returns the exact target at
$\beta = 1$ — or `posterior(beta=…)` at any level, so an intermediate step is
checkable too. {doc}`../user-guide/toy-models` works that comparison
through.

## Next

Both forms depend on the ladder, and this page has taken it as given. How far
each step should go — and what a ladder that is too coarse costs, which is
more than it looks — is {doc}`04-tempering-schedules`.
