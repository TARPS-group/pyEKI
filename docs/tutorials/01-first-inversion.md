# 1. Your first inversion

You have a model that turns parameters into predictions, and observations of
those predictions that are noisy. You want to know which parameter values are
consistent with the observations, and how much room the observations leave.

This page does that calculation from start to finish. It assumes Python and
NumPy, and nothing about Ensemble Kalman Inversion. Everything else is
introduced where it is needed.

## What the method needs from you

Four things:

- a **forward model** — a function from parameters to predicted observations;
- the **observations**, as one vector;
- an **observation error covariance** — how much noise is in each
  observation, and whether the errors are related to each other;
- a **prior** — the range of parameter values you would have considered before
  seeing the observations.

The forward model is only ever called. No derivatives, no adjoint, no access
to its internals. It can be a Python function, a compiled simulator, or a job
submitted to a cluster.

What comes back is not one best parameter vector. It is an **ensemble**: a
collection of parameter vectors, whose spread is the answer's uncertainty.
Nothing in this method estimates uncertainty separately from the ensemble, so
the ensemble is the answer.

## A problem to work with

`pyeki.toy` ships three small problems, so that everything on this page can be
run with no model and no data of your own. This page uses the decay problem:
two parameters, an amplitude and a decay rate, observed at twelve times.

```python
import pyeki  # enables float64; import this before creating any array
import jax
import jax.numpy as jnp
from pyeki import toy

problem = toy.exponential_decay()

problem.u_dim     # 2 -- the number of parameters
problem.v_dim     # 12 -- the number of observations
problem.u_true    # [2.  1.5] -- the amplitude and rate the data came from
problem.y[:4]     # [1.3705  0.929   0.6856  0.45  ]
```

:::{important}
Import `pyeki` before creating any array. It switches JAX to 64-bit floating
point. Arrays created before that import stay 32-bit and are **not** promoted
later, and a 32-bit forward model costs accuracy in the answer.
:::

Two parameters is a deliberate choice for a first page: everything about this
problem can be drawn in a plane, so the figures below show what is happening
rather than illustrating it. The parameters are called `u` throughout pyEKI,
and the predictions `v`.

`problem.u_true` exists because the observations here are synthetic — they
were generated from a known amplitude and rate, and then had noise added. Your
own problem will not have it. It is useful here only as a target to check
against.

## The forward model takes the whole ensemble

This is the one mechanical thing to get right, and the one that most often
goes wrong first.

**A forward model receives the whole ensemble at once.** It takes a `(J, P)`
array — `J` members, each a vector of `P` parameters — and returns a `(J, N)`
array of `N` predictions per member. It is called once per step with every
member, never once per member.

For the decay model, $v_i = u_0 e^{-u_1 t_i}$:

```python
times = problem.times          # (12,) -- 0.25 to 3.0, evenly spaced

def forward(ensemble):         # (J, 2) in
    amplitude = ensemble[:, 0:1]                 # (J, 1)
    rate = ensemble[:, 1:2]                      # (J, 1)
    return amplitude * jnp.exp(-rate * times)    # (J, 12) out

ensemble = jnp.array([[2.0, 1.5], [1.0, 0.5]])
forward(ensemble).shape        # (2, 12)
```

The slices are written `0:1` rather than `0` so that they keep their second
axis and broadcast against `times`. This computes the same values as
`problem.forward`, exactly.

If your model is written for one parameter vector at a time — which most
models are — wrap it rather than rewriting it:

```python
def one_member(member):        # (2,) in
    return member[0] * jnp.exp(-member[1] * times)   # (12,) out

forward = jax.vmap(one_member)                       # (J, 2) -> (J, 12)
```

`jax.vmap` only works on a model written in JAX. For a model that is not — a
subprocess, a compiled binary, a call to a job scheduler — the wrapper is a
loop or a parallel map over the rows, and
{doc}`../user-guide/writing-a-forward-model` works one through.

A model with the wrong convention usually fails somewhere inside the update
rather than at the call, with a shape error that does not name the cause. So
check it before running anything:

```python
from pyeki.eki.testing import check_forward_model

check_forward_model(forward, u_dim=2, v_dim=12)      # raises, or returns None
```

That calls your model five times and checks the shape at two ensemble sizes,
the floating-point type, that the model is deterministic, and that each
member's predictions depend only on that member. Use a cheap configuration of
an expensive model.

## The prior and the observation error

Both are covariance matrices, and pyEKI represents a covariance matrix by how
it acts rather than by storing it. For independent errors that is
`PSDDiagonal`, which takes the variances:

```python
from pyeki.gauss import Gaussian
from pyeki.linalg import PSDDiagonal

prior = Gaussian(mean=jnp.array([1.0, 1.0]),
                 cov=PSDDiagonal(jnp.array([1.0, 1.0])))

noise_cov = PSDDiagonal(jnp.full(12, 0.02 ** 2))     # a standard deviation of 0.02
```

The prior says: both parameters are somewhere around 1, give or take about 1.
The observation error says: each of the twelve observations is off by about
0.02, independently of the others. These are exactly `problem.prior` and
`problem.noise_cov`, so the rest of the page uses the problem's own.

Note the units. `PSDDiagonal` takes **variances**, not standard deviations —
`0.02 ** 2`, not `0.02`. Observation error is usually known as a standard
deviation, so this is worth squaring carefully.

Correlated errors, and priors that are not diagonal, are
{doc}`06-covariances-as-operators`. Everything below is unchanged by them.

## One step

Start with a set of parameter vectors drawn from the prior:

```python
from pyeki.eki import EKIState

state = EKIState.from_prior(jax.random.key(0), problem.prior, n_members=64)
state.ensemble.shape         # (64, 2)
```

`jax.random.key` makes the typed keys JAX uses for random numbers. The same
key gives the same ensemble, so this page's numbers are reproducible.

Now the method's one idea. Run every member through the forward model. That
gives 64 pairs: a parameter vector, and the predictions it produced. Fit a
Gaussian distribution to those pairs — the mean and covariance of the
parameters, of the predictions, and between the two. Then ask that Gaussian a
question it can answer exactly: *given that the predictions came out equal to
the observations, what are the parameters?* Conditioning a Gaussian on part of
itself is a closed-form calculation. The answer is a new distribution over the
parameters, and the ensemble is moved so that its mean and spread match it.

```python
from pyeki.eki import FixedSchedule, run

one_step = run(state, problem.forward, problem.y, problem.noise_cov,
               schedule=FixedSchedule.constant(1.0, n_steps=1))

one_step.mean                            # [2.0129  1.6505]
one_step.ensemble.std(axis=0, ddof=1)    # [0.0781  0.7632]
```

```{figure} ../_generated/figures/01-one-step.png
:alt: Four panels: the prior ensemble in the parameter plane, its predictions against the observations, the fitted Gaussian over the decay rate and one prediction with the observed value marked, and the predictions after one conditioning step.
:width: 100%

One conditioning step. **(a)** 64 members drawn from the prior, with the
prior's own contours. **(b)** every member through the forward model, against
the observations and their error bars. **(c)** the fitted Gaussian, for the
decay rate and the prediction at $t = 1$: the members are the points, the
ellipses are the Gaussian fitted to them, and conditioning on the observed
value picks out the band of rates. **(d)** the predictions after that one
step, on the same axes as (b).
```

Both prediction panels are clipped. Three members' predictions at $t = 1$ lie
above the top of panel (b), and nine of the 64 members have a negative decay
rate, which makes their predictions grow rather than decay. That is what a
prior of $\mathcal{N}(1, 1)$ on the rate allows, and it is a fair picture of
what an ensemble drawn from a wide prior looks like.

Panel (c) is the whole method in one picture, and it also shows what the method
approximates. The 64 points are not shaped like an ellipse — they curve, and a
few sit far out — but an ellipse is what gets conditioned. The conditioning
itself is exact; fitting that ellipse is where the approximation enters. The
other place it enters is the ellipse's own coefficients, which are estimated
from 64 members rather than known.

The fit is not a metaphor. Doing it by hand gives the same numbers as the step
above, to floating point:

```python
from pyeki.gauss import GaussianJoint

joint = GaussianJoint.from_samples(u_samples=state.ensemble,
                                   v_samples=problem.forward(state.ensemble))
conditioned = joint.condition(problem.y, problem.noise_cov)

conditioned.mean                     # [2.0129  1.6505]
conditioned.cov.diag() ** 0.5        # [0.0781  0.7632]
```

## Why one step is not enough

One step already found roughly the right amplitude: 2.013 against a true 2.0.
The decay rate is the problem. Its standard deviation is 0.76, which is most
of the prior's own spread of 1.0 — the step barely narrowed it. In prediction
space, panel (d), the fan of curves is much wider than the error bars: at
$t = 1$ the members' predictions have a standard deviation of 0.578, nearly
thirty times the observation error of 0.02.

The reason is visible in panel (c). Over the prior's whole range the
relationship between the rate and the predictions is strongly curved, and one
ellipse cannot describe it. The fit is worst exactly where it is being asked
to do the most work.

So the observations are not assimilated in one go. Instead the method builds a
sequence of intermediate distributions between the prior and the answer,

$$\pi_\beta(u) \;\propto\; \pi_0(u)\,
  e^{-\beta \Phi(\mathcal{G}(u))},
  \qquad \Phi(v) = \tfrac12 \lVert W(y - v)\rVert^2 ,$$

where $\pi_0$ is the prior, $\mathcal{G}$ is the forward model, and $\Phi$ is
the **misfit**: half the sum of a member's squared prediction errors, each
divided by that observation's error standard deviation. At $\beta = 0$ this is
the prior. At
$\beta = 1$ it is the distribution the observations imply — the target. In
between, the observations are counted at partial weight.

A run walks up that sequence in steps. Each step takes the ensemble from one
value of $\beta$ to a slightly larger one, and each step is one conditioning
step of the kind above, applied with the observation error covariance divided
by that step's increment — a small increment means noisier observations, and so
a smaller move. Because each step covers a short distance, the Gaussian fit
only has to be good locally, which is where it is good.

The values of $\beta$ are called **tempering levels**, and the sequence of
them is a **ladder**. How far each step should go is
{doc}`04-tempering-schedules`, and it matters: the single step above and the
six-step ladder below differ by a factor of 21 in the decay rate's spread.

## The whole run

`AdaptiveESSSchedule` picks the increments from the ensemble as it goes, and
is the sensible default:

```python
from pyeki.eki import AdaptiveESSSchedule

result = run(state, problem.forward, problem.y, problem.noise_cov,
             schedule=AdaptiveESSSchedule())

result.status            # 'schedule_exhausted' -- the ladder reached beta = 1
result.n_evaluations     # 6 -- forward model calls
result.beta              # 1.0
```

Six calls of the forward model, each on all 64 members: 384 member
evaluations, which is the cost that matters when a model is expensive.

Note the call. `run` takes `forward`, `y` and `noise_cov` as **three separate
arguments**. A toy problem is a container for setting one up, not something
you hand to `run`; it is not even callable. When you replace the toy problem
with your own model, those three arguments are what you replace.

## Reading the answer

```python
result.ensemble.shape                 # (64, 2)
result.mean                           # [1.9802  1.4741]
result.ensemble.std(axis=0, ddof=1)   # [0.0396  0.0363]
```

The true parameters were `[2.0, 1.5]`. The answer's mean is 0.020 and 0.026
away from them, and its standard deviations are 0.040 and 0.036 — so the truth
is within one standard deviation in both parameters. The decay rate went from a
standard deviation of 0.76 after one step to 0.036 after the ladder, a
narrowing of a factor of 21 for five more forward evaluations.

```{figure} ../_generated/figures/01-answer.png
:alt: Two panels: the ensemble's predictions against the observations, and the final ensemble in the amplitude-rate plane with the true parameters marked.
:width: 100%

The finished run. Left, the 64 members' predictions against the observations —
the fan is now narrower than the error bars, with a standard deviation of
0.0099 at $t = 1$ against an observation error of 0.02. Right, the members in
the parameter plane, with the parameters the data came from marked. The
ensemble is tilted because the amplitude and the rate are not separately
identified by these observations: a larger amplitude with a faster decay fits
about as well.
```

Read the ensemble, not just the mean. `result.mean` is a summary; the spread
and the shape are the part that says how much the observations actually
determined. Here they say that the two parameters cannot be pinned down
independently, which no single number reports.

An ensemble's members are **not** independent draws from the target
distribution. The update couples them, and that coupling is the sampling error
that the ensemble size controls. Treat the spread as the answer and the
individual members as the machinery.

## What this page left out

The answer above is an approximation. It is exact only in a case this problem
is not — an affine forward model and a Gaussian prior, among other conditions
that {doc}`03-sampling-or-optimizing` lists in full. There is no guarantee
attached to it, and different choices of ladder give different answers.

That makes the next question how to tell whether a given answer is any good,
which is {doc}`02-reading-a-run`. Then:

- {doc}`03-sampling-or-optimizing` — the two things a run can be aiming at,
  and which one you want;
- {doc}`04-tempering-schedules` — how far each step should go, and what a
  ladder that is too coarse costs;
- {doc}`05-transform-or-pathwise` — the two ways a step can move the ensemble;
- {doc}`06-covariances-as-operators` — correlated observation error, and
  non-diagonal priors;
- {doc}`08-when-the-model-fails` — forward models that crash or return `nan`,
  which is most real ones.

{doc}`../user-guide/running-an-inversion` is the same material organized by
question rather than in sequence, and is the page to return to once you know
what you are looking for.
