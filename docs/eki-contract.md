# Ensemble Kalman Inversion contract

This page specifies `pyeki.eki`: the objects it provides, the contract of every
method, and the iteration that all of them serve. It is normative — an
implementation that violates a rule here is defective even if its tests pass —
and it is the reference for two audiences: contributors implementing or
reviewing the layer, and users who want a more precise account of what a pyEKI
run actually computes than the user guide gives.

Throughout, *must* and *never* state requirements, *should* states a strong
default that a documented reason may override, and *may* states a permission.
{doc}`design` records *why* the load-bearing decisions were made; this page
records *what* they require. The layer is built on {doc}`linop-contract` and
{doc}`gaussian-contract`, and references both freely rather than restating
them.

:::{admonition} Status: specification ahead of code
:class: important

`pyeki.eki` does not exist yet. This document is the design it will be built
to, and it is the artifact to review and iterate on before implementation
begins. Once the module ships, this page remains as the normative reference
for its behaviour.
:::

(eki-scope)=
## Scope

The layer is the algorithmic top of the package: it turns the Gaussian
conditioning of {doc}`gaussian-contract` into a *run* — an initial ensemble, a
ladder of tempered targets, an ensemble update per rung, and a record of what
happened. It provides

- **the iteration** itself, as a driver that owns the loop, the temperature
  bookkeeping, the pseudo-random number stream, and the failure handling;
- **tempering schedules**, fixed and adaptive, which decide how far each step
  moves;
- **ensemble updates**, stochastic and deterministic, which are thin
  assemblies over `pyeki.gauss`;
- **inflation**, which maintains ensemble spread;
- **stopping rules** and **per-step diagnostics**.

Two well-known modes of EKI fall out of the same driver, and the layer is
designed so that neither is privileged ({ref}`eki-axes`):

- the **sampling form**, which draws from the prior and runs a finite ladder of
  tempered targets summing to $\beta = 1$, producing an approximate posterior
  ensemble; and
- the **optimization form**, which runs the same update without a temperature
  budget until the data are fit, producing a collapsing ensemble whose centre
  approximates a regularized least-squares solution.

Deliberately outside the layer: the forward model (any callable, per the
package's permanent scope boundary), parameter transformations and
constraints, and localization (`pyeki.localize`, which plugs in as an update
rule). {ref}`eki-excluded` lists what else is left out and why.

**This layer implements no covariance arithmetic of its own.** It *calls*
`whiten`, `factor` and the conditioning methods of the layers below — the
diagnostics whiten a $(J, N)$ residual block every step
({ref}`eki-diagnostics`), and `AdditiveInflation` requires a factor
({ref}`eki-inflation`) — but every such operation is dispatched to
`pyeki.linalg` or `pyeki.gauss`, and nothing here decomposes, inverts or
assembles a covariance. What this layer computes itself is elementwise:
whitened residuals' norms, log-space weight ratios, and mean-and-subtract on an
ensemble. A step that assembled a covariance, or that re-derived any part of
the conditioning kernel, would be a layering violation
({ref}`eki-updates`).

The implementation PR must also add the layer's user-guide page, per the
package rule that every user-facing feature has a user-guide home; this page
remains the deeper reference.

(eki-notation)=
## Notation and conventions

The notation of {doc}`gaussian-contract` carries over unchanged: $u$ is the
parameter block of dimension $P$, $v$ the predicted-observation block of
dimension $N$, $y$ the observation, $R$ the observation-noise covariance as a
{class}`~pyeki.linalg.PSDLinOp`, $W$ a whitener of $R$, and $J$ the number of
ensemble members. Ensembles are stored row-wise as `(J, dim)` arrays.
Additional symbols:

| symbol | meaning |
| ------ | ------- |
| $G$ | the forward model, a callable from parameters to predicted observations |
| $\beta$ | the accumulated tempering level, $\beta \ge 0$ |
| $\Delta\beta_t$ | the increment taken at step $t$, strictly positive |
| $\Phi_j$ | the whitened misfit of member $j$ (below) |
| $T$ | the number of completed steps of a run |

Conventions, each normative:

- **The misfit carries the factor $\tfrac12$.** For a prediction $v$,

  $$
  \Phi(v) \;=\; \tfrac{1}{2}\,\bigl\lVert W(y - v) \bigr\rVert^2
  \;=\; \tfrac{1}{2}\,(y-v)^\top R^{-1} (y-v),
  $$

  computed against the **base** noise covariance $R$, never a tempered one.
  Every criterion, diagnostic and stopping rule in the layer is written in
  terms of this quantity, so that the convention is fixed in exactly one
  place. The value is whitener-invariant by the operator layer's `whiten`
  guarantee, so it does not depend on which valid $W$ the noise operator
  chose. The factor of $\tfrac12$ is what makes $e^{-\beta\Phi}$ the tempered
  likelihood and $2\overline{\Phi} \approx N$ the well-specified-fit
  benchmark; halving or doubling it silently rescales every schedule
  parameter, which is why it is pinned here.
- **The tempering variable is a level, and steps take increments.** State
  carries $\beta$; a step takes $\Delta\beta$ and conditions with per-step
  noise $R/\Delta\beta$ — **never** $R/\beta$
  ({ref}`eki-iteration`).
- **Vectors passed to the layer are exactly core-shaped**, as in
  `pyeki.gauss`: `y` is a `(N,)` array, an ensemble a `(J, P)` array. The
  batched exception is `misfits`, which follows the operator layer's
  batch contract.
- **PRNG keys are typed keys** — the output of `jax.random.key`, of shape
  `()`. Raw `uint32` key arrays are rejected at construction
  ({ref}`eki-validation`), because a stored key of shape `(2,)` would make
  the `batch_shape` of the state ambiguous.
- **A run's driver loop is ordinary Python, not `lax.scan`.** The forward
  model is an arbitrary callable — possibly a subprocess, a job-scheduler
  submission, or a non-traceable legacy code — so it can never be traced.
  Every *array* computation in the layer is nonetheless `jit`- and
  `vmap`-safe with static shapes ({ref}`eki-jax`).

(eki-iteration)=
## The iteration

Everything in the layer serves one iteration, specified here once.

### The tempered family

For a prior $\pi_0$ on $\mathbb{R}^P$, the layer's family of targets is

$$
\pi_\beta(u) \;\propto\; \pi_0(u)\, e^{-\beta \Phi(G(u))},
\qquad \beta \ge 0 ,
$$

which is the prior at $\beta = 0$, the Bayesian posterior at $\beta = 1$, and
increasingly concentrated on the minimizers of $\Phi \circ G$ as
$\beta \to \infty$.

The bridge to Gaussian conditioning is an identity, not an approximation: for
any $\delta > 0$,

$$
\frac{\pi_{\beta+\delta}(u)}{\pi_\beta(u)}
\;\propto\; e^{-\delta \Phi(G(u))}
\;\propto\; \mathcal{N}\!\bigl(y;\, G(u),\, R/\delta\bigr) ,
$$

so **moving one increment up the ladder is conditioning on the same
observation with the noise covariance divided by that increment.** This is why
`pyeki.gauss` needs nothing new to serve tempering, and why the noise
interface is `whiten` only: $R/\delta$ is a
{class}`~pyeki.linalg.PSDScaled`, whose whitener is the base whitener scaled
by $\sqrt{\delta}$, so a traced increment flows through without refactorizing
anything.

### One step

Step $t$ carries the ensemble from level $\beta_t$ to
$\beta_{t+1} = \beta_t + \Delta\beta_t$:

1. evaluate the forward model on every member, $v_j = G(u_j)$;
2. form the joint Gaussian of the pairs $(u_j, v_j)$ — an
   {class}`~pyeki.gauss.EnsembleJoint`;
3. condition it on $y$ with noise $R/\Delta\beta_t$, obtaining an updated
   parameter ensemble.

Step 3 is exact Gaussian conditioning of a Gaussian *fitted to* the ensemble.
Two approximations are inherent to the step: the joint law of $(u, G(u))$ under
$\pi_{\beta_t}$ is replaced by a Gaussian, and its moments are replaced by
$J$-member empirical estimates. Both are exact when $G$ is affine and the
ensemble's empirical moments equal $\pi_{\beta_t}$'s, which is what makes the
exactness claim of {ref}`eki-honesty` reachable at all.

Those two are the *only* inherent ones, but they are not the only ones a
configured run makes. The stochastic update adds Monte Carlo noise of order
$KRK^\top/J$ per realization ({ref}`eki-updates`); inflation adds spread on
purpose ({ref}`eki-inflation`); and repairing failed members conditions on a
damped covariance ({ref}`eki-failures`). Each is opt-in or opt-out and each is
recorded where it arises; {ref}`eki-honesty` collects the consequences.

### Telescoping

Per-step precisions add. For an affine $G$ and a Gaussian prior, conditioning
with $R/\Delta\beta_t$ at each rung contributes
$\Delta\beta_t\, G^\top R^{-1} G$ to the posterior precision, so after $T$
steps the accumulated precision is
$\Lambda_0 + \bigl(\sum_t \Delta\beta_t\bigr) G^\top R^{-1} G$: **the ladder
composes to one-shot conditioning at level $\beta_T$**, and at $\beta_T = 1$
to the exact posterior. This identity is the layer's central correctness
property and its first conformance obligation ({ref}`eki-conformance`).

:::{warning}
Using $R/\beta_t$ — the accumulated level — instead of $R/\Delta\beta_t$ is
the layer's signature silent failure. It raises nothing, produces a
plausible-looking posterior, and gets the answer wrong by an amount that
*grows with ladder length*. On a uniform $T$-rung ladder the level form
accumulates $\sum_{t=1}^{T} t/T = (T+1)/2$ times the data precision instead of
one — a factor of 3 at $T = 5$ and 5.5 at $T = 10$ — so it is exact only at
$T = 1$ and diverges from there. Conformance test 2 pins this against a named
fixture.
:::

The identity holds exactly for the ensemble's empirical moments under the
square-root update, and in expectation under the stochastic update
({ref}`eki-updates`). Inflation breaks it deliberately
({ref}`eki-inflation`).

(eki-subspace)=
### The subspace property

For both shipped updates the new members satisfy
$u_j' \in \bar u + \operatorname{span}\{u_k - \bar u\}$: the update is a
combination of the ensemble's own anomalies
({doc}`gaussian-contract`). By induction over steps, **every iterate of a run
lies in the affine subspace spanned by the initial ensemble**, whose dimension
is at most $J - 1$, however many steps are run and however the schedule is
chosen.

Three consequences worth stating plainly, because they explain the rest of the
layer's surface:

- $J$ bounds what a run can represent, not merely how accurately it estimates
  moments. Directions absent from the initial ensemble are unreachable.
- Multiplicative inflation preserves the subspace; **additive inflation is the
  only shipped mechanism that leaves it** ({ref}`eki-inflation`).
- Localization escapes it a different way, by giving each parameter block its
  own weight vector, so the global update is no longer a single combination of
  whole-ensemble anomalies (`pyeki.localize`).

(eki-honesty)=
### What the layer does not promise

A library that reports a posterior must be precise about what that word means
here.

- **Exactness is claimed for the affine-Gaussian case only.** With an affine
  $G$, a Gaussian prior, an ensemble whose empirical moments equal the
  prior's, the square-root update, no inflation, no failed members, and a
  ladder whose increments sum **exactly** to 1, the run reproduces the exact
  posterior mean and covariance, to floating point. That is the whole
  exactness claim, and it is what the conformance suite checks. Every clause
  is load-bearing, and the last one is easy to miss:
  `FixedSchedule.uniform(T)` sums to 1 only to round-off
  ({ref}`eki-schedules`), so it satisfies the claim to round-off and not
  better.
- **For nonlinear $G$ the output is an approximation with no consistency
  guarantee.** It depends on the schedule, the ensemble size, and the update
  rule; two different ladders give two different answers, and neither is a
  bound on the other. A finer ladder makes each step's Gaussian approximation
  more accurate but accumulates more sampling error and collapses the
  ensemble further, so refinement is not monotone improvement.
- **Members are not independent posterior draws.** With the *exact* gain, the
  stochastic update is Matheron's rule and would return independent draws;
  the empirical gain couples the members, and that coupling is precisely the
  sampling error that $J$ controls. Uncertainty is carried by the ensemble's
  spread, not by any individual member.
- **The optimization form deliberately collapses the ensemble.** Its terminal
  spread is a measure of numerical convergence, not of posterior uncertainty,
  and reporting it as an uncertainty is a misuse the layer cannot prevent.
- **Nothing in the loop re-consults the prior.** The prior enters once,
  through the initial ensemble, which is why a prior covariance with no cheap
  inverse is perfectly usable ({doc}`design`) — and why, as $\beta$ grows past
  1, the prior's relative weight decays and the iteration becomes an
  optimizer. Two constructions deliberately reintroduce it, and both then
  require more of the prior covariance than a bare run does: the Tikhonov
  augmentation needs `whiten`, and a Langevin-type update needs `solve`
  ({ref}`eki-variants`).
- **The three departures from the ladder are opt-in, and each is named where
  it happens.** Inflation widens the ensemble on purpose and breaks
  telescoping ({ref}`eki-inflation`); the stochastic update adds Monte Carlo
  noise, so the identity holds in expectation rather than per realization
  ({ref}`eki-updates`); and repairing failed members conditions on a
  covariance damped by $(J_v-1)/(J-1)$ ({ref}`eki-failures`). A run's
  configuration therefore determines which of the claims above survive, and
  the default configuration — square-root update, no inflation — keeps all of
  them.
- **A run's total cost is not knowable in advance under an adaptive
  schedule.** The number of rungs, and hence of forward-model evaluations,
  depends on the misfits the model produces. A caller who must budget
  evaluations uses a `FixedSchedule`, whose cost is exactly its length, or
  drives `step` directly ({ref}`eki-step`). Exceeding `max_steps` raises
  rather than returning a partial answer — and the exception carries the
  state and the history, so the work is not lost ({ref}`eki-driver`).

(eki-axes)=
## Three orthogonal choices

The design's organizing claim is that a variant of EKI is a choice on three
independent axes, and that the driver is the same in every case.

| axis | object | question it answers |
| ---- | ------ | ------------------- |
| how far, and when to stop | `Schedule`, `StoppingRule` | what is $\Delta\beta_t$, and when does the run end? |
| how the ensemble moves | `EnsembleUpdate` | given the increment, what is the new ensemble? |
| how spread is maintained | `Inflation` | what happens to the ensemble before each forward evaluation? |

The two forms of EKI named in {ref}`eki-scope` are therefore **not two
drivers, and not a flag**: they are two schedules.

| | sampling form | optimization form |
| --- | --- | --- |
| schedule | a budget: increments sum to $\beta = 1$ | no budget: $\beta$ grows without bound |
| ends when | the budget is exhausted | a stopping rule fires |
| update | either | either |
| inflation | usually none | often used, to delay collapse |
| interpretation | approximate posterior ensemble | collapsing ensemble around a regularized fit |

```python
from pyeki.eki import (
    AdaptiveESSSchedule, DiscrepancyStop, EKIState, FixedSchedule, run,
)

state = EKIState.from_prior(key, prior, n_members=64)

# Sampling form: adaptive ladder, budget 1, deterministic update (default).
sampled = run(state, forward, y, noise_cov,
              schedule=AdaptiveESSSchedule())
ensemble = sampled.ensemble          # (J, P) approximate posterior ensemble
centre = sampled.mean                # (P,) its mean

# Optimization form: unit steps, no budget, stop on the discrepancy principle.
fit = run(state, forward, y, noise_cov,
          schedule=FixedSchedule.constant(1.0, n_steps=200),
          stop=DiscrepancyStop(tau=1.0))
assert fit.stop_fired                # False means the ladder ran out first
```

Neither result is named `posterior`. `sampled.ensemble` is an approximate
posterior ensemble under the caveats of {ref}`eki-honesty`, and `fit.ensemble`
is a collapsing ensemble around a regularized fit, which is not a posterior at
all; naming either one `posterior` in a script invites reporting the second as
though it were the first.

Each axis is a **protocol**, not a base class, and each has shipped
implementations. This is the one place where pyEKI is deliberately open to
extension at the algorithm level: `pyeki.linalg` is extended by writing an
operator, `pyeki.gauss` is closed, and `pyeki.eki` is extended by writing a
schedule, an update rule, or an inflation. {ref}`eki-variants` works through
what that buys.

### Orthogonal mechanically, coupled in meaning

The axes compose without restriction: every schedule works with every update
and every inflation, and the driver never inspects one to decide another. That
independence is mechanical, and it is deliberate — a driver that rejected
combinations would be encoding a taste rather than a contract.

It is **not** a claim that every combination is meaningful, and the layer
cannot tell the difference. A short guide, which the implementation must not
turn into a runtime check:

| combination | reading |
| ----------- | ------- |
| budgeted schedule + either update, no inflation | the sampling form; every claim of {ref}`eki-honesty` applies |
| unbounded schedule + a stopping rule | the optimization form; the terminal spread is not an uncertainty |
| budgeted schedule + inflation | a deliberately widened variant; not a tempering ladder for the target family |
| `DiscrepancyStop` on a budgeted ladder | constructible and usually a mistake: it can end a sampling run at $\beta = 0.4$, whose ensemble is neither a posterior nor a fit |
| `AdaptiveESSSchedule` with a non-Kalman update | the ESS criterion presumes a tempering $\beta$ and a conditioning step; against a Langevin-type update it is a step-size rule with no interpretation |

The one combination worth calling out as a trap is the fourth. A stopping rule
fires on the misfits regardless of how much budget remains, so pairing one with
a budgeted schedule silently converts a sampling run into an early exit at an
arbitrary intermediate level. `EKIResult.status` records which happened, and
`EKIResult.beta` records where the run stopped; nothing else will warn.

(eki-objects)=
## The objects

| object | kind | role |
| ------ | ---- | ---- |
| `EKIState` | pytree | the loop-carried state: ensemble, level, step index, key |
| `Evaluation` | pytree | everything one forward evaluation produced: the members, their predictions, and the whitened residuals |
| `HistoryRecord` | pytree | one row of the run's history: scalars only |
| `EKIResult` | frozen dataclass | the final state, the history, and why the run ended |
| `EnsembleUpdate` | protocol | one ensemble update, given an increment |
| `Schedule` | protocol | the increment, plus the two attributes that say how long the ladder is |
| `StoppingRule` | protocol | whether to stop, given the current misfits |
| `Inflation` | protocol | a transformation of the ensemble, applied before each forward evaluation |
| `run`, `iterate`, `step` | functions | the driver, as a function, as a generator, and as one iteration |
| `misfits`, `effective_sample_size`, `repair_failed_members` | functions | the array-level pieces schedules and custom drivers need |

Rules governing the set:

1. **The value classes are unbatched frozen pytrees, exactly like operators
   and gauss objects.** Construction, validation, family and JAX-integration
   rules are inherited verbatim ({ref}`eki-jax`). `EKIResult` is the one
   exception: it is a report, never an argument to traced code, so it is a
   plain frozen dataclass.
2. **The policy protocols are structural.** An implementation is anything with
   the right call signature — a shipped class, a user's frozen dataclass, or a
   plain function where the protocol has a single method. Nothing subclasses
   anything.
3. **Policies are pure and stateless.** A schedule, stopping rule, update rule
   or inflation must be a pure function of its arguments and its own frozen
   fields, and must not carry iteration state. This is what makes a run
   resumable from an `EKIState` alone ({ref}`eki-state`) and reproducible;
   it is why `Schedule` receives the step index instead of counting calls.

(eki-state)=
## `EKIState`

Everything a run needs in order to continue.

**Fields**, in declaration order: `ensemble`, a `(J, P)` array; `beta`, a 0-d
array; `step`, a static `int`; and `key`, a typed PRNG key of shape `()`.
Construction validates exact ranks, $J \ge 2$ and $P \ge 1$, `step >= 0`, and
that `key` is a typed key — a check on **shape and dtype** (tier 2), since a
typed key and a 0-d float array agree on shape, and only
`jnp.issubdtype(key.dtype, jax.dtypes.prng_key)` separates them. A Python float or 0-d array is accepted for `beta`
and converted to a 0-d float array before storing, as the operator layer's
scalar dunders do; `beta` must not be negative.

**Derived attributes.** `n_members` and `u_dim` are `int` properties, named as
in {doc}`gaussian-contract`; `mean` is the ensemble mean, a `(P,)` array, the
same property `EKIResult` exposes and delegates to — a caller in an `iterate`
loop holds states, not results.

**`EKIState.restart()`** returns a copy with `step = 0` and `beta = 0.0`: a
state carrying the same ensemble and key, ready to begin a *new* ladder rather
than resume the old one. It exists because the alternative is
`dataclasses.replace(state, step=0, beta=0.0)`, which requires knowing that
`step` is a counter a schedule indexes and that a budgeted schedule exhausts on
`beta`, and which is the remedy for a trap that otherwise raises nothing — see
the warning below.

**`EKIState.from_prior(key, prior, n_members)`** draws the initial ensemble
from the prior — the layer's one piece of work that is neither storing a field
nor validating one, and therefore the only reason the class has an alternate
constructor at all. It is a classmethod rather than logic inside `EKIState`,
per the operator layer's rule that constructors store and classmethods compute
({ref}`contract-jax`): the dataclass constructor is what pytree unflattening
bypasses, so anything it computed would silently vanish at a trace boundary.
Sampling in particular must not live there, since it would redraw the ensemble
on every reconstruction.

`prior` is a {class}`~pyeki.gauss.Gaussian`; the draw is pinned as

```python
key_sample, key_state = jax.random.split(key)
EKIState(prior.sample(key_sample, n_members), 0.0, 0, key_state)
```

so the initial ensemble is exactly `Gaussian.sample`'s pinned draw and the
state's stream is independent of it. Requires `prior.cov.supports("factor")`;
an unsupported covariance raises `UnsupportedOpError` from the operator layer,
unmodified.

A warm start — an ensemble from a previous run, a Latin hypercube, a
hand-built design — is direct construction with `beta=0.0`, `step=0`. Nothing
in the layer requires that the initial ensemble came from the prior; what the
prior contributes is only its samples, and the tempered family's $\pi_0$ is
whatever the initial ensemble represents.

**Resumption.** `run(state, ...)` on a state returned by a previous run
continues it: same schedule, same policies, and the tail of the run is
bit-identical to an uninterrupted one ({ref}`eki-conformance`). This is the
sole mechanism for checkpointing, and it is why policies may not hold
iteration state.

:::{warning}
**`step` is cumulative across runs, and position-dependent schedules read it.**
Resuming a partially-completed ladder is the case this is designed for: a
`FixedSchedule` interrupted after four of ten rungs resumes at rung four
because `state.step` is 4.

The same property makes *chaining a second, different ladder* onto a finished
state a silent no-op. After `FixedSchedule.uniform(10)` completes,
`state.step == 10`; handing that state to a fresh `FixedSchedule.uniform(10)`
finds `step >= n_steps` already true, and `run` returns immediately with
`status="schedule_exhausted"`, an empty history, and the ensemble unchanged.
Nothing raises, because an already-finished ladder legitimately returns.

A new ladder therefore needs a new counter, which is what `restart` is for:

```python
phase2 = state.restart()          # step = 0, beta = 0.0, same ensemble and key
```

Adaptive *budgeted* schedules are not immune, for the mirror-image reason:
their exhaustion reads `beta` rather than `step`, so a finished budget is
equally finished on re-entry. `restart` resets both, which is why it resets
both.

**The driver also warns.** A run that returns with `n_steps == 0` performed no
work at all, which is essentially never what the caller wanted, so the driver
logs it at `WARNING` ({ref}`eki-driver`). That does not make the trap
impossible — a run legitimately returns when its ladder is finished — but it
means the silent case is no longer silent.

`EKIResult.n_steps` counts the records of that run, not `state.step`; after a
resumption the two differ, and the histories concatenate in call order.
:::

(eki-step)=
## The step

One iteration, as **two public phases** and one convenience that composes them.
`iterate` and `run` are loops over the same two phases.

| function | does | returns |
| -------- | ---- | ------- |
| `evaluate` | inflate, call the forward model, repair, summarize | an `Evaluation` |
| `apply` | validate the increment, update, check finiteness, advance | `(EKIState, HistoryRecord)` |
| `advance` | `evaluate` then `apply`, at a given increment | `(EKIState, HistoryRecord)` |

### `evaluate(state, forward, y, noise_cov, *, inflation=None, on_failure="repair")`

Runs the forward model once and returns everything that evaluation produced. It
takes **no increment** and moves nothing: `state` is untouched, and the level
and the step index are carried into the result unchanged.

This is the phase that costs a forward-model evaluation, and making it public
separately is what lets a caller spend that resource deliberately. Three
consumers need it:

- **Backtracking and damping.** One `Evaluation` serves any number of trial
  increments, because `apply` takes it as an argument
  ({ref}`eki-step-backtracking`).
- **Consulting a policy outside the driver.** A `Schedule` or `StoppingRule`
  takes an `Evaluation`, so a caller driving the loop by hand can use the
  shipped policies rather than reimplementing their criteria.
- **Diagnostics the record does not carry.** The singular values of $S$, the
  per-member misfits, the posterior predictive — all live on the `Evaluation`
  ({ref}`eki-diagnostics`).

### `apply(state, evaluation, *, increment, y, noise_cov, update=TransformUpdate())`

Moves `state` forward by the given increment, using an `Evaluation` already
obtained from it. Chooses nothing: no schedule, no stopping rule, no
`max_steps`. Those are the driver's, and everything they decide is passed in.

**`evaluation` must belong to `state`.** `apply` checks
`evaluation.step == state.step` and `evaluation.beta == state.beta` and raises
`ValueError` otherwise. Without the check, pairing an evaluation with a
different state would take the key split from one and the members from the
other and return finite, plausible nonsense.

`y` and `noise_cov` are passed again rather than carried on the `Evaluation`,
which holds no problem data — it is a record of an evaluation, not a bound
problem. This is also what makes `apply` the entry point for anything that
varies the *data* between rungs, a subsampled or randomized observation vector
among them, which the driver deliberately fixes for a whole run
({ref}`eki-excluded`).

### `advance(state, forward, y, noise_cov, *, increment, update=TransformUpdate(), inflation=None, on_failure="repair")`

Exactly `apply(state, evaluate(state, forward, y, noise_cov, inflation=...,
on_failure=...), increment=increment, y=y, noise_cov=noise_cov, update=...)`,
provided because one rung at a known increment is the common case. Named
`advance` rather than `step` because `step` is an index on three classes and a
keyword argument, and `step = step(...)` is a shadowing mistake waiting to
happen.

(eki-step-backtracking)=
### Backtracking, as the two phases make it

The damped and backtracking family — propose an increment, judge it, accept it
or shrink and retry — is why the phases are public. States are immutable, so
re-applying from the same state with a smaller increment is exactly "reject and
retry", and the rejected trial costs **no** re-evaluation at the current state:

```python
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

The accepted branch reuses `probe` as the next rung's evaluation, so the
steady-state cost is **one forward evaluation per rung plus one per rejection**
— which is what backtracking costs in any implementation. The layer neither
hides that nor pays it for callers who do not want it. A loop written against
`advance` alone would instead re-evaluate the current state on every trial,
doubling the cost, which is why the phases and not just their composition are
public.

**The order of operations is normative.** Steps 1–5 are `evaluate`; steps
0 and 6–9 are `apply`.

1. **Split the key**, always into three, whatever the policies are:

   ```python
   key_next, key_inflate, key_update = jax.random.split(state.key, 3)
   ```

   Fixed arity means turning inflation on or off does not shift the update's
   random stream ({ref}`eki-prng`). **Both phases perform this split
   independently** from the same `state.key`, rather than `evaluate` handing
   `key_update` and `key_next` on through the `Evaluation`. The split is
   deterministic, so they agree; it costs nothing; and it keeps the
   fixed-arity rule stated in one place instead of putting the update's key on
   an object that every `Schedule` and `StoppingRule` receives.
2. **Inflate.** `u = inflation(key_inflate, ensemble=state.ensemble,
   step=state.step, beta=state.beta)`, or `u = state.ensemble` bit-exactly
   when `inflation is None`. Inflation happens here — before the forward
   evaluation, on the ensemble that is about to be updated — so that
   predictions always correspond to the members they will update, and so that
   the ensemble a run *returns* is never an inflated one
   ({ref}`eki-inflation`).
3. **Evaluate the forward model.** `v = forward(u)`, the only non-JAX call in
   the layer. Its result must be an array of shape `(J, N)`; anything else
   raises `ValueError` naming the expected and received shapes. This check is
   static and runs every step.
4. **Validity and repair.** A member is *invalid* when its prediction row
   contains any non-finite entry. With $J_v$ the number of valid members:
   if $J_v = J$, the ensemble and predictions pass through **bit-exactly
   untouched**; if `on_failure == "raise"`, raise `EKIError`; if $J_v < 2$,
   raise `EKIError` regardless of `on_failure`; otherwise apply
   `repair_failed_members` ({ref}`eki-failures`).
5. **Summarize.** Build the `Evaluation` from the repaired ensemble and
   predictions ({ref}`eki-diagnostics`), and return it. This ends `evaluate`.

0. **Validate, before anything else.** `apply` checks its `increment` and that
   its `evaluation` belongs to its `state` **on entry**, before the key split.
   Numbered zero because it precedes step 6's position in the original
   ordering, and the placement matters: validating the increment after the
   forward model has run — which a single fused `step` had to do, since the
   driver cannot know the increment until it has seen the evaluation — spends
   $J$ model evaluations before rejecting an argument the caller got wrong.
   Splitting the phases removes that cost for every caller of `apply`, and the
   driver still validates a schedule-returned increment at the same point it
   always did, because that is the earliest it exists.
6. **Validate the increment.** It must be a scalar, finite, and **strictly
   positive**, converted to a 0-d float array; otherwise `ValueError`. A zero
   increment is rejected even though it raises nothing: $R/0$ is a
   {class}`~pyeki.linalg.PSDScaled` with an infinite scalar, which whitens to
   zero, so the gain vanishes and the ensemble is returned **unchanged** while
   $\beta$ never advances. An adaptive ladder would then spin until
   `max_steps`.
7. **Update.** `u_new = update(key_update, ensemble=evaluation.ensemble,
   predictions=evaluation.predictions, y=y, noise_cov=noise_cov,
   increment=dbeta, step=state.step, beta=state.beta)`, validated to be
   `(J, P)` with a real floating dtype matching the incoming ensemble's.
8. **Finiteness.** If `u_new` contains a non-finite entry, raise `EKIError`
   naming the step and the level. Silent `nan` propagation through a long run
   is the worst outcome available to this layer, and the cost is one $O(JP)$
   reduction.
9. **Advance.** Return
   `EKIState(u_new, state.beta + dbeta, state.step + 1, key_next)` and the
   step's `HistoryRecord`.

Steps 4, 6 and 8 read concrete values, so one rung **synchronizes with the
device a small fixed number of times** — at most three across the two phases,
and at most four in `iterate`, which also reads the exhaustion check and the
stopping rule. The reads cannot be coalesced: step 6 decides whether to
dispatch the update, and step 8 reads a value the update produces. This is a deliberate cost: $O(1)$ scalars and one
reduction against $J$ forward-model evaluations, and it is what allows
termination, validation and adaptive increments to be ordinary Python.

### What the driver adds

`iterate` wraps `step` with the decisions `step` refuses to make, in this
order, before each call:

1. **Ladder exhaustion.** If the schedule's attributes say the ladder is
   finished ({ref}`eki-schedules`), end the run with status
   `"schedule_exhausted"`. This is checked *before* the forward model is
   evaluated, which is why exhaustion is separated from the increment: a fixed
   ladder of $T$ rungs must cost exactly $T$ ensemble evaluations, not $T + 1$.
2. **Safety bound.** If this call has already completed `max_steps`
   iterations, raise `EKIError`. The message must name `max_steps`, the
   schedule, and whether a stopping rule was supplied, since an unbounded
   schedule with no stopping rule is the usual cause.

   **`max_steps` bounds the iterations of this call, not `state.step`.** The
   distinction is invisible on a fresh run and decisive on a resumed one.
   Bounding the cumulative index would make the recovery this layer
   advertises — catch `EKIError`, checkpoint `exc.state`, resume
   ({ref}`eki-validation`) — a guaranteed no-op for the `max_steps` raise
   itself, since the resumed run would re-raise on entry before any
   evaluation, with an empty history; and it would silently shrink a resumed
   run's allowance to `max_steps - state.step` regardless of what the caller
   passed. The counter is therefore local to the call, and a resumption gets
   the bound the caller asked for.

   The order of these two decisions is normative and the inversion is a bug: a
   schedule that exhausts at step $T$ **must complete under
   `max_steps == T`**, which is the value a caller naturally passes. Checking
   the bound first would turn every such run into an `EKIError` on its final
   re-entry, blaming the schedule for a completed ladder.
3. **Evaluate and summarize** — steps 1–5 of `step` above, which `iterate`
   needs before it can consult the stopping rule or the schedule.

   This forces the internal factoring, so the contract states it rather than
   leaving it to be rediscovered. The layer has two private phases: *evaluate*
   (steps 1–5: split, inflate, call the model, repair, summarize) and *apply*
   (steps 6–9: validate the increment, update, check finiteness, advance).
   Public `step` is evaluate-then-apply with the increment given; `iterate` is
   evaluate, then its own decisions, then apply with the increment chosen. The
   forward model is therefore called **exactly once per rung** in both, which
   an implementation that had `iterate` call `step` naively would violate by
   evaluating twice — the one cost the layer exists to economize.
4. **Stopping rule.** If `stop is not None and stop(evaluation)`, end the run
   with status `"stopping_rule"`, emitting a terminal record
   ({ref}`eki-diagnostics`) and leaving the state unchanged.
5. **Increment.** `dbeta = schedule.next_increment(evaluation)`. A schedule may
   return `None` to declare the ladder finished on evidence only the evaluation
   carries ({ref}`eki-schedules`), which ends the run with status
   `"schedule_exhausted"` and emits the same terminal record as a stopping
   rule; any other value is validated and used as the increment.

(eki-updates)=
## Update rules

An `EnsembleUpdate` is a callable

```python
update(key, *, ensemble, predictions, y, noise_cov, increment, step, beta,
       **_) -> Array
```

mapping a `(J, P)` ensemble, its `(J, N)` predictions, the observation, the
**base** noise covariance and a 0-d increment to a new `(J, P)` ensemble.

**Everything after the key is keyword-only, and this is normative.** Two of the
arguments are arrays whose shapes coincide whenever $P = N$, so a positional
protocol would let `ensemble` and `predictions` be transposed with no error at
all — in exactly the small hand-built fixtures where the mistake is hardest to
see, and in the degeneracy cases the conformance suite is required to run.
Keyword-only makes that swap unrepresentable.

Implementations **should** accept and ignore `**_`. The argument list is the
layer's forwards-compatibility seam: a future rule may need something not
listed here, and a signature that tolerates unknown keywords keeps existing
rules working when it is added.

Requirements on any implementation:

- **It implements one rung of the ladder**: the move from an ensemble
  representing $\pi_\beta$ to one representing $\pi_{\beta + \Delta\beta}$.
  The two shipped rules do so by conditioning with `noise_cov / increment`.
- **It consumes the key whole**, per {doc}`gaussian-contract`'s randomness
  rule, and is a deterministic function of its arguments including the key.
  A deterministic rule ignores the key.
- **It receives both the increment and the absolute level**, and the two mean
  different things: `increment` is how far this rung moves, `beta` is where
  the rung starts, and `step` is which rung it is. The shipped rules use only
  the increment.

  A rule that varies with `beta` or `step` — an annealed threshold, a decaying
  damping — **breaks the telescoping identity of {ref}`eki-iteration`**, for
  the same reason inflation does, and the layer states that consequence rather
  than preventing it. Withholding the arguments would not prevent the
  behaviour: it would only push callers into keeping a counter inside the
  rule, which violates the purity requirement above and silently breaks
  resumption — a failure the conformance suite can catch in the package's own
  rules and cannot catch in a user's. Given the choice between an unenforced
  rule and an unresumable workaround, the contract takes the unenforced rule.
- **It is `jit`- and `vmap`-safe with static shapes**, and holds any arrays it
  needs as pytree data so that it can be passed through a trace boundary.
- **It receives the base noise covariance and the increment separately**,
  rather than the pre-scaled per-step operator, so that a rule needing the
  increment as a step size in its own right — a Langevin-type sampler, for
  instance — has it ({ref}`eki-variants`).

Two rules ship, and both are two lines over `pyeki.gauss`:

| rule | delegates to | character |
| ---- | ------------ | --------- |
| `TransformUpdate()` | `EnsembleJoint(u_samples=u, v_samples=v).transform_update(y, noise_cov / increment)` | deterministic square-root transform; ignores the key; **the default** |
| `PathwiseUpdate()` | `EnsembleJoint(u_samples=u, v_samples=v).pathwise_update(key, y, noise_cov / increment)` | perturbed-observation; consumes the key |

Neither holds any field. The entire numerical content of the update — the
whitened-SVD kernel, the bounded gain multiplier, the identity-completed
square-root transform, the whitener invariance, the graceful degradation at
zero prediction anomalies — belongs to {doc}`gaussian-contract` and must not
be re-derived here. If a step of this layer ever contains covariance
arithmetic of its own, the layering has been violated.

**Choosing between them** is the caller's decision, and beyond the default the
layer takes no position — it records the facts {doc}`gaussian-contract`
establishes. The square-root update's output has the posterior moments exactly,
per realization, and adds no Monte Carlo noise, at the cost of a deterministic
member-to-member coupling. The stochastic update's output has those moments in
expectation, with per-realization spread of order $KRK^\top/J$, and is the form
for which the pathwise-sampling reading of the run is available; it is also the
form most of the ensemble-Kalman-inversion literature is written in. The
deterministic alternative descends from the perturbation-free ensemble
square-root filters of Whitaker and Hamill and their successors
({ref}`eki-references`); {doc}`gaussian-contract` owns the transform itself.

**The two rules are named for the `pyeki.gauss` methods they delegate to**, so
that one vocabulary spans both layers: `TransformUpdate` calls
`transform_update`, `PathwiseUpdate` calls `pathwise_update`. The alternative
pairing — naming one for its mechanism, the square root, and the other for its
character, stochastic — is not an antonym pair, so a reader could not infer
from the names that the two are alternatives, and it introduced a third
vocabulary for a distinction the layer below had already named twice. The
mechanism and the character are both still stated, in the table above and in
the paragraphs below; they are simply not what the classes are called.

**`TransformUpdate` is the default**, and the reason is the layer's own
claims. Telescoping holds exactly under it and only in expectation under the
stochastic update, so the exactness statement of {ref}`eki-honesty` — the one
property this layer proves about itself — is a property of the default
configuration rather than of a configuration a caller has to know to ask for.
It is also deterministic, so a user's first two runs agree, and a difference
between them is a bug rather than a seed. A caller who wants the classical
perturbed-observation form passes `update=PathwiseUpdate()`, which is one
keyword and is what the sampling-form literature describes.

(eki-schedules)=
## Schedules

A `Schedule` is **one method and two declarative attributes**.

| member | kind | contract |
| ------ | ---- | -------- |
| `n_steps` | `int \| None` | the ladder's length in rungs, or `None` if it is not step-bounded |
| `beta_target` | `float \| None` | the temperature budget, or `None` for an unbounded ladder |
| `next_increment` | `(evaluation: Evaluation) -> Array \| float \| None` | the next increment, or `None` for "finished after all". A returned value must be scalar, finite and strictly positive. |

**The driver decides exhaustion, not the schedule.** A ladder is finished when

```python
(sched.n_steps is not None and step >= sched.n_steps) or (
 sched.beta_target is not None and beta >= sched.beta_target - budget_tol)
```

with `budget_tol` as {ref}`eki-adaptive` gives it. The check is
misfit-independent and runs before any forward evaluation, so a run whose
ladder is already finished pays nothing; `next_increment` may read the current
misfits, so adaptivity costs no evaluation either.

The attributes are read, never called, and must be constant for the life of the
object — they are static metadata on a frozen policy, not state. A schedule
with both `None` is legal and unbounded, and must be ended by a stopping rule.

**Why exhaustion is data rather than a method.** An earlier form of this
protocol had a second method, `exhausted(step, beta) -> bool`. Four things go
wrong with it, and all four are fixed by the attributes:

- *A step-size rule could not be a plain function.* {ref}`eki-objects`'s rule 2
  permits a bare function wherever a protocol has a single method, and a
  two-method protocol withdraws that permission from every schedule — including
  the unbounded step-size rules whose `exhausted` is `return False`, a required
  method with a constant body.
- *"Is this a budgeted ladder?" was unanswerable without calling something.*
  The driver can now check the schedule against `max_steps` at entry
  ({ref}`eki-driver`), and warn on the `DiscrepancyStop`-on-a-budget trap
  ({ref}`eki-axes`), neither of which is possible when the answer is hidden
  behind a method.
- *`budget_tol` was duplicated into every schedule* that wanted a budget, and
  silently omitted by any that forgot it.
- *The two shipped families already carry the attributes as fields*, so the
  protocol now costs them nothing: `FixedSchedule.n_steps` is
  `len(increments)`, and the adaptive schedules' `beta_target` is the budget
  field itself.

**Why `next_increment` may also end the ladder.** The attributes describe
bookkeeping only, so a schedule whose finishing condition depends on the
*misfits* has no way to express it there — an adaptive ladder that should give
up when its target has become unattainable above the floor, for instance.
Without a second exit such a schedule must either crawl at its floor until
`max_steps` raises, spending the whole evaluation budget to report a failure,
or the caller must restate the schedule's own configuration and criterion
inside a separate `StoppingRule` and keep the two consistent by hand. Returning
`None` is the smaller mechanism: it ends the run with
`status="schedule_exhausted"`, and the evaluation that produced it is recorded
as the terminal record, exactly as for a stopping rule
({ref}`eki-diagnostics`).

The two exits mean different things and both are needed. The attributes say "I
can tell from the bookkeeping that there is nothing left to do", and cost no
evaluation; `None` says "having seen this ensemble, there is nothing useful
left to do", and costs the evaluation that produced it. Neither shipped
schedule returns `None`; both reach their budgets through the attributes.

`next_increment` must be **pure**: a schedule that counted its own calls could
not be resumed from a checkpoint, and `evaluation.step` is passed precisely so
that it need not.

**No schedule performs a trial update or a trial forward evaluation.** Both
shipped criteria read nothing but the misfit vector $(\Phi_1,\dots,\Phi_J)$ and
the current level, so adaptivity costs $O(\texttt{n\_bisect}\cdot J)$ per step
for a bisecting schedule and $O(J)$ for a closed-form one, on top of a whitening
the driver pays anyway. Two consequences: the increment-rescaling identity
{doc}`gaussian-contract` records for candidate steps is never needed here,
because no candidate is ever evaluated; and choosing an increment can never
cost a model evaluation, which is the resource the whole layer is organized
around.

A custom schedule is not confined to the misfits, though — the evaluation
carries the whole whitened residual matrix, and hence the whitened prediction
anomalies, as well as the members and their predictions
({ref}`eki-diagnostics`), which is what a step-size rule of the Langevin family
needs. What no schedule can do is evaluate the forward model again; that
restriction is the protocol's, and it is deliberate ({ref}`eki-excluded`).

### `FixedSchedule(increments)`

A ladder given in advance. `increments` is a non-empty tuple of Python floats,
static metadata, each strictly positive and finite (`ValueError` otherwise).
`n_steps` is `len(increments)` and `beta_target` is `None`;
`next_increment(evaluation)` returns `increments[evaluation.step]` and ignores
everything else.

Because it indexes by the state's cumulative step, a `FixedSchedule` resumes a
partially-completed ladder correctly and treats a *finished* state as finished
— see the warning in {ref}`eki-state` before chaining one run onto another.

Its `repr` summarizes rather than enumerates: `FixedSchedule(n_steps=200,
total=200.0)`. The general rule that policy objects print their static fields
({ref}`eki-repr`) assumes those fields are small, and this one's is not — a
200-rung optimization ladder would otherwise print 200 floats into every
traceback and test id.

Two classmethods cover the common ladders:

| constructor | increments | form |
| ----------- | ---------- | ---- |
| `FixedSchedule.uniform(n_steps)` | `(1/T,) * T` | sampling: a uniform ladder to $\beta = 1$ |
| `FixedSchedule.constant(increment, n_steps)` | `(c,) * T` | optimization when $c \cdot T > 1$; a single Kalman update at `constant(1.0, 1)` |

`uniform` reaches $\beta = 1$ to round-off, not exactly: $T \cdot (1/T)$ is
not exactly 1 in floating point. This is a documented consequence, not a
defect to paper over with a correction on the last rung.

(eki-adaptive)=
### Shared semantics of the adaptive schedules

Both adaptive schedules answer the same question — *how far can the target
move before this ensemble stops describing it?* — and differ only in how they
measure it. Everything except the measurement is shared, and is specified here
once.

**Fields common to both.**

| field | default | meaning |
| ----- | ------- | ------- |
| `beta_target` | `1.0` | the temperature budget, or `None` for an unbounded ladder |
| `min_increment` | `1e-3` | a floor guaranteeing progress |
| `max_increment` | `1.0` | a ceiling |

All three are static metadata. `beta_target`, when given, must be strictly
positive; the floor must be positive and no greater than the ceiling
(`ValueError` otherwise).

**Exhaustion.** Both schedules set `n_steps = None` and expose the budget as
`beta_target`, so the driver's check is
`beta_target is not None and beta >= beta_target - budget_tol`, with
`budget_tol = 1e-12 * beta_target` — relative, so that a small budget is not
swallowed whole: an absolute floor would make any `beta_target` at or below it
exhaust at $\beta = 0$, returning an untouched ensemble and an empty history
with nothing raised. It is never satisfied when `beta_target is None`, so an
unbounded ladder must be ended by a stopping rule; a run with neither is a
`max_steps` `EKIError`, and that error's message must say so
({ref}`eki-driver`).

**Clamping, and its precedence.** Each schedule computes an unclamped
criterion value $\delta^\star$ and returns

$$
\delta \;=\; \min\Bigl(\,
\max\bigl(\delta^\star,\ \delta_{\min}\bigr),\ \
\delta_{\max},\ \
\beta_{\text{target}} - \beta
\,\Bigr).
$$

**The budget term is present only when `beta_target is not None`.** For an
unbounded ladder the clamp is
$\min(\max(\delta^\star, \delta_{\min}), \delta_{\max})$; writing the
three-term form unconditionally is a `TypeError` on every unbounded run, which
is a shipped configuration ({ref}`eki-variants`). For the same reason
`max_increment` must be **finite**: an unbounded ladder with an infinite ceiling
and a degenerate ensemble has no upper clamp at all, and
$\delta^\star = +\infty$ then reaches the increment validation of
{ref}`eki-step` and raises there instead.

The order is normative, and both inversions of it are bugs. The floor beats
the criterion, so a step is always taken even where the criterion would demand
an arbitrarily small one — without which an adaptive ladder can stall short of
its budget for an unbounded number of steps. The **budget cap beats the
floor**, so the ladder cannot overshoot $\beta_{\text{target}}$ — without which
a sampling run silently conditions on more data than it has. Positivity of the
result follows from the exhaustion check having been false, which guarantees a
remaining budget strictly above `budget_tol`.

With a floor $\delta_{\min}$ and a budget, a run reaches $\beta_{\text{target}}$
in at most $\lceil \beta_{\text{target}} / \delta_{\min}\rceil$ steps.

**The shipped defaults are chosen so that this bound is reachable, and the
arithmetic is part of the contract.** With `beta_target=1.0` and
`min_increment=1e-3` the worst case is $\lceil 1/10^{-3}\rceil = 1000$ steps,
which is exactly the driver's default `max_steps=1000`
({ref}`eki-driver`). A budgeted adaptive schedule at the defaults therefore
**cannot** raise on the safety bound: the floor-bound ladder finishes on its
last permitted step. A caller who lowers `min_increment`, or *raises*
`beta_target`, breaks that relation — a floor of $10^{-4}$, or a budget of 2,
against a bound of 1000 needs 10000 or 2000 rungs respectively. Neither is
left to be discovered: the driver checks the arithmetic at entry and raises
`ValueError` before spending an evaluation ({ref}`eki-driver`).

**The degenerate ensemble.** When every member has the same misfit — a
collapsed ensemble, or a forward model insensitive to the current spread — no
increment changes the target's shape relative to the ensemble, and both
schedules must take the **largest allowed step**, that is
$\min(\delta_{\max}, \beta_{\text{target}} - \beta)$. Each criterion below
reaches that conclusion on its own; the requirement is recorded here so that an
implementation cannot satisfy one schedule's version of it and not the other's.

### `AdaptiveESSSchedule`

Measures the move by the effective sample size of the importance weights that
would carry the ensemble from one target to the next. The construction is the
standard ESS-based adaptive tempering of the sequential Monte Carlo literature
(Jasra and co-authors; convergence theory in Beskos and co-authors —
{ref}`eki-references`), and is used here purely as a **step-size heuristic**:
pyEKI computes no importance weights, does no resampling, and makes no
importance-sampling correctness claim. Its extra field is `ess_fraction`,
the ESS level sought as a fraction of $J$, default `0.5`, required to lie in
$(0,\ 1 - 10^{-6}]$ — bounded away from 1 because $\mathrm{ESS}(0)$ evaluates
to `exp(log J)` rather than exactly $J$, so a `ess_fraction` within round-off of 1
makes $\delta = 0$ an invalid lower bracket and the invariant below false from
the first iteration; and `n_bisect`, a static `int` bisection count, default
`50`, required to be at least 1. Any `n_bisect` at or above 30 attains the
target to float64 resolution.

For an increment $\delta$, the weights and their effective sample size are

$$
w_j(\delta) = e^{-\delta \Phi_j},
\qquad
\mathrm{ESS}(\delta)
= \frac{\bigl(\sum_j w_j\bigr)^2}{\sum_j w_j^2}
\;\in\; [1, J] .
$$

$\mathrm{ESS}$ is $J$ at $\delta = 0$ and **monotone non-increasing** in
$\delta$, strictly so unless every $\Phi_j$ is equal — the degenerate case
{ref}`eki-adaptive` covers — so the target level is reached by bisection.
{doc}`design` derives it. Two consequences of that derivation bind the
implementation. The rate of decay is a variance of the misfits under the tilted
weights, so the criterion measures ensemble disagreement about the data. And
**the derivative vanishes at $\delta = 0$**, so the function is flat at the left
endpoint and a derivative-based root find started there stalls, which is why the
method is bisection.

The criterion is

$$
\delta^\star \;=\; \sup\{\delta \,:\, \mathrm{ESS}(\delta) \ge \texttt{ess\_fraction}\cdot J\},
$$

clamped as {ref}`eki-adaptive` specifies. Three implementation requirements:

- **Bisect on a bracket, and return the safe end.** Take
  $\delta_{\mathrm{hi}} = \min(\delta_{\max},\, \beta_{\text{target}} -
  \beta)$, or $\delta_{\max}$ when `beta_target is None`. Bisect
  $[0, \delta_{\mathrm{hi}}]$ for exactly `n_bisect` iterations, maintaining
  $\mathrm{ESS}(\text{lo}) \ge \text{ess\_fraction}\cdot J >
  \mathrm{ESS}(\text{hi})$, and return `lo`. The guarantee is therefore
  one-sided: the returned increment meets the target before the floor is
  applied.

  When $\mathrm{ESS}(\delta_{\mathrm{hi}})$ already meets the target the cap
  binds and $\delta_{\mathrm{hi}}$ is the answer — but that case must be folded
  in **branchlessly**, by initialising
  `lo = jnp.where(ess_hi >= ess_fraction * J, delta_hi, 0.0)` and letting the loop run
  unchanged, never as an early Python return on a device value. It is not an
  optimization: without it the degenerate ensemble of {ref}`eki-adaptive`
  reaches the largest allowed step only to $2^{-\texttt{n\_bisect}}$, and a
  budgeted ladder never consumes its budget exactly. A fixed iteration count
  with no data-dependent control flow is what keeps the whole criterion one
  traced array computation; implement the loop with `lax.fori_loop`, not as an
  unrolled Python loop.
- **Compute the ESS in log space.** $\delta\Phi_j$ is routinely in the
  hundreds, so
  $\mathrm{ESS} = \exp\bigl(2\,\mathrm{lse}(-\delta\Phi) -
  \mathrm{lse}(-2\delta\Phi)\bigr)$
  with `lse` a max-shifted log-sum-exp. Naive exponentiation underflows every
  weight to zero and returns `nan` — a `0/0` that would silently poison the
  bisection.
- **Rely on the shift invariance rather than working around it.** The ratio is
  unchanged by a common shift of every $\Phi_j$, which is exactly why the
  max-shifted form above is not merely a numerical convenience but computes
  the same quantity — and why the criterion remains meaningful when every
  misfit is enormous, as it is early in a run.

`ess_fraction` defaults to `0.5`, which is pyEKI's choice rather than a canonical
value; the tempering literature uses targets between about a third and a half.
A smaller target takes longer steps and fewer of them.

### `AdaptiveMisfitSchedule`

Measures the move against the **noise level** rather than against the
ensemble, and solves for the increment in closed form instead of bisecting.
Its one extra field is `divergence_budget`, written $\theta$ in the formulas
below, default `None` meaning $N/2$ — and with that default the schedule has
**no tuning parameter at all**, which is its main attraction.

The field is named for what it bounds rather than for the source's symbol. It
is the ceiling on the Jeffreys divergence between consecutive tempered
measures, which is where the criterion comes from and what a caller overriding
it is actually choosing; $\theta$ means nothing to a reader who has not read
the source, and the default exists precisely so that most callers never name it
at all. The symbol is kept in the mathematics, where it is the source's.

This is the **data misfit controller** of Iglesias and Yang
({ref}`eki-references`), reproduced here rather than invented: their equation
for $\alpha_n^{-1}$ is the criterion below, their $\theta = M/2$ is this
section's $\theta = N/2$, and their budget clamp $1 - t_n$ is
{ref}`eki-adaptive`'s. Two things this contract adds are presentational rather
than mathematical — the criterion is expressed in increments rather than in the
inflation factors $\alpha_n$ that the multiple-data-assimilation literature
uses, and the clamps are specified with an explicit precedence — and one is a
genuine requirement the source does not need: the guarded divisions below,
because pyEKI must not return `nan` on a collapsed ensemble.

The criterion. Write $\chi_j = 2\,\delta\,\Phi_j$ for member $j$'s whitened
misfit measured at *this step's own* noise level $R/\delta$, rather than at
the base level. If the step's target were well specified and the ensemble were
distributed according to it, each $\chi_j$ would be a $\chi^2_N$ variate, whose
mean is $N$ and whose variance is $2N$. Those two benchmarks give two
thresholds on $\delta$ — **alternatives, not a conjunction**; which one applies
is settled by the derivation below, and requiring both at once is the silent
error this section exists to prevent:

$$
\text{(mean)}\quad \overline{\chi} \le N
\iff \delta \le \frac{N}{2\overline{\Phi}} ,
\qquad
\text{(variance)}\quad \operatorname{Var}(\chi) \le 2N
\iff \delta \le \sqrt{\frac{N}{2\,\sigma^2_\Phi}} ,
$$

with $\overline{\Phi}$ and $\sigma^2_\Phi$ the ensemble mean and variance of
the base misfits, the variance taken with the package's fixed $J-1$ divisor.
The criterion is the **larger** of the two bounds,

$$
\delta^\star \;=\; \max\left\{\, \frac{\theta}{\overline{\Phi}},\ \
\sqrt{\frac{\theta}{\sigma^2_\Phi}} \,\right\},
\qquad \theta = N/2 \ \text{ by default},
$$

clamped as {ref}`eki-adaptive` specifies — where the budget clamp
$\beta_{\text{target}} - \beta$ is what makes a run terminate exactly at
$\beta_{\text{target}}$ with the increments summing to it.

**The `max` is deliberate and is not a `min`.** Read as two independent
benchmarks the `max` looks merely permissive, but the source's derivation makes
it forced: the Jeffreys divergence between consecutive tempered measures is
approximated by the *smaller* of two expressions valid in different regimes, and
a `min` bounded by $\theta$ is satisfied exactly when $\delta$ is below the
*larger* of the two thresholds. Writing `min` in the criterion would impose both
approximations at once, including whichever is invalid in the current regime,
and would stall the ladder on the invalid one. {doc}`design` gives the
derivation; inverting this is a silent error, and conformance obligation 4 pins
each regime.

The threshold $\theta = N/2$ is then fixed by a *statistical discrepancy
principle* rather than tuned. At the step's own noise level the whitened misfit
$\chi_j$ would be $\chi^2_N$ if the target were well specified, so requiring the
mean below $N$ and the variance below $2N$ — the two benchmarks stated above —
is exactly $\theta = N/2$ in both bounds. That is the sense in which the
schedule has no free parameter: the observation dimension supplies it.

Which bound binds is decided by the misfits' coefficient of variation: the mean
bound is the larger exactly when $\sigma_\Phi/\overline{\Phi} > 1/\sqrt{\theta}$,
which at the default $\theta = N/2$ is $\sqrt{2/N}$ — itself exactly the
coefficient of variation of a $\chi^2_N$ variate. For
any appreciable $N$ that is the common case — the source reports it holding
throughout its own experiments, and notes that it corresponds to a prior wide
enough to contain the truth. The variance bound takes over for an unusually
tightly clustered ensemble, or a prior centred far from the truth with little
spread, where it correctly permits a longer step than the mean alone would.

**How this differs from the ESS criterion — and why both ship.** The two are
not variants of one idea, and they do not agree. Under the small-increment
increment the misfit schedule chooses, the effective sample size is at or near
its **floor**: the variance benchmark puts $\delta^2\sigma^2_\Phi$ at $N/2$,
which for any appreciable $N$ is far outside the small-increment regime in
which $\mathrm{ESS}$ is close to $J$, and drives the weights onto one to three
members regardless of $N$ or $J$. (For Gaussian misfits the weights are
lognormal and $\mathrm{ESS}/J = e^{-\delta^2\sigma^2_\Phi}$, so the floor
$\mathrm{ESS} \ge 1$ binds well before the benchmark is reached.) The misfit
schedule therefore takes **far longer steps** than an ESS target near $1/2$
would allow. That is not a defect in either: they control different
things. The ESS criterion keeps each intermediate target representable by the
ensemble that must describe it, which is what the *sampling* form needs; the
misfit criterion absorbs as much data per rung as the noise at that rung can
explain, which is the *optimization* form's logic, and it is calibrated to the
observation dimension instead of to the ensemble size.

The practical recommendation follows: `AdaptiveESSSchedule` when the posterior
ensemble is the deliverable, `AdaptiveMisfitSchedule` when the fit is — or
when model evaluations are scarce enough that a tuning-free schedule taking
few large steps is worth its coarser approximation.

**The degenerate cases are guarded divisions.** Both bounds must yield
$+\infty$ rather than `nan` when their denominator vanishes: a collapsed
ensemble gives $\sigma^2_\Phi = 0$, and an ensemble that already fits the data
exactly gives $\overline{\Phi} = 0$. Compute each as

```python
jnp.where(d > 0, theta / jnp.where(d > 0, d, 1.0), jnp.inf)
```

— the **doubled** `where`, because `jnp.where` evaluates both branches, so the
single-`where` form still forms `theta / 0` and yields a `nan` derivative at
`d = 0`. The clamps of {ref}`eki-adaptive` then deliver the largest allowed
step, as that section requires. Dividing unguarded
happens to give `inf` for positive `divergence_budget`, but relies on the sign of a zero
and gives `nan` at `divergence_budget = 0`, so the guard is contractual rather than
stylistic.

(eki-stopping)=
## Stopping rules

A `StoppingRule` is a callable `stop(evaluation) -> bool`, returning a Python
`bool`, pure, and — like schedules — forbidden from holding iteration state.
It is consulted before the increment is chosen, so a run that already fits the
data takes no further update.

A misfit-based stopping rule necessarily costs one forward evaluation whose
update is discarded: the misfits that trigger the stop are the misfits of an
ensemble that is then returned unchanged. That evaluation is visible in the
history as the terminal record ({ref}`eki-diagnostics`), and it is inherent,
not an implementation artifact.

**`DiscrepancyStop(tau=1.0)`** implements Morozov's discrepancy principle,
in the form Iglesias adapted to ensemble Kalman methods
({ref}`eki-references`): stop as soon as the ensemble centre fits the data to
within the noise level,

$$
2\,\Phi(\bar v) \;\le\; \tau^2 N ,
$$

where $\bar v = \frac1J\sum_j v_j$ is the **mean prediction** and
$\Phi(\bar v)$ is the evaluation's `centre_misfit`.

The scaling is the natural one. At the true parameter, the whitened residual is
the whitened noise, so $2\Phi$ is a $\chi^2_N$ variate with mean $N$ and
standard deviation $\sqrt{2N}$: $\tau = 1$ stops when the centre's residual
reaches the size the noise alone explains *in expectation*, and $\tau^2 = 1 + k\sqrt{2/N}$
puts the threshold $k$ standard deviations above that, values up to about
$\tau = 2$ being the common conservative choice. Fitting *below* the noise
level is over-fitting, which is what the rule exists to prevent and why the
optimization form needs a discrepancy criterion rather than a fixed step count.
`tau` must be strictly positive (`ValueError` otherwise).

:::{note}
The residual is measured at the mean **prediction**, $\bar v$, not at the
prediction of the mean parameter, $G(\bar u)$. These differ at second order in
the ensemble spread, and $\bar v$ is the one available: $G(\bar u)$ would cost
an extra forward-model evaluation per step ({ref}`eki-excluded`). A caller who
needs the stricter reading evaluates it themselves in an `iterate` loop.
:::

(eki-inflation)=
## Inflation

An `Inflation` is a callable
`inflate(key, *, ensemble, step, beta, **_) -> Array`, shape preserving, pure,
`jit`-safe. It runs at the top of every step, on the ensemble that is about to
be evaluated and updated ({ref}`eki-step`). The keyword-only convention and the
`**_` recommendation are the update protocol's, for the same reasons
({ref}`eki-updates`); `step` and `beta` are supplied because a decaying
inflation schedule is common practice and the alternative is a rule that
mutates a counter and cannot be resumed.

Both "before" and "after" the analysis appear in the literature. **In the interior of a run they coincide**: inflating
the analysis ensemble at the end of step $t$ and inflating it at the start of
step $t+1$ produce identical sequences, since nothing happens in between. They
differ at the two ends, and the difference is why the start is chosen —
placing it at the start means the ensemble a run *returns* is a clean posterior
ensemble rather than an inflated one, and that predictions always match the
members they update.

The cost of that choice: the **initial**
ensemble is inflated before it is ever evaluated, so a run with inflation on
never evaluates exactly the ensemble the caller supplied, and the final
ensemble is never inflated. A caller who needs the pristine initial ensemble
evaluated drives the first rung with `inflation=None` and the rest with the
inflation. Applying inflation between the forward evaluation and the update
would instead invalidate the predictions, and is forbidden.

The equivalence above also assumes an inflation that does not look at the
update. One that does — the relaxation-to-prior family, which blends the
posterior anomalies or spread back toward the pre-update ensemble's — is not an
`Inflation` under this signature at all, and is a custom `EnsembleUpdate`
instead ({ref}`eki-excluded`).

**Inflation breaks the telescoping identity of {ref}`eki-iteration`, by
design.** A run with inflation on is not a tempering ladder for the target
family; it is a deliberately widened variant. Sampling-form runs should
therefore default to no inflation, and `inflation=None` is the default.

:::{warning}
**"Inflation" is an overloaded word, and this layer uses the narrower sense.**
Here it means *ensemble* inflation: a transformation that widens the spread of
the members. In much of the ensemble-Kalman-inversion literature the same word
names something else entirely — inflating the *observation noise* by a factor
$\alpha$, which in this contract is the tempering increment
$\Delta\beta = 1/\alpha$ ({ref}`eki-iteration`) and is the schedule's business,
not this section's. The two are unrelated knobs that happen to share a name, so
read any external formula's definition before transcribing it.
:::

| implementation | field | effect |
| -------------- | ----- | ------ |
| `MultiplicativeInflation(anomaly_factor)` | 0-d array | $u_j \mapsto \bar u + r\,(u_j - \bar u)$ |
| `AdditiveInflation(cov)` | a `PSDLinOp` of side $P$ | adds centred draws from `cov` |

**`MultiplicativeInflation`** is the classical covariance inflation of Anderson
and Anderson ({ref}`eki-references`). It scales the anomalies, so the empirical
covariance is multiplied by $r^2$, the field **squared**. The field is therefore named
`anomaly_factor` rather than `factor`: the literature is genuinely split
between the anomaly convention and the covariance convention $C \mapsto \gamma C$,
uses the same symbols for both, and the two are related by $\gamma = r^2$.
Naming the field for what it multiplies is the only way to keep a caller from
passing an intended variance inflation of $1.2$ and silently getting $1.44$ —
an error that is invisible at the small values normally used ($r$ a few percent
above 1, where $r$ and $\sqrt{\gamma}$ barely differ) and severe at large ones.
The field is held as a 0-d array (pytree data) so a traced value can flow
through it. The mean is preserved exactly, and so is the subspace of
{ref}`eki-subspace`.

**`AdditiveInflation`** requires `cov.supports("factor")` and applies, with
`L = cov.factor()` of width $k$,

```python
pert = Gaussian(jnp.zeros(P), cov).sample(key, J)
ensemble + (pert - pert.mean(axis=0))
```

defined **by delegation** to {meth}`~pyeki.gauss.Gaussian.sample` rather than
by restating its factor-and-normal recipe. The draw is then pinned in exactly
one place in the package, the gauss layer's snapshot test covers this one too,
and the two cannot drift apart — which they could if this layer wrote out its
own `normal(key, (J, k))`, since nothing but review would notice the shapes
diverging. The perturbations
are **centred**, which preserves the ensemble mean exactly. The empirical
covariance is inflated by `cov` in expectation under the $J-1$ divisor —
exactly, and independently of the centring, since subtracting the perturbation
mean leaves the anomalies about the new mean unchanged. A
scale is folded into the operator by the caller — `AdditiveInflation(0.01 *
prior.cov)` — rather than carried as a second field. This is the only shipped
mechanism that moves the ensemble out of its initial affine subspace, which is
its main reason for existing.

Composing two inflations is a three-line callable and is not packaged
({ref}`eki-excluded`).

(eki-failures)=
## Forward models and failed members

**The forward model is any callable** `(J, P) -> (J, N)`. That is the whole
interface, and it is fixed by the package's permanent scope boundary: pyEKI
ships no forward models and defines no forward-model base class. The callable
may be `jit`-ed by the caller, may fan out over processes, may block on a job
scheduler; the driver never traces it and never inspects it.

**Failure is signalled by non-finite predictions.** A member whose prediction
row contains any `nan` or `inf` is invalid. Deriving validity this way rather
than adding a mask to the return type keeps the interface at one array, and
matches what failing simulators actually produce.

:::{important}
**The wrapper owns its own exceptions.** A one-array interface can express a
failed member but not a raised one, so the obligation falls on the caller: a
forward model that may crash, time out, return a non-zero exit code, or lose a
worker **must catch that itself and return a non-finite row** for the affected
members. An exception that escapes the callable propagates out of the driver;
the run stops, and the work is recoverable only because `EKIError` and the
propagating exception leave the history reachable ({ref}`eki-driver`) — which
is a worse outcome than a `nan` row the layer knows how to handle.

This is a real obligation on every caller wrapping an external simulator, and
it is the one thing the layer needs from a forward model beyond its shape.
:::

**What non-finiteness cannot detect.** The signal catches the failures that
announce themselves, and the layer should not be read as catching more. Named
so their absence is a decision:

- **Plausible-but-wrong output** — a solver that returns zeros, its initial
  condition, or a sentinel fill value such as `-9999`. These are finite, so
  every member is "valid". A fill value is the dangerous case: it produces an
  enormous $\Phi_j$, which an adaptive schedule reads as genuine ensemble
  disagreement and answers by shrinking the increment, so the run *stalls* on
  a broken member instead of flagging it.
- **Finite but unphysical output** — a diverged but non-overflowing solve.
  Indistinguishable from a poor fit at this layer.
- **Slow failure** — a member that never returns. There is no per-member
  timeout, and there should not be one here; it belongs to the wrapper that
  owns the process.
- **Systematically failing members.** A member repaired to the centre is
  updated to the posterior mean, may fail again next step, and can be absent
  from the run's effective ensemble throughout while every step still reports
  $J_v$ close to $J$. Only the `n_valid` history reveals it, and only if a
  caller looks.

A caller who can distinguish these in their own wrapper should map them to
non-finite rows there, where the information exists. The layer's contribution
is to make the failures it *can* see loud and its handling exact.

Handling is governed by `on_failure`, whose value must be one of the two
strings below; anything else is a `ValueError` at the call
({ref}`eki-validation`), never a silent fallback to the default:

| value | behaviour |
| ----- | --------- |
| `"repair"` (default) | replace failed members by the valid centre, leaving valid members untouched |
| `"raise"` | raise `EKIError` naming the number and indices of failed members |

Either way, $J_v < 2$ raises: a single valid member has no anomalies.

**Failures are surfaced three ways, because `n_valid` alone is not enough.**
Every record carries it, the driver logs a `WARNING` on any step with a failed
member, and `EKIResult.min_n_valid` reports the worst step without the caller
having to stack the history. The third matters because the first two are easy
to miss: reading `n_valid` requires stacking the history and looking, and the
`logging` record reaches nobody until a handler is installed, which
{ref}`eki-driver` states plainly is not the default. On top of that, `run`
issues one `warnings.warn` per run in which any member ever failed —
`warnings` being on by default where `logging` is not, which is the whole
difference between a mitigation and a mitigation on paper. Under `"repair"` a
run can otherwise complete, return a normal-looking result, and have been
conditioning on a covariance damped at every rung
({ref}`eki-failures`).

### `repair_failed_members(ensemble, predictions, valid)`

The repair keeps the ensemble size static — a requirement, since a
data-dependent $J$ would make every downstream shape dynamic. With
$m_j \in \{0,1\}$ the validity indicator, $J_v = \sum_j m_j$, and
$\hat u, \hat v$ the means over the valid members alone,

$$
u_j \;\longmapsto\; \hat u + m_j\,(u_j - \hat u) ,
$$

and identically for $v_j$: **failed members are moved to the valid centre and
valid members are left exactly where they are.** Then, exactly:

- the all-$J$ mean equals $\hat u$, because the valid anomalies sum to zero
  about $\hat u$ and the invalid ones are now zero;
- the all-$J$ empirical covariance with divisor $J-1$ equals the valid-member
  covariance with divisor $J_v-1$, scaled by $(J_v-1)/(J-1) < 1$, and
  likewise the cross-covariance — both blocks carry the same factor;
- failed members become $(\hat u, \hat v)$, so their residual is $y - \hat v$;
  they rejoin the ensemble rather than being lost. Under
  `TransformUpdate` their update lands **exactly** at the posterior mean:
  zeroing row $j$ of $A_u$ and of $A_v$ zeroes row $j$ of $S$, hence
  $U_{ji} = 0$ for every $\sigma_i > 0$, so the transform leaves that row's
  anomaly at zero. Under `PathwiseUpdate` they land at the posterior mean
  plus their own $\mathcal{N}(0, KRK^\top)$ perturbation.

**The covariance is damped, and that is the intended trade.** The step
conditions with $\widehat C_{uv}$ and $\widehat C_{vv}$ both multiplied by
$c = (J_v-1)/(J-1)$, giving the gain $c\,\widehat C_{uv}(c\,\widehat C_{vv} +
R/\delta)^{-1} = \widehat C_{uv}(\widehat C_{vv} + R/(c\delta))^{-1}$ — neither
$K$ nor $cK$, but exactly the gain at the shorter increment $c\delta$. For the
mean, then, a failed member costs a slightly smaller step, which is the safe
direction and is bounded by the failure fraction. The posterior *spread* is
additionally narrowed by $c$, and that part **does** carry into the next step:
with failures recurring at rates $c_t$ the ensemble carries $\prod_t c_t$. What
does not accumulate is the factor within a single step.

:::{admonition} Why the moment-exact repair is not the default
:class: note

There is an alternative that makes the fixed-$J$ moments equal the valid
members' moments *exactly*, by rescaling every member's anomaly:

$$
u_j \;\longmapsto\; \hat u + m_j\,(u_j - \hat u)\,\sqrt{\tfrac{J-1}{J_v-1}} .
$$

It is the construction {doc}`gaussian-contract` anticipates as the reason its
anomaly divisor is fixed, and on the moments it is strictly better. pyEKI does
not use it, because of what it does to the members. The factor is applied to
the *surviving* members too, so each one is moved outward from the centre; the
pair $(u_j, v_j)$ is no longer forward-model-consistent; and the run's returned
ensemble can contain members never evaluated at their own parameters.

The factor is not small. At $J = 100$ with $10\%$ of
members failing, the factor is $\sqrt{99/89} \approx 1.055$ — a $5.5\%$
re-injection of spread, *per step*, larger than the multiplicative inflation
factors practitioners actually use, applied silently and by default. The layer
is emphatic that inflation breaks telescoping and must therefore be opt-in
({ref}`eki-inflation`); a default that inflates by a data-dependent amount
under the name "repair" would contradict that in the configuration most users
never change.

So the damping is preferred to the silent inflation, and the moment-exact
variant is recorded here rather than shipped. A caller who wants it writes six
lines and passes the result through a custom `EnsembleUpdate`.
:::

`repair_failed_members` is public because the moment identities above are worth
testing directly, and because a caller driving the analysis from outside the
loop needs it.

**No-failure steps must be bit-exact.** When $J_v = J$ the formula is
mathematically the identity but *not* bit-exactly so — it subtracts and adds
the mean. The driver therefore branches in Python on the synchronized
`n_valid` and skips the repair entirely when nothing failed, so that adding
failure handling changes nothing about a run in which nothing fails. The
consequence for compilation is at most one extra traced variant
({ref}`eki-jax`).

**Misfits are computed after repair.** A failed member would otherwise
contribute a `nan` misfit and poison every statistic and every adaptive
criterion. After repair its prediction is $\hat v$, so it contributes
$\Phi(\hat v)$ — the misfit of the valid centre, which by the identity of
{ref}`eki-diagnostics` sits *below* the valid members' mean misfit by
$\tfrac{J_v-1}{2J_v}\operatorname{tr}(W \widehat C_{vv} W^\top)$. The bias in
$\overline{\Phi}$ is therefore downward, which makes the reported misfit
spread and the ESS slightly optimistic and makes
`AdaptiveMisfitSchedule`'s mean bound slightly permissive; `n_valid` is in the evaluation so that a
criterion may account for it.

(eki-diagnostics)=
## Diagnostics

Two classes, split by lifetime rather than by subject. An **`Evaluation`** is
transient: it exists for the duration of one step, carries the members, their
predictions and the whitened residual matrix, and is what a schedule and a
stopping rule are shown. A **`HistoryRecord`** is kept for the whole run: it
carries scalars only, so a history of hundreds of steps costs nothing, and it is
what a caller reads afterwards.

The driver builds the record from the evaluation and the chosen increment, so
the evaluation is the single source of truth for everything both describe — a
record field that could disagree with the evaluation it came from would be a
defect ({ref}`eki-conformance`, obligation 21). The one field the record adds
from outside is the increment, which is not known until the schedule has seen
the evaluation.

### `Evaluation`

Everything one forward evaluation produced: what a schedule's `next_increment`
and a stopping rule see, what {func}`apply` consumes, and what a run reports as
its `last_evaluation`. Returned by {func}`evaluate` ({ref}`eki-step`).

**Fields.**

| field | shape | meaning |
| ----- | ----- | ------- |
| `step` | static `int` | the index of this step |
| `beta` | 0-d | the level *entering* the step |
| `ensemble` | `(J, P)` | the members that were evaluated — after inflation and after repair |
| `predictions` | `(J, N)` | their predictions, after repair |
| `whitened_residuals` | `(J, N)` | row $j$ is $W(y - v_j)$, after repair |
| `rms_parameter_spread` | 0-d | $\lVert A_u \rVert_F / \sqrt{(J-1)P}$, the root-mean-square per-coordinate ensemble standard deviation |
| `n_valid` | static `int` | how many members' predictions were finite |

**`ensemble` is not the ensemble the caller supplied.** It is post-inflation
and post-repair: the members that were actually run through the forward model
and that the update will actually move. Both transformations happen inside
{func}`evaluate` ({ref}`eki-inflation`, {ref}`eki-failures`), and neither is
recoverable by the caller — inflation consumes a split key, and repair depends
on which rows came back non-finite. Carrying the pair is what makes
`last_evaluation` answerable at all, and it costs nothing: the driver holds
both arrays live for the update regardless.

**`rms_parameter_spread` is scale-dependent**, and the name says so. It
averages per-coordinate standard deviations across parameters that may carry
unrelated units, so for an unscaled parameter vector it is dominated by the
largest-magnitude coordinate. The scale-free collapse diagnostic is the leading
singular value of $S$, which this layer deliberately does not compute
({ref}`eki-excluded`) and which is recoverable from `whitened_residuals`.

**Why `whitened_residuals` rather than just the misfit vector.** Write
$b_j = W(y - v_j)$ for the rows. Then

$$
\Phi_j = \tfrac12\lVert b_j\rVert^2,
\qquad
\bar b = W(y - \bar v),
\qquad
b_j - \bar b = -\,W\,(v_j - \bar v) ,
$$

so the misfits, the misfit of the mean prediction, **and the whitened
prediction anomalies** are all recoverable from this one array — the last
being, up to a sign and the $\sqrt{J-1}$, exactly the scaled whitened anomaly
matrix $S$ that {doc}`gaussian-contract`'s kernel is built on. It costs nothing:
the driver must whiten the residuals to compute the misfits at all, so this is
the array it already has. It buys the expressibility of a step-size rule that
needs the anomaly structure — a Langevin-type scheme sets its step from the
$J \times J$ matrix $\langle W(v_k - \bar v),\, b_j\rangle$. $N$ is likewise
recoverable from the trailing axis, so a criterion may be calibrated to the
observation dimension without the schedule storing it.

Every method and derived property below raises on a vmapped family, and the
class is otherwise an ordinary unbatched frozen pytree ({ref}`eki-jax`).

:::{note}
**The whitening is computed twice per step, and the layer accepts that.** The
driver whitens $y - v_j$ against the base $R$ to build this array, and the
update then whitens $A_v$ against $R/\Delta\beta$ inside
{class}`~pyeki.gauss.EnsembleJoint`. The two are the same computation up to a
factor: centring the rows of `whitened_residuals` gives $-A_v W^\top$ in exact
arithmetic, and the tempered whitener is $\sqrt{\Delta\beta}\,W$. The two
routes are **not** interchangeable in floating point: `pyeki.gauss` centres and
then whitens, precisely because centring already-whitened predictions cancels a
common $W\bar v$ and loses
$O(\varepsilon\sqrt{\kappa(R)}\,\lVert W\bar v\rVert / \lVert W A_v\rVert)$
— an error that grows as the ensemble collapses. The recovered matrix is a
diagnostic, never a substitute for the update's own.

For structured whiteners applying in $O(N)$ per vector this is absorbed by the
update's own $O(NJ^2)$. For a **dense** whitener it is not: the $O(JN^2)$
whitening dominates an update there, so the layer doubles the dominant cost of a
step on exactly the worst-conditioned problems. The duplication buys a strict
separation — diagnostics here, the conditioning kernel sealed in `pyeki.gauss`,
no intermediate crossing between them. If a dense-whitener consumer ever makes
the constant factor worth removing, the fix is to thread a precomputed whitened
matrix through the update's signature, and it is a contract change rather than
an optimization.
:::

**Derived properties**, computed on access, not cached:

| property | value |
| -------- | ----- |
| `misfits` | `(J,)`, $\Phi_j = \tfrac12\lVert b_j\rVert^2$ |
| `centre_misfit` | 0-d, $\Phi(\bar v) = \tfrac12\lVert \bar b\rVert^2$ |
| `n_members`, `u_dim`, `v_dim` | `int`, named as in {doc}`gaussian-contract` |

`centre_misfit` is *not* the mean of `misfits`, and the two must never be
confused: they differ by half the whitened prediction spread,

$$
\overline{\Phi_j} \;=\; \Phi(\bar v)
\;+\; \tfrac{J-1}{2J}\operatorname{tr}\bigl(W \widehat{C}_{vv} W^\top\bigr),
$$

coinciding only as the ensemble collapses. Both are wanted — a discrepancy
principle asks about the centre, a tempering criterion about the individual
members — so both are provided, under names that cannot be mistaken for one
another.

A third quantity, the misfit of the prediction *at the mean parameter*
$\Phi(G(\bar u))$, is a genuinely different object again and is deliberately
absent: computing it costs an extra forward-model evaluation
({ref}`eki-excluded`).

### `HistoryRecord`

One row of the history, built by the driver from the evaluation and the chosen
increment. **Every field is a 0-d array**, `step` and `n_valid` included (as
0-d integer arrays): `step`, `n_valid`, `beta`, `increment`, `beta_next`,
`misfit_mean`, `misfit_min`, `misfit_max`, `centre_misfit`, `spread`, and
`ess`.

**No field of `HistoryRecord` is static metadata, and that is a requirement
rather than an oversight.** Static fields live in a pytree's treedef, so two
records with different `step` values would have different treedefs, and
`jax.tree.map` across a history would raise instead of stacking. The history is
the one collection in the package meant to be stacked, so its element type must
be homogeneous as a pytree. `Evaluation.step` stays a static `int` for the
opposite reason: `FixedSchedule` indexes a Python tuple with it, which a traced
value cannot do.

- `ess` is $\mathrm{ESS}(\Delta\beta_t)$ at the increment actually taken,
  computed by the driver for **every** schedule, not only the ESS one. It is
  the single most informative number about whether a ladder is too coarse, and
  it costs $O(J)$. It does carry a tempering reading: for a schedule whose
  `beta` is accumulated pseudo-time rather than a temperature
  ({ref}`eki-variants`), $\mathrm{ESS}$ of $e^{-\Delta t\,\Phi}$ is a
  diagnostic of a target family that run is not traversing, and should be
  ignored rather than interpreted. It is computed unconditionally anyway, so
  that the history has one shape for every run.
- `beta_next` is `beta + increment`, stored rather than derived so that
  plotting a ladder needs no arithmetic.
- The per-member `misfits` vector is deliberately absent, as is anything else
  of size $J$ or larger: a history of hundreds of steps must stay cheap. A
  caller who wants per-member or per-observation quantities uses `iterate` and
  keeps them themselves, from `evaluation.whitened_residuals`.

**Conditioning diagnostics are the caller's to compute, and they are
reachable.** The most-watched number in a run is the leading singular value of
the scaled whitened anomaly matrix $S$ — whether the gain is saturating and the
ensemble collapsing — and no field here reports it. Two reasons: the update
computes its own SVD internally and has no channel to return one, and having
the driver compute a second SVD would add an $O(NJ^2)$ cost per step, the same
order as the update itself, for a diagnostic. It is not lost, because
`evaluation.whitened_residuals` **determines $S$ completely** — centring its rows
gives $-A_v W^\top$, as the evaluation's own section shows, to the accuracy that
section records — so a caller who wants the spectrum takes one SVD in an
`iterate` loop and pays for it deliberately. The related identity
$\log\det(\widehat C_{vv} + R) = \log\det R + \sum_i \log(1+\sigma_i^2)$,
which {doc}`gaussian-contract` records but does not expose, is available the
same way.

**The terminal record.** When a run ends on an evaluation whose update is then
discarded, that evaluation is recorded as a final `HistoryRecord` with `increment`
exactly `0.0`, `beta_next == beta`, and `ess` the literal `float(J)` — written
by the driver rather than obtained from `effective_sample_size`, since
`exp(log J)` is not `J` in floating point. It appears at most once, always
last, and in exactly two cases: a stopping rule fired
({ref}`eki-stopping`), or a schedule's `increment` returned `None`
({ref}`eki-schedules`). A run ended by the schedule's attributes performs no
such evaluation
and emits no such record. A zero increment in a
record therefore means "evaluated, then stopped"; this is the one zero
increment the layer permits, and it is written by the driver, never returned
by a schedule.

**Stacking the history** is `result.stacked`: a single `HistoryRecord` whose
`batch_shape` is `(T,)` and whose every field is `(T,)`-shaped — a family in
the sense of {ref}`contract-families`, inert to methods and legible in its
repr, which is exactly what a stacked history should be.

```python
plt.plot(result.stacked.step, result.stacked.misfit_mean)
```

It is a property rather than a documented one-liner because the one-liner is
`jax.tree.map(lambda *xs: jnp.stack(xs), *result.history)`, which **raises on
an empty history** — `jax.tree.map` with no trees has nothing to map over — so
every caller would have to guard on `n_steps` first, for the first thing anyone
does with a run. `stacked` returns `(0,)`-shaped fields there instead, which is
the answer that needs no branch. Records are homogeneous pytrees precisely so
that this works ({ref}`eki-diagnostics`).

The one case to guard is the **empty history**: a run whose ladder is already
exhausted on entry performs no evaluation, so `history == ()` and
`jax.tree.map` with no trees raises. Callers stacking a history must check
`result.n_steps` first. A stopping rule that fires at step 0 is *not* this case
— it emits a terminal record, so `n_steps == 1`. The layer ships no
tabular or plotting machinery.

(eki-driver)=
## The driver

Two loops over the two phases of {ref}`eki-step`: `iterate` is a generator,
`run` collects its output. **Neither is implemented over the other**, and `run`
is not implemented over `iterate`: both are thin wrappers around one private
driver. That is deliberate — see `run` below.

### `iterate(state, forward, y, noise_cov, *, schedule, update=TransformUpdate(), inflation=None, stop=None, on_failure="repair", max_steps=1000)`

A generator yielding `(EKIState, HistoryRecord, Evaluation)` after each
iteration, including the terminal evaluation-only iteration, and **returning**
the terminating `status` as its `StopIteration` value. It is the extension
point for anything that needs to *observe* or *interrupt* a run: per-step
checkpointing, custom logging, a wall-clock budget, stopping on parameter
stagnation, an early `break`. Exceptions propagate; abandoning the generator is
safe. Anything that needs to *revisit* a rung — backtracking, damping, trial
increments — uses `evaluate` and `apply` directly instead ({ref}`eki-step`).

The `Evaluation` is yielded because every recipe this contract recommends needs
it and the record cannot carry it: the record holds scalars only, by design
({ref}`eki-diagnostics`), so per-member misfits, the whitened residual matrix,
the singular values of $S$ and the posterior predictive are all reachable only
here. A caller who wants none of them ignores the third element.

A caller who ends the loop themselves has everything `EKIResult` needs — the
last yielded state, the records they accumulated, `INTERRUPTED`, and the last
yielded `Evaluation`:

```python
records = []
for state, record, evaluation in (it := iterate(state, forward, y, noise_cov,
                                                schedule=sched)):
    records.append(record)
    if time.monotonic() > deadline:
        break
else:
    ...
result = EKIResult(state=state, history=tuple(records),
                   status=INTERRUPTED, last_evaluation=evaluation)
```

### `run(...)`

The same arguments, returning an `EKIResult`. It is the interface the user guide
leads with.

**`run` does not consume `iterate`, and the reason is `status`.** All three
end-paths — the schedule's attributes, `next_increment` returning `None`, and a
stopping rule — look alike from outside: two of them emit an identical terminal
record and the third emits none ({ref}`eki-diagnostics`), so a consumer of the
yield stream cannot tell `"stopping_rule"` from `"schedule_exhausted"`. Routing
`run` through `iterate` would therefore mean either recovering the status from
`StopIteration.value` — which a plain `for` loop discards, making the obvious
implementation silently wrong — or reconstructing it from the record shape,
which is exactly the inference this layer refuses to ask of anyone. Both
functions instead wrap one private driver that knows why it stopped. `iterate`
still returns the status, for a caller who wants it and drives the generator
explicitly.

**`EKIResult`** is a plain frozen dataclass — the one value class in the layer
that is not a pytree, because it is a report and never an argument to traced
code. Fields:

| field | contents |
| ----- | -------- |
| `state` | the final `EKIState` |
| `history` | a tuple of `HistoryRecord`, one per ensemble evaluation |
| `status` | why the run ended (below) |
| `last_evaluation` | the `Evaluation` of the final forward evaluation, or `None` if the run made none |

All four are **keyword-only**, since the type is user-constructible
({ref}`eki-driver`) and `state`/`history` are as swappable as any other
same-arity pair. It is declared `eq=False` like every other class in the
package: a generated `__eq__` would raise on array comparison, and a generated
`__hash__` with it.

Properties: `ensemble` (`state.ensemble`), `beta` (`state.beta`), `mean` (the
ensemble mean, `(P,)`), `n_steps` (`len(history)`), `n_evaluations` (the number
of *forward calls*, which is `n_steps`, one per record — not $J\,n_{\text{steps}}$,
which counts member evaluations and is the caller's own multiplication),
`min_n_valid` (the smallest `n_valid` over the history, or `None` on an empty
one), `stacked` and the two termination booleans (all below).

**`last_evaluation` exists because the returned ensemble has never been
evaluated.** On a `"schedule_exhausted"` run the last update produces
`state.ensemble` and the loop then ends, so `state.ensemble` is one update
*past* the final forward evaluation and has no predictions at all; the last
record's misfits describe the ensemble *before* that update. Without this
field, a user's first two questions — what is my final misfit, and what does
the posterior predictive look like — would each cost another $J$ forward
evaluations for data the run already bought and threw away.

`result.last_evaluation.ensemble` is therefore **not** `result.ensemble` on a
schedule-exhausted run. On a stopping-rule termination the state is left
unchanged, so there they *are* the same array — the off-by-one is a property of
the exit path, not an invariant. Naming the fields rather than indexing a
2-tuple is what keeps the two readable apart; `[0]` and `[1]` on an
`(ensemble, predictions)` pair is the same hazard `EnsembleJoint`'s
keyword-only fields exist to forbid.

Moments beyond the mean are one line through the layer below:

```python
fit = Gaussian.from_samples(result.ensemble)
fit.cov.diag()                       # (P,) per-coordinate variances
fit.sample(key, 1000)                # draws from the fitted moments
```

{meth}`~pyeki.gauss.Gaussian.from_samples` holds the covariance as a
{class}`~pyeki.linalg.PSDLowRank` of width $J$, so nothing $P \times P$ is ever
formed and the rank ceiling of {ref}`eki-subspace` is visible in the type. The
result therefore carries `mean` for convenience and stops there.

Two things that line is **not**. It is not a further conditioning step:
`EnsembleJoint.condition` would apply another Kalman update and return an
ensemble shrunk one extra rung, which is the over-confident direction. And it
is not a posterior, whatever the run's configuration — it is the fit to the
terminal ensemble, under every caveat of {ref}`eki-honesty`.

`EKIResult.status` is one of exactly three strings, exported as module-level
constants so that a comparison cannot be misspelled. `run` produces only the
first two; the third is for a caller reporting an `iterate` loop they ended
themselves:

| status | meaning |
| ------ | ------- |
| `"schedule_exhausted"` | the ladder finished, by the schedule's attributes or by `next_increment` returning `None` |
| `"stopping_rule"` | the stopping rule fired; the last record is terminal |
| `"interrupted"` | the run was ended by its caller, not by a policy |

`"interrupted"` is never produced by `run`. It exists because `iterate` is the
sanctioned way to impose a wall-clock budget or an external cancellation, and a
caller who `break`s out of that loop needs a legal way to report what happened:
**`EKIResult` is user-constructible**, and this is the status for a run that
ended on the caller's terms. Without it the recommended pattern for the
recommended extension point would produce an object the type's own contract
forbids.

Comparing against a literal is the expected use, so the constants matter:
`status == "stopping-rule"` with a hyphen is `False` forever and reads as a
completed budget. The implementation annotates `status` and `on_failure` as
`Literal[...]`, validates `on_failure` at the call, and exports
`SCHEDULE_EXHAUSTED`, `STOPPING_RULE` and `INTERRUPTED`.

**Two termination booleans, not one.** `stop_fired` is
`status == STOPPING_RULE`; `budget_complete` is
`status == SCHEDULE_EXHAUSTED`. Both are properties rather than further
statuses, because they answer questions about *intent* that `status`
deliberately does not model — and there are two such questions, one per form of
the iteration:

- The optimization form asks *did it fit?* A run with
  `FixedSchedule.constant(1.0, 200)` and a `DiscrepancyStop` that never fires
  ends at $\beta = 200$ with `status="schedule_exhausted"` — the *same* status
  a successfully completed sampling ladder reports — having failed to fit the
  data. `stop_fired` is `False` there, and that is the one-word answer.
- The sampling form asks *did the ladder finish?* `budget_complete` answers it,
  and the pair together makes the layer's one real trap legible from the result
  alone: `DiscrepancyStop` on a budgeted ladder, which {ref}`eki-axes` calls
  constructible and usually a mistake, reports `stop_fired=True,
  budget_complete=False`, ending a sampling run at an arbitrary intermediate
  level.

:::{note}
**There is deliberately no single `converged`.** A boolean by that name has to
choose one of the two questions and then answers the other one wrongly: defined
as "a stopping rule fired" it is `False` on a perfectly completed $\beta = 1$
sampling ladder, which is the layer's headline use and did everything right,
and `True` on the $\beta = 0.4$ early exit above, which is the case
{ref}`eki-axes` says nothing will warn about. The name promises a statement
about the answer while the definition can only report which branch the loop
exited, so the layer does not offer it.
:::

There is no `"max_steps"` status, because **exceeding `max_steps` raises**. The
bound is a safety net against a schedule that can never be exhausted and a run
with no stopping rule; a genuinely step-limited run is a `FixedSchedule` with
that many rungs, or a `break` in an `iterate` loop. A sampling run that
silently returned an ensemble at $\beta = 0.7$ labelled as a posterior is
the failure `stop_fired` and `budget_complete` exist to expose.

**Progress reporting.** The driver emits one record per step at `INFO` on the
logger named `pyeki.eki`, carrying the step, the level, the increment and the
mean misfit, and one at `WARNING` when any member fails. This is the standard
library's `logging` and nothing more: no handler is installed, no configuration
is read, and a caller who does nothing sees nothing. A run of an expensive
model can last hours, and a library that emits *nothing* over that span, with
no API to add without dropping to `iterate`, is a gap rather than a design
choice — while instrumentation as a feature (timings, profiles, progress bars)
stays excluded ({ref}`eki-excluded`).

**The budget and the bound are checked against each other at entry.** A
schedule exposing `beta_target` and a floor implies a worst case of
$\lceil \beta_{\text{target}} / \delta_{\min} \rceil$ rungs, and the driver
raises `ValueError` **before the first forward evaluation** when `max_steps` is
below it. This is possible only because exhaustion is declarative
({ref}`eki-schedules`), and it converts the layer's sharpest remaining
foot-gun — a run that spends its entire evaluation budget and then reports
`EKIError` on precisely the badly-conditioned problems the floor exists to
rescue — into an immediate, actionable error. A schedule that does not expose a
floor is not checked.

The shipped defaults satisfy it with no slack: `beta_target=1.0` and
`min_increment=1e-3` give $\lceil 1/10^{-3} \rceil = 1000 = $ the default
`max_steps` ({ref}`eki-adaptive`). Note that **raising the budget breaks the
relation just as lowering the floor does** — `beta_target=2.0` at the default
floor needs 2000 — which is why the check is arithmetic on the attributes
rather than a note telling the caller to keep two defaults in step.

**Arguments not otherwise specified.** `y` must be a `(N,)` array;
`noise_cov` a `PSDLinOp` of side $N$ supporting `whiten`, with
`batch_shape == ()`; `max_steps` a positive `int`; `on_failure` one of the two
strings of {ref}`eki-failures`. The driver validates the problem's shapes once,
before the first evaluation, and the forward model's output shape at every
evaluation.

(eki-prng)=
## Randomness

- **The state owns the stream.** `EKIState.key` is the only source of
  randomness in a run, and a run is fully determined by its initial state and
  its policies.
- **Splitting is pinned.** `EKIState.from_prior` splits once, into
  `(key_sample, key_state)`. Each step splits into exactly three,
  `(key_next, key_inflate, key_update)`, in that order, **whether or not
  inflation and the update consume theirs**. Fixed arity is what makes the
  update's stream independent of the inflation choice, so that switching
  inflation on does not silently change the update's perturbations.
- **Policies consume their key whole**, per {doc}`gaussian-contract`. No
  policy stores a key or advances hidden state.
- The pinning is subject to the same caveat as the gauss layer's: identical
  arrays across pyEKI releases for a fixed JAX version and PRNG
  configuration. pyEKI never changes the draw on its side; the test suite
  snapshots a short run so that a JAX-side stream change is detected rather
  than absorbed.
- Resuming from a checkpointed state reproduces the tail exactly, which is
  the operational point of all of the above ({ref}`eki-conformance`).

(eki-validation)=
## Validation and errors

The four-tier scheme of {ref}`contract-validation` applies, with the gauss
layer's extension that tier 4 also runs at call time in debug mode. Tier 3
here covers the driver's per-step static checks, which are unconditional
because they are pure Python over shapes.

| tier | checks | examples |
| ---- | ------ | -------- |
| 2. construction | ranks, static sizes, operator types, field domains | `ensemble` rank ≠ 2; $J < 2$; a key that is not a typed key, by shape **or** dtype; `FixedSchedule` increments not all positive; `ess_fraction` outside $(0,\ 1-10^{-6}]$; `n_bisect` $< 1$; `divergence_budget` not `None`, not finite, or $\le 0$; `max_increment` not finite; `min_increment` $>$ `max_increment`; `beta_target` $\le 0$; `tau \le 0` |
| 3. call | problem and per-step shapes, policy outputs, string arguments | `y` not `(N,)` or not finite; `noise_cov` side ≠ $N$ or a family; `max_steps` not a positive `int`; the forward model's output not `(J, N)` or not of a real floating dtype; an inflation's output not `(J, P)`; an update's output not `(J, P)`; `AdditiveInflation.cov` of side ≠ $P$; a schedule increment that is non-scalar, non-finite, or not strictly positive; `on_failure` not one of the two permitted strings |
| 4. value (debug) | finiteness of the initial ensemble and of `beta`; finiteness of inflation fields; positivity of `anomaly_factor` | violations yield `nan` or a silently wrong ladder outside debug mode |

`y`'s finiteness is checked **unconditionally**, not at tier 4, because it is
$O(N)$ once per run and because a non-finite `y` otherwise surfaces as one of
two unrelated and misdiagnosing errors — a full-budget run of `nan` updates
under `AdaptiveMisfitSchedule`, or "increment not finite" under
`AdaptiveESSSchedule` — neither of which names `y`.

**String arguments are validated, never silently defaulted.** `on_failure` is
checked against its two permitted values at the call, and an unrecognized one
raises rather than falling back to `"repair"` — a typo such as `"Raise"` must
not quietly select the opposite behaviour on a run that then discards failures
it was asked to reject. `status` is produced by the layer rather than consumed
from it, and is exported as constants for the same class of reason
({ref}`eki-driver`).

The layer defines **one** new exception, because a run is long and expensive
enough that a caller wants to catch its failures specifically — to checkpoint
and investigate — without catching every `RuntimeError` in the process:

**`EKIError(RuntimeError)`**, raised for the four conditions under which a
run cannot continue: `max_steps` exceeded; fewer than two valid members; any
invalid member under `on_failure="raise"`; and a non-finite updated ensemble.
Each message must name the step index, the level, and the condition; the
`on_failure="raise"` message must additionally name the number and indices of
the failed members; the `max_steps` message must additionally name the schedule and
whether a stopping rule was supplied.

**`EKIError` carries the run, and this is normative.** It has two attributes,
`state` and `history`: the last good `EKIState` and the records accumulated up
to the failure, populated on **every** raise path. Without them a caller
catching it has nothing to checkpoint — `run` builds its `EKIResult` internally
and the state lives inside the driver — so the four failures that lose a run
become four recoverable ones for the cost of two attribute assignments. On an
expensive model with most of a long ladder completed, that is the difference
between a resumable checkpoint and discarded work:

```python
try:
    result = run(state, forward, y, noise_cov, schedule=sched)
except EKIError as exc:
    checkpoint(exc.state)          # resume from here after investigating
    diagnose(exc.history)
```

A caller resuming from `exc.state` is doing exactly what {ref}`eki-state`
specifies, so nothing further is needed to make the recovery exact.

Everything else follows the layers below: `ValueError` and `TypeError` for
validation, and `UnsupportedOpError` propagated **unmodified** from the
operator layer whenever a covariance lacks an operation a policy needs. The
layer never catches `UnsupportedOpError`, never wraps it, and never falls back
to dense linear algebra on the caller's behalf; the escape hatch is the same
`densify` at the same call site.

| condition | raises |
| --------- | ------ |
| wrong rank, size, or disagreeing shapes at construction | `ValueError` |
| non-operator or wrong-level covariance; a non-typed PRNG key | `TypeError` |
| problem shapes, forward-model output shape, or a policy's output invalid | `ValueError` |
| `on_failure` not `"repair"` or `"raise"` | `ValueError`, at the call |
| `noise_cov` a vmapped family | `ValueError` |
| covariance lacking `whiten` or `factor` where required | `UnsupportedOpError`, from the operator layer |
| `max_steps` exceeded; $J_v < 2$; non-finite updated ensemble | `EKIError` |
| violated value precondition | `ValueError` in debug mode; `nan` or a silently wrong result otherwise |

(eki-jax)=
## JAX integration

Every rule of {ref}`contract-jax` binds the value classes: the same field
allowlist, `static_field()` for everything else, constructor-bypassing
unflatten, `eq=False`, identity hashing, never `static_argnums`,
constructors-store-classmethods-compute. `pyeki.eki` exports no class
decorator; the classes reuse the operator layer's machinery internally, as
`pyeki.gauss` does.

Beyond that, three requirements specific to a layer that owns a loop:

- **The forward model is never traced.** The driver must not `jit`, `vmap` or
  otherwise transform it, and must not require it to be traceable. Callers who
  want their model compiled do it themselves.
- **Compilations must not grow with the number of steps.** The driver applies
  `jax.jit` to its array computations internally; the exact partition is an
  implementation detail, but the number of traces over a run must be bounded
  by a small constant. In particular the increment must reach the update as a
  traced 0-d array, never as a Python float baked into a constant — the
  reason {doc}`gaussian-contract` and {doc}`linop-contract` insist that the
  scaled operator's scalar be a 0-d array field. The repair branch of
  {ref}`eki-failures` and the two `on_failure` modes may each add a variant;
  the number of steps may not. This is a conformance obligation, not an
  aspiration.
- **Debugging is `jax.disable_jit()`**, not a driver flag. The layer ships no
  `jit=` argument ({ref}`eki-excluded`).

### Families

The family machinery of {ref}`contract-families` applies to the pytree value
classes, in its gauss instantiation ({ref}`gauss-jax`):

- `EKIState`, `Evaluation` and `HistoryRecord` each have a required
  `batch_shape` property, computed from the leading axes of each array **field**
  beyond its core rank — `EKIState.ensemble` 2,
  `Evaluation.whitened_residuals` 2, `Evaluation.ensemble` 2,
  `Evaluation.predictions` 2, `key` 0, and `beta`,
  `rms_parameter_spread` and every
  `HistoryRecord` field 0 — combined by broadcasting with `ValueError` on
  mismatch. Derived properties such as `Evaluation.misfits` are not fields and
  do not enter the computation. Directly constructed objects always report `()`.
- When `batch_shape` is non-empty the object is **inert**: every method and
  array-computing property raises `ValueError` naming the object, the
  operation, the batch shape and the remedy. The static `int` properties,
  `batch_shape` itself, and `repr` still answer.
- Family `repr` takes the `vmapped(...)` form of {ref}`contract-repr`.

Families are a legibility concern here rather than a use case: the driver loop
is Python and calls an untraceable forward model, so a run cannot be `vmap`-ed
({ref}`eki-excluded`). A state that acquired batch axes did so by accident,
and the point of the machinery is that it says so.

(eki-variants)=
## Expressing variants

Not normative. This section is the evidence for the three-axis claim of
{ref}`eki-axes`: a variant that cannot be expressed here is a reason to revisit
the design.

| variant | expressed as |
| ------- | ------------ |
| approximate posterior sampling by tempering (the ensemble smoother with multiple data assimilation of Emerick and Reynolds) | `FixedSchedule.uniform(T)` or `AdaptiveESSSchedule(beta_target=1.0)`, with the default `TransformUpdate` |
| the same, in its classical perturbed-observation form | those schedules with `update=PathwiseUpdate()` |
| a single Kalman update (the one-step linearized approximation) | `FixedSchedule.constant(1.0, 1)` |
| EKI as an iterative regularization method | `FixedSchedule.constant(1.0, n)` with `stop=DiscrepancyStop()` |
| adaptive-regularization EKI (the EKI-DMC scheme of Iglesias and Yang) | `AdaptiveMisfitSchedule` with `stop=DiscrepancyStop()` |
| inflation-stabilized variants | any of the above with `inflation=` |
| Tikhonov-regularized EKI | an augmented problem; see below — no new code |
| localized EKI | `update=` an update rule from `pyeki.localize` |
| a Langevin-type ensemble sampler (the ensemble Kalman sampler of Garbuno-Iñigo and co-authors) | a custom `EnsembleUpdate` holding the prior, using `increment` as its step size |

**Two need detail.**

*Tikhonov regularization needs no code at all.* Appending the parameters to
the predictions and the prior mean to the data,

```python
forward_aug = lambda u: jnp.concatenate([forward(u), u], axis=-1)
y_aug = jnp.concatenate([y, prior.mean])
noise_aug = block_diag(noise_cov, prior.cov)
```

adds $\tfrac12\lVert C_0^{-1/2}(u - m_0)\rVert^2$ to the tempered misfit,
because the whitened residual of the appended block is exactly that. The
prior returns as data, re-imposed at every rung **in addition to** whatever it
already contributes through the initial ensemble — which is the whole content
of the variant. It requires `prior.cov` to support `whiten`, uses
`pyeki.linalg`'s existing {func}`~pyeki.linalg.block_diag`, and needs nothing
from this layer. The observation dimension becomes $N + P$, so the cost is the
one the augmentation implies and nothing more.

:::{warning}
**The augmentation is for the optimization form, and combining it with a
$\beta = 1$ budget double-counts the prior.** The words "in addition to" above
are the whole hazard. Started from a prior ensemble and run to $\beta = 1$, the
prior enters twice — once through the initial ensemble and once through the
appended data block — and the result is over-concentrated, by a factor that
grows with how informative the prior is. Nothing raises; the run looks
completely normal.

The variant is sound where it comes from: as $\beta \to \infty$ the initial
ensemble's contribution is negligible against the accumulated data term, so the
double counting vanishes and the augmentation is simply the regularizer. That
is the optimization form. To use the augmentation with a $\beta = 1$ budget and
still get a posterior, the initial ensemble must be **diffuse** rather than
prior-distributed, so that the appended block is the only place the prior
enters.

Pairing this recipe with `AdaptiveESSSchedule(beta_target=1.0)` from a prior
ensemble is therefore a confidently wrong posterior with no error, which is
the failure this warning exists to expose.
:::

The regularization weight is the noise block's scale: `block_diag(noise_cov, (1 / lam) * prior.cov)` gives the penalty
$\tfrac{\lambda}{2}\lVert C_0^{-1/2}(u - m_0)\rVert^2$, a
{class}`~pyeki.linalg.PSDScaled` that whitens as cheaply as `prior.cov` does.
Centring at $m_0$ rather than at the origin is a choice, and the origin is
recovered by passing a zero mean.

*A Langevin-type update fits the protocol.* Such rules — the ensemble Kalman
sampler and its affine-invariant relatives ({ref}`eki-references`) — add a
prior-drift and a diffusion term to the Kalman-like term and are driven by a step size rather
than a temperature budget. They need the ensemble, the predictions, $y$, the
noise, a step size, a key, and the prior — which is exactly the update
signature plus a field on the rule, and it is why `increment` is passed
separately from `noise_cov` ({ref}`eki-updates`) and why `beta_target=None`
exists. For such a rule `EKIState.beta` reads as accumulated pseudo-time
rather than a tempering level; the layer keeps the name and the bookkeeping,
which are identical.

One caveat, which cuts against a property the package advertises. The drift term involves the prior **precision** $C_0^{-1}$, so such
a rule needs `prior.cov.supports("solve")` — while a plain run needs only
`factor`, which is why {doc}`design` can say that a prior with no cheap inverse
is perfectly usable. The most structurally interesting priors are exactly the
ones where that matters: a `factor`-only covariance drives every shipped
configuration and raises `UnsupportedOpError` at the first Langevin step. The
step-size rule such a variant wants is expressible as a `Schedule`, since the
evaluation carries the whitened residual matrix ({ref}`eki-schedules`).

**What does not fit** is settled by one criterion: a variant belongs here when
its members move freely and are carried between steps, when there is one
ensemble and one forward model, and when the data are fixed for the run.
Mean-field and unscented schemes fail the first, multilevel schemes the second,
mini-batch schemes the third. {ref}`eki-excluded` gives each its entry and its
reason.

(eki-consumers)=
## How the layers around this one connect

Not normative, but the design was shaped against these call sites.

**`pyeki.gauss` is consumed only through `EnsembleJoint`'s two update
methods**, once per step, with the tempered operator `noise_cov / increment`.
No other gauss surface is used by the shipped rules: not `condition`, not
`Gaussian.log_density`, not the conditioning primitives. `Gaussian` is used
once, by `EKIState.from_prior`, for its `sample`.

**`pyeki.localize` will supply an `EnsembleUpdate`.** The driver needs no
knowledge of localization and localization needs no change to the driver: the
update signature gives it the ensemble, the predictions, the residual data and
the increment; the conditioning primitives give it the per-block analyses;
{ref}`contract-composites` gives it the noise operator's block anatomy. Two
things it must bring itself, neither of which this layer supplies: observation
**locations**, which appear nowhere in `pyeki.eki` and so live as static fields
on the rule, and the neighbourhood and taper definitions.

One real limit. A local analysis needs the noise covariance restricted to a
neighbourhood, and {ref}`gauss-consumers` states that extracting a principal
submatrix of a *correlated* block is not an operator-layer operation. So
localization composes cleanly for **diagonal noise, or neighbourhoods aligned
to the noise operator's blocks**, and not for arbitrary neighbourhoods cutting
across a correlated block — which is the case {doc}`design` spends a section
motivating. This is a constraint on `pyeki.localize`'s neighbourhood
construction, not a gap in this layer, but the claim that everything it needs
is already contractual would be false without it.

**Hyperparameter estimation, if it ever arrives**, differentiates a marginal
likelihood at fixed data — not a run. The loop is not differentiable, since
the forward model may not be; `log_density` in the gauss layer is, which is
where that consumer belongs.

(eki-repr)=
## `repr`

Type name and static sizes, never array contents, matching
{ref}`contract-repr` and {ref}`gauss-repr`:
`EKIState(n_members=64, u_dim=12, step=3)`,
`Evaluation(step=3, n_members=64)`, `HistoryRecord(step=3)`. Policy objects
print their static fields, which are small and informative:
`AdaptiveESSSchedule(ess_fraction=0.5, beta_target=1.0)`,
`MultiplicativeInflation(anomaly_factor=1.02)` — with the one exception that a
policy holding a *large* static field summarizes it instead
(`FixedSchedule(n_steps=200, total=200.0)`, {ref}`eki-schedules`), since the
general rule assumes those fields are small. `EKIResult` prints its status and
counts: `EKIResult(status='schedule_exhausted', n_steps=17, beta=1.0)`.
`repr` never raises.

(eki-surface)=
## Public surface

`pyeki.eki` exports exactly: the value classes `EKIState`, `Evaluation`,
`HistoryRecord`, `EKIResult`; the protocols `EnsembleUpdate`, `Schedule`,
`StoppingRule`, `Inflation`; the update rules `TransformUpdate`,
`PathwiseUpdate`; the schedules `FixedSchedule`, `AdaptiveESSSchedule`,
`AdaptiveMisfitSchedule`; the stopping rule `DiscrepancyStop`; the inflations
`MultiplicativeInflation`, `AdditiveInflation`; the driver `run` and `iterate`
and the three step functions `evaluate`, `apply` and `advance`; the helpers
`misfits`, `effective_sample_size`, `repair_failed_members`; the
status constants `SCHEDULE_EXHAUSTED`, `STOPPING_RULE`, `INTERRUPTED`; and the
exception `EKIError`. Anything else is private, and no consumer may depend on
it.

The three helpers are public for three different reasons, and it is worth
saying which: `misfits` because the misfit convention of
{ref}`eki-notation` must be applicable outside a run — to a validation set, or
to a candidate parameter — without reimplementing the factor of $\tfrac12$;
`effective_sample_size` because it is the one criterion a custom schedule is
most likely to want and the least likely to get right in log space; and
`repair_failed_members` because its moment identities are worth testing
directly, and a caller driving the phases by hand needs it.

| helper | signature | returns |
| ------ | --------- | ------- |
| `misfits(y, predictions, noise_cov)` | `(N,), (..., N), PSDLinOp -> (...)` | $\tfrac12\lVert W(y - v)\rVert^2$, batched per the operator layer's contract |

`misfits` and {attr}`Evaluation.misfits` are deliberately the same name for the
same quantity, and the conformance suite asserts they agree: the free function
is how a caller applies the convention of {ref}`eki-notation` outside a run,
and the property is how a policy reads it inside one. "Whitened" is not in
either name — the misfit is defined as the whitened quadratic form, so there is
no unwhitened one to distinguish it from, and the word would attach to the
misfit rather than to the residual it actually describes.
| `effective_sample_size(misfits, increment)` | `(J,), scalar -> 0-d` | $\mathrm{ESS}$ of $e^{-\delta\Phi}$, computed in log space |
| `repair_failed_members(*, ensemble, predictions, valid)` | `(J,P), (J,N), (J,) bool -> (J,P), (J,N)` | the mean-preserving repair of {ref}`eki-failures` |

`repair_failed_members` is **keyword-only** for the reason
{ref}`eki-updates` gives for the update protocol: its first two arguments are
arrays whose shapes coincide whenever $P = N$, so a positional signature would
let them be transposed with no error at all, returning finite and plausibly
wrong numbers. `valid` must be a `(J,)` boolean array, and $J_v \ge 2$ is a
precondition the helper checks, since at $J_v \le 1$ the valid-member mean is
`nan`.

(eki-testing)=
### `pyeki.eki.testing`

Unlike `pyeki.gauss`, this layer ships a conformance harness, because unlike
`pyeki.gauss` it is open to extension: {ref}`eki-axes` names it the one place
where pyEKI is deliberately extensible at the algorithm level, and the same
asymmetry that gives `pyeki.linalg` a `check_operator` applies here.

| function | checks |
| -------- | ------ |
| `check_schedule(schedule, evaluation)` | `n_steps` and `beta_target` are present, of the right types, and unchanged by reads; `next_increment` returns `None` or a scalar, finite, strictly positive value; **purity**, by calling twice on the same evaluation and comparing bit-exactly |
| `check_update(update, key, **operands)` | the result is `(J, P)` with the incoming dtype; determinism given the key; the subspace property of {ref}`eki-subspace`, unless the rule declares it leaves the span; `jit`-safety with static shapes |
| `check_inflation(inflation, key, ensemble)` | shape preservation; purity; that the mean is preserved, unless the rule declares otherwise |
| `check_stopping_rule(stop, evaluation)` | a Python `bool` is returned; purity |

Each takes a policy and a small synthetic `Evaluation`, which the module also
provides a constructor for, since a user testing their own schedule should not
have to run a forward model to get one.

**Purity is the reason the harness exists.** {ref}`eki-updates` records that a
policy holding iteration state silently breaks resumption, and concedes that
this is "a failure the conformance suite can catch in the package's own rules
and cannot catch in a user's". A harness is exactly the answer to that: calling
a policy twice on one evaluation and comparing is two lines, and it catches the
schedule-that-counts-its-own-calls bug — which this contract lists among the
silent-failure classes — in a user's code rather than only in ours.

The shipped policies are all run through these checks, as the operators are
through `check_operator`.

(eki-conformance)=
## Conformance

Obligations on `tests/`. The package's conventions apply: exactness tests
check closed forms rather than tolerances chosen to pass, and references are
written independently of the code under test. The suite must verify at least:

1. **Telescoping exactness — the headline test.** An affine forward model, a
   Gaussian prior, an ensemble whose empirical moments equal the prior's
   exactly (the QR-of-ones fixture of {ref}`gauss-consumers`), the square-root
   update, and a multi-rung ladder summing to 1: the final ensemble's mean and
   covariance equal the one-shot posterior to floating point. Checked for
   several ladder lengths and for non-uniform increments.
2. **The sensitivity of that test is itself verified.** A targeted regression
   test computes the same run with the $R/\beta_t$ mis-scaling written out
   locally and asserts that it disagrees by the documented margin, so that
   test 1's tolerance is known to be tight enough to catch the layer's
   signature bug rather than merely to pass.
3. **Stochastic update composition.** On the same affine problem, a ladder
   with `PathwiseUpdate` reproduces, elementwise for a fixed key, a
   hand-written dense reference that applies the perturbed-observation formula
   rung by rung; and its posterior moments match the one-shot posterior in
   expectation, tested as a mean over many keys with a tolerance derived from
   the $KRK^\top/J$ scale rather than tuned.
4. **Schedules.** `FixedSchedule` takes exactly its increments in order and
   exhausts after exactly $T$ steps, having made exactly $T$ ensemble
   evaluations, **and completes under `max_steps == T`** — the executable form
   of the exhaustion-before-bound ordering of {ref}`eki-step`, and a
   regression test for the inversion that would make the natural `max_steps`
   always raise. A schedule whose `next_increment` returns `None` ends the run
   with `status="schedule_exhausted"` and a terminal record.

   **Both** adaptive schedules reach `beta_target` to within `budget_tol`
   without ever exceeding it, leave the exhaustion check true on arrival,
   never return a non-positive increment, and respect the clamp precedence of
   {ref}`eki-adaptive` — a case where the floor binds, a case where the budget
   cap beats the floor, and a case where `max_increment` binds. Each takes
   exactly $\min(\delta_{\max}, \beta_{\text{target}} - \beta)$ when every
   misfit is identical, and each runs unbounded (`beta_target=None`) without
   raising, which is the executable form of the conditional budget term.

   **Rung counts are asserted exactly, not approximately**, by an instrumented
   model: a budgeted schedule whose criterion is `inf` and whose ceiling is
   $0.3$ against a budget of 1 takes exactly four rungs, with increments
   $(0.3, 0.3, 0.3, 0.1)$. One test pins the `>=` in the exhaustion check,
   `budget_tol`, cap-beats-floor, and the absence of a trailing dribble rung.

   `AdaptiveESSSchedule`'s returned increment attains its ESS target before
   the floor is applied — asserted at a **small `n_bisect`**, since at the
   default of 50 the two bracket ends differ by $2^{-50}$ and no float64
   tolerance can tell them apart, so the obligation would not distinguish
   returning `lo` from returning `hi`. A second test reproduces the increment
   against a hand-written bisection at the same `n_bisect`, pinning the
   iteration count and the log-space computation together. Both run in the
   interior regime, with neither clamp binding.

   `AdaptiveMisfitSchedule` returns the larger of its two bounds, with a test
   for each regime — the mean bound binding when the misfit coefficient of
   variation exceeds $1/\sqrt{\theta}$ and the variance bound below it — and
   is `inf`-guarded rather than `nan` at zero misfit spread and at zero mean
   misfit. A `nan` misfit must **not** yield the largest allowed step, which
   the single-`where` guard would deliver silently.

   **The entry-time budget check** of {ref}`eki-driver` raises before the
   first evaluation when `max_steps` cannot accommodate the schedule's own
   floor-bound worst case, and does not raise at the shipped defaults. Both
   halves are asserted by arithmetic on the attributes rather than by running
   $10^3$ steps; a scaled-down ladder (`beta_target=0.01`,
   `min_increment=1e-3`, `max_steps=10`) exercises the bound executably at 1%
   of the cost.
5. **The ESS criterion.** `effective_sample_size` is $J$ at zero increment,
   monotone non-increasing along a grid of increments, correct against a
   direct small-value computation, equal to $J/(1+\mathrm{cv}^2)$ for the
   weights it defines, and finite for misfits of order $10^{4}$ where the
   naive non-log-space formula returns `nan` — the latter as a targeted
   regression test.
6. **The two adaptive criteria are distinct, and measurably so.** On the same
   ensemble, `AdaptiveMisfitSchedule`'s increment drives the effective sample
   size to within a few members of its floor of 1, and is at least five times
   `AdaptiveESSSchedule`'s at a target of $1/2$ — pinning the comparison in
   both directions, so that neither schedule can silently drift into
   implementing the other's criterion. Both halves are asserted; the first is
   the executable form of {ref}`eki-schedules`' floor claim.
7. **Stopping.** `DiscrepancyStop` fires exactly when $2\Phi(\bar v) \le
   \tau^2 N$, is consulted before the increment, ends the run with status
   `"stopping_rule"` and a terminal record whose increment is exactly zero,
   and can end a run at step 0 with an empty update history.
8. **`max_steps` and the error payload.** An unbounded schedule with no
   stopping rule raises `EKIError` at the bound, with a message naming the
   schedule; the bound is never reached by a schedule with a budget and a
   positive floor. **Every `EKIError` raise path carries `state` and
   `history`** — the bound, $J_v < 2$, `on_failure="raise"`, and a non-finite
   update — and resuming a run from a caught `exc.state` continues it exactly,
   which is what makes the attributes worth having.
9. **Failure handling.** `repair_failed_members` leaves valid members
   **bit-identical**, maps failed members to the valid centre, reproduces the
   valid-member mean exactly, and gives an all-$J$ covariance and
   cross-covariance equal to the valid-member ones scaled by exactly
   $(J_v-1)/(J-1)$ — the damping of {ref}`eki-failures`, pinned as an equality
   so that a reintroduced anomaly rescaling fails the test rather than passing
   a tolerance. A step in which nothing fails is bit-identical to the same step
   with failure handling absent. `on_failure="raise"` raises; an unrecognized
   `on_failure` raises `ValueError` rather than defaulting; $J_v < 2$ raises
   under either mode; a `nan` prediction row does not produce a `nan` misfit
   statistic.
10. **Inflation.** `MultiplicativeInflation` preserves the mean exactly and
    scales the empirical covariance by `anomaly_factor**2` exactly; a run with it is
    unchanged in the members' span. `AdditiveInflation` preserves the mean
    exactly, matches its pinned elementwise definition, and moves the ensemble
    out of its initial affine subspace — while a run without it stays inside,
    which is the executable form of {ref}`eki-subspace`.
11. **Misfits.** `misfits` matches a dense
    $\tfrac12 (y-v)^\top R^{-1} (y-v)$ at batch ranks 0, 1 and 2, is invariant
    across two noise operators representing the same $R$ with different
    whiteners, and pins the factor of $\tfrac12$ against a closed-form case.
    `centre_misfit` differs from `misfit_mean` by exactly
    $\tfrac{J-1}{2J}\operatorname{tr}(W\widehat{C}_{vv}W^\top)$ — a divisor
    the derivation gets wrong easily and which no tolerance-based test would
    catch. `misfits` and `centre_misfit` are recovered from the evaluation's
    stored `whitened_residuals` alone, and the whitened prediction anomalies
    recovered from it agree with $-W A_v$.
12. **Reproducibility and resumption.** The same initial state and policies
    give bit-identical results. Stopping a run after $k$ steps and resuming
    from the returned state reproduces the uninterrupted run's remaining
    records and final ensemble bit-exactly — the executable form of the
    policy-purity rule. `iterate` and `run` agree.
13. **The optimization form.** On an affine problem, a run with
    `FixedSchedule.constant(1.0, n)` and `DiscrepancyStop` terminates, its
    misfit decreases monotonically in the affine case, and its centre
    approaches the least-squares solution restricted to the initial
    ensemble's affine subspace — the correct target given
    {ref}`eki-subspace`, not the unrestricted minimizer.
14. **JAX.** The array computations run under `jit`; a multi-step run
    **compiles a bounded number of times, independent of the number of
    steps** (asserted against a compilation counter, not inspected by eye) —
    the executable form of the traced-increment requirement; flatten and
    unflatten preserve type and behaviour for all three pytree classes, with
    sentinel leaves; families report their batch shape, take the
    `vmapped(...)` repr, and refuse every method.
15. **The history stacks.** `result.stacked` yields a `HistoryRecord` with
    `batch_shape == (T,)` and `(T,)`-shaped fields, including `step` and
    `n_valid`, and `(0,)`-shaped fields on an empty history. This is a targeted
    regression test for the treedef trap of {ref}`eki-diagnostics`: declaring
    either field as static metadata makes every record a different pytree type,
    and the underlying `jax.tree.map` then raises.
16. **The two phases are public and compose.** `advance` equals
    `apply(state, evaluate(...))` bit-exactly, and reproduces the corresponding
    rung of an equivalent `run`. Applying twice from one `Evaluation` at
    different increments leaves both the state and the evaluation untouched and
    yields the two corresponding results — the property the backtracking loop of
    {ref}`eki-step` relies on — with the forward model called **once**, counted
    by an instrumented model. `apply` raises `ValueError` on an `Evaluation`
    belonging to a different state, and validates its increment before the
    model would have run. `iterate` yields what a hand-written loop over the two
    phases yields, given the same schedule decisions.
17. **Degeneracy.** $J = 2$, $N = 1$, and $P = 1$ all work. A collapsed
    ensemble produces no `nan` and, with zero prediction anomalies, no
    movement. A forward model returning `nan` for every member raises rather
    than propagating. A `nan`-producing update raises `EKIError` naming the
    step.
18. **Validation and repr.** Every tier-2 and tier-3 rule of
    {ref}`eki-validation` raises as specified; reprs match {ref}`eki-repr`
    with no array data, and `FixedSchedule`'s summarizes rather than printing
    its increments; the pinned prior draw and a short run's output are
    snapshotted so a JAX-side PRNG change is detected rather than absorbed.
19. **The result reports the run.** `status` takes only the three permitted
    values and equals the exported constants. `stop_fired` and
    `budget_complete` are each true exactly on their own status, tested on
    four runs: a completed budget with no stopping rule (`budget_complete`
    only — the case a single `converged` would report as failure), an
    optimization run whose ladder ran out before its `DiscrepancyStop` fired
    (`budget_complete` only), a `DiscrepancyStop` firing on a budgeted ladder
    (`stop_fired` only, at $\beta < \beta_{\text{target}}$ — the trap of
    {ref}`eki-axes`), and a user-built `INTERRUPTED` result (neither). The
    third fixture must have `stop` fire on a problem the ladder has **not**
    fit, so that an implementation deriving either boolean from the last
    record's misfit fails rather than passing.

    `last_evaluation` holds the final forward evaluation and is `None` exactly
    when the run made none. Its `ensemble` and `predictions` are asserted
    **equal** to the last call recorded by an instrumented model — an equality,
    not the inequality against `result.ensemble` that a pre-inflation or
    pre-repair pair would also satisfy — and the off-by-one against
    `result.ensemble` is asserted on a schedule-exhausted run and its *absence*
    asserted on a stopping-rule run, where the state is unchanged.
    `n_evaluations` equals the instrumented forward-call count.
    `result.stacked` succeeds on a multi-step run with `(T,)`-shaped fields
    including `step` and `n_valid`, and returns `(0,)`-shaped fields on an
    empty history rather than raising.
20. **Inflation sees the ladder, and is applied where the contract says.** An
    `Inflation` and an `EnsembleUpdate` that record their `step` and `beta`
    arguments observe the true sequence, and a rule varying with `beta` gives a
    run that is still exactly resumable — the point of passing the arguments
    instead of forcing a stateful
    workaround ({ref}`eki-updates`).

    Placement is asserted against an instrumented model, not assumed: with an
    inflation that adds a constant, the model's **first** recorded input equals
    the initial ensemble plus that constant, bit-exactly. That pins both the
    before-the-evaluation placement — which {ref}`eki-inflation` calls the
    alternative forbidden — and the consequence that section states and which
    was otherwise untested, that the initial ensemble is inflated before it is
    ever evaluated. `result.ensemble` is asserted to be an update output, never
    an inflation output.
21. **The record agrees with the evaluation it came from.** Over a multi-rung
    run with distinct increments, every `HistoryRecord` field is checked
    against an independently recomputed value: `step` and `beta` against the
    state entering the rung, `beta_next` against the state leaving it,
    `increment` against their difference, and `misfit_mean`, `misfit_min`,
    `misfit_max`, `centre_misfit`, `rms_parameter_spread`, `n_valid` and `ess`
    against NumPy over the model's recorded input and output. Exact where the
    quantity is exact. {ref}`eki-diagnostics` calls a record field that could
    disagree with its evaluation a defect; this is the obligation that makes
    the claim mean something, and it is the only guard against a swapped
    `misfit_min`/`misfit_max`, an `ess` computed at `beta` instead of at the
    increment, or a `beta_next` off by one rung — none of which any other test
    would notice, since every one of them stays a plausible scalar.
22. **`rms_parameter_spread` is exact against a closed form.** On a
    QR-of-ones fixture with covariance factor $c I_P$, every coordinate has
    empirical variance exactly $c^2$, so the field is exactly $c$; a second
    fixture with distinct per-coordinate variances pins the root-mean-square
    reading as $\sqrt{(a^2+b^2+c^2)/3}$. The two errors this catches — a
    divisor of $J$ rather than $J-1$, and a missing $1/\sqrt{P}$ — are
    separately distinguishable at $J = 5$, $P = 3$, and the field is otherwise
    covered by no obligation at all despite being reported in the history.
23. **A finished ladder is a no-op, and says so.** Running a completed
    `FixedSchedule` state through a fresh schedule of the same length gives
    `status="schedule_exhausted"`, `n_steps == 0`, an empty history,
    `last_evaluation is None`, a bit-identical ensemble, and **zero** forward
    calls by an instrumented model; `EKIState.restart()` then gives the full
    ladder. The adaptive counterpart is asserted too, since it exhausts on
    `beta` rather than on `step`. {ref}`eki-state` devotes a warning to this
    trap and nothing tested it.
24. **The driver hands the update what it repaired.** With a model failing one
    member of six, a recording `EnsembleUpdate` receives an `ensemble` and
    `predictions` bit-identical to `repair_failed_members` applied to the
    inflated ensemble and the raw predictions — **both** elements.
    {doc}`gaussian-contract` warns that differing masks between the two blocks
    corrupt $\widehat C_{uv}$ with no exception, so repairing one and not the
    other is a silent wrong answer that obligation 9 does not reach, since it
    exercises the helper rather than the driver.
25. **The axes compose.** A smoke matrix over every schedule, both updates,
    every inflation and both stopping-rule settings, on a tiny affine problem
    of three rungs, asserting only that each run terminates, reports a
    permitted `status`, stacks, stays finite, and matches an instrumented
    forward-call count. This is the executable form of {ref}`eki-axes`'
    orthogonality claim, which is the design's organizing premise and is
    otherwise checked at a single point of the space. It also catches a driver
    that inspects one axis to decide another — an `ess` computed only for the
    ESS schedule, say.
26. **The document's own recipes run.** Every runnable block in this page is
    executed by a test: the two-form example, the pinned prior draw,
    `restart()`, the backtracking loop, `AdditiveInflation`'s definition, the
    `stacked` one-liner, the `EKIError` checkpoint pattern, the `iterate`
    construction of an `INTERRUPTED` result, and the Tikhonov augmentation.
    The last is load-bearing: it is this contract's evidence for the
    "no new code" claim of {ref}`eki-variants`, it depends on `block_diag` and
    on the appended-block whitening identity, and its documented
    double-counting hazard is asserted as an over-concentration so that a later
    change cannot quietly "fix" it.

Alongside conformance, targeted regression tests guard this layer's
silent-failure classes under the same do-not-delete rule as the layers below.
**The list is derived from this document rather than curated**: every place the
prose says a mistake raises nothing, warns nobody, or looks normal earns an
entry, and the two must be kept in step.

The $R/\beta$ mis-scaling. The non-log-space ESS. An inflation factor applied
to the covariance rather than the anomalies (the `anomaly_factor` convention).
A repair applied when nothing failed. A repair that rescales the surviving
members, and so inflates silently. A misfit computed before repair. A schedule
that counts its own calls, and so breaks resumption. A `HistoryRecord` field
declared static, and so unstackable. The safety bound checked before ladder
exhaustion, and so raising on a completed run. The `min`/`max` inversion in
`AdaptiveMisfitSchedule` and each inversion of the clamp precedence — all three
named in the prose as easy and silent. The bisection returning `hi`. The
key-split arity or order changed, which {ref}`eki-prng` says must not shift the
update's stream, and which **no numeric test can catch** in the default
configuration, since it consumes no randomness at all: the guard is a
`jax.random.key_data` snapshot of a multi-rung `PathwiseUpdate` run. A record
field disagreeing with the evaluation it was built from, which
{ref}`eki-diagnostics` calls a defect and which every plausible-scalar bug
hides behind. Two forward evaluations per rung. Chaining a fresh ladder onto a
finished state, which {ref}`eki-state` warns returns unchanged with nothing
raised. `DiscrepancyStop` on a budgeted ladder. A fill-value forward model
stalling an adaptive ladder. A systematically failing member, visible only in
`n_valid`. The Tikhonov augmentation at $\beta = 1$ from a prior ensemble,
which {ref}`eki-variants` says looks completely normal. And a `float32`
forward model or update quietly demoting a run's precision, where every
downstream test still passes at its own tolerance.

An increment baked in as a Python constant is **not** on the list, and the
reason is worth recording: a Python float passed as a `jit` *argument* does not
retrace, so the test that would have guarded it cannot fail. What does force a
retrace per step is a static field on an object crossing a `jit` boundary,
which {ref}`eki-jax` states as a rule and which the compilation-count
obligation covers.

(eki-references)=
## References

Where a shipped policy reproduces a published method, this is the source. The
list is deliberately short: it covers what the layer *implements*, not the
ensemble-Kalman literature at large, and it is the one place in the package
that points outside the repository — the docstring rule that keeps
documentation self-contained applies to the API, and a design contract that
implemented someone else's criterion without saying so would be worse for being
tidy.

| shipped as | source |
| ---------- | ------ |
| `AdaptiveMisfitSchedule` | M. A. Iglesias and Y. Yang, *Adaptive regularisation for ensemble Kalman inversion*, Inverse Problems, 2021; arXiv:2006.14980. The data misfit controller, its Jeffreys-divergence derivation, and the statistical discrepancy principle fixing $\theta = N/2$. |
| `DiscrepancyStop` | V. A. Morozov, for the discrepancy principle itself; M. A. Iglesias, *A regularizing iterative ensemble Kalman method for PDE-constrained inverse problems*, Inverse Problems, 2016, for its use as a stopping rule in an ensemble Kalman method. |
| `AdaptiveESSSchedule` | A. Jasra, D. A. Stephens, A. Doucet and T. Tsagaris, *Inference for Lévy-driven stochastic volatility models via adaptive sequential Monte Carlo*, Scandinavian Journal of Statistics 38(1):1–22, 2010, for ESS-based adaptive tempering; A. Beskos, A. Jasra, N. Kantas and A. H. Thiéry, *On the convergence of adaptive sequential Monte Carlo methods*, Annals of Applied Probability 26(2):1111–1146, 2016, for its convergence theory; Y. Zhou, A. M. Johansen and J. A. D. Aston, *Towards automatic model comparison: an adaptive sequential Monte Carlo approach*, Journal of Computational and Graphical Statistics 25(3):701–726, 2016, for the conditional-ESS generalization pyEKI does **not** implement. |
| the tempered ladder as an algorithm | A. A. Emerick and A. C. Reynolds, *Ensemble smoother with multiple data assimilation*, Computers & Geosciences 55:3–15, 2013. The inflation factors $\alpha_n$ of that literature are this contract's reciprocal increments, and the requirement that they sum appropriately is {ref}`eki-iteration`'s telescoping. |
| `MultiplicativeInflation` | J. L. Anderson and S. L. Anderson, *A Monte Carlo implementation of the nonlinear filtering problem to produce ensemble assimilations and forecasts*, Monthly Weather Review 127:2741–2758, 1999. |
| `TransformUpdate`'s lineage | J. S. Whitaker and T. M. Hamill, *Ensemble data assimilation without perturbed observations*, Monthly Weather Review 130:1913–1924, 2002. The transform this layer uses is specified in {doc}`gaussian-contract`, not here. |
| the Langevin variant of {ref}`eki-variants` | A. Garbuno-Iñigo, F. Hoffmann, W. Li and A. M. Stuart, *Interacting Langevin diffusions: gradient structure and ensemble Kalman sampler*, SIAM Journal on Applied Dynamical Systems, 2020. |

Two notes on what is *not* claimed. The relaxation-to-prior inflation family and
the damped iterative ensemble smoothers named in {ref}`eki-excluded` and
{ref}`eki-step` are referred to by description rather than cited, because the
layer does not ship them — it only undertakes not to foreclose them. And
`AdaptiveESSSchedule`'s default target of `0.5` is pyEKI's choice, not a value
any of the above prescribes.

(eki-excluded)=
## Deliberately excluded

Their absence is a decision, not an oversight.

**A `lax.scan` driver.** The forward model is an arbitrary callable and may
not be traceable at all, so the loop cannot be a scan. Everything that *can*
be traced is, and the loop stays Python. A run whose forward model happens to
be pure JAX gains a fully-scanned variant only by giving up the per-step
Python decisions — adaptive increments, termination, failure branching — that
this contract is largely about; no consumer has asked for the trade.

**A `Problem` container** bundling `(forward, y, noise_cov)`. The triple is
passed to `run`, `iterate`, `step` and the helpers, and a container would
document its shape agreement once.

Not excluded because it could not be a pytree — `EKIResult` is already a plain
frozen dataclass on the ground that it never crosses a trace boundary, and a
`Problem` would qualify the same way. Excluded because three arguments are not
many, the container would carry no behaviour, and every call site taking the
triple would then accept either form or force a conversion. The shape agreement
it would document is validated once per run anyway ({ref}`eki-driver`). Revisit
if a fourth element joins the triple.

**A `callback` argument.** `iterate` is the extension point, and a callback
would be a second way to do the same thing. `run`'s documentation shows the
`iterate` loop for checkpointing and logging.

**A `jit=` flag.** `jax.disable_jit()` already exists and is global, which is
what a debugging switch should be.

**Trial evaluations *as a policy interface*** — a `Schedule` that may propose
an increment, see it evaluated, and revise it. Such an interface would let a
schedule spend forward-model evaluations, which is the one cost the layer is
organized around, and it would make every schedule's cost unpredictable from
its type. The shipped adaptive schedules are deliberately the kind that need no
trial evaluation at all ({ref}`eki-schedules`).

The *algorithms* that want trial evaluations — line search, backtracking on a
rejected increment, damped Gauss–Newton-style iterative ensemble smoothers —
are not excluded. They are written against the public `step`, which is exactly
the propose-evaluate-accept-or-retry loop and is shown as one
({ref}`eki-step`). The distinction is between the driver spending evaluations
on a caller's behalf, which it never does, and a caller spending them
deliberately, which it may.

**Changing the data between rungs.** `run` and `iterate` bind
`(forward, y, noise_cov)` once, and a generator cannot be handed a new
observation, so subsampled, mini-batch and randomized-observation variants —
which draw a fresh subset of the data at every rung — are not expressible
against the driver. They are expressible against `step`, which takes the triple
per call, so the variant is a five-line loop rather than a missing feature
({ref}`eki-step`). The driver keeps the binding because a fixed $N$ is what
lets it validate shapes once and lets a stopping rule mean the same thing at
every rung. Noted because the three-axis table of {ref}`eki-axes` names schedule, update
and inflation, and never the data.

**Importance weights and resampling.** The ESS appears as a step-size
heuristic and a diagnostic only. Carrying weights, resampling, and the
correctness claims that come with them would make this a sequential Monte
Carlo library; EKI's appeal is precisely that it needs neither.

**Parameter transformations and constraints.** Bounds, positivity and
sparsity-promoting reparameterizations belong to the caller, who works in
unconstrained coordinates and composes the transformation into the forward
model and the prior. A projection applied after each update would be a
different algorithm, expressible as a custom update rule that wraps a shipped
one.

**Covariance localization.** {doc}`design` records why: the Hadamard taper
destroys the low-rank factorization the conditioning kernel depends on.
Domain localization is the plan, as an update rule.

**Mean-field and unscented variants.** Their members are deterministic
quadrature nodes regenerated from a mean and a covariance each step, so the
state they need is $(m, C)$ rather than an ensemble. The distinguishing feature
is that the members do not move freely and are not carried between steps — not
the value of $J$, which such a scheme also has.

**Multilevel and multi-fidelity variants.** They carry several coupled
ensembles at different forward-model resolutions and estimate moments by a
telescoping sum across them. `EKIState` holds one ensemble, `run` takes one
`forward`, and the update protocol sees one $(J, P)$ array, so this is a
different driver rather than a policy. Adjacent to the fixed-$J$ decision
below.

**The moment-exact repair of failed members.** {ref}`eki-failures` gives the
construction, its arithmetic, and why the shipped repair damps instead.

**Stopping on parameter stagnation.** A rule comparing consecutive ensembles
would need history, which would make stopping rules stateful and runs
unresumable. Write it as an `iterate` loop.

**Multiple runs in one call.** A family of runs — several observations,
several priors, several seeds — is a Python loop over `run`, not a `vmap`:
the driver cannot be traced. Vectorizing across runs would require a
traceable forward model and a scanned driver, which is the excluded item
above.

**Composing inflations, adaptive inflation, and the relaxation-to-prior
family.** Two inflations compose in a three-line callable. An inflation that
adapts to the misfits would need the evaluation, enlarging the protocol for no
current consumer. The relaxation schemes — blending the posterior ensemble's
anomalies or spread back toward the prior step's — need *both* the pre-update
and post-update ensembles, so they are not `Inflation`s at all under this
layer's signature; they are post-update transformations, and adding them would
mean a fifth protocol. All three are writable as a custom `EnsembleUpdate` wrapping a shipped one,
which does see both ensembles.

**The misfit at the mean parameter, $\Phi(G(\bar u))$.** An extra forward-model
evaluation per step for a diagnostic, a $1/J$ increase in the run's dominant
cost. The caller's to compute in an `iterate` loop; {ref}`eki-diagnostics`
distinguishes it from the two the evaluation carries.

**Conditioning diagnostics in the history.** The singular values of the scaled
whitened anomaly matrix, and the marginal-likelihood term
$\sum_i\log(1+\sigma_i^2)$ built from the same decomposition, would cost the
driver a second $O(NJ^2)$ decomposition — the update's own order, for a
diagnostic — because the update's SVD is internal to `pyeki.gauss`.
{ref}`eki-diagnostics` gives the recovery route. Revisit if `pyeki.gauss` ever
grows a decomposition accessor, which it records as awaiting a consumer.

**Wall-clock and profiling instrumentation.** The record holds the algorithmic
diagnostics, and the driver emits one `logging` record per step
({ref}`eki-driver`). Timings, profiles and progress bars belong to the caller,
around an `iterate` loop.

**Checkpointing to disk.** `EKIState` is a pytree of arrays and a small
static; serializing it is the caller's choice of format. The layer's
obligation is that resumption from a deserialized state is exact, which is
{ref}`eki-conformance`'s test 11.

**An adaptive ensemble size.** $J$ is fixed for a run, because every
downstream shape depends on it and a data-dependent $J$ would make the
analysis untraceable. Failed members are handled by repair, not by shrinking
the ensemble ({ref}`eki-failures`).
