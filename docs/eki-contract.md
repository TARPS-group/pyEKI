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

The layer's arithmetic is deliberately shallow, and the boundary is worth
stating precisely rather than loosely: **this layer forms no covariance and
factorizes nothing.** Every operation on a covariance — whitening, factoring,
solving, the conditioning kernel itself — happens in `pyeki.linalg` or
`pyeki.gauss`. What does happen here is elementwise: whitened residuals and
their norms, log-space weight ratios, mean-and-subtract on an ensemble, and
centred draws from an operator's factor. A step of this layer that assembled a
covariance, or that re-derived any part of the conditioning kernel, would be a
layering violation ({ref}`eki-updates`).

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
  batched exception is `whitened_misfits`, which follows the operator layer's
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
*grows with ladder length*: on a linear-Gaussian problem with a five-step
uniform ladder, the increment form reproduces the one-shot posterior to
$10^{-15}$ while the level form is off by $0.12$ in the mean and $0.25$ in
the covariance. Write the telescoping test first.
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
| how spread is maintained | `Inflation` | what happens to the ensemble between steps? |

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
assert fit.converged                 # False means the ladder ran out first
```

Neither result is named `posterior`. `sampled.ensemble` is an approximate
posterior ensemble under the caveats of {ref}`eki-honesty`, and `fit.ensemble`
is a collapsing ensemble around a regularized fit, which is not a posterior at
all; naming either one `posterior` in a script is the first step towards
reporting the second as though it were the first.

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
| `StepSummary` | pytree | what a schedule or stopping rule sees about the current step |
| `StepRecord` | pytree | one row of the run's history: scalars only |
| `EKIResult` | frozen dataclass | the final state, the history, and why the run ended |
| `EnsembleUpdate` | protocol | one ensemble update, given an increment |
| `Schedule` | protocol | the increment, and whether the ladder is finished |
| `StoppingRule` | protocol | whether to stop, given the current misfits |
| `Inflation` | protocol | a transformation of the ensemble between steps |
| `run`, `iterate`, `step` | functions | the driver, as a function, as a generator, and as one iteration |
| `whitened_misfits`, `effective_sample_size`, `repair_failed_members` | functions | the array-level pieces schedules and custom drivers need |

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

**Fields.** `ensemble`, a `(J, P)` array; `beta`, a 0-d array; `key`, a typed
PRNG key of shape `()`; and `step`, a static `int`. Construction validates
exact ranks, $J \ge 2$ and $P \ge 1$, `step >= 0`, and that `key` is a typed
key (tier 2, shape-only). A Python float or 0-d array is accepted for `beta`
and converted to a 0-d float array before storing, as the operator layer's
scalar dunders do; `beta` must not be negative.

**Derived attributes.** `n_members` and `u_dim` are `int` properties, named as
in {doc}`gaussian-contract`.

**`EKIState.from_prior(key, prior, n_members)`** is the classmethod that
computes, per the operator layer's constructors-store rule. `prior` is a
{class}`~pyeki.gauss.Gaussian`; the draw is pinned as

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
finds `exhausted(10, ...)` already true, and `run` returns immediately with
`status="schedule_exhausted"`, an empty history, and the ensemble unchanged.
Nothing raises, because an already-finished ladder legitimately returns.

A new ladder therefore needs a new counter:

```python
phase2 = dataclasses.replace(state, step=0)
```

Adaptive *budgeted* schedules are immune, since their exhaustion reads
`beta` rather than `step` — but for the same reason they need `beta` reset if
a second budget is intended. `EKIResult.n_steps` counts the records of that
run, not `state.step`; after a resumption the two differ, and the histories
concatenate in call order.
:::

(eki-step)=
## The step

One iteration, specified once and used three times: `iterate` is a loop over
it, `run` is a loop over `iterate`, and it is public in its own right.

### `step(state, forward, y, noise_cov, *, increment, update=SquareRootUpdate(), inflation=None, on_failure="repair")`

Carries `state` forward by one rung at the **given** increment, returning
`(EKIState, StepRecord)`. It chooses nothing: no schedule, no stopping rule, no
`max_steps`. Those are the driver's, and everything they decide is passed in.

Making the single iteration public is what keeps {ref}`eki-driver`'s claim —
that `iterate` is the extension point — true for the algorithms that need to
*revisit* a step rather than merely observe it. A generator can be abandoned
but not re-entered with a revised decision, so the whole damped/backtracking
family (propose a step, evaluate it, accept it or shrink the damping and retry)
is unwritable against `iterate` alone. Against `step` it is an ordinary loop,
because states are immutable and re-stepping from the same one is exactly
"reject and retry":

```python
s, delta = state, 1.0
while not done(s):
    trial, record = step(s, forward, y, noise_cov, increment=delta)
    _, probe = step(trial, forward, y, noise_cov, increment=delta)
    if probe.misfit_mean < record.misfit_mean:
        s, delta = trial, delta * 1.5      # accept, lengthen
    else:
        delta = delta / 2                  # reject, retry from the same state
```

One evaluation per rejection is wasted, which is what backtracking costs in any
implementation; the layer neither hides that nor pays it for callers who do not
want it. `step` is also the entry point for anything that varies the *data*
between rungs — a subsampled or randomized observation vector — which the
driver deliberately fixes for a whole run ({ref}`eki-excluded`).

**The order of operations is normative.**

1. **Split the key**, always into three, whatever the policies are:

   ```python
   key_next, key_inflate, key_update = jax.random.split(state.key, 3)
   ```

   Fixed arity means turning inflation on or off does not shift the update's
   random stream ({ref}`eki-prng`).
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
5. **Summarize.** Build the `StepSummary` from the repaired ensemble and
   predictions ({ref}`eki-diagnostics`). `step` returns the summary's contents
   through the record; the driver consults the summary itself.
6. **Validate the increment.** It must be a scalar, finite, and **strictly
   positive**, converted to a 0-d float array; otherwise `ValueError`. A zero
   increment is not a harmless no-op — $R/0$ is infinite and the update
   returns `nan`.
7. **Update.** `u_new = update(key_update, ensemble=u, predictions=v, y=y,
   noise_cov=noise_cov, increment=dbeta, step=state.step, beta=state.beta)`,
   validated to be `(J, P)`.
8. **Finiteness.** If `u_new` contains a non-finite entry, raise `EKIError`
   naming the step and the level. Silent `nan` propagation through a long run
   is the worst outcome available to this layer, and the check is free given
   that step 4 already synchronized.
9. **Advance.** Return
   `EKIState(u_new, state.beta + dbeta, state.step + 1, key_next)` and the
   step's `StepRecord`.

Steps 4, 6 and 8 read concrete values, so a step **synchronizes with the device
once**. This is a deliberate cost: $O(1)$ scalars and one reduction against $J$
forward-model evaluations, and it is what allows termination, validation and
adaptive increments to be ordinary Python.

### What the driver adds

`iterate` wraps `step` with the decisions `step` refuses to make, in this
order, before each call:

1. **Ladder exhaustion.** If `schedule.exhausted(state.step, state.beta)`, end
   the run with status `"schedule_exhausted"`. This is checked *before* the
   forward model is evaluated, which is why `Schedule` splits exhaustion from
   the increment: a fixed ladder of $T$ rungs must cost exactly $T$ ensemble
   evaluations, not $T + 1$.
2. **Safety bound.** If `state.step >= max_steps`, raise `EKIError`. The
   message must name `max_steps`, the schedule, and whether a stopping rule
   was supplied, since an unbounded schedule with no stopping rule is the
   usual cause.

   The order of these two is normative and the inversion is a bug: a schedule
   that exhausts at step $T$ **must complete under `max_steps == T`**, which is
   the value a caller naturally passes. Checking the bound first would turn
   every such run into an `EKIError` on its final re-entry, blaming the
   schedule for a completed ladder.
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
4. **Stopping rule.** If `stop is not None and stop(summary)`, end the run
   with status `"stopping_rule"`, emitting a terminal record
   ({ref}`eki-diagnostics`) and leaving the state unchanged.
5. **Increment.** `dbeta = schedule.increment(summary)`. A schedule may return
   `None` to declare the ladder finished on evidence only the summary carries
   ({ref}`eki-schedules`), which ends the run with status
   `"schedule_exhausted"` and emits the same terminal record as a stopping
   rule; any other value is validated and used as `step`'s increment.

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
| `SquareRootUpdate()` | `EnsembleJoint(u, v).transform_update(y, noise_cov / increment)` | deterministic square-root transform; ignores the key; **the default** |
| `StochasticUpdate()` | `EnsembleJoint(u, v).pathwise_update(key, y, noise_cov / increment)` | perturbed-observation; consumes the key |

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
form most of the ensemble-Kalman-inversion literature is written in.

**`SquareRootUpdate` is the default**, and the reason is the layer's own
claims. Telescoping holds exactly under it and only in expectation under the
stochastic update, so the exactness statement of {ref}`eki-honesty` — the one
property this layer proves about itself — is a property of the default
configuration rather than of a configuration a caller has to know to ask for.
It is also deterministic, so a user's first two runs agree, and a difference
between them is a bug rather than a seed. A caller who wants the classical
perturbed-observation form passes `update=StochasticUpdate()`, which is one
keyword and is what the sampling-form literature describes.

(eki-schedules)=
## Schedules

A `Schedule` has two methods.

| method | signature | contract |
| ------ | --------- | -------- |
| `exhausted` | `(step: int, beta: Array) -> bool` | is the ladder finished, on bookkeeping alone? Decided before any forward evaluation. Must return a Python `bool`. |
| `increment` | `(summary: StepSummary) -> Array \| float \| None` | the next increment, or `None` for "finished after all". Called only when `exhausted` is false. A returned value must be scalar, finite and strictly positive. |

The split is the load-bearing part of the interface. `exhausted` is cheap and
misfit-independent, so a run whose ladder is finished pays nothing; `increment`
may look at the current misfits, so adaptivity costs nothing extra either.
Both must be pure: a schedule that counted its own calls could not be resumed
from a checkpoint, and `step` is passed precisely so that it need not.

**Why `increment` may also end the ladder.** `exhausted` sees only `(step,
beta)`, so a schedule whose finishing condition depends on the *misfits* has no
way to express it there — an adaptive ladder that should give up when its
target has become unattainable above the floor, for instance. Without a second
exit such a schedule must either crawl at its floor until `max_steps` raises,
spending the whole evaluation budget to report a failure, or the caller must
restate the schedule's own configuration and criterion inside a separate
`StoppingRule` and keep the two objects consistent by hand. Returning `None` is
the smaller mechanism: it ends the run with `status="schedule_exhausted"`, and
the evaluation that produced the summary is recorded as the terminal record,
exactly as for a stopping rule.

The two exits mean different things and both are needed. `exhausted` is "I can
tell from the bookkeeping that there is nothing left to do", and costs no
evaluation; `None` is "having seen this ensemble, there is nothing useful left
to do", and costs the evaluation that produced the summary. Neither shipped
schedule returns `None`; both reach their budgets through `exhausted`.

**No schedule performs a trial update or a trial forward evaluation.** Both
shipped criteria read nothing but the misfit vector $(\Phi_1,\dots,\Phi_J)$ and
the current level, so adaptivity costs $O(J)$ per step on top of a whitening
the driver pays anyway. Two consequences: the increment-rescaling identity that
{doc}`gaussian-contract` records for candidate steps
($S(R/\delta) = \sqrt{\delta}\,S(R)$) is never needed here, because no
candidate is ever evaluated; and choosing an increment can never cost a
model evaluation, which is the resource the whole layer is organized around.

A custom schedule is not confined to the misfits, though — the summary carries
the whole whitened residual matrix, and hence the whitened prediction anomalies
({ref}`eki-diagnostics`), which is what a step-size rule of the
Langevin family needs. What no schedule can do is evaluate the forward model
again; that restriction is the protocol's, and it is deliberate
({ref}`eki-excluded`).

### `FixedSchedule(increments)`

A ladder given in advance. `increments` is a non-empty tuple of Python floats,
static metadata, each strictly positive and finite (`ValueError` otherwise).
`exhausted(step, beta)` is `step >= len(increments)`; `increment(summary)`
returns `increments[summary.step]` and ignores everything else.

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
| `beta_final` | `1.0` | the temperature budget, or `None` for an unbounded ladder |
| `min_increment` | `1e-3` | a floor guaranteeing progress |
| `max_increment` | `1.0` | a ceiling |

All three are static metadata. `beta_final`, when given, must be strictly
positive; the floor must be positive and no greater than the ceiling
(`ValueError` otherwise).

**Exhaustion.** `exhausted(step, beta)` is
`beta_final is not None and beta >= beta_final - budget_tol`, with
`budget_tol = 1e-12 * max(1.0, beta_final)`. It is always false when
`beta_final is None`, so an unbounded ladder must be ended by a stopping rule;
a run with neither is a `max_steps` `EKIError`, and that error's message must
say so ({ref}`eki-driver`).

**Clamping, and its precedence.** Each schedule computes an unclamped
criterion value $\delta^\star$ and returns

$$
\delta \;=\; \min\Bigl(\,
\max\bigl(\delta^\star,\ \delta_{\min}\bigr),\ \
\delta_{\max},\ \
\beta_{\text{final}} - \beta
\,\Bigr).
$$

The order is normative, and both inversions of it are bugs. The floor beats
the criterion, so a step is always taken even where the criterion would demand
an arbitrarily small one — without which an adaptive ladder can stall short of
its budget for an unbounded number of steps. The **budget cap beats the
floor**, so the ladder cannot overshoot $\beta_{\text{final}}$ — without which
a sampling run silently conditions on more data than it has. Positivity of the
result follows from `exhausted` having returned false, which guarantees a
remaining budget strictly above `budget_tol`.

With a floor $\delta_{\min}$ and a budget, a run reaches $\beta_{\text{final}}$
in at most $\lceil \beta_{\text{final}} / \delta_{\min}\rceil$ steps.

**The shipped defaults are chosen so that this bound is reachable, and the
arithmetic is part of the contract.** With `beta_final=1.0` and
`min_increment=1e-3` the worst case is $\lceil 1/10^{-3}\rceil = 1000$ steps,
which is exactly the driver's default `max_steps=1000`
({ref}`eki-driver`). A budgeted adaptive schedule at the defaults therefore
**cannot** raise on the safety bound: the floor-bound ladder finishes on its
last permitted step. A caller who lowers `min_increment` must raise
`max_steps` correspondingly, and the implementation must keep the two defaults
consistent if either changes — a floor of $10^{-4}$ against a bound of 1000
would guarantee an `EKIError` on precisely the badly-conditioned problems the
floor exists to rescue, after spending the entire evaluation budget.

**The degenerate ensemble.** When every member has the same misfit — a
collapsed ensemble, or a forward model insensitive to the current spread — no
increment changes the target's shape relative to the ensemble, and both
schedules must take the **largest allowed step**, that is
$\min(\delta_{\max}, \beta_{\text{final}} - \beta)$. Each criterion below
reaches that conclusion on its own; the requirement is recorded here so that an
implementation cannot satisfy one schedule's version of it and not the other's.

### `AdaptiveESSSchedule`

Measures the move by the effective sample size of the importance weights that
would carry the ensemble from one target to the next. The construction is
standard in adaptive tempering, and is used here purely as a **step-size
heuristic**: pyEKI computes no importance weights, does no resampling, and
makes no importance-sampling correctness claim. Its extra field is `target`,
the ESS level sought as a fraction of $J$, in $(0, 1)$, default `0.5`; and
`n_bisect`, a static `int` bisection count, default `50`.

For an increment $\delta$, the weights and their effective sample size are

$$
w_j(\delta) = e^{-\delta \Phi_j},
\qquad
\mathrm{ESS}(\delta)
= \frac{\bigl(\sum_j w_j\bigr)^2}{\sum_j w_j^2}
\;\in\; [1, J] .
$$

$\mathrm{ESS}$ is $J$ at $\delta = 0$ and **monotone non-increasing** in
$\delta$, so the target level is reached by bisection. The monotonicity is a
fact rather than a hope, and the whole construction rests on it, so the
derivation is recorded here. Write $E_\lambda$ for expectation under the
tilted weights $w_j(\lambda) \propto e^{-\lambda\Phi_j}$. Then

$$
\frac{d}{d\delta}\log \mathrm{ESS}(\delta)
= 2\bigl(E_{2\delta}[\Phi] - E_{\delta}[\Phi]\bigr)
= -\,2\delta \int_{1}^{2} \operatorname{Var}_{s\delta}(\Phi)\,ds
\;\le\; 0 ,
$$

the second equality by integrating
$\frac{d}{d\lambda}E_\lambda[\Phi] = -\operatorname{Var}_\lambda(\Phi)$ from
$\lambda = \delta$ to $\lambda = 2\delta$. The inequality is strict for
$\delta > 0$ unless every $\Phi_j$ is equal, which is the degenerate case
{ref}`eki-adaptive` covers.

Two consequences of that identity are load-bearing for the implementation.
The rate of decay is a variance of the misfits under the tilted weights, so
the criterion is genuinely measuring ensemble disagreement about the data.
And **the derivative vanishes at $\delta = 0$**: the function is flat at the
left endpoint, so a derivative-based root find started there stalls. Bisection
is not merely convenient, it is the right method.

One precondition, which pyEKI satisfies structurally: the argument assumes the
ensemble arrives **equally weighted**. It does — this layer carries no
importance weights at all ({ref}`eki-excluded`) — but a variant that
introduced them would have to revisit the monotonicity before reusing this
bisection.

The criterion is

$$
\delta^\star \;=\; \sup\{\delta \,:\, \mathrm{ESS}(\delta) \ge \texttt{target}\cdot J\},
$$

clamped as {ref}`eki-adaptive` specifies. Three implementation requirements:

- **Bisect on a bracket, and return the safe end.** Take
  $\delta_{\mathrm{hi}} = \min(\delta_{\max},\, \beta_{\text{final}} -
  \beta)$; if $\mathrm{ESS}(\delta_{\mathrm{hi}})$ already meets the target,
  return it — the cap binds and no search is needed. Otherwise bisect
  $[0, \delta_{\mathrm{hi}}]$ for exactly `n_bisect` iterations, maintaining
  $\mathrm{ESS}(\text{lo}) \ge \text{target}\cdot J >
  \mathrm{ESS}(\text{hi})$, and return `lo`. The guarantee is therefore
  one-sided: the returned increment meets the target before the floor is
  applied. A fixed iteration count keeps the computation
  `jit`-compatible with no data-dependent control flow.
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

`target` defaults to `0.5`, which is pyEKI's choice rather than a canonical
value; the tempering literature uses targets between about a third and a half.
A smaller target takes longer steps and fewer of them.

### `AdaptiveMisfitSchedule`

Measures the move against the **noise level** rather than against the
ensemble, and solves for the increment in closed form instead of bisecting.
Its one extra field is `theta`, a static override for the benchmark scale,
default `None` meaning $N/2$ — and with that default the schedule has **no
tuning parameter at all**, which is its main attraction.

The criterion. Write $\chi_j = 2\,\delta\,\Phi_j$ for member $j$'s whitened
misfit measured at *this step's own* noise level $R/\delta$, rather than at
the base level. If the step's target were well specified and the ensemble were
distributed according to it, each $\chi_j$ would be a $\chi^2_N$ variate, whose
mean is $N$ and whose variance is $2N$. The schedule asks the increment to
respect those two benchmarks:

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
$\beta_{\text{final}} - \beta$ is what makes a run terminate exactly at
$\beta_{\text{final}}$ with the increments summing to it.

**The `max` is deliberate and is not a `min`.** The two lines are separate
sufficient benchmarks, not joint requirements, so taking the larger admits the
longest step meeting at least one of them. Which one binds is decided by the
misfits' coefficient of variation: the mean bound is the larger exactly when
$\sigma_\Phi/\overline{\Phi} > \sqrt{2/N}$, so for any appreciable $N$ the
mean bound normally sets the step and the variance bound takes over only for
an unusually tightly clustered ensemble — where it correctly permits a longer
step than the mean alone would. A `min` would let whichever bound is
momentarily pessimistic stall the ladder; a caller who wants that conservative
reading writes a four-line schedule of their own.

**How this differs from the ESS criterion — and why both ship.** The two are
not variants of one idea, and they do not agree. Under the small-increment
expansion of {ref}`eki-adaptive`'s companion section,
$\mathrm{ESS}(\delta) \approx J/(1 + \delta^2\sigma^2_\Phi)$, so the variance
benchmark $\delta^2\sigma^2_\Phi = N/2$ sits at a relative ESS of about
$2/(N+2)$ — for a hundred observations, an ESS of roughly two members. The
misfit schedule therefore takes **far longer steps** than an ESS target near
$1/2$ would allow. That is not a defect in either: they control different
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
`jnp.where(d > 0, theta / d, jnp.inf)`; the clamps of {ref}`eki-adaptive` then
deliver the largest allowed step, as that section requires. Dividing unguarded
happens to give `inf` for positive `theta`, but relies on the sign of a zero
and gives `nan` at `theta = 0`, so the guard is contractual rather than
stylistic.

(eki-stopping)=
## Stopping rules

A `StoppingRule` is a callable `stop(summary) -> bool`, returning a Python
`bool`, pure, and — like schedules — forbidden from holding iteration state.
It is consulted before the increment is chosen, so a run that already fits the
data takes no further update.

A misfit-based stopping rule necessarily costs one forward evaluation whose
update is discarded: the misfits that trigger the stop are the misfits of an
ensemble that is then returned unchanged. That evaluation is visible in the
history as the terminal record ({ref}`eki-diagnostics`), and it is inherent,
not an implementation artifact.

**`DiscrepancyStop(tau=1.0)`** implements the discrepancy principle: stop as
soon as the ensemble centre fits the data to within the noise level,

$$
2\,\Phi(\bar v) \;\le\; \tau^2 N ,
$$

where $\bar v = \frac1J\sum_j v_j$ is the **mean prediction** and
$\Phi(\bar v)$ is the summary's `centre_misfit`.

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

That placement deserves a word, because both "before" and "after" the analysis
appear in the literature. **In the interior of a run they coincide**: inflating
the analysis ensemble at the end of step $t$ and inflating it at the start of
step $t+1$ produce identical sequences, since nothing happens in between. They
differ at the two ends, and the difference is why the start is chosen —
placing it at the start means the ensemble a run *returns* is a clean posterior
ensemble rather than an inflated one, and that predictions always match the
members they update.

The cost of that choice, stated so it is not discovered later: the **initial**
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

**`MultiplicativeInflation`** scales the anomalies, so the empirical covariance
is multiplied by $r^2$, the field **squared**. The field is therefore named
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
are **centred**, so the ensemble mean is preserved exactly and the empirical
covariance is inflated by `cov` in expectation under the $J-1$ divisor. A
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
Diagnostics always record `n_valid`, so failures are never silent even under
`"repair"`.

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
- failed members become $(\hat u, \hat v)$, so their residual is $y - \hat v$
  and their update lands at the posterior mean; they rejoin the ensemble
  rather than being lost.

**The covariance is damped, and that is the intended trade.** The step
conditions with $\widehat C_{uv}$ and $\widehat C_{vv}$ both multiplied by
$c = (J_v-1)/(J-1)$, giving the gain $c\,\widehat C_{uv}(c\,\widehat C_{vv} +
R)^{-1}$ — neither $K$ nor $cK$, but a mildly *shortened* step, equivalent in
effect to a slightly smaller increment. The bias is towards moving less, which
is the safe direction, it is bounded by the failure fraction, and it does not
accumulate.

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

Quantitatively that is not a rounding concern. At $J = 100$ with $10\%$ of
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
$\Phi(\hat v)$ — the misfit of the valid centre. The bias is therefore
towards the average, which makes the reported misfit spread and the ESS
slightly optimistic when members fail; `n_valid` is in the summary so that a
criterion may account for it.

(eki-diagnostics)=
## Diagnostics

Two classes, split by lifetime rather than by subject. The **summary** is
transient: it exists for the duration of one step, carries the full whitened
residual matrix, and is what a schedule and a stopping rule are shown. The
**record** is kept for the whole run: it carries scalars only, so a history of
hundreds of steps costs nothing, and it is what a caller reads afterwards.

The driver builds the record from the summary and the chosen increment, so the
summary is the single source of truth for everything both describe — a record
field that could disagree with the summary it came from would be a defect. The
one field the record adds from outside is the increment, which is not known
until the schedule has seen the summary.

### `StepSummary`

What a schedule's `increment` and a stopping rule see.

**Fields.**

| field | shape | meaning |
| ----- | ----- | ------- |
| `step` | static `int` | the index of this step |
| `beta` | 0-d | the level *entering* the step |
| `whitened_residuals` | `(J, N)` | row $j$ is $W(y - v_j)$, after repair |
| `spread` | 0-d | $\lVert A_u \rVert_F / \sqrt{(J-1)P}$, the root-mean-square per-coordinate ensemble standard deviation |
| `n_valid` | static `int` | how many members' predictions were finite |

**The single array field is deliberate, and it is the layer's choice of
sufficient statistic.** Write $b_j = W(y - v_j)$ for the rows. Then

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
matrix $S$ that {doc}`gaussian-contract`'s kernel is built on. Carrying the
matrix rather than the misfit vector costs nothing: the driver must whiten the
residuals anyway in order to compute the misfits at all, so this is the array
it already has. What it buys is that a step-size rule needing the anomaly
structure — a Langevin-type scheme sets its step from the $J \times J$ matrix
$\langle W(v_k - \bar v),\, b_j\rangle$ — is expressible as a `Schedule`
instead of being locked out of the protocol. $N$ is likewise recoverable, from
the trailing axis, so a criterion may be calibrated to the observation
dimension without the schedule storing it.

:::{note}
**The whitening is computed twice per step, and the layer accepts that.** The
driver whitens $y - v_j$ against the base $R$ to build this array, and the
update then whitens $A_v$ against $R/\Delta\beta$ inside
{class}`~pyeki.gauss.EnsembleJoint`. The two are the same computation up to a
factor: centring the rows of `whitened_residuals` gives $-A_v W^\top$ exactly,
and the tempered whitener is $\sqrt{\Delta\beta}\,W$.

For structured whiteners applying in $O(N)$ per vector this is absorbed by the
update's own $O(NJ^2)$. For a **dense** whitener it is not:
{doc}`gaussian-contract` notes that the $O(JN^2)$ whitening dominates an update
there, so this layer doubles the dominant cost of a step on exactly the
worst-conditioned problems. The duplication buys a strict separation —
diagnostics are computed here, the conditioning kernel stays sealed in
`pyeki.gauss`, and no intermediate crosses between them — and removing it would
mean either threading a precomputed whitened matrix through the update's
signature or hoisting the conditioning primitives into this layer's driver.
Neither is worth a constant factor today; if a dense-whitener consumer makes it
worth one, the fix is the first of the two, and it is a contract change rather
than an optimization.
:::

**Derived properties**, computed on access, not cached:

| property | value |
| -------- | ----- |
| `misfits` | `(J,)`, $\Phi_j = \tfrac12\lVert b_j\rVert^2$ |
| `centre_misfit` | 0-d, $\Phi(\bar v) = \tfrac12\lVert \bar b\rVert^2$ |
| `n_members`, `n_obs` | `int` |

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

### `StepRecord`

One row of the history, built by the driver from the summary and the chosen
increment. **Every field is a 0-d array**, `step` and `n_valid` included (as
0-d integer arrays): `step`, `n_valid`, `beta`, `increment`, `beta_next`,
`misfit_mean`, `misfit_min`, `misfit_max`, `centre_misfit`, `spread`, and
`ess`.

**No field of `StepRecord` is static metadata, and that is a requirement
rather than an oversight.** Static fields live in a pytree's treedef, so two
records with different `step` values would have different treedefs, and
`jax.tree.map` across a history would raise instead of stacking. The history is
the one collection in the package meant to be stacked, so its element type must
be homogeneous as a pytree. `StepSummary.step` stays a static `int` for the
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
  keeps them themselves, from `summary.whitened_residuals`.

**Conditioning diagnostics are the caller's to compute, and they are
reachable.** The most-watched number in a run is the leading singular value of
the scaled whitened anomaly matrix $S$ — whether the gain is saturating and the
ensemble collapsing — and no field here reports it. Two reasons: the update
computes its own SVD internally and has no channel to return one, and having
the driver compute a second SVD would add an $O(NJ^2)$ cost per step, the same
order as the update itself, for a diagnostic. It is not lost, because
`summary.whitened_residuals` **determines $S$ completely** — centring its rows
gives $-A_v W^\top$, as the summary's own section shows — so a caller who wants
the spectrum takes one SVD in an `iterate` loop and pays for it deliberately. The related identity
$\log\det(\widehat C_{vv} + R) = \log\det R + \sum_i \log(1+\sigma_i^2)$,
which {doc}`gaussian-contract` records but does not expose, is available the
same way.

**The terminal record.** When a stopping rule ends a run, the forward
evaluation that triggered it is recorded as a final `StepRecord` with
`increment` exactly `0.0`, `beta_next == beta`, and `ess == J`. It appears at
most once, always last, and only for a stopping-rule termination — a
schedule-exhausted run performs no such evaluation. A zero increment in a
record therefore means "evaluated, then stopped"; this is the one zero
increment the layer permits, and it is written by the driver, never returned
by a schedule.

**Stacking the history** is one line, since records are homogeneous pytrees:

```python
h = jax.tree.map(lambda *xs: jnp.stack(xs), *result.history)
h.misfit_mean  # (T,)
```

The result is a `StepRecord` whose `batch_shape` is `(T,)` — a family in the
sense of {ref}`contract-families`, inert to methods and legible in its repr,
which is exactly what a stacked history should be.

The one case to guard is the **empty history**: a run that ends before its
first update — an already-exhausted ladder, or a stopping rule that fires at
step 0 — has `history == ()`, and `jax.tree.map` with no trees raises. Callers
stacking a history must check `result.n_steps` first. The layer ships no
tabular or plotting machinery.

(eki-driver)=
## The driver

Three entry points over one loop: `step` is one iteration ({ref}`eki-step`),
`iterate` is the loop, and `run` collects its output.

### `iterate(state, forward, y, noise_cov, *, schedule, update=SquareRootUpdate(), inflation=None, stop=None, on_failure="repair", max_steps=1000)`

A generator yielding `(EKIState, StepRecord)` after each iteration, including
the terminal evaluation-only iteration. It is the extension point for anything
that needs to *observe* or *interrupt* a run: per-step checkpointing, custom
logging, a wall-clock budget, stopping on parameter stagnation, an early
`break`. Exceptions propagate; abandoning the generator is safe. Anything that
needs to *revisit* a rung — backtracking, damping, trial increments — uses
`step` directly instead ({ref}`eki-step`).

### `run(...)`

The same arguments, consuming `iterate` and returning an `EKIResult`. It is
sugar, and it is the interface the user guide leads with.

**`EKIResult`** is a plain frozen dataclass — the one value class in the layer
that is not a pytree, because it is a report and never an argument to traced
code. Fields:

| field | contents |
| ----- | -------- |
| `state` | the final `EKIState` |
| `history` | a tuple of `StepRecord`, one per ensemble evaluation |
| `status` | why the run ended (below) |
| `last_evaluated` | the `(ensemble, predictions)` pair of the final forward evaluation, or `None` if the run made none |

and properties `ensemble` (`state.ensemble`), `beta` (`state.beta`), `mean`
(the ensemble mean, `(P,)`), `n_steps` (`len(history)`), `n_evaluations`
(`n_steps * J`, exact because every record corresponds to exactly one ensemble
evaluation), and `converged` (below).

**`last_evaluated` exists because the returned ensemble has never been
evaluated.** On a `"schedule_exhausted"` run the last update produces
`state.ensemble` and the loop then ends, so `state.ensemble` is one update
*past* the final forward evaluation and has no predictions at all; the last
record's misfits describe the ensemble *before* that update. Without this
field, a user's first two questions — what is my final misfit, and what does
the posterior predictive look like — would each cost another $J$ forward
evaluations for data the run already bought and threw away. The pair is
labelled rather than merged into the result's other fields precisely so that
which ensemble it belongs to cannot be misread:
`result.last_evaluated[0]` is *not* `result.ensemble`.

Moments beyond the mean are one line through the layer below —
`EnsembleJoint(result.ensemble, preds).condition(...)` gives the structured
posterior, and `jnp.cov` gives the raw one — so the result carries `mean` for
convenience and stops there.

`status` is one of exactly three strings, exported as module-level constants so
that a comparison cannot be misspelled:

| status | meaning |
| ------ | ------- |
| `"schedule_exhausted"` | the ladder finished, by `exhausted` or by an `increment` returning `None` |
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

**`converged`** is `True` exactly when a stopping rule was supplied and it
fired — that is, `stop is not None and status == "stopping_rule"`. It is a
property rather than a fourth status because "did this run achieve what it was
for" is a question about intent, which `status` deliberately does not model.
It answers the optimization form's version of the honesty problem: a run with
`FixedSchedule.constant(1.0, 200)` and a `DiscrepancyStop` that never fires
ends at $\beta = 200$ with `status="schedule_exhausted"` — the *same* status a
successfully completed sampling ladder reports — having failed to fit the data.
The layer argues at length that returning an ensemble at $\beta = 0.7$ labelled
a posterior is unacceptable; reporting an unconverged fit as a completed budget
is the symmetric failure, and `converged` is the one-word answer.

There is no `"max_steps"` status, because **exceeding `max_steps` raises**. The
bound is a safety net against a schedule that can never be exhausted and a run
with no stopping rule; a genuinely step-limited run is a `FixedSchedule` with
that many rungs, or a `break` in an `iterate` loop. A sampling run that
silently returned an ensemble at $\beta = 0.7$ labelled as a posterior is
exactly the kind of quiet wrongness this package refuses.

**Progress reporting.** The driver emits one record per step at `INFO` on the
logger named `pyeki.eki`, carrying the step, the level, the increment and the
mean misfit, and one at `WARNING` when any member fails. This is the standard
library's `logging` and nothing more: no handler is installed, no configuration
is read, and a caller who does nothing sees nothing. A run of an expensive
model can last hours, and a library that emits *nothing* over that span, with
no API to add without dropping to `iterate`, is a gap rather than a design
choice — while instrumentation as a feature (timings, profiles, progress bars)
stays excluded ({ref}`eki-excluded`).

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
| 2. construction | ranks, static sizes, operator types, field domains | `ensemble` rank ≠ 2; $J < 2$; a raw `uint32` key; `FixedSchedule` increments not all positive; `target` outside $(0,1)$; `tau \le 0` |
| 3. call | problem and per-step shapes, policy outputs, string arguments | `y` not `(N,)`; `noise_cov` side ≠ $N$ or a family; the forward model's output not `(J, N)`; a schedule increment that is non-scalar, non-finite, or not strictly positive; `on_failure` not one of the two permitted strings |
| 4. value (debug) | finiteness of the initial ensemble, `y`, and inflation fields; positivity of `anomaly_factor` | violations yield `nan` or a silently wrong ladder outside debug mode |

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

**`EKIError(RuntimeError)`**, raised for the three conditions under which a
run cannot continue: `max_steps` exceeded, fewer than two valid members (or
any invalid member under `on_failure="raise"`), and a non-finite updated
ensemble. Each message must name the step index, the level, and the
condition; the `max_steps` message must additionally name the schedule and
whether a stopping rule was supplied.

**`EKIError` carries the run, and this is normative.** It has two attributes,
`state` and `history`: the last good `EKIState` and the records accumulated up
to the failure, populated on **every** raise path in the driver. Without them
the exception's stated reason for existing is empty — there would be nothing
to checkpoint, because `run` builds its `EKIResult` internally and the state
lives inside the generator, so the object a caller catches would be a bare
`RuntimeError` subclass carrying a sentence.

The scenario is concrete and not rare: a 200-member ensemble on a cluster, 40
of 50 rungs completed, one step returns $J_v = 1$. With the attributes that is
a resumable checkpoint and a diagnosable failure; without them it is thirty
hours of forward-model evaluations discarded. The same applies to a non-finite
update at rung 39 and to `max_steps` on an adaptive run. The cost is two
attribute assignments, so the three "lose the run" failures become three
recoverable ones:

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

- `EKIState`, `StepSummary` and `StepRecord` each have a required
  `batch_shape` property, computed from the leading axes of each array field
  beyond its core rank (`ensemble` 2, `misfits` 1, `beta` and the scalar
  statistics 0, `key` 0), combined by broadcasting with `ValueError` on
  mismatch. Directly constructed objects always report `()`.
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

Not normative, but it is the evidence for the three-axis claim of
{ref}`eki-axes`, and a variant that cannot be expressed here is a reason to
revisit the design.

| variant | expressed as |
| ------- | ------------ |
| approximate posterior sampling by tempering (multiple data assimilation) | `FixedSchedule.uniform(T)` or `AdaptiveESSSchedule(beta_final=1.0)`, with the default `SquareRootUpdate` |
| the same, in its classical perturbed-observation form | those schedules with `update=StochasticUpdate()` |
| a single Kalman update (the one-step linearized approximation) | `FixedSchedule.constant(1.0, 1)` |
| EKI as an iterative regularization method | `FixedSchedule.constant(1.0, n)` with `stop=DiscrepancyStop()` |
| adaptive-regularization EKI | `AdaptiveMisfitSchedule(beta_final=None)` with `stop=DiscrepancyStop()` |
| inflation-stabilized variants | any of the above with `inflation=` |
| Tikhonov-regularized EKI | an augmented problem; see below — no new code |
| localized EKI | `update=` an update rule from `pyeki.localize` |
| a Langevin-type ensemble sampler | a custom `EnsembleUpdate` holding the prior, using `increment` as its step size |

**Two of these deserve detail**, because they are where the design either
earns its keep or does not.

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

Pairing this recipe with `AdaptiveESSSchedule(beta_final=1.0)` from a prior
ensemble is therefore a confidently wrong posterior with no error, which is
precisely the quiet wrongness this package refuses — so it is stated here
rather than left to be discovered.
::: The regularization weight is the noise block's
scale: `block_diag(noise_cov, (1 / lam) * prior.cov)` gives the penalty
$\tfrac{\lambda}{2}\lVert C_0^{-1/2}(u - m_0)\rVert^2$, a
{class}`~pyeki.linalg.PSDScaled` that whitens as cheaply as `prior.cov` does.
Centring at $m_0$ rather than at the origin is a choice, and the origin is
recovered by passing a zero mean.

*A Langevin-type update fits the protocol.* Such rules add a prior-drift and a
diffusion term to the Kalman-like term and are driven by a step size rather
than a temperature budget. They need the ensemble, the predictions, $y$, the
noise, a step size, a key, and the prior — which is exactly the update
signature plus a field on the rule, and it is why `increment` is passed
separately from `noise_cov` ({ref}`eki-updates`) and why `beta_final=None`
exists. For such a rule `EKIState.beta` reads as accumulated pseudo-time
rather than a tempering level; the layer keeps the name and the bookkeeping,
which are identical.

One caveat, worth a line because it cuts against a property the package
advertises. The drift term involves the prior **precision** $C_0^{-1}$, so such
a rule needs `prior.cov.supports("solve")` — while a plain run needs only
`factor`, which is why {doc}`design` can say that a prior with no cheap inverse
is perfectly usable. The most structurally interesting priors are exactly the
ones where that matters: a `factor`-only covariance drives every shipped
configuration and raises `UnsupportedOpError` at the first Langevin step. The
step-size rule such a variant wants is expressible as a `Schedule`, since the
summary carries the whitened residual matrix ({ref}`eki-schedules`).

**What does not fit**, recorded so the boundary is visible. Three families, and
the reason differs in each case:

- **Mean-field and unscented schemes.** Their members are deterministic
  quadrature nodes regenerated from a mean and a covariance at every step, so
  their *state* is $(m, C)$ and not an ensemble at all. The distinguishing
  feature is that the members do not move freely and are not carried between
  steps — not the value of $J$, which such a scheme also has.
- **Multiple coupled ensembles.** Multilevel schemes carry several ensembles at
  different forward-model fidelities and combine their moments through a
  telescoping-sum estimator. Every member moves freely, so the criterion above
  admits them, but `EKIState` holds one ensemble and one $J$ and the driver
  takes one `forward`. They would need a different state and a different
  driver.
- **Anything that changes the data between rungs.** The driver binds
  `(forward, y, noise_cov)` for the whole run. See {ref}`eki-excluded`, where
  `step` is the answer.

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

One real limit, recorded here rather than discovered during that
implementation. A local analysis needs the noise covariance restricted to a
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
`StepSummary(step=3, n_members=64)`, `StepRecord(step=3)`. Policy objects
print their static fields, which are small and informative:
`AdaptiveESSSchedule(target=0.5, beta_final=1.0)`,
`MultiplicativeInflation(anomaly_factor=1.02)` — with the one exception that a
policy holding a *large* static field summarizes it instead
(`FixedSchedule(n_steps=200, total=200.0)`, {ref}`eki-schedules`), since the
general rule assumes those fields are small. `EKIResult` prints its status and
counts: `EKIResult(status='schedule_exhausted', n_steps=17, beta=1.0)`.
`repr` never raises.

(eki-surface)=
## Public surface

`pyeki.eki` exports exactly: the value classes `EKIState`, `StepSummary`,
`StepRecord`, `EKIResult`; the protocols `EnsembleUpdate`, `Schedule`,
`StoppingRule`, `Inflation`; the update rules `SquareRootUpdate`,
`StochasticUpdate`; the schedules `FixedSchedule`, `AdaptiveESSSchedule`,
`AdaptiveMisfitSchedule`; the stopping rule `DiscrepancyStop`; the inflations
`MultiplicativeInflation`, `AdditiveInflation`; the driver `run`, `iterate`
and `step`; the helpers `whitened_misfits`, `effective_sample_size`,
`repair_failed_members`; the status constants `SCHEDULE_EXHAUSTED`,
`STOPPING_RULE`, `INTERRUPTED`; and the exception `EKIError`. Anything else is
private, and no consumer may depend on it.

The three helpers are public because writing a schedule requires them:

| helper | signature | returns |
| ------ | --------- | ------- |
| `whitened_misfits(y, predictions, noise_cov)` | `(N,), (..., N), PSDLinOp -> (...)` | $\tfrac12\lVert W(y - v)\rVert^2$, batched per the operator layer's contract |
| `effective_sample_size(misfits, increment)` | `(J,), scalar -> 0-d` | $\mathrm{ESS}$ of $e^{-\delta\Phi}$, computed in log space |
| `repair_failed_members(ensemble, predictions, valid)` | `(J,P), (J,N), (J,) bool -> (J,P), (J,N)` | the moment-preserving repair of {ref}`eki-failures` |

There is no `pyeki.eki.testing`: the policy protocols are structural and have
no invariants a harness could check beyond the signature, so conformance is a
set of obligations on the package's own tests, as in `pyeki.gauss`.

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
   with `StochasticUpdate` reproduces, elementwise for a fixed key, a
   hand-written dense reference that applies the perturbed-observation formula
   rung by rung; and its posterior moments match the one-shot posterior in
   expectation, tested as a mean over many keys with a tolerance derived from
   the $KRK^\top/J$ scale rather than tuned.
4. **Schedules.** `FixedSchedule` takes exactly its increments in order and
   exhausts after exactly $T$ steps, having made exactly $T$ ensemble
   evaluations, **and completes under `max_steps == T`** — the executable form
   of the exhaustion-before-bound ordering of {ref}`eki-step`, and a
   regression test for the inversion that would make the natural `max_steps`
   always raise. A schedule whose `increment` returns `None` ends the run with
   `status="schedule_exhausted"` and a terminal record. The shipped defaults
   are mutually reachable: a budgeted adaptive schedule at
   `min_increment=1e-3` never raises at `max_steps=1000`, asserted by
   arithmetic on the two defaults rather than by running $10^3$ steps. **Both** adaptive schedules reach `beta_final` without
   overshoot, never return a non-positive increment, respect the documented
   clamp precedence of {ref}`eki-adaptive` (a case where the floor binds, a
   case where the budget cap beats the floor, a case where `max_increment`
   binds), and take the largest allowed step when every misfit is identical.
   `AdaptiveESSSchedule`'s returned increment attains its ESS target before
   the floor is applied. `AdaptiveMisfitSchedule` returns the larger of its two
   bounds, with a test for each regime — the mean bound binding when the misfit
   coefficient of variation exceeds $\sqrt{2/N}$ and the variance bound below
   it — and is `inf`-guarded rather than `nan` at zero misfit spread, at zero
   mean misfit, and at `theta = 0`.
5. **The ESS criterion.** `effective_sample_size` is $J$ at zero increment,
   monotone non-increasing along a grid of increments, correct against a
   direct small-value computation, equal to $J/(1+\mathrm{cv}^2)$ for the
   weights it defines, and finite for misfits of order $10^{4}$ where the
   naive non-log-space formula returns `nan` — the latter as a targeted
   regression test.
5a. **The two adaptive criteria are distinct, and measurably so.** On the same
   ensemble, `AdaptiveMisfitSchedule`'s increment sits near the relative ESS of
   $2/(N+2)$ that {ref}`eki-schedules` predicts for it, and is therefore much
   larger than `AdaptiveESSSchedule`'s at a target near $1/2$ — pinning the
   comparison in both directions, so that neither schedule can silently drift
   into implementing the other's criterion.
6. **Stopping.** `DiscrepancyStop` fires exactly when $2\Phi(\bar v) \le
   \tau^2 N$, is consulted before the increment, ends the run with status
   `"stopping_rule"` and a terminal record whose increment is exactly zero,
   and can end a run at step 0 with an empty update history.
7. **`max_steps` and the error payload.** An unbounded schedule with no
   stopping rule raises `EKIError` at the bound, with a message naming the
   schedule; the bound is never reached by a schedule with a budget and a
   positive floor. **Every `EKIError` raise path carries `state` and
   `history`** — the bound, $J_v < 2$, `on_failure="raise"`, and a non-finite
   update — and resuming a run from a caught `exc.state` continues it exactly,
   which is what makes the attributes worth having.
8. **Failure handling.** `repair_failed_members` leaves valid members
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
9. **Inflation.** `MultiplicativeInflation` preserves the mean exactly and
   scales the empirical covariance by `anomaly_factor**2` exactly; a run with it is
   unchanged in the members' span. `AdditiveInflation` preserves the mean
   exactly, matches its pinned elementwise definition, and moves the ensemble
   out of its initial affine subspace — while a run without it stays inside,
   which is the executable form of {ref}`eki-subspace`.
10. **Misfits.** `whitened_misfits` matches a dense
    $\tfrac12 (y-v)^\top R^{-1} (y-v)$ at batch ranks 0, 1 and 2, is invariant
    across two noise operators representing the same $R$ with different
    whiteners, and pins the factor of $\tfrac12$ against a closed-form case.
    `centre_misfit` differs from `misfit_mean` by exactly
    $\tfrac{J-1}{2J}\operatorname{tr}(W\widehat{C}_{vv}W^\top)$ — a divisor
    the derivation gets wrong easily and which no tolerance-based test would
    catch. `misfits` and `centre_misfit` are recovered from the summary's
    stored `whitened_residuals` alone, and the whitened prediction anomalies
    recovered from it agree with $-W A_v$.
11. **Reproducibility and resumption.** The same initial state and policies
    give bit-identical results. Stopping a run after $k$ steps and resuming
    from the returned state reproduces the uninterrupted run's remaining
    records and final ensemble bit-exactly — the executable form of the
    policy-purity rule. `iterate` and `run` agree.
12. **The optimization form.** On an affine problem, a run with
    `FixedSchedule.constant(1.0, n)` and `DiscrepancyStop` terminates, its
    misfit decreases monotonically in the affine case, and its centre
    approaches the least-squares solution restricted to the initial
    ensemble's affine subspace — the correct target given
    {ref}`eki-subspace`, not the unrestricted minimizer.
13. **JAX.** The array computations run under `jit`; a multi-step run
    **compiles a bounded number of times, independent of the number of
    steps** (asserted against a compilation counter, not inspected by eye) —
    the executable form of the traced-increment requirement; flatten and
    unflatten preserve type and behaviour for all three pytree classes, with
    sentinel leaves; families report their batch shape, take the
    `vmapped(...)` repr, and refuse every method.
13a. **The history stacks.** `jax.tree.map(lambda *xs: jnp.stack(xs),
    *result.history)` succeeds on a multi-step run and yields a `StepRecord`
    with `batch_shape == (T,)` and `(T,)`-shaped fields — including `step` and
    `n_valid`. This is a targeted regression test for the treedef trap of
    {ref}`eki-diagnostics`: declaring either as static metadata makes every
    record a different pytree type and the documented one-liner raises. An
    empty history is checked to raise from `jax.tree.map` rather than to
    return something misleading.
13b. **The step is public and composes.** `step` with an explicit increment
    reproduces the corresponding rung of an equivalent `run` bit-exactly;
    re-stepping twice from the same state with different increments leaves the
    original state untouched and yields the two corresponding results, which is
    the property the backtracking pattern of {ref}`eki-step` relies on; and
    `iterate` yields exactly what a hand-written loop over `step` with the same
    schedule decisions yields.
14. **Degeneracy.** $J = 2$, $N = 1$, and $P = 1$ all work. A collapsed
    ensemble produces no `nan` and, with zero prediction anomalies, no
    movement. A forward model returning `nan` for every member raises rather
    than propagating. A `nan`-producing update raises `EKIError` naming the
    step.
15. **Validation and repr.** Every tier-2 and tier-3 rule of
    {ref}`eki-validation` raises as specified; reprs match {ref}`eki-repr`
    with no array data, and `FixedSchedule`'s summarizes rather than printing
    its increments; the pinned prior draw and a short run's output are
    snapshotted so a JAX-side PRNG change is detected rather than absorbed.
16. **The result reports the run.** `status` takes only the three permitted
    values and equals the exported constants; `converged` is true exactly when
    a stopping rule was supplied and fired, and **false** for an optimization
    run whose ladder ran out before its `DiscrepancyStop` fired — the case
    {ref}`eki-driver` exists to distinguish. `last_evaluated` holds the
    ensemble and predictions of the final forward evaluation, and is `None`
    exactly when the run made none; its ensemble is *not* `result.ensemble` on
    a schedule-exhausted run, and the test asserts the off-by-one rather than
    assuming it. `n_evaluations` equals the number of forward calls, counted
    by an instrumented model.
17. **Inflation sees the ladder.** An `Inflation` and an `EnsembleUpdate` that
    record their `step` and `beta` arguments observe the true sequence, and a
    rule varying with `beta` gives a run that is still exactly resumable —
    the point of passing the arguments instead of forcing a stateful
    workaround ({ref}`eki-updates`).

Alongside conformance, targeted regression tests guard this layer's
silent-failure classes under the same do-not-delete rule as the layers below:
the $R/\beta$ mis-scaling, the non-log-space ESS, an inflation factor applied
to the covariance rather than the anomalies (the `anomaly_factor` convention),
a repair applied when nothing failed, a repair that rescales the surviving
members (and so inflates silently), a misfit computed before repair, a schedule
that counts its own calls (and so breaks resumption), an increment baked in as
a Python constant (and so recompiles every step), a `StepRecord` field declared
static (and so unstackable), and the safety bound checked before ladder
exhaustion (and so raising on a completed run).

(eki-excluded)=
## Deliberately excluded

Recorded so their absence reads as a decision, not an oversight.

**A `lax.scan` driver.** The forward model is an arbitrary callable and may
not be traceable at all, so the loop cannot be a scan. Everything that *can*
be traced is, and the loop stays Python. A run whose forward model happens to
be pure JAX gains a fully-scanned variant only by giving up the per-step
Python decisions — adaptive increments, termination, failure branching — that
this contract is largely about; no consumer has asked for the trade.

**A `Problem` container** bundling `(forward, y, noise_cov)`. The triple is
passed to `run`, `iterate`, `step` and the helpers, and a container would
document its shape agreement once.

The tempting objection — that a field holding an arbitrary callable can be
neither pytree data nor hashable static metadata, so the container could not be
a pytree — **does not hold**, and is recorded as rejected so it is not
rediscovered: `EKIResult` is already a plain frozen dataclass rather than a
pytree, on the ground that it is never an argument to traced code, and a
`Problem` would qualify under exactly the same rule since the driver unpacks it
and never traces the forward model.

It is excluded on the honest ground instead: three arguments are not many, the
container would carry no behaviour, and every call site that takes the triple
would then accept either form or force a conversion. The shape agreement it
would document is validated once per run anyway ({ref}`eki-driver`). Revisit if
a fourth element joins the triple, at which point the balance changes.

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
every rung. Recorded because the three-axis table of {ref}`eki-axes` names
schedule, update and inflation, and never the data — an omission a reader would
otherwise assume was an oversight.

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
state they need is $(m, C)$ rather than an ensemble; see {ref}`eki-variants`.

**Multilevel and multi-fidelity variants.** They carry several coupled
ensembles at different forward-model resolutions and estimate moments by a
telescoping sum across them. `EKIState` holds one ensemble, `run` takes one
`forward`, and the update protocol sees one $(J, P)$ array, so this is a
different driver rather than a policy. Adjacent to the fixed-$J$ decision
below.

**The moment-exact repair of failed members.** Rescaling every anomaly by
$\sqrt{(J-1)/(J_v-1)}$ matches the valid members' moments exactly, but moves
the surviving members outward by a data-dependent factor at every step —
silent inflation, in the configuration nobody changes. The shipped repair
damps instead; {ref}`eki-failures` gives the arithmetic.

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
adapts to the misfits would need the summary, enlarging the protocol for no
current consumer. The relaxation schemes — blending the posterior ensemble's
anomalies or spread back toward the prior step's — need *both* the pre-update
and post-update ensembles, so they are not `Inflation`s at all under this
layer's signature; they are post-update transformations, and adding them would
mean a fifth protocol. All three are recorded rather than designed, and all
three are writable as a custom `EnsembleUpdate` wrapping a shipped one, which
does see both ensembles.

**The misfit at the mean parameter, $\Phi(G(\bar u))$.** A third misfit
diagnostic alongside the two the summary provides, and the one some
discrepancy criteria are written against. It costs an extra forward-model
evaluation per step — a $1/J$ increase in the run's dominant cost for a
diagnostic — so it is the caller's to compute in an `iterate` loop if wanted.

**Conditioning diagnostics in the history.** The singular values of the scaled
whitened anomaly matrix, and the marginal-likelihood term
$\sum_i\log(1+\sigma_i^2)$ that {doc}`gaussian-contract` derives from the same
decomposition, are the natural answer to "is my ensemble collapsing". They are
excluded from `StepRecord` because the update's SVD is internal to
`pyeki.gauss` and has no return channel, so the driver would have to compute a
second one at $O(NJ^2)$ — the update's own order — for a diagnostic. They stay
reachable: `summary.whitened_residuals` determines the matrix completely, so a
caller takes the SVD in an `iterate` loop and pays for it on purpose
({ref}`eki-diagnostics`). Revisit if `pyeki.gauss` ever grows a decomposition
accessor, which it records as awaiting a consumer.

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
