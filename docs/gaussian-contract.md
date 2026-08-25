# Joint Gaussian contract

This page specifies `pyeki.gauss`: the classes it provides, the contract of
every method, and the conditioning mathematics that all of them share. It is normative — an implementation that violates a rule here is
defective even if its tests pass — and it is the reference for two audiences:
contributors implementing or reviewing the layer, and users who want a more
precise account of Gaussian conditioning in pyEKI than the user guide gives.

Throughout, *must* and *never* state requirements, *should* states a strong
default that a documented reason may override, and *may* states a permission.
{doc}`design` records *why* the load-bearing decisions were made; this page
records *what* they require. The layer is built on the operator layer, and
this contract freely references {doc}`linop-contract` rather than restating
its rules.

:::{admonition} Status: specification ahead of code
:class: important

`pyeki.gauss` does not exist yet. This document is the design it will be
built to, and it is the artifact to review and iterate on before
implementation begins. Once the module ships, this page remains as the
normative reference for its behaviour.
:::

## Scope

The layer represents joint Gaussian distributions over a pair of blocks — in
EKI, parameters and predicted observations — and conditions them on a noisy
observation of the second block. It provides exactly what Ensemble Kalman
Inversion needs from Gaussian machinery:

- **sampling** from a marginal Gaussian whose covariance is a structured
  operator (drawing the initial ensemble from the prior);
- **the ensemble update**, in both its stochastic (perturbed-observation) and
  deterministic (square-root transform) forms — Gaussian conditioning applied
  to the Gaussian determined by an ensemble's empirical moments — together
  with that posterior as a distribution, in structured low-rank form;
- **exact moment conditioning** of an operator-represented joint, as the
  reference the ensemble forms are tested against;
- **the array-level conditioning primitives** underneath both, exposed at the
  granularity that domain localization consumes.

It is deliberately not a probabilistic-programming layer: there are no
densities over anything non-Gaussian, no transformations of distributions,
and no inference machinery. {ref}`gauss-excluded` lists what is left out and
why.

Everything numerical routes through one algorithm, the **whitened-SVD
conditioning kernel** ({ref}`gauss-kernel`) — *kernel* in the computational
sense, the shared numerical core of every conditioning routine; nothing in
this package uses the word to mean a covariance kernel. The competing route — the
Woodbury identity applied to the normal equations — squares a condition
number that is already the failure mode of late-stage EKI, and is excluded;
{doc}`design` gives the full comparison.

(gauss-notation)=
## Notation and conventions

One convention set governs the whole layer. Symbols used throughout:

| symbol | meaning |
| ------ | ------- |
| $u$    | the block to be updated (in EKI: the parameters), dimension $P$ |
| $v$    | the observed block (in EKI: the predicted observations), dimension $N$ |
| $y$    | the observation, a vector of length $N$ |
| $R$    | the observation-noise covariance, an $N \times N$ PSD operator |
| $W$    | a whitener of $R$: the fixed matrix `noise_cov.whiten` applies, $W R W^\top = I_N$ |
| $J$    | the number of ensemble members |
| $r$    | a residual $y - v$, of length $N$ |

Conventions, each normative:

- **Samples are stored row-wise: an ensemble is a `(J, dim)` array**, one
  member per row. This is what a `vmap`-ed forward model produces and what
  the operator layer's batch contract ({ref}`contract-batch`) treats as a
  batch of vectors, so ensembles flow between the two layers with no
  transposes. Displayed mathematics in this page follows the same
  convention: the sample matrices $\mathsf{U} \in \mathbb{R}^{J \times P}$
  and $\mathsf{V} \in \mathbb{R}^{J \times N}$ have one member per row.
- **Anomalies are raw deviations from the sample mean.** With
  $\bar{u} = \frac{1}{J}\sum_j u_j$, the anomaly matrix is
  $A_u = \mathsf{U} - \mathbf{1}\bar{u}^\top \in \mathbb{R}^{J \times P}$,
  and likewise $A_v$. No normalization is folded into the anomalies
  themselves; the divisor appears explicitly in the formulas.
- **Empirical covariances use the divisor $J - 1$**:
  $\widehat{C}_{uv} = A_u^\top A_v / (J-1)$, and so on. The choice is fixed,
  not configurable ({ref}`gauss-excluded`).
- **Vectors passed to methods are exactly core-shaped.** An observation `y`
  is a `(N,)` array, a mean a `(dim,)` array. The gauss classes do not
  accept batched operands the way operator methods do; a family of
  distributions or updates is expressed with `jax.vmap` over the pytree
  ({ref}`gauss-jax`). The two conditioning *primitives* are the exception:
  they are array-level and follow the operator layer's batch contract
  ({ref}`gauss-primitives`).
- **Noise covariances are `PSDLinOp`s.** Every `noise_cov` argument must be
  a {class}`~pyeki.linalg.PSDLinOp` of side $N$; a non-operator or an
  operator of the wrong shape is a `TypeError` / `ValueError` at call time.
  The ensemble updates use it only through `whiten` — by design, the one
  operation the noise covariance must support ({doc}`design`) — so a noise
  operator with no factorization at all still drives every update, and
  tempering's per-step covariance `noise_cov / dbeta` (a
  {class}`~pyeki.linalg.PSDScaled`) whitens as cheaply as the base operator.

(gauss-objects)=
## The objects

The public surface is three classes and two functions. Each class is one
*representation* of a Gaussian object, and the representation is explicit in
the type — the layer never converts between representations behind the
caller's back.

| object          | represents                            | representation                                | role in EKI |
| --------------- | ------------------------------------- | --------------------------------------------- | ----------- |
| `Gaussian`      | one Gaussian distribution             | mean vector + `PSDLinOp` covariance           | the prior; the posteriors `condition` returns |
| `EnsembleJoint` | the joint Gaussian with an ensemble's empirical moments | $J$ paired samples                | every update — the hot path |
| `JointGaussian` | a joint Gaussian over $(u, v)$        | mean vectors + covariance blocks as operators | the exact reference the ensemble forms are tested against |
| `gain_weights`  | the Kalman gain, in whitened variables| pure array function                           | the shared conditioning core; localization's entry point |
| `sqrt_transform`| the square-root update transform      | pure array function                           | the shared conditioning core; localization's entry point |

Three rules govern the set:

1. **Representations are explicit and never coerced.** An `EnsembleJoint`
   stores samples and computes through anomalies; a `JointGaussian` stores
   means and covariance blocks as operators. No method converts one
   representation into the other implicitly — the conformance suite builds
   both from the same data *in the tests*, where the equivalence is the
   thing being checked.
2. **The set is closed.** Unlike the operator layer, `pyeki.gauss` has no
   extension story: users do not subclass these classes. User extensibility
   lives one layer down, in the operators supplied as covariances — a custom
   noise or prior operator that passes `check_operator` works here unchanged.
   There is consequently no public class decorator and no
   `pyeki.gauss.testing` module ({ref}`gauss-conformance`).
3. **Objects are unbatched frozen pytrees, exactly like operators.** The
   construction, validation, and JAX-integration rules of the operator
   contract apply verbatim ({ref}`gauss-jax`); a family of joints is a
   `vmap`-ed pytree, never stored batch axes.

(gauss-kernel)=
## The conditioning kernel

All conditioning in the layer is one computation, specified here once. The
class methods ({ref}`gauss-ensemble`) are thin assemblies of these pieces;
the two public functions ({ref}`gauss-primitives`) expose them directly.

### The whitened anomaly matrix

Given prediction anomalies $A_v$ and a noise operator with whitener $W$,
define the **scaled whitened anomaly matrix**

$$
S \;=\; \frac{1}{\sqrt{J-1}}\, A_v W^\top \;\in\; \mathbb{R}^{J \times N},
$$

whose $j$-th row is $W a_j / \sqrt{J-1}$ — in code, exactly
`noise_cov.whiten(A_v) / sqrt(J - 1)`, one call under the operator layer's batch
contract. Let

$$
S = U \Sigma V^\top \quad \text{(thin SVD)}, \qquad
U \in \mathbb{R}^{J \times r},\;
V \in \mathbb{R}^{N \times r},\;
r = \min(J, N),
$$

with singular values $\sigma_1 \ge \dots \ge \sigma_r \ge 0$. Because the
rows of $A_v$ sum to zero, $S^\top \mathbf{1} = 0$: the all-ones direction
is in the null space of $S^\top$, so every $\sigma_i > 0$ has
$U_{\cdot i} \perp \mathbf{1}$. Mean-centering also caps the rank at
$J - 1$, so at least one singular value is exactly zero whenever
$N \ge J$; every formula below is continuous at $\sigma = 0$ and needs no
special-casing.

### The gain

For the empirical covariances $\widehat{C}_{uv} = A_u^\top A_v/(J-1)$ and
$\widehat{C}_{vv} = A_v^\top A_v/(J-1)$, the Kalman gain
$K = \widehat{C}_{uv}\,(\widehat{C}_{vv} + R)^{-1}$ applied to a residual
$r$ satisfies

$$
K r \;=\; \frac{1}{\sqrt{J-1}}\, A_u^\top\, w, \qquad
w \;=\; U \operatorname{diag}\!\Bigl(\frac{\sigma_i}{1+\sigma_i^2}\Bigr) V^\top\, (W r)
\;\in\; \mathbb{R}^{J}.
$$

The vector $w$ is the **weight vector**: the update to $u$ is the
combination $A_u^\top w / \sqrt{J-1}$ of the ensemble's own anomalies, so
no matrix of dimension $P$ or $N$ is ever formed and the update stays in
the span of the $u$-anomalies. In closed form, independent of the SVD,

$$
w \;=\; S\,(S^\top S + I_N)^{-1}\,(Wr)
\;=\; \frac{1}{\sqrt{J-1}}\, A_v\,(\widehat{C}_{vv} + R)^{-1}\, r ,
$$

which exhibits two contract-level properties:

- **Whitener invariance.** $w$ depends on the noise operator only through
  $R$ itself, not through which valid $W$ the operator chose — the freedom
  the operator contract's `whiten` specification grants
  ({doc}`linop-contract`): every whitener
  satisfying $WRW^\top = I$ yields the same weights. Implementations are
  free to route through `whiten`, and results must not depend on the
  routing.
- **Unconditional boundedness.** $\sigma/(1+\sigma^2) \le 1/2$ for all
  $\sigma \ge 0$, so the gain cannot blow up however collapsed or
  ill-conditioned the ensemble becomes, with no regularization parameter to
  tune.

The SVD form is the normative implementation: it never forms
$S^\top S$ or $S S^\top$, whose condition numbers are the *squares* of
$S$'s. See {doc}`design` for the numerical comparison with the
normal-equations route.

### The square-root transform

The deterministic update replaces sampling noise with an exact transform of
the anomalies. Define

$$
T \;=\; (I_J + S S^\top)^{-1/2}
\;=\; I_J + U\bigl((I_r+\Sigma^2)^{-1/2} - I_r\bigr)U^\top
\;\in\; \mathbb{R}^{J \times J},
$$

symmetric, built from the same SVD. The second form is the normative one:
for a thin SVD the naive $U(I+\Sigma^2)^{-1/2}U^\top$ *omits the identity on
the orthogonal complement* and is simply wrong whenever $r < J$ — the
$I_J + U(\cdot - I)U^\top$ form is exact for every rank.

Transformed anomalies $T A_u$ have empirical covariance exactly equal to the
posterior covariance of the empirical joint:

$$
\frac{(T A_u)^\top (T A_u)}{J-1}
\;=\; \widehat{C}_{uu} - K\,\widehat{C}_{vu}.
$$

Two structural facts, both load-bearing and both conformance-checked:

- **The transform preserves mean-centering**: $T\mathbf{1} = \mathbf{1}$
  (each $U_{\cdot i}$ with $\sigma_i > 0$ is orthogonal to $\mathbf{1}$, and
  the modifier vanishes at $\sigma_i = 0$), so transformed anomalies still
  sum to zero and the posterior ensemble's mean is not silently shifted.
- **The identity is exact in exact arithmetic**, not asymptotic in $J$. The
  conformance suite checks it to floating-point tolerance, per the
  exactness-test convention.

### Cost

Whitening costs $J$ applications of $W$; the thin SVD is
$O(J N \min(J, N))$; forming weights is $O(J r)$ per residual and combining
anomalies $O(JP)$ per member. A full ensemble update is
$O(NJ^2 + PJ^2)$ for $J \le N$ — linear in both dimensions, cubic only in
the ensemble size.

(gauss-primitives)=
## Conditioning primitives

The two pieces above are public functions in `pyeki.gauss`, defined as pure
matrix functions of their array arguments — no divisor, no whitening, and no
randomness folded in. They exist as the layer's *advanced tier*: the class
methods cover EKI's global update, and domain localization (the planned
`pyeki.localize`) calls these directly, once per local analysis, under
`jax.vmap` over fixed-size neighbourhoods.

| function | signature | returns |
| -------- | --------- | ------- |
| `gain_weights(s, b)`   | `(J, N), (..., N) -> (..., J)` | $U \operatorname{diag}\bigl(\sigma_i/(1+\sigma_i^2)\bigr) V^\top b$ for the thin SVD $s = U\Sigma V^\top$ |
| `sqrt_transform(s)`    | `(J, N) -> (J, J)`             | $(I_J + s s^\top)^{-1/2}$ |

Rules:

- **`s` is exactly 2-D** — it plays the operator's role, so it carries no
  batch axes; a family of local analyses is a `vmap` over the function.
  `b` follows the operator layer's batch contract: trailing core axis of
  length $N$, any number of leading batch axes carried through. Violations
  raise `ValueError` with the same message obligations as tier-3 operand
  checks ({ref}`contract-validation`).
- **Both functions compute one SVD per call.** Batch the residuals into a
  single `gain_weights` call rather than looping; the $J$ per-member
  residuals of an update are one `(J, N)` operand.
- The functions are deterministic, differentiable JAX code: safe under
  `jit`, `vmap`, and `grad`, with no data-dependent shapes.
- Callers own the semantics of `s`. For conditioning, `s` must be the
  *scaled, whitened* anomaly matrix of {ref}`gauss-kernel`, and `b` a
  *whitened* residual; the functions cannot check this, which is why the
  class methods — where the conventions are enforced — are the default
  interface and these functions are the escape hatch.

(gauss-marginal)=
## `Gaussian`

A single Gaussian distribution $\mathcal{N}(m, C)$: the prior a caller
supplies, and the posterior that conditioning returns — from either joint
class ({ref}`gauss-ensemble`, {ref}`gauss-joint`).

**Fields.** `mean`, a `(n,)` array, and `cov`, a
{class}`~pyeki.linalg.PSDLinOp` of side `n`. Construction validates the rank
of `mean`, the operator type of `cov`, and their agreement
(tier 2, shape-only). `n` is a property computed from `mean`.

**Capabilities delegate to the covariance.** `Gaussian` defines no
capability system of its own: each method requires specific operations of
`cov`, and an unsupported one raises the operator layer's
`UnsupportedOpError`, unmodified, from the inner call. `gaussian.cov`
is a public field, so callers gate exactly as they do on operators:
`gaussian.cov.supports("factor")`.

### `sample(key, n_samples)`

Returns a `(n_samples, n)` array of independent draws. Requires
`cov.supports("factor")`.

- `n_samples` must be a Python `int`, at least 1 — it determines an output
  shape, so it can never be traced.
- The draw is **pinned elementwise**, not merely distributionally: with `L`
  the operator `cov.factor()` returns and `k` its width, the result is
  exactly

  ```python
  mean + L.matvec(jax.random.normal(key, (n_samples, k)))
  ```

  one batched `matvec` under the operator layer's batch contract. Pinning
  the draw makes sampling reproducible across releases for a fixed
  covariance representation and makes the conformance check elementwise
  rather than statistical. What is *not* pinned is anything across
  representations: two operators for the same matrix may return different
  factors, hence different samples from the same key — only the
  distribution is shared.

### `log_density(x)`

Returns the log-density at `x`, batched: `(..., n) -> (...)`, each element a
0-d real JAX scalar (never a Python float). Requires
`cov.supports("whiten")` and `cov.supports("logdet")`. The value is

$$
\log p(x) \;=\; -\tfrac{1}{2}\Bigl(\, n\log 2\pi \;+\; \log\det C \;+\;
\lVert W(x - m)\rVert^2 \,\Bigr),
$$

correct for every valid whitener by the invariant
$\lVert W r \rVert^2 = r^\top C^{-1} r$ that the operator contract
guarantees. Precondition: $C$ nonsingular — a singular covariance yields
`nan`/`inf` downstream of the operator layer, per its tier-4 convention.

(gauss-ensemble)=
## `EnsembleJoint`

The joint Gaussian determined by an ensemble's empirical moments, held in
sample form: the object EKI builds once per step, uses for one update, and
discards. Its distribution is
$\mathcal{N}\!\left(\begin{pmatrix}\bar u\\ \bar v\end{pmatrix},
\begin{pmatrix}\widehat{C}_{uu} & \widehat{C}_{uv}\\
\widehat{C}_{vu} & \widehat{C}_{vv}\end{pmatrix}\right)$ — a Gaussian *fit
to* the members, not the empirical (equal-weight point-mass) distribution
of the members themselves: every conditioning method below is exact
Gaussian conditioning applied to this fitted Gaussian. The samples are the
*representation* from which its moments are read, and no moment matrix is
ever formed.

**Fields.** `u_samples`, a `(J, P)` array, and `v_samples`, a `(J, N)`
array — member-aligned: row $j$ of each belongs to the same member.
Construction validates exact rank 2 on both, agreement of the leading axes,
and $J \ge 2$ (a single sample has no anomalies; the check is shape-only,
so it is tier 2 and unconditional). $P \ge 1$ and $N \ge 1$, as everywhere.

**Derived attributes.** `n_members`, `u_dim`, `v_dim` are `int` properties.
`u_mean`, `v_mean`, `u_anomalies`, `v_anomalies` are array properties
computed on access from the stored samples. This is not the forbidden lazy
factorization ({ref}`contract-jax`): nothing is cached and nothing
factorizes — each access is an $O(J\cdot\mathrm{dim})$ mean-and-subtract
that fuses under `jit`. The samples are the whole state, and
`(mean, anomalies)` is recovered from them exactly.

**No factorization at construction — deliberately.** The operator layer's
rule is *factorize at construction*; here there is nothing to factorize:
the SVD of $S$ depends on the noise operator, which arrives per update
call and changes every tempering step. Each update method computes its SVD
once, uses it, and discards it — never caching it on the instance, which
under `jit` would be the discarded-cache bug the operator contract
documents. One method call, one SVD.

All three conditioning methods share two trailing arguments: `y`, the
observation, a `(N,)` array (tier-3 core-shape check), and `noise_cov`, a
`PSDLinOp` of side $N$ supporting `whiten` (`UnsupportedOpError` from the
inner call otherwise). The two update methods return a `(J, P)` array of
updated members — they update $u$ only, since EKI re-evaluates the forward
model to get the next step's $v$ — while `condition` returns the same
posterior as a distribution. All are deterministic functions of their
arguments (`pathwise_update` includes the key among them), safe under
`jit`, and all degrade gracefully when the prediction anomalies are zero:
the updates return `u_samples` unchanged and `condition` returns the prior
marginal's moments — a collapsed ensemble is a no-op, not `nan`.

### `pathwise_update(key, y, noise_cov)`

The stochastic (perturbed-observation) EKI update: pathwise conditioning of
the joint's own samples, one fresh perturbation per member. The result is

$$
u_j' \;=\; u_j + \frac{1}{\sqrt{J-1}}\, A_u^\top\, w_j, \qquad
w_j = \texttt{gain\_weights}\bigl(S,\; b_j\bigr), \qquad
b_j = W(y - v_j) - \varepsilon_j,
$$

with $\varepsilon_j$ the rows of the **pinned draw**
`jax.random.normal(key, (J, N))`. The perturbation enters *only* in
whitened space: $b_j$ is the whitened residual against the perturbed
prediction $v_j + W^{-1}\varepsilon_j$ (equivalently: $u_j' = u_j +
K(y - v_j - \eta_j)$ with $\eta_j = W^{-1}\varepsilon_j \sim
\mathcal{N}(0, R)$ — the identity the conformance suite checks
elementwise against a dense reference). Pinning the draw fixes the exact
output for a given key, which makes runs reproducible and the method
testable without statistics.

:::{warning}
The method **neither accepts nor exposes the perturbations**: no `eps`
argument, no perturbed observations in the return value. This is the
mixed-representation warning of the operator contract's `whiten` section
({doc}`linop-contract`), made structural: a perturbation used through the
whitened shortcut must never also be pushed
through `factor()` in the same update, and the surest way to prevent that
is for the same code to own both the draw and its single use. A caller who
needs perturbed observations materialized is building a different
algorithm and should use the conditioning primitives directly, drawing its own
$\varepsilon$ and choosing one representation for it.
:::

### `transform_update(y, noise_cov)`

The deterministic (square-root) update: the moment-form posterior of the
empirical joint, returned in ensemble representation. The result is

$$
u_j' \;=\; \underbrace{\bar u + \frac{1}{\sqrt{J-1}}\, A_u^\top\,
\texttt{gain\_weights}\bigl(S,\, W(y - \bar v)\bigr)}_{\text{posterior mean}}
\;+\; \bigl(T A_u\bigr)_j, \qquad T = \texttt{sqrt\_transform}(S).
$$

No randomness, no `key`. The returned ensemble is an *exact* ensemble
representation of the moment posterior: its sample mean equals the
posterior mean and its sample covariance (divisor $J-1$) equals the
posterior covariance, both in exact arithmetic — the identity of
{ref}`gauss-kernel`, and the bridge the conformance suite uses between the
ensemble and block representations. Because $T\mathbf{1} = \mathbf{1}$, the
transformed anomalies remain centred, so the two summands above really are
the posterior mean and posterior anomalies.

`transform_update` and `pathwise_update` produce ensembles with the same
first and second moments in expectation; they differ in that the transform
is exact per-realization while the pathwise form is exact in distribution.
Which to use is the EKI layer's decision, not this layer's.

### `condition(y, noise_cov)`

Moment-form conditioning in the ensemble representation: the same posterior
that `transform_update` represents as members, returned as a
{class}`Gaussian` — for sampling the posterior at any size, and for
diagnostics. The result has

$$
m_{\text{post}} \;=\; \bar u + \frac{1}{\sqrt{J-1}}\, A_u^\top\,
\texttt{gain\_weights}\bigl(S,\, W(y - \bar v)\bigr),
\qquad
C_{\text{post}} \;=\; F F^\top, \quad
F = \frac{1}{\sqrt{J-1}}\,(T A_u)^\top \in \mathbb{R}^{P \times J},
$$

with the covariance returned **in structured form**: a `PSDLowRank`
operator holding the factor $F$ — never a dense $P \times P$ matrix, so no
size guard is needed. The exact relationship to the transform update, a
conformance obligation, is

$$
\texttt{transform\_update}(y, R)_j
\;=\; m_{\text{post}} + \sqrt{J-1}\; F_{\cdot j}.
$$

The returned covariance is honest about rank:
$\operatorname{rank}(C_{\text{post}}) \le J - 1$, so it is singular
whenever $J - 1 < P$ — the usual EKI regime. The posterior `Gaussian`
therefore supports `sample` (the factor is the stored representation) but
not `log_density`, which raises `UnsupportedOpError` from the covariance:
a rank-deficient Gaussian has no density on $\mathbb{R}^P$, and the
capability system says so instead of returning `-inf` or `nan`.

:::{admonition} Prerequisite: `PSDLowRank` in `pyeki.linalg`
:class: note

This method needs one new elementary operator, sketched here and to be
specified in the operator contract when it is added. `PSDLowRank` holds a
single data field, a factor `F` of shape `(n, k)` with no relation imposed
between `n` and `k`, and represents $F F^\top$. It implements `matvec`
($F(F^\top x)$, two trailing-axis contractions), `diag` (rowwise
$\sum_j F_{ij}^2$), `to_dense` ($FF^\top$ assembled from the stored array,
never via `matvec`), and `factor` (wrapping `F` as a `Dense`); it
implements **no** `_solve`, `_whiten`, or `_logdet` — it is the operator
contract's singular-by-construction class made concrete, the static
class-level decision that contract already anticipates. Nothing is
computed at construction: the stored field *is* the factorization. Like
every operator, it must pass `check_operator` before merging.
:::

(gauss-joint)=
## `JointGaussian`

The operator-represented joint: means and covariance blocks held explicitly.
Its role is to be the **obviously correct reference** — exact moment
conditioning computed by dense linear algebra on materialized blocks,
against which the whitened-SVD path is tested — and the exact conditioner
for problems small enough to afford it. It is deliberately not a hot-path
object, and its one conditioning method is deliberately dense: the
reference must be a *different, simpler code path* than the whitened-SVD
path it checks, or the comparison is vacuous — the same principle as the operator
contract's `to_dense` independence rule.

**Fields.** `u_mean` `(P,)`, `v_mean` `(N,)`, `u_cov` a `PSDLinOp` of side
$P$, `cross_cov` a {class}`~pyeki.linalg.LinOp` of shape `(P, N)` — the
block $C_{uv}$, so *rows index $u$* — and `v_cov` a `PSDLinOp` of side $N$.
Construction validates ranks, operator types (`TypeError` for a non-PSD
marginal block or a non-operator, mirroring the composite rules of
{ref}`contract-composites`), and shape agreement across all five fields
(tier 2).

**Joint validity is a value precondition.** The types guarantee each
marginal block is PSD, but positive semi-definiteness of the *joint* —
$C_{vv} \succeq C_{vu} C_{uu}^{-1} C_{uv}$ — is a property of the values,
unverifiable from structure. It is the caller's responsibility, tier 4: in
debug mode ({ref}`contract-validation`), `condition` assembles the dense
joint covariance it has already materialized and checks its smallest
eigenvalue; outside debug mode a violation yields a posterior that is
simply wrong, silently, like every tier-4 violation.

### `condition(y, noise_cov, *, max_n=4096)`

Exact moment conditioning: returns the posterior over $u$ given the
observation $y = v + \eta$, $\eta \sim \mathcal{N}(0, R)$, as a
{class}`Gaussian`. With materialized blocks and $M = C_{vv} + R$ (dense):

$$
m_{\text{post}} = m_u + C_{uv} M^{-1} (y - m_v), \qquad
C_{\text{post}} = C_{uu} - C_{uv} M^{-1} C_{vu},
$$

computed by dense Cholesky factorization of $M$, the posterior covariance
symmetrized as $\tfrac{1}{2}(C + C^\top)$ against floating-point asymmetry
and returned as a {class}`~pyeki.linalg.DensePSD` (via `from_matrix`, so
debug mode's positive-definiteness check applies to it). Requirements:

- **Size guard before allocation**: `ValueError` when $\max(P, N)$ exceeds
  `max_n`, raised before any dense block is materialized. Raising the limit
  is a deliberate act at the call site, exactly as with
  {func}`~pyeki.linalg.densify`.
- `noise_cov` is used through `to_dense()` — always available — so *any*
  `PSDLinOp` works here, including ones with no `whiten`. The reference
  path and the whitened-SVD path deliberately have complementary
  requirements.
- Cost is $O\bigl((P+N)^3\bigr)$ and the method must not pretend otherwise:
  no structured shortcuts, no routing through `gain_weights`. Its value is
  its simplicity.
- Precondition: $M$ nonsingular (guaranteed when $R \succ 0$). A posterior
  covariance that is singular — $u$ fully determined by the observation —
  is legitimate as a distribution but fails inside
  `DensePSD.from_matrix` in the usual tier-4 way: `nan` silently, an error
  in debug mode.

(gauss-prng)=
## Randomness

- Every stochastic method takes a JAX PRNG `key` as its first argument and
  **consumes it whole**: the caller owns splitting. No method stores a key,
  advances a hidden state, or splits the key it was given for purposes
  beyond the one call.
- The two draws in the layer are **pinned by this contract**:
  `Gaussian.sample` draws `normal(key, (n_samples, k))` with `k` the factor
  width, and `pathwise_update` draws `normal(key, (J, N))`. Same key, same
  arguments, same representation ⇒ identical output arrays, across
  releases. Changing either draw is a breaking change and must be treated
  as one.
- Everything else is deterministic. There is exactly one source of
  randomness per stochastic call, which is what makes EKI runs resumable
  from a stored key — a requirement the EKI layer inherits and this layer
  must not undermine.

(gauss-validation)=
## Validation and errors

The four-tier scheme of the operator contract ({ref}`contract-validation`)
applies unchanged: everything static is checked always, values only on
request. What each tier means here:

| tier | checks | examples |
| ---- | ------ | -------- |
| 2. construction | ranks, static sizes, operator types, cross-field shape agreement | `u_samples` rank ≠ 2; $J = 1$; `cov` not a `PSDLinOp`; `cross_cov` shape disagreeing with the means |
| 3. call | operand core shapes and operator arguments | `y` not `(N,)`; `noise_cov` side ≠ $N$; `noise_cov` not a `PSDLinOp`; primitive operands mis-shaped; `n_samples` not a positive `int` |
| 4. value (debug) | finiteness of samples and means; joint PSD-ness in `JointGaussian.condition`; the operator layer's own checks via `from_matrix` | violations yield `nan` or a wrong posterior silently outside debug mode |

Tier-1 (field declaration) is inherited with the class machinery
({ref}`gauss-jax`). Error messages follow the operator contract's
obligations: name the object (its `repr`), the method, the expectation, and
the offending value's shape or type.

The layer defines **no new exception types**. `UnsupportedOpError` arises
only from the operator layer, propagated unmodified from the covariance
that lacks the operation — the gauss layer never catches it, never wraps
it, and never falls back to dense linear algebra on the caller's behalf.
The explicit escape hatch is the same one as everywhere:
`densify` the covariance, deliberately, at the call site.

| condition | raises |
| --------- | ------ |
| wrong rank / disagreeing shapes at construction | `ValueError` |
| non-operator or wrong-level covariance field | `TypeError` |
| `y`, `s`, or `b` core shape mismatch at call | `ValueError` |
| `noise_cov` not a `PSDLinOp` / wrong side | `TypeError` / `ValueError` |
| `n_samples` not a positive Python `int` | `TypeError` / `ValueError` |
| covariance lacking `factor` / `whiten` / `logdet` where required | `UnsupportedOpError`, from the operator layer |
| `condition` size guard | `ValueError`, before allocation |
| violated value precondition | `ValueError` in debug mode; `nan` or a silently wrong result otherwise |

(gauss-jax)=
## JAX integration

The three classes are frozen-dataclass pytrees declared with the same
machinery as operators, and every rule of the operator contract's JAX
section ({ref}`contract-jax`) binds them:

- **Field classification is the same allowlist**: a field is pytree data iff
  its annotation is `Array` or a `LinOp` subtype; everything else is
  `static_field()` or a definition-time `TypeError`. (No gauss class stores
  another gauss class, so the allowlist needs no extension.)
- **Unflattening bypasses the constructor**, so tier-2 validation is strict
  at genuine construction and absent at trace boundaries; `vmap`-ed
  families reconstruct exactly as operator families do
  ({ref}`contract-batching-operators`), and the same conformance round-trip
  applies.
- **Identity semantics**: `eq=False`, hash by identity, never
  `static_argnums` — a joint is always a traced argument.
- **Constructors store and validate; classmethods compute.** Nothing in the
  layer computes at construction today ({ref}`gauss-ensemble` records why
  the SVD cannot live there); the rule binds any future classmethod that
  does.
- **`pyeki.gauss` exports no class decorator.** The class set is closed
  (rule 2 of {ref}`gauss-objects`), so there is nothing for users to
  declare; internally the classes may reuse the operator layer's
  registration machinery, generalized as needed, without exposing it.
- Conditioning methods and primitives must be `jit`- and `vmap`-safe with
  no data-dependent shapes, and `log_density` must be differentiable with
  respect to `x` and the array leaves (hyperparameter estimation
  differentiates through it; the updates carry no such requirement).

(gauss-consumers)=
## How the layers above consume this one

Not normative for `pyeki.gauss` itself, but the design was shaped against
these call sites, and a change that breaks them is a change to reconsider.

**The EKI driver** (`pyeki.eki`, planned) is a loop over tempering steps.
Per step: build the joint from the current ensemble, update, re-evaluate
the forward model.

```python
prior = Gaussian(m0, C0)                    # C0: any PSDLinOp with factor
u = prior.sample(key_init, n_members)

for key_t, dbeta in schedule:               # increments, not levels
    v = forward(u)                          # caller's vmap-ed model: (J, N)
    joint = EnsembleJoint(u, v)
    u = joint.pathwise_update(key_t, y, noise_cov / dbeta)
```

The tempered noise `noise_cov / dbeta` is the operator layer's scalar
scaling — a traced increment flows through the 0-d scalar field, so the
adaptive schedule never re-factorizes the noise. The deterministic variant
is the same loop with `transform_update(y, noise_cov / dbeta)` and no key.

**Domain localization** (`pyeki.localize`, planned) runs one small analysis
per parameter block against its nearby observations, with per-observation
noise inflation by the reciprocal taper. It consumes the conditioning
primitives under `vmap`, not the classes: per block, slice the local
prediction anomalies and residuals, build the local noise covariance as
`diag_congruence(noise_cov_block, 1/sqrt(taper))`, whiten, and call
`gain_weights` — obtaining that block's weight vector in $\mathbb{R}^J$ to
apply to its own $u$-anomalies. The primitives' array-purity, their
one-SVD-per-call rule, and the block anatomy that
{ref}`contract-composites` makes contractual are what this plan relies on.

**Tests and diagnostics** consume `JointGaussian`: on a linear-Gaussian
problem the empirical moments of `transform_update`'s output must
reproduce `JointGaussian.condition`'s posterior exactly, and a tempered run's
posterior must telescope to the one-shot posterior. The telescoping test
itself belongs to the EKI layer; the per-step exactness it composes from
lives here.

(gauss-repr)=
## `repr`

Type name and static sizes, never array contents, matching the operator
rule ({ref}`contract-repr`): `Gaussian(n=12)`,
`EnsembleJoint(n_members=100, u_dim=12, v_dim=40)`,
`JointGaussian(u_dim=12, v_dim=40)`. `JointGaussian` may append its blocks'
type names but never recurses into their arrays.

(gauss-surface)=
## Public surface

`pyeki.gauss` exports exactly: the classes `Gaussian`, `EnsembleJoint`,
`JointGaussian`, and the conditioning primitives `gain_weights` and
`sqrt_transform`.
Anything else is private, and no consumer may depend on it. There is no
`pyeki.gauss.testing`: the conformance obligations below bind the package's
own test suite, since the class set is closed.

(gauss-conformance)=
## Conformance

The layer has no user-extension point, so conformance is not a public
harness but a set of obligations on `tests/`. Exactness tests check against
closed forms and independent code paths, not tolerances chosen to pass
({doc}`linop-contract` sets the convention). The suite must verify at
least:

1. **Gain against dense**: `gain_weights` composed with `whiten` reproduces
   the dense $K r$ elementwise on small random problems, in all three shape
   regimes $N > J$, $N = J$, $N < J$, with `b` at batch ranks 0, 1, and 2.
2. **Whitener invariance**: two noise operators representing the same $R$
   with different whiteners yield identical weights.
3. **Transform against dense**: `sqrt_transform` matches a dense
   `eigh`-based $(I + ss^\top)^{-1/2}$ — including the rank-deficient case
   $r < J$ that the naive thin-SVD formula gets wrong — and satisfies
   $T = T^\top$ and $T\mathbf{1} = \mathbf{1}$.
4. **Moment exactness of the ensemble posterior**: `transform_update`'s
   output has sample mean and covariance equal to the dense posterior
   moments of the fitted joint Gaussian, to floating-point tolerance;
   `condition`'s mean and the dense form of its low-rank covariance equal
   the same moments; and the two methods satisfy the elementwise identity
   $u_j' = m_{\text{post}} + \sqrt{J-1}\, F_{\cdot j}$ of
   {ref}`gauss-ensemble`.
5. **Pathwise against dense, elementwise**: with $\varepsilon$ recomputed
   from the same key (the pinned draw), `pathwise_update` equals
   $u_j + K(y - v_j - W^{-1}\varepsilon_j)$ computed densely, with $W$
   recovered by whitening the columns of $I_N$.
6. **Cross-representation agreement**: a `JointGaussian` built from an
   ensemble's densified empirical moments conditions to the same posterior
   that `EnsembleJoint.condition` returns and `transform_update`
   represents.
7. **Reference independence**: `JointGaussian.condition` is tested against
   hand-written dense Bayes formulas, not against the whitened-SVD path —
   and must not route through `gain_weights` or `sqrt_transform`, so that
   check 6 compares two genuinely independent paths.
8. **Marginal formulas**: `sample` matches its pinned elementwise
   definition; `log_density` matches the dense closed form at batch ranks
   0, 1, and 2 and differentiates.
9. **Degeneracy**: zero prediction anomalies make both updates the
   identity on `u_samples`; $J = 2$ and $N = 1$ work; a collapsed ensemble
   produces no `nan`.
10. **Capability propagation**: a noise covariance without `whiten`, and a
    covariance without `factor` or `logdet`, raise `UnsupportedOpError`
    from the update, `sample`, and `log_density` respectively — and
    `log_density` on the singular posterior `EnsembleJoint.condition`
    returns raises the same way.
11. **Validation**: every tier-2 and tier-3 rule of
    {ref}`gauss-validation` raises as specified, including the size guard
    firing before allocation.
12. **JAX round trips**: flatten/unflatten preserves type and behaviour for
    all three classes; updates run under `jit`; constructing an
    `EnsembleJoint` inside `vmap` round-trips and a `vmap`-ed family of
    joints agrees with a Python loop; sentinel-leaf unflattening succeeds.
13. **Reproducibility and repr**: same key, same output, elementwise;
    different keys differ; reprs match {ref}`gauss-repr` with no array
    data.

Alongside conformance, targeted regression tests guard the layer's own
silent-failure classes once found — the thin-SVD completion term (check 3),
a mixed-representation perturbation, a mean shift from an uncentred
transform — under the same do-not-delete rule as the operator layer's.

(gauss-excluded)=
## Deliberately excluded

Recorded so their absence reads as a decision, not an oversight.

**A structured posterior from `JointGaussian.condition`.** The ensemble
path returns structured posteriors (`EnsembleJoint.condition`); the
operator path deliberately does not. Its posterior covariance
$C_{uu} - C_{uv} M^{-1} C_{vu}$ is a low-rank *downdate*, and an operator
for one — `PSDDowndate(base, F)` representing $\mathrm{base} - FF^\top$ —
has a lopsided capability set: `matvec` and `diag` are immediate, `solve`
and `logdet` follow from the Woodbury identity given `base.solve` and
`base.logdet` plus one $k \times k$ factorization at construction, but
`factor` and `whiten` require writing $F = LG$ against the base's own
factor $L$, which only some base types can solve for — colliding with the
rule that support is static. More decisively, `condition`'s whole role is
to be the dense oracle ({ref}`gauss-joint`), and check 7 of
{ref}`gauss-conformance` depends on its independence. If a consumer ever
needs large exact operator-path posteriors, the shape of the addition is
`PSDDowndate` in `pyeki.linalg` first, then a *new* structured
conditioning method here — with `condition` staying dense.

**Joint sampling and pathwise conditioning for `JointGaussian`.** Matheron-style
posterior sampling — draw from the joint, condition each draw — requires a
*coherent factor pair* $(F_u, F_v)$ sharing one latent vector, which the
block fields $(C_{uu}, C_{uv}, C_{vv})$ cannot supply: consistency of the
cross block with the factors is a value property no constructor can check.
EKI does not need it (the ensemble *is* the joint sample), so it waits for
a real consumer, likely arriving together with a factor-pair construction
such as a linear-map classmethod.

**A linear-map constructor** (`from_linear_map(prior, H)`). Convenient, but
its natural output — `v_cov = product(H, C, H.T)` — cannot be *typed* PSD
under the operator layer's rule that PSD-ness is never inferred through
composition. Callers assemble the blocks explicitly (for the reference
role, three lines of dense algebra); revisit alongside the previous item.

**A reified gain object.** Computing the SVD once and reusing it across
conditioning methods on the same `(joint, noise_cov)` pair would need a
returned decomposition object. Each EKI step calls exactly one method once,
so today it would be surface without a consumer; the conditioning
primitives already serve anyone assembling custom flows.

**Regime selection.** The whitened-SVD kernel is correct in every shape
regime, including $N \le J$ where a dense factorization of
$\widehat{C}_{vv} + R$ would also be affordable. Selecting between them is
an optimization to add behind the same method signatures if profiles ever
demand it, not an API decision.

**Batched observations.** `y` is one observation vector. A family of
updates — multiple observations, multiple noise levels — is `jax.vmap`
over the method, consistent with the layer-wide batching story.

**A configurable anomaly divisor.** $J - 1$ everywhere. A $1/J$ convention
changes every formula's scaling for no consumer; inflation, which is the
principled way to widen an ensemble, belongs to `pyeki.eki`.

**Perturbation injection.** `pathwise_update` takes no `eps` argument; see
the warning in {ref}`gauss-ensemble`. Determinism needs are met by the
pinned key-derived draw.

**Non-Gaussian anything.** No mixtures, no transformations of variables,
no likelihoods other than additive Gaussian noise. The layer is the
Gaussian core of EKI, not an inference framework.

**Mean-only conditioning.** `JointGaussian.condition` always computes the
posterior covariance. A mean-only fast path saves one dense solve on an
object whose whole role is small exact reference computations; not worth a
second signature.
