# Joint Gaussian contract

This page specifies `pyeki.gauss`: the classes it provides, the contract of
every method, and the conditioning mathematics that all of them share. It is
normative — an implementation that violates a rule here is defective even if
its tests pass — and it is the reference for two audiences: contributors
implementing or reviewing the layer, and users who want a more precise
account of Gaussian conditioning in pyEKI than the user guide gives.

Throughout, *must* and *never* state requirements, *should* states a strong
default that a documented reason may override, and *may* states a permission.
{doc}`design` records *why* the load-bearing decisions were made; this page
records *what* they require. The layer is built on the operator layer, and
this contract freely references {doc}`linop-contract` rather than restating
its rules.

:::{admonition} Status: implemented
:class: note

`pyeki.gauss` implements this specification, and this page is the normative
reference for its behaviour. The conformance obligations of
{ref}`gauss-conformance` are met by `tests/test_gauss.py`; the user guide's
{doc}`user-guide/conditioning` page covers when to reach for each piece.
:::

Readers after precision rather than implementation want
{ref}`gauss-notation`, {ref}`gauss-kernel`, and the method sections; the
remainder — validation, JAX integration, conformance, exclusions — binds
implementers.

(gauss-scope)=
## Scope

The layer represents joint Gaussian distributions over a pair of blocks — in
practice, parameters and predicted observations — and conditions them on a noisy
observation of the second block. It provides exactly what Ensemble Kalman
Inversion needs from Gaussian machinery:

- **sampling** from a marginal Gaussian whose covariance is a structured
  operator (drawing an initial set of samples from a prior);
- **the update**, in both its stochastic (perturbed-observation)
  and deterministic (square-root transform) forms — Gaussian conditioning
  applied to the Gaussian determined by paired samples' empirical moments —
  together with that posterior as a distribution, in structured low-rank form;
- **the array-level conditioning primitives** underneath both, exposed at the
  granularity that domain localization consumes.

There is deliberately no packaged "exact" joint alongside the empirical one:
the dense reference that everything is tested against is hand-written in the
test suite ({ref}`gauss-conformance`), and an operator-represented joint
class waits for a consumer ({ref}`gauss-excluded`). Nor is the layer a
probabilistic-programming toolkit: there are no densities over anything
non-Gaussian, no transformations of distributions, and no inference
machinery.

Everything numerical routes through one algorithm, the **whitened-SVD
conditioning kernel** ({ref}`gauss-kernel`) — *kernel* in the computational
sense, the shared numerical core of every conditioning routine; nothing in
this package uses the word to mean a covariance kernel. The algebraically
equivalent alternatives — the Woodbury identity on the normal equations, and
a dense factorization of the predictive covariance — are respectively
rejected and deferred, with reasons, in {ref}`gauss-excluded`.

The implementation PR must also add the layer's user-guide page, per the
package rule that every user-facing feature has a user-guide home; this
page remains the deeper reference.

(gauss-notation)=
## Notation and conventions

One convention set governs the whole layer. Symbols used throughout:

| symbol | meaning |
| ------ | ------- |
| $u$    | the block to be updated, dimension $P$ |
| $v$    | the block that is observed, dimension $N$ |
| $y$    | the observation, a vector of length $N$ |
| $R$    | the observation-noise covariance, an $N \times N$ PSD operator |
| $W$    | a whitener: the fixed matrix an operator's `whiten` applies, $W C W^\top = I$ for that operator's $C$; in conditioning, of $R$ |
| $J$    | the number of samples |
| $r$    | a residual $y - v$, of length $N$ (the thin SVD's width is written $\rho$) |

Conventions, each normative:

- **Samples are stored row-wise: a block of samples is a `(J, dim)` array**, one
  sample per row. This is what `vmap` produces and what
  the operator layer's batch contract ({ref}`contract-batch`) treats as a
  batch of vectors, so sample blocks flow between the two layers with no
  transposes. Displayed mathematics in this page follows the same
  convention: the sample matrices $\mathsf{U} \in \mathbb{R}^{J \times P}$
  and $\mathsf{V} \in \mathbb{R}^{J \times N}$ have one sample per row.
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
  ({ref}`gauss-jax`). The exceptions are the two conditioning
  *primitives* — array-level, following the operator layer's batch
  contract ({ref}`gauss-primitives`) — and the evaluation-point argument
  of `Gaussian.log_density`, which is batched the same way.
- **Noise covariances are `PSDLinOp`s.** Every `noise_cov` argument must be
  a {class}`~pyeki.linalg.PSDLinOp` of side $N$; a non-operator or an
  operator of the wrong shape is a `TypeError` / `ValueError` at call time.
  The conditioning methods use it only through `whiten` — by design, the one
  operation the noise covariance must support ({doc}`design`) — so a noise
  operator with no factorization at all still drives every update, and
  a scaled covariance such as `noise_cov / dbeta` (a
  {class}`~pyeki.linalg.PSDScaled`) whitens as cheaply as the base operator.

(gauss-objects)=
## The objects

The public surface is two classes and two functions.

| object | represents | representation | where it is used |
| ------ | ---------- | -------------- | ---------------- |
| `Gaussian` | one Gaussian distribution | mean vector + `PSDLinOp` covariance | a prior or marginal; what `condition` returns; a sample block's moments, via `from_samples` |
| `EmpiricalJoint` | the joint Gaussian with paired samples' empirical moments | $J$ paired samples | every conditioning operation |
| `gain_weights` | the sample weights for one whitened residual | pure array function | the shared conditioning core |
| `sqrt_transform` | the square-root update transform | pure array function | the shared conditioning core |

Three rules govern the set:

1. **One joint abstraction.** `EmpiricalJoint` stores samples and *acts as*
   the Gaussian obtained by matching moments to them; every conditioning
   path — the posterior as a distribution, the deterministic transform, the
   stochastic pathwise update — is a method on it ({ref}`gauss-empirical`).
   There is no second, operator-represented joint class: the one
   load-bearing role such a class had — being the reference implementation —
   belongs to the test suite by design, precisely so that no package code
   has to be trusted ({ref}`gauss-conformance`), and packaging it would add
   surface without a consumer ({ref}`gauss-excluded`).
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
class methods ({ref}`gauss-empirical`) are thin assemblies of these pieces;
the two public functions ({ref}`gauss-primitives`) expose them directly.
"Thin assembly" names the mathematics, not an obligation to call the
public functions: the class methods share a single SVD of $S$ internally,
while each public primitive recomputes its own — routing a class method
through the primitives is permitted only where it preserves the
one-SVD-per-call rule of {ref}`gauss-empirical`.

### The whitened anomaly matrix

Given prediction anomalies $A_v$ and a noise operator with whitener $W$,
define the **scaled whitened anomaly matrix**

$$
S \;=\; \frac{1}{\sqrt{J-1}}\, A_v W^\top \;\in\; \mathbb{R}^{J \times N},
$$

whose $j$-th row is $W a_j / \sqrt{J-1}$ — in code, exactly
`noise_cov.whiten(A_v) / sqrt(J - 1)`, one call under the operator layer's
batch contract. Let

$$
S = U \Sigma V^\top \quad \text{(thin SVD)}, \qquad
U \in \mathbb{R}^{J \times \rho},\;
V \in \mathbb{R}^{N \times \rho},\;
\rho = \min(J, N),
$$

with singular values $\sigma_1 \ge \dots \ge \sigma_\rho \ge 0$. Because the
sample rows of $A_v$ sum to zero ($\mathbf{1}^\top A_v = 0$),
$S^\top \mathbf{1} = 0$: the all-ones direction is in the null space of
$S^\top$, so every $\sigma_i > 0$ has $U_{\cdot i} \perp \mathbf{1}$.
Mean-centering also caps the rank at $J - 1$, so at least one singular
value is exactly zero whenever $N \ge J$ — in exact arithmetic; floating
point returns $O(\varepsilon \lVert S \rVert)$ instead. Every formula
below is continuous at $\sigma = 0$ in value and needs no special-casing;
differentiability is another matter ({ref}`gauss-primitives`).

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
combination $A_u^\top w / \sqrt{J-1}$ of the samples' own anomalies, so
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
  ill-conditioned the fitted Gaussian becomes, with no regularization parameter to
  tune.

The SVD form is the normative implementation: it never forms
$S^\top S$ or $S S^\top$, whose condition numbers are the *squares* of
$S$'s. The algebraically equivalent routes that do form them are excluded
({ref}`gauss-excluded`); {doc}`design` gives the full comparison with the
normal-equations route.

### The square-root transform

The deterministic update replaces sampling noise with an exact transform of
the anomalies. Define

$$
T \;=\; (I_J + S S^\top)^{-1/2}
\;=\; I_J + U\bigl((I_\rho+\Sigma^2)^{-1/2} - I_\rho\bigr)U^\top
\;\in\; \mathbb{R}^{J \times J},
$$

symmetric, built from the same SVD. The second form is the normative one:
for a thin SVD the naive $U(I+\Sigma^2)^{-1/2}U^\top$ *omits the identity on
the orthogonal complement* and is simply wrong whenever $\rho < J$ — the
$I_J + U(\cdot - I)U^\top$ form is exact for every rank.

Transformed anomalies $T A_u$ have empirical covariance exactly equal to the
posterior covariance of the fitted joint Gaussian:

$$
\frac{(T A_u)^\top (T A_u)}{J-1}
\;=\; \widehat{C}_{uu} - K\,\widehat{C}_{vu}.
$$

Two structural facts, both load-bearing and both conformance-checked:

- **The transform preserves mean-centering**: $T\mathbf{1} = \mathbf{1}$
  (each $U_{\cdot i}$ with $\sigma_i > 0$ is orthogonal to $\mathbf{1}$, and
  the modifier vanishes at $\sigma_i = 0$), so transformed anomalies still
  sum to zero and the posterior mean is not silently shifted.
  In floating point the numerically-zero $\sigma$'s $U$ column need not be
  orthogonal to $\mathbf{1}$; the property survives because the modifier
  decays *quadratically*: the induced mean shift is
  $O\bigl((\varepsilon\sigma_{\max})^2\bigr)$ rather than
  $O(\varepsilon\sigma_{\max})$ — the modifier is exactly $0.0$ while
  $\varepsilon\sigma_{\max} \lesssim 10^{-8}$, and negligible above it.
- **The identity is exact in exact arithmetic**, not asymptotic in $J$. The
  conformance suite checks it to floating-point tolerance
  ({ref}`gauss-conformance`).

### Cost

Whitening costs $J + 1$ applications of $W$ — every prediction and the
observation, from one call on the stacked rows $[\mathsf{V}; y]$. Whitening is
a fixed linear map applied row-wise, so it commutes with centering and with
subtraction: $W A_v^\top$ is the whitened predictions minus *their* mean, and
$W(y - v_j)$ is $Wy - Wv_j$. The grouping is **not** free, however: centering
and differencing must happen *before* the whitener is applied. The two orders
agree in exact arithmetic but are not equally stable — centering *whitened*
predictions makes the cancellation ratio
$\lVert W\bar v\rVert / \lVert W a_j\rVert$ in place of
$\lVert \bar v\rVert / \lVert a_j\rVert$, so the error grows with
$\kappa(W) = \sqrt{\kappa(R)}$ whenever the prediction mean is aligned
with a precise direction of the noise. Measured against an exact reference at
$\kappa(R) = 10^4$ with a prediction mean of $10^8$ along $R$'s most precise
direction, whitening first gives a posterior-mean error of $4.9$ where
centering first gives $2\times10^{-6}$. Whitening $[A_v;\, y-\bar v]$ costs the
same $J+1$ applications and does not have that failure mode. What must **not**
be done is whitening the anomalies and the residuals in two separate calls,
which costs $2J$ applications in the stochastic update — twice the necessary
figure, and for a dense whitener on the dominant term. The thin SVD is
$O(J N \min(J, N))$; forming weights is $O((N + J)\,\rho)$ per residual and
combining anomalies $O(JP)$ per sample. A full update is
$O(NJ^2 + PJ^2)$ for $J \le N$, plus the $J + 1$ whitener applications —
which the total absorbs for structured whiteners applying in $O(N)$, and which
dominate ($O(JN^2)$) for a dense $W$ whenever $P \lesssim N^2/J$; at larger $P$
the $O(PJ^2)$ anomaly combination dominates instead. Linear in both dimensions
for structured whiteners, cubic only in the number of samples. The conditioning
that matters degrades in the *small-noise* direction (large
$\sigma_{\max}$, a large whitener), not the collapse direction
($\sigma \to 0$), which is numerically pristine.

(gauss-primitives)=
## Conditioning primitives

The two pieces above are public functions in `pyeki.gauss`, defined as pure
matrix functions of their array arguments — no divisor, no whitening, and no
randomness folded in. They exist as the layer's *advanced tier*: the class
methods cover the global update, and domain localization (the planned
`pyeki.localize`) calls these directly, once per local analysis, under
`jax.vmap` over fixed-size neighbourhoods.

| function | signature | returns |
| -------- | --------- | ------- |
| `gain_weights(s, b)`   | `(J, N), (..., N) -> (..., J)` | $U \operatorname{diag}\bigl(\sigma_i/(1+\sigma_i^2)\bigr) V^\top b$ for the thin SVD $s = U\Sigma V^\top$ |
| `sqrt_transform(s)`    | `(J, N) -> (J, J)`             | $(I_J + s s^\top)^{-1/2}$ |

Rules:

- **`s` is exactly 2-D, with both sizes at least 1** (`ValueError`
  otherwise; there is no lower bound of 2 on $J$ — the divisor lives in
  the caller). It plays the operator's role, so it carries no batch axes;
  a family of local analyses is a `vmap` over the function. `b` follows
  the operator layer's batch contract: trailing core axis of length $N$,
  any number of leading batch axes carried through. Violations raise
  `ValueError` with the same message obligations as tier-3 operand checks
  ({ref}`contract-validation`), the function's name standing in for the
  operator repr.
- **Both functions compute one SVD per call.** Batch the residuals into a
  single `gain_weights` call rather than looping; the $J$ per-sample
  residuals of an update are one `(J, N)` operand.
- The functions are deterministic JAX code, safe under `jit` and `vmap`,
  with no data-dependent shapes. They are differentiable wherever the
  singular values of `s` are distinct and nonzero. At *exactly* repeated
  or exactly zero singular values — an exactly collapsed `s`, or the
  zero-padded columns a masked local analysis may produce when the masking
  drops the rank below $\min(J, N)$ — the SVD's
  gradient is `nan`, even though the functions themselves are smooth
  there — `gain_weights` equals the rational $s(s^\top s + I)^{-1}b$, and
  `sqrt_transform` is real-analytic, the spectrum of $I + ss^\top$ being
  bounded below by 1. The
  float-generic degeneracy of mean-centering ($\sigma_{\min} \sim
  10^{-16}$ when $N \ge J$) is not an exact tie and differentiates
  finitely. No conditioning path in this layer *requires* differentiation
  with respect to `s` ({ref}`gauss-jax`) — a caller who differentiates an
  update with respect to `v_samples` does differentiate through the SVD,
  and inherits these cases. An implementation may restore gradients
  everywhere with a custom JVP routed through the closed forms, but is not
  required to; note that for `sqrt_transform` that means a Fréchet
  derivative of $A \mapsto A^{-1/2}$, materially more work than
  `gain_weights`'s rational form.
- `sqrt_transform` imposes no centering requirement on `s`. The
  $T\mathbf{1} = \mathbf{1}$ property of {ref}`gauss-kernel` follows from
  $\mathbf{1}^\top s = 0$ and holds only for such `s`; on general `s` the
  transform is still $(I + ss^\top)^{-1/2}$, and $T\mathbf{1}$ is
  whatever that matrix makes it.
- Both return values are functions of `s` alone, invariant to the SVD's
  sign and degenerate-rotation freedom — they equal the closed forms of
  {ref}`gauss-kernel` — so any correct thin SVD gives the same output
  elementwise.
- Callers own the semantics of `s`. For conditioning, `s` must be the
  *scaled, whitened* anomaly matrix of {ref}`gauss-kernel`, and `b` a
  *whitened* residual; the functions cannot check this, which is why the
  class methods — where the conventions are enforced — are the default
  interface and these functions are the escape hatch.

(gauss-marginal)=
## `Gaussian`

A single Gaussian distribution $\mathcal{N}(m, C)$: the prior a caller
supplies, and the posterior that `EmpiricalJoint.condition` returns
({ref}`gauss-empirical`).

**Fields.** `mean`, a `(n,)` array, and `cov`, a
{class}`~pyeki.linalg.PSDLinOp` of side `n`. Construction validates the rank
of `mean`, the operator type of `cov`, and their agreement
(tier 2, shape-only — {ref}`gauss-validation`). `n` is a property computed from `mean`.

The size properties across the layer follow one rule: an object with a
single dimension names it `n`, matching {class}`~pyeki.linalg.SquareLinOp`
one layer down; an object with several qualifies each one, as
`EmpiricalJoint`'s `n_samples`, `u_dim` and `v_dim` do. So `Gaussian.n` and
`EmpiricalJoint.u_dim` are the same convention applied to different arities,
not an inconsistency.

**Capabilities delegate to the covariance.** `Gaussian` defines no
capability system of its own: each method requires specific operations of
`cov`, and an unsupported one raises the operator layer's
`UnsupportedOpError`, unmodified, from the inner call. `gaussian.cov`
is a public field, so callers gate exactly as they do on operators:
`gaussian.cov.supports("factor")`.

### `Gaussian.from_samples(samples)`

The Gaussian fit to a `(J, n)` array of samples, $J \ge 2$: mean the sample
mean, covariance the empirical covariance with this layer's fixed $J-1$
divisor, held as a {class}`~pyeki.linalg.PSDLowRank` whose factor is
$A^\top/\sqrt{J-1}$. A classmethod rather than logic in the constructor, per
{ref}`contract-jax`'s rule that constructors store and classmethods compute.

It is the **one-block counterpart of {class}`EmpiricalJoint`**, which fits a
joint to two row-aligned blocks; the two agree on the $u$ block by
construction, and the conformance suite pins that. Its purpose is to let a
caller read a block of samples' moments as a distribution: `cov.diag()` gives the
per-coordinate variances, and `sample` draws from the fit.

The covariance is never formed as an $n \times n$ matrix — the stored factor
*is* the empirical covariance, at $O(nJ)$ rather than $O(n^2)$. Its rank is at
most $J-1$, so it is singular whenever $J - 1 < n$, which is the usual sample
block regime; `PSDLowRank` accordingly provides `diag` and `factor` and
withholds `solve`, `whiten` and `logdet`, so {meth}`log_density` raises
`UnsupportedOpError` on the result. That is correct rather than restrictive: a
density against a singular covariance is not defined.

Anomalies are formed with the same centring the conditioning methods use
({ref}`gauss-kernel`), so identical samples give exactly zero spread rather
than round-off.

:::{note}
This is a **fit**, not a conditioning result, and the distinction is the
caller's to keep. `pyeki.eki` uses it to report the moments of a terminal
ensemble, and states there what such an ensemble does and does not represent;
nothing here licenses calling the result a posterior.
:::

### `sample(key, n_samples)`

Returns a `(n_samples, n)` array of independent draws. Requires
`cov.supports("factor")`.

- `n_samples` must be a Python `int`, at least 1 — it determines an output
  shape, so it can never be traced. Anything that is not an `int` —
  including `bool` and NumPy integers — is a `TypeError`; an `int` below 1
  is a `ValueError`. Since `bool` *is* an `int` subclass, the rule is a check
  on the exact type, not an `isinstance` test; it consequently also rejects
  other `int` subclasses, which is intended.
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
0-d real JAX scalar (never a Python float). `x` is a tier-3 operand: its
trailing axis must be exactly `n` and its rank at least 1, checked before
anything is computed — without that check a shorter `x` broadcasts and
returns a finite, plausible, wrong number. Requires
`cov.supports("whiten")` and `cov.supports("logdet")`, checked in that
order. The value is

$$
\log p(x) \;=\; -\tfrac{1}{2}\Bigl(\, n\log 2\pi \;+\; \log\det C \;+\;
\lVert W(x - m)\rVert^2 \,\Bigr),
$$

correct for every valid whitener by the invariant
$\lVert W r \rVert^2 = r^\top C^{-1} r$ that the operator contract
guarantees. Precondition: $C$ nonsingular — a singular covariance yields
`nan`/`inf` downstream of the operator layer, per its tier-4 convention.

(gauss-empirical)=
## `EmpiricalJoint`

The joint Gaussian determined by paired samples' empirical moments, held in
sample form: built for one conditioning operation, used once, and
discards. Its distribution is
$\mathcal{N}\!\left(\begin{pmatrix}\bar u\\ \bar v\end{pmatrix},
\begin{pmatrix}\widehat{C}_{uu} & \widehat{C}_{uv}\\
\widehat{C}_{vu} & \widehat{C}_{vv}\end{pmatrix}\right)$ — a Gaussian *fit
to* the samples, not the empirical (equal-weight point-mass) distribution
of the samples themselves: every conditioning method below is exact
Gaussian conditioning applied to this fitted Gaussian. The samples are the
*representation* from which its moments are read, and no moment matrix is
ever formed.

**Fields.** `u_samples`, a `(J, P)` array, and `v_samples`, a `(J, N)`
array — row-aligned: row $j$ of each belongs to the same sample. Both are
**keyword-only**: they are arrays of the same rank agreeing on the sample
axis, so exchanging them is shape-valid whenever $P = N$ and no check can
detect it — the update is then computed from the wrong blocks and returns
finite, plausible numbers. The cost is that a family is built through a
lambda, `jax.vmap(lambda u, v: EmpiricalJoint(u_samples=u, v_samples=v))`,
rather than by mapping the constructor directly. `Gaussian` stays
positional: an array and an operator cannot be exchanged silently, since
the type check catches it.
Construction validates exact rank 2 on both, agreement of the leading axes,
and $J \ge 2$ (a single sample has no anomalies; the check is shape-only,
so it is tier 2 and unconditional). $P \ge 1$ and $N \ge 1$, as everywhere.

**Derived attributes.** `n_samples`, `u_dim`, `v_dim` are `int` properties.
`u_mean`, `v_mean`, `u_anomalies`, `v_anomalies` are array properties
computed on access from the stored samples. This is not the forbidden lazy
factorization ({ref}`contract-jax`): nothing is cached and nothing
factorizes — each access is an $O(J\cdot\mathrm{dim})$ mean-and-subtract
that fuses under `jit`. The samples are the whole state, and
`(mean, anomalies)` is recovered from them exactly.

**No factorization at construction — deliberately.** The operator layer's
rule is *factorize at construction*; here there is nothing to factorize:
the SVD of $S$ depends on the noise operator, which arrives per update
call and may differ on every call. Each update method computes its SVD
once, uses it, and discards it — never caching it on the instance, which
under `jit` would be the discarded-cache bug the operator contract
documents. One method call, one SVD.

All three conditioning methods condition on the same observation model —
$y = v + \eta$ with $\eta \sim \mathcal{N}(0, R)$ — and share two trailing
arguments: `y`, the observation, an array of rank exactly 1 and length $N$
(unlike operator operands, no batch axes — a family of updates is
`jax.vmap` over the method), and `noise_cov`, a `PSDLinOp` of side $N$
supporting `whiten` (`UnsupportedOpError` from the inner call otherwise).
Precondition: `noise_cov` is nonsingular — `whiten`'s own precondition.
The whitened formulation requires it even though $\widehat{C}_{vv} + R$
is generically invertible for singular $R$ (it fails only when
$\operatorname{nullity}(R) + \operatorname{nullity}(\widehat{C}_{vv}) > N$);
a singular noise operator that nonetheless
types as whitening-capable yields `nan`, or the tier-4 result check of
{ref}`gauss-validation` in debug mode. The two update methods
return a `(J, P)` array of updated samples, row $j$ updating sample $j$ —
they update $u$ only, leaving a caller that needs a matching $v$ to get the
next step's $v$ — while `condition` returns the same posterior as a
distribution. All are deterministic functions of their arguments
(`pathwise_update` includes the key among them), safe under `jit`, and all
degrade gracefully when the prediction anomalies are zero: the updates
return `u_samples` unchanged and `condition` returns the prior marginal's
moments — a collapsed sample block is a no-op, not `nan`, for finite inputs.
This requires the anomalies of identical samples to be *exactly* zero,
which a plain subtraction of a summed-and-divided mean does not deliver;
the anomaly properties are formed so that they do, because the alternative
is not a `nan` but a wrong finite update, of order 1 for samples of order
$10^{23}$. Whitening can still overflow for finite inputs — a prediction of
$10^{300}$ against a noise variance of $10^{-20}$ — and there the result is
`nan`.

### `pathwise_update(key, y, noise_cov)`

The stochastic (perturbed-observation) update: pathwise conditioning of
the joint's own samples, one fresh perturbation per sample. The result is

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
fitted joint Gaussian, returned in sample form. The result is

$$
u_j' \;=\; \underbrace{\bar u + \frac{1}{\sqrt{J-1}}\, A_u^\top\,
\texttt{gain\_weights}\bigl(S,\, W(y - \bar v)\bigr)}_{\text{posterior mean}}
\;+\; \bigl(T A_u\bigr)_j, \qquad T = \texttt{sqrt\_transform}(S).
$$

No randomness, no `key`. The returned block is an *exact* sample
representation of the moment posterior: its sample mean equals the
posterior mean and its sample covariance (divisor $J-1$) equals the
posterior covariance, both in exact arithmetic — the identity of
{ref}`gauss-kernel`, and the bridge the conformance suite uses between the
update and the hand-written dense reference. Because
$T\mathbf{1} = \mathbf{1}$, the transformed anomalies remain centered, so
the two summands above really are the posterior mean and posterior
anomalies.

`pathwise_update`'s output has sample
mean and sample covariance (divisor $J-1$) that are *unbiased estimators*
of the same posterior moments. The sample mean has variance exactly
$K R K^\top / J$; the sample covariance fluctuates at the usual
$O(J^{-1/2})$ rate, with entrywise standard deviation of order
$\bigl(C^{\text{post}}_{ii}(KRK^\top)_{jj} +
C^{\text{post}}_{jj}(KRK^\top)_{ii} +
2(KRK^\top)_{ij}^2\bigr)^{1/2}\!/\sqrt{J}$ — in relative terms
$\sqrt{J}$ times looser than the mean's. The unbiasedness of the
covariance is particular to the $J-1$ divisor, whose centering of the
perturbations cancels exactly.
Individual pathwise samples are not posterior draws — conditional on the
sample block, sample $j$ is distributed
$\mathcal{N}\bigl(u_j + K(y - v_j),\, K R K^\top\bigr)$ — so no
"exact in distribution" claim is made. Which update to use is the calling
layer's decision, not this layer's.

### `condition(y, noise_cov)`

Moment-form conditioning: the same posterior that `transform_update`
represents as samples, returned as a `Gaussian` — for sampling the
posterior at any size, and for diagnostics. The result has

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
whenever $J - 1 < P$ — the usual regime here. The posterior `Gaussian`
supports `sample` (the factor is the stored representation) but not
`log_density`, which raises `UnsupportedOpError` from the covariance. In
the usual regime that is forced: a rank-deficient Gaussian has no density
on $\mathbb{R}^P$, and the capability system says so instead of returning
`-inf` or `nan`. When $J - 1 \ge P$ the posterior is typically full-rank
and the density exists mathematically — the class's static capability
choice still raises, and a caller wanting that density densifies the
covariance deliberately.

:::{admonition} `PSDLowRank` in `pyeki.linalg`
:class: note

This method's return type is specified by the operator contract, at
{ref}`contract-psd-low-rank`; that entry governs. In outline:
`PSDLowRank(F)` holds a single data field, an `Array` of shape `(n, k)`
with $n, k \ge 1$ and no relation imposed between them, and represents
$F F^\top$. Its `capabilities()` is exactly `frozenset({"diag",
"factor"})` — `solve`, `solve_mat`, `logdet`, `whiten` and `whiten_mat`
all raise `UnsupportedOpError`, at every width — and nothing is computed
at construction, because the stored field *is* the factorization.
:::

(gauss-prng)=
## Randomness

- Every stochastic method takes a JAX PRNG `key` as its first argument and
  **consumes it whole**: the caller owns splitting. No method stores a key,
  advances a hidden state, or splits the key it was given for purposes
  beyond the one call.
- The two draws in the layer are **pinned by this contract**:
  `Gaussian.sample` draws `normal(key, (n_samples, k))` with `k` the factor
  width, and `pathwise_update` draws `normal(key, (J, N))`. Same key, same
  arguments, same representation ⇒ identical output arrays across *pyEKI*
  releases, **for a fixed JAX version and PRNG configuration** — the bit
  stream itself belongs to JAX and has changed across JAX releases and
  flags (`jax_threefry_partitionable`, x64 mode, PRNG implementation).
  pyEKI never changes the draw on its side; doing so is a breaking change.
  The test suite snapshots both draws so a JAX-side stream change is
  detected rather than silently absorbed. The pinning is over evaluation
  in one mode: `jit`-compiled and eager evaluation of the same call may
  differ in the last bits, as anywhere in JAX.
- - Everything else is deterministic. There is exactly one source of randomness
  per stochastic call, which is what makes a caller's runs resumable from a
  stored key — a requirement the layer above inherits and this layer must not
  undermine.

(gauss-validation)=
## Validation and errors

The four-tier scheme of the operator contract ({ref}`contract-validation`)
applies with one extension: everything static is checked always, values
only on request — and, unlike the operator layer, tier 4 here also runs at
*call* time: in debug mode the conditioning methods check `y`,
`log_density` its evaluation point `x`, and the primitives their operands,
for finiteness, and the conditioning methods additionally check **what they
return**. Like the operator layer's,
these checks read array values and are therefore skipped on tracers: under
`jit` or `vmap` they do not run, in debug mode or otherwise. A singular
`noise_cov` is not among the *operand* checks — nothing here can detect one
before the fact; it surfaces as `nan`, or as the result check firing after
the fact. What each tier means here:

| tier | checks | examples |
| ---- | ------ | -------- |
| 2. construction | ranks, static sizes, operator types, cross-field shape agreement; a vmapped-family `cov` | `u_samples` rank ≠ 2; $J = 1$; `cov` not a `PSDLinOp`; `mean` and `cov` sides disagreeing |
| 3. call | operand core shapes, operator arguments, and static non-array arguments; a vmapped-family `noise_cov` — `ValueError` for shape violations, `TypeError` for type violations, per the taxonomy below | `y` not `(N,)`; `noise_cov` side ≠ $N$; `noise_cov` not a `PSDLinOp`; primitive operands mis-shaped; `n_samples` not a positive `int` |
| 4. value (debug) | finiteness of `u_samples`, `v_samples` and `mean` at construction; of `y`, `x`, and the primitives' `s` and `b` at call; of the conditioning methods' *returned* values | violations yield `nan` or a silently wrong posterior outside debug mode |

Tier-1 (field declaration) is inherited with the class machinery
({ref}`gauss-jax`). Error messages follow the operator contract's
obligations: name the object (its `repr`), the method, the expectation, and
the offending value's shape or type.

Within a method the checks run in the operator layer's order: the family
guard ({ref}`gauss-jax`) first, then the required-capability checks in the
order the method names them, then tier-3 operand and operator-argument
validation, then — in debug mode — tier-4 value checks.

Result checks run **last**, once there is a result to check. This is the
layer's one tier-4 *postcondition*: all three conditioning methods assert in
debug mode that what they return is finite — the updated sample block, or, for
`condition`, both the posterior mean and the covariance factor, checked
before the `PSDLowRank` and the `Gaussian` are built, so the diagnosis names
the conditioning call rather than a constructor below it. Three points fix
its scope:

- **Why this layer checks outputs when the operator layer does not.** A
  conditioning result becomes the *next* step's input: a `nan` sample block
  is handed straight to an expensive forward-model evaluation, and a model
  that returns finite nonsense for `nan` parameters launders it beyond
  recovery. An operator's result goes back to the caller who asked for it.
  The asymmetry is deliberate, and is not an argument for adding output
  checks to `pyeki.linalg`.
- **It is the only cheap detection of a singular `noise_cov`.** The three
  methods behave identically under it. Before this rule `condition` alone
  raised, because it happened to route its mean through a constructor — and
  even then it left the covariance factor unchecked, so the check covered
  half of what it appeared to guard.
- **`sample` is deliberately excluded**, and `log_density` with it. Their
  covariance arrives already constructed, so a non-finite result implicates
  the operator rather than this call, and the operator layer validates its
  own fields at construction — which {class}`~pyeki.linalg.PSDLowRank` and
  {class}`~pyeki.linalg.DensePSD` both do for their factors. The gap this
  leaves is deliberate and worth naming: a *singular* covariance with
  entirely finite fields makes `log_density` return `nan` with no check
  firing, in debug mode or out. Every shipped operator rejects that at
  construction, but the level is user-extensible, so a custom `PSDLinOp`
  that whitens and is singular reaches it. Nonsingularity is
  `log_density`'s stated precondition, not something this layer detects.

Because tier 4 is skipped on tracers, none of this fires inside a
`jit`-compiled driver loop. Detection there is a different mechanism, and
belongs to the layer that owns the loop.

The ordering rule has one forced exception, for the operator argument that
*is* the capability bearer. A conditioning method cannot consult
`noise_cov.supports("whiten")` before it knows `noise_cov` is an operator at
all, so two of its tier-3 checks — that it is a `PSDLinOp` (`TypeError`) and
that it is not a vmapped family (`ValueError`) — run **ahead** of the
capability check; the side check and `y`'s shape check stay behind it. No
such exception applies to `Gaussian`, whose `cov` is validated at
construction.

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
| `y`, `x`, `s`, or `b` core shape mismatch at call | `ValueError` |
| `noise_cov` not a `PSDLinOp` / wrong side | `TypeError` / `ValueError` |
| `n_samples` not a positive Python `int` | `TypeError` / `ValueError` |
| covariance lacking `factor` / `whiten` / `logdet` where required | `UnsupportedOpError`, from the operator layer |
| any operation or array-computing property on a vmapped family | `ValueError`, at call — apply the family under `jax.vmap` |
| a vmapped-family `cov` at construction, or `noise_cov` at call | `ValueError` |
| violated value precondition | `ValueError` in debug mode; `nan` or a silently wrong result otherwise |
| a non-finite conditioning result (typically a singular `noise_cov`) | `ValueError` in debug mode, from the method; `nan` otherwise |

(gauss-jax)=
## JAX integration

Both classes are frozen-dataclass pytrees declared with the same machinery
as operators, and every rule of the operator contract's JAX section
({ref}`contract-jax`) binds them:

- **Field classification is the same allowlist**: a field is pytree data iff
  its annotation is `Array` or a `LinOp` subtype; everything else is
  `static_field()` or a definition-time `TypeError`. (No gauss class stores
  another gauss class, so the allowlist needs no extension.)
- **Unflattening bypasses the constructor**, so tier-2 validation is strict
  at genuine construction and absent at trace boundaries; `vmap`-ed
  families reconstruct exactly as operator families do
  ({ref}`contract-batching-operators`), and behave as the Families
  subsection below specifies.
- **Identity semantics**: `eq=False`, hash by identity, never
  `static_argnums` — a joint is always a traced argument.
- **Constructors store and validate; classmethods compute.** Nothing in the
  layer computes at construction today ({ref}`gauss-empirical` records why
  the SVD cannot live there); the rule binds any future classmethod that
  does.
- **`pyeki.gauss` exports no class decorator.** The class set is closed
  (rule 2 of {ref}`gauss-objects`), so there is nothing for users to
  declare; internally the classes may reuse the operator layer's
  registration machinery, generalized as needed, without exposing it.
- Conditioning methods and primitives must be `jit`- and `vmap`-safe with
  no data-dependent shapes, and `log_density` must be differentiable with
  respect to `x` and the array leaves (hyperparameter estimation
  differentiates through it; the updates carry no such requirement — see
  the differentiability caveat in {ref}`gauss-primitives`).

### Families

The operator contract's family machinery ({ref}`contract-families`) is
per-class; here is its gauss instantiation:

- **Both classes have a required `batch_shape` property**, computed like
  the operators' — each array field contributes its leading axes beyond
  its core rank, and an operator field contributes its own `batch_shape`,
  combined by broadcasting with `ValueError` on mismatch. Core ranks:
  `Gaussian.mean` 1 (with `cov` contributing its own); `u_samples` and
  `v_samples` 2. Directly constructed objects always report `()`.
- **Genuine construction rejects families**: `Gaussian(mean, cov)` with
  `cov.batch_shape != ()` is a tier-2 `ValueError` (the batch shape is
  static information), and a family `noise_cov` at a conditioning call is
  a tier-3 `ValueError` — apply a family of noise operators with
  `jax.vmap`, not directly.
- **Inertness**: when `batch_shape` is non-empty, every method and every
  array-computing property (`sample`, `log_density`, the three
  conditioning methods, the means and anomalies) raises `ValueError`
  naming the object, the operation, the batch shape, and the remedy —
  apply the family under `jax.vmap` — before any capability or operand
  check. The static `int` properties (`n`, `n_samples`, `u_dim`,
  `v_dim`), `batch_shape` itself, and `repr` still answer — and the size
  properties report **core** (trailing) sizes, never batch sizes, exactly
  as an operator's `shape` does: `Gaussian.n` is `mean.shape[-1]`, and
  `EmpiricalJoint`'s three are `u_samples.shape[-2]`, `u_samples.shape[-1]`
  and `v_samples.shape[-1]`.
- **Family repr** wraps the ordinary form, as for operators; the form and
  the never-raises rule are in {ref}`gauss-repr`.

(gauss-consumers)=
## How the layers above consume this one

Not normative for `pyeki.gauss` itself, but the design was shaped against
these call sites, and a change that breaks them is a change to reconsider.

**The EKI driver** (`pyeki.eki`, planned) is a loop over tempering steps.
Per step: build the joint from the current ensemble, update, re-evaluate
the forward model.

```python
prior = Gaussian(m0, C0)                    # C0: any PSDLinOp with factor
u = prior.sample(key_init, n_samples)

for key_t, dbeta in schedule:               # increments, not levels
    v = forward(u)                          # caller's vmap-ed model: (J, N)
    joint = EmpiricalJoint(u_samples=u, v_samples=v)
    u = joint.pathwise_update(key_t, y, noise_cov / dbeta)
```

The tempered noise `noise_cov / dbeta` is the operator layer's scalar
scaling — a traced increment flows through the 0-d scalar field, so the
adaptive schedule never re-factorizes the noise. The deterministic variant
is the same loop with `transform_update(y, noise_cov / dbeta)` and no key.

Two facts shape how the driver uses this layer. First, a candidate
tempering increment rescales the kernel, not the data: the whitener of
$R/\delta$ is $\sqrt{\delta}\,W$, so
$S(R/\delta) = \sqrt{\delta}\,S(R)$ **and** the whitened residual
becomes $\sqrt{\delta}\,Wr$. Both rescalings enter, giving weight
multipliers $\delta\sigma_i/(1+\delta\sigma_i^2)$ and transform
modifiers $(1+\delta\Sigma^2)^{-1/2} - I$ from a single base SVD —
rescaling $\sigma_i$ alone is wrong by $1/\sqrt{\delta}$. The public
primitives cannot exploit this, since each recomputes its own SVD
({ref}`gauss-kernel`); an adaptive search that wants the saving needs an
internal entry point, which this layer does not currently expose
({ref}`gauss-excluded`).

Second, forward-model failures are handled by *sample preprocessing*, not
by a masked joint: with validity mask $m_j$ and $J_v \ge 2$ valid
members, replacing each member by
$\hat u + m_j\,(u_j - \hat u)\,\sqrt{(J-1)/(J_v-1)}$ (with $\hat u$
the valid-member mean, and the *same* mask applied to $v$ — a failed
evaluation invalidates the pair) makes the fixed-$J$ joint's moments,
cross-covariance included, equal the masked moments exactly, and keeps
shapes static under `jit`. Failed members rejoin at the posterior mean
under `transform_update` and `condition`; under `pathwise_update` they
land at the posterior mean plus their own perturbation, a
$\mathcal{N}(0, KRK^\top)$ draw about it. Surviving members are moved
outward by the factor $\sqrt{(J-1)/(J_v-1)} > 1$ relative to a genuinely
$J_v$-member analysis — that is the moment matching working, but it is a
real change to the parameters handed to the next forward-model
evaluation. Two preconditions are the driver's to enforce, because both
fail silently here: $J_v \ge 2$ (at $J_v \le 1$ the rescale is `inf` or
`nan` and the whole ensemble becomes `nan`), and mask identity between
$u$ and $v$ (differing masks corrupt $\widehat{C}_{uv}$ with no
exception).

:::{important}
**`pyeki.eki` does not ship this construction, and the reason is worth
recording here rather than only there.** The rescaling is applied to the
*surviving* members too, so each is moved outward from the centre by a
data-dependent factor at every step — $\sqrt{99/89} \approx 1.055$ at
$J = 100$ with a tenth of the members failing, which is larger than the
multiplicative inflation practitioners actually use, applied silently and by
default. The pair $(u_j, v_j)$ also stops being forward-model-consistent.
`pyeki.eki` therefore uses the undamped map
$u_j \mapsto \hat u + m_j (u_j - \hat u)$, accepting a covariance damped by
$(J_v-1)/(J-1)$ in exchange for leaving valid members bit-identical.

What is written above is the *moment-exact* option and its arithmetic, which
this layer records because its fixed divisor is what makes the option
available at all. It is not a recommendation to the driver. The two
preconditions named above bind either map, since both concern the mask rather
than the scaling.
:::

This is why the anomaly divisor stays fixed
({ref}`gauss-excluded`).

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
Two idioms make the fixed-size-neighbourhood plan exact. Zeroing column
$i$ of `s` and entry $i$ of each *whitened* residual is exactly the
analysis in which whitened coordinate $i$ is absent — elementwise, for
both primitives — so padded slots mask to exact no-ops under static
shapes. That is "observation $i$ removed" only when the whitener does not
mix coordinate $i$ with the kept ones: for correlated noise a within-block
mask is not an observation removal. Hence the second idiom: the per-block
noise must align with the noise operator's contractual `block_shapes` —
extracting a principal submatrix of a *correlated* block is not an
operator-layer operation, so partial-block neighbourhoods require diagonal
noise or block-aligned neighbourhoods.

**The test suite** holds the reference implementation. On small problems
the dense Bayes formulas are written out in the tests themselves — plain
dense linear algebra over the empirical moments, independent of this
layer's code — and every conditioning method is checked against them
({ref}`gauss-conformance`). Exact-moment fixtures extend the comparison to
analytic posteriors: for any target joint with a covariance factor $G$ of
width $k$, an ensemble of $J \ge k + 1$ members whose empirical
moments equal the target's exactly can be constructed — concretely, take
the complete QR of $\mathbf{1} \in \mathbb{R}^J$, let $E$ be its last
$k$ columns (orthonormal, each $\perp \mathbf{1}$), and set the members
to $\mu + \sqrt{J-1}\,E\,G^\top$ — so closed-form linear-Gaussian
posteriors are reachable through `EmpiricalJoint` alone. Only $J \ge k + 1$
binds (at $J = k$ the construction fails, and silently); reducing a wide
factor first — a thin QR of $G^\top$, or an eigendecomposition of
$GG^\top$, never a Cholesky, which raises on the rank-deficient targets
that matter — merely lowers the ensemble size that condition demands. A
tempered run's posterior telescoping to the one-shot posterior is the EKI
layer's test; the per-step exactness it composes from lives here.

(gauss-repr)=
## `repr`

Type name and static sizes, never array contents, matching the operator
rule ({ref}`contract-repr`): `Gaussian(n=12)`,
`EmpiricalJoint(n_samples=100, u_dim=12, v_dim=40)`. A vmapped family wraps
that form and names its batch —
`vmapped(EmpiricalJoint(n_samples=100, u_dim=12, v_dim=40), batch=(8,))`
({ref}`gauss-jax`) — and `repr` never raises: an instance whose sizes
cannot be read falls back to a marker form, unspecified beyond its not
raising.

(gauss-surface)=
## Public surface

`pyeki.gauss` exports exactly: the classes `Gaussian` and `EmpiricalJoint`,
including `Gaussian.from_samples`, and the conditioning primitives
`gain_weights` and `sqrt_transform`.
Anything else is private, and no consumer may depend on it. There is no
`pyeki.gauss.testing`: the conformance obligations below bind the package's
own test suite, since the class set is closed.

(gauss-conformance)=
## Conformance

The layer has no user-extension point, so conformance is not a public
harness but a set of obligations on `tests/`. Two rules govern the
reference: exactness tests check against closed forms, not tolerances
chosen to pass — where a closed form exists the suite compares against it,
at a tolerance of a few $\varepsilon$ times the natural scale of the
quantity — and **the dense
reference is hand-written in the tests** — plain dense linear algebra over
means, anomalies, and materialized operators, never routed through
`gain_weights`, `sqrt_transform`, or any other code of this layer — so
every comparison is between two genuinely independent paths. The suite
must verify at least:

1. **Gain against dense**: `gain_weights` composed with `whiten` reproduces
   the dense $K r$ elementwise on small random problems, in all three shape
   regimes $N > J$, $N = J$, $N < J$, with `b` at batch ranks 0, 1, and 2.
2. **Whitener invariance**: two noise operators representing the same $R$
   with different whiteners yield the same weights to floating-point
   tolerance — the invariance is exact in exact arithmetic, but the two
   routes round differently, so this is not the bit-exact SVD invariance
   of {ref}`gauss-primitives`. (No shipped operator
   pair differs — Cholesky uniqueness makes every in-package whitener
   identical — so the test defines a local operator whose `_whiten`
   applies a fixed orthogonal rotation of a valid whitener.)
3. **Transform against dense**: `sqrt_transform` matches a dense
   `eigh`-based $(I + ss^\top)^{-1/2}$ at moderate scale
   ($\sigma_{\max} \lesssim 10^2$, where that reference is trustworthy —
   forming $I + ss^\top$ squares the conditioning, so at large
   $\sigma_{\max}$ the *reference* is the inaccurate side), including the
   case $\rho < J$, i.e. $N < J$, that the naive thin-SVD formula gets wrong.
   At large $\sigma_{\max}$ it instead satisfies the invariant in its
   **stably formed** version,
   $T T^\top + (Ts)(Ts)^\top = I$, to a tolerance scaling as
   $\varepsilon \max(1, \sigma_{\max})$ — the algebraically equivalent
   $T\,(I + ss^\top)\,T^\top = I$ must not be used, because forming
   $ss^\top$ reintroduces the $\sigma_{\max}^2$-sized intermediate whose
   rounding this check exists to avoid, pushing the achievable residual to
   $\varepsilon\sigma_{\max}^2$. **The spectrum of `s` must carry
   singular values of order 1 or below alongside the large ones for this
   comparison to mean anything.** What hides the re-formed version's loss is
   $TT^\top \sim \sigma^{-2}$ being negligible in every direction at once,
   which happens whenever *all* the singular values are large — spanning
   decades is not sufficient, and is the wrong criterion: at $J=5$, $N=8$,
   $\sigma = (10^{10}, 10^9, 10^8, 10^7, 10^6)$ spans four decades and the
   two forms agree to a factor of $1.0$, testing nothing. With
   $\sigma = (10^{10}, 10^5, 1, 1, 1)$ they separate by eight orders of
   magnitude, which is the regime the check must use.
   It satisfies $T = T^\top$ for every
   `s`, and $T\mathbf{1} = \mathbf{1}$ **for mean-centered `s`**
   ($\mathbf{1}^\top s = 0$, which is the only case the conditioning
   kernel produces) to a tolerance of
   $c_1 J \varepsilon + c_2 (\varepsilon \sigma_{\max})^2$; for general
   `s` no such identity holds ({ref}`gauss-primitives`). Both terms are
   needed and **both scale**. The quadratic term is the modifier-induced
   mean shift of {ref}`gauss-kernel`, but the *computed* $T\mathbf{1}$
   carries ordinary round-off from its $J$-term dot products on top, and
   that floor dominates until $\sigma_{\max}$ reaches
   $\varepsilon^{-1/2}$: at $\sigma_{\max} = 3.3$ the observed residual is
   $4\times10^{-16}$ against $5\times10^{-31}$ for the quadratic term
   alone. The floor grows with $J$, so a *constant* floor calibrated at one
   number of samples expires at another — measured worst ratios over $J$ up to
   400 and $\sigma_{\max}$ up to $10^{13}$ are $0.83$ against
   $J\varepsilon$ and $2.0$ against $(\varepsilon\sigma_{\max})^2$. The
   check must therefore be exercised at more than one $J$.
4. **Moment exactness of the posterior**: `transform_update`'s output has
   sample mean and covariance equal to the hand-written dense posterior
   moments of the fitted joint Gaussian, to floating-point tolerance;
   `condition`'s mean and the dense form of its low-rank covariance equal
   the same moments; and the two methods satisfy the elementwise identity
   $u_j' = m_{\text{post}} + \sqrt{J-1}\, F_{\cdot j}$ of
   {ref}`gauss-empirical`.
5. **Pathwise against dense, elementwise**: with $\varepsilon$ recomputed
   from the same key (the pinned draw), `pathwise_update` equals
   $u_j + K(y - v_j - W^{-1}\varepsilon_j)$ computed densely, with $W$
   recovered as the transpose of `noise_cov.whiten(jnp.eye(N))` — the
   row-wise batch contract returns $W^\top$.
6. **Exact-moment fixtures**: on samples constructed so that their
   empirical moments equal an analytic linear-Gaussian joint
   ({ref}`gauss-consumers`), `condition` and `transform_update` reproduce
   that joint's closed-form posterior moments.
7. **Marginal formulas**: `sample` matches its pinned elementwise
   definition; `log_density` matches the dense closed form at batch ranks
   0, 1, and 2 and differentiates. `from_samples` reproduces the sample mean
   and the $J-1$ covariance exactly against a dense reference, agrees with
   `EmpiricalJoint`'s $u$-block moments from the same samples, holds a
   `PSDLowRank` of width $J$ that withholds `solve`, `whiten` and `logdet`,
   gives identical samples exactly zero spread, and validates rank and
   sample count.
8. **Degeneracy**: zero prediction anomalies — every row of `v_samples`
   given the *same, exactly representable* value, since a collapsed
   block of arbitrary values leaves anomalies at $O(\varepsilon)$
   rather than bit-zero — make both updates the identity on `u_samples` —
   bit-exact for `pathwise_update`, to round-off for `transform_update`,
   which reconstructs $\bar u + a_j$ — and `condition` return the prior
   marginal's moments; $J = 2$ and $N = 1$ work; a collapsed sample block
   with finite inputs produces no `nan`.
9. **Capability propagation** (`PSDLowRank` is the one shipped
   `PSDLinOp` that disclaims operations, and it covers the `whiten` and
   `logdet` cases; the `factor` case needs a test-local `PSDLinOp`
   implementing no `_factor`): a noise covariance without `whiten`, and a
   covariance without `factor` or `logdet`, raise `UnsupportedOpError`
   from the conditioning methods, `sample`, and `log_density`
   respectively — and `log_density` on the posterior `condition` returns
   raises the same way, regardless of $J$ versus $P$.
10. **Validation**: every tier-2 and tier-3 rule of
    {ref}`gauss-validation` raises as specified.
11. **JAX round trips**: flatten/unflatten preserves type and behaviour for
    both classes; the conditioning methods run under `jit`; constructing an
    `EmpiricalJoint` inside `vmap` round-trips and a `vmap`-ed family of
    joints agrees with a Python loop; sentinel-leaf unflattening succeeds.
12. **Reproducibility and repr**: same key, same output, elementwise;
    different keys differ; reprs match {ref}`gauss-repr` with no array
    data; and the pinned draws are snapshotted so a JAX-side PRNG stream
    change is detected ({ref}`gauss-prng`).
13. **Families**: stacking either class's leaves and unflattening yields a
    family that reports its batch shape, takes the `vmapped(...)` repr
    form, and refuses every method and array-computing property with the
    family `ValueError`, while the static `int` properties and
    `batch_shape` still answer; genuine construction with a family
    covariance raises ({ref}`gauss-jax`).

Beyond the tests, the implementation PR owes two deliverables named
earlier: the layer's user-guide page ({ref}`gauss-scope`) and the
`PSDLowRank` operator with its operator-contract entry
({ref}`gauss-empirical`).

Alongside conformance, targeted regression tests guard the layer's own
silent-failure classes once found — the thin-SVD completion term (check 3),
a mixed-representation perturbation, a mean shift from an uncentered
transform, a `nan` gradient at an exactly collapsed `s`, a singular
noise covariance turning an update into an all-`nan` result with no
exception outside debug mode (assert the `nan`, so the day it starts raising
is visible), the whitening grouping of {ref}`gauss-kernel` (checked for
accuracy against an exact reference, not only for its application count),
and a collapsed sample block at large magnitude, where a mean-and-subtract that
does not cancel exactly turns an exact no-op into a wrong, finite update —
under the same
do-not-delete rule as the operator layer's.

(gauss-excluded)=
## Deliberately excluded

Recorded so their absence reads as a decision, not an oversight.

**An operator-represented joint class.** An earlier draft of this contract
had one — mean vectors plus covariance blocks $(C_{uu}, C_{uv}, C_{vv})$ as
operators, with a dense `condition` as the packaged oracle. It was dropped:
the conformance rules require the oracle to be hand-written in the tests
precisely so that no package code is trusted, and with the empirical path
supplying the moment posterior ({ref}`gauss-empirical`) the class had no
remaining consumer. If exact conditioning of operator joints is ever
needed — GP-style prediction with full-rank structured priors — the right
re-entry is a **factor-pair joint**: coherent operators $(F_u, F_v)$
sharing one latent vector, with $C_{uu} = F_uF_u^\top$,
$C_{uv} = F_uF_v^\top$, $C_{vv} = F_vF_v^\top$. That representation runs
through the same whitened-SVD kernel (`EmpiricalJoint` is its dense special
case, $F = A^\top/\sqrt{J-1}$), gives joint sampling and Matheron-style
pathwise conditioning for free, and avoids the two dead ends of the block
form: a linear-map constructor whose $C_{vv}$ cannot be *typed* PSD
(composition never proves PSD-ness), and a downdate operator
(`PSDDowndate(base, F)` for $\mathrm{base} - FF^\top$) whose `factor` and
`whiten` require solving $F = LG$ against the base's own factor, which
only some base types can do.

**The Woodbury route.** Applying the Woodbury identity to
$(\widehat{C}_{vv} + R)^{-1}$ is the same algebra as the whitened SVD with
worse arithmetic: it forms
$(J-1)I_J + A_v R^{-1} A_v^\top = (J-1)(I + S S^\top)$ — and *forming*
the Gram rounds away singular values below
$\sqrt{\varepsilon}\,\sigma_{\max}$, even though the assembled matrix
itself is regularized by the shift. The loss is governed by
$\sigma_{\max}$, not by $\kappa(S)$: collapse ($\sigma \to 0$) costs the
Gram route nothing, while at $\sigma_{\max} = 10^8$ it destroys every
singular value below $1.5$ — the largest gain multipliers
$\sigma/(1+\sigma^2)$ among them — for a relative error around $10^{-1}$
where the SVD route holds $10^{-8}$. The SVD route
gets the bounded multiplier $\sigma/(1+\sigma^2)$ for free; a factorization
of the formed Gram matrix does not. Its one advantage — needing `solve` on
the noise instead of `whiten` — has no consumer, since `whiten` is the
operation the noise interface is designed around. Rejected, not deferred;
{doc}`design` records the full comparison.

**The dense $N \le J$ route.** When observations are few, materializing
$\widehat{C}_{vv} + R$ and factorizing it is affordable and can beat the
SVD. It may arrive later as an *internal* optimization behind the same
method signatures, with results equal to floating point — and it must
never become the in-package reference: the oracle stays hand-written in
the tests.

**A reified gain object.** Computing the SVD once and reusing it across
conditioning methods on the same `(joint, noise_cov)` pair would need a
returned decomposition object. A caller calls exactly one method per step
once, so today it would be surface without a consumer; the conditioning
primitives already serve anyone assembling custom flows. The one
prospective consumer is an adaptive-step search, which could reuse a
single SVD across candidate increments ({ref}`gauss-consumers`); it pays
one SVD per candidate until that consumer exists and justifies the
object.

**Batched observations.** `y` is one observation vector. A family of
updates — multiple observations, multiple noise levels — is `jax.vmap`
over the method, consistent with the layer-wide batching story.

**A configurable anomaly divisor.** $J - 1$ everywhere. A $1/J$ convention
changes every formula's scaling for no consumer — the masked-sample
consumer is served by sample preprocessing in `pyeki.eki`
({ref}`gauss-consumers`); inflation, which is the principled way to widen
a set of samples, belongs to `pyeki.eki`.

**An empirical marginal-likelihood accessor.**
$\log\det(\widehat{C}_{vv} + R) = \log\det R + \sum_i \log(1 + \sigma_i^2)$
falls out of the update's own SVD, and a future evidence or
tempering-diagnostic consumer may want it. Excluded until that consumer
exists; recorded so its absence reads as a decision and the identity is
not rediscovered.

**Perturbation injection.** `pathwise_update` takes no `eps` argument; see
the warning in {ref}`gauss-empirical`. Determinism needs are met by the
pinned key-derived draw.

**Non-Gaussian anything.** No mixtures, no transformations of variables,
no likelihoods other than additive Gaussian noise. The layer is the
Gaussian core of EKI, not an inference framework.
