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
- **conditioning** a joint Gaussian on a noisy observation of its second
  block, returning the posterior as a distribution in structured low-rank
  form, and transporting realizations of the joint to the posterior by
  Matheron's rule;
- **the two sample-to-sample updates** built on that conditioning, in both
  the stochastic (perturbed-observation) and deterministic (square-root
  transform) forms;
- **the array-level conditioning primitives** underneath all of it, exposed
  at the granularity that domain localization consumes.

The layer is not a probabilistic-programming toolkit: there are no densities
over anything non-Gaussian, no transformations of distributions, and no
inference machinery. Nor does it supply a dense reference implementation —
the reference everything is tested against is hand-written in the test suite
({ref}`gauss-conformance`), so that no package code is trusted.

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
| $k$    | the latent width of a joint factor |
| $F$    | a joint factor, with row blocks $F_u \in \mathbb{R}^{P \times k}$ and $F_v \in \mathbb{R}^{N \times k}$ |
| $G$    | a linear map, $N \times P$ |
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
- **Factors are column-wise: a factor of an $n$-dimensional covariance is
  an $(n, k)$ operator**, as `PSDLinOp.factor()` returns it. The two
  conventions meet whenever samples become a factor, and the conversion
  carries a transpose *and* the divisor's square root:
  $F_u = A_u^\top/\sqrt{J-1}$. Displayed mathematics uses whichever
  convention the object at hand is stored in, and says which.
- **Vectors passed to methods are exactly core-shaped.** An observation `y`
  is a `(N,)` array, a mean a `(dim,)` array. The gauss classes do not
  accept batched operands the way operator methods do; a family of
  distributions or updates is expressed with `jax.vmap` over the pytree
  ({ref}`gauss-jax`). There are three exceptions, each following the
  operator layer's batch contract: the two conditioning *primitives*
  ({ref}`gauss-primitives`), the evaluation-point argument of
  `Gaussian.log_density`, and the three realization arguments of
  `GaussianJoint.pathwise` ({ref}`gauss-pathwise`).
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

The public surface is three classes and two functions.

| object | represents | representation | where it is used |
| ------ | ---------- | -------------- | ---------------- |
| `Gaussian` | one Gaussian distribution | mean vector + `PSDLinOp` covariance | a prior or marginal; what `condition` returns |
| `GaussianJoint` | a joint Gaussian over the two blocks | mean pair + joint factor | every conditioning identity |
| `EmpiricalJoint` | $J$ paired samples | two row-aligned sample blocks | the two sample-to-sample updates |
| `gain_weights` | the latent weights for one whitened residual | pure array function | the shared conditioning core |
| `sqrt_transform` | the square-root update transform | pure array function | the shared conditioning core |

Four rules govern the set:

1. **The two joint classes divide the work by what they return.**
   `GaussianJoint` owns the mathematics: it holds moments and returns
   distributions, or transports realizations handed to it
   ({ref}`gauss-joint`). `EmpiricalJoint` owns the samples: it holds them
   and returns updated samples aligned with the ones it holds
   ({ref}`gauss-empirical`). Neither duplicates the other's role, and
   nothing on `EmpiricalJoint` returns a distribution.
2. **The Gaussian fit is written at the call site.** Conditioning a set of
   samples means conditioning a Gaussian fitted to their moments, and
   `EmpiricalJoint.to_gaussian_joint()` is where that fit happens. There is
   no `condition` on `EmpiricalJoint`: a method of that name on a class
   named for the empirical distribution reads like conditioning the
   empirical measure, which is not what it would do.
3. **The set is closed.** Unlike the operator layer, `pyeki.gauss` has no
   extension story: users do not subclass these classes. User extensibility
   lives one layer down, in the operators supplied as covariances and as
   factor row blocks — a custom operator that passes `check_operator` works
   here unchanged. There is consequently no public class decorator and no
   `pyeki.gauss.testing` module ({ref}`gauss-conformance`).
4. **Objects are unbatched frozen pytrees, exactly like operators.** The
   construction, validation, and JAX-integration rules of the operator
   contract apply verbatim ({ref}`gauss-jax`); a family of joints is a
   `vmap`-ed pytree, never stored batch axes.

(gauss-kernel)=
## The conditioning kernel

All conditioning in the layer is one computation, specified here once. The
class methods ({ref}`gauss-joint`, {ref}`gauss-empirical`) are thin
assemblies of these pieces; the two public functions
({ref}`gauss-primitives`) expose them directly. "Thin assembly" names the
mathematics, not an obligation to call the public functions: the class
methods share a single SVD of $S$ internally, while each public primitive
recomputes its own — routing a class method through the primitives is
permitted only where it preserves the one-SVD-per-call rule of
{ref}`gauss-joint`.

### The joint factor

A joint Gaussian is represented by a **joint factor**: a single factor of
the whole block covariance, cut into two **row blocks**.

$$
F = \begin{pmatrix} F_u \\ F_v \end{pmatrix}
\in \mathbb{R}^{(P+N) \times k},
\qquad
\begin{pmatrix} C_{uu} & C_{uv} \\ C_{vu} & C_{vv} \end{pmatrix}
= F F^\top ,
$$

so that $C_{uu} = F_u F_u^\top$, $C_{uv} = F_u F_v^\top$ and
$C_{vv} = F_v F_v^\top$ — all three at once, with no side condition to
check. Equivalently, one shared latent vector drives both blocks:

$$
\begin{pmatrix} u \\ v \end{pmatrix}
= \begin{pmatrix} \bar u \\ \bar v \end{pmatrix}
+ \begin{pmatrix} F_u \\ F_v \end{pmatrix}\xi,
\qquad \xi \sim \mathcal{N}(0, I_k).
$$

That shared $\xi$ is the whole content of the representation. Two factors
chosen independently, one of $C_{uu}$ and one of $C_{vv}$, say nothing at
all about $C_{uv}$; the pair must come from one factorization of $F F^\top$.
The latent width $k$ is otherwise unconstrained.

A factor is **centred** when $F\mathbf{1}_k = 0$. That is a property of the
representation, not of the distribution, and it is what makes the latent
index a sample index ({ref}`gauss-empirical`).

### The whitened factor

Given a joint factor and a noise operator with whitener $W$, define the
**whitened factor**

$$
S \;=\; \bigl(W F_v\bigr)^\top \;\in\; \mathbb{R}^{k \times N},
$$

whose $j$-th row is $W$ applied to the $j$-th column of $F_v$ — in code,
exactly `noise_cov.whiten_mat(F_v)` transposed, one call. Let

$$
S = U \Sigma V^\top \quad \text{(thin SVD)}, \qquad
U \in \mathbb{R}^{k \times \rho},\;
V \in \mathbb{R}^{N \times \rho},\;
\rho = \min(k, N),
$$

with singular values $\sigma_1 \ge \dots \ge \sigma_\rho \ge 0$. For a
centred factor, $S^\top \mathbf{1} = W F_v \mathbf{1} = 0$: the all-ones
direction is in the null space of $S^\top$, so every $\sigma_i > 0$ has
$U_{\cdot i} \perp \mathbf{1}$. A factor read off $J$ samples is centred and
so has rank at most $J - 1$, putting at least one singular value exactly at
zero whenever $N \ge J$ — in exact arithmetic; floating point returns
$O(\varepsilon \lVert S \rVert)$ instead. Every formula below is
continuous at $\sigma = 0$ in value and needs no special-casing;
differentiability is another matter ({ref}`gauss-primitives`).

### The gain

The Kalman gain $K = C_{uv}\,(C_{vv} + R)^{-1}$ applied to a residual $r$
satisfies

$$
K r \;=\; F_u\, w, \qquad
w \;=\; U \operatorname{diag}\!\Bigl(\frac{\sigma_i}{1+\sigma_i^2}\Bigr) V^\top\, (W r)
\;\in\; \mathbb{R}^{k}.
$$

The vector $w$ is the **weight vector**: the update to $u$ is the
combination $F_u w$ of the factor's own columns, so no matrix of dimension
$P$ or $N$ is ever formed and the update stays in the span of $F_u$. In
closed form, independent of the SVD,

$$
w \;=\; S\,(S^\top S + I_N)^{-1}\,(Wr)
\;=\; F_v^\top\,(C_{vv} + R)^{-1}\, r ,
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
  ill-conditioned $s$ becomes, with no regularization parameter to
  tune.

The SVD form is the normative implementation: it never forms
$S^\top S$ or $S S^\top$, whose condition numbers are the *squares* of
$S$'s. The algebraically equivalent routes that do form them are excluded
({ref}`gauss-excluded`); {doc}`design` gives the full comparison with the
normal-equations route.

### The square-root transform

Conditioning multiplies the factor's $u$ block on the right. Define

$$
T \;=\; (I_k + S S^\top)^{-1/2}
\;=\; I_k + U\bigl((I_\rho+\Sigma^2)^{-1/2} - I_\rho\bigr)U^\top
\;\in\; \mathbb{R}^{k \times k},
$$

symmetric, built from the same SVD. The second form is the normative one:
for a thin SVD the naive $U(I+\Sigma^2)^{-1/2}U^\top$ *omits the identity on
the orthogonal complement* and is simply wrong whenever $\rho < k$ — the
$I_k + U(\cdot - I)U^\top$ form is exact for every rank.

$F_u T$ is a factor of the posterior covariance, exactly:

$$
\bigl(F_u T\bigr)\bigl(F_u T\bigr)^\top
\;=\; F_u\bigl(I_k + S S^\top\bigr)^{-1}F_u^\top
\;=\; C_{uu} - K\,C_{vu},
$$

the middle step by the push-through identity
$S(I_N + S^\top S)^{-1}S^\top = I_k - (I_k + SS^\top)^{-1}$. So the whole of
conditioning is the map
$(\bar u, \bar v, F_u, F_v) \mapsto (\bar u + F_u w,\; F_u T)$, from one SVD.

Two structural facts, both load-bearing and both conformance-checked:

- **The transform preserves centring**: $T\mathbf{1} = \mathbf{1}$
  (each $U_{\cdot i}$ with $\sigma_i > 0$ is orthogonal to $\mathbf{1}$, and
  the modifier vanishes at $\sigma_i = 0$), so a centred factor conditions
  to a centred factor and the posterior mean is not silently shifted.
  In floating point the numerically-zero $\sigma$'s $U$ column need not be
  orthogonal to $\mathbf{1}$; the property survives because the modifier
  decays *quadratically*: the induced mean shift is
  $O\bigl((\varepsilon\sigma_{\max})^2\bigr)$ rather than
  $O(\varepsilon\sigma_{\max})$ — the modifier is exactly $0.0$ while
  $\varepsilon\sigma_{\max} \lesssim 10^{-8}$, and negligible above it.
- **The identity is exact in exact arithmetic**, not asymptotic in $k$. The
  conformance suite checks it to floating-point tolerance
  ({ref}`gauss-conformance`).

### Cost

Conditioning whitens $k + 1$ vectors — the factor's $k$ columns and the mean
residual — from one `whiten_mat` call on the stacked columns
$[F_v \mid y - \bar v]$. At $k = J$ that is the $J + 1$ a sample update
spends.

Whitening is a fixed linear map applied column-wise, so it commutes with
subtraction and with the centring that built the factor. The grouping is
**not** free, however: centring and differencing must happen *before* the
whitener is applied. The two orders agree in exact arithmetic but are not
equally stable — centring *whitened* vectors makes the cancellation ratio
$\lVert W\bar v\rVert / \lVert W F_v\rVert$ in place of
$\lVert \bar v\rVert / \lVert F_v\rVert$, so the error grows with
$\kappa(W) = \sqrt{\kappa(R)}$ whenever $\bar v$ is aligned with a precise
direction of the noise. Measured against an exact reference at
$\kappa(R) = 10^4$ with $\bar v$ of magnitude $10^8$ along $R$'s most precise
direction, whitening first gives a posterior-mean error of $4.9$ where
centring first gives $2\times10^{-6}$. Holding a factor rather than samples
settles this structurally: the factor is centred at construction, so there is
no ordering left to get wrong inside a conditioning call.

Two pathwise routes have deliberately different costs. Transporting the
realizations a centred factor already holds
({ref}`gauss-empirical`) gets each whitened residual from the whitened factor,
$W(y - v_j) = W(y - \bar v) - \sqrt{J-1}\,S_{j\cdot}$, for $J + 1$
applications in total; what must **not** be done there is whitening the factor
and the residuals separately, which costs $2J$ — twice the necessary figure,
and for a dense whitener on the dominant term. Transporting *arbitrary*
realizations ({ref}`gauss-pathwise`) has no such shortcut, since $v$ is data
rather than the factor, and spends $k$ plus one per realization.

The thin SVD is $O(k N \min(k, N))$; forming weights is $O((N + k)\,\rho)$ per
residual and applying $F_u$ is $O(Pk)$ per weight vector for a dense factor,
less for a structured one. A full sample update is $O(NJ^2 + PJ^2)$ for
$J \le N$, plus the $J + 1$ whitener applications — which the total absorbs
for structured whiteners applying in $O(N)$, and which dominate ($O(JN^2)$)
for a dense $W$ whenever $P \lesssim N^2/J$; at larger $P$ the $O(PJ^2)$
factor application dominates instead. Linear in both dimensions for structured
whiteners, cubic only in the latent width. The conditioning that matters
degrades in the *small-noise* direction (large $\sigma_{\max}$, a large
whitener), not the collapse direction ($\sigma \to 0$), which is numerically
pristine.

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
| `gain_weights(s, b)`   | `(k, N), (..., N) -> (..., k)` | $U \operatorname{diag}\bigl(\sigma_i/(1+\sigma_i^2)\bigr) V^\top b$ for the thin SVD $s = U\Sigma V^\top$ |
| `sqrt_transform(s)`    | `(k, N) -> (k, k)`             | $(I_k + s s^\top)^{-1/2}$ |

Rules:

- **`s` is exactly 2-D, with both sizes at least 1** (`ValueError`
  otherwise; there is no lower bound of 2 on $k$ — the divisor lives in
  the caller). It plays the operator's role, so it carries no batch axes;
  a family of local analyses is a `vmap` over the function. `b` follows
  the operator layer's batch contract: trailing core axis of length $N$,
  any number of leading batch axes carried through. Violations raise
  `ValueError` with the same message obligations as tier-3 operand checks
  ({ref}`contract-validation`), the function's name standing in for the
  operator repr.
- **Both functions compute one SVD per call.** Batch the residuals into a
  single `gain_weights` call rather than looping; the $J$ per-sample
  residuals of a stochastic update are one `(J, N)` operand.
- The functions are deterministic JAX code, safe under `jit` and `vmap`,
  with no data-dependent shapes. They are differentiable wherever the
  singular values of `s` are distinct and nonzero. At *exactly* repeated
  or exactly zero singular values — an exactly collapsed `s`, or the
  zero-padded columns a masked local analysis may produce when the masking
  drops the rank below $\min(k, N)$ — the SVD's
  gradient is `nan`, even though the functions themselves are smooth
  there — `gain_weights` equals the rational $s(s^\top s + I)^{-1}b$, and
  `sqrt_transform` is real-analytic, the spectrum of $I + ss^\top$ being
  bounded below by 1. The
  float-generic degeneracy of centring ($\sigma_{\min} \sim
  10^{-16}$ when $N \ge k$) is not an exact tie and differentiates
  finitely. No conditioning path in this layer *requires* differentiation
  with respect to `s` ({ref}`gauss-jax`) — a caller who differentiates an
  update with respect to `v_samples` does differentiate through the SVD,
  and inherits these cases. An implementation may restore gradients
  everywhere with a custom JVP routed through the closed forms, but is not
  required to; note that for `sqrt_transform` that means a Fréchet
  derivative of $A \mapsto A^{-1/2}$, materially more work than
  `gain_weights`'s rational form.
- `sqrt_transform` imposes no centring requirement on `s`. The
  $T\mathbf{1} = \mathbf{1}$ property of {ref}`gauss-kernel` follows from
  $s^\top\mathbf{1} = 0$, which holds exactly when the factor the whitening
  came from is centred; on general `s` the transform is still
  $(I + ss^\top)^{-1/2}$, and $T\mathbf{1}$ is whatever that matrix makes it.
- Both return values are functions of `s` alone, invariant to the SVD's
  sign and degenerate-rotation freedom — they equal the closed forms of
  {ref}`gauss-kernel` — so any correct thin SVD gives the same output
  elementwise.
- Callers own the semantics of `s`. For conditioning, `s` must be the
  *whitened factor* of {ref}`gauss-kernel`, and `b` a *whitened* residual;
  the functions cannot check this, which is why the class methods — where
  the conventions are enforced — are the default interface and these
  functions are the escape hatch.

(gauss-marginal)=
## `Gaussian`

A single Gaussian distribution $\mathcal{N}(m, C)$: the prior a caller
supplies, and the posterior that `GaussianJoint.condition` returns
({ref}`gauss-condition`).

**Fields.** `mean`, a `(n,)` array, and `cov`, a
{class}`~pyeki.linalg.PSDLinOp` of side `n`. Construction validates the rank
of `mean`, the operator type of `cov`, and their agreement
(tier 2, shape-only — {ref}`gauss-validation`). `n` is a property computed from `mean`.

The size properties across the layer follow one rule: an object with a
single dimension names it `n`, matching {class}`~pyeki.linalg.SquareLinOp`
one layer down; an object with several qualifies each one, as
`EmpiricalJoint`'s `n_samples`, `u_dim` and `v_dim` do, and as
`GaussianJoint`'s `u_dim`, `v_dim` and `latent_dim` do. So `Gaussian.n` and
`GaussianJoint.u_dim` are the same convention applied to different arities,
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

(gauss-joint)=
## `GaussianJoint`

A joint Gaussian over the two blocks, and the home of every conditioning
identity. Its distribution is
$\mathcal{N}\!\left(\begin{pmatrix}\bar u\\ \bar v\end{pmatrix},
FF^\top\right)$ for the joint factor $F$ of {ref}`gauss-kernel`.

**Fields.** `u_mean` and `v_mean`, arrays of shapes `(P,)` and `(N,)`; and
`u_factor` and `v_factor`, the row blocks $F_u$ and $F_v$, both
{class}`~pyeki.linalg.LinOp` s, of shapes `(P, k)` and `(N, k)`. All four
are **keyword-only**: the two means, and the two factors, are pairs of
like-shaped objects agreeing on their trailing size, so exchanging a pair is
valid whenever $P = N$ and no check can detect it. Construction validates
the rank of each mean, the operator type of each factor, each factor's
agreement with its own mean, and the two factors' shared $k$; $P, N, k \ge 1$,
as everywhere. `u_dim`, `v_dim` and `latent_dim` are `int` properties.

The row blocks are operators rather than arrays so that a structured
covariance stays structured. What the kernel asks of them is narrow: $F_u$ is
applied to `k`-vectors and to one $k \times k$ matrix, through `matvec` and
`matmat`; $F_v$ is materialized as an `(N, k)` array, which the singular
value decomposition needs and no representation can avoid.

**Coherence is the caller's, and only partly checkable.** Any pair of row
blocks of matching width defines *some* joint Gaussian — $FF^\top$ is PSD
whatever $F$ is — so an implementation cannot verify that the pair came from
one factorization of the intended covariance. Factorizing $C_{uu}$ and
$C_{vv}$ separately yields the intended marginals and a wrong cross-block,
and conditioning then answers correctly for a different joint. The shared-$k$
check catches this whenever $P \ne N$, two square factorizations having
widths $P$ and $N$; at $P = N$ it cannot. The two arithmetic constructors are
therefore the documented routes, and `from_factors` is the escape hatch.

**Three constructors.**

| constructor | builds | $k$ |
| ----------- | ------ | --- |
| `from_linear_map(u_marginal, linear_map)` | the joint of $u$ and $Gu$ | the width of `u_marginal.cov.factor()` |
| `from_samples(*, u_samples, v_samples)` | the joint fitted to paired samples' moments | $J$ |
| `from_factors(*, u_mean, v_mean, u_factor, v_factor)` | row blocks supplied directly | as given |

`from_linear_map` takes a `Gaussian` whose covariance supports `factor` and
a {class}`~pyeki.linalg.LinOp` $G$ of shape `(N, P)`, and builds

$$
\bar u = m_0, \quad \bar v = G m_0, \quad
F_u = L, \quad F_v = G L,
\qquad C_0 = L L^\top,
$$

for the $L$ that `u_marginal.cov.factor()` returns. Conditioning the result
is closed-form linear-Gaussian conditioning: nothing is inverted, no matrix
of either block's dimension is formed, and a singular $C_0$ is admissible —
the case for which no precision-form posterior exists at all. $F_u$ stays
the operator `factor()` returned; $F_v = GL$ is materialized here, from $k$
applications of $G$, which is the factorization this constructor owes the
kernel and the reason $G$'s shape never enters a conditioning call.
`UnsupportedOpError` if the covariance has no `factor`; `TypeError` on a
non-`Gaussian` or non-`LinOp`; `ValueError` if $G$'s input size is not
`u_marginal.n`, or either argument is a family.

`from_samples` matches moments to two row-aligned sample blocks: the means
are the sample means and the covariance blocks the empirical covariances with
the layer's $J-1$ divisor, held as

$$
F_u = \frac{A_u^\top}{\sqrt{J-1}}, \qquad
F_v = \frac{A_v^\top}{\sqrt{J-1}}, \qquad k = J,
$$

which reproduces all three blocks exactly and forms none of them. It
validates exact rank 2 on both blocks, agreement of the sample axes, and
$J \ge 2$ (a single sample has no anomalies; shape-only, so tier 2 and
unconditional). The factor it builds is **centred**, $F\mathbf{1}_J = 0$,
because anomalies sum to zero — the property {ref}`gauss-empirical` depends
on. The result is a Gaussian *fit to* the samples, not the equal-weight
point-mass distribution of the samples themselves.

`from_factors` wraps a bare array as a {class}`~pyeki.linalg.Dense` and
leaves an operator alone, so a caller mixing the two gets one
representation.

**Marginals.** `u_marginal` and `v_marginal` return
$\mathcal{N}(\bar u, F_uF_u^\top)$ and $\mathcal{N}(\bar v, F_vF_v^\top)$ as
`Gaussian`s whose covariances are `PSDLowRank` holding the corresponding row
block, materialized as an array. `v_marginal` is the **noise-free** marginal
of $v$, not of an observation of it: $R$ does not appear.

**No factorization at construction — deliberately.** The operator layer's
rule is *factorize at construction*; the decomposition the kernel needs is
of $W F_v$, and the noise operator arrives per call and may differ on every
call. Each method computes its SVD once, uses it, and discards it — never
caching it on the instance, which under `jit` would be the discarded-cache
bug the operator contract documents. One method call, one SVD.

Both conditioning methods take a trailing `noise_cov`, a `PSDLinOp` of side
$N$ supporting `whiten` (`UnsupportedOpError` from the inner call
otherwise), and condition on the same observation model, $y = v + \eta$ with
$\eta \sim \mathcal{N}(0, R)$. Precondition: `noise_cov` is nonsingular —
`whiten`'s own precondition. The whitened formulation requires it even though
$C_{vv} + R$ is generically invertible for singular $R$ (it fails only when
$\operatorname{nullity}(R) + \operatorname{nullity}(C_{vv}) > N$); a singular
noise operator that nonetheless types as whitening-capable yields `nan`, or
the tier-4 result check of {ref}`gauss-validation` in debug mode. Both
degrade gracefully when $F_v$ is zero: `condition` returns the $u$ marginal's
own moments and `pathwise` returns its `u` argument unchanged — a collapsed
observed block is a no-op, not `nan`, for finite inputs. Whitening can still
overflow for finite inputs — $F_v \sim 10^{300}$ against a noise variance of
$10^{-20}$ — and there the result is `nan`.

(gauss-condition)=
### `condition(y, noise_cov)`

Standard Gaussian conditioning: the posterior over $u$, as a `Gaussian`.

$$
m_{\text{post}} \;=\; \bar u + F_u\,
\texttt{gain\_weights}\bigl(S,\, W(y - \bar v)\bigr),
\qquad
C_{\text{post}} \;=\; \bigl(F_uT\bigr)\bigl(F_uT\bigr)^\top,
\quad T = \texttt{sqrt\_transform}(S),
$$

with the covariance returned **in structured form**: a `PSDLowRank`
operator holding the `(P, k)` factor $F_uT$ — never a dense $P \times P$
matrix, so no size guard is needed. `y` is an array of rank exactly 1 and
length $N$; unlike operator operands it takes no batch axes, a family of
conditioning calls being a `vmap` over the method.

The returned covariance is honest about rank:
$\operatorname{rank}(C_{\text{post}}) \le \min(k, P)$, so it is singular
whenever $k < P$. The posterior `Gaussian` supports `sample` (the factor is
the stored representation) but not `log_density`, which raises
`UnsupportedOpError` from the covariance. Where $k < P$ that is forced: a
rank-deficient Gaussian has no density on $\mathbb{R}^P$, and the capability
system says so instead of returning `-inf` or `nan`. When $k \ge P$ the
posterior is typically full-rank and the density exists mathematically — the
static capability choice still raises, and a caller wanting that density
densifies the covariance deliberately.

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

(gauss-pathwise)=
### `pathwise(*, u, v, whitened_noise, y, noise_cov)`

Matheron's rule: transport realizations of the joint to the posterior.

$$
\Phi(u, v, \eta) \;=\; u + K\bigl(y - v - \eta\bigr)
\;=\; u + F_u\,
\texttt{gain\_weights}\bigl(S,\, W(y - v) - \varepsilon\bigr),
\qquad \varepsilon = W\eta ,
$$

an affine map depending on the joint only through $K$ — that is, only
through its moments. If $(u, v)$ is distributed as the joint and
$\eta \sim \mathcal{N}(0, R)$ independently, then $\Phi(u,v,\eta)$ is
distributed as the posterior `condition` returns: the mean is
$m_{\text{post}}$, and

$$
\operatorname{Cov}\Phi
= C_{uu} - KC_{vu} - C_{uv}K^\top + K(C_{vv}+R)K^\top
= C_{uu} - KC_{vu},
$$

since $K(C_{vv} + R) = C_{uv}$. Each realization is transported
independently of every other, so a batch of them is one call.

**The three realization arguments follow the operator layer's batch
contract**: `u` of shape `(..., P)`, `v` and `whitened_noise` of shape
`(..., N)`, broadcasting over leading axes, result of shape `(..., P)`. They
are one of the three departures from the layer's core-shape rule
({ref}`gauss-notation`), and the departure is what makes a batch of
residuals one `gain_weights` call rather than a loop.

**Every argument is keyword-only.** `v` and `whitened_noise` are both
`(..., N)` and so freely exchangeable, and `u` joins them at $P = N$; a
positional call would turn an argument-order slip into a silent wrong
answer.

**`whitened_noise` is $\varepsilon$, not $\eta$.** The method applies
$W^{-1}$ to nothing, so the argument is a draw in whitened coordinates. This
is the mixed-representation warning of the operator contract's `whiten`
section ({doc}`linop-contract`): $WL$ has orthonormal rows but is not the
identity, so pushing one $\varepsilon$ through both `whiten` and `factor()`
in the same update corrupts the joint law of the result while every marginal
statistic still looks right. Draw it once, and use it only here.

Cost: $k$ applications of $W$ for $S$, plus one per realization. Unlike
`condition`, the residuals cannot come free from $S$ — `v` is arbitrary data
rather than the joint's own factor ({ref}`gauss-empirical` has the route
that can).

(gauss-empirical)=
## `EmpiricalJoint`

$J$ paired samples, and the two updates that carry them forward. It holds
the samples and nothing else; reading a *distribution* out of them is
`to_gaussian_joint()`.

**Fields.** `u_samples`, a `(J, P)` array, and `v_samples`, a `(J, N)`
array — row-aligned: row $j$ of each belongs to the same sample. Both are
**keyword-only**: they are arrays of the same rank agreeing on the sample
axis, so exchanging them is shape-valid whenever $P = N$ and no check can
detect it — the update is then computed from the wrong blocks and returns
finite, plausible numbers. The cost is that a family is built through a
lambda, `jax.vmap(lambda u, v: EmpiricalJoint(u_samples=u, v_samples=v))`,
rather than by mapping the constructor directly. `Gaussian` stays
positional: an array and an operator cannot be exchanged silently, since the
type check catches it. Construction validates exact rank 2 on both,
agreement of the leading axes, and $J \ge 2$ — the same checks
`GaussianJoint.from_samples` applies, shared so that the two agree on what a
sample pair is.

**Derived attributes.** `n_samples`, `u_dim`, `v_dim` are `int` properties.
`u_mean`, `v_mean`, `u_anomalies`, `v_anomalies` are array properties
computed on access from the stored samples. This is not the forbidden lazy
factorization ({ref}`contract-jax`): nothing is cached and nothing
factorizes — each access is an $O(J\cdot\mathrm{dim})$ mean-and-subtract that
fuses under `jit`. The samples are the whole state, and
`(mean, anomalies)` is recovered from them exactly.

Both updates take the trailing `y` and `noise_cov` of {ref}`gauss-joint`,
under the same preconditions, and return a `(J, P)` array of updated
samples, row $j$ updating sample $j$ — they update $u$ only, leaving a
caller who needs a matching $v$ to recompute it. Both are deterministic
functions of their arguments (`pathwise_update` counts the key among them),
safe under `jit`, and both degrade gracefully when the $v$ anomalies are
zero, returning `u_samples` unchanged. That requires the anomalies of
identical samples to be *exactly* zero, which a plain subtraction of a
summed-and-divided mean does not deliver; `GaussianJoint.from_samples` forms
them so that they are, because the alternative is not a `nan` but a wrong
finite update, of order 1 for samples of order $10^{23}$.

### `to_gaussian_joint()`

The joint Gaussian fitted to these samples' moments: exactly
`GaussianJoint.from_samples(u_samples=..., v_samples=...)`, raising
`ValueError` on a family.

This call is the layer's one crossing from samples to a distribution, and it
is deliberately explicit. Conditioning a set of samples means conditioning a
Gaussian fitted to their moments; that fit is a modelling step, and it
belongs in the caller's source text rather than inside a method name.

The conversion **loses nothing**. With the mean kept and a factor of width
$J$, the samples are recovered exactly,

$$
u_j = \bar u + \sqrt{J-1}\,\bigl(F_u\bigr)_{\cdot j},
$$

a conformance obligation, and the same identity `transform_update` uses on
the *conditioned* factor. So the map from a sample set to a mean and a
centred width-$J$ factor is a bijection. What the conversion drops is the
*interpretation* of the latent index as a sample index — which is why the two
updates below live here and not on the joint.

The joint must be **derived on each call, not stored**. Holding both
representations would hold the same numbers twice, and the derivation is an
$O(J(P+N))$ centre-and-scale that fuses under `jit`.

### `transform_update(y, noise_cov)`

The deterministic (square-root) update: the fitted joint's posterior, read
back as samples. Conditioning multiplies the factor on the right by $T$, and
because the factor is centred so is $F_uT$, so its columns are again a
sample set:

$$
u_j' \;=\; m_{\text{post}} + \sqrt{J-1}\,\bigl(F_uT\bigr)_{\cdot j}
\;=\; m_{\text{post}} + \bigl(T A_u\bigr)_j .
$$

No randomness, no `key`. The returned block is an *exact* sample
representation of the posterior: its sample mean equals the posterior mean
and its sample covariance (divisor $J-1$) equals the posterior covariance,
both in exact arithmetic — the identity of {ref}`gauss-kernel`, and the
bridge the conformance suite uses between the update and the hand-written
dense reference. In terms of {ref}`gauss-condition`, the obligation is

$$
\texttt{transform\_update}(y, R)_j
\;=\; m_{\text{post}} + \sqrt{J-1}\; F_{\cdot j},
\qquad
F = \texttt{to\_gaussian\_joint().condition}(y, R)\texttt{.cov.F},
$$

and the implementation must share that method's single decomposition rather
than computing a second one.

:::{important}
**This reading is a method here, not a function of the posterior, and the
reason is a precondition no type can carry.** It is valid exactly when the
factor is centred. Applied to a joint built any other way — `from_linear_map`,
say — it returns samples whose covariance is right and whose mean is shifted
by $\sqrt{k-1}\,F\mathbf{1}_k/k$, silently. Held on the class that owns
samples, centredness is structural: `from_samples` is the only constructor
reachable from here, and it centres. Moved to `GaussianJoint`, it would be a
value precondition detectable only in debug mode.

The failure it prevents is not hypothetical in the other direction either.
Were `transform_update` to take a sample set as an *argument*, applying $T$
to any set other than the joint's own would leave the posterior mean intact —
$\mathbf{1}^\top A' = 0$ and $T\mathbf{1} = \mathbf{1}$ both still hold — and
get the covariance wrong, finitely and without an exception. Measured on an
unrelated sample set of the same shape: mean correct to $1.1\times10^{-16}$,
covariance wrong by 59% of its own scale. The argument is already inside the
joint, so it is not offered.
:::

### `pathwise_update(key, y, noise_cov)`

The stochastic (perturbed-observation) update: Matheron's rule applied to
the samples themselves, one fresh perturbation each.

$$
u_j' \;=\; u_j + F_u\, w_j, \qquad
w_j = \texttt{gain\_weights}\bigl(S,\; b_j\bigr), \qquad
b_j = W(y - v_j) - \varepsilon_j,
$$

with $\varepsilon_j$ the rows of the **pinned draw**
`jax.random.normal(key, (J, N))` and $F_u = A_u^\top/\sqrt{J-1}$. The
perturbation enters *only* in whitened space, so equivalently $u_j' = u_j +
K(y - v_j - \eta_j)$ with $\eta_j = W^{-1}\varepsilon_j \sim \mathcal{N}(0,
R)$ — the identity the conformance suite checks elementwise against a dense
reference. Pinning the draw fixes the exact output for a given key, which
makes runs reproducible and the method testable without statistics.

It is {ref}`gauss-pathwise`'s map on the joint's own realizations, and takes
the cheaper route to them. Because the samples *are* the fitted factor, each
whitened residual follows from the whitened factor,

$$
W(y - v_j) \;=\; W(y - \bar v) - \sqrt{J-1}\,S_{j\cdot},
$$

so the update spends $J + 1$ applications of $W$, not $2J$. Routing the same
samples through `GaussianJoint.pathwise` computes the same thing to
round-off and spends $2J$; the two are conformance-checked against each
other, and the counts are conformance-checked apart.

:::{warning}
The method **neither accepts nor exposes the perturbations**: no `eps`
argument, no perturbed observations in the return value. This is the
mixed-representation warning of the operator contract's `whiten` section
({doc}`linop-contract`), made structural: a perturbation used through the
whitened shortcut must never also be pushed through `factor()` in the same
update, and the surest way to prevent that is for the same code to own both
the draw and its single use. A caller who needs the perturbations
materialized wants `GaussianJoint.pathwise`, which takes them explicitly, or
the conditioning primitives.
:::

`pathwise_update`'s output has sample mean and sample covariance (divisor
$J-1$) that are *unbiased estimators* of the posterior moments
`transform_update` represents exactly. The sample mean has variance exactly
$K R K^\top / J$; the sample covariance fluctuates at the usual
$O(J^{-1/2})$ rate, with entrywise standard deviation of order
$\bigl(C^{\text{post}}_{ii}(KRK^\top)_{jj} +
C^{\text{post}}_{jj}(KRK^\top)_{ii} +
2(KRK^\top)_{ij}^2\bigr)^{1/2}\!/\sqrt{J}$ — in relative terms
$\sqrt{J}$ times looser than the mean's. The unbiasedness of the covariance
is particular to the $J-1$ divisor, whose centring of the perturbations
cancels exactly. Individual pathwise samples are not posterior draws —
conditional on the sample block, sample $j$ is distributed
$\mathcal{N}\bigl(u_j + K(y - v_j),\, K R K^\top\bigr)$ — so no "exact in
distribution" claim is made. Which update to use is the calling layer's
decision, not this layer's.

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
- Everything else is deterministic. There is exactly one source of randomness
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
| 2. construction | ranks, static sizes, operator types, cross-field shape agreement; a vmapped-family `cov` or factor row block | `u_samples` rank ≠ 2; $J = 1$; `cov` not a `PSDLinOp`; a factor row block not a `LinOp`; `mean` and `cov` sides disagreeing; the two row blocks disagreeing on $k$ |
| 3. call | operand core shapes, operator arguments, and static non-array arguments; a vmapped-family `noise_cov` — `ValueError` for shape violations, `TypeError` for type violations, per the taxonomy below | `y` not `(N,)`; `noise_cov` side ≠ $N$; `noise_cov` not a `PSDLinOp`; primitive operands mis-shaped; `n_samples` not a positive `int` |
| 4. value (debug) | finiteness of `u_samples`, `v_samples` and the means at construction; of `y`, `x`, `pathwise`'s three realization arguments, and the primitives' `s` and `b` at call; of every conditioning method's *returned* value | violations yield `nan` or a silently wrong posterior outside debug mode |

Tier-1 (field declaration) is inherited with the class machinery
({ref}`gauss-jax`). Error messages follow the operator contract's
obligations: name the object (its `repr`), the method, the expectation, and
the offending value's shape or type.

Within a method the checks run in the operator layer's order: the family
guard ({ref}`gauss-jax`) first, then the required-capability checks in the
order the method names them, then tier-3 operand and operator-argument
validation, then — in debug mode — tier-4 value checks.

Result checks run **last**, once there is a result to check. This is the
layer's one tier-4 *postcondition*: every conditioning method asserts in
debug mode that what it returns is finite — the updated sample block, the
transported realizations, or, for `condition`, both the posterior mean and
the covariance factor, checked before the `PSDLowRank` and the `Gaussian`
are built, so the diagnosis names the conditioning call rather than a
constructor below it. Three points fix its scope:

- **Why this layer checks outputs when the operator layer does not.** A
  conditioning result is fed back in as the next call's input: a `nan` block
  is handed straight to whatever expensive computation produces the next
  :math:`v`, and a computation that returns finite nonsense for `nan` inputs
  launders it beyond recovery. An operator's result goes back to the caller who
  asked for it. The asymmetry is deliberate, and is not an argument for adding
  output checks to `pyeki.linalg`.
- **It is the only cheap detection of a singular `noise_cov`.** Every
  conditioning method behaves identically under it. Before this rule
  `condition` alone raised, because it happened to route its mean through a
  constructor — and even then it left the covariance factor unchecked, so the
  check covered half of what it appeared to guard.
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
| a vmapped-family `cov` or factor row block at construction, or `noise_cov` at call | `ValueError` |
| the two factor row blocks disagreeing on the latent width | `ValueError` |
| `pathwise` called positionally | `TypeError`, from Python |
| violated value precondition | `ValueError` in debug mode; `nan` or a silently wrong result otherwise |
| a non-finite conditioning result (typically a singular `noise_cov`) | `ValueError` in debug mode, from the method; `nan` otherwise |

(gauss-jax)=
## JAX integration

All three classes are frozen-dataclass pytrees declared with the same
machinery as operators, and every rule of the operator contract's JAX section
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
- **Constructors store and validate; classmethods compute.**
  `GaussianJoint.from_samples` centres and scales, and `from_linear_map`
  materializes $GL$; the plain constructor only stores.
  {ref}`gauss-joint` records why the SVD cannot live at construction either
  way.
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

- **Every class has a required `batch_shape` property**, computed like
  the operators' — each array field contributes its leading axes beyond
  its core rank, and an operator field contributes its own `batch_shape`,
  combined by broadcasting with `ValueError` on mismatch. Core ranks:
  `Gaussian.mean` 1 (with `cov` contributing its own); `GaussianJoint`'s two
  means 1 (with each row block contributing its own); `u_samples` and
  `v_samples` 2. Directly constructed objects always report `()`.
- **Genuine construction rejects families**: `Gaussian(mean, cov)` with
  `cov.batch_shape != ()` is a tier-2 `ValueError`, as is a `GaussianJoint`
  row block with a non-empty `batch_shape` (the batch shape is static
  information), and a family `noise_cov` at a conditioning call is a tier-3
  `ValueError` — apply a family of noise operators with `jax.vmap`, not
  directly.
- **Inertness**: when `batch_shape` is non-empty, every method and every
  array-computing property (`sample`, `log_density`, `condition`,
  `pathwise`, the two updates, `to_gaussian_joint`, the marginals, the means
  and the anomalies) raises `ValueError` naming the object, the operation,
  the batch shape, and the remedy — apply the family under `jax.vmap` —
  before any capability or operand check. The static `int` properties (`n`,
  `n_samples`, `u_dim`, `v_dim`, `latent_dim`), `batch_shape` itself, and
  `repr` still answer — and the size properties report **core** (trailing)
  sizes, never batch sizes, exactly as an operator's `shape` does:
  `Gaussian.n` is `mean.shape[-1]`, `EmpiricalJoint`'s three are
  `u_samples.shape[-2]`, `u_samples.shape[-1]` and `v_samples.shape[-1]`,
  and `GaussianJoint.latent_dim` is `u_factor.shape[1]`.
- **Family repr** wraps the ordinary form, as for operators; the form and
  the never-raises rule are in {ref}`gauss-repr`.

(gauss-consumers)=
## How the layers above consume this one

Not normative for `pyeki.gauss` itself, but the design was shaped against
these call sites, and a change that breaks them is a change to reconsider.

**The EKI driver** (`pyeki.eki`) is a loop over tempering steps.
Per step: hold the ensemble and its predictions as a sample pair, update,
re-evaluate the forward model. It consumes `EmpiricalJoint`'s two updates
and nothing else; `GaussianJoint` is reached only inside them.

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
internal entry point on `GaussianJoint`, which is where such a thing now
belongs and which this layer does not yet expose ({ref}`gauss-excluded`).

Second, forward-model failures are handled by *sample preprocessing*, not
by a masked joint: with validity mask $m_j$ and $J_v \ge 2$ valid
members, replacing each member by
$\hat u + m_j\,(u_j - \hat u)\,\sqrt{(J-1)/(J_v-1)}$ (with $\hat u$
the valid-member mean, and the *same* mask applied to $v$ — a failed
evaluation invalidates the pair) makes the fixed-$J$ joint's moments,
cross-covariance included, equal the masked moments exactly, and keeps
shapes static under `jit`. Failed members rejoin at the posterior mean
under `transform_update`; under `pathwise_update` they
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
dense linear algebra over means, anomalies and materialized operators,
independent of this layer's code — and every conditioning method is checked
against them ({ref}`gauss-conformance`).

Two fixtures reach analytic posteriors. `from_linear_map` reaches them
directly: a `Gaussian` prior and a linear map give a joint whose closed-form
posterior the tests compute densely, and at a rank-deficient prior factor
there is no precision form to compare against, which is the point.
Exact-moment sample sets reach them through the sample path: for any target
covariance with a factor $F$ of width $k$, a set of $J \ge k + 1$ samples
whose empirical moments equal the target's exactly can be constructed — take
the complete QR of $\mathbf{1} \in \mathbb{R}^J$, let $E$ be its last $k$
columns (orthonormal, each $\perp \mathbf{1}$), and set the rows to
$\mu + \sqrt{J-1}\,E\,F^\top$. Only $J \ge k + 1$ binds (at $J = k$ the
construction fails, and silently); reducing a wide factor first — a thin QR
of $F^\top$, or an eigendecomposition of $FF^\top$, never a Cholesky, which
raises on the rank-deficient targets that matter — merely lowers the sample
count that condition demands.

The same construction is what makes `pathwise`'s distributional claim
checkable *exactly* rather than by Monte Carlo: build realizations of
$(u, v, \eta)$ together, from the block factor
$\bigl(\begin{smallmatrix} F_u & 0 \\ F_v & 0 \\ 0 &
R^{1/2}\end{smallmatrix}\bigr)$ of width $k + N$, so that $\eta$ has
covariance exactly $R$ and exactly zero cross-covariance with the pair. The
map is affine, so exact input moments give exact output moments.

A tempered run's posterior telescoping to the one-shot posterior is the EKI
layer's test; the per-step exactness it composes from lives here.

(gauss-repr)=
## `repr`

Type name and static sizes, never array contents, matching the operator
rule ({ref}`contract-repr`): `Gaussian(n=12)`,
`GaussianJoint(u_dim=12, v_dim=40, latent_dim=100)`,
`EmpiricalJoint(n_samples=100, u_dim=12, v_dim=40)`. A vmapped family wraps
that form and names its batch —
`vmapped(EmpiricalJoint(n_samples=100, u_dim=12, v_dim=40), batch=(8,))`
({ref}`gauss-jax`) — and `repr` never raises: an instance whose sizes
cannot be read falls back to a marker form, unspecified beyond its not
raising.

(gauss-surface)=
## Public surface

`pyeki.gauss` exports exactly: the classes `Gaussian`, `GaussianJoint` and
`EmpiricalJoint`, including `Gaussian.from_samples` and `GaussianJoint`'s
three constructors, and the conditioning primitives `gain_weights` and
`sqrt_transform`.
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
   `s`, and $T\mathbf{1} = \mathbf{1}$ **for `s` from a centred factor**
   ($s^\top\mathbf{1} = 0$) to a tolerance of
   $c_1 J \varepsilon + c_2 (\varepsilon \sigma_{\max})^2$; for general
   general `s` no such identity holds ({ref}`gauss-primitives`). Both terms are
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
   the same moments; and the two satisfy the elementwise identity
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
   definition; `log_density` matches the dense closed form at batch ranks 0, 1,
   and 2 and differentiates. `from_samples` reproduces the sample mean and the
   $J-1$ covariance exactly against a dense reference, agrees with
   `GaussianJoint.from_samples`'s $u$ marginal on the same samples, holds a
   `PSDLowRank` of width $J$ that withholds `solve`, `whiten` and `logdet`,
   gives identical samples exactly zero spread, and validates rank and sample
   count.
8. **Degeneracy**: zero prediction anomalies — every row of `v_samples`
   given the *same, exactly representable* value, since a collapsed
   block of arbitrary values leaves anomalies at $O(\varepsilon)$
   rather than bit-zero — make both updates the identity on `u_samples` —
   bit-exact for `pathwise_update`, to round-off for `transform_update`,
   which reconstructs $\bar u + a_j$ — and `condition` return the $u$
   marginal's own moments; $J = 2$ and $N = 1$ work; a collapsed sample block
   with finite inputs produces no `nan`.
9. **Capability propagation** (`PSDLowRank` is the one shipped
   `PSDLinOp` that disclaims operations, and it covers the `whiten` and
   `logdet` cases; the `factor` case needs a test-local `PSDLinOp`
   implementing no `_factor`): a noise covariance without `whiten`, and a
   covariance without `factor` or `logdet`, raise `UnsupportedOpError`
   from the conditioning methods, `sample`, and `log_density`
   respectively — and `log_density` on the posterior `condition` returns
   raises the same way, regardless of $k$ versus $P$.
10. **Validation**: every tier-2 and tier-3 rule of
    {ref}`gauss-validation` raises as specified.
11. **JAX round trips**: flatten/unflatten preserves type and behaviour for
    all three classes; the conditioning methods run under `jit`; constructing
    an `EmpiricalJoint` inside `vmap` round-trips and a `vmap`-ed family of
    joints agrees with a Python loop; sentinel-leaf unflattening succeeds.
12. **Reproducibility and repr**: same key, same output, elementwise;
    different keys differ; reprs match {ref}`gauss-repr` with no array
    data; and the pinned draws are snapshotted so a JAX-side PRNG stream
    change is detected ({ref}`gauss-prng`).
13. **Families**: stacking any class's leaves and unflattening yields a
    family that reports its batch shape, takes the `vmapped(...)` repr
    form, and refuses every method and array-computing property with the
    family `ValueError`, while the static `int` properties and
    `batch_shape` still answer; genuine construction with a family
    covariance or factor row block raises ({ref}`gauss-jax`).
14. **The joint factor and its constructors**: `from_linear_map`'s posterior
    equals the dense block form, and — where the prior is invertible — the
    precision form $(C_0^{-1} + G^\top R^{-1}G)^{-1}$; at a rank-deficient
    prior factor ($k < P$) it equals the dense form, the case no precision
    form covers. `from_linear_map` leaves the $u$ row block the operator
    `factor()` returned and materializes only the $v$ block, and whitens
    $k + 1$ vectors. `from_samples` reproduces all three moment blocks. The
    projection is lossless: $u_j = \bar u + \sqrt{J-1}(F_u)_{\cdot j}$
    elementwise, and the factor it builds is centred. `from_factors` wraps a
    bare array and leaves an operator alone.
15. **The pathwise map**: `pathwise` equals the dense
    $u + K(y - v - W^{-1}\varepsilon)$ elementwise; on realizations of
    $(u, v, \eta)$ whose empirical moments equal the target joint's exactly
    ({ref}`gauss-consumers`), the transported sample moments equal the
    closed-form posterior's; it agrees with `pathwise_update` to round-off on
    the joint's own samples, and the two whitener counts differ as
    {ref}`gauss-empirical` states.

Beyond the tests, the implementation PR owes two deliverables named
earlier: the layer's user-guide page ({ref}`gauss-scope`) and the
`PSDLowRank` operator with its operator-contract entry
({ref}`gauss-condition`).

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

**A block-represented joint class.** An earlier draft of this contract had
one — mean vectors plus covariance blocks $(C_{uu}, C_{uv}, C_{vv})$ as
independent operators. `GaussianJoint` is the re-entry that draft's own notes
pointed to, and it is a *factor* joint rather than a block one, for three
reasons the block form cannot answer.

The kernel needs a shared latent, and recovering one from blocks means
solving $F_u F_v^\top = C_{uv}$, that is
$F_u = C_{uv}F_v(F_v^\top F_v)^{-1}$ — which forms the $k \times k$ Gram and
squares the condition number, the very defect that rejected the Woodbury
route below. Measured: a factor with $\kappa(F_v) = 7.7$ whose last column
is scaled by $10^{-8}$ has $\kappa(F_v^\top F_v) = 1.3\times10^{32}$, and the
recovered $F_u$ is wrong by a relative factor of $1.5\times10^{8}$ — no
correct digits at all. It also requires $C_{uv}$, a matrix of *both* block
dimensions, which no code path in the layer forms. And the joint's
PSD-ness cannot be typed: a valid joint needs
$\operatorname{col}(C_{vu}) \subseteq \operatorname{col}(C_{vv})$,
which three independently supplied operators do not satisfy in general and no
operator type can assert — composition never proves PSD-ness.

A joint factor has none of these problems: coherence is structural, nothing
is squared, no matrix of both dimensions appears, and $C = FF^\top$ is PSD by
construction. It also avoids the block form's second dead end, a downdate
operator (`PSDDowndate(base, F)` for $\mathrm{base} - FF^\top$) whose
`factor` and `whiten` require solving against the base's own factor, which
only some base types can do — the posterior is $(F_uT)(F_uT)^\top$ instead,
a `PSDLowRank` and nothing more.

What remains excluded is a **dense reference implementation**: the oracle
stays hand-written in the tests, precisely so that no package code is
trusted ({ref}`gauss-conformance`).

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
conditioning calls on the same `(joint, noise_cov)` pair would need a
returned decomposition object. `GaussianJoint` is where it would live, and
`transform_update` already shows the shape of the need — it reuses
`condition`'s decomposition, through a private helper rather than a public
object. That suffices while the sharing stays inside the layer. The one
prospective *external* consumer is an adaptive-step search, which could reuse
a single SVD across candidate increments ({ref}`gauss-consumers`); it pays
one SVD per candidate until that consumer exists and justifies the
object.

**Batched observations.** `y` is one observation vector. A family of
updates — multiple observations, multiple noise levels — is `jax.vmap`
over the method, consistent with the layer-wide batching story.

**A sample-set argument on `transform_update`.** It would be redundant when
correct and silently wrong otherwise; {ref}`gauss-empirical` gives the
arithmetic and the measured error.

**A configurable anomaly divisor.** $J - 1$ everywhere. A $1/J$ convention
changes every formula's scaling for no consumer — the masked-sample
consumer is served by sample preprocessing in `pyeki.eki`
({ref}`gauss-consumers`); inflation, which is the principled way to widen
a set of samples, belongs to `pyeki.eki`.

**A marginal-likelihood accessor.**
$\log\det(C_{vv} + R) = \log\det R + \sum_i \log(1 + \sigma_i^2)$
falls out of a conditioning call's own SVD, and a future evidence or
tempering-diagnostic consumer may want it. It would belong on
`GaussianJoint`, alongside `v_marginal`. Excluded until that consumer
exists; recorded so its absence reads as a decision and the identity is
not rediscovered.

**Perturbation injection into `pathwise_update`.** It takes no `eps`
argument; see the warning in {ref}`gauss-empirical`. Determinism needs are
met by the pinned key-derived draw, and a caller who genuinely needs to
supply perturbations uses `GaussianJoint.pathwise`, where the whitened
representation is pinned by the parameter's name and documented obligation.

**Non-Gaussian anything.** No mixtures, no transformations of variables,
no likelihoods other than additive Gaussian noise. The layer is the
Gaussian core of EKI, not an inference framework.
