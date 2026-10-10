# Distribution contract

This page specifies `enskit.distribution`: the classes and functions it
provides, the contract of every method, and the conditioning mathematics they
share. It is normative: an implementation that violates a rule here is
defective even if its tests pass. It is the reference for two audiences:
contributors implementing or reviewing the layer, and users who want a more
precise account of distributions, conditioning and conditional maps in EnsKit
than the user guide gives.

Throughout, *must* and *never* state requirements, *should* states a strong
default that a documented reason may override, and *may* states a
permission. The layer is built on the operator layer, and this page refers to
{doc}`linop-contract` rather than restating its rules. {doc}`joint-factor`
derives the factor representation this layer generalizes, and
{doc}`redesign/index` records the design this page makes precise.

:::{admonition} Status: implemented
:class: note

This page was written and reviewed before the code, as the operator
contract was, and PR 4 of the redesign plan implemented it in
`enskit.distribution`. Where implementing it showed the page to be wrong or
incomplete, the page was corrected in the same pull request; each change is
listed in {ref}`dist-implementation-changes`. The page absorbs the joint Gaussian
contract that governed the retired `enskit.gauss`, which PR 7 deleted with
that module; its regression tests are ported in {ref}`dist-conformance`.

Where this page departs from the design's stubs (`docs/redesign/stubs/`),
it says so in {ref}`dist-departures`, and this page wins.
:::

(dist-scope)=
## Scope

The layer represents probability distributions over **named blocks**, and
provides the operations on them that ensemble Kalman methods are built from:

- **two distributions**: an empirical distribution of particles, optionally
  weighted (`Ensemble`), and a joint Gaussian held as a shared factor plus
  an independent term per block (`Gaussian`), with a subclass whose latent
  coordinates are an ensemble's particles (`EnsembleGaussian`);
- **the crossings between them**: moment matching (`Ensemble.project`),
  sampling (`Gaussian.sample`), and reading an `EnsembleGaussian`'s particles
  back out (`realize_particles`);
- **exact Gaussian algebra**: marginals, adding independent noise, moving an
  independent term into the shared factor, re-factoring, conditioning on
  noisy or exact values, and log densities;
- **conditional maps**: the pointwise affine map that carries samples of a
  Gaussian joint to samples of its conditional (`MatheronMap`), and the map
  that moves an `EnsembleGaussian`'s own particle set (`SquareRootMap`);
- **weights**: reweighting in log space, the effective sample size, and
  resampling;
- **one fixture**: particles whose sample moments equal a Gaussian's exactly
  (`exact_moment_ensemble`).

What the layer is not:

- It has no notion of a map from one block to another, a simulator, an
  update, an observation or time. Pushing a distribution through a map is
  `enskit.maps`; an ensemble Kalman update is `enskit.kalman`. A
  block becomes a *given* block only at the moment it is conditioned on.
- It has no densities over anything non-Gaussian, no mixtures, no
  transformations of variables, and no inference loop.
- It supplies no dense reference implementation. The reference everything is
  tested against is hand-written in the test suite ({ref}`dist-conformance`).

Every conditioning path goes through one computation, the **conditioning
core** of {ref}`dist-core`: one whitened factor, one
{class}`~enskit.linalg.IdentityPlusGram`, one thin singular value
decomposition. The algebraically equivalent routes that form a Gram matrix
are rejected ({ref}`dist-excluded`).

(dist-notation)=
## Notation and conventions

| symbol | meaning |
| ------ | ------- |
| $b$, $c$, $x$, $y$ | block names; $c$ ranges over conditioned (*given*) blocks, $x$ over *target* blocks |
| $d_b$ | the dimension of block $b$ |
| $N$ | the total dimension of the given blocks, $\sum_c d_c$ |
| $k$ | the latent width, `latent_dim` |
| $J$ | the number of particles, `n_particles` |
| $m_b$ | block $b$'s mean, a $(d_b,)$ vector |
| $F_b$ | block $b$'s row of the shared factor, a $(d_b, k)$ operator |
| $D_b$ | block $b$'s independent term, a $(d_b, d_b)$ PSD operator |
| $F_c$, $D_c$, $m_c$ | the given blocks' factor rows stacked, their independent terms block-diagonally, their means stacked |
| $W$ | a whitener of $D_c$: the fixed matrix the operators' `whiten` applies, $W D_c W^\top = I_N$ |
| $S$ | the whitened factor $(W F_c)^\top$, $(k, N)$ |
| $U\Sigma V^\top$ | the thin SVD of $S$, with $\rho = \min(k, N)$ singular values $\sigma_1 \ge \dots \ge \sigma_\rho \ge 0$ |
| $A$ | $I_k + S S^\top$ |
| $y^*$ | a value for the given blocks |
| $\varepsilon$ | the float64 unit roundoff, $2^{-52} \approx 2.2 \times 10^{-16}$ |

Conventions, each normative:

- **Blocks are named vectors.** A block is a `str` name and a dimension
  $d_b \ge 1$. A block's value is a `(d_b,)` array. Names are ordered: the
  order in which blocks were given at construction is the order every method
  reports them in, and the order of the result of every method that does not
  say otherwise.
- **Particles are stored row-wise.** An ensemble block is a `(J, d_b)` array,
  row $j$ being particle $j$. The particle axis is a batch axis to the
  operator layer ({ref}`contract-batch`), so `op.matvec(ens["x"])` applies an
  operator to every particle and `D.whiten(y - ens["g"])` whitens every
  residual.
- **Factors are column-wise.** A factor of a $d$-dimensional covariance is a
  $(d, k)$ operator, as `PSDLinOp.factor()` returns it. Where particles become
  a factor, the conversion carries a transpose and the divisor's square root:
  $F_b = A_b^\top / \sqrt{\delta}$ for the $(J, d_b)$ anomalies $A_b$.
- **Anomalies are raw deviations from the mean.** No normalization is folded
  into them. The divisor of a covariance computed from particles is
  specified once, in {ref}`dist-divisor`.
- **Samples and particles.** A *sample* is a draw from any distribution. A
  *particle* is an element of an `Ensemble`. Where a function asks for
  samples, an ensemble's particles are those samples.
- **Distributions are unbatched.** A family of distributions is a
  `jax.vmap` over the pytree, never stored batch axes ({ref}`dist-jax`). The
  methods that conceptually take one value (`condition`, the maps' given
  values) require exactly `(d_b,)`; a family of values is a `vmap`. One
  argument takes leading batch axes, as the operator layer's operands do: the
  values of `Gaussian.log_density`, so that one density is evaluated at many
  points in one call.
- **Scalars are returned as 0-d JAX arrays**, never Python floats.

(dist-divisor)=
### The divisor

Every covariance computed from particles has the form

$$
\hat C_{ab} = \frac{1}{\delta_w}\sum_{j=1}^J w_j\, a^{(a)}_j \big(a^{(b)}_j\big)^\top ,
\qquad
\delta_w = \begin{cases}
1 - \sum_j w_j^2 & \text{unbiased (the default)},\\
1 & \text{empirical},
\end{cases}
$$

with $a_j$ the anomalies and $w_j$ the normalized weights. Unweighted,
$w_j = 1/J$, and this is $\frac{1}{\delta}\sum_j a_j a_j^\top$ with

$$
\delta = J\,\delta_w = \begin{cases} J - 1 & \text{unbiased},\\ J & \text{empirical}. \end{cases}
$$

- **The unbiased divisor is the default everywhere.** It treats the
  particles as independent samples: unweighted, $\hat C$ is unbiased for
  their covariance; weighted, $1 - \sum_j w_j^2$ is the same correction for
  fixed weights (self-normalized importance weights make $\hat C$ only
  consistent). Uniform weights give the unweighted divisor,
  $J(1 - 1/J) = J - 1$.
- **The empirical divisor** gives the moments of the empirical distribution
  $\hat p = \sum_j w_j\delta_{x_j}$ itself. It is the right one when the
  particles are the atoms of a deterministic rule rather than samples, such
  as a quadrature or sigma-point rule with nonnegative weights (one set of
  weights for both moments) that reproduces a mean and covariance exactly.
- **The override** is a keyword-only `unbiased: bool = True`, taken by
  `Ensemble.cov` and `Ensemble.project`, and in the layer above by
  `enskit.maps.statistical_linearization`. Nothing else takes it: the
  Kalman layer's approximation, `exact_moment_ensemble` and the algorithms'
  diagnostics always use the unbiased divisor. Any other value than a
  `bool` raises `TypeError`.
- **An `EnsembleGaussian` carries its divisor** ({ref}`dist-ensemble-gaussian`):
  its factor rows are $A_b^\top/\sqrt\delta$, and every operation that
  reads particles out of them, $a^{(b)}_j = \sqrt\delta\,F_b e_j$, uses the
  stored $\delta$. A reader that assumed $J - 1$ of a projection formed with
  $J$ would return particles scaled by $\sqrt{(J-1)/J}$ about their mean,
  without raising.

(dist-block-arguments)=
### Block arguments

Every argument that is a set of block values or covariances accepts a
positional-only mapping, keywords, or both:

```python
g.condition({"y": y_obs})      # mapping
g.condition(y=y_obs)           # keywords
g.condition({"y": y_obs}, z=z_obs)  # both
```

- The mapping form is always available, and is required for a name that is
  not a Python identifier or that collides with one of the method's own
  parameters (`key`, for instance).
- A block given in both forms raises `TypeError`, as a repeated keyword does
  in Python.
- A name that is not a block raises `KeyError`, with a message listing the
  blocks.
- Where the order of the given blocks matters to a result (the column order
  of $S$, the layout of a perturbation draw), it is the **distribution's
  block order**, not the order the arguments were written in, so a set of
  block values gives the same result however it was spelled. (Methods that
  take names to *select* blocks, such as `marginal`, return them in the
  order given, as they say.)

The same rules govern every method below that takes names as positional
arguments (`marginal("u", "g")`): unknown names raise `KeyError`, and a name
repeated raises `ValueError`.

(dist-objects)=
## The objects

| object | represents | constructed by |
| ------ | ---------- | -------------- |
| `Ensemble` | $J$ particles over named blocks, optionally weighted | its constructor; `Gaussian.sample`; `realize_particles`; the maps; `resample`; `exact_moment_ensemble` |
| `Gaussian` | a joint Gaussian over named blocks: shared factor plus independent terms | its constructor; `Gaussian.independent`; `Ensemble.project` (weighted); every Gaussian method |
| `EnsembleGaussian` | a `Gaussian` whose latent coordinate $j$ is particle $j$ | `Ensemble.project` (unweighted); its constructor; the methods that keep the latent space |
| `ConditionalMap` | protocol: a pointwise map from samples of a joint and a value $y^*$ to samples of the conditional | user code; `MatheronMap` is the shipped instance |
| `MatheronMap` | the exact conditional of a `Gaussian`, $x \mapsto x + K(y^* - y)$ | `Gaussian.conditional_map` |
| `SquareRootMap` | the square-root reading of an `EnsembleGaussian`'s conditional, for its own particles | `EnsembleGaussian.square_root_map` |
| `Regression` | the conditional of one block given others, as coefficients, an intercept and a residual covariance | `Gaussian.regression` |
| `reweight`, `effective_sample_size`, `resample` | operations on weights | — |
| `exact_moment_ensemble` | particles whose sample moments equal a Gaussian's exactly | — |

Four rules govern the set:

1. **Distributions return distributions; maps return particles.** Every
   `Gaussian` method returns a `Gaussian`, an `Ensemble`, an operator, an
   array, or a `Regression`, which holds only operators and an array;
   nothing returns a decomposition object. The two maps hold
   everything that does not depend on the given value, built once, and are
   called with the value.
2. **The Gaussian approximation is written at the call site.** Conditioning
   particles means conditioning a Gaussian fitted to them, and
   `Ensemble.project()` is where that fit happens. There is no `condition` on
   `Ensemble`.
3. **Particle alignment is a type.** Reading particles out of a Gaussian, and
   moving them as a set, are valid only for a factor whose latent coordinates
   are particles. Those two operations exist only on `EnsembleGaussian`
   ({ref}`dist-ensemble-gaussian`); a `Gaussian` built any other way does not
   have them, so the precondition is checked by the type rather than by a
   value check.
4. **The set is closed except for one protocol.** Users do not subclass
   `Ensemble`, `Gaussian` or `EnsembleGaussian`. User extensibility lives in
   the operators supplied as factor rows and independent terms (any operator
   that passes `check_operator` works here unchanged) and in the
   `ConditionalMap` protocol, which a fitted nonlinear transport map
   implements.

(dist-representation)=
## The Gaussian representation

A `Gaussian` over blocks $b$ is

$$
x_b = m_b + F_b\,\xi + e_b, \qquad \xi \sim \mathcal N(0, I_k), \qquad
e_b \sim \mathcal N(0, D_b),
$$

with the $e_b$ independent of $\xi$ and of each other, so that

$$
\operatorname{cov}(x_a, x_b) = F_a F_b^\top \quad (a \ne b), \qquad
\operatorname{cov}(x_b, x_b) = F_b F_b^\top + D_b .
$$

The shared latent vector $\xi$ is what keeps every cross-covariance consistent
by construction ({doc}`joint-factor` derives this for two blocks). The
independent terms are kept apart from the factor because they carry the
structure the operator layer exists to exploit: a noise covariance used only
through `whiten`, a prior with a Kronecker or diagonal form that is never
densified.

**Each block may lack a factor row or an independent term, but not both.**

- A block with no factor row is independent of $\xi$: `factor(name)` returns
  `None`. A missing row is `None`, never a width-0 operator or a row of
  zeros; the operator layer has no empty operators ({ref}`contract-zero`).
- A block with no independent term has $D_b = 0$: `block_cov(name)` returns
  `None`.
- A block with neither would be a point mass. It is rejected at construction
  (`ValueError`); a known constant is not a block of a distribution.

**The latent width $k$ is static metadata**, an `int` $\ge 0$. It is inferred
from the factor rows when they exist, and otherwise given (`latent_dim=`) or
zero. A marginal over blocks that have no factor row keeps the joint's $k$,
so that later operations on the same latent space agree on it. At $k = 0$ no
block has a factor row.

**A factor is centered** when $F_b \mathbf 1_k = 0$ for every block. That is
a property of the representation, not of the distribution. An
`EnsembleGaussian`'s factor is centered, and its latent coordinate $j$ is
particle $j$'s ({ref}`dist-ensemble-gaussian`).

What the representation asks of its operators is narrow:

| operand | operations used |
| ------- | --------------- |
| $F_b$, a target's row | `matvec`, `matmat` (applying it to latent vectors), `rmatvec` through products, `to_dense` where a method says it materializes the row |
| $F_c$, a given block's row | `to_dense`: the SVD needs $W F_c$ as an array, which no representation avoids |
| $D_c$, a given block's term | `whiten`, `whiten_mat`; `logdet` for densities |
| $D_b$, any term that is sampled or absorbed | `factor` |

Each method names the operations it needs, checks them before doing any work,
and raises the operator layer's `UnsupportedOpError` unmodified when one is
missing ({ref}`dist-validation`).

(dist-core)=
## The conditioning core

All conditioning on noisy values is one computation, specified here once.
`condition`, `log_density`, `conditional_map` and `square_root_map` are
assemblies of it.

### The whitened factor

For given blocks $c$, every one of which has an independent term, stack their
factor rows into the $(N, k)$ array $F_c$ (a block with no factor row
contributes zero rows) and their terms block-diagonally into $D_c$. With $W$
the block-diagonal whitener whose block $c$ is the matrix `D_c.whiten`
applies,

$$
S = (W F_c)^\top \in \mathbb R^{k \times N}, \qquad
S = U \Sigma V^\top \ \text{(thin SVD)}, \qquad A = I_k + S S^\top .
$$

The SVD is computed by constructing `IdentityPlusGram(S)`
({ref}`contract-gram`), **once per build or call that needs it**, and every
quantity below is read from that one object. No code in this layer computes
an SVD of its own, and none forms $S S^\top$ or $S^\top S$: forming either
squares the condition number and rounds away every
$\sigma_i < \sqrt{\varepsilon}\,\sigma_{\max}$, which are the singular values
with the largest gain multipliers.

### The gain

The Kalman gain $K = \operatorname{cov}(x, y)\operatorname{cov}(y, y)^{-1}$
applied to a residual $r = y^* - m_c$ satisfies

$$
K r = F_x w, \qquad
w = A^{-1} S\, W r
  = U \operatorname{diag}\!\Big(\frac{\sigma_i}{1 + \sigma_i^2}\Big) V^\top W r
  \in \mathbb R^k,
$$

which is `IdentityPlusGram(S).solve_factor(W r)`. The change to block $x$'s
mean is a combination of its own factor's columns, so no $(d_x, N)$ matrix is formed. In
closed form $w = F_c^\top (F_c F_c^\top + D_c)^{-1} r$, which exhibits two
properties:

- **Whitener invariance.** $w$ depends on the independent terms only through
  $D_c$, not through which valid $W$ each operator chose. Results must not
  depend on that choice beyond round-off.
- **Unconditional boundedness.** $\sigma/(1 + \sigma^2) \le 1/2$, so the gain
  cannot blow up however collapsed $S$ is, with no regularization parameter.

### The square-root transform

$$
T = A^{-1/2} = I_k + U\big((I_\rho + \Sigma^2)^{-1/2} - I_\rho\big) U^\top ,
$$

which is `IdentityPlusGram(S).inverse_sqrt()`, an operator costing
$O(k\rho)$ to apply. The identity on the complement of $U$'s columns is
required: without it the thin form is singular whenever $\rho < k$. By the
push-through identity $S(I_N + S^\top S)^{-1} S^\top = I_k - A^{-1}$,

$$
(F_x T)(F_x T)^\top = F_x A^{-1} F_x^\top
  = F_x F_x^\top - K \operatorname{cov}(y, x),
$$

so $F_x T$ is a factor row of the conditional. Conditioning is the map
$(m_x, F_x) \mapsto (m_x + F_x w,\ F_x T)$ on every target block, from one
SVD; the targets' independent terms are unchanged.

Two structural facts, both exact in exact arithmetic:

- **The transform preserves centering.** If the factor is centered then
  $S^\top \mathbf 1 = W F_c \mathbf 1 = 0$, so every $U_{\cdot i}$ with
  $\sigma_i > 0$ is orthogonal to $\mathbf 1$ and the modifier vanishes at
  $\sigma_i = 0$: $T \mathbf 1 = \mathbf 1$, and $F_x T$ is centered. In
  floating point $T\mathbf 1 = \mathbf 1$ holds to the tolerance of
  {ref}`dist-conformance`'s obligation 3, which grows like
  $(\varepsilon\sigma_{\max})^2$ at large $\sigma_{\max}$.
- **$T$ is symmetric**, which is what makes the square-root reading of
  {ref}`dist-square-root-map` preserve the particles' mean
  (Wang et al., 2004).

### Centering and whitening order

Centering and differencing **must** happen before whitening. For each given
block the factor row and the mean residual are whitened together, in one
call: `D_c.whiten_mat` of the $(d_c, k + 1)$ array $[F_c \mid y^*_c - m_c]$.
Whitening first and then subtracting loses accuracy in proportion to
$\sqrt{\kappa(D_c)}$ when the mean lies along a precise direction of the
noise: $9 \times 10^{-6}$ against $10^{-12}$ at $\kappa = 10^{10}$
({doc}`joint-factor` derives the ratio). A projected factor is centered at
construction, which settles the factor's half structurally; only the residual
remains, and it is differenced before whitening.

Where an `EnsembleGaussian`'s particles' own residuals are needed (by
`MatheronMap.particle_coefficients`), they are read off the whitened factor
rather than whitened again:

$$
W(y^* - g_j) = W(y^* - m_c) - \sqrt{\delta}\, S_{j\cdot}^\top ,
$$

for $g_j = m_c + \sqrt{\delta}\,F_c e_j$ the particle's noise-free given
value and $\delta$ the stored divisor ({ref}`dist-divisor`).
This is exact in exact arithmetic and agrees with whitening $y^* - g_j$
directly to round-off, not bit-exactly ({ref}`dist-matheron`).

(dist-cost)=
### Cost

Counted in vectors whitened per given block (a dense whitener applies in
$O(d_c^2)$ per vector, a diagonal one in $O(d_c)$):

| operation | whitened vectors | SVDs |
| --------- | ---------------- | ---- |
| `condition` (noisy) | $k + 1$, in one `whiten_mat` call | 1 |
| `log_density` (noisy), at a batch of $n$ values | $k + n$ ($n$ alone when $k = 0$) | 1, or 0 when $k = 0$ |
| `conditional_map` (build) | $k$ | 1 |
| `MatheronMap` called on $n$ samples | $n$ | 0 |
| `MatheronMap.particle_coefficients` (call) | $1$ | 0 |
| `square_root_map` (build) | $k$ | 1 |
| `SquareRootMap` (call) | $1$ | 0 |
| `cov(a)` of a block with both parts | $k$ (`LowRankUpdate` whitens $F_a$, materialized as an array, at construction) | 1 |

A given block with no factor row contributes zero rows to $S$ and whitens
only its residual: one vector rather than $k + 1$ for `condition`, and none at
a map's build. The counts above are for given blocks with factor rows.

At $k = 0$ there is no $S$: `conditional_map` builds a map with no
`IdentityPlusGram`, whose call returns the targets unchanged, and
`condition` and `log_density` compute no SVD. The capabilities a method
requires are checked at $k = 0$ as at any other width, so a missing one does
not hide behind an empty latent space.

So transporting an `EnsembleGaussian`'s $J$ particles through its Matheron
map costs $J + 1$ whitened vectors when it reads its residuals off $S$
(`particle_coefficients`), and $2J$ when it whitens each particle's residual
(calling the map on the particles). Only a count tells the two apart: the
numbers agree to round-off. Implementations may group the whitening calls as
they like, but the counts above are normative and are count-tested
({ref}`dist-conformance`, obligation 15).

The thin SVD costs $O(kN\rho)$. Applying $F_x$ to the coefficients costs
$O(d_x k)$ per vector for a dense row, less for a structured one. The whole
computation is linear in every block dimension for structured whiteners and
cubic only in the latent width.

(dist-accuracy)=
## Accuracy, and singular independent terms

This section settles issues #10 and #25: what conditioning does when a given
block's independent term is singular or nearly so, how accurate it is at
extreme ratios of factor to noise, and what is reported.

**Precondition.** Each given block's independent term is nonsingular, which
is `whiten`'s own precondition ({ref}`contract-validation`). The conditional
itself may exist without it (it needs only
$F_c F_c^\top + D_c$ nonsingular), but the whitened formulation does not
represent that case.

What happens when the precondition fails depends on whether the operator can
still whiten, and the layer's behavior in each case is normative:

| independent term | what `whiten` returns | result | reported |
| ---------------- | --------------------- | ------ | -------- |
| nonsingular | a whitening | the conditional | — |
| exactly singular and whitening produces `inf` or `nan` (a `PSDDiagonal` with a zero entry, say) | non-finite values | `nan` | in debug mode, by the **whitening check**: `ValueError` naming the method, the block and the likely cause |
| numerically singular but factorizable (a `DensePSD` of rank $N - 1$ whose Cholesky succeeds with a tiny pivot) | finite values with very large entries | the conditional of the operator as represented; see below | not reported |

The third row is not a silent wrong answer, and must not be documented as
one. The represented operator is $D_c + \delta$ for a perturbation $\delta$
of order $\varepsilon\lVert D_c\rVert$, and when $F_c F_c^\top + D_c$ is
nonsingular the conditional is continuous in that perturbation. Measured
against the exact conditional, computed densely from
$(F_cF_c^\top + D_c)^{-1}$ with the singular $D_c$: for a dense term of rank
7 of 8 drawn at random ($\kappa = 2.7 \times 10^{16}$; measured with
`enskit.gauss` on `main` at 989be85, which conditions through the same
core), the mean agrees to
$4.5 \times 10^{-9}$ and the covariance to $1.5 \times 10^{-9}$, relative,
with $\kappa(F_cF_c^\top + D_c) = 196$. What is lost is accuracy, at the
rate the next paragraph states, and nothing in the layer can tell a
numerically singular term from a merely precise one.

**Accuracy at large $\sigma_{\max}$.** The computation is backward stable:
its results are the exact conditional for given rows perturbed by about
$\varepsilon$ relative to $\sigma_{\max}$, as every normwise-stable
factorization of $S$ is. How much that costs depends on the conditional's own
sensitivity to such a perturbation, and in general it is large. For given
rows with singular values $(\sigma_{\max}, 1, 1)$ in random directions, a
perturbation of $F_c$ by $\varepsilon\sigma_{\max}$ entrywise changes the
*exact* conditional mean by $7.5 \times 10^{-12}$, $7.5 \times 10^{-9}$ and
$7.5 \times 10^{-5}$, relative, at $\sigma_{\max} = 10^5$, $10^8$ and
$10^{12}$, and the layer's errors against exact rational references are at or
below that ($3.7 \times 10^{-13}$, $2.2 \times 10^{-9}$, $4.7 \times 10^{-5}$
for the mean, and the same order for the covariance, whether $\rho = k$ or
$\rho < k$). Past $\sigma_{\max} \approx 1/\varepsilon$ such a conditional is
not determined by the stored floats at all: the mean is wrong by order 1.

Where the decomposition of $S$ is exact, as for given rows along latent
coordinate axes, the results are accurate to a few $\varepsilon$: the mean at
every measured $\sigma_{\max}$ from $10^5$ to $10^{20}$, and the covariance
and `log_density` when $\rho = k$ (for `log_density`, $N \le k$). When
$\rho < k$, the usual case for an ensemble ($J > N$), the covariance loses
accuracy even then: $T$'s complement term $x - UU^\top x$ leaves a residue of
order $\varepsilon\lVert x\rVert$ in the directions the given blocks
constrain, where the true result is of order $\lVert x\rVert/\sigma$, so the
conditional variance in those directions carries a relative error of about
$\varepsilon\,\sigma_{\max}$, of either sign. Measured against exact
references computed in extended precision, for three given coordinates and
$k = 5$:

| $\sigma_{\max}$ | relative error, latent direction along a coordinate axis | relative error, random latent direction |
| --------------- | ------------------------------------------------- | --------------------------------------- |
| $10^5$ | $2.0 \times 10^{-11}$ | — |
| $10^8$ | $1.2 \times 10^{-8}$ | $-8.9 \times 10^{-9}$ |
| $10^{11}$ | — | $8.3 \times 10^{-5}$ |
| $10^{13}$ | $1.6 \times 10^{-3}$ | $5.0 \times 10^{-3}$ |
| $10^{15}$ | — | $1.9 \times 10^{-1}$ |
| $10^{17}$ | $-1$: the variance is exactly $0$ | $3.8 \times 10^{2}$ |

The axis-aligned column is the thin basis alone: at $\rho = k$ the same rows
give $2 \times 10^{-16}$. The random column is dominated by the sensitivity of
the previous paragraph, which the thin basis does not add to. Once
$\sigma_{\max}$ passes about $1/\varepsilon$ the axis-aligned variance is
exactly zero, which reports certainty. A ratio of factor to noise standard
deviations of $10^{11}$ already costs about $10^{-4}$. The noisy `log_density`
has the thin-basis mechanism through $V$ when $N > k$ ({ref}`dist-log-density`).

The thin-basis loss is a property of
{class}`~enskit.linalg.IdentityPlusGram`, not of this layer, and is recorded
as issue #54. This layer does not check for it: no threshold on
$\sigma_{\max}$ separates a numerically singular term from a deliberately
vague factor against precise values, and the operators that could tell (a
condition estimate of $D_c$) are not part of the operator contract.

**The checks this layer does run** are tier 4 (debug mode, concrete arrays
only; {ref}`dist-validation`):

1. **The whitening check.** Immediately after whitening, every operation that
   whitens checks that the whitened factor and the whitened residuals are
   finite. On failure it raises `ValueError` naming the method, the given
   block, and the likely cause: an independent term that is singular or
   cannot whiten these values, or values so large relative to it that
   whitening overflows. `IdentityPlusGram` deliberately does not check its
   `S` ({ref}`contract-gram`), so this layer, which built `S` and knows the
   cause, must.
2. **The result check.** Every conditioning result is checked for finiteness
   once its arrays exist and **before any result object is constructed**;
   the result is then built through the constructor-bypassing path, so no
   constructor's own tier-4 check can fire first and name the wrong call
   (the old contract's rule). What is checked: for noisy `condition`, the
   decomposition's `U` and `sigma`, the coefficients $w$ and the target
   means; for exact `condition`, the diagonal of $R$, $w$, the target means
   and the materialized target rows; for `log_density`, the value; for the
   maps, the coefficients or particles returned. A non-finite result with
   finite whitened inputs means overflow (applying a target factor row, or
   squaring a whitened residual) or a decomposition that failed, and the
   message says so.

The two checks are independently testable: a singular diagonal term trips
the first, and a target factor row of order $10^{300}$ against finite,
well-scaled given blocks trips the second (for `log_density`, whitened
values of order $10^{200}$, whose square overflows). A `PSDDiagonal` with a
zero entry is rejected by its own constructor in debug mode, so the test
constructs it with debug checks off and conditions with them on. Every conditioning path runs both,
so a caller who wraps a loop in `debug_checks` gets the same exception from
every path on identical inputs. This was the asymmetry of issue #10: the old
`condition` raised where the two sample-moving methods returned `nan`.

`sample` and `realize_particles` are outside the result check, as in the old
contract: their factor rows and terms arrive constructed, the operator layer
validates its own fields, and a non-finite draw implicates an operator rather
than this call.

(dist-ensemble)=
## `Ensemble`

$J$ particles over named blocks, optionally weighted: the empirical
distribution

$$
\hat p = \sum_{j=1}^{J} w_j\,\delta_{x_j}, \qquad
w_j = \frac{\exp(\ell_j)}{\sum_i \exp(\ell_i)} ,
$$

with log weights $\ell$, or $w_j = 1/J$ when unweighted.

**Construction.** `Ensemble(blocks=None, /, *, log_weights=None, **block_arrays)`.

- Blocks follow {ref}`dist-block-arguments`; mapping entries come first, in
  their order, then keywords in the order written. At least one block.
  A block named `log_weights` must be given in the mapping.
- Each block is an array of rank exactly 2, `(J, d_b)`, with $d_b \ge 1$, one
  shared $J \ge 2$, and one shared real floating dtype. A single particle has
  no anomalies, so $J = 1$ is rejected; so is a block with no columns.
- `log_weights` is `None` or a `(J,)` real floating array of the blocks'
  dtype. `None` means equal weights, and differs from an array of equal
  values only in `is_weighted`.
- Arrays are stored as given (converted with `jnp.asarray`, never copied to a
  different dtype).
- Particles need not be finite, in debug mode or not: a non-finite row is
  how a failed particle is marked ({ref}`dist-failed-particles`).

**Introspection.** `names` (tuple, in order), `dims` (dict, in order),
`n_particles` (`int`), `is_weighted` (`bool`), `log_weights` (the stored array
or `None`), and `ens[name]`, the `(J, d_b)` array (`KeyError` listing the
blocks otherwise).

**`weights`** is the `(J,)` normalized weights, a max-shifted softmax of the
log weights, so log weights far outside the floating range normalize
correctly; $1/J$ each when unweighted.

**`all_finite`** is a `(J,)` boolean array: particle $j$ is finite in every
block. Non-finite rows are how the layers above mark failed particles. It
reads values, so it is an array, never a Python `bool`.

**Structure.** All keep the weights and return an `Ensemble`.

- `marginal(*names)`: the named blocks, in the order given; at least one.
  Marginalizing an empirical distribution is dropping coordinates, so the
  particles are unchanged.
- `drop(*names)`: every block not named, in block order; at least one must
  remain (`ValueError`).
- `assign(blocks=None, /, **block_arrays)`: replace existing blocks, which
  keep their positions, or append new ones in the order given. Each array
  must be `(J, d)` with the ensemble's $J$ and dtype.
- `rename(mapping=None, /, **new_names)`: `old=new`; positions are kept.
  Every old name must be a block. New names must be distinct and must not
  collide with a block that is not itself being renamed (`ValueError`), so a
  swap is allowed.
- `pipe(f, *args, **kwargs)` returns `f(self, *args, **kwargs)`. It exists so
  operations defined in a layer above can be chained without this layer
  importing them; the user guide introduces reassignment first.

**Moments.**

- `mean(name)`: $\bar x = \sum_j w_j x_j$, `(d,)`, computed as
  $x_r + \sum_{i : w_i > 0} w_i (x_i - x_r)$.
- `anomalies(name)`: $a_j = x_j - \bar x$, `(J, d)`. Formed by subtracting
  a reference particle $x_r$, then the mean of the differences:

  $$
  a_j = (x_j - x_r) - \sum_{i : w_i > 0} w_i (x_i - x_r),
  $$

  so particles that are identical give exactly zero anomalies however large
  their magnitude. A plain mean-and-subtract does not: identical particles of
  magnitude $6\times10^{23}$ give anomalies of order 1, and conditioning then
  moves them by order 1 where it should leave them unchanged. The means are taken over the particle axis with the
  axis kept, so an operand whose leading axis happens to equal $J$ cannot
  broadcast against the wrong axis.

  The reference $x_r$ is the first particle when unweighted, and a particle
  of largest log weight when weighted, so it never has weight zero.
  The sums run over the particles of positive weight $w_i$, as `weights`
  reports it: the term of a particle of weight zero is not computed, rather
  than computed as $0 \cdot (x_i - x_r)$, which is `nan` when the particle
  is not finite. Its own anomaly $a_j$ is still formed, and is non-finite
  when the particle is. For a particle of weight zero whose difference
  $x_i - x_r$ is finite, the term was an exact zero anyway, so the mask
  changes nothing there. The choice of reference is a separate matter: a
  reference of weight zero far from the others made every difference
  inaccurate (a particle at $10^{12}$ cost $10^{-4}$ in the mean), and a
  particle of largest weight does not.
- `cov(a, b=None, *, unbiased=True)`: the covariance $\hat C_{ab}$ of
  {ref}`dist-divisor`, with the divisor `unbiased` selects. `cov(a)` is a
  {class}`~enskit.linalg.PSDLowRank` of the projected factor row
  ({ref}`dist-project`), width $J$; `cov(a, b)` for $a \ne b$ is the product
  operator $F_a F_b^\top$, shape `(d_a, d_b)`. Neither forms a
  $(d_a, d_b)$ matrix. `cov(a, a)` is `cov(a)`.

(dist-project)=
### `project()`

The moment-matching Gaussian: the Gaussian with this ensemble's mean and
covariance, jointly over all blocks. It is the one place a Gaussian
approximation enters, and it is always written by the caller (or inside
`enskit.kalman.gaussian_approximation`, which says so).

`project(*, unbiased=True)` uses the divisor `unbiased` selects
({ref}`dist-divisor`), so its covariance is `cov`'s.

**Unweighted**, whichever the divisor, the result is an `EnsembleGaussian`
with $k = J$, no independent terms, the divisor $\delta$ stored, and

$$
m_b = \bar x^{(b)}, \qquad
F_b = \frac{1}{\sqrt{\delta}}\big[a^{(b)}_1, \dots, a^{(b)}_J\big]
\in \mathbb R^{d_b \times J},
$$

each row a {class}`~enskit.linalg.Dense` of the scaled, transposed anomalies.
The factor is centered because anomalies sum to zero. The conversion loses
nothing: `realize_particles()` returns these particles to round-off.

**Weighted**, the result is a plain `Gaussian` with $k = J$ and

$$
m_b = \sum_j w_j x^{(b)}_j, \qquad
(F_b)_{:,j} = \frac{\sqrt{w_j}\,a^{(b)}_j}{\sqrt{\delta_w}} .
$$

$\sqrt{w_j}$ is computed in log space, $\exp\big((\ell_j - \operatorname{lse}(\ell))/2\big)$,
and the anomaly is masked to zero before it is scaled, so a particle of
weight zero contributes an exact zero column with a finite derivative, even
when the particle is not finite. The divisor is computed as
$1 - \sum_i w_i^2 = -\operatorname{expm1}\big(\operatorname{lse}(2\ell) - 2\operatorname{lse}(\ell)\big)$,
which stays accurate when one weight dominates. Each $\operatorname{lse}$
factors out the largest term and sums the rest through `log1p`: a plain
$\log(1 + t)$ loses $\varepsilon/t$ relative accuracy, which the
near-cancellation of the two terms then exposes (about $10^{-10}$ at an ESS of
$1 + 10^{-6}$). It is not particle-aligned:
reading particles back would divide by $\sqrt{w_j}$. So it is a plain
`Gaussian`, and the rules that need alignment refuse it.

When the weights are concentrated on one particle to working precision, the
unbiased divisor rounds to zero and the weighted covariance is undefined.
That is a value precondition of the unbiased projection: in debug mode
`project` raises `ValueError` naming the effective sample size and
`unbiased=False`; otherwise the factor rows are `inf` or `nan`. The
empirical divisor is $1$, and has no such precondition.

Every particle of positive weight must be finite. That too is a value
precondition, of `project` and of `cov`: in debug mode they raise
`ValueError` naming the block and the number of such particles, checked
block by block before the concentrated weights; otherwise the means or
factor rows are `nan`.

(dist-failed-particles)=
### Failed particles

A particle with a non-finite entry in any block is *failed*; `all_finite`
finds them. The layers above produce them and decide what to do about
them. This layer never repairs, and it treats a failed particle as a
valid state of an `Ensemble`:

- construction, `assign`, the structural methods, `weights`, `all_finite`,
  `effective_sample_size`, `reweight` and `resample` accept one, in debug
  mode or not;
- the moments and `project` leave out every particle of weight zero, so
  a failed particle given log weight $-\infty$, or a finite log weight
  whose weight underflows to zero, affects no mean, covariance or
  projection, no other particle's anomaly, and none of their derivatives;
  `resample` never selects it;
- `project` and `cov` refuse a failed particle of positive weight, in debug
  mode ({ref}`dist-project`). With one, `mean` is `nan`, and so is every
  anomaly.

The conditional maps are not on the first list: their tier-4 check on
samples covers every row.

Giving every failed particle weight zero, without resampling, is

```python
ens = reweight(ens, jnp.where(ens.all_finite, 0.0, -jnp.inf))
```

and the alternative is the algorithms' repair, which replaces the failed
rows.

(dist-gaussian)=
## `Gaussian`

**Construction.** `Gaussian(means, *, factors=None, block_covs=None, latent_dim=None)`.

`factors` and `block_covs` are keyword-only. Both are mappings of operators,
and a factor row may be any `LinOp`, a `PSDLinOp` included, so a positional
slip would make a covariance $C$ a factor row (covariance $CC^\top$) whenever
$k = d_b$, with nothing raised.

- `means`: a non-empty mapping from block name to an array of rank exactly 1,
  `(d_b,)`, $d_b \ge 1$, all of one real floating dtype. Its order is the
  block order. Names are `str`.
- `factors`: a mapping from a subset of the names to factor rows, each a
  {class}`~enskit.linalg.LinOp` or a 2-D array (wrapped in
  {class}`~enskit.linalg.Dense`), of shape `(d_b, k)` with one shared $k$.
- `block_covs`: a mapping from a subset of the names to
  {class}`~enskit.linalg.PSDLinOp`s of shape `(d_b, d_b)`.
- `latent_dim`: keyword-only, a Python `int` $\ge 0$ (exact type; `bool` is
  rejected). Inferred from the factor rows when omitted, and $0$ when there
  are none. Given together with factor rows, it must equal their width.
- Every block must have a factor row or an independent term.

The constructor stores what it is given; it computes nothing. Its value
precondition (tier 4) is that the means are finite.

**`Gaussian.independent(blocks=None, /, **block_specs)`**: mutually
independent blocks, each given as a `(mean, cov)` pair,

$$
x_b \sim \mathcal N(m_b, C_b) \ \text{independently for each } b,
$$

with no shared factor ($k = 0$) and each covariance as that block's
independent term. This is how a prior is written:
`Gaussian.independent(u=(m0, C0))`.

**Introspection.** `names`, `dims`, `latent_dim` (static `int`),
`mean(name)`, `factor(name)` (a `LinOp` or `None`), `block_cov(name)` (a
`PSDLinOp` or `None`); unknown names raise `KeyError`.

**`cov(a, b=None)`**, as an operator; nothing of size $(d_a, d_b)$ is
formed by the call.

| block $a$ has | `cov(a)` |
| ------------- | -------- |
| $F_a$ and $D_a$ | `LowRankUpdate(D_a, F_a)`, $D_a + F_aF_a^\top$ ({ref}`contract-low-rank-update`) |
| $D_a$ only | $D_a$ itself |
| $F_a$ only | `PSDLowRank(F_a.to_dense())`, which materializes the $(d_a, k)$ row |

`LowRankUpdate` needs `whiten` of its base, so `cov(a)` of a block with both
parts raises `UnsupportedOpError` naming `whiten` when $D_a$ cannot whiten.
Its `solve` is a Woodbury solve that loses accuracy when $F_aF_a^\top$
dominates $D_a$ by many orders of magnitude (issue #51); `whiten` and
`logdet` do not. For $a \ne b$, `cov(a, b)` is `product(F_a, F_b.T)`, or
`Zero(d_a, d_b)` of the distribution's dtype when either row is absent. `cov(a, a)` is `cov(a)`.

**Structure.** Each returns a Gaussian of the same kind (an
`EnsembleGaussian` stays one), on the same latent space.

- `marginal(*names)`: the named blocks, in the order given; their means,
  factor rows and terms are kept and the rest discarded. $k$ is unchanged.
  Exact and free.
- `drop(*names)`: the marginal over every other block, in block order; at
  least one must remain.
- `rename(mapping=None, /, **new_names)`: as for `Ensemble.rename`.
- `pipe(f, *args, **kwargs)`: `f(self, *args, **kwargs)`.

(dist-add-noise)=
### `add_noise(covs=None, /, **block_covs)`

Add independent Gaussian noise to the named blocks, in place:

$$
x_b \mapsto x_b + e_b, \qquad e_b \sim \mathcal N(0, R_b)
\ \text{independent of everything else.}
$$

Each $R_b$ is a `PSDLinOp` of side $d_b$; at least one block.

- On a block with no independent term, this sets $D_b = R_b$ and changes
  nothing else. The latent space is unchanged, so an `EnsembleGaussian` stays
  one, and $R_b$ is used afterward only through what later operations ask of
  it (`whiten`, for conditioning).
- On a block that already has one, its existing term is first moved into the
  shared factor (`absorb`), which needs `D_b.factor()`, and $R_b$
  becomes the new term. The latent space grows, so the result is a plain
  `Gaussian`.

The noise-free block is not kept. To keep it, push it to a new block through
`enskit.maps.AdditiveNoise` instead.

(dist-absorb)=
### `absorb(*names)`

Move the independent terms of the named blocks into the shared factor. Each
named block must have one (`ValueError` otherwise) whose `factor()` is
supported (`UnsupportedOpError`, checked for all named blocks before any
work). The distribution is unchanged; the representation now lets a later
map of block $b$ stay correlated with $b$.

For the named blocks in **block order**, with $L_b$ = `D_b.factor()` of width
$w_b$, the latent space grows to $k' = k + \sum_b w_b$, with block $b$'s
columns appended in that order. Each absorbed block's row becomes its old
row, if any, followed by zero columns except $L_b$ in its own columns; every
other block with a factor row gains zero columns; a block with no factor row
keeps none. The zero columns are {class}`~enskit.linalg.Zero` blocks, of the
distribution's dtype, inside an {class}`~enskit.linalg.HStack`. They store no
array and `HStack` skips them when applying; a path that densifies the row
(`to_dense`, as conditioning does for given rows) materializes them. A row that would consist of a single operator (an absorbed
block with no previous row, at $k = 0$, absorbing alone) is that operator
itself, not a one-element `HStack`.

The result is a plain `Gaussian`: the latent coordinates are no longer
particles.

(dist-compress)=
### `compress(*, max_dim=4096)`

Re-factor the shared factor to a width at most the total dimension of the
blocks that have factor rows. Let $F$ be the $(D_F, k)$ array stacking those
rows. When $k > D_F$, with the thin QR $F^\top = QR$,

$$
F F^\top = R^\top R, \qquad F \mapsto R^\top \in \mathbb R^{D_F \times D_F},
$$

and the rows of $R^\top$, as `Dense`s, replace the factor rows; $k$ becomes
$D_F$. The distribution is unchanged. When $k \le D_F$ nothing is computed
and the rows are kept as they are. When no block has a factor row but
$k > 0$ (a marginal over blocks without rows), $D_F = 0$ and the result has
$k = 0$. In every case the result is a plain
`Gaussian`, so code does not depend on whether compression happened.

Use it where repeated `absorb` would otherwise grow the latent width without
bound, as absorbing a new independent term after every conditioning does.
Before allocating, when compression would happen ($k > D_F > 0$), `compress`
raises `ValueError` if $D_F > $ `max_dim` or $D_F k > $ `max_dim`$^2$
(stacking the rows allocates $D_F \times k$), with `max_dim` a Python `int`
$\ge 1$. Compression materializes every factor row, so a structured
row (a Kronecker factor) does not survive it.

(dist-condition)=
### `condition(values=None, /, **block_values)`

The exact conditional distribution of the other blocks, given values for
some blocks. Values follow {ref}`dist-block-arguments`; each is exactly
`(d_c,)`; at least one block is given, and at least one block must remain
(`ValueError`). The result is over the remaining blocks, in block order,
of the same kind and on the same latent space.

Two cases, chosen by the given blocks' structure. Some but not all given
blocks having independent terms is a `ValueError`: the two cases do not
combine.

**Noisy values: every given block has an independent term.** With the
quantities of {ref}`dist-core`, for each target block $x$ with a factor row,

$$
m_x' = m_x + F_x w, \qquad w = A^{-1} S\, W(y^* - m_c), \qquad
F_x' = F_x T, \qquad T = A^{-1/2},
$$

and $D_x$ unchanged. One `IdentityPlusGram` serves both. $F_x'$ is the
product operator `product(F_x, T)`, so a structured $F_x$ is never densified.
A target with no factor row is independent of the given blocks and is
returned unchanged. At $k = 0$ every block is independent: the result is the
marginal over the targets, after the values' shapes are checked. Repeated
conditioning nests products; `compress` collapses them.

Each given block needs `whiten`. The whitened vectors and the SVD count are
those of {ref}`dist-cost`.

**Exact values: no given block has an independent term.** With the thin QR
of the stacked given rows' transpose, $F_c^\top = QR$, $Q \in \mathbb R^{k
\times N}$,

$$
m_x' = m_x + F_x w, \qquad w = Q R^{-\top}(y^* - m_c), \qquad
F_x' = F_x (I_k - Q Q^\top).
$$

$F_x'$ is materialized as the $(d_x, k)$ array $F_x - (F_x Q) Q^\top$, a
`Dense`; exact conditioning therefore densifies a structured target row, and
is meant for factors already held as arrays, such as an ensemble's. It
requires $F_c$ of full row rank $N$:

- $N \le k$ always, and for an `EnsembleGaussian`, whose centered columns
  have rank at most $J - 1$, $N \le J - 1$. Both are shape conditions and
  always run (`ValueError`). Without the second, an aligned Gaussian with
  $J = N$ passes the first, and under `jit` returned a log density of
  $-3 \times 10^{31}$.
- Rank deficiency beyond the shape conditions is a value precondition. In
  debug mode `condition` raises `ValueError` when
  $\min_i |R_{ii}| \le \max(N, k)\,\varepsilon\,\max_i |R_{ii}|$.

$I - QQ^\top$ fixes $\mathbf 1$ when the factor is centered ($Q$'s columns
lie in the range of $F_c^\top$, which is orthogonal to $\mathbf 1$), so an
`EnsembleGaussian` stays aligned.

(dist-conditional-map-method)=
### `conditional_map(given)`

The exact conditional of this Gaussian as a pointwise transport map,
`MatheronMap` ({ref}`dist-matheron`), built from the given block
*names* (a `str` or a sequence of `str`). Every given block must have an
independent term: there is no exact-value conditional map (`ValueError`). At
least one block must remain as a target. The given blocks' `whiten` is
checked at build.

(dist-regression)=
### `regression(target, *, given, min_norm=False)`

The regression of block $y$ = `target` on the blocks $x$ = `given` (a `str`
or a sequence of `str`, reordered to block order): the conditional of $y$
given $x$ as an affine function of the given values,

$$
y \mid x \sim \mathcal N(Ax + c,\ \Omega), \qquad
A = C_{yx}C_{xx}^{-1}, \qquad c = m_y - Am_x, \qquad
\Omega = C_{yy} - AC_{xy},
$$

returned as a `Regression`: `coefficients`, a mapping from each given block,
in block order, to its $(d_y, d_b)$ operator $A_b$; `intercept`, the
$(d_y,)$ array $c$; and `residual_cov`, $\Omega$, as `cov` would return the
covariance of a block with the conditional's factor row and $y$'s own term.
Its static attributes are `target` and `given`. It is a frozen pytree, so a
`jax.vmap` over the call returns a family. $A$ minimizes
$\mathbb E\lVert y - Ax - c\rVert^2$; for an ensemble's projection it is the
least-squares regression of the particles, weighted by their weights when
there are any.

Write $F_c$ for the given blocks' stacked factor rows, $N$ for their total
dimension and $F_y$ for the target's row. As for `condition`, mixed given
blocks are a `ValueError`, and otherwise the case is chosen by structure:

| case | computation | $\Omega$ |
| --- | --- | --- |
| every given block has a term | $S$ and its `IdentityPlusGram` as in {ref}`dist-core`; $A = F_y (I + SS^\top)^{-1}F_c^\top D_c^{-1}$, from `D_c.solve_mat(F_c)` and the gram's `solve_mat` | $F_y T T^\top F_y^\top + D_y$ |
| no term, $N \le r$ | thin QR $F_c^\top = QR$; $A = F_y QR^{-\top}$ | $F_y(I - QQ^\top)F_y^\top + D_y$ |
| no term, $N > r$, `min_norm=True` | thin QR $F_cH = QR$; $A = F_yHR^{-1}Q^\top = F_yF_c^{+}$ | $(F_y - AF_c)(F_y - AF_c)^\top + D_y$, which is $D_y$ to round-off |

Here $r = J - 1$ for an `EnsembleGaussian`, whose factor rows annihilate
$\mathbf 1$, and $H$ is then a fixed orthonormal basis of
$\mathbf 1^\perp$; otherwise $r = k$ and $H = I_k$. A target with no factor
row has zero coefficients (`Zero` operators) and $\Omega = D_y$.

- **The noisy case is ridge regression.** On an ensemble's projection,
  `add_noise(x=Lam)` before `regression` gives
  $A = \hat C_{yx}(\hat C_{xx} + \Lambda)^{-1}$. It needs `whiten` and
  `solve` of the term of every given block with a factor row
  (`UnsupportedOpError` before any work); a block without one contributes
  zero coefficients and uses neither. It
  loses relative accuracy like $\varepsilon(1 + \sigma_{\max}^2)$, since
  $(I + SS^\top)^{-1}$ is applied to $F_c^\top D_c^{-1} = SW$ rather than
  read off the decomposition, which would need $W^\top$, an operation
  operators do not provide.
- **The exact case without `min_norm` and with $N > r$ raises**
  `ValueError`, naming both remedies. The minimum-norm solution is a choice:
  it depends on the given blocks' coordinates, and with several blocks it
  weighs their units against each other.
- **Representation.** No $(d_y, d_b)$ array is formed except in the unique
  exact case, where $N \le k$. Otherwise $A_b$ is `product(F_y, Dense(C_b))`
  (noisy, $C$ the $(k, N)$ array) or `product(Dense(L), Dense(Q_b).T)`
  (minimum norm, $L$ the $(d_y, r)$ array), of width at most $k$. The
  residual factor row is a $(d_y, k)$ `Dense` in both exact cases, as in
  `condition`.
- **Rank is a value precondition**, as for exact `condition`. The case is
  chosen from the type and the sizes, assuming the given factor rows have
  rank $\min(N, r)$. In debug mode the exact cases raise `ValueError` when
  the triangular factor $R$ fails
  $\min_i|R_{ii}| > \max(\text{its QR's sizes})\,\varepsilon\max_i|R_{ii}|$.
  Outside it, lower rank gives non-finite or wrong coefficients without
  raising. It arises from duplicated particles (after `resample`), a given
  coordinate constant across particles, an `EnsembleGaussian` conditioned
  exactly on other blocks (rank $J - 1 - N'$ for $N'$ conditioned
  coordinates), and a weighted projection, a plain `Gaussian` with $r = k =
  J$ whose rank is at most $J - 1$, and less with zero weights. Issue #86
  records the instances and the options.
- **Counts.** One QR and no SVD in the exact cases; one SVD and no QR in the
  noisy case.

(dist-log-density)=
### `log_density(values=None, /, **block_values)`

The log density of the marginal over exactly the blocks in `values`,
evaluated at those values. Blocks not named are marginalized out, so
`g.log_density(a=v)` is the density of the $a$ marginal and
`g.log_density(a=v, b=w)` that of the joint. Each value is
`(*batch, d_b)`, with one batch shape shared by all of them (`ValueError`
otherwise; values are not broadcast against each other). The result has shape
`batch`, a 0-d array when unbatched, never a Python float.

**Noisy values** (every named block has an independent term), with
$b = W(v - m_c)$ batched over the leading axes,

$$
\log p(v) = -\tfrac12\Big(\lVert b\rVert^2 - (Sb)^\top A^{-1} S b
  + \log\det D_c + \textstyle\sum_i \log(1 + \sigma_i^2) + N\log 2\pi\Big),
$$

computed through $C = $ `LowRankUpdate(D_c, F_c)`, with $D_c$ the
block-diagonal operator of the named terms (`block_diag`) and $F_c$ their
stacked rows as a `Dense`: the quadratic term is
$\lVert W_C(v - m_c)\rVert^2$ with $W_C = (I_N + S^\top S)^{-1/2}W$, which
is `C.whiten`, and the determinant is `C.logdet()`, both from the one
decomposition `C` stores. At $k = 0$, or when no named block has a factor
row, the $S$ terms vanish and $C$ is $D_c$.
Requires `whiten` and `logdet` of every named block's term, checked in that
order.

The quadratic term must **not** be computed as
$\lVert b\rVert^2 - \langle Sb, A^{-1}Sb\rangle$, the form the formula
suggests: the two terms nearly cancel for a typical value, and the relative
error grows like $\varepsilon\sigma_{\max}^2$. Measured against exact values
in extended precision, at a value drawn from the marginal:

| $N$, $k$ | $\sigma_{\max}$ | through `LowRankUpdate.whiten` | $\lVert b\rVert^2 - \langle Sb, A^{-1}Sb\rangle$ |
| -------- | ---------------- | ------------------------------ | ------------------------------------------------- |
| 1, 3 | $10^8$ | $0$ | $-4.5$ |
| 3, 8 | $10^{12}$ | $4 \times 10^{-16}$ | $-7 \times 10^{8}$ |
| 5, 2 | $10^8$ | $-1.6 \times 10^{-8}$ | $-6.7$ |
| 8, 3 | $10^{12}$ | $6.6 \times 10^{-5}$ | $3.3 \times 10^{7}$ |

The stable route is exact to round-off when $N \le k$, and loses about
$\varepsilon\sigma_{\max}$ when $N > k$, by the thin-basis mechanism of
{ref}`dist-accuracy` applied to $V$ (issue #54).

**Exact values** (no named block has one), with $F_c^\top = QR$ and the same
size conditions as exact `condition`,

$$
\log p(v) = -\tfrac12\Big(\lVert R^{-\top}(v - m_c)\rVert^2
  + 2\textstyle\sum_i \log|R_{ii}| + N \log 2\pi\Big).
$$

Mixed sets raise `ValueError`, as for `condition`.

This is the evidence of values under a joint, and the objective for tuning
hyperparameters ({ref}`dist-derivatives`).

(dist-sample)=
### `sample(key, n_particles)`

Draws `n_particles` independent samples as an unweighted `Ensemble` over
every block, in block order:

$$
x_j^{(b)} = m_b + F_b\,\xi_j + L_b\,\eta_j^{(b)}, \qquad
\xi_j \sim \mathcal N(0, I_k), \quad \eta^{(b)}_j \sim \mathcal N(0, I_{w_b}),
$$

with $L_b$ = `D_b.factor()` of width $w_b$, and either term absent when the
block lacks it. `n_particles` is a Python `int` (exact type) $\ge 2$, the
`Ensemble` minimum: `TypeError` for a non-`int`, `ValueError` below 2. Every
independent term must support `factor`, checked before drawing. The draw is
pinned ({ref}`dist-prng`).

(dist-ensemble-gaussian)=
## `EnsembleGaussian`

A `Gaussian` whose latent coordinates are an ensemble's particles: $k = J$
and

$$
F_b = \frac{1}{\sqrt{\delta}}\big[a^{(b)}_1, \dots, a^{(b)}_J\big], \qquad
F_b \mathbf 1 = 0 ,
$$

with $\delta$ its divisor, $J - 1$ or $J$ ({ref}`dist-divisor`), so latent
coordinate $j$ belongs to particle $j$. Two operations need that
correspondence and exist only here: reading the particles back out
(`realize_particles`) and moving them as a set (`square_root_map`).

**Construction.** `EnsembleGaussian(means, *, factors=None, block_covs=None, n_particles, unbiased=True)`,
with the `Gaussian` arguments, keyword-only for the same reason,
`n_particles` a Python `int` $\ge 2$ equal to the factor rows' width, and
`unbiased` a `bool` saying which divisor the factor rows were formed with
(`TypeError` otherwise). The normal route is `Ensemble.project()`, which
centers by construction. Centering is a value precondition, checked in debug
mode only: for each factor row,
$\lVert F_b\mathbf 1\rVert_\infty \le 10\,J\,\varepsilon\,\max_{ij}|(F_b)_{ij}|$,
computed densely.

**Why it is a type and not a flag.** The square-root reading is valid only
for a centered factor. Applied to an uncentered one it returns, silently, a
particle set whose mean is displaced by $\sqrt{k-1}\,F_xT\mathbf 1/k$ and
whose sample covariance falls short of the conditional's by the rank-one term
$(F_xT\mathbf 1)(F_xT\mathbf 1)^\top/k$: measured at 18% to 62% of the
covariance's own scale. Carried by the type, the precondition is checked by
construction rather than by value.

**What keeps the type.** Every method that keeps the latent space returns an
`EnsembleGaussian` with the same `n_particles` and `unbiased`: `marginal`, `drop`,
`rename`, `condition` (both cases), and `add_noise` on blocks with no
independent term. `absorb`, `compress`, and `add_noise` on a block that
already has one return a plain `Gaussian`. So does
`enskit.maps.pushforward` whenever it changes the latent space; that
layer's contract says when.

`n_particles` is a static `int`, and `latent_dim == n_particles` always.
`unbiased` is a static `bool`, so the two divisors are two tree structures,
and `divisor` is the property $\delta$: `n_particles - 1` when `unbiased`,
`n_particles` otherwise. Every reading of particles out of the factor rows
uses `divisor`: `realize_particles`, `square_root_map`,
`MatheronMap.particle_coefficients`, and the Kalman layer's aligned paths.

(dist-realize)=
### `realize_particles(*, key=None, exclude_block_covs=())`

The particles this Gaussian's latent coordinates belong to, as an unweighted
`Ensemble` over every block:

$$
x_j^{(b)} = m_b + \sqrt{\delta}\,F_b e_j \;\big[+\, L_b\,\eta^{(b)}_j\big],
\qquad j = 1, \dots, J,
$$

with $e_j$ the $j$-th unit vector and $\delta$ the stored divisor, computed
as $m_b$ plus the rows of $\sqrt{\delta}\,F_b^\top$ (`F_b.to_dense()`, transposed). A block with no factor
row realizes as its mean. The bracketed draw from $D_b = L_bL_b^\top$ is added
for each block that has an independent term and is not named in
`exclude_block_covs`, a sequence of block names.

- For the result of `Ensemble.project()` this returns the ensemble's
  particles to round-off. After `condition` it returns the conditioned
  particles: the deterministic square-root reading of the conditional.
- `key` is keyword-only, required exactly when some realized block has an
  independent term that is not excluded (`ValueError` naming the blocks when
  missing), and otherwise ignored. The draw is pinned ({ref}`dist-prng`).
- `exclude_block_covs` returns blocks without their independent terms: the
  noise-free given blocks that `MatheronMap.particle_coefficients` reasons
  about while drawing that noise itself, in whitened coordinates. Each name
  must be a block (`KeyError`) with an independent term (`ValueError`).

(dist-square-root-map-method)=
### `square_root_map(given)`

The square-root reading of the conditional, as a map from given values to
this Gaussian's particles: a `SquareRootMap`
({ref}`dist-square-root-map`). Built from the given block names; every given
block must have an independent term (`ValueError` otherwise; for exact values,
`condition(...).realize_particles()` is the route). At least one target.

(dist-maps)=
## Conditional maps

(dist-conditional-map)=
### The `ConditionalMap` protocol

A map carrying samples of a joint distribution to samples of a conditional.
It is built from a joint and the *names* of the given blocks, and called with
samples and a value $y^*$:

```python
class ConditionalMap(Protocol):
    given: tuple[str, ...]
    targets: tuple[str, ...]

    def __call__(self, samples: Ensemble, values=None, /, *, key=None,
                 **block_values) -> Ensemble: ...
```

The obligations of an implementation, which
`enskit.testing.check_conditional_map` checks (from PR 6):

1. `given` and `targets` are disjoint tuples of block names, fixed at build.
2. Values follow {ref}`dist-block-arguments`, one exactly `(d_c,)` array for
   each given block and no other; anything else raises.
3. `samples` must contain every given and every target block. The result has
   every target block replaced, the given blocks dropped, any other block
   passed through unchanged and in its position, and the weights kept. It has
   the same number of particles.
4. It is **pointwise**: the image of sample $j$ depends only on sample $j$,
   $y^*$, and row $j$ of whatever the map draws from its key. So without a
   key, permuting the samples permutes the result exactly. With a key, the
   draw's row $j$ is tied to position $j$, not to the sample, so a
   permutation changes which noise each sample receives: equivariance then
   holds in distribution only, and the check tests it on the keyless call. A map
   that couples samples (as a square-root map does) is not a
   `ConditionalMap`.
5. A key is consumed whole, and the map is deterministic given it.

A nonlinear triangular transport map fitted to particles is the same
protocol with a different fitting procedure.

(dist-matheron)=
### `MatheronMap`

The exact conditional of a `Gaussian` as an affine transport map, built by
`Gaussian.conditional_map(given)`:

$$
T_{y^*}(x, y) = x + K(y^* - y) = x + F_x\, A^{-1} S\, W(y^* - y),
\qquad K = \operatorname{cov}(x, y)\operatorname{cov}(y, y)^{-1},
$$

for every target block $x$ with a factor row (targets without one are passed
through unchanged: their gain is zero). If $(x, y)$ is a sample of the
Gaussian, $T_{y^*}(x, y)$ is a sample of its conditional at $y^*$. This is
Matheron's rule (Journel & Huijbregts, 1978), known in machine learning as
pathwise conditioning (Wilson et al., 2021), and applied to ensemble
particles with perturbed values it is the stochastic ensemble Kalman update
(Burgers et al., 1998). The mean is $m_x'$ and

$$
\operatorname{cov} T = C_{xx} - KC_{yx} - C_{xy}K^\top + K C_{yy} K^\top
= C_{xx} - K C_{yx},
$$

since $K C_{yy} = C_{xy}$. Here $y$ includes the given blocks' independent
noise; when the samples hold only the noise-free part, the map draws that
noise itself (the `key` below).

**At build** it holds the `IdentityPlusGram` of $S$, the given blocks' terms
(for whitening) and means, the targets' factor rows, and, when built from an
`EnsembleGaussian`, its `n_particles`. It is not constructed directly.
Attributes: `given` and `targets` (block order), `latent_dim`.

**`__call__(samples, values=None, /, *, key=None, **block_values)`**
follows the protocol. The `key` says what the samples hold: without one,
each sample's given blocks are taken to include their independent noise
(joint samples); with one, they are taken to be noise-free, and the map
draws the noise. For each sample $j$ it whitens the residual $y^*_c - y_{j,c}$ of
each given block (differenced before whitening), forms
$b_j = W(y^* - y_j)$, and, when `key` is given, subtracts
$\varepsilon_j$, a row of one pinned `(n, N)` standard normal draw
({ref}`dist-prng`):

$$
W\big(y^* - (y_j + e_j)\big) = W(y^* - y_j) - \varepsilon_j, \qquad
e_j = W^{-1}\varepsilon_j \sim \mathcal N(0, D_c).
$$

The noise therefore enters only in whitened coordinates, and the
independent terms need only `whiten`, never `factor`. One
`solve_factor` call on the batch gives the `(n, k)` coefficients
$w_j$, and $x_j' = x_j + F_x w_j$.

:::{warning}
The map neither accepts nor returns the perturbations. A perturbation used
through the whitened shortcut must never also be pushed through `factor()` in
the same transport (the warning under `whiten` in {doc}`linop-contract`): $WL$ has orthonormal rows but is not the identity,
so mixing the two representations corrupts the joint law of the result while
every marginal statistic still looks right. Owning the draw and its single
use in one place makes that structural.
:::

**`coefficients(samples, values=None, /, *, key=None, **block_values)`**
returns the `(n, k)` coefficients $w_j$ with $K(y^* - y_j - e_j) = F_x w_j$,
without applying them: the escape hatch for code that works in latent space.

**`particle_coefficients(values=None, /, *, key, **block_values)`**
returns the same coefficients for the particles of the `EnsembleGaussian` the
map was built from, reading their residuals off the whitened factor:

$$
b_j = W(y^* - m_c) - \sqrt{\delta}\,S_{j\cdot}^\top - \varepsilon_j ,
$$

with $\delta$ the Gaussian's divisor and $\varepsilon$ the same pinned
`(J, N)` draw. The map stores $J$ and $\delta$ as its static attributes
`n_particles` and `divisor`, both `None` when built from a plain
`Gaussian`. The key is required
(`ValueError` when missing): the particles' given blocks are noise-free by
construction, and without the draw the result would be the unperturbed
transport, whose sample covariance falls short of the conditional's by
$KD_cK^\top$. It whitens one vector per given block, so transporting $J$
particles costs $J + 1$ whitened vectors in all ({ref}`dist-cost`), against $2J$ for calling the map on the
particles. It agrees with
`coefficients(g.realize_particles(exclude_block_covs=given), ...)` to
round-off, not bit-exactly. A map built from a plain `Gaussian` raises
`ValueError`: its latent coordinates are not particles.

Unlike `SquareRootMap`, this method returns coefficients rather than
particles. The caller applies them to its own particle arrays,
$x_j + F_x w_j$, so particles whose given anomalies are exactly zero come
back bit-identical rather than re-realized to round-off.

A `MatheronMap` applied to samples of the joint, with their noise, returns
exact samples of the conditional. Applied to particles fitted by the
Gaussian with the divisor $J - 1$, its output's sample mean and covariance
(divisor $J-1$) are unbiased for the conditional's moments under the noise
draw; individual images are not conditional samples, since given the
particles, image $j$ is distributed
$\mathcal N\big(x_j + K(y^* - y_j),\ K D_c K^\top\big)$. The sample mean has
variance exactly $K D_c K^\top / J$.

(dist-square-root-map)=
### `SquareRootMap`

Moves the particle set of the `EnsembleGaussian` it was built from, as a
whole. Called with $y^*$, for each target block $x$,

$$
x_j' = m_x + F_x\, A^{-1}S\,W(y^* - m_c) + \sqrt{\delta}\,F_x T e_j
\;\big[+\, L_x\,\eta^{(x)}_j\big], \qquad j = 1, \dots, J,
$$

with $\delta$ the Gaussian's divisor. This is
`g.condition(values).realize_particles(key=key)` with everything that
does not depend on $y^*$ computed once, at build: the `IdentityPlusGram` of
$S$, and each target's realized conditional anomalies
$\sqrt{\delta}\,(F_xT)^\top$ as a `(J, d_x)` array. A call whitens one vector
per given block, computes $w$ by `solve_factor`, and adds $F_x w$ to the
target means and stored anomalies. A target with no factor row realizes as
its mean. A target with an independent term is sampled, which needs `key`
(`ValueError` when missing) and its term's `factor`, checked at build; the
draw is `realize_particles`'s ({ref}`dist-prng`).

It has no samples argument: it acts through each particle's latent
coordinate $e_j$, so it is defined only for the particles the Gaussian was
built from. It is therefore not a `ConditionalMap`.

`__call__(values=None, /, *, key=None, **block_values) -> Ensemble`,
unweighted, over the targets in block order. Attributes: `given`, `targets`,
`n_particles`.

This is the symmetric square-root update of the ensemble transform Kalman
filter (Bishop et al., 2001; Hunt et al., 2007), written in the latent
coordinates of the projected Gaussian. For a linear-Gaussian joint whose
particles' moments equal the joint's, and whose targets have no
independent terms (a sampled term adds sampling error), the output's
mean and covariance (divisor $\delta$) equal the exact conditional's, in
exact arithmetic.

(dist-weights)=
## Weights

**`reweight(ensemble, log_weight_increments)`** multiplies each particle's
weight by $\exp(\Delta\ell_j)$:

$$
\ell_j \mapsto \ell_j + \Delta\ell_j ,
$$

treating an unweighted ensemble as all zeros, so importance weights,
likelihood factors compose by repeated calls without leaving log space. `log_weight_increments` is exactly `(J,)` (tier 3). An entry of
$-\infty$ gives that particle weight zero. `nan` or $+\infty$ entries, and an
increment that leaves every log weight $-\infty$, are value violations: in
debug mode `ValueError`; otherwise the weights are `nan`. The result is
always weighted.

**`effective_sample_size(ensemble)`**, a 0-d array:

$$
\mathrm{ESS} = \frac{1}{\sum_j w_j^2}
= \exp\big(2\operatorname{lse}(\ell) - \operatorname{lse}(2\ell)\big),
$$

computed in the second form, which stays finite when the log weights span
hundreds of units (Kong et al., 1994). It is $J$ when unweighted.

**`resample(key, ensemble, n_particles=None, *, scheme="systematic")`**
draws an unweighted ensemble from a weighted one, with duplicated particles
where the weights were large. `n_particles` defaults to the input's and is
otherwise a Python `int` $\ge 2$. `scheme` is `"systematic"` (lower variance)
(Kitagawa, 1996) or `"multinomial"` (Gordon et al., 1993); anything else is
`ValueError`. An unweighted input is
resampled with equal weights. The draws are pinned ({ref}`dist-prng`). The
selected indices are discrete, so the result's derivative with respect to the
weights is zero.

(dist-exact-moments)=
## `exact_moment_ensemble(key, gaussian, n_particles)`

Particles whose sample mean and covariance (divisor $J-1$) equal
`gaussian`'s exactly, jointly: every block's mean and covariance and every
cross-covariance (the second-order exact sampling of Pham, 2001). The
Gaussian has $s = k + \sum_b w_b$ **sources**: the
latent vector and the columns of each independent term's factor, $w_b$ for
block $b$ (so every term must support `factor`).

1. Draw $Z$, a `(J, s)` standard normal array (pinned, {ref}`dist-prng`).
2. Center its columns, $Z \mapsto Z - \mathbf 1\bar z^\top$.
3. Orthonormalize with a thin QR, $Z = QR$; $Q$'s columns are orthonormal and
   orthogonal to $\mathbf 1$. Never a Cholesky of a sample covariance, which
   raises on the rank-deficient targets that matter.
4. Split $Q$'s columns into the latent part $Q_\xi$ (the first $k$) and one
   part $Q_b$ per independent term, in block order, and set

   $$
   x_j^{(b)} = m_b + \sqrt{J-1}\,\big(F_b\, (Q_\xi)_{j\cdot}^\top
     + L_b\,(Q_b)_{j\cdot}^\top\big).
   $$

Since $Q^\top Q = I$ and $Q^\top\mathbf 1 = 0$, the sample mean is $m$ and the
sample covariance is $FF^\top + \operatorname{blockdiag}(L_bL_b^\top)$, to
round-off. It requires $J > s$, a shape condition that always runs
(`ValueError`): at $J = s$ the centered columns have rank at most $s - 1$ and
the construction fails silently.

It is a **fixture**, not a reference implementation. Tests use it to build
inputs whose moments are known exactly; the results those inputs produce are
still compared against references written out by hand in the tests
({ref}`dist-conformance`). Documentation uses it where numbers must not depend
on sampling error.

(dist-prng)=
## Randomness

- Keys are **typed** keys (`jax.random.key`). A raw `uint32` key raises
  `TypeError`: its trailing shape makes a family of keys ambiguous.
- Always-random functions take the key first (`sample`, `resample`,
  `exact_moment_ensemble`). Sometimes-random ones take a keyword-only `key=`
  (`realize_particles`, `SquareRootMap`) and raise `ValueError` when a needed
  key is missing; a key passed where none is needed is ignored, though it is
  still type-checked. Two are different: in `MatheronMap`'s call and
  `coefficients`, the key's presence says whether the samples carry their
  noise ({ref}`dist-matheron`), and `particle_coefficients` always requires
  one.
- A key is **consumed whole**. Nothing stores a key, advances one, or splits
  it for uses beyond the one call.
- The draws are **pinned**. Same key, same arguments, same representation give
  identical arrays across EnsKit releases, for a fixed JAX version and PRNG
  configuration (the bit stream belongs to JAX). Changing a draw is a
  breaking change, and the test suite snapshots each so that a JAX-side
  change is detected:

| call | draw |
| ---- | ---- |
| `Gaussian.sample` | `keys = split(key, 1 + n_D)`, with $n_D$ the number of blocks with an independent term; `normal(keys[0], (n, k))` for $\xi$ when $k > 0$ (`keys[0]` is unused at $k = 0$); `normal(keys[1 + i], (n, w_b))` for the $i$-th such block in block order |
| `realize_particles` | `keys = split(key, n_R)`, with $n_R$ the number of realized blocks with an independent term that is not excluded; `normal(keys[i], (J, w_b))` for the $i$-th in block order |
| `MatheronMap`, `coefficients`, `particle_coefficients` | `normal(key, (n, N))`, its columns the given blocks' whitened coordinates in block order |
| `SquareRootMap` | as `realize_particles`, over the target blocks |
| `exact_moment_ensemble` | `normal(key, (J, s))`, columns the latent vector then each term's factor columns in block order |
| `resample`, systematic | $u =$ `uniform(key, (), dtype)`, `c = cumsum(w)`; indices `minimum(searchsorted(c, c[-1] * (u + arange(n)) / n, side="right"), last)`, with `last` the index of the last particle of positive weight |
| `resample`, multinomial | `categorical(key, log_weights, shape=(n,))`, with zeros for `log_weights` when unweighted |

  All normal draws have the distribution's dtype. Pinning is over evaluation
  in one mode: `jit`-compiled and eager evaluation of the same call may differ
  in the last bits.

(dist-validation)=
## Validation and errors

The four tiers of the operator contract ({ref}`contract-validation`) apply:
everything static is checked always, and values only in debug mode
(`enskit.linalg.debug_checks`), on concrete arrays. Tier 4 here also runs at
call time, on operands, on intermediate whitened values, and on results
({ref}`dist-accuracy`). Like the operator layer's, tier-4 checks read array
values and are skipped inside a trace, including when the arrays are concrete
but closed over by a live trace: `jnp.isfinite` of a concrete array is staged
into the trace, so testing whether the operand is a tracer is not enough.
The guard is the operator layer's `value_check`, which skips a check whose
outcome is not concrete.

| tier | checks | examples |
| ---- | ------ | -------- |
| 2. construction | ranks, sizes, dtypes, operator types and shapes, block names, the representation rules | a mean not 1-D; an ensemble block not 2-D; $J < 2$; a factor row of the wrong height or width; a term not a `PSDLinOp`; a block with neither part; `n_particles` not the factor width; mixed dtypes |
| 3. call | block names; static arguments; value core shapes; structural conditions on the given blocks; capabilities; key presence and type | an unknown block; a block given twice; a value not `(d_c,)`; mixed noisy and exact given blocks; exact size conditions; `n_particles` not an `int` $\ge 2$; a raw `uint32` key |
| 4. value (debug) | finiteness of means at construction; finiteness of particles of positive weight in `project` and `cov`; centering of an `EnsembleGaussian`; finiteness of values and samples at call; the whitening check; exact-case rank; concentrated weights in `project`; non-finite increments in `reweight`; the result check | violations give `nan`, `inf`, or a finite wrong answer outside debug mode |

**Order of checks** within a method, so that when two things are wrong the
same one is always reported:

1. the family guard ({ref}`dist-jax`);
2. block names and argument structure (unknown names, names given twice or
   repeated, static non-array arguments such as `n_particles`, `scheme` and
   `max_dim`, then the key: its type, then its presence where needed);
3. structural conditions on the named blocks (each given block's kind, mixed
   noisy and exact, size conditions, at least one target);
4. required capabilities, block by block in block order, operations in the
   order the method names them;
5. operand core shapes;
6. tier 4 on operands;
7. the work, with the whitening check immediately after whitening;
8. the result check.

Names come before capabilities because which operator to ask depends on which
blocks were named.

The layer defines **no exception types**. `UnsupportedOpError` comes only
from the operator layer, raised unmodified, and the layer never falls back to
dense linear algebra on the caller's behalf except inside the explicit
{func}`~enskit.linalg.dense_fallback` context, which acts at the operator
layer. Error messages name the object (its `repr`), the method, the block,
the expectation, and the offending value's shape or type.

| condition | raises |
| --------- | ------ |
| wrong rank, size or disagreeing shapes at construction | `ValueError` |
| wrong type of field (a term not a `PSDLinOp`, a row not a `LinOp` or array, a non-floating block) or mixed dtypes | `TypeError` |
| a block with neither a factor row nor an independent term | `ValueError` |
| an unknown block name at a call | `KeyError` |
| a `factors` or `block_covs` key naming no block, at construction | `ValueError` |
| a block given twice, by mapping and keyword | `TypeError` |
| a name repeated in a positional list of names | `ValueError` |
| a value's core shape mismatched | `ValueError` |
| no blocks left (conditioning or dropping every block) | `ValueError` |
| mixed noisy and exact given blocks | `ValueError` |
| exact-value size conditions | `ValueError` |
| `conditional_map` or `square_root_map` on a block with no independent term | `ValueError` |
| `particle_coefficients` on a map not built from an `EnsembleGaussian` | `ValueError` |
| `n_particles` not an `int`, or below 2 | `TypeError` / `ValueError` |
| a needed key missing | `ValueError` |
| a raw `uint32` key | `TypeError` |
| a capability missing (`whiten`, `logdet`, `factor`) | `UnsupportedOpError`, from the operator layer |
| `compress` over `max_dim` | `ValueError`, before allocating |
| any method or array-computing property on a vmapped family | `ValueError` |
| a value precondition violated, a whitening or a result not finite | `ValueError` in debug mode; `nan`, `inf` or a wrong finite result otherwise |

Arguments that two coinciding dimensions would let a caller swap silently are
keyword-only (`n_particles` on `EnsembleGaussian`, `latent_dim`, `max_dim`,
`key` on sometimes-random calls).

(dist-derivatives)=
## Differentiability

Every operation in this layer is differentiable with respect to every array
leaf it reads (means, factor rows, operator parameters inside independent
terms, particle arrays, given values and log weights), with `jax.grad`,
`jax.jvp` and `jax.vjp` returning finite, correct derivatives wherever the
function is smooth. What holds beyond that is narrower than the design's
§9.1 stated, as PR 2 measured ({ref}`contract-gram`, issue #52):

| operation | first derivatives | second derivatives |
| --------- | ----------------- | ------------------ |
| noisy `condition`, both maps, noisy `log_density`, noisy `regression`, `cov(a)` with both parts (`LowRankUpdate`'s `solve`, `whiten`, `logdet`) | finite and correct at every input, including exactly repeated and exactly zero singular values of $S$ (a collapsed ensemble, constant given coordinates, zero-padded columns); forward and reverse mode, under `jit` and `vmap` | exact at well-separated singular values; near a tie of gap $\delta$ they lose accuracy like $\varepsilon/\delta$, and `jax.hessian` may be `nan` at an exact tie. Only the log-determinant parts are exact everywhere |
| exact-value `condition` and `log_density`, `compress` (QR), `regression` without independent terms | finite where the factorized matrix has full rank, which is also where the function is defined or unique | not promised |
| `resample`, keys | zero by declaration (discrete) | — |
| `exact_moment_ensemble` | linear in the Gaussian's parameters: its QR reads only the key's draw | exact |

- At degenerate spectra, forward mode computes the SVD's own `nan` tangents
  and discards them, so it raises under the `jax_debug_nans` flag although
  its result is finite. Reverse mode does not.
- **Stochastic operations are reparameterized.** For a fixed key, `sample`,
  `realize_particles`, the maps and `exact_moment_ensemble` are deterministic
  functions of their parameters, and their derivative is the pathwise
  derivative of that realization. Because the maps draw noise in whitened
  coordinates, the dependence on an independent term enters through $W$ and
  is differentiated exactly.
- `log_density` is the hyperparameter objective: its first derivative is
  degenerate-safe, and a second-order optimizer through it is exact only at
  well-separated spectra, except for the log-determinant term.

(dist-jax)=
## JAX integration

All classes are frozen pytrees, declared with explicitly separated data and
metadata, and every rule of the operator contract's JAX section
({ref}`contract-jax`) binds them: construction validates, unflattening
bypasses the constructor, objects compare and hash by identity and are never
`static_argnums`, and nothing is cached lazily.

**Fields.** A field is data if and only if it is an array, a `LinOp`, a
distribution, or a tuple of these in which entries may be `None`; everything
else is static metadata and hashable. The fields are:

| class | data | static |
| ----- | ---- | ------ |
| `Ensemble` | the block arrays (a tuple, in order); `log_weights` (an array or `None`) | `names` |
| `Gaussian` | means, factor rows and terms (three tuples in block order, absent entries `None`) | `names`, `latent_dim` |
| `EnsembleGaussian` | as `Gaussian` | as `Gaussian`, and `n_particles` |
| `MatheronMap` | the `IdentityPlusGram`, the given terms and means, the target rows | `given`, `targets`, `latent_dim`, and `n_particles` or `None` |
| `SquareRootMap` | the `IdentityPlusGram`, the given terms and means, the target means, rows, conditional anomalies and terms | `given`, `targets`, `n_particles` |

A `None` entry is part of the tree structure, so whether a block has a factor
row or an independent term is static: two Gaussians that differ in it are
different pytree types, and a `vmap` or `jax.tree.map` across them raises.
So is whether an ensemble is weighted. Consequently the carry of `lax.scan`
or `lax.while_loop` must leave each iteration with the structure it entered
with (issue #56): an unweighted ensemble carried through `reweight` and then
`resample` is fine, since it comes back unweighted, but one carried through
`reweight` alone, or a Gaussian that gains an independent term from
`add_noise`, is not; nor is resampling conditionally under `lax.cond`, whose
two branches return different structures. PR 4
extends the registration machinery to these fields (tuples containing
`None`) without exposing a public decorator; the set of classes is closed.

**Families.** Every class has a `batch_shape` property computed as the
operators' is ({ref}`contract-families`): each array contributes its leading
axes beyond its core rank (1 for a mean, 2 for an ensemble block, 1 for
`log_weights`), each operator its own `batch_shape`, combined by
broadcasting, `ValueError` on mismatch. Directly constructed objects report
`()`. When `batch_shape` is non-empty, every method and every
array-computing property (`weights`, `all_finite`, `mean`, `anomalies`,
`cov`, `project`, every `Gaussian` method, the maps' calls) raises
`ValueError` naming the object, the operation, the batch shape and the
remedy, apply the family under `jax.vmap`, before any other check. The
static properties (`names`, `dims`, `latent_dim`, `n_particles`,
`is_weighted`, `given`, `targets`), `batch_shape` and `repr` still answer,
and sizes are core sizes, never batch sizes. Construction with a family
operator is a tier-2 `ValueError`.

Every conditioning method and map is `jit`- and `vmap`-safe with no
data-dependent shapes.

(dist-repr)=
## `repr`

Type name and static sizes, never array contents:

```text
Ensemble(n_particles=100, blocks={'x': 40, 'theta': 1}, weighted=False)
Gaussian(blocks={'u': 4, 'g': 6}, latent_dim=104, block_covs=('g',))
EnsembleGaussian(n_particles=100, blocks={'u': 4, 'g': 6}, block_covs=('g',))
EnsembleGaussian(n_particles=100, blocks={'u': 4}, block_covs=(), unbiased=False)
MatheronMap(given=('g',), targets=('u',), latent_dim=100)
SquareRootMap(given=('g',), targets=('u',), n_particles=100)
```

A vmapped family wraps that form and names its batch,
`vmapped(Ensemble(n_particles=100, blocks={'x': 40}, weighted=False), batch=(8,))`.
`repr` never raises: an instance whose sizes cannot be read falls back to a
marker form.

(dist-surface)=
## Public surface

`enskit.distribution` exports exactly: `Ensemble`, `Gaussian`,
`EnsembleGaussian`, `Regression`, `ConditionalMap`, `MatheronMap`, `SquareRootMap`,
`reweight`, `effective_sample_size`, `resample` and `exact_moment_ensemble`.
Anything else is private. The module's private modules are imported only from
inside `enskit.distribution`. There is no `enskit.distribution.testing`;
`check_conditional_map` lives in `enskit.testing`, which may import this
layer and is imported by nothing.

(dist-consumers)=
## How the layers above consume this one

Not normative for this layer, but the design was shaped against these call
sites, and a change that breaks them is a change to reconsider.

**`enskit.kalman`** builds every update from three pieces of this layer:

```python
approx = ens.project().add_noise(g=R)           # gaussian_approximation
smap = approx.square_root_map("g")                # SymmetricSquareRoot.build
post = smap(g=y_obs)

cmap = approx.conditional_map("g")                # Matheron.build
w = cmap.particle_coefficients(g=y_obs, key=k)    # J + 1 whitened vectors
u_new = ens["u"] + approx.factor("u").matvec(w)
```

`Matheron` uses `particle_coefficients` on an aligned approximation and the
map's call on any other (a hybrid covariance, a plain `Gaussian`), adding
draws from target blocks' independent terms itself. Both rules refuse
weighted ensembles, whose projection is not aligned; the message says to
resample first. The tempered noise $R/\Delta\beta$ is the operator layer's
scalar scaling, so a traced increment flows through a 0-d field and nothing
re-factorizes.

**`enskit.maps`** builds new Gaussians through the public constructor and
methods: a linear map of a block with no independent term appends a row
$HF_b$ on the same latent space (an `EnsembleGaussian` stays one); of a block
with one, it calls `absorb` first. `AdditiveNoise` on a Gaussian is
`add_noise` to a new block.

**Localization** (PR 9) reads the particles and factor rows of an aligned
approximation and builds one `IdentityPlusGram` per neighborhood itself: its
local noise is tapered (`diag_congruence`), so the local whitened factor
differs from any slice of a global $S$ and is computed anew in any case.

**The algorithms** read `all_finite` to find failed particles, and decide
repair themselves; this layer never repairs. An `Ensemble` holding failed
particles is valid, so `pushforward` builds one directly from a simulator's
output ({ref}`dist-failed-particles`).

(dist-conformance)=
## Conformance

Conformance is a set of obligations on `tests/`. Exactness tests check
against closed forms at a tolerance of a few $\varepsilon$ times the
quantity's natural scale, never a tolerance chosen to pass, and the dense
reference is **hand-written in the tests** (plain dense linear algebra over
means, anomalies and materialized operators, never routed through this layer
or `IdentityPlusGram`), so every comparison is between independent paths.
`exact_moment_ensemble` may build inputs; it is never the reference for an
output. The suite must verify at least:

1. **Gain against dense.** The conditional mean of noisy `condition` equals
   the dense $m_x + C_{xc}(C_{cc} + D_c)^{-1}(y^* - m_c)$ on small random
   problems, in the three regimes $N > k$, $N = k$, $N < k$, with one and with
   several given blocks, and with given blocks lacking a factor row.
2. **Whitener invariance.** Two independent terms for the same matrix with
   different whiteners give the same results to round-off. No shipped pair
   differs, so the test defines a local operator that rotates a valid
   whitener.
3. **The transform.** The conditional factor $F_xT$ reproduces the dense
   conditional covariance; at large $\sigma_{\max}$, the stably formed
   invariant $TT^\top + (TS)(TS)^\top = I$ holds to
   $\varepsilon\max(1, \sigma_{\max})$, with a spectrum mixing large singular
   values and ones of order 1 (the re-formed $T(I + SS^\top)T^\top$ must not
   be used). For a centered factor, $T\mathbf 1 = \mathbf 1$ to
   $c_1 J\varepsilon + c_2(\varepsilon\sigma_{\max})^2$, at more than one $J$.
4. **Exact values.** Exact `condition` and exact `log_density` equal the dense
   forms; the size conditions raise, including $J = N$ on an
   `EnsembleGaussian`; the rank check raises in debug mode.
5. **Square root.** `SquareRootMap` equals
   `condition(...).realize_particles()` to round-off; its output's sample mean
   and covariance equal the dense conditional's on exact-moment inputs; it
   computes one SVD at build and none per call.
6. **Matheron.** The map equals the dense $x + K(y^* - y - W^{-1}\varepsilon)$
   elementwise, with $\varepsilon$ recomputed from the same key and $W$
   recovered as the transpose of `D.whiten(eye(N))`. On joint samples built
   with exact moments together with an independent noise block, the
   transported sample moments equal the conditional's. `particle_coefficients`
   agrees with `coefficients` on the realized noise-free particles to
   round-off.
7. **Project and realize.** `project()` reproduces every mean, block
   covariance and cross-covariance against dense references, unweighted and
   weighted; `realize_particles()` after `project()` returns the particles to
   round-off; the unweighted factor is centered; the weighted divisor stays
   accurate at an ESS of $1 + 10^{-6}$ and raises, in debug mode, when the
   weights are concentrated to working precision.
8. **Exact moments.** `exact_moment_ensemble` reproduces means, covariances and
   cross-covariances jointly, with and without independent terms, and raises
   at $J = s$.
9. **Densities and sampling.** `log_density` matches the dense closed form at
   batch ranks 0, 1 and 2, for one block, several, a marginal, and $k = 0$;
   `sample` and `realize_particles` match their pinned definitions
   elementwise.
10. **Structure.** `marginal`, `drop`, `rename`, `add_noise`, `absorb` and
    `compress` leave the distribution unchanged (dense covariance compared
    before and after) and return the documented kind; `absorb` pads with
    `Zero` and materializes nothing; `compress` raises over `max_dim` before
    allocating; `cov` returns the documented operator for each case.
11. **Weights.** `weights` and `effective_sample_size` are correct for log
    weights spanning hundreds of units; `reweight` composes; both resampling
    schemes match their pinned definitions.
12. **Degeneracy.** Zero given anomalies (every particle given the same,
    exactly representable value) make `condition` return the targets'
    moments, `SquareRootMap` return the particles to round-off, and
    `particle_coefficients` return exact zeros, so the particles plus
    $F_x w$ are bit-identical to the particles; $J = 2$ and $N = 1$ work; a
    collapsed ensemble with finite inputs gives no `nan`.
13. **Validation.** Every tier-2 and tier-3 rule of {ref}`dist-validation`
    raises as specified, and the check order is pinned with two simultaneous
    violations per case (one violation at a time pins no order).
14. **Capabilities.** A given block's term without `whiten`, a density without
    `logdet`, and a sampled or absorbed term without `factor` raise
    `UnsupportedOpError` before any work.
15. **Counts.** Whitened vectors and SVDs per operation equal
    {ref}`dist-cost`'s table, by instrumented operators: in particular
    $J + 1$ for `particle_coefficients` plus the map's build, $2J$ through the
    map's call, $k + 1$ for `condition`, and one SVD per build.
16. **Diagnostics.** In debug mode the whitening check fires on a singular
    diagonal term, and the result check fires on an overflowing target row,
    each from every conditioning path, with the cause in the message; neither
    fires under `jit` or `vmap`, nor inside a trace over closed-over concrete
    arrays; outside debug mode the singular term gives `nan` without raising.
17. **Accuracy.** The measured regimes of {ref}`dist-accuracy`: on given
    rows along latent coordinate axes, the mean accurate to a few
    $\varepsilon$ at $\sigma_{\max}$ up to $10^{20}$, and the covariance and
    `log_density` when $\rho = k$ (for `log_density`, $N \le k$); on rotated
    rows, at $\rho = k$ and $\rho < k$, mean and covariance errors no larger
    than the exact conditional's own change under a perturbation of $F_c$ by
    $\varepsilon\sigma_{\max}$; `log_density` far more accurate than the
    cancelling form at $\sigma_{\max} = 10^8$; and, as regressions pinned
    until issue #54 is settled, the $\varepsilon\sigma_{\max}$ loss when
    $\rho < k$, on the axis-aligned case and on one seeded random direction,
    each at its current value.
18. **Derivatives.** First derivatives of noisy `condition`, both maps and
    `log_density` are finite and agree with finite differences at exactly
    repeated singular values, at zero-padded columns and at a collapsed
    ensemble, in forward and reverse mode, under `jit` and `vmap`.
19. **JAX.** Flatten and unflatten preserve type and behavior for every
    class; conditioning and the maps run under `jit`; a vmapped family of
    each class reports its batch shape, takes the `vmapped(...)` repr, refuses
    every method, and answers the static properties; constructing inside
    `vmap` round-trips; a vmapped family agrees with a Python loop over its
    members; unflattening with sentinel leaves (`object()`, as
    `jax.custom_vjp` does) succeeds, with `None` entries in the tuples; and a
    gradient through every conditioning operation computes one SVD.
21. **Dtypes.** No operation promotes a distribution's dtype: a float32
    Gaussian conditions, absorbs (its `Zero` padding included), samples and
    realizes in float32, so its results can be a `lax.scan` carry.
22. **`repr` and messages.** Every class's `repr` matches {ref}`dist-repr`
    and contains no array data, and never raises on a malformed instance;
    error messages name the object, the method, the block and the offending
    value's shape or type.
20. **Reproducibility.** Same key, same output elementwise; different keys
    differ; every pinned draw is snapshotted.
23. **Regression** (`tests/test_linearization.py`). Each case against a
    hand-written reference: the unique case against least squares with an
    unpenalized intercept, and against `condition` at several values; the
    minimum-norm case against the centered pseudo-inverse, and against
    $GP_{\mathrm{span}}$ when the target is affine in the given block; the
    noisy case against
    $\hat C_{yx}(\hat C_{xx} + \Lambda)^{-1}$ for a diagonal and a dense
    $\Lambda$; a weighted ensemble against weighted least squares; a
    linear-Gaussian joint returning $(H, R)$ one way and
    $C H^\top(HCH^\top + R)^{-1}$ with the conditional covariance the other. The two exact
    computations agree at $N = J - 1$; every returned operator passes
    `check_operator`; derivatives in each case are finite and equal a dense
    reference's; the counts above hold; the validation and debug rank
    checks raise. Two regressions: square, non-symmetric coefficients are
    recovered untransposed, and the minimum-norm intercept is not penalized
    (`pinv([1, X])` differs).
24. **The divisor** (`tests/test_divisor.py`). `cov` and `project` against
    the closed forms of {ref}`dist-divisor` for both divisors, unweighted
    and weighted; uniform weights give the unweighted divisor; the two differ
    by exactly $(J-1)/J$ or $1 - \sum_j w_j^2$; concentrated weights have an
    empirical covariance. No path mixes divisors: every method that keeps the
    latent space keeps `unbiased`; `realize_particles` returns the particles;
    the square-root update with $\delta = J$ and noise $cR$ equals the one with
    $J - 1$ and $R$, $c = (J-1)/J$; `particle_coefficients` equals
    `coefficients` of the realized particles. Each of these fails when any
    one reading of the divisor is replaced by $J - 1$.

### Ported regression tests

Each regression test of `tests/test_gauss.py` guards a silent failure class
that this layer can also have, and is ported to it in PR 4 rather than
deleted. Under the do-not-delete rule, the ported tests keep their
docstrings' reasoning, updated to this layer's names:

| `tests/test_gauss.py` | becomes, for this layer |
| --------------------- | ----------------------- |
| `the_square_root_reading_needs_a_centered_factor` | the uncentered-factor error measured against a plain `Gaussian`'s conditional, the reason `realize_particles` exists only on `EnsembleGaussian` |
| `independently_chosen_factors_lose_the_cross_covariance` | the same, for factor rows of two blocks chosen separately |
| `transporting_arbitrary_realizations_costs_one_whitening_each` | obligation 15: the map's call against `particle_coefficients` |
| `thin_svd_needs_the_identity_completion` | obligation 3, through `condition` at $N < k$ |
| `stably_formed_invariant_beats_the_re_formed_one` | obligation 3's invariant |
| `mixing_perturbation_representations_corrupts_the_update` | a `MatheronMap` whose noise is pushed through `factor()` too |
| `uncentered_transform_shifts_the_ensemble_mean` | `SquareRootMap` and `realize_particles` after `condition` |
| `gradient_is_finite_at_an_exactly_collapsed_operand` | obligation 18 |
| `singular_noise_covariance_yields_nan_without_raising` | obligation 16, outside debug mode |
| `all_three_methods_report_a_nan_result_in_debug_mode` | obligation 16, every conditioning path |
| `result_checks_are_skipped_under_jit` | obligation 16 |
| `sample_is_outside_the_result_check_by_design` | `sample` and `realize_particles`, unchanged |
| `each_update_applies_the_whitener_j_plus_one_times` | obligation 15 |
| `anomalies_are_centered_before_whitening` | the whitening order of {ref}`dist-core`, checked for accuracy against an exact reference |
| `a_collapsed_ensemble_is_exact_at_any_magnitude` | `Ensemble.anomalies` at magnitude $6\times10^{23}$, and obligation 12 |
| `debug_checks_survive_a_trace_over_closed_over_arrays` | obligation 16 |
| `check_order_with_two_simultaneous_violations` | obligation 13 |
| `anomalies_are_formed_over_the_member_axis_when_batched` | `Ensemble.anomalies` and `project` under `vmap` with a leading axis equal to $J$ |

New regression tests join them: the weighted projection with one dominant
weight; exact moments jointly across blocks; exact conditioning with $J = N$;
the latent width kept by a marginal over blocks without factor rows; the
cancelling log-density form; a structured factor row left undensified by
noisy conditioning; and, for failed particles ({ref}`dist-failed-particles`),
a `nan` particle at weight zero, first and in the middle, affecting no
moment, projection or derivative, a finite outlier at weight zero costing
nothing, a weighted collapsed ensemble staying exact beside one, and debug
mode accepting one at construction and refusing it at positive weight in
`project` and `cov`. (The design review's dropped target noise, a variance of
0.044 where the conditional has 4.04, was a defect of the stochastic rule,
which samples target terms; its regression belongs to PR 6.)

(dist-departures)=
## Departures from the design

The stubs in `docs/redesign/stubs/` are the design; where this page differs,
this page governs. Each departure, and why:

1. **`MatheronMap.particle_coefficients` is new.** The stubs left the
   $J + 1$ route to `kalman.Matheron` "reading the residuals off $S$", which
   would mean reaching into the map's private state from another layer. The
   route is now a public method of the map that holds $S$.
2. **Value checks follow the tiers.** The stubs had `project` check
   concentrated weights eagerly and `reweight` raise on `nan` and $+\infty$.
   Both read values, so both are tier 4: debug mode only, `nan` otherwise.
3. **Every block has a factor row or an independent term.** The stubs
   allowed a block with neither.
4. **`square_root_map` takes only blocks with independent terms**, as
   `conditional_map` does. Exact values go through
   `condition(...).realize_particles()`.
5. **`cov(a)` of a block with only a factor row** materializes the row as a
   `PSDLowRank`; `cov(a, b)` with an absent row is a `Zero`.
6. **`absorb` raises on a block with no independent term** rather than
   skipping it; `add_noise` absorbs only blocks that have one.
7. **`compress` always returns a plain `Gaussian`**, whether or not it
   compressed, and its guard applies to the dimension of the blocks that have
   factor rows.
8. **Exact-value conditioning materializes target rows.** The stubs did not
   say how $I - QQ^\top$ was held.
9. **`sample` requires `n_particles >= 2`**, the `Ensemble` minimum.
10. **`Gaussian.rename` takes keywords too**, as `Ensemble.rename` does.
11. **The pinned draws are specified** ({ref}`dist-prng`), including the
    split arity, and `resample`'s systematic scheme uses
    `searchsorted(..., side="right")`.
12. **Differentiability is narrowed** to first derivatives at degenerate
    spectra ({ref}`dist-derivatives`), as PR 2 measured; the design's §9.1
    claimed it for the operations without qualification.
13. **`Gaussian`'s `factors` and `block_covs` are keyword-only**; the stubs
    had them positional, where a covariance passed as a factor row is silent.
14. **`MatheronMap.particle_coefficients` requires a key**, and the call's
    key semantics (presence means the samples are noise-free) are stated
    rather than implied.
15. **`log_density` computes its quadratic term through
    `LowRankUpdate.whiten`**, not the cancelling
    $\lVert b\rVert^2 - \langle Sb, A^{-1}Sb\rangle$ the stub's formula
    suggests.

(dist-implementation-changes)=
## Changes made while implementing

PR 4 implemented this page and corrected it where the implementation showed
it to be wrong or incomplete; change 8 followed it, from issue #60. Each
change, and why:

1. **The accuracy claim is narrowed** ({ref}`dist-accuracy`). The page said
   the conditional mean was accurate to a few $\varepsilon$ at every
   $\sigma_{\max}$, and the covariance whenever $\rho = k$. That holds only
   where the decomposition of $S$ is exact, as for the axis-aligned rows it
   was measured on. For rows in general directions the conditional itself
   moves by about $\varepsilon\sigma_{\max}$ under a perturbation of $F_c$ of
   one rounding, so no normwise-stable algorithm can do better, and the layer
   does not do worse. Obligation 17 now tests both statements.
2. **A given block with no factor row whitens only its residual**
   ({ref}`dist-cost`), rather than $k$ columns of zeros. The results are
   identical, since whitening is linear.
3. **`EnsembleGaussian`'s `factors` and `block_covs` are keyword-only**, as
   `Gaussian`'s are and for the same reason; the page's signature had them
   positional.
4. **`compress`'s size guard applies only when compression would
   allocate**, and `compress` drops the latent space when no block has a
   factor row ($D_F = 0 < k$), which the page did not cover and which raised.
   The page's guard read as unconditional, which would refuse a call that
   computes nothing.
5. **The log-sum-exps of the weighted projection use `log1p`**
   ({ref}`dist-project`). The page's formula was right; computing its
   $\operatorname{lse}$ terms with a plain logarithm lost about $10^{-10}$ at
   an ESS of $1 + 10^{-6}$, which obligation 7 measures.
6. **`log_density` skips $S$ when no named block has a factor row**, as it
   does at $k = 0$; the result is the same.
7. **Systematic resampling never selects a particle of weight zero.** The
   pinned formula `clip(searchsorted(cumsum(w), (u + arange(n)) / n), 0, J - 1)`
   selected the last particle when the cumulative sum rounded below 1 and the
   last position fell past it, with probability about $J\varepsilon$ per call
   even when that particle had weight zero, which is how a failed particle is
   marked. The positions are now scaled by the cumulative sum's last entry,
   and the index is bounded by the last particle of positive weight
   ({ref}`dist-prng`).
8. **Failed particles are a valid state** ({ref}`dist-failed-particles`,
   issue #60). Tier 4 checked the particles' finiteness at construction,
   which rejected, in debug mode, every ensemble holding a failed particle,
   before the layers above could repair it. And a `nan` particle
   of weight zero made every weighted moment `nan`, since $0 \cdot
   \mathrm{nan} = \mathrm{nan}$, whether it was the reference of the
   differences (particle 1, as it was) or not. The finiteness check moved
   to `project` and `cov`, for particles of positive weight; the sums leave
   out particles of weight zero; and the reference of a weighted ensemble is
   a particle of largest weight.

(dist-excluded)=
## Deliberately excluded

Recorded so their absence reads as a decision.

**The Woodbury route.** Applying the Woodbury identity to
$(F_cF_c^\top + D_c)^{-1}$ forms $I + SS^\top$, and forming it rounds away
every singular value below $\sqrt\varepsilon\,\sigma_{\max}$: at
$\sigma_{\max} = 10^8$ it destroys every singular value below 1.5, the
largest gain multipliers among them, for a relative error near $10^{-1}$
where the SVD route holds $10^{-8}$. Rejected, not deferred. ({doc}`design`
records the full comparison.)

**A data-space route for few given coordinates.** When $N$ is small,
factorizing $F_cF_c^\top + D_c$ densely is affordable and can beat the SVD.
It may arrive as an internal optimization behind the same signatures, with
results equal to round-off, and never as the test reference.

**Conditioning with a singular independent term.** The conditional exists
whenever $F_cF_c^\top + D_c$ is nonsingular, but representing it needs a
route that does not whiten $D_c$, and no consumer needs it. Exact-value
conditioning covers the case where every given block is noise-free.

**A condition-number diagnostic on $S$.** It cannot separate a numerically
singular term from a vague factor against precise values
({ref}`dist-accuracy`).

**A reified decomposition object.** The two maps are what a reusable
decomposition would be; they are built once and called per value. An
adaptive search over noise scalings that wants one SVD across candidate
scalings would need more ($S(D/\delta) = \sqrt\delta\,S(D)$, with weight
multipliers $\delta\sigma_i/(1 + \delta\sigma_i^2)$), and waits for that
consumer.

**Weighted square roots and alignment.** A weighted projection is not
aligned, and a weighted square-root reading is a different estimator with
its own literature. Kalman rules refuse weighted ensembles and tell the caller to
resample.

**An operator for $FF^\top$ with a structured $F$.** `cov(a)` of a block with
only a factor row materializes the row. An operator that keeps it structured
is a linalg addition, made when a method needs it.

**Blocks with event shapes.** Blocks are vectors; a field on a grid is a
flattened block.

**Non-Gaussian anything.** No mixtures, no transformations of variables, no
likelihoods other than additive Gaussian noise, no MCMC kernels, no SMC
driver. Weighted ensembles and their three functions are what an SMC loop
written elsewhere needs from this layer.

## References

- Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
  with the ensemble transform Kalman filter. Part I: Theoretical aspects.
  *Monthly Weather Review*, 129(3), 420–436.
- Gordon, N. J., Salmond, D. J. & Smith, A. F. M. (1993). Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F
  (Radar and Signal Processing)*, 140(2), 107–113.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Journel, A. G. & Huijbregts, C. J. (1978). *Mining Geostatistics*.
  Academic Press.
- Kitagawa, G. (1996). Monte Carlo filter and smoother for non-Gaussian
  nonlinear state space models. *Journal of Computational and Graphical
  Statistics*, 5(1), 1–25.
- Kong, A., Liu, J. S. & Wong, W. H. (1994). Sequential imputations and
  Bayesian missing data problems. *Journal of the American Statistical
  Association*, 89(425), 278–288.
- Pham, D. T. (2001). Stochastic methods for sequential data assimilation in
  strongly nonlinear systems. *Monthly Weather Review*, 129(5), 1194–1207.
- Wang, X., Bishop, C. H. & Julier, S. J. (2004). Which is better, an
  ensemble of positive–negative pairs or a centered spherical simplex
  ensemble? *Monthly Weather Review*, 132(7), 1590–1605.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
  M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
  Machine Learning Research*, 22(105), 1–47.
