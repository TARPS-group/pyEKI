# The joint factor

`pyeki.gauss` represents a joint Gaussian by a single factor of its
covariance, cut into two row blocks. This page derives what follows from that
choice: how conditioning becomes one matrix multiplication, why a set of
samples and a factor of width $J$ are the same object, and where each
operation therefore belongs.

{doc}`gaussian-contract` states the resulting rules normatively and is what an
implementation must satisfy; this page is the reasoning behind them, and
{doc}`design` records the layer-level decisions above it. Every measurement
quoted below was taken in float64 against a dense reference; they are
collected in [Measurements](#measurements).

## Notation

| symbol | meaning |
| ------ | ------- |
| $u$, $v$ | the two blocks, of dimensions $P$ and $N$ |
| $y$ | the observation, a vector of length $N$ |
| $R$, $W$ | the observation-noise covariance, and a whitener of it: $WRW^\top = I_N$ |
| $J$ | the number of samples |
| $k$ | the latent width of a joint factor |
| $G$ | a linear map, $N \times P$ |
| $\mathbf{1}_m$, $I_m$ | the all-ones vector and the identity, in $m$ dimensions |

Samples are rows: the sample matrices are $\mathsf{U} \in \mathbb{R}^{J
\times P}$ and $\mathsf{V} \in \mathbb{R}^{J \times N}$, row-aligned, with
sample means $\bar u$ and $\bar v$. The **anomaly matrices** are the
deviations from those means,

$$
A_u = \mathsf{U} - \mathbf{1}_J\bar u^\top, \qquad
A_v = \mathsf{V} - \mathbf{1}_J\bar v^\top ,
$$

whose rows sum to zero: $\mathbf{1}_J^\top A_u = 0$. Empirical covariances
use the divisor $J-1$, so $\widehat{C}_{uv} = A_u^\top A_v/(J-1)$.

Factors are columns, as the operator layer's `factor()` returns them: a
factor of an $n$-dimensional covariance is an $(n, k)$ operator. The two
conventions meet whenever samples become a factor, and the conversion
carries both a transpose and the divisor's square root.

## The representation

Write the pair as one vector $x = (u, v) \in \mathbb{R}^{P+N}$, distributed
$\mathcal{N}(\mu, C)$ with $\mu = (\bar u, \bar v)$ and $C$ the block
covariance. A **joint factor** is any $F \in \mathbb{R}^{(P+N)\times k}$ with
$C = FF^\top$, split along the block structure of $x$:

$$
F = \begin{pmatrix} F_u \\ F_v \end{pmatrix},
\qquad F_u \in \mathbb{R}^{P\times k}, \quad F_v \in \mathbb{R}^{N\times k},
$$

which gives all three covariance blocks at once and with nothing left to
check:

$$
C_{uu} = F_uF_u^\top, \qquad
C_{uv} = F_uF_v^\top, \qquad
C_{vv} = F_vF_v^\top .
$$

This is the same notion of a factor the operator layer already has, read in
two row blocks. In generative form it says that **one shared latent vector
drives both blocks**:

$$
\begin{pmatrix} u \\ v \end{pmatrix}
= \begin{pmatrix} \bar u \\ \bar v \end{pmatrix}
+ \begin{pmatrix} F_u \\ F_v \end{pmatrix}\xi,
\qquad \xi \sim \mathcal{N}(0, I_k).
$$

That shared $\xi$ is the whole content of the representation. Two factors
chosen independently — one of $C_{uu}$, one of $C_{vv}$ — carry no
information about $C_{uv}$ at all; the pair has to come from one
factorization.

The two that matter here:

**From samples**, with $k = J$: the factor is the scaled anomalies,

$$
F = \frac{1}{\sqrt{J-1}}\begin{pmatrix} A_u^\top \\ A_v^\top \end{pmatrix},
$$

which reproduces the three empirical covariances exactly and forms none of
them. This is {meth}`~pyeki.gauss.GaussianJoint.from_samples`.

**From a linear map**, with $k$ the width of the prior's own factor: for $u
\sim \mathcal{N}(m_0, C_0)$ with $C_0 = LL^\top$ and $v = Gu$,

$$
F = \begin{pmatrix} L \\ GL \end{pmatrix},
\qquad \bar u = m_0, \quad \bar v = Gm_0 .
$$

This is {meth}`~pyeki.gauss.GaussianJoint.from_linear_map`, and conditioning
it gives the closed-form linear-Gaussian posterior.

## Conditioning

Let $S = (WF_v)^\top \in \mathbb{R}^{k\times N}$ be the whitened factor of
the observed block, with thin SVD $S = U\Sigma V^\top$.

**The gain.** Since $R = W^{-1}W^{-\top}$,

$$
C_{vv} + R = W^{-1}\bigl(S^\top S + I_N\bigr)W^{-\top},
$$

so $(C_{vv}+R)^{-1} = W^\top(S^\top S + I_N)^{-1}W$, and with $F_v^\top
W^\top = S$ the Kalman gain $K = C_{uv}(C_{vv}+R)^{-1}$ applies as

$$
K r \;=\; F_u\, S\bigl(S^\top S + I_N\bigr)^{-1}(Wr)
\;=\; F_u\,\texttt{gain\_weights}(S,\, Wr).
$$

The update to $u$ is a combination of $F_u$'s own columns, so no matrix of
dimension $P$ or $N$ appears.

**The posterior.** The mean follows immediately, $m_{\text{post}} = \bar u +
K(y - \bar v)$. For the covariance, apply the push-through identity
$S(I_N + S^\top S)^{-1}S^\top = I_k - (I_k + SS^\top)^{-1}$:

$$
C_{\text{post}}
= C_{uu} - KC_{vu}
= F_u\bigl(I_k - S(I_N + S^\top S)^{-1}S^\top\bigr)F_u^\top
= F_u\bigl(I_k + SS^\top\bigr)^{-1}F_u^\top ,
$$

so with $T = \texttt{sqrt\_transform}(S) = (I_k + SS^\top)^{-1/2}$,

$$
C_{\text{post}} = \bigl(F_uT\bigr)\bigl(F_uT\bigr)^\top .
$$

**Conditioning is a right multiplication of the factor by $T$.** The whole
of it is the map

$$
\bigl(\bar u,\ \bar v,\ F_u,\ F_v\bigr)
\;\longmapsto\;
\bigl(\bar u + F_u w,\ \ F_u T\bigr),
\qquad w = \texttt{gain\_weights}\bigl(S, W(y - \bar v)\bigr),
$$

one SVD supplying both halves. Nothing is inverted, neither $S^\top S$ nor
$SS^\top$ is formed, and the result is exact at every latent width — including
$k < P$, where the prior is singular and the precision form $(C_0^{-1} +
G^\top R^{-1}G)^{-1}$ does not exist at all.

## Centred factors and sample sets

$T$ multiplies $F_u$ on the *right*: it acts on the latent index. Whether
that constitutes an update of a set of samples is exactly the question of
whether the latent index *is* a sample index. Two facts settle it.

Call a factor **centred** when $F\mathbf{1}_k = 0$. This is a property of the
representation, not of the distribution it represents.

:::{admonition} A centred factor is a sample set
:class: important

For $F \in \mathbb{R}^{n\times k}$ and $m \in \mathbb{R}^n$, form the $k$
rows

$$
\mathsf{X} = \mathbf{1}_k m^\top + \sqrt{k-1}\,F^\top
\in \mathbb{R}^{k \times n}.
$$

Then $\mathsf{X}$ has sample mean $m$ **if and only if** $F\mathbf{1}_k =
0$, and in that case its empirical covariance with divisor $k-1$ is exactly
$FF^\top$.

*Why.* The sample mean is $m + \sqrt{k-1}F\mathbf{1}_k/k$, which is $m$
exactly when $F\mathbf{1}_k = 0$. Under that condition $\mathsf{X}$'s
anomaly matrix is $\sqrt{k-1}F^\top$, whose empirical covariance is
$FF^\top$.
:::

So a mean together with a centred factor of width $k$ *is* a set of $k$
samples whose empirical moments equal the Gaussian's — and the map runs both
ways, since $\mathsf{U} \mapsto (\bar u,\ A_u^\top/\sqrt{J-1})$ sends a
$J$-sample set to a centred factor of width $J$. A factor read off samples is
centred precisely because anomalies sum to zero.

The second fact is that conditioning preserves this. If $F_v\mathbf{1}_k =
0$ then $S^\top\mathbf{1}_k = WF_v\mathbf{1}_k = 0$, hence
$SS^\top\mathbf{1}_k = 0$, hence $(I_k + SS^\top)\mathbf{1}_k = \mathbf{1}_k$
and so $T\mathbf{1}_k = \mathbf{1}_k$. Therefore $F_uT\mathbf{1}_k =
F_u\mathbf{1}_k = 0$: **a centred factor conditions to a centred factor.**

Together they give the square-root update. Conditioning maps sample sets to
sample sets, and

$$
\texttt{transform\_update}(y, R)_j
= m_{\text{post}} + \sqrt{J-1}\,\bigl(F_uT\bigr)_{\cdot j}
= m_{\text{post}} + \bigl(TA_u\bigr)_j .
$$

Column $j$ of the conditioned factor is updated sample $j$.

Two consequences for the API.

**The projection from samples to a joint loses nothing.** Running the
equivalence backwards, $u_j = \bar u + \sqrt{J-1}(F_u)_{\cdot j}$, so
{meth}`~pyeki.gauss.EmpiricalJoint.to_gaussian_joint` is a bijection onto
(mean, centred width-$J$ factor) pairs. Neither update needs its samples
supplied a second time: everything they use is in the joint. What the
projection drops is the *reading* of the latent index as a sample index.

**A sample-set argument would be redundant or wrong.** Suppose
`transform_update` took one. $T \in \mathbb{R}^{k\times k}$, so the call only
conforms at $k = J$; and the result's empirical covariance would be
$A_u'^\top(I_J + SS^\top)^{-1}A_u'/(J-1)$, which equals $C_{\text{post}}$ only
when $A_u'^\top/\sqrt{J-1} = F_u$ — that is, only when the sample set is the
one the joint was fitted to. Worse, the *mean* survives regardless, since
$\mathbf{1}^\top A_u' = 0$ and $T\mathbf{1} = \mathbf{1}$ hold for any
centred set. On an unrelated sample set of the same shape the mean is correct
to round-off and the covariance is wrong by a large fraction of its own scale
— 62% on the conformance fixture, though the exact figure depends on the
draw, so the test asserts only that it exceeds a tenth. Finite, with nothing
raised.

## The pathwise map

Matheron's rule is the affine map

$$
\Phi(u, v, \eta) = u + K\bigl(y - v - \eta\bigr),
$$

which depends on the joint only through $K$ — only through its moments. If
$(u,v)$ is distributed as the joint and $\eta \sim \mathcal{N}(0,R)$
independently, then $\Phi$ is distributed as the posterior: its mean is
$m_{\text{post}}$, and

$$
\operatorname{Cov}\Phi
= C_{uu} - KC_{vu} - C_{uv}K^\top + K\bigl(C_{vv}+R\bigr)K^\top
= C_{uu} - KC_{vu},
$$

because $K(C_{vv}+R) = C_{uv}$ by the definition of $K$.

Two properties distinguish it from the square-root update, and both put it on
{class}`~pyeki.gauss.GaussianJoint` with its realizations as arguments. It is
a **per-realization** map: each triple is transported independently, so a
batch of them is one call rather than a loop. And it is correct for **any**
realization of the joint, not only for realizations that produced the
moments.

The noise enters in whitened coordinates. With $\varepsilon = W\eta \sim
\mathcal{N}(0, I_N)$,

$$
\Phi = u + F_u\,
\texttt{gain\_weights}\bigl(S,\, W(y - v) - \varepsilon\bigr),
$$

and {meth}`~pyeki.gauss.GaussianJoint.pathwise` takes $\varepsilon$ rather
than $\eta$. That is deliberate: $WL$ has orthonormal rows but is not the
identity, so pushing one perturbation through both `whiten` and `factor()` in
the same update corrupts the joint law while every marginal statistic still
looks right. Pinning the argument to one representation, by name and by
documented obligation, is what closes that off.

{meth}`~pyeki.gauss.EmpiricalJoint.pathwise_update` is this map on the
samples the joint was fitted to, and takes a cheaper route to them. Because
those samples *are* the factor, $v_j = \bar v + \sqrt{J-1}(F_v)_{\cdot j}$
and so

$$
W(y - v_j) = W(y - \bar v) - \sqrt{J-1}\,S_{j\cdot} ,
$$

which costs nothing beyond the whitened factor already in hand. That update
spends $J+1$ applications of $W$; the general map, whose $v$ is arbitrary
data, spends $k$ plus one per realization.

## Why not covariance blocks

The obvious alternative is to hold $(C_{uu}, C_{uv}, C_{vv})$ as three
operators and let callers supply whatever structured types they have. It
fails three ways, independently.

**Recovering a shared latent requires squaring.** Given $C_{vv}$ and a factor
$F_v$ of it, the coherent $F_u$ solves $F_uF_v^\top = C_{uv}$, that is

$$
F_u = C_{uv}\bigl(F_v^\top\bigr)^{+} = C_{uv}F_v\bigl(F_v^\top F_v\bigr)^{-1}.
$$

This fails outright when $k > N$: the Gram is then singular by construction,
the solve is ill-posed, and there is no coherent $F_u$ to recover. Measured on
a $(5, 6)$ factor, the recovered $F_u$ has relative error $1.4\times10^{9}$ —
not an approximation but a different matrix. When $k \le N$ the recovery is
well-posed but forms $F_v^\top F_v$, squaring the condition number, which is
the defect that rules out the Woodbury route ({doc}`design`): at
$\kappa(F_v) = 1.4\times10^{8}$ the Gram reaches $2\times10^{16}$ and the
recovered $F_u$ keeps 8 of its 16 digits.

Squaring is the lesser half of the objection, though, and worth being precise
about: an SVD-based pseudo-inverse avoids forming the Gram and does no better
here ($4.7\times10^{-8}$ on the same fixture), because the ill-conditioning
is in $F_v$ itself. The decisive objections are the two below.

**It needs a matrix of both dimensions.** $C_{uv}$ is $P \times N$. No code
path in the package forms one; the point of the whitened-SVD kernel is that
nothing of either block's dimension is materialized.

**PSD-ness cannot be typed.** A valid joint covariance requires
$\operatorname{col}(C_{vu}) \subseteq \operatorname{col}(C_{vv})$. Three
independently supplied operators do not satisfy that in general, and no
operator type can assert that they do: composition never proves PSD-ness.
Supply blocks that violate it and the answer is finite, plausible and not a
posterior.

A joint factor has none of these problems. Coherence is structural, nothing
is squared, no matrix of both dimensions appears, and $C = FF^\top$ is PSD by
construction.

What the factor form cannot check is that its two row blocks came from *one*
factorization. Any pair of matching width defines some joint Gaussian, so
factorizing $C_{uu}$ and $C_{vv}$ separately yields the intended marginals
and a wrong cross-block — and conditioning then answers correctly for a
different joint. The shared-width check catches this whenever $P \ne N$, two
square factorizations having widths $P$ and $N$; at $P = N$ it cannot, which
is why {meth}`~pyeki.gauss.GaussianJoint.from_factors` is documented as the
escape hatch and the two arithmetic constructors are the recommended routes.

## Where each operation lives

The dividing line is what an operation returns.

{class}`~pyeki.gauss.GaussianJoint` owns the mathematics: operations
determined by the moments, returning a distribution or transporting
realizations handed to them. {class}`~pyeki.gauss.EmpiricalJoint` owns the
samples: operations whose result is a set of samples aligned with the ones it
holds.

| operation | lives on | because |
| --------- | -------- | ------- |
| `condition` | `GaussianJoint` | moment-determined, returns a distribution |
| `pathwise` | `GaussianJoint` | per-realization, correct for realizations the joint never saw |
| `transform_update` | `EmpiricalJoint` | valid only for a centred factor |
| `pathwise_update` | `EmpiricalJoint` | returns samples aligned with the ones held |

Two placements deserve comment.

**`condition` is not on `EmpiricalJoint`.** Conditioning a set of samples
means conditioning a Gaussian fitted to their moments. That fit is a
modelling step, and a method named `condition` on a class named for the
empirical distribution reads like conditioning the empirical measure, which
is not what it would do. Writing
`joint.to_gaussian_joint().condition(y, noise_cov)` costs one call and puts
the step in the source text.

**`transform_update` is not on `GaussianJoint`.** It could be: the reading is
a pure function of the conditioned factor. But its hypothesis —
$F\mathbf{1}_k = 0$ — is a *value* precondition. On a general joint it could
only be checked in debug mode, so a `from_linear_map` joint would silently
return samples with the right covariance and a mean shifted by
$\sqrt{k-1}F\mathbf{1}_k/k$. Held on the class that owns samples,
centredness is structural: the only constructor reachable from there centres,
and no check is needed.

## What it costs

Conditioning whitens $k+1$ vectors — the factor's $k$ columns and the mean
residual — from one `whiten_mat` call on the stacked columns $[F_v \mid y -
\bar v]$. At $k = J$ that is the $J+1$ a sample update spends.

Holding a factor rather than samples settles the centre-before-whiten
question structurally. The two orders agree in exact arithmetic but not in
stability: centring already-whitened vectors makes the cancellation ratio
$\lVert W\bar v\rVert / \lVert WF_v\rVert$ in place of $\lVert \bar v\rVert /
\lVert F_v\rVert$, so the error grows with $\kappa(W) = \sqrt{\kappa(R)}$
whenever $\bar v$ is aligned with a precise direction of the noise. Measured
against an exact rational reference at $\kappa(R) = 10^{10}$, with the
observed block's mean of magnitude $10^{10}$ along $R$'s most precise
direction: whitening first gives a posterior-mean relative error of
$9.0\times10^{-6}$ where centring first gives $1.0\times10^{-12}$, six orders
apart. Because the factor is centred at construction, there is no ordering
left to get wrong inside a conditioning call — only the mean residual, which
is differenced before whitening.

The thin SVD is $O(kN\min(k,N))$, forming weights $O((N+k)\rho)$ per
residual, applying $F_u$ is $O(Pk)$ per weight vector for a dense factor and
less for a structured one. A full sample update is $O(NJ^2 + PJ^2)$ for $J
\le N$: linear in both dimensions for structured whiteners, cubic only in the
latent width.

Exact conditioning through `from_linear_map` adds $O(NPk)$ to form $GL$, less
for a structured $G$. At a full-rank dense prior ($k = P$) that is the same
order as a dense solve, with the advantages of inverting nothing and
remaining valid for a singular $C_0$; the real gain is at $k \ll P$, and in
`diag()` and `sample()` on the low-rank posterior, both $O(Pk)$.

One limitation. {class}`~pyeki.linalg.PSDLowRank` holds a
dense array, so the posterior covariance factor $F_uT$ is materialized at
$(P, k)$ even when $F_u$ is structured. For EKI that costs nothing — a
$(J, P)$ ensemble is already held and $k \le J$ — but a caller with $P$ too
large to hold $(P, k)$ would need a low-rank operator over an operator
factor. That is the trigger if it is ever wanted.

## Measurements

Every claim above, checked in float64 against dense references. For scale,
$\varepsilon = 2.22\times10^{-16}$.

| claim | result |
| ----- | ------ |
| factor conditioning against dense block conditioning, $k = P$ | mean $3.3\times10^{-16}$, covariance $2.7\times10^{-15}$ |
| the same, against the precision form | mean $1.6\times10^{-15}$, covariance $5.3\times10^{-15}$ |
| the same, at a rank-deficient prior factor $k < P$ | mean $4.4\times10^{-16}$, covariance $1.3\times10^{-15}$ |
| fully operator-form path, structured prior and operator map | mean $7.8\times10^{-16}$, covariance $1.6\times10^{-15}$ |
| samples recovered from mean and factor | $4.4\times10^{-16}$ |
| $F\mathbf{1} = 0$ for a factor read off samples | $3.9\times10^{-16}$ |
| $T\mathbf{1} = \mathbf{1}$, and $F_uT$ still centred | $1.1\times10^{-16}$, $4.7\times10^{-16}$ |
| `transform_update` against `condition` plus the reading | $4.4\times10^{-16}$ |
| Matheron on exact-moment realizations, against the closed-form posterior | mean $2.2\times10^{-16}$, covariance $3.3\times10^{-16}$ |
| a sample-set argument: mean survives, covariance does not | mean $5.6\times10^{-17}$; covariance error $1.17$ on a scale of $1.88$ |
| centring before whitening, $\kappa(R) = 10^{10}$ | $1.0\times10^{-12}$, against $9.0\times10^{-6}$ for the reverted grouping |
| block recovery of $F_u$, wide factor ($k > N$) | ill-posed; recovered $F_u$ off by $1.4\times10^{9}$ |
| block recovery of $F_u$, Gram route at $\kappa(F_v) = 1.4\times10^{8}$ | $\kappa(F_v^\top F_v) = 2\times10^{16}$; $F_u$ to $1.8\times10^{-8}$, against $4.7\times10^{-8}$ for an SVD pseudo-inverse |

All but the last two are exercised by `tests/test_gauss.py` — most as
conformance obligations of {doc}`gaussian-contract`, the sample-set row as a
targeted regression test. The two block-recovery rows describe the rejected
design, so they are recorded here rather than tested.
