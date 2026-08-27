# Design notes

Why pyEKI's interfaces look the way they do. This page records decisions that
are not obvious from the code and are easy to undo by accident. It is aimed at
contributors; users need only the {doc}`user-guide/quickstart`.

## Three covariances, three sets of requirements

EKI involves three covariance objects that play different roles and need
different operations. Conflating them into one interface is what pushes these
codebases toward dense linear algebra.

| object | where it appears | operations needed | structure to exploit |
| --- | --- | --- | --- |
| prior over parameters | drawing the initial ensemble; predicting at new points; hyperparameter inference | `factor` always; `solve` and `logdet` only for the latter two | Kronecker, spectral, sparse |
| observation noise | **every** update | whitening, and nothing else | block diagonal; temporal correlation |
| ensemble covariance | every update | never materialized | low rank by construction |

Two consequences shape the library.

**The prior is off the hot path.** In tempered EKI the prior covariance is used
exactly once, to draw the initial ensemble; everything afterwards runs on
empirical moments. A prior with no cheap inverse at all is therefore perfectly
usable. Cheap `solve` and `logdet` become load-bearing only when estimating
hyperparameters, running variants whose drift term involves the prior
precision, or predicting at unobserved points.

**The noise covariance needs only whitening.** Writing the perturbed-observation
residual with $R$ the (possibly tempered) noise covariance,

$$
r^{(j)} = y - y^{(j)}, \qquad y^{(j)} = g^{(j)} + R^{1/2}\varepsilon_j
\;\Longrightarrow\;
R^{-1/2} r^{(j)} = R^{-1/2}\bigl(y - g^{(j)}\bigr) - \varepsilon_j .
$$

The $R^{1/2}$ cancels. Combined with the gain below, $R$ is used only through
$R^{-1/2}\cdot$, which is why `whiten` is a first-class method rather than
something callers assemble from `cholesky`.

## The conditioning kernel

Let $\Theta'$ and $G'$ be mean-centered parameter and prediction anomalies, $J$
the ensemble size, and define the **whitened anomaly matrix**. (This page
writes members as *columns*, $G' \in \mathbb{R}^{N \times J}$; the
normative {doc}`gaussian-contract` writes them as rows — its $U$ and $V$
are this page's $V$ and $U$.)

$$
S := \frac{1}{\sqrt{J-1}}\,R^{-1/2}G', \qquad S = U\Sigma V^\top \ \text{(thin SVD)}.
$$

Then the Kalman gain applies as

$$
K r = \frac{1}{\sqrt{J-1}}\;\Theta'\,V \operatorname{diag}\!\Bigl(\frac{\sigma_i}{1+\sigma_i^{2}}\Bigr) U^\top \bigl(R^{-1/2}r\bigr).
$$

This is the form pyEKI will implement, in preference to the algebraically
equivalent Woodbury identity applied to the normal equations. Four reasons:

- **Nothing is squared.** The competing route forms
  $(J-1)I + G'^\top R^{-1} G'$ and factorizes it. *Forming* that Gram
  rounds away every singular value of $S$ below
  $\sqrt{\varepsilon}\,\sigma_{\max}$ — and those are the ones carrying
  the largest gain multipliers. The loss is governed by $\sigma_{\max}$
  (small noise, a large whitener), not by $\kappa(S)$: ensemble collapse
  alone, which drives $\sigma_{\min} \to 0$, costs neither route
  accuracy.
- **The multiplier is bounded unconditionally.** $\sigma/(1+\sigma^2) \le 1/2$
  for all $\sigma \ge 0$, so the gain cannot blow up however ill-conditioned
  the ensemble becomes, and no regularization parameter needs tuning.
- **It is one object away from the deterministic variant.** The square-root
  transform is $T = I_J + V\bigl((I+\Sigma^2)^{-1/2} - I\bigr)V^\top$,
  reusing the same decomposition. The identity completion matters: the
  naive $V(I+\Sigma^2)^{-1/2}V^\top$ omits the identity on the orthogonal
  complement and is wrong whenever the thin SVD's rank is below $J$ — see
  the {doc}`gaussian-contract` for the normative form.
- **The update stays in the ensemble span**, $\theta^{(j)} \mapsto \theta^{(j)}
  + \Theta' w^{(j)}$ with $w^{(j)} \in \mathbb{R}^J$, so no matrix of parameter
  or observation dimension is ever formed.

Cost is $O(NJ)$ to whiten (for a whitener applying in $O(N)$ per vector;
a dense whitener costs $O(N^2)$ each), $O(NJ^2)$ for the thin SVD and $O(PJ)$ to apply
$\Theta'$ — linear in both the observation dimension $N$ and parameter
dimension $P$ for a structured whitener, cubic only in the ensemble size;
a dense whitener adds its own $O(N^2J)$.

### The algorithm space

A survey, not a dispatch plan: `pyeki.gauss` routes everything through the
whitened SVD and selects nothing at runtime ({doc}`gaussian-contract`).

| regime | method | cost |
| --- | --- | --- |
| $J < N$, noise whitenable | whitened SVD — what pyEKI implements | $O(NJ^2)$ |
| $N$ very large, noise block diagonal | same, whitening per block | $O(NJ^2)$ |
| localized | per-block whitened SVD on local data | $O(n N_{\text{loc}} J^2)$ |
| $N \le J$ | dense factorization of the predictive covariance — deferred as a possible internal optimization behind the same signatures | $O(N^3)$ |

## Localization

Two schemes exist and they are **not** interchangeable here.

**Covariance localization** tapers the sample cross-covariance elementwise. The
Hadamard product destroys the rank-$J$ factorization, so the predictive
covariance loses its low-rank structure and must be inverted at $O(N^3)$.

**Domain localization** instead restricts each parameter block to nearby
observations and inflates their noise by the reciprocal of the taper. The
subspace algebra is preserved exactly, the per-block analyses are independent,
and each yields a weight vector in $\mathbb{R}^J$ applied to that block's own
anomalies.

pyEKI will implement domain localization, because it preserves the conditioning
kernel above and parallelizes cleanly.

Two hazards worth designing against. Parameters with no location — globally
shared ones — must be exempt from tapering, or the pooling that makes them
global is silently destroyed. And local neighbourhoods must be fixed-size with
a validity mask, since variable-size domains cannot be vectorized.

## Structured covariance results

Results that inform which operators are worth implementing. Each was verified
numerically against a dense reference.

### Kronecker plus nugget

$C = K \otimes B + I_n \otimes C^{l}$ admits a simultaneous diagonalization,
giving `solve`, `logdet` and `factor` after one $O(n^3 + D^3)$ factorization.
Two cautions:

- The determinant needs a term that is easy to omit. With $W$ from the
  generalized eigenproblem, $\log\det C = n\log\det C^{l} + \sum_{ij}
  \log(\lambda_i\mu_j + 1)$; the congruence contributes because $Q \otimes W$
  is not orthogonal.
- It requires $C^{l} \succ 0$ on the whole block. A rank-deficient $B$ is fine,
  but a singular nugget makes the pencil singular and no simultaneous
  diagonalization exists.

JAX has no generalized `eigh`, so compute it by whitening: Cholesky
$C^{l} = L_cL_c^\top$, standard `eigh` of $L_c^{-1}BL_c^{-\top}$, then
$W = L_cZ$.

### Spectral (circulant) structure

For stationary kernels on a regular lattice, circulant embedding gives a shared
Fourier basis, and a sum $\sum_q K_q \otimes B_q$ becomes block diagonal with
$n$ blocks of size $D \times D$.

:::{warning}
This gives exact `matvec` and exact **sampling** on the original grid, but
**not** `solve` or `logdet`. The grid covariance is a principal *submatrix* of
the circulant, and restriction does not commute with inversion. In a
one-dimensional check, `matvec` was exact to $2\times10^{-16}$ and restricted
draws reproduced the covariance to Monte Carlo error, while the spectral
`solve` was 8% off and the log-determinant simply wrong.

Either use it for `matvec` and sampling only — which is all the prior needs —
or adopt the periodic covariance as the model, in which case all four
operations are exact by construction.
:::

Two further practicalities: the padding factor needed for a positive
semi-definite embedding is set by the smoothest, longest-lengthscale component,
and costs must be quoted in the padded size. Squared-exponential kernels do not
embed at any practical padding; use Matérn.

### Exponential correlation in one dimension

For the correlation matrix $\rho^{|i-j|}$ it is the **whitener** $L^{-1}$ that
is bidiagonal, not the factor $L$: the Cholesky of the covariance is a dense
lower triangle. So whitening is a vectorized linear-time operation, while
*sampling* is a first-order recurrence — also linear, but sequential, and the
only such operation in this layer.

A single scalar $\rho$ is the wrong parameterization whenever observation times
are irregular, which missing data guarantees. The Markov property survives
subsampling, so the precision stays tridiagonal when built from per-interval
coefficients; only the interface must change. Higher-order Matérn correlations
are not autoregressive of order one and need a state-space representation.

### Sparse representations are duals

A tapered covariance gives cheap `matvec` because it is a sparse *covariance*.
A Vecchia-style factorization gives cheap `solve` and `logdet` because it is a
sparse *precision*. Neither gives the other cheaply, and a tapered matrix is
only positive semi-definite if the taper is itself a valid positive-definite
function — which is dimension-dependent, so a taper must come from a known
family rather than an arbitrary sparsity mask.

## Implementation constraints

Verified against JAX 0.8.3. Each produces wrong numbers or silent slowness
rather than an error.

**Pytree registration is unforgiving.** Undeclared non-array fields become
pytree children and arrive as tracers under `jit`. A NumPy array used as static
metadata works once and then raises, permanently poisoning that function's
compilation cache. Declare data and metadata fields explicitly and keep static
metadata to tuples of integers.

**Lazy factorization caching does not work.** A factor cached inside a traced
function is written to a temporary copy of the operator and discarded, so the
operator re-factorizes on every call — around ten times slower in a
representative case. Factorizations belong in the constructor.

**Operators compare by identity.** Dataclass equality would compare arrays
elementwise and raise on the ambiguous truth value, and hashing fails on array
fields. Operators are always traced arguments, never static ones.

**Numerical failure is silent.** A Cholesky of an indefinite matrix returns
`nan` without raising, and the log-determinant propagates it. Guard with
runtime checks in a debug mode rather than assuming an exception.

**Capability dispatch must be static.** An error raised for an unsupported
operation fires at trace time even inside the untaken branch of a conditional,
because both branches are traced.

## Why not an existing library

pyEKI implements its own operator layer rather than depending on a
general-purpose one. The three structures EKI most needs are the three that
general libraries tend to lack: a sum of Kronecker products, a spectral
representation of that sum, and rectangular square roots for sampling. Existing
libraries also standardize on column-stacked operands rather than leading batch
axes, which would push that convention through every consumer.

The layer here is deliberately small and aimed at what EKI requires. If
stochastic trace or log-determinant estimation later becomes routine, borrowing
a specialist implementation for those calls is preferable to growing one here.

## EKI schedule criteria

Two derivations the EKI contract states as facts and uses, recorded here so
that page can stay a specification. Notation is that page's.

### Why the ESS is monotone in the increment

Write $E_\lambda$ for expectation under the tilted weights
$w_j(\lambda) \propto e^{-\lambda\Phi_j}$. Since
$\log \mathrm{ESS}(\delta) = 2\,\mathrm{lse}(-\delta\Phi) - \mathrm{lse}(-2\delta\Phi)$
and $\frac{d}{d\lambda}\log\sum_j e^{-\lambda\Phi_j} = -E_\lambda[\Phi]$,

$$
\frac{d}{d\delta}\log \mathrm{ESS}(\delta)
= 2\bigl(E_{2\delta}[\Phi] - E_{\delta}[\Phi]\bigr)
= -\,2\delta \int_{1}^{2} \operatorname{Var}_{s\delta}(\Phi)\,ds
\;\le\; 0 ,
$$

the second equality by integrating
$\frac{d}{d\lambda}E_\lambda[\Phi] = -\operatorname{Var}_\lambda(\Phi)$ from
$\lambda = \delta$ to $\lambda = 2\delta$ and substituting
$\lambda = s\delta$. The inequality is strict for $\delta > 0$ unless every
$\Phi_j$ is equal.

Two things follow that the contract relies on. The decay rate is a variance of
the misfits under the tilted weights, so the criterion is measuring ensemble
disagreement about the data rather than an arbitrary monotone quantity. And the
$-2\delta$ prefactor vanishes at $\delta = 0$: the function is flat at the left
endpoint, so bisection is the right root find and a derivative-based one is not.

One precondition, which pyEKI satisfies structurally: the argument assumes the
ensemble arrives equally weighted. It does, since the layer carries no
importance weights at all — but a variant that introduced them would have to
revisit this before reusing the bisection.

### Why the misfit criterion takes a max of its two bounds

Iglesias and Yang obtain the criterion by controlling the Jeffreys
(symmetrized Kullback–Leibler) divergence between consecutive tempered
measures, requiring it to stay below $\theta$. That divergence is exactly
$\delta\,(E_\beta[\Phi] - E_{\beta+\delta}[\Phi])$ — the $\log Z$ terms
cancel — and cannot be evaluated at step $t$, since it depends on the next
measure. It is therefore approximated in two regimes: dropping the unknown
non-negative term gives the upper bound $\delta\,\overline{\Phi}$, accurate
when the mean misfit falls substantially across the rung; and a first-order
expansion of $\overline{\Phi}$ in $\delta$ gives $\delta^2\sigma^2_\Phi$,
accurate when it barely moves. Neither is valid throughout, so the divergence is
approximated by the **smaller** of the two,

$$
D_{\mathrm{J}} \;\approx\; \min\bigl\{\, \delta\,\overline{\Phi},\ \
\delta^2 \sigma^2_\Phi \,\bigr\} \;\le\; \theta .
$$

Both expressions increase in $\delta$, so a `min` bounded by $\theta$ holds
exactly on the interval up to the **larger** of the two thresholds. The `max` in
the criterion is that `min` turned inside out.

The threshold $\theta = N/2$ is then fixed by a statistical discrepancy
principle rather than tuned. At the step's own noise level the whitened misfit
$\chi_j = 2\delta\Phi_j$ would be $\chi^2_N$ if the target were well
specified, so requiring its mean below $N$ and its variance below $2N$ gives
$\theta = N/2$ in both bounds — which is the sense in which the schedule has no
free parameter: the observation dimension supplies it.
