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

Let $\Theta'$ and $G'$ be mean-centred parameter and prediction anomalies, $J$
the ensemble size, and define the **whitened anomaly matrix**

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
  $(J-1)I + G'^\top R^{-1} G'$ and factorizes it, squaring the condition number
  of $S$ — which is exactly the quantity that degrades as the ensemble
  collapses in late tempering steps.
- **The multiplier is bounded unconditionally.** $\sigma/(1+\sigma^2) \le 1/2$
  for all $\sigma \ge 0$, so the gain cannot blow up however ill-conditioned
  the ensemble becomes, and no regularization parameter needs tuning.
- **It is one object away from the deterministic variant.** The square-root
  transform is $T = V(I+\Sigma^2)^{-1/2}V^\top$, reusing the same
  decomposition.
- **The update stays in the ensemble span**, $\theta^{(j)} \mapsto \theta^{(j)}
  + \Theta' w^{(j)}$ with $w^{(j)} \in \mathbb{R}^J$, so no matrix of parameter
  or observation dimension is ever formed.

Cost is $O(NJ)$ to whiten, $O(NJ^2)$ for the thin SVD and $O(PJ)$ to apply
$\Theta'$ — linear in both the observation dimension $N$ and parameter
dimension $P$, cubic only in the ensemble size.

### Choosing a variant

| regime | method | cost |
| --- | --- | --- |
| $N \le J$ | dense factorization of the predictive covariance | $O(N^3)$ |
| $J < N$, noise whitenable | whitened SVD (the default) | $O(NJ^2)$ |
| $N$ very large, noise block diagonal | same, whitening per block | $O(NJ^2)$ |
| localized | per-block whitened SVD on local data | $O(n N_{\text{loc}} J^2)$ |
| exact joint, small | direct factorization | $O(N^3)$ |
| exact joint, large | iterative solve | iterations $\times$ `matvec` |

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

### Kronecker products

$A \otimes B$ needs no factorization of its own. Every operation reduces to the
same operation on each factor: `matvec` by reshaping the operand's trailing
axis to $(n_A, n_B)$ and applying each factor along its own axis, `solve`
because $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$, and `factor` because
$\mathrm{factor}(A \otimes B) = \mathrm{factor}(A) \otimes \mathrm{factor}(B)$.
The last of those is why a rectangular Kronecker class is needed as well as a
square one: the factors' square roots need not be square, so the operand is
reshaped to $(k_A, k_B)$ while the result is reshaped to $(n_A, n_B)$, with a
mixed intermediate of shape $(k_A, n_B)$. An implementation that reshapes
operand and result alike — all the square case requires — cannot express the
rectangular one.

Three results, each of which produces wrong numbers rather than an error.

**Orientation is silent.** The convention is that the first factor is the slow
index, matching `numpy.kron`. Reversing it gives $B \otimes A$, which is a
different matrix but is positive definite whenever $A \otimes B$ is, and has
the same shape whenever the factors are the same size. So the reversed
implementation returns a valid covariance with the wrong meaning. Checked: for
two $3 \times 3$ PSD factors, the reversed product has smallest eigenvalue
$9.40$ — comfortably definite — while differing from the correct product by
$17.1$ in the largest entry.

**The log-determinant pairs each factor with the other factor's size.**

$$\log\det(A \otimes B) = n_B \log\det A + n_A \log\det B.$$

Swapping the two coefficients is the natural slip and it is invisible to any
test whose factors are the same size, since the two expressions then coincide
identically. With $n_A = 3$ and $n_B = 5$ the correct value is $57.5627$ and
the swapped pairing gives $70.5677$.

**Consistency is not correctness.** The conformance suite compares `matvec`
against `to_dense`, so an implementation that reverses the factors in *both*
agrees with itself and passes. Verified by mutation: reversing the orientation
throughout `Kron` — `matvec`, `solve`, `whiten`, `factor`, `cholesky`, `diag`
and `to_dense` together — leaves the whole conformance suite passing for
equal-size factors. Only comparison against an external `numpy.kron` reference
detects it. So the orientation of `matvec` *and* of `to_dense` must each be
pinned to `numpy.kron` directly; pinning them to each other constrains nothing.

The Kronecker product of two lower-triangular matrices is lower triangular and
equals the Cholesky factor of the product exactly, so a triangular square root
does exist. It is not exposed as one: `cholesky()` returns the same rectangular
class as `factor()`, which carries no `solve`, and `whiten` is implemented on
the factors instead. That is the second operator in this layer to hit the
mismatch between what `cholesky()` promises and what whitening needs.

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
