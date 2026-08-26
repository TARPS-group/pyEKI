# Handoff

Written 2026-08-24 and updated 2026-08-25, after the operator layer was
reworked against the normative contract. Read `CLAUDE.md` first for
conventions, then this for state and next steps.

## Where things stand

**Done.** `pyeki.linalg` is implemented to the specification in
`docs/linop-contract.md` — the normative reference for the layer's behaviour,
written and adversarially reviewed before this implementation. Three-level
hierarchy (`LinOp`/`SquareLinOp`/`PSDLinOp`) with template methods (public
methods gate and validate; authors implement `_`-prefixed hooks), transposes
(`rmatvec`, `T`), operator arithmetic (`@` composition, scalar `*`/`/`),
six elementary operators, nine composites with factory functions, a debug
mode for value
preconditions, and a 14-check conformance harness; the full test suite passes.
Documentation builds: landing page, installation, quickstart, operator
catalogue, a guide to writing an operator, the operator contract, design
notes, and an API reference.

**Not started.** `pyeki.gauss`, `pyeki.localize`, `pyeki.eki`. The design
background for all three is in `docs/design.md`; `pyeki.gauss` additionally
has its full normative contract in `docs/gaussian-contract.md`.

**Origin.** This package was extracted from a research repository where the
operator layer was first written. That repository keeps the domain-specific
work — forward models, priors, experiment configuration — and will depend on
pyEKI. Nothing domain-specific should come back across.

## Next steps, in order

### 1. `Kron` (start here)

The one operator whose absence blocks the rest. Two variants, and they are not
the same code:

- **Square** `Kron(A, B)` representing $A \otimes B$. Convention: the first
  factor's index is the *slow* one, so block $(i,j)$ of the result is
  $A_{ij}B$. Implement `matvec` by reshaping the trailing axis to `(n_A, n_B)`,
  applying `B` then `A`, and flattening back.
- **Rectangular**, needed because `factor(A ⊗ B) == factor(A) ⊗ factor(B)` and
  those factors need not be square. The square implementation does **not**
  generalize: it reshapes input and output to the same shape, whereas a
  rectangular Kronecker product needs input reshaped to `(k_A, k_B)` and output
  to `(n_A, n_B)`.

:::{warning}
A transposed Kronecker orientation is silent — it yields a valid PSD matrix
with the wrong meaning. Test `matvec` against `np.kron` directly, at leading
batch rank 0, 1 and 2. Do **not** rely on a `to_dense` comparison: `to_dense`
is built from `jnp.kron` of the children, a different code path, so it passes
even when `matvec` is wrong.
:::

Then `KronLMC` (a sum $\sum_q A_q \otimes B_q$), `KronPlusNugget` and
`LowRankPlus`, in that order. `docs/design.md` records the closed forms and
their preconditions, including a log-determinant term that is easy to omit.

### 2. `pyeki.gauss`

The conditioning layer. `docs/design.md` gives the kernel: the whitened-SVD
gain, which is the form to implement — not the algebraically equivalent
Woodbury identity on the normal equations, which squares a condition number
that is already the problem.

Its normative design is now `docs/gaussian-contract.md` — adversarially
reviewed and ready to implement. The shape differs from the suggestion this
section previously recorded: one closed `EnsembleJoint` class plus a
`Gaussian` marginal and two array-level conditioning primitives; no
operator-represented joint (the dense reference is hand-written in the
tests, deliberately); and the whitened-SVD kernel as the single algorithm.
The contract's `gauss-excluded` section records why each earlier suggestion
was dropped.

### 3. `pyeki.eki`

Tempering ladder, ensemble updates, inflation, driver loop.

:::{important}
The per-step observation noise is $\Sigma/\Delta\beta_t$, using the tempering
*increment*, not $\Sigma/\beta_t$. Per-step precisions must telescope to the
total. On a linear-Gaussian problem with a five-step ladder, the increment form
reproduces the one-shot posterior to $10^{-15}$ while the other is off by 0.12
in the mean and 0.25 in the covariance — and the error grows with ladder
length. Write the telescoping test first.
:::

Also: choose the increment adaptively rather than fixing the ladder; handle
forward-model failures with a validity mask and a fixed ensemble size; and
carry the PRNG key in the state so runs are reproducible and resumable.

### 4. `pyeki.localize`

Domain localization, not covariance localization — `docs/design.md` explains
why the latter destroys the low-rank structure the conditioning kernel depends
on. Watch the two hazards recorded there: exempting unlocated parameters from
tapering, and fixed-size neighbourhoods with masks so the local analyses
vectorize.

## Open decisions

Deferred deliberately, with enough context to settle later:

**Operator addition dispatch.** There is no `__add__` on operators and no
registry of simplification rules. With the current type list a registry would
carry about two rules. When one is added it needs: a walk over the method
resolution order rather than exact type lookup; an n-ary flattened sum rather
than binary nesting; and a way for a rule to decline. A reasonable alternative
is not to simplify on addition at all, and instead dispatch on structure inside
`solve` and `logdet`.

(Two decisions previously listed here — capability declaration and whitening
versus triangularity — were settled by the operator contract: `supports()` is
defined by hook presence with derived-dependency resolution, and `cholesky()`
was removed in favour of `factor()` plus a primitive `whiten()`.)

## Things not to rediscover

Each of these cost real effort to find and produces wrong numbers rather than
errors. All are recorded in `docs/design.md`; this is the index.

| finding | consequence |
| --- | --- |
| Circulant embedding gives `matvec` and sampling but **not** `solve` or `logdet` on a restricted grid | a spectral log-determinant would be silently wrong |
| For exponential correlation, the *whitener* is bidiagonal, not the factor | sampling is a sequential recurrence, not a banded solve |
| A scalar correlation coefficient is wrong for irregular observation times | build the precision from per-interval coefficients |
| `KronPlusNugget` log-determinant needs an $n\log\det C^l$ term | omitting it is off by a factor, silently |
| Kronecker-plus-nugget needs a strictly positive-definite nugget | a singular one has no simultaneous diagonalization |
| A tapered covariance is PSD only if the taper is a valid PD function | dimension-dependent; use a known family |
| `M @ x` contracts the wrong axis for `ndim >= 2` | wrong answer, no error, whenever the operator is square |
| A `to_dense`-based test does not exercise `matvec` | the obvious guard test is vacuous |
| Lazy factorization caches are discarded inside traces | silent ~10x slowdown |
| Undeclared non-array dataclass fields become tracers | fails later, far from the declaration |
| JAX has no generalized `eigh` | reformulate via Cholesky whitening |

## Working agreements

- `uv` for everything. `uv run pytest` before every commit.
- Docstrings follow `CLAUDE.md`. Every user-facing feature gets a user-guide
  entry, not only an API entry.
- Keep the package domain-agnostic. If a docstring wants to mention a specific
  application, that is a sign it belongs in the calling repository.
