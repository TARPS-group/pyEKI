# Handoff

Written 2026-08-24, updated 2026-08-25 after the operator layer was reworked
against the normative contract, 2026-08-27 after `pyeki.eki` shipped, and
2026-08-28 after the forward-model contract was specified. Read
`CLAUDE.md` first for conventions, then this for state and next steps.

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
Documentation builds with zero warnings: landing page, installation,
quickstart, three user-guide pages plus a guide to writing an operator, the
three normative contracts, design notes, and an API reference.

`pyeki.gauss` is implemented to `docs/gaussian-contract.md`: `Gaussian`,
`EnsembleJoint` and the two array-level conditioning primitives, all routed
through the whitened-SVD kernel. `PSDLowRank`, the operator it needed, is in
`pyeki.linalg`.

`pyeki.eki` is implemented to `docs/eki-contract.md`: the four value classes,
the three policy protocols with eight shipped implementations, the two public
phases of a rung, `run` and `iterate`, the three array-level helpers, and the
`pyeki.eki.testing` conformance harness for user-written policies. Its
user-guide page is `docs/user-guide/running-an-inversion.md`.

The **forward-model contract** is specified in one place as of 2026-08-28, in
the contract's *Forward models and failed members* and in the user-guide page
`docs/user-guide/writing-a-forward-model.md`. Three properties the driver had
been deciding on its own are now stated and tested (obligations 27-29): what
the callable receives, what it may return, and what it must be. It landed on
its own branch, deliberately ahead of the toy forward models and the first
tutorial, both of which consume it — a session writing the contract *and* the
models satisfying it is under pressure to bend the first toward the second.

**Not started.** `pyeki.localize`, and the Kronecker family of operators. The
design background for both is in `docs/design.md`. The toy forward-model module
and the tutorial series are also unwritten; both are now unblocked.

**Origin.** This package was extracted from a research repository where the
operator layer was first written. That repository keeps the domain-specific
work — forward models, priors, experiment configuration — and will depend on
pyEKI. Nothing domain-specific should come back across.

## Next steps, in order

### 1. `Kron`

The operator that blocks the rest of the linalg roadmap. No shipped layer
needs it — `pyeki.gauss` and `pyeki.eki` run on the operators already there —
so this is a capability step rather than an unblocking one. Two variants, and
they are not the same code:

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

### 2. `pyeki.localize`

Domain localization, not covariance localization — `docs/design.md` explains
why the latter destroys the low-rank structure the conditioning kernel depends
on. Watch the two hazards recorded there: exempting unlocated parameters from
tapering, and fixed-size neighbourhoods with masks so the local analyses
vectorize.

`pyeki.eki` is ready for it: localization plugs in as an `EnsembleUpdate`,
and the driver needs no knowledge of it. Two things localization must bring
itself, neither of which `pyeki.eki` supplies: observation **locations**,
which appear nowhere in the layer and so live as static fields on the rule,
and the neighbourhood and taper definitions. One real limit, recorded in the
EKI contract's *How the layers around this one connect*: extracting a
principal submatrix of a *correlated* noise block is not an operator-layer
operation, so localization composes cleanly for diagonal noise or for
neighbourhoods aligned to the noise operator's blocks, and not for arbitrary
neighbourhoods cutting across a correlated block. That is a constraint on
neighbourhood construction rather than a gap in the layer below.

## Open decisions

Deferred deliberately, with enough context to settle later:

**Operator addition dispatch.** There is no `__add__` on operators and no
registry of simplification rules. With the current type list a registry would
carry about two rules. When one is added it needs: a walk over the method
resolution order rather than exact type lookup; an n-ary flattened sum rather
than binary nesting; and a way for a rule to decline. A reasonable alternative
is not to simplify on addition at all, and instead dispatch on structure inside
`solve` and `logdet`.

**The validity mask on `Evaluation`.** Only `n_valid` is carried, not the
`(J,)` boolean mask, so an update that wanted to down-weight repaired members
cannot see which they were. Deferred rather than declined: the update
protocol's `**_` seam makes adding the field non-breaking, and the consumer
that would use it — `pyeki.localize` — does not exist yet and so cannot say
what shape it wants. Revisit when localization lands. The EKI contract's
*Diagnostics* section records the argument.

(Three decisions previously listed here were settled. Capability declaration
and whitening versus triangularity went to the operator contract: `supports()`
is defined by hook presence with derived-dependency resolution, and
`cholesky()` was removed in favour of `factor()` plus a primitive `whiten()`.
`AdditiveInflation`'s supposed per-rung refactorization turned out not to
exist: every shipped PSD operator factorizes at construction, so `factor()`
returns a stored factor and the update path contains no Cholesky at all.)

## Things not to rediscover

Each of these cost real effort to find and produces wrong numbers rather than
errors. Each is recorded in `docs/design.md` or in the contract for its
layer; this is the index.

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
| Per-step noise is $\Sigma/\Delta\beta_t$, never $\Sigma/\beta_t$ | a plausible posterior, wrong by $(T+1)/2$ times the data precision on a uniform $T$-rung ladder, growing with ladder length |
| A single-`where` guard sends a `nan` misfit to the `inf` branch | `nan > 0` is `False`, so the schedule silently returns the *largest* allowed step |
| A Python float passed as a `jit` **argument** does not retrace | the retrace-per-step bug is a *static field* on an object crossing the boundary, so never pass an `EKIState` or `Evaluation` whole |
| `Evaluation.centre_misfit` is not the mean of `Evaluation.misfits` | they differ by exactly $\tfrac{J-1}{2J}\operatorname{tr}(W \widehat C_{vv} W^\top)$ |
| The repair formula is not bit-exactly the identity when nothing failed | it must be `jnp.where(valid, ensemble, centre)` *and* skipped in Python on the synchronized `n_valid` |
| A static field on a `HistoryRecord` makes every record a different pytree | `jax.tree.map` across a history raises instead of stacking |
| `step` is cumulative across runs | chaining a fresh ladder onto a finished state returns unchanged, with nothing raised — use `restart()` |
| `cov.factor()` is free, because operators factorize at construction | "hoisting" it by storing a densified factor turns an $O(P)$ diagonal into a $P \times P$ array |
| Every comparison against `nan` is `False` | a bisection on such a comparison silently returns its lower bracket, and a floor then makes that look like an ordinary step |
| `jnp.mean(x, axis=-2)` without `keepdims` | the subtraction right-aligns against the batch axis, so an operand whose leading axis equals $J$ broadcasts and returns wrong anomalies without raising |
| `np.asarray` on the forward model's argument returns a **read-only view**, not a copy | writing into it raises `assignment destination is read-only` from wherever you wrote, not at the conversion; copy with `np.array` |
| A `float32` forward model is promoted and warned about, not rejected | it still costs ~$7\times10^{-5}$ relative in the posterior mean where the prediction mean exceeds the spread by $10^4$; promotion recovers only about half, since the digits are gone before the array arrives |
| `ensemble @ G` instead of `ensemble @ G.T` is silent when $G$ is square | the transposed model's predictions, right shape, no error; `G @ ensemble` raises, so it is the harmless mistake |

## Working agreements

- `uv` for everything. `uv run pytest` before every commit.
- Docstrings follow `CLAUDE.md`. Every user-facing feature gets a user-guide
  entry, not only an API entry.
- Keep the package domain-agnostic. If a docstring wants to mention a specific
  application, that is a sign it belongs in the calling repository.
