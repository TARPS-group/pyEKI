# Handoff

Written 2026-08-24, updated 2026-08-25 after the operator layer was reworked
against the normative contract, 2026-08-27 after `pyeki.eki` shipped,
2026-08-28 after the forward-model contract was specified and the layer
vocabulary was fixed, and 2026-09-02 after the joint was split into a
Gaussian and a sample container. Read `CLAUDE.md` first for conventions — including the
layer-boundary rules, which are new — then this for state and next steps.

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
`GaussianJoint`, `EmpiricalJoint` and the two array-level conditioning
primitives, all routed through the whitened-SVD kernel. `PSDLowRank`, the
operator it needed, is in `pyeki.linalg`.

As of 2026-09-02 the joint is **two** classes. `GaussianJoint` holds a joint
Gaussian as a *joint factor* — one factor of the block covariance, cut into
the row blocks that drive both blocks from a shared latent vector — and owns
`condition` and the pathwise (Matheron) map. `EmpiricalJoint` holds paired
samples, offers `to_gaussian_joint()`, and keeps the two updates that return
samples. `condition` is gone from it: conditioning samples means conditioning
a Gaussian fitted to them, and that fit is now written at the call site.

The point of the split is `GaussianJoint.from_linear_map`, which builds the
joint of $u$ and $Gu$ and so gives closed-form linear-Gaussian posteriors —
previously unreachable, since the only entrance to conditioning was to
present samples. `docs/joint-factor.md` derives the representation and records
why it is a factor rather than three covariance blocks. The square-root update
stayed on `EmpiricalJoint` because its reading of the conditioned factor is
valid only for a centred one, which holding samples makes structural; the
contract and that page both give the measured failure it avoids.

`pyeki.eki` was not touched: both update policies call the same two methods
with the same signatures.

**`pyeki.toy`** shipped on 2026-09-02, and is the module `CLAUDE.md` had been
promising: three toy problems, each a forward model bundled with a prior, a
noise covariance, synthetic data and the parameters that generated it.
`linear_gaussian` at any pair of dimensions, whose `posterior(level)`
delegates to `GaussianJoint.from_linear_map(...).condition(y, R / level)` —
two lines, and the reason the split above had to land first; the
`exponential_decay` problem, tuned until a unit step and an adaptive ladder
differ reliably rather than coincidentally; and `restricted_decay`, the same
model with a valid domain, so a member whose rate is not positive returns a
non-finite row. Its user-guide page is `docs/user-guide/toy-models.md`.

The problems are frozen dataclasses of plain values and are deliberately
**not callable**: `run` takes the triple, and the contract excludes a
container accepted in its place. They are not pytrees, hold no mutable state
and count no calls — the recorder in `tests/test_eki.py`'s `_AffineProblem` is
a test instrument and stays there. The existing local closures in `tests/`
were **not** migrated: several are instrumented and several write their
reference locally on purpose, which is what makes them regression tests for
the layer.

`pyeki.eki.testing` gained **`check_forward_model`**, which checks a user's
own model from outside a run: shape at two ensemble sizes, dtype, determinism,
and row independence — twice, because permuting the members is bit-exact and
catches order-dependent coupling while a symmetric coupling survives it and
needs a subset re-evaluation, to a tolerance. The contract's "nothing detects
this" about row coupling is now "no *run* detects this", in both the contract
and the guide.

`pyeki.eki` is implemented to `docs/eki-contract.md`: the four value classes,
the three policy protocols with eight shipped implementations, the two public
phases of a step, `run` and `iterate`, the three array-level helpers, and
the `pyeki.eki.testing` conformance harness for user-written policies. Its
user-guide page is `docs/user-guide/running-an-inversion.md`.

The **forward-model contract** is specified in one place as of 2026-08-28, in
the contract's *Forward models and failed members* and in the user-guide page
`docs/user-guide/writing-a-forward-model.md`. Three properties the driver had
been deciding on its own are now stated and tested (obligations 27-30): what
the callable receives, what it may return, and what it must be. It landed on
its own branch, deliberately ahead of the toy forward models and the first
tutorial, both of which consume it — a session writing the contract *and* the
models satisfying it is under pressure to bend the first toward the second.

The **layer vocabulary** was unified at the same time, and the rules are now
in `CLAUDE.md`. One word per concept: a *run* contains *steps*, each step has
two *phases* (`evaluate` and `assimilate`) made of numbered *operations*, and
each step is preceded by one *evaluation* of the forward model. "Rung" and
"iteration" as a countable noun are retired. Vocabulary flows downward only:
`pyeki.gauss` has samples, not members, and `pyeki.linalg` speaks of neither
Gaussians nor conditioning. The renames that followed: `EnsembleJoint` ->
`EmpiricalJoint`, its `n_members` -> `n_samples`, `apply` -> `assimilate`,
`EKIResult.n_steps` -> `n_evaluations` plus a new `n_completed_steps`, and
`n_obs` -> `v_dim`. `docs/user-guide/conditioning.md` was rewritten in the
gauss layer's own vocabulary at the same time; it was the last prose describing
that layer in EKI's words.

Two things came out of the adversarial review of that branch. An **inflation's
output dtype is now checked**, as an update's already was — the inflated
members are what the forward model is called on, so an inflation returning
`int64` used to hand the model an integer ensemble with nothing raised.
And **a forward model returning a dtype *wider* than the run is not demoted**,
so it fails at the update's dtype check with an error naming the update rule;
that is deliberate for now and recorded as issue #19.

**Tutorial 1** shipped on 2026-09-04, complete and revised, with the figure
machinery the rest of the series needs. Tutorials 2 and 3 are drafted, tested
and building, but have **not** been through a revision pass — they are marked
as unreviewed drafts on the series index, and they run the library's default
update rule rather than the pathwise one tutorial 1 selects. Revising them is
its own PR.

The series was also restructured: tutorial 3 was carrying four lessons, so it
split into three — sampling against optimizing (the destination), tempering
schedules (the path), and the two update rules — which pushed the four
remaining stubs down by two. The series is now nine pages, of which 4 to 9 are
stubs.

**Tutorial 1 runs `PathwiseUpdate`, not the default.** On
`exponential_decay`, `TransformUpdate` reproduces the target's covariance but
leaves the ensemble strung along one direction: the across-ridge projection
has a kurtosis of 19 against a Gaussian's 3, and at 64 members two members
carry 72% of that variance. It is the nonlinearity, not sampling error —
present after a single step, and no better at 4096 members, while on
`linear_gaussian` the same rule leaves the ensemble perfectly Gaussian at any
number of steps. `PathwiseUpdate` draws a perturbation per member and refills
the cloud, at about one extra forward evaluation. Issue #29 asks whether the
package default should change; if it does, the explicit argument and its
explanation come out of the page.

Three things the first three pages settled, none of which had a precedent:

- **Figures are generated at build time.** `docs/figures.py` is both the figure
  module and a Sphinx extension: a `builder-inited` handler writes every figure
  into `docs/_generated/figures`, which is gitignored. Nothing is committed, so
  a figure cannot disagree with the code that made it. Regeneration is skipped
  when every output postdates both that module and every source file of the
  package — 6 s to regenerate all of them, 1.6 s cached — and
  `PYEKI_DOCS_FIGURES=force` overrides. That alone catches only a figure whose
  code *raises*, so each figure function also returns the numbers it plotted
  and `tests/test_tutorials.py` pins them; a figure drawing the wrong array
  fails the test rather than merely looking wrong. Pixels are deliberately not
  compared, and the module records why.
- **The quickstart stays.** Tutorial 6 (was 4) does not absorb it: four pages
  link to it, and it serves a reader who came for `pyeki.linalg` alone.
  Tutorial 6 is short and problem-led instead, and its stub now says so.
- **Notebook wiring is still undecided**, deliberately. `myst-nb` would
  *replace* `myst_parser` rather than join it, which changes how all 23 existing
  pages are parsed, and that does not belong in a tutorials branch. The three
  sub-decisions at the bottom of `docs/examples/index.md` stand as written, and
  the figure machinery above forecloses none of them.

The nonlinear problem has no closed form, so the pages compare against grid
quadrature — `figures._tempered_moments`, which **refines its box**, because
one grid does not serve every level: a box wide enough for the prior resolves
the $\beta = 1$ target with about five points across its width and reports a
mean wrong in the fourth decimal, while a box sized for the target puts the
prior's mean at `[1.48, 1.31]` instead of `[1, 1]`. After two refinements the
moments agree to six digits across three resolutions and recover the prior
exactly at $\beta = 0$. Both failure modes are silent and invisible in a
contour plot; `tests/test_tutorials.py` asserts against each.

**Not started.** `pyeki.localize`, and the Kronecker family of operators. The
design background for both is in `docs/design.md`. Tutorials 4 to 9 and the
example notebooks are unwritten. Tutorial 7's stub (was 5) carries re-measured
numbers from the shipped problem; issue #24 covers what its page still owes.
Issue #26 proposes a fixed-budget ablation study and leaves open whether it is
a tutorial or a notebook.

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
`AdditiveInflation`'s supposed per-step refactorization turned out not to
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
| Per-step noise is $\Sigma/\Delta\beta_t$, never $\Sigma/\beta_t$ | a plausible posterior, wrong by $(T+1)/2$ times the data precision on a uniform $T$-step ladder, growing with ladder length |
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
| A run's evaluations and its completed steps differ by one whenever it ends on a stopping rule or a `None` increment | a single `n_steps` naming both is how the ambiguity arose; the terminal record is the one with a zero increment, at most one, always last |
| A policy's output needs its **dtype** checked, not only its shape | an inflation returning `int64` handed the forward model an integer ensemble and the run completed silently; the shape check passed |
| A `float32` forward model is promoted and warned about, not rejected | it still costs ~$7\times10^{-5}$ relative in the posterior mean where the prediction mean exceeds the spread by $10^4$; promotion recovers only about half, since the digits are gone before the array arrives |
| `ensemble @ G` instead of `ensemble @ G.T` is silent when $G$ is square | the transposed model's predictions, right shape, no error; `G @ ensemble` raises, so it is the harmless mistake |
| A **symmetric** coupling across ensemble members survives a permutation of them | mean-centring is permutation-equivariant, so a permutation test alone passes a model that normalizes across the ensemble; it takes re-evaluating a *subset* to catch, and that comparison cannot be bit-exact |
| The same members in a differently *sized* batch round differently | a dense contraction picks a different kernel per batch shape, so a subset comparison holds only to round-off while a permutation one is bit-exact |
| The closed form's cost is set by the prior factor's width $k$, not by $P$ | a full-rank prior at $P = 2000$ means a $(2000, 2000)$ posterior factor — 32 MB, 0.07 s — so `LinearGaussian.posterior` guards on $Pk$ and the *run* has no such limit |
| A run at $P \gg J$ reports a spread the exact posterior contradicts | at $P = 2000$, $N = 40$, $J = 40$ the ensemble's mean posterior sd is 0.014 against an exact 0.990, a factor of seventy, with nothing raised and no history field flagging it |
| A deterministic square-root update can get the covariance right and the shape wrong | on a nonlinear problem `TransformUpdate` leaves the least-varying direction with a kurtosis of 19 and two members holding 72% of its variance; it is a linear recombination of existing anomalies, so more members do not help. Only visible in a scatter plot or a shape statistic — no `HistoryRecord` field reports it |
| One grid cannot serve every tempering level | a box wide enough for the prior reports the $\beta = 1$ mean wrong in the fourth decimal; a box sized for the target reports the prior's mean as `[1.48, 1.31]` rather than `[1, 1]`. Both are silent, and invisible in a contour plot |
| A multi-line `:alt:` value breaks a MyST `{figure}` | the continuation lines are absorbed into the caption, and the build fails with "Figure caption must be a paragraph" pointing at the directive rather than at the option |

## Working agreements

- `uv` for everything. `uv run pytest` before every commit.
- Docstrings follow `CLAUDE.md`. Every user-facing feature gets a user-guide
  entry, not only an API entry.
- Keep the package domain-agnostic. If a docstring wants to mention a specific
  application, that is a sign it belongs in the calling repository.
