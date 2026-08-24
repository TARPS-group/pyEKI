# Handoff

Written 2026-08-24, at the point where the package core exists and development
moves to a fresh session. Read `CLAUDE.md` first for conventions, then this for
state and next steps.

## Where things stand

**Done.** `pyeki.linalg` is implemented, documented and tested — the three-level
operator hierarchy, six leaf operators, six composites, and a conformance
harness. 38 tests pass. Documentation builds: landing page, installation,
quickstart, operator catalogue, a guide to writing an operator, design notes,
and an API reference.

`Kron` and `KronGeneral` landed on 2026-08-24, with the orientation and
log-determinant results recorded in `docs/design.md` and in the table below.

**Not started.** `pyeki.gauss`, `pyeki.localize`, `pyeki.eki`. The design for
all three is in `docs/design.md`, which is the single most useful thing to read
before writing any of them.

**Origin.** This package was extracted from a research repository where the
operator layer was first written. That repository keeps the domain-specific
work — forward models, priors, experiment configuration — and will depend on
pyEKI. Nothing domain-specific should come back across.

## Next steps, in order

### 1. `KronLMC`, `KronPlusNugget`, `LowRankPlus` (start here)

`Kron` and `KronGeneral` are done. The remaining operators, in this order:

- **`KronLMC`**, a sum $\sum_q A_q \otimes B_q$. Build on `Kron`; note that a
  sum of Kronecker products has no shared eigenbasis in general, so `solve` and
  `logdet` need either the spectral route or no implementation at all.
- **`KronPlusNugget`**. `docs/design.md` records the simultaneous
  diagonalization, the $n\log\det C^l$ term that is easy to omit, and the
  requirement that the nugget be strictly positive definite.
- **`LowRankPlus`**.

Follow the orientation convention `Kron` establishes — first factor slow,
matching `numpy.kron` — and pin each new operator's `matvec` *and* `to_dense`
to `numpy.kron` separately rather than to each other. `docs/design.md` explains
why the latter constrains nothing, with the mutation result that demonstrates
it.

### 2. `pyeki.gauss`

The conditioning layer. `docs/design.md` gives the kernel: the whitened-SVD
gain, which is the form to implement — not the algebraically equivalent
Woodbury identity on the normal equations, which squares a condition number
that is already the problem.

Suggested shape: a `JointGaussian` protocol supplying means, a `gain_apply`,
and joint sampling; two implementations, one from ensemble anomalies and one
from structured operators; and two conditioning functions, moment-based and
pathwise. The pathwise form is what EKI uses; the moment form is the reference
the tests compare against.

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

**Capability declaration.** `supports()` compares the resolved method against
the base class's raising default, and composites override it to intersect over
their children. This works, but it is implicit. If it becomes a problem, derive
it at class creation from `cls.__dict__` instead.

**Whitening versus triangularity.** `cholesky()` requires a square triangular
factor, but whitening only needs a square invertible one. Some structures have
the latter without the former. If that becomes limiting, decouple `whiten` from
`cholesky` rather than forcing triangularity. Now tracked as issue #1, with the
two instances that exist today — `BlockDiag` and `Kron` — written up there; a
third is likely when `KronPlusNugget` and `LowRankPlus` land.

## Things not to rediscover

Each of these cost real effort to find and produces wrong numbers rather than
errors. All are recorded in `docs/design.md`; this is the index.

| finding | consequence |
| --- | --- |
| Circulant embedding gives `matvec` and sampling but **not** `solve` or `logdet` on a restricted grid | a spectral log-determinant would be silently wrong |
| For exponential correlation, the *whitener* is bidiagonal, not the factor | sampling is a sequential recurrence, not a banded solve |
| A scalar correlation coefficient is wrong for irregular observation times | build the precision from per-interval coefficients |
| Kronecker orientation is silent: $B \otimes A$ is PSD whenever $A \otimes B$ is, and the same shape when the factors match in size | a valid covariance with the wrong meaning, no error |
| `Kron` log-determinant pairs each factor with the size of the *other* factor | off by a factor, and invisible to any test with equal-size factors |
| A self-consistent `matvec`/`to_dense` pair passes the whole conformance suite | orientation must be pinned to `numpy.kron`, not to the other method |
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
