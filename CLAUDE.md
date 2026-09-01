# CLAUDE.md — pyEKI

## What this project is

pyEKI is a small, robust, efficient implementation of Ensemble Kalman Inversion
(EKI) and its common variants, for derivative-free Bayesian calibration of
expensive forward models.

It is a **library**, not a research repository. The deliverable is a
well-documented, well-tested package that colleagues can depend on. Prefer
clarity and correctness over cleverness, and keep the public surface small.

Four layers, each building on the one below:

1. `pyeki.linalg` — a lean structured linear operator layer, scoped to what EKI
   needs rather than to general-purpose linear algebra.
2. `pyeki.gauss` — joint Gaussian distributions and conditioning. *(planned)*
3. `pyeki.localize` — distance-based localization. *(planned)*
4. `pyeki.eki` — tempering schedules, ensemble updates, inflation, the driver
   loop, and variants. *(planned)*

## What this project is NOT

Out of scope, deliberately and permanently:

- **Forward models.** The forward model is any callable from parameters to
  predicted observations. pyEKI ships toy models for testing only.
- **Priors, Gaussian process kernels, coregionalization.** A prior is any
  operator satisfying the covariance interface. Constructing covariances from
  kernels belongs to the caller.
- **Domain-specific anything.** No knowledge of the systems being calibrated
  should appear in this package, including in docstrings and examples.
- **A general-purpose linear algebra library.** `pyeki.linalg` exists because
  EKI needs structured operators. Add a structure when EKI needs it, not
  because it would be nice to have.

## Layer boundaries and vocabulary

The four layers are a hierarchy, and **vocabulary flows downward only**. A
layer may name a layer above it to justify its own scope; it may not borrow
that layer's concepts to define its own behaviour.

| layer | speaks of | must not speak of |
| ----- | --------- | ----------------- |
| `pyeki.linalg` | operators, batches, rows, factors, whiteners | Gaussians, conditioning, priors, posteriors, samples, ensembles, members, steps |
| `pyeki.gauss` | Gaussians, conditioning, samples, the `u` and `v` blocks | ensembles, members, steps, tempering, forward models, EKI |
| `pyeki.eki` | runs, steps, ensembles, members, tempering, forward models | — |

The one permitted upward reference is **naming a consumer to justify scope**:
"a square variant will be added when an EKI consumer needs `solve`" is fine,
because this package scopes the lower layers by what EKI needs. Using the
consumer's *time or domain vocabulary* to define lower-layer semantics is not:
write "a scale that may itself be traced", never "the per-step noise
covariance". A lower layer must read correctly to someone who has never heard
of EKI.

The boundary is crossed in exactly one place, where `pyeki.eki` builds an
`EmpiricalJoint` from its ensemble: members go in, samples come out.

**Names for sizes.** `n_<plural noun>` counts things — `n_members` (EKI),
`n_samples` (gauss), `n_valid`, `n_steps`, `n_evaluations`, `n_in`/`n_out`
(linalg). `<block>_dim` is the dimension of a named vector block — `u_dim`,
`v_dim`. There is exactly one name per quantity: $N$ is `v_dim`, never also
`n_obs`.

**Names for the parts of an EKI run** are normative and specified in
`docs/eki-contract.md` under *Terminology*: a **run** contains **steps**, each
step has two **phases** (`evaluate` and `assimilate`) made of numbered
**operations**, and each step is preceded by one **evaluation** of the forward
model. "Rung" and "iteration" as a countable noun are retired; do not
reintroduce them.

## Package management

`uv`. Use `uv sync` and `uv sync --group dev`; never `pip install` into the
environment. Run tests with `uv run pytest`.

## Docstring conventions

These apply to all docstrings, and strictly to module-, class-, and
public-function-level ones.

**Write for the person calling the code.** Lead with what the thing is and how
to use it. Explain behaviour, arguments, return values, and errors — not the
reasoning that led to the implementation.

**Use clear, precise language and no unnecessary jargon.** Prefer a plain
description over a compressed technical phrase. Do not editorialize about the
design: sentences like "the split is load-bearing rather than cosmetic" state a
low-level design judgement and do not belong at the top of an API.

**Organize with sections.** Use numpydoc headings — `Parameters`, `Returns`,
`Raises`, `Notes` — and tables when listing several classes or functions. A
reader should be able to skim the structure.

**Put design rationale in a `Notes` section, or leave it out.** Consequential
lower-level decisions are worth recording when they are non-obvious or easy to
undo by accident, but they go at the bottom under `Notes`, never in the opening
description. Extended rationale belongs in `docs/design.md`.

**Scope each level distinctly; do not repeat yourself.**
- *Module*: what the module provides, an index of its contents, and any
  convention shared across everything in it.
- *Class*: what this class represents and its parameters. Do not restate
  module-level conventions.
- *Method/function*: what this call does, its arguments and return value. Do
  not restate class-level context.

**Keep docstrings self-contained.** Do not reference anything outside the
repository. A reader with only the source must be able to follow them.
Cross-reference other modules and classes within the package freely, using
Sphinx roles (`:class:`, `:mod:`, `:meth:`, `:func:`).

## Documentation

Sphinx with the furo theme, `myst-parser` for Markdown pages, and `napoleon`
for numpydoc-style docstring sections. Build with:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Every user-facing feature needs a place in the user guide, not only an API
entry. The user guide explains *when and why*; the API reference explains
*what*.

## Code conventions

**Array shapes: leading batch axes, core operand shape trailing.** This is the
NumPy generalized-ufunc rule and what `vmap` produces. It applies everywhere,
not only in `linalg`.

**Contract the trailing axis.** Never write `M @ x` in an operator
implementation — for arrays of two or more dimensions it contracts the
second-to-last axis, which silently returns a wrong answer when the operator is
square. Use `pyeki.linalg.dense_matvec`.

**Fail loudly.** Unsupported operations raise rather than falling back to dense
linear algebra. Size guards raise before allocating.

**Return JAX scalars, not Python floats.** Converting fails on a tracer under
`jit`, and on any complex intermediate.

**Factorize at construction time, in `from_matrix`-style classmethods.** The
dataclass constructor only stores: pytree reconstruction rebuilds operators
from their stored fields alone, bypassing the constructor. Never cache a
factorization lazily — a cache written inside a traced function is discarded,
so the operator silently re-factorizes on every call.

**Every new operator gets `check_operator`.** The conformance suite in
`pyeki.linalg.testing` catches the batch-rank and square-root bugs that
otherwise produce wrong numbers without raising.

## JAX notes

- Float64 is enabled in `pyeki/__init__.py`. Worker processes do not inherit
  it; that needs `JAX_ENABLE_X64=1` in the environment.
- Operators are pytrees via the `@linop` decorator, with data and metadata
  fields declared explicitly; its unflatten bypasses the constructor, so
  validation runs only at genuine construction.
- Operators compare by identity and are never `static_argnums`.
- `shape` is a property, not a stored field, so it stays concrete under `jit`.
- JAX has no generalized `eigh`; use a Cholesky whitening reformulation.

## Testing

`pytest`, in `tests/`. Three kinds:

1. **Conformance** — every operator instance through `check_operator`.
2. **Targeted regression** — one test per bug class that produces wrong numbers
   without raising. These are the valuable ones; do not delete them as
   redundant.
3. **Exactness** — where a closed form exists, check against it rather than
   against a tolerance chosen to make the test pass.
