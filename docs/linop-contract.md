# Linear operator contract

This page specifies the operator layer of `pyeki.linalg`: the class
hierarchy, the contract of every method, and the rules an operator
implementation must satisfy. It is normative — an implementation that
violates a rule here is defective even if its tests pass — and it is the
reference for two audiences: contributors writing or reviewing operators,
and users who want a more precise account of the layer than the
{doc}`user-guide/operators` catalogue gives.

Throughout, *must* and *never* state requirements, *should* states a strong
default that a documented reason may override, and *may* states a
permission. {doc}`design` records *why* the load-bearing decisions were
made; this page records *what* they require.

:::{admonition} Status: specification ahead of code
:class: important

This document describes the design the implementation is being brought to,
not the layer as currently shipped. Where the two disagree, this document
wins. {ref}`contract-changes` lists every difference. The section will be
removed once the implementation conforms.
:::

## Scope

The layer represents matrices implicitly, by how they act on vectors, so
that structure — diagonal, block, triangular, low-rank, Kronecker — is
exploited rather than materialized. It is scoped to what Ensemble Kalman
Inversion needs: applying operators and their transposes, solving against
them, and taking square roots to sample and whiten. It is deliberately not
a general-purpose linear algebra library; {ref}`contract-excluded` lists
what is left out and why.

(contract-hierarchy)=
## The hierarchy

Operators form three levels. Each level is a mathematical claim about the
map, and each adds exactly the operations that the claim makes well
defined.

| class         | claim                                        | adds                                                  |
| ------------- | -------------------------------------------- | ----------------------------------------------------- |
| `LinOp`       | a linear map, possibly rectangular           | `matvec`, `rmatvec`, `matmat`, `rmatmat`, `T`, `to_dense` |
| `SquareLinOp` | the map is square                            | `solve`, `solve_mat`, `logdet`, `diag`                |
| `PSDLinOp`    | the map is symmetric positive semi-definite  | `factor`, `whiten`, `whiten_mat`                      |

The split is what makes unrepresentable states unrepresentable: a
rectangular operator does not *refuse* `solve`, it does not *have* it. This
matters most for the operators `factor()` returns — square roots are
generally rectangular and never assumed self-adjoint, so they must not be
able to advertise an inverse or a determinant.

Three rules govern the hierarchy:

1. **A class's level is its strongest true claim.** An operator that is
   square but not known PSD subclasses `SquareLinOp`; an operator that is
   PSD subclasses `PSDLinOp`, even when some inherited operations have no
   cheap implementation for it.
2. **Capabilities are monotone.** A subclass never removes or disables an
   operation its base class provides. If a would-be subclass cannot honour
   an inherited operation *in principle*, the subclass relationship is
   wrong, not the operation. (Whether an operation has a *cheap
   implementation* is a separate, per-instance question — see
   {ref}`contract-capabilities`.)
3. **Level membership is static.** Whether an operator is square or PSD is
   decided by its Python type, never by inspecting array values. This keeps
   every dispatch decision available at trace time.

`LinOp` is an abstract base class: instantiating a subclass that lacks any
required piece (`shape`, `_matvec`, `_rmatvec`, `_to_dense`) fails at
instantiation, not on first use.

## Shape contract

Every operator has a `shape` property returning `(n_out, n_in)` — rows,
then columns, matching the dense array `to_dense()` returns.

- `shape` **is a property, not a stored field**, computed from static
  information (stored array shapes, static integer fields). It is therefore
  a concrete tuple of Python ints even under `jit`, and is usable in shape
  arithmetic, `jnp.split` points, and Python-level branches.
- `SquareLinOp` adds `n`, the side length, equal to both entries of
  `shape`.
- **Operators are unbatched.** Every stored array has exactly its core
  rank: a `Dense` stores a 2-D array, a `PSDDiagonal` a 1-D array. The
  user-facing constructors — classmethods like `from_matrix` and the
  factories — enforce this exactly; the storing constructor enforces only
  a rank *floor*, for reasons the pytree machinery imposes (see
  {ref}`contract-validation` and {ref}`contract-batching-operators`). A
  "batch of operators" is expressed with `jax.vmap` over the operator
  pytree, never by handing arrays with extra leading axes to a strict
  constructor.
- **Shapes are strictly positive.** Both entries of `shape` are at least
  1; the strict constructors reject empty operators. (The `k` axis of a
  matrix *operand* may be 0 — see the batch contract.)

(contract-batch)=
## Batch contract

Operands carry the batching. Every array-accepting method obeys one rule:

> **The operand's core shape is trailing; any number of leading axes are
> batch axes. The method is applied independently over the batch axes, and
> the result has the same batch shape.**

This is the NumPy generalized-ufunc convention and what `jax.vmap`
produces, so operators compose with `vmap` without axis bookkeeping.

The core signatures, with `...` the (possibly empty) batch shape:

| method       | core operand           | signature                                | contracted axis |
| ------------ | ---------------------- | ---------------------------------------- | --------------- |
| `matvec`     | vector `(n_in,)`       | `(..., n_in) -> (..., n_out)`            | `-1`            |
| `rmatvec`    | vector `(n_out,)`      | `(..., n_out) -> (..., n_in)`            | `-1`            |
| `matmat`     | matrix `(n_in, k)`     | `(..., n_in, k) -> (..., n_out, k)`      | `-2`            |
| `rmatmat`    | matrix `(n_out, k)`    | `(..., n_out, k) -> (..., n_in, k)`      | `-2`            |
| `solve`      | vector `(n,)`          | `(..., n) -> (..., n)`                   | `-1`            |
| `solve_mat`  | matrix `(n, k)`        | `(..., n, k) -> (..., n, k)`             | `-2`            |
| `whiten`     | vector `(n,)`          | `(..., n) -> (..., n)`                   | `-1`            |
| `whiten_mat` | matrix `(n, k)`        | `(..., n, k) -> (..., n, k)`             | `-2`            |

Methods that take no operand return fixed shapes: `diag()` returns `(n,)`,
`logdet()` returns a 0-d array, `to_dense()` returns `(n_out, n_in)`.

### Why every vector method has a matrix sibling

Under a trailing-core convention, an array of shape `(n, k)` is genuinely
ambiguous: it is both a stack of `n` vectors of length `k` and a single
`n × k` matrix. For a square operator $A$ of side $n$ applied to an
$n \times n$ array $X$, the two readings are *both shape-valid and give
different answers*:

$$
\texttt{matvec}(X) = (A X^\top)^\top = X A^\top,
\qquad
\texttt{matmat}(X) = A X .
$$

No method can infer which was meant from the number of dimensions, so the
layer never tries: `matvec`, `solve` and `whiten` always mean "a batch of
vectors", and `matmat`, `solve_mat` and `whiten_mat` always mean "a matrix
operand, possibly batched". The pairing is a requirement of the batch
contract, not an API convenience.

The two readings are nevertheless consistent in exactly the way intuition
expects: **`matmat` is batched `matvec` — batched over columns** rather
than over leading axes,

$$
\texttt{matmat}(X)[\ldots,\, :,\, j] \;=\; \texttt{matvec}\bigl(X[\ldots,\, :,\, j]\bigr),
$$

and the default `_matmat` hook implements precisely this identity (the
same holds for `solve_mat`/`solve` and `whiten_mat`/`whiten`). What
distinguishes this layer from column-stacked operator libraries is not
whether matrix application is batched vector application — it is — but
which axis carries a batch of vectors in the *vector* methods: leading,
because that is what `vmap` produces and the layout in which ensembles
arrive from a `vmap`-ed forward model.

The matrix methods are the single exception to "contract the trailing
axis": their core operand is 2-D, so they contract axis `-2` and carry `k`
along. `k` is part of the core shape, never a batch axis.

### Rules for implementations

- Contract the trailing axis explicitly. **Never write `M @ x`** inside an
  implementation: for operands with two or more dimensions, `@` contracts
  the second-to-last axis, which silently returns a wrong answer exactly
  when the operator is square. Use the provided helpers, which encode the
  convention: `dense_matvec(A, x)` contracts the trailing axis of `x`
  against the trailing axis of the unbatched matrix `A`, so
  `(m, n) × (..., n) -> (..., m)`; `tri_solve(L, b, *, lower, trans=0)` is
  the trailing-axis triangular solve under the same batch rule.
- Batch axes are carried, never broadcast against. The operator itself has
  no batch axes (see above), and the operand's batch shape passes through
  unchanged. An implementation must produce the same result as looping the
  core operation over the batch — `vmap` semantics, which the conformance
  suite checks by comparing against `jax.vmap` directly.
- Results at batch rank 0, 1 and 2 must agree with the dense reference.
  Rank-dependent bugs (a reshape that hard-codes one batch axis, a
  reduction over the wrong axis) are the layer's most dangerous class of
  defect because they produce wrong numbers without raising; the
  conformance suite tests all three ranks for this reason.

(contract-batching-operators)=
### Batching over operators

A family of operators — one covariance per ensemble member, say — is a
*batched pytree*: apply `jax.vmap` over the operator argument, and inside
the mapped function the operator behaves exactly like an unbatched one.

```python
covs = jax.vmap(DensePSD.from_matrix)(As)          # As: (m, n, n)
outs = jax.vmap(lambda C, x: C.solve(x))(covs, xs) # xs: (m, n)
```

This is the only supported way to batch an operator. Handing batched
arrays to a strict constructor (`Dense.from_matrix` on a 3-D array) is
rejected.

The mechanics deserve precision, because they dictate where validation
may live. *Inside* a `vmap` trace, each leaf is presented at its unbatched
core shape, so strict checks would pass there. But at the **vmap exit
boundary** — and on every other unflatten of a batched family — the
operator is reconstructed through its *storing constructor* with fully
batched concrete leaves, outside any trace. Exact-rank enforcement in the
storing constructor would therefore make the family above unconstructible.
Hence the split stated in the shape contract: **classmethods and
factories, which are never on an unflatten path, enforce exact core rank;
the storing constructor rejects only rank below the core** and accepts
extra leading axes. An operator whose leaves carry extra leading axes is a
*vmapped family*: it is valid as a `vmap` argument and for pytree
operations, and nothing else — the contract defines no behaviour for
calling its methods directly (its `shape` reads only the trailing core
axes). The conformance suite checks that constructing and returning an
operator inside `vmap` round-trips.

## Method contracts

### Structure: public methods and implementation hooks

Every operation comes in two parts:

- a **public method** (`matvec`, `solve`, ...), defined once on the base
  classes and not overridden by operator classes, which checks the
  capability gate, validates the operand's core shape, and delegates; and
- an **implementation hook** (`_matvec`, `_solve`, ...), which subclasses
  implement and which may assume a valid operand.

Operator authors implement hooks only. This makes operand validation
impossible to forget, gives every operator identical error behaviour, and
gives the capability system ({ref}`contract-capabilities`) a single
enforcement point.

Three clauses pin the division of labour:

- **Hooks receive the operand unchanged, batch axes included.** The public
  method validates and passes the array through as-is; every hook
  implements the full batch contract itself — usually one broadcasting
  expression, which the helpers encode. The base classes never loop or
  `vmap` on a hook's behalf.
- **The capability gate runs before operand validation**, so an
  unsupported operation raises `UnsupportedOpError` regardless of the
  operand.
- **The two properties, `shape` and `T`, are exempt from the no-override
  rule.** They take no operand, so there is nothing to validate or gate —
  and `T` is expressly overridden where the transpose is structured
  (`Triangular`), by `PSDLinOp` (returns `self`), and by `Transposed`
  (returns the wrapped operator).

The full method set:

| public       | hook          | level         | availability                     |
| ------------ | ------------- | ------------- | -------------------------------- |
| `shape`      | (property)    | `LinOp`       | required                         |
| `matvec`     | `_matvec`     | `LinOp`       | required                         |
| `rmatvec`    | `_rmatvec`    | `LinOp`       | required¹                        |
| `matmat`     | `_matmat`     | `LinOp`       | derived from `_matvec`           |
| `rmatmat`    | `_rmatmat`    | `LinOp`       | derived from `_rmatvec`          |
| `T`          | (property)    | `LinOp`       | always                           |
| `to_dense`   | `_to_dense`   | `LinOp`       | required                         |
| `solve`      | `_solve`      | `SquareLinOp` | optional                         |
| `solve_mat`  | `_solve_mat`  | `SquareLinOp` | derived from `_solve`            |
| `logdet`     | `_logdet`     | `SquareLinOp` | optional                         |
| `diag`       | `_diag`       | `SquareLinOp` | optional                         |
| `factor`     | `_factor`     | `PSDLinOp`    | optional                         |
| `whiten`     | `_whiten`     | `PSDLinOp`    | optional                         |
| `whiten_mat` | `_whiten_mat` | `PSDLinOp`    | derived from `_whiten`           |

¹ `PSDLinOp` provides a delegating `_rmatvec` — a real method whose body
returns `self._matvec(x)`, since a PSD operator is self-adjoint — so PSD
authors implement nothing extra. It must be a delegating method, not a
class-body alias `_rmatvec = _matvec`: an alias of the abstract `_matvec`
carries `__isabstractmethod__` (leaving every PSD subclass abstract) and
would bind the base function, never seeing subclass overrides.

*Derived* hooks have a default written in terms of another hook: `matmat`
transposes into `matvec` and back, `solve_mat` into `solve`, `whiten_mat`
into `whiten`. A derived operation is available exactly when its dependency
is, and an operator with a faster direct implementation may override the
hook (a `Dense` can implement `_matmat` as one `einsum`). *Optional* hooks
have no default; calling the public method on an operator that does not
implement the hook raises `UnsupportedOpError`
({ref}`contract-unsupported`).

### `matvec`, `rmatvec`, `matmat`, `rmatmat`

`matvec(x)` applies the operator; `rmatvec(x)` applies its transpose.
Both are exact linear maps — no stochastic or iterative approximation.
`rmatvec` is required at the base level because the layer's main producers
of rectangular operators are `factor()` and cross-covariance blocks, and
both are useless downstream without their transposes (sampling applies
$L$, conditioning applies $L^\top$).

### `T`

`op.T` returns the transpose as an operator. Requirements:

- `op.T.shape == (n_in, n_out)`, `op.T.matvec == op.rmatvec` pointwise,
  and `op.T.to_dense()` equals `op.to_dense().T`.
- `op.T.T` behaves identically to `op`. The default view guarantees this
  by identity — transposing a `Transposed` returns the original object —
  while a structured override (`Triangular.T`) returns a new operator and
  guarantees it by equality of the dense forms.
- On `PSDLinOp`, `T` returns `self`.

The default implementation wraps the operator in a `Transposed` view that
swaps `matvec`/`rmatvec` and reverses `shape`. `Transposed` is a plain
`LinOp`: transposition does not preserve the layer's *knowledge* of
solvability, even though it preserves solvability itself. An operator whose
transpose supports more should override `T` to return a structured result —
`Triangular.T` returns a `Triangular` of the opposite orientation, which
keeps its `solve`.

`Transposed` is itself a public `@linop` operator with a single data
field, the wrapped operator. Its matrix hooks delegate to the wrapped
operator's `_rmatmat`/`_matmat`, so a fast matrix override is not lost
under transposition; its `_to_dense` transposes the wrapped dense form
(never touching `matvec`); its repr follows the composite rule; its
constructor validates nothing beyond the field. Constructing it directly
is allowed but — like all direct construction — never upgrades the level:
`densify` of a directly-wrapped PSD operator therefore returns a `Dense`.

### `to_dense`

Returns the operator as a dense `(n_out, n_in)` array.

- It **must not route through `matvec`/`rmatvec` or their hooks** — it
  may be built from stored arrays and static metadata by any independent
  code path (a spectral operator legitimately assembles an explicit DFT
  matrix from metadata), but never by applying the operator to an
  identity matrix. `to_dense` is the reference the conformance suite
  compares `matvec` against; an implementation routed through `matvec`
  would make that comparison vacuous. The suite enforces this mechanically
  ({ref}`contract-conformance`), not by convention.
- It has no size guard: it is the honest primitive that tests and
  {func}`densify` build on. The guard belongs to `densify`, the
  user-facing escape hatch.

### `solve`, `solve_mat`

`solve(b)` returns the exact solution of $A x = b$; `solve_mat(B)` of
$A X = B$. Preconditions: the operator is nonsingular. Like everything in
the layer, these are exact direct methods — an iterative solver would have
a tolerance parameter and a failure mode, which belong in the caller's
hands, not behind this interface.

### `logdet`

Returns $\log \lvert \det A \rvert$ as a **0-d real JAX array** — never a
Python float, which would fail on a tracer under `jit`; never complex, even
when an intermediate diagonalization produces complex values. For a
`PSDLinOp` the absolute value is inert and `logdet` is
$\log \det A$. The absolute value matters only for square non-PSD
operators (a `Triangular` with negative diagonal entries), and matches
`slogdet`'s convention of separating magnitude from sign.

### `diag`

Returns the diagonal as a `(n,)` array.

### `factor`

Returns an operator $L$ with $L L^\top = A$, of shape `(n, k)`. This is
the sampling interface: for standard normal `eps` of length `k`,
`L.matvec(eps)` has covariance $A$.

- $L$ satisfies the full `LinOp` contract, including `rmatvec` — so
  $L^\top v$ is always available, which pathwise conditioning requires.
- **No triangularity, squareness, or orientation is promised.** `k > n`
  means the operator is a sum of simpler pieces (low-rank-plus-diagonal
  factors as a horizontal stack); `k < n` means the operator is singular.
  A class that is singular *by construction* — its factor statically thin
  — must not implement `_solve` or `_whiten`: support is static, so this
  is a design rule for the class, not a runtime reaction to array values.
- The factorization is computed when the *operator* is constructed, not
  when `factor()` is called ({ref}`contract-jax`); `factor()` only wraps
  stored arrays, so it is cheap and stable across calls.

### `whiten`, `whiten_mat`

`whiten(x)` applies a fixed matrix $W$ satisfying $W A W^\top = I_n$, so
that data with covariance $A$ becomes uncorrelated with unit variance.
Precondition: $A$ is nonsingular.

- $W$ is **chosen by the implementation and fixed per instance**: every
  call to the same instance applies the same $W$. Beyond that it is
  unspecified — it need not be triangular, and it need not invert the $L$
  that `factor()` returns.
- Consequently `whiten(factor().matvec(eps))` is a standard normal vector
  **of length $n$** — equal to `eps` in distribution exactly when the
  factor is square, and never promised elementwise: for any valid
  whitener, $WL$ has orthonormal rows ($WL(WL)^\top = WAW^\top = I_n$) but
  need not be the identity, and for a wide factor ($k > n$) the two
  vectors do not even share a dimension. Downstream code must rely only on
  distributional identities and on the invariant
  $\lVert W x \rVert^2 = x^\top A^{-1} x$, which every valid whitener
  satisfies and which is what the conformance suite checks.
- `whiten` is a first-class optional operation, not a derived one, because
  for some structures the whitener is the cheap object and any factor of
  $A$ is expensive — for the one-dimensional exponential correlation, the
  whitener is bidiagonal while the Cholesky factor is a full triangle (see
  {doc}`design`). Deriving `whiten` from a triangular factorization would
  make those structures needlessly quadratic.

:::{warning}
The unspecified-$W$ freedom imposes one usage rule on consumers drawing
perturbations. A perturbation $\varepsilon$ used through the
whitened-space shortcut — writing the whitened perturbed residual as
$W(y - g_j) - \varepsilon_j$ directly — must **never** also be pushed
through `factor()` in the same update (say, to materialize
$y_j = g_j + L\varepsilon_j$ for storage). $WL$ is orthonormal-rowed but
not the identity, so mixing the two representations of the same
$\varepsilon$ corrupts the *joint* law of the update while every marginal
statistic still looks right — the layer's signature silent failure mode.
Choose one representation per $\varepsilon$.
:::

There is deliberately no `cholesky()` in the contract. Its two former
roles are covered: sampling by `factor()`, whitening by `whiten()`. A
guaranteed-triangular accessor cannot be honoured by exactly the operators
that matter (a block-diagonal factor is not triangular; the exponential
correlation's factor is dense), so the promise would be either broken or a
dense fallback in disguise. Operators for which the natural factor *is*
triangular simply return a `Triangular` from `factor()`, where callers can
detect it by type.

(contract-capabilities)=
## Capabilities

Not every operator implements every optional operation, and for composites
the answer depends on the instance: a block-diagonal operator can solve
only if all of its blocks can. The capability system makes support
queryable, with one load-bearing invariant:

> **For every operation the operator's type defines,
> `op.supports(name)` is `True` exactly when calling it completes without
> raising `UnsupportedOpError`.** For operations below the operator's
> level — which the type does not define at all — `supports` answers
> `False` without raising, and calling them fails with `AttributeError`,
> like any missing attribute.

Both directions matter. `supports` must not report an operation that then
raises, and must not deny an operation that works — including the derived
ones: an operator implementing `_solve` supports `solve_mat`, and
`supports("solve_mat")` must say so. The public methods gate on
`supports()` itself, so reporting and enforcement share one code path and
cannot drift apart.

- The known operation names are exactly the twelve operations of the
  method table: `matvec`, `rmatvec`, `matmat`, `rmatmat`, `to_dense`,
  `solve`, `solve_mat`, `logdet`, `diag`, `factor`, `whiten`,
  `whiten_mat`. `shape` and `T` are properties, not operations, and are
  not queryable.
- `supports(name)` accepts every known name regardless of the operator's
  level: `supports("solve")` on a rectangular operator returns `False`
  (the type does not even have the method), it does not raise. Generic
  code can probe any operator uniformly.
- An **unknown name raises `ValueError`**. Returning `False` would make a
  typo — `supports("choleksy")` — silently steer callers onto their dense
  fallback branch forever.
- Required operations report `True` always.
- A derived operation reports `True` exactly when its dependency does, or
  when the operator overrides the derived hook directly.
- Composites refine `supports` per instance by intersecting over their
  children. This is why `supports` is an instance method: class-level
  declarations cannot express it.
- `capabilities()` returns the frozen set of supported names among the
  seven operations that are not unconditionally available — `solve`,
  `solve_mat`, `logdet`, `diag`, `factor`, `whiten`, `whiten_mat` — for
  error messages and diagnostics. The five required operations, always
  available, are not listed.

There is no mechanism for withdrawing an inherited capability (no
`_WITHDRAWN` marker): by hierarchy rule 2, a class that must disown an
operation is at the wrong level. Instance-dependent support is the
composite `supports` override, which both reports *and* enforces, since
the public method checks it before dispatching to the hook.

Because dispatch on capabilities happens in Python over static information,
it is compatible with `jit`: branch on `supports()` *before* tracing, never
inside a traced conditional (see the trace-time warning below).

(contract-unsupported)=
## Unsupported operations raise

Calling an optional operation the operator does not support raises
`UnsupportedOpError`. There is **no silent dense fallback**: an accidental
$O(n^3)$ materialization is precisely the failure this layer exists to
prevent, so it must be visible at the call site. The exception:

- subclasses `NotImplementedError`;
- names the operator type, the operation, and the operator's actual
  capabilities, and points at the explicit fallback (`densify`);
- is picklable and reconstructible from its `args`, because forward-model
  evaluation runs in worker processes and exceptions cross that boundary
  by pickling. Concretely, it is constructed as
  `UnsupportedOpError(name, operator_type, capabilities)` — a string, a
  string, and a tuple of strings — so `type(e)(*e.args)` rebuilds it.

:::{warning}
The raise happens **at trace time**, and inside `jit` both branches of a
`lax.cond` are traced — so an unsupported call raises even from the branch
that would never execute. Gate on `op.supports(name)` in Python, outside
the traced conditional. This is a feature: support is static information,
and consulting it statically is the only implementation that composes with
`jit` at all.
:::

### The explicit fallback: `densify`

`densify(op, *, max_n=4096)` is the deliberate escape hatch. It
materializes the operator and returns a dense operator **at the same level
of the hierarchy**, so the advertised fallback always actually provides
the operation that just raised:

| `op` is a       | `densify(op)` returns | backed by            |
| --------------- | --------------------- | -------------------- |
| `PSDLinOp`      | `DensePSD`            | a Cholesky factor    |
| `SquareLinOp`   | `DenseSquare`         | an LU factorization  |
| `LinOp`         | `Dense`               | the array itself     |

- The size guard raises `ValueError` **before allocating** when either
  side exceeds `max_n`. Raising the limit is a deliberate act at the call
  site, which is the point.
- Densifying a *singular* PSD operator is a caller error: the Cholesky of
  a singular matrix is `nan` without an exception (see
  {ref}`contract-validation` for how debug mode catches this). Singular
  PSD operators are legitimate — as instances of classes that are singular
  *by construction* and therefore implement `factor` but not
  `_solve`/`_whiten` — and `densify` cannot manufacture what does not
  exist. Feeding a singular matrix to a class that *does* advertise
  `solve` (`DensePSD.from_matrix` of a rank-deficient array) is the other
  case: a tier-4 value violation, on which `supports` still answers `True`
  — support is static — and the result is `nan`.

`DenseSquare` is the LU-backed elementary square operator this mapping
requires.
`DenseSquare.from_matrix(A)` computes the LU factorization once and stores
`A` together with the factors; `to_dense` returns the stored `A`; it
supports `solve`, `solve_mat`, `logdet` ($\log\lvert\det\rvert$ from the
LU diagonal — pivot signs are irrelevant under the log-magnitude
convention), and `diag`; and it overrides `T` to return the transposed
operator backed by the same factorization, so transposition does not cost
it its `solve`.

(contract-validation)=
## Validation and errors

Validation is layered by *when it can run* and *what it may read*. The
governing principle: **everything static is checked always; values are
checked only on request**, because value checks read array contents, which
is impossible on tracers and forces a device sync on concrete arrays.

| tier | what is checked | when | cost | on failure |
| ---- | --------------- | ---- | ---- | ---------- |
| 1. class definition | every dataclass field is declared data or static correctly | `import` time | free | `TypeError` |
| 2. construction | structural validity: array ranks, block shape agreement, block types | every construction, including pytree unflatten | O(#fields) Python | `ValueError` / `TypeError` |
| 3. call | operand core shape matches the operator | every public method call | free (shapes are static) | `ValueError` |
| 4. value (debug) | positivity, finiteness, definiteness preconditions | opt-in, concrete arrays only | a device sync | `ValueError` |

Tier 3 exists because structured operators otherwise fail *silently*:
`Identity(6).matvec(ones(3))` would happily return the wrong-shaped array,
and `PSDDiagonal` would broadcast a length-1 operand. Precisely: the vector
methods require `ndim >= 1` and `shape[-1]` equal to the core length; the
matrix methods require `ndim >= 2` and `shape[-2]` equal to the core
length, with `shape[-1]` — the `k` axis, possibly 0 — unconstrained.
Violating either condition raises `ValueError`; operand dtypes are not
validated (the package runs float64 end to end). Shape checks are pure
Python over static information, so they cost nothing under `jit` and are
therefore unconditional. Error messages must name the operator (its
`repr`), the method, the expected core shape, and the offending shape.

Tier 2 has three rules imposed by JAX's pytree machinery, which
reconstructs operators through their storing constructor on every
`jit`/`vmap` boundary ({ref}`contract-jax`):

- **Shape-only.** Constructor validation may inspect `ndim`, `shape` and
  Python-level structure. It must never read values (tracers pass through
  constructors routinely).
- **Sentinel-tolerant, transitively.** Any field that does not expose
  `ndim` passes validation untouched: JAX internals unflatten pytrees with
  placeholder objects — `jax.custom_vjp`, for one, rebuilds arguments with
  bare `object()` leaves — and validation must not break them. The rule
  extends through composites: any check that reads a *child operator's*
  `shape` must first verify the child's stored arrays expose `ndim` and
  skip the check otherwise, because a child built around sentinel leaves
  constructs successfully but cannot answer `shape`.
- **Rank floor, not exact rank.** The storing constructor rejects arrays
  whose rank is *below* the field's core rank and accepts extra leading
  axes — a vmapped family, reconstructed at `vmap` exit boundaries with
  batched concrete leaves ({ref}`contract-batching-operators`). Exact core
  rank, and the positivity of core sizes, are enforced by the classmethods
  and factories, which are never on an unflatten path.

Tier-2 and tier-3 code runs in Python on every eager call, and — because
unflatten calls the constructor — tier 2 runs again on every
reconstruction of a `jit` *output*; under `jit` the call-site checks run
at trace time only. Both tiers must therefore stay O(#fields) of pure
Python, with no array work.

Tier 4 covers preconditions that are *values*: `PSDDiagonal` entries strictly
positive, `from_matrix` arguments actually positive definite, factors
finite. Outside debug mode these are the caller's responsibility, and
violating them yields `nan` (or `±inf` — `logdet` of a singular operator)
downstream rather than an exception — silently, which is why the debug
mode exists. Debug mode is a process-global boolean, toggled by
`set_debug_checks` and usable as the `debug_checks` context manager, with
`value_check` the helper that consults it inside constructors; it is
consulted when constructors and `from_matrix`-style classmethods execute —
call-time value checks are out of scope. `densify`'s singular-PSD hazard
is caught this way: `densify` routes through `from_matrix`, which in debug
mode asserts positive definiteness of the materialized array. On tracers,
all tier-4 checks are skipped.

The full error taxonomy:

| condition | raises |
| --------- | ------ |
| field neither array-like nor marked static | `TypeError`, at class definition |
| structural mismatch (wrong array rank, disagreeing block shapes) | `ValueError`, at construction |
| wrong block *type* (non-PSD block in a PSD composite) | `TypeError`, at construction |
| operand core shape mismatch | `ValueError`, at call |
| unsupported operation (type defines it, no cheap implementation) | `UnsupportedOpError`, at call/trace |
| operation below the operator's level (type does not define it) | `AttributeError`, at call — see {ref}`contract-capabilities` |
| unknown capability name | `ValueError`, at call |
| `densify` size guard | `ValueError`, before allocation |
| violated value precondition | `ValueError` in debug mode; `nan` otherwise |

(contract-jax)=
## JAX integration

### The `@linop` decorator

Every concrete operator is declared with the `@linop` class decorator,
which makes the class a frozen dataclass and registers it as a pytree with
explicitly separated data and metadata:

- **Field classification is an allowlist.** A field is a pytree *child*
  (data) if and only if its annotation is `Array`, a `LinOp` subtype, or a
  tuple of those. Every other field must be marked with `static_field()`;
  an unmarked field of any other annotation is a `TypeError` at class
  definition. The polarity matters: the failure mode of a misclassified
  field is a tracer arriving in shape arithmetic far from the declaration,
  so the default must be the safe side, with no heuristic denylist to have
  holes in. The rule's operating conditions: annotations must resolve at
  class-definition time in the defining module's runtime namespace (via
  `typing.get_type_hints`) — no `TYPE_CHECKING`-only names, no self- or
  forward-references. Optional children (`Array | None`) are
  unrepresentable: a field is either always an array/operator, or it is
  static. A `np.ndarray` annotation is a definition-time `TypeError` like
  any other non-allowlisted type — store JAX arrays.
- **Static metadata must be hashable and cheap to compare** — ints, bools,
  strings, tuples of those. Never arrays (a NumPy array as metadata
  poisons the compilation cache).
- `eq=False`: operators compare by identity. Dataclass equality would
  compare arrays elementwise and raise on the ambiguous truth value.
- `repr=False`: the generated dataclass `__repr__` would shadow the
  base-class one and print whole arrays into tracebacks and test ids. See
  {ref}`contract-repr`.

### Constructors store; classmethods compute

The dataclass constructor **must only store and validate** — never
factorize, never allocate more than trivially. Anything computed from the
inputs (a Cholesky factor, an eigendecomposition, LU) is done in an
alternate constructor (`DensePSD.from_matrix(A)`), which computes once and
passes the results to the storing constructor.

Two independent reasons, either sufficient:

- JAX **reconstructs pytrees through the constructor** on every
  `jit`/`vmap`/`grad` boundary. A constructor that computes would
  silently recompute at each of them.
- The lazy alternative — caching a factorization on first use — does not
  work at all under JAX: a cache written inside a traced function lands on
  a temporary copy and is discarded, so the operator re-factorizes on
  every call, silently.

Corollary: what `factor()` and `solve()` need must already be sitting in
the operator's fields when they are called.

### Identity, hashing, and `static_argnums`

Operators compare by identity and hash by identity. They are **never valid
`static_argnums`**: every call would hash to a fresh value and silently
retrace. Pass operators as ordinary traced arguments — the pytree
registration exists precisely so that this works. Static information that
callers may legitimately close over is the `shape` tuple and the results
of `supports()`.

### Float64

`import pyeki` enables JAX float64 for the process. Operators assume it:
the conditioning arithmetic this layer feeds loses several digits to
cancellation in float32. Worker processes do not inherit the setting and
need `JAX_ENABLE_X64=1` in their environment.

(contract-composites)=
## Constructing composites

A composite's hierarchy level depends on its contents: a block-diagonal of
PSD blocks is PSD, of mixed blocks merely square or rectangular; the
factors of a PSD block-diagonal form a block-diagonal that is not *known*
to be PSD, and is typed as a plain `LinOp`.
Since level membership is static (hierarchy rule 3) and the ingredients'
types and shapes are known at construction, **factory functions choose the
class**:

| factory                 | returns                                      |
| ----------------------- | -------------------------------------------- |
| `block_diag(*blocks)`   | `PSDBlockDiag` if every block is a `PSDLinOp`, else `BlockDiag` |
| `product(*ops)`         | `Product` (a square variant will be added when an EKI consumer needs `solve`/`logdet` through a product) |
| `hstack(*ops)`          | `HStack`                                     |
| `kron(A, B)`            | the PSD Kron variant if both children are `PSDLinOp`, else the general variant — the rectangular factor-Kron is the general one *(classes arrive with the Kron milestone)* |
| `diag_congruence(op, scale)`| `PSDDiagCongruence` (see below)             |

Variadic factories require at least one operand (`ValueError` otherwise);
applied to a single operand, `block_diag`, `product` and `hstack` return
that operand unchanged. Factories do not flatten nested composites.

Two further composites are constructed through operator arithmetic rather
than a named factory: `A @ B` is `product(A, B)`, and `c * op` builds a
level-preserving scaled operator ({ref}`contract-arithmetic`).

The factories are the stable public construction API: as structured
variants are added, factory calls transparently start returning them,
without breaking callers. The concrete classes remain public — they are
what `isinstance` checks and `factor()` return types are written against —
and constructing them directly is allowed, but a direct construction
validates only; it never upgrades to a more capable class.

Semantics fixed by this contract:

- `Product(ops)` applies right-to-left, like matrix multiplication;
  adjacent shapes must agree at construction. `HStack` blocks must share a
  row count, validated at construction.
- **Composite anatomy is contract, not implementation detail.** The
  children are a public data field — `blocks` on the block-diagonal
  classes, `ops` on `Product` and `HStack` — and the block classes expose
  the per-block shapes as a static `block_shapes` property, computed like
  `shape`. The consumer is domain localization, which must align
  sub-vectors of the observation space with the noise operator's blocks.
- `HStack(ops)` is a block **row** `[A_1 ... A_m]`: it splits the operand's
  trailing axis and sums the blocks' outputs. Its transpose — the block
  column — is `hstack(...).T`, whose `matvec` concatenates the blocks'
  `rmatvec` outputs; no separate `VStack` class is needed.
- `PSDBlockDiag` applies every operation blockwise; its `factor()` returns
  a general `BlockDiag` of the blocks' factors (rectangular blocks). Its
  `supports(name)` intersects over blocks, and the public-method gate
  enforces the same answer.
- Composites do not track definiteness through composition: a `Product` of
  PSD operators is not PSD in general, and the layer never infers PSD-ness
  from structure. An operator family that *is* closed under a composition
  gets its own class — which is exactly what `PSDDiagCongruence` is.

### Diagonal congruence

`diag_congruence(op, scale)` represents $\mathrm{diag}(s)\,A\,\mathrm{diag}(s)$
for a `PSDLinOp` $A$ and a strictly positive vector $s$ (a tier-4
precondition, like every value constraint), returning `PSDDiagCongruence`, a
`PSDLinOp` holding the scale as a 1-D data field (`scale`). Like scalar scaling, every
capability delegates to the base operator with the diagonal folded in:

| operation on $SAS$, $S = \mathrm{diag}(s)$ | in terms of $A$              |
| ------------------------------------------ | ---------------------------- |
| `matvec(x)`                                | $s \odot A(s \odot x)$       |
| `solve(b)`                                 | $A^{-1}(b / s)\, /\, s$      |
| `logdet()`                                 | $2\sum_i \log s_i + \log\det A$ |
| `diag()`                                   | $s^2 \odot \operatorname{diag}(A)$ |
| `factor()`                                 | $S L$, a `Product`           |
| `whiten(x)`                                | $A$'s whitening of $x / s$   |

`supports()` delegates to the base operator. The consumer is domain
localization: inflating each local observation's noise by the reciprocal
taper is the congruence with $s_i = 1/\sqrt{\tau_i}$, so inflated noise
whitens as cheaply as the original — including for correlated
(non-diagonal) noise blocks.

(contract-arithmetic)=
## Operator arithmetic

Operators support a deliberately small arithmetic surface: composition
with `@`, and scalar scaling with `*` and `/`. Both are sugar over
structured composites, both are unambiguous under the batch contract, and
neither accepts a raw array. Addition remains excluded
({ref}`contract-excluded`).

### Composition: `A @ B`

When **both operands are operators**, `A @ B` returns `product(A, B)` —
the lazy composition, with adjacent shapes validated at construction and
nothing multiplied out. It is exactly the factory call, so as structured
product variants are added, `@` picks them up transparently.

### `@` never accepts an array

`op @ x` and `x @ op` raise `TypeError` for array operands — deliberately,
and the dunders are defined so that the error is a *guided* one, naming
the methods to use instead (`matvec`/`matmat`, or `rmatvec`/`T` on the
right-hand side) and the batch convention. The reasoning:

- NumPy fixed what `@` means for operands with two or more dimensions:
  contract axis `-2`, treat leading axes as batches of *matrices*. So
  there is exactly one consistent array semantics available for `op @ X`,
  and it is `matmat`.
- That semantics collides with this layer's canonical data layout. A
  `vmap`-ed forward model produces ensembles as `(J, n)` row-batches, and
  applying an operator to one is a *batch-of-vectors* operation. Given
  `R` of side $n$, the natural-looking `R @ ensemble` is a confusing
  shape error when $J \ne n$ — and when $J = n$ it is shape-valid,
  contracts the wrong axis, and returns a wrong answer silently. The
  small-observation regime makes $J = n$ a case that actually occurs, not
  a corner.
- A `@` that works for 1-D operands and column matrices but misbehaves on
  the package's own preferred layout would be worse than no `@`: partial
  support teaches a habit, then punishes it where the cost is a wrong
  posterior. The named methods are total — `matvec` handles every batch
  rank uniformly — and their names force the caller to say which of the
  two shape-identical operations they mean.

Restricting `@` to operator–operator composition is also the convention of
the closest analogue among JAX operator libraries (lineax): the collision
above is a property of leading-batch layouts, not of this package.

For the guided errors to be *reachable*, the left operand must defer to
the reflected dunders. JAX arrays do — their binary operators return
`NotImplemented` for operands they do not recognize — but NumPy arrays do
not by default: `np_array * op` would silently build an **object array of
per-element scaled operators**, with no error at all. `LinOp` therefore
sets `__array_ufunc__ = None`, which makes NumPy defer as well. Operators
must never define `__jax_array__`: it would make `x @ op` silently
densify instead of deferring. The conformance suite pins the deferral
behaviour for both array libraries.

### Scalar scaling: `c * op`, `op * c`, `op / c`

Scalar multiplication returns a **scaled composite at the same hierarchy
level as its operand** — `PSDScaled`, `SquareScaled`, or `Scaled`,
selected by the operand's level — delegating every capability of the base
operator with the scalar folded in:

| operation on $cA$ | in terms of $A$                              |
| ----------------- | -------------------------------------------- |
| `matvec(x)`       | $c \cdot A x$                                |
| `solve(b)`        | $A^{-1} b \,/\, c$                           |
| `logdet()`        | $n \log\lvert c\rvert + \log\lvert\det A\rvert$ |
| `diag()`          | $c \cdot \operatorname{diag}(A)$             |
| `factor()`        | $\sqrt{c}\, L$, as a scaled factor           |
| `whiten(x)`       | $A$'s whitening of $x$, divided by $\sqrt{c}$ |

Requirements:

- The scalar is held as a **0-d array field** (pytree data), never a
  Python float, so a traced value flows through it. This is the point:
  the consumer is tempering, where the per-step noise covariance is
  $\Sigma / \Delta\beta_t$ with an increment chosen adaptively — and
  therefore traced — inside the step. Scaling must not rebuild or
  re-factorize the base operator.
- `supports()` delegates to the base operator: the scaled composite
  supports exactly what its base supports.
- As with the other composites ({ref}`contract-composites`), the concrete
  class is selected by the operand's level — the dunder plays the factory
  role — so scaling a `PSDLinOp` yields a `PSDLinOp`, and scaling a
  factor yields a plain `LinOp`.
- The dunders accept Python and NumPy real scalars and 0-d arrays,
  converting to a 0-d `jnp` array before storing; anything with
  `ndim > 0` gets the guided error below.
- Scaling an already-scaled operator folds the scalars into a single
  wrapper rather than nesting. The repr follows the composite rule (type
  and shape).
- Value preconditions are tier 4 ({ref}`contract-validation`): $c > 0$
  when the operand is PSD (a traced sign cannot be checked eagerly),
  $c \ne 0$ when `solve` is used. Violations produce `nan`, or an error
  in debug mode, like every other value precondition.
- Only true scalars scale: `array * op` for a non-0-d array is a guided
  `TypeError`, for the same reason `@` rejects arrays — elementwise and
  batched readings would be ambiguous.

There is no unary negation, and `op1 * op2` raises a guided `TypeError`
naming `@`: the first cannot preserve any level above `SquareLinOp`, and
the second already has a name.

(contract-repr)=
## `repr`

`repr(op)` is the type name and shape — `Dense(200, 300)`,
`PSDBlockDiag(5, 5)` — and never includes array contents. Reprs appear in
tracebacks, pytest ids, and error messages of this very contract, where a
dumped array would bury the signal. Composites may append a summary of
their structure (block count) but never recurse into children's arrays.

(contract-surface)=
## Public surface

For the avoidance of doubt, `pyeki.linalg` exports exactly: the levels
`LinOp`, `SquareLinOp`, `PSDLinOp`; the elementary operators `Identity`,
`PSDDiagonal`, `Dense`, `DenseSquare`, `Triangular`,
`DensePSD`; the composites `Product`, `HStack`, `BlockDiag`,
`PSDBlockDiag`, `Transposed`, `Scaled`, `SquareScaled`, `PSDScaled`,
`PSDDiagCongruence`; the factories `block_diag`, `product`, `hstack`, `kron`
(with the Kron classes, once that milestone lands), `diag_congruence`; the
helpers `dense_matvec` and `tri_solve`; `densify`, `UnsupportedOpError`,
`linop`, `static_field`, and the debug switch (`set_debug_checks`, the
`debug_checks` context manager, and the `value_check` helper it gates). The
conformance suite lives in `pyeki.linalg.testing`. Anything else is private, and no consumer may
depend on it.

(contract-conformance)=
## Conformance

`pyeki.linalg.testing.check_operator(op)` is the executable form of this
contract. **Every new operator type must pass it**, on an instance small
enough to densify, before it is merged. It must verify at least:

1. **Dense agreement at batch ranks 0, 1, 2** for `matvec` and `rmatvec`,
   with distinct random operands per rank (a shared operand can mask
   rank-dependent bugs).
2. **Matrix siblings**: `matmat`, `rmatmat` and `solve_mat` against the
   dense reference, and `whiten_mat` against columnwise `whiten` (the
   column identity of the batch contract — a dense reference for
   `whiten_mat` does not exist, since the contract fixes no particular
   $W$), each batched and unbatched.
3. **Transpose**: `op.T` itself passes checks 1–2, with its dense
   reference the dense transpose, and `op.T.T` matches `op`'s dense form.
   (A structured `T` override with a correct `_to_dense` and a wrong-axis
   `_matvec` must not slip through on the dense comparison alone.)
4. **Solve**: `solve` against the dense inverse at batch ranks 0, 1 and 2,
   when claimed.
5. **Square roots**: `factor()` returns an `L` whose dense form satisfies
   $L L^\top = A$, and `L` itself passes the `LinOp` checks (including
   `rmatvec`).
6. **Whitening**: with $W$ recovered by applying `whiten` to the columns
   of $I_n$, (a) $W A W^\top \approx I_n$, and (b) `whiten(x)` agrees
   **elementwise** with $Wx$ on random operands at batch ranks 0, 1 and 2.
   The elementwise form is deliberate: it pins linearity, per-instance
   fixedness of $W$, and batch-rank behaviour at once, and it implies
   $\lVert Wx\rVert^2 = x^\top A^{-1} x$ (for square invertible $W$,
   $W^\top W = A^{-1}$) — whereas the norm identity alone is satisfied by
   maps that are not a fixed matrix at all. No comparison against any
   particular factorization — the contract does not promise one.
7. **Scalars**: `diag` and `logdet` against the dense reference; `logdet`
   is a 0-d real JAX array.
8. **`to_dense` independence, enforced**: with the class-level `matvec`,
   `_matvec`, `matmat` and `_matmat` all temporarily replaced by raising
   stubs (instances are frozen, so the patch must be on the class),
   `to_dense()` still succeeds. (A check that merely inspects where
   `to_dense` is *defined* can be satisfied by an implementation that
   routes through `matvec` — or through `matmat` — which would make every
   dense comparison above compare `matvec` with itself.)
9. **Capability honesty, both directions, at the operator's level**: every
   operation with `supports(name) == True` runs without
   `UnsupportedOpError`; every type-defined operation with
   `supports(name) == False` raises it; operations below the operator's
   level are absent from the type (`hasattr` is `False`) and `supports`
   answers `False` for them. Operands are synthesized at the core shapes
   of the method table, with `k = 3`. This is the invariant of
   {ref}`contract-capabilities` made mechanical.
10. **Operand validation**: for each operand-taking method, an operand
    whose *contracted* axis has the wrong length — axis `-1` for the
    vector methods, axis `-2` for the matrix methods — raises
    `ValueError`, as does an operand of insufficient rank. The
    uncontracted `k` axis is unconstrained and must not raise.
11. **Pytree round trip**: flatten/unflatten preserves type and behaviour;
    unflattening with bare `object()` sentinel leaves succeeds for every
    operator type, composites included; the operator works under `jit`;
    `vmap` over the *operand* agrees with native batching; constructing
    and returning the operator inside `jax.vmap` round-trips (the
    vmap-exit reconstruction of {ref}`contract-batching-operators`), and
    `vmap` over the resulting family agrees with a Python loop; `grad` of
    a fixed scalar (the sum of `matvec` on a fixed operand) through array
    leaves returns the same tree structure.
12. **Repr hygiene**: `repr(op)` matches the type-and-shape form and
    contains no array data.
13. **Arithmetic dispatch**: JAX and NumPy left operands defer to the
    reflected dunders — `x @ op` and `array * op` raise the guided
    `TypeError`, and `scalar * op` returns the scaled composite — pinning
    the `__array_ufunc__ = None` deferral the guided errors depend on.

The suite skips what `supports()` disclaims (that is check 9's other
half), so the same driver applies to every operator type unchanged.

Beyond conformance, the test suite keeps *targeted regression tests* — one
per class of bug that produces wrong numbers without raising (the
wrong-axis contraction, the vacuous `to_dense`, the discarded lazy cache).
These encode why the contract's rules exist and must not be deleted as
redundant with conformance.

(contract-excluded)=
## Deliberately excluded

Recorded so their absence reads as a decision, not an oversight.

**Operator addition (`+`).** Composition and scalar scaling are supported
({ref}`contract-arithmetic`); addition is not. A sum is only worth
representing when its structure survives — the factor of a sum of PSD
operators is a horizontal stack, but its `solve` and `logdet` require
either a simplification rule per pair of types (two `PSDDiagonal`s,
low-rank plus diagonal) or a dedicated class per structured sum. A
registry of such rules is machinery the current type count does not
justify, and a generic sum class would advertise almost nothing. Structured
sums get their own classes as EKI needs them; revisit `__add__` when
`pyeki.gauss` exists and real call sites are visible.

**`@` between an operator and an array.** Excluded with a guided error;
the reasoning is in {ref}`contract-arithmetic`.

**`cholesky()`.** Removed from the contract; see the square-roots section
for the reasoning.

**Batched operators.** One operator holds one matrix; families are
`vmap`ed pytrees ({ref}`contract-batching-operators`). Supporting stored
batch axes would force every `shape`, `split`, and validation rule to
carry a second convention through the whole layer. (The storing
constructor's tolerance of extra leading axes is `vmap` plumbing only —
the contract defines no direct-call behaviour for such a family.)

**dtype tracking.** The package runs float64 end to end (enabled at
import). A per-operator dtype attribute and promotion rules would be
machinery without a consumer; `to_dense()` answers the question where it
arises. Revisit if mixed precision ever becomes a requirement.

**Iterative and matrix-free solves.** `solve` is exact and direct.
Iterative methods have tolerances, preconditioners and failure modes that
belong to the caller; hiding them behind the same method name as an exact
solve would make `solve`'s contract untestable.

**In-place or mutating operations.** Operators are frozen; every method
returns new arrays. This is the only sane convention under JAX.

**A general operator algebra.** The layer grows one structure at a time,
when EKI needs it. The catalogue of shipped operators lives in
{doc}`user-guide/operators`; this contract constrains *how* any of them
behave, not *which* exist.

(contract-changes)=
## Appendix: changes from the implemented layer

:::{admonition} Temporary section
:class: note

This appendix exists while the implementation is brought up to this
specification, and will be deleted afterwards.
:::

| area | implemented today | specified here |
| ---- | ----------------- | -------------- |
| naming | `PSDOperator`; `@operator` (shadows the stdlib module); `Diagonal`/`ScaledIdentity` despite their positivity preconditions | `PSDLinOp`; `@linop`; `PSDDiagonal` — generic mathematical names are reserved for unrestricted classes. `ScaledIdentity` is dropped entirely: `c * Identity(n)` is the same operator with the same costs |
| implementation surface | subclasses override public methods; base classes hold raising defaults, detected by comparing against a snapshot | subclasses implement `_`-prefixed hooks; public methods validate, gate on `supports`, and dispatch |
| transpose | none; `factor()`'s result cannot be transposed without densifying | `rmatvec`/`rmatmat` required; `T` on every operator; `Transposed` view |
| `cholesky()` | in the PSD interface; contract already unhonourable for `BlockDiag` (returns a non-triangular, non-solving factor) | removed; `factor()` + `whiten()` cover both roles |
| `whiten` | derived from `cholesky().solve()` | primitive optional hook; contract is $W A W^\top = I$ for a fixed, otherwise unspecified $W$; `whiten_mat` added |
| `supports()` | reports `False` for working derived methods (`solve_mat`, `whiten`); returns `False` for unknown names | derived operations resolve through their dependencies; unknown names raise `ValueError`; invariant "supports ⟺ succeeds" enforced by the public-method gate |
| `_WITHDRAWN` | class variable consulted by `supports` but unenforced, and unused | removed; monotone-capability rule replaces it |
| unsupported-op error | not reconstructible from `args`; not picklable | picklable, rebuilds from `args` |
| `densify` of a square non-PSD operator | returns `Dense`, which has no `solve` — the advertised fallback fails | returns `DenseSquare` (new LU-backed elementary operator); mapping preserves the hierarchy level |
| operand validation | none; `Identity(6).matvec(ones(3))` returns the wrong shape silently | tier-3 core-shape checks on every public method |
| constructor validation | none; batched arrays accepted then misbehave | tier-2 structural checks — exact core rank in classmethods/factories, a rank floor in the storing constructor (vmap-exit reconstruction requires it); `vmap`-over-pytree is the batching story |
| value preconditions | documented only; violations yield `nan` | opt-in debug mode checks them on concrete inputs |
| field classification | denylist heuristic over annotation strings (misses `int \| None`, `Optional[int]`, `Literal`, `np.ndarray`) | allowlist: data iff `Array` / `LinOp` / tuples thereof; everything else must be static |
| abstractness | missing required methods surface on first use | `LinOp` is an ABC; instantiation fails |
| composites | direct class construction; `BlockDiag` + `BlockDiagGeneral` pair, the latter with a stub `solve` raising `AttributeError` | factories `block_diag`/`product`/`hstack` select the class; general `BlockDiag` is a plain `LinOp` with no stub methods |
| arithmetic | none; `op @ x` fails with Python's generic unsupported-operand error; `np_array * op` silently builds an object array | `@` composes operators (guided `TypeError` on arrays); `*`/`/` return a level-preserving scaled composite; `__array_ufunc__ = None` so NumPy defers; `__jax_array__` forbidden; `+` still excluded |
| composite anatomy | `blocks`/`ops` fields exist but are implementation detail | children and static block sizes are contract ({ref}`contract-composites`); `PSDDiagCongruence` added for localization noise inflation |
| conformance | `to_dense` independence checked by definition site (bypassable); one PRNG key reused; no transpose, capability-honesty, operand-validation, or vmap-over-operator checks | checks 1–13 above, with the monkeypatch guard and per-check keys |
