# Operator catalogue

## The hierarchy

Operators form three levels. Each level is a mathematical claim about the
map, and adds the operations that claim makes well defined — so an operator
never advertises something meaningless: a rectangular matrix has no `solve`
at all, rather than a `solve` that raises.

| class | represents | adds |
| --- | --- | --- |
| `LinOp` | any linear map, possibly rectangular | `matvec`, `rmatvec`, `matmat`, `rmatmat`, `T`, `to_dense` |
| `SquareLinOp` | a square map | `solve`, `solve_mat`, `logdet`, `diag` |
| `PSDLinOp` | a symmetric positive semi-definite map | `factor`, `whiten`, `whiten_mat` |

`rmatvec` applies the transpose, and `op.T` returns the transpose as an
operator. The full behavioural specification is the
{doc}`../linop-contract` reference page.

## Operators defined by their own arrays

| class | represents | notes |
| --- | --- | --- |
| `Identity(size)` | $I_n$ | every operation is free |
| `PSDDiagonal(diagonal)` | $\mathrm{diag}(d)$ | all operations linear in $n$ |
| `Dense(A)` | an explicit array | may be rectangular; no structure assumed |
| `DenseSquare.from_matrix(A)` | a dense square matrix | stored with its LU; what `densify` returns for square non-PSD operators |
| `Triangular(L, lower)` | a triangular matrix | what `DensePSD.factor()` returns |
| `DensePSD.from_matrix(A)` | a dense PSD matrix | stored as its Cholesky factor |
| `PSDLowRank(F)` | $FF^\top$ for a factor $F$ of shape $(n, k)$ | singular when $k < n$; provides `diag` and `factor` only |

`PSDLowRank` imposes no relation between $n$ and $k$, and computes nothing
at construction: the stored factor *is* the factorization, so `factor()`
hands it straight back as a `Dense`. It withholds `solve`, `whiten` and
`logdet` at *every* width — forced when $k < n$, where the operator is
singular by construction, and a deliberate choice when $k \ge n$, since
capabilities belong to the type and no shape can rule out a rank-deficient
wide factor. Densify an instance you know to be full rank if you need them.

`DensePSD` and `DenseSquare` are built with `from_matrix`, which factorizes
once at construction. Operators never factorize lazily on first use, because
a factor cached inside a traced function is discarded when the trace ends,
which would silently re-factorize on every call.

## Operators built from other operators

Build composites through the factory functions, which pick the most capable
class for the ingredients:

| factory | returns | represents |
| --- | --- | --- |
| `block_diag(*blocks)` | `PSDBlockDiag` if every block is PSD, else `BlockDiag` | a block-diagonal matrix |
| `product(*ops)` | `Product` | $A_1 A_2 \cdots A_m$, applied right to left |
| `hstack(*ops)` | `HStack` | $[A_1\ A_2\ \cdots\ A_m]$, a block *row* |
| `diag_congruence(op, scale)` | `PSDDiagCongruence` | $\mathrm{diag}(s)\,A\,\mathrm{diag}(s)$ for PSD $A$ |

`HStack` splits its input along the trailing axis and sums the blocks'
outputs, so $[A_1\ A_2]\,[x_1; x_2] = A_1 x_1 + A_2 x_2$; its transpose
`hstack(...).T` is the corresponding block column. The block-diagonal
classes expose their children as `op.blocks` and their shapes as
`op.block_shapes`.

## Operator arithmetic

```python
C = A @ B            # composition: product(A, B)
R_t = R / dbeta      # tempered noise: a scaled operator, same capabilities
S = 2.0 * A          # scaling preserves the hierarchy level
At = A.T             # the transpose, as an operator
```

`@` composes **operators only**. Applying an operator to an array is always
`matvec`/`matmat` — `op @ x` raises a `TypeError` that says so, because
NumPy's `@` contracts axis `-2`, which is silently wrong for the
leading-batch vector layout everything in pyEKI uses. Scalars for `*` and
`/` may be traced values, which is what tempering needs.

(operator-batches)=
## Batches of operators

A batch of operators — one covariance per ensemble member, say — is built
with `jax.vmap` over the constructor, and used with `jax.vmap` over the
operator argument:

```python
covs = jax.vmap(DensePSD.from_matrix)(As)          # As: (100, n, n)
outs = jax.vmap(lambda C, x: C.solve(x))(covs, xs) # xs: (100, n)
```

What `vmap` hands back is a *vmapped family*: a single operator object
whose stored arrays carry an extra leading axis. A family identifies
itself — `covs.batch_shape` is `(100,)` and its repr reads
`vmapped(DensePSD(3, 3), batch=(100,))` — and it is deliberately **inert**:
calling any operation on it directly, or scaling or composing it, raises a
`ValueError` telling you to apply it under `jax.vmap`, because outside of
`vmap` there is no defined way to line its members up with your data.
Passing arrays with extra leading axes to a constructor does *not* build a
family; it is rejected outright.

## Debugging value preconditions

Some requirements are about values, not shapes: `PSDDiagonal` entries must
be positive, `DensePSD.from_matrix` needs a symmetric positive-definite
matrix. JAX cannot check values inside `jit`, so by default a violation
produces `nan` or `inf` downstream rather than an error. When a `nan`
appears and you want to find where, turn on debug checks:

```python
from pyeki.linalg import debug_checks

with debug_checks():
    cov = DensePSD.from_matrix(A)   # raises here if A is not symmetric PD
```

`set_debug_checks(True)` enables them process-wide. The checks run only on
concrete arrays and are skipped on traced values, so enabling them never
changes `jit`-ed behaviour.

## Conditional support

A composite's capabilities depend on its contents. A block-diagonal operator
supports `solve` only if every block does, so check before calling:

```python
if cov.supports("solve"):
    x = cov.solve(b)
else:
    x = densify(cov).solve(b)
```

`op.capabilities()` returns everything an operator supports beyond the
always-available operations. An unknown name raises `ValueError`, so a typo
cannot silently steer you onto the dense branch.

## Square roots and whitening

Two related operations, with different guarantees:

- **`factor()`** returns an operator `L` with `L @ L.T == op`, of shape
  `(n, k)`. Use it to draw samples: `L.matvec(eps)` for standard normal
  `eps` of length `k` has covariance `op`, and `L.rmatvec` applies `L.T`.
- **`whiten(x)`** applies a fixed matrix `W` with `W @ op @ W.T == I`, so
  data with covariance `op` becomes uncorrelated with unit variance.
  `whiten_mat` is the matrix-operand form.

The shape of `factor()` is informative. `k > n` means the operator is a sum
of simpler pieces, as in low-rank-plus-diagonal. `k < n` means it is
genuinely singular, so it supports neither `solve` nor `whiten` —
`PSDLowRank` is the shipped operator in that position.

The whitener is *not* promised to invert the factor: `whiten(L.matvec(eps))`
agrees with `eps` in distribution, never elementwise. Use one representation
per random draw.

## Cost summary

For an operator of side $n$:

| operator | `matvec` | `solve` | `whiten` | `logdet` |
| --- | --- | --- | --- | --- |
| `Identity` | $O(n)$ | $O(n)$ | $O(n)$ | $O(1)$ |
| `PSDDiagonal` | $O(n)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| `DensePSD`, `DenseSquare` | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | $O(n)$ after the constructor's $O(n^3)$ |
| `Triangular` | $O(n^2)$ | $O(n^2)$ | — | $O(n)$ |
| `PSDLowRank` (factor width $k$) | $O(nk)$ | — | — | — |
| block diagonals | sum over blocks | sum over blocks | sum over blocks | sum over blocks |
| `PSDDiagCongruence`, scaled operators | base $+ O(n)$ | base $+ O(n)$ | base $+ O(n)$ | base $+ O(n)$ |
| `Product`, `HStack` | sum over factors | — | — | — |
