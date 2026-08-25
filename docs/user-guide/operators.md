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
genuinely singular, so it supports neither `solve` nor `whiten`.

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
| block diagonals | sum over blocks | sum over blocks | sum over blocks | sum over blocks |
| `PSDDiagCongruence`, scaled operators | base $+ O(n)$ | base $+ O(n)$ | base $+ O(n)$ | base $+ O(n)$ |
| `Product`, `HStack` | sum over factors | — | — | — |
