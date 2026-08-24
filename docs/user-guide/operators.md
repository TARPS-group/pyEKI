# Operator catalogue

## The hierarchy

Operators form three levels. Each adds the operations that become well defined
at that level, so an operator never advertises something meaningless — a
rectangular matrix has no `solve` at all, rather than a `solve` that raises.

| class | represents | adds |
| --- | --- | --- |
| `LinOp` | any linear map, possibly rectangular | `matvec`, `matmat`, `to_dense` |
| `SquareLinOp` | a square map | `solve`, `solve_mat`, `logdet`, `diag` |
| `PSDOperator` | a symmetric positive semi-definite map | `factor`, `cholesky`, `whiten` |

## Operators defined by their own arrays

| class | represents | notes |
| --- | --- | --- |
| `Identity(size)` | $I_n$ | every operation is free |
| `ScaledIdentity(c, size)` | $cI_n$ | `c` is an array, so it can be differentiated |
| `Diagonal(d)` | $\mathrm{diag}(d)$ | all operations linear in $n$ |
| `Dense(A)` | an explicit array | may be rectangular; no structure assumed |
| `Triangular(L, lower)` | a triangular matrix | what `cholesky()` returns |
| `DensePSD(L)` | a dense PSD matrix | stored as its Cholesky factor |

`DensePSD` is built with `DensePSD.from_matrix(A)`, which factorizes once at
construction. Operators never factorize lazily on first use, because a cached
factor computed inside a traced function is discarded when the trace ends,
which would silently re-factorize on every call.

## Operators built from other operators

| class | represents | notes |
| --- | --- | --- |
| `Product(ops)` | $A_1 A_2 \cdots A_m$ | applied right to left |
| `HStack(ops)` | $[A_1\ A_2\ \cdots\ A_m]$ | a block *row*: splits the input and sums |
| `BlockDiag(blocks)` | block diagonal, PSD blocks | operations applied blockwise |
| `BlockDiagGeneral(blocks)` | block diagonal, any blocks | what `BlockDiag.factor()` returns |

`HStack` splits its input along the trailing axis and sums the blocks' outputs,
so that $[A_1\ A_2]\,[x_1; x_2] = A_1 x_1 + A_2 x_2$. It does not apply each
block to the whole input.

## Conditional support

A composite's capabilities depend on its contents. A `BlockDiag` supports
`solve` only if every block does, so check before calling:

```python
if cov.supports("solve"):
    x = cov.solve(b)
else:
    x = densify(cov).solve(b)
```

`op.capabilities()` returns everything an operator supports.

## Square roots

Two related methods, with different guarantees:

- **`factor()`** returns any `L` with `L @ L.T == op`, of shape `(n, k)`. It
  need not be square. Use it to draw samples.
- **`cholesky()`** returns the square triangular `L`. Available only when the
  factor is square.
- **`whiten(x)`** returns `L^-1 x`. Prefer it over `cholesky().solve(x)` — it
  does not require the factor to be triangular, only invertible.

The shape of `factor()` is informative. `k > n` means the operator is a sum of
simpler pieces, as in low-rank-plus-diagonal. `k < n` means it is genuinely
singular, so it will have no `solve`.

## Cost summary

For an operator of side $n$ with $r$ the rank and $\eta$ the number of
nonzeros:

| operator | `matvec` | `solve` | `logdet` |
| --- | --- | --- | --- |
| `Identity`, `ScaledIdentity` | $O(n)$ | $O(n)$ | $O(1)$ |
| `Diagonal` | $O(n)$ | $O(n)$ | $O(n)$ |
| `DensePSD` | $O(n^2)$ | $O(n^2)$ | $O(n)$ after the constructor's $O(n^3)$ |
| `Triangular` | $O(n^2)$ | $O(n^2)$ | $O(n)$ |
| `BlockDiag` | sum over blocks | sum over blocks | sum over blocks |
| `Product`, `HStack` | sum over factors | — | — |
