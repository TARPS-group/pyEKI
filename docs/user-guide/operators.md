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
| `Kron(A, B)` | $A \otimes B$, PSD factors | operations applied factorwise |
| `KronGeneral(A, B)` | $A \otimes B$, any factors | what `Kron.factor()` returns |

`HStack` splits its input along the trailing axis and sums the blocks' outputs,
so that $[A_1\ A_2]\,[x_1; x_2] = A_1 x_1 + A_2 x_2$. It does not apply each
block to the whole input.

## Kronecker products

`Kron(A, B)` represents $A \otimes B$, a covariance that separates into one
factor per index of the problem. The typical use is a quantity indexed by two
things at once, with a covariance over each: $A$ of side $n_A$ over the first
index and $B$ of side $n_B$ over the second, giving a covariance of side
$n_A n_B$.

Representing that as a `Kron` rather than as its dense $n_A n_B$ matrix is
worth it because every operation reduces to the same operation on each factor:

| operation | closed form | cost |
| --- | --- | --- |
| `matvec` | apply each factor along its own axis | $O(n_A n_B (n_A + n_B))$ |
| `solve` | $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$ | the two factors' solves |
| `logdet` | $n_B \log\det A + n_A \log\det B$ | the two factors' logdets |
| `diag` | $\mathrm{diag}(A) \otimes \mathrm{diag}(B)$ | $O(n_A n_B)$ |
| `factor` | $\mathrm{factor}(A) \otimes \mathrm{factor}(B)$ | the two factors' factors |

Nothing of side $n_A n_B$ is ever formed or factorized. Two $500 \times 500$
factors give a covariance with 250 000 rows that is still solved and whitened
by working only on the factors.

`Kron` requires both factors to be square and PSD, and raises in the
constructor otherwise.

### Index ordering

The **first factor is the slow index**. The entry of `Kron(A, B)` at row
$i\,n_B + k$ and column $j\,n_B + l$ is $A_{ij}B_{kl}$ — block $(i,j)$ of the
result is $A_{ij}B$. This is the ordering `numpy.kron` uses. An operand is
therefore laid out with the *second* factor's index varying fastest, so that
reshaping its trailing axis to `(n_A, n_B)` gives rows indexed by $A$ and
columns by $B$.

:::{warning}
Reversing the two factors is not a loud error. `Kron(B, A)` is a different
matrix, but it is positive definite whenever `Kron(A, B)` is, and when the
factors have the same size it has the same shape too — so a reversed argument
order yields a perfectly valid covariance with the wrong meaning, and nothing
raises. If your two factors are the same size, no shape check anywhere will
catch it. Confirm the ordering once against a small dense `numpy.kron`.
:::

### Rectangular Kronecker products

`Kron.factor()` returns a `KronGeneral`, not a `Kron`, because a square root
of a Kronecker product is the Kronecker product of the factors' square roots
and those need not be square:

```python
op = Kron(A, B)          # (n_A n_B, n_A n_B)
L = op.factor()          # (n_A n_B, k_A k_B) -- a KronGeneral
```

`KronGeneral` accepts factors of any shape and provides `matvec`, `matmat` and
`to_dense` only. Its operand has core shape $(k_A, k_B)$ while its result has
core shape $(n_A, n_B)$, and these differ whenever either factor is
rectangular — which is why it is a separate class rather than a flag on
`Kron`.

`Kron.cholesky()` also returns a `KronGeneral`. The Kronecker product of two
lower-triangular matrices genuinely is the lower Cholesky factor of the
product, but the returned type does not advertise a `solve`, so use
`whiten(x)` rather than `cholesky().solve(x)` — `Kron` implements it on the
factors directly.

## Conditional support

A composite's capabilities depend on its contents. A `BlockDiag` supports
`solve` only if every block does, and a `Kron` only if both factors do, so
check before calling:

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
nonzeros; for `Kron`, $n = n_A n_B$:

| operator | `matvec` | `solve` | `logdet` |
| --- | --- | --- | --- |
| `Identity`, `ScaledIdentity` | $O(n)$ | $O(n)$ | $O(1)$ |
| `Diagonal` | $O(n)$ | $O(n)$ | $O(n)$ |
| `DensePSD` | $O(n^2)$ | $O(n^2)$ | $O(n)$ after the constructor's $O(n^3)$ |
| `Triangular` | $O(n^2)$ | $O(n^2)$ | $O(n)$ |
| `BlockDiag` | sum over blocks | sum over blocks | sum over blocks |
| `Kron` | $O(n(n_A + n_B))$ | the factors' solves | the factors' logdets |
| `Product`, `HStack`, `KronGeneral` | sum over factors | — | — |
