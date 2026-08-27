# Quickstart

This page introduces the operator layer, which everything else in pyEKI is
built on. It takes about ten minutes. For the conditioning layer above it, see
{doc}`conditioning`.

## Why operators rather than arrays

EKI spends most of its time doing three things with covariance matrices:
applying them, solving against them, and taking square roots in order to draw
samples and to whiten residuals. For realistic problems those matrices are far
too large to store, but they are rarely arbitrary — observation error is
typically block diagonal, priors are often Kronecker-structured, and ensemble
covariances are low rank by construction.

An **operator** represents a matrix by how it acts, so that structure is
preserved and used. A `PSDDiagonal` of length one million costs one million
numbers and applies in linear time; the equivalent dense array does not fit in
memory.

## Creating operators

```python
import pyeki  # enables float64; import this before creating arrays
import jax.numpy as jnp
from pyeki.linalg import PSDDiagonal, DensePSD, Identity

d = PSDDiagonal(jnp.array([1.0, 4.0, 9.0]))
d.shape          # (3, 3)
d.matvec(jnp.ones(3))    # Array([1., 4., 9.])
d.logdet()               # Array(3.583..., dtype=float64)
```

Every operator exposes the same core interface, so code written against it
works regardless of the structure underneath:

```python
def log_density_quadratic_term(cov, residual):
    """Compute r^T C^-1 r without knowing how `cov` is stored."""
    whitened = cov.whiten(residual)
    return -0.5 * jnp.sum(whitened ** 2)
```

## Array shapes

Every method takes **leading batch axes**, with the operand's core shape
trailing. This is the same rule NumPy's `matmul` follows, and it is what
`jax.vmap` produces by default.

| method | core operand | signature |
| --- | --- | --- |
| `matvec` | vector `(n_in,)` | `(..., n_in) -> (..., n_out)` |
| `rmatvec` | vector `(n_out,)` | `(..., n_out) -> (..., n_in)` |
| `matmat` | matrix `(n_in, k)` | `(..., n_in, k) -> (..., n_out, k)` |
| `solve` | vector `(n,)` | `(..., n) -> (..., n)` |
| `solve_mat` | matrix `(n, k)` | `(..., n, k) -> (..., n, k)` |

So an ensemble of `J` parameter vectors, stored `(J, n)`, is simply a batch:

```python
ensemble = jnp.ones((100, 3))     # 100 members
d.matvec(ensemble).shape          # (100, 3)
```

Use `matvec` for a batch of vectors and `matmat` for a single matrix operand.
The `k` in `matmat` is part of the core shape, not a batch axis, and neither
method infers which you meant from the number of dimensions.

A batch of *operators* is a different thing from a batch of operands, and it
is built with `jax.vmap` — see {ref}`operator-batches` in the catalogue.

## Composing operators

Observation error covariances are usually block diagonal — independent errors
in one data stream, correlated errors in another:

```python
from pyeki.linalg import block_diag

noise = block_diag(
    PSDDiagonal(jnp.array([0.5, 0.5, 2.0])),
    DensePSD.from_matrix(jnp.eye(2) + 0.3),
)

noise.shape        # (5, 5)
noise.logdet()     # sum over blocks
```

Every operation is applied block by block, so cost is the sum over blocks
rather than cubic in the total size. Operators also compose and scale
directly — and the scalar may be a traced value, which is what tempering
needs:

```python
tempered = noise / dbeta        # same structure, same capabilities
C = A @ B                       # composition of two operators
At = A.T                        # the transpose, as an operator
```

`@` composes operators only; applying an operator to an array is always
`matvec` or `matmat`, and `op @ x` raises an error that says so.

## Square roots, sampling and whitening

A PSD operator can produce a square root `L` satisfying `L @ L.T == cov`. That
is what you need to draw samples:

```python
import jax

key = jax.random.key(0)
L = noise.factor()
eps = jax.random.normal(key, (L.shape[1],))
sample = L.matvec(eps)          # covariance equals `noise`
```

and its inverse is what you need to whiten a residual:

```python
z = noise.whiten(residual)      # z has identity covariance
```

`factor()` need not be square. A low-rank-plus-diagonal operator has more
columns than rows, and a reduced-rank operator has fewer — in which case it is
singular and has no `solve`.

## Asking what an operator can do

Not every operator supports every operation cheaply, and support can depend on
an operator's contents. A block-diagonal operator can only `solve`
if all of its blocks can:

```python
noise.supports("solve")      # True
noise.capabilities()         # frozenset({'solve', 'whiten', 'factor', ...})
```

Calling an unsupported operation raises `UnsupportedOpError` rather than quietly
falling back to dense linear algebra, so an accidental cubic cost is visible:

```python
from pyeki.linalg import densify

densify(op).solve(b)         # explicit dense fallback, with a size guard
```

## Next steps

- {doc}`operators` — the full catalogue and what each one costs.
- {doc}`conditioning` — the Gaussian layer built on these operators.
- {doc}`writing-an-operator` — adding a new structure.
- {doc}`../linop-contract` — the precise behavioural contract.
- {doc}`../design` — why the interface looks the way it does.
