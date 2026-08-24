# Writing an operator

Adding a structure pyEKI does not ship takes a class, a decorator, and three
methods.

## A minimal example

Suppose you have a covariance of the form $\sigma^2 I + uu^\top$ — a scalar
multiple of the identity plus a rank-one update.

```python
import jax.numpy as jnp
from jax import Array
from pyeki.linalg import PSDOperator, operator, static_field

@operator
class IdentityPlusRankOne(PSDOperator):
    """A scalar multiple of the identity plus a rank-one update.

    Parameters
    ----------
    sigma2
        Scalar array, strictly positive.
    u
        Update vector of length ``n``.
    """

    sigma2: Array
    u: Array

    @property
    def shape(self):
        n = self.u.shape[-1]
        return (n, n)

    def matvec(self, x):
        return self.sigma2 * x + self.u * jnp.sum(self.u * x, axis=-1, keepdims=True)

    def to_dense(self):
        return self.sigma2 * jnp.eye(self.shape[0]) + jnp.outer(self.u, self.u)
```

That is enough for the operator to work anywhere a `matvec` is needed, and to
pass through `jit`, `vmap` and `grad`.

## Required pieces

**`@operator`** makes the class a frozen dataclass and registers it as a JAX
pytree. Array fields become pytree children automatically.

**`shape`** is a property, not a stored field, so that it stays a concrete
tuple under `jit` and can be used in shape expressions.

**`matvec`** must contract the **trailing** axis of its argument and allow any
number of leading batch axes.

:::{warning}
Do not write `self.A @ x`. For arrays with two or more dimensions, `@`
contracts the second-to-last axis. When the operator is square, that returns a
wrong answer without raising. Use `pyeki.linalg.dense_matvec`, which does the
right contraction.
:::

**`to_dense`** should be built from the stored arrays, not by applying the
operator to an identity matrix. The conformance suite compares `matvec` against
`to_dense`, so an implementation via `matvec` would make that check compare
`matvec` with itself.

## Adding cheap operations

Implement whichever of `solve`, `logdet`, `diag`, `factor`, `cholesky` and
`whiten` your structure supports cheaply, and leave the rest alone. The base
classes provide versions that raise `UnsupportedOpError`, and `supports()` reports
which you overrode — you do not maintain a separate list.

For the example above, the Sherman–Morrison formula gives a cheap solve:

```python
    def solve(self, b):
        ub = jnp.sum(self.u * b, axis=-1, keepdims=True)
        denom = self.sigma2 + jnp.sum(self.u * self.u)
        return (b - self.u * ub / denom) / self.sigma2

    def logdet(self):
        n = self.shape[0]
        return n * jnp.log(self.sigma2) + jnp.log1p(
            jnp.sum(self.u * self.u) / self.sigma2
        )
```

The factor is a horizontal stack, which composes from operators pyEKI already
provides:

```python
    def factor(self):
        from pyeki.linalg import Dense, HStack, ScaledIdentity
        n = self.shape[0]
        return HStack((
            ScaledIdentity(jnp.sqrt(self.sigma2), n),
            Dense(self.u[:, None]),
        ))
```

Note the resulting factor is `(n, n + 1)` — wider than the operator, which is
expected and allowed.

## Non-array fields

Any field that is not an array must be marked static, or it becomes a pytree
child and arrives as a tracer under `jit`:

```python
    size: int = static_field()
```

The `@operator` decorator raises at class-definition time if you forget, rather
than letting it fail later inside a shape expression.

## Testing

Run the conformance suite on an instance. It checks every operation you
implemented against a dense reference, sweeps the leading batch rank, verifies
the square roots agree with each other, and round-trips the operator through
`jit`, `vmap` and `grad`:

```python
from pyeki.linalg.testing import check_operator

def test_identity_plus_rank_one():
    op = IdentityPlusRankOne(jnp.array(2.0), jnp.array([1.0, -1.0, 0.5]))
    check_operator(op)
```

Operations you did not implement are skipped, so the same call works for every
operator.
