# Writing an operator

Adding a structure pyEKI does not ship takes a class, a decorator, and a few
short methods. Operator authors implement `_`-prefixed **hooks**; the public
methods (`matvec`, `solve`, ...) are defined once on the base classes, where
they validate the operand and check support before dispatching to your hook.

## A minimal example

Suppose you have a covariance of the form $\sigma^2 I + uu^\top$ — a scalar
multiple of the identity plus a rank-one update.

```python
import jax.numpy as jnp
from jax import Array
from pyeki.linalg import PSDLinOp, linop

@linop
class IdentityPlusRankOne(PSDLinOp):
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

    def _matvec(self, x):
        return self.sigma2 * x + self.u * jnp.sum(self.u * x, axis=-1, keepdims=True)

    def _to_dense(self):
        return self.sigma2 * jnp.eye(self.shape[0]) + jnp.outer(self.u, self.u)
```

That is enough for the operator to work anywhere application is needed, and
to pass through `jit`, `vmap` and `grad`. Because the class is a
`PSDLinOp`, `_rmatvec` comes for free (a PSD operator is self-adjoint), and
so do `matmat`, `rmatmat` and `T`.

## Required pieces

**`@linop`** makes the class a frozen dataclass and registers it as a JAX
pytree. A field whose annotation is `Array`, a `LinOp` subtype, or a tuple
of those becomes a pytree child; any other field must be marked with
`static_field()`, or the class is rejected at definition time.

**`shape`** is a property computed from static information, never a stored
field, so that it stays a concrete tuple under `jit` and can be used in
shape expressions.

**`_matvec`** receives its operand already validated, **batch axes
included**: it must contract the trailing axis and carry any number of
leading batch axes through, usually as one broadcasting expression. The
same applies to every other hook.

:::{warning}
Do not write `self.A @ x` inside a hook. For arrays with two or more
dimensions, `@` contracts the second-to-last axis; when the operator is
square, that returns a wrong answer without raising. Use
`pyeki.linalg.dense_matvec`, which does the right contraction.
:::

**`_to_dense`** must not route through `matvec` — build the array from the
stored fields directly. The conformance suite compares `matvec` against
`to_dense`, and it verifies the independence mechanically by stubbing out
`matvec` and calling `to_dense()`.

**`check_operator`** from `pyeki.linalg.testing` is the executable contract:
run it on a small instance of every new operator type. It checks
application at several batch ranks, transposition, solves, square roots,
whitening, capability honesty, operand validation, pytree behaviour, and
arithmetic dispatch.

```python
from pyeki.linalg.testing import check_operator

check_operator(IdentityPlusRankOne(jnp.asarray(0.5), jnp.arange(1.0, 5.0)))
```

## Adding cheap operations

Implement whichever of the optional hooks — `_solve`, `_logdet`, `_diag`,
`_factor`, `_whiten` — your structure supports cheaply, and leave the rest
alone. `supports()` reports what you implemented, and calling anything else
raises `UnsupportedOpError`; you never maintain a separate list.

For the example above, the Sherman–Morrison formula gives a cheap solve:

```python
    def _solve(self, b):
        ub = jnp.sum(self.u * b, axis=-1, keepdims=True)
        denom = self.sigma2 + jnp.sum(self.u * self.u)
        return (b - self.u * ub / denom) / self.sigma2

    def _logdet(self):
        n = self.shape[0]
        return n * jnp.log(self.sigma2) + jnp.log1p(
            jnp.sum(self.u * self.u) / self.sigma2
        )
```

The factor is a horizontal stack, which composes from operators pyEKI
already provides:

```python
    def _factor(self):
        from pyeki.linalg import Dense, Identity, hstack
        n = self.shape[0]
        return hstack(
            jnp.sqrt(self.sigma2) * Identity(n),
            Dense(self.u[:, None]),
        )
```

Note the resulting factor is `(n, n + 1)` — wider than the operator, which
is expected and allowed.

## Non-array fields

Any field that is not an array or an operator must be marked static, or the
class is rejected when it is defined:

```python
    size: int = static_field()
```

Static metadata must be hashable and cheap — ints, bools, strings, tuples
of those — never NumPy arrays.

## Two rules imposed by JAX

**The dataclass constructor only stores.** Anything computed from the
inputs — a Cholesky factor, an eigendecomposition — belongs in a
`from_matrix`-style classmethod. JAX rebuilds operators through the
constructor on every `jit`/`vmap` boundary, so a computing constructor
silently recomputes there, and a factorization cached lazily inside a
traced function is discarded when the trace ends.

**Constructor validation is shape-only and tolerant.** A `__post_init__`
may check ranks and static structure, but it must never read array values,
must skip fields that do not expose `ndim` (JAX sometimes rebuilds pytrees
with placeholder objects), and must reject only ranks *below* the field's
core rank — extra leading axes are how `vmap` reconstructs a batched family
of operators. Value-level preconditions (positivity, definiteness) belong
in `pyeki.linalg.value_check` assertions, which run only under
`pyeki.linalg.debug_checks()`.
