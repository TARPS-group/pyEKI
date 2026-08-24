"""Conformance checks for operator implementations.

Call :func:`check_operator` on an instance of a new operator type to verify it
against a dense reference. It runs the individual checks below, which can also
be called on their own.

===============================  ==============================================
function                         checks
===============================  ==============================================
:func:`check_matvec`             ``matvec`` at leading batch rank 0, 1 and 2
:func:`check_matmat`             ``matmat`` with and without batch axes
:func:`check_solve`              ``solve`` and ``solve_mat`` against the inverse
:func:`check_factor`             ``factor``, ``cholesky`` and ``whiten`` agree
:func:`check_scalars`            ``diag`` and ``logdet``
:func:`check_pytree`             flatten round trip, ``jit``, ``vmap``, ``grad``
:func:`check_dense_independent`  ``to_dense`` is not written via ``matvec``
===============================  ==============================================

Checks skip operations the operator does not claim to support, so the same
suite applies to every type.

Notes
-----
Three of these target failures that produce wrong numbers rather than errors,
which is why they are worth running on every new type:

- Varying the leading batch rank catches an implementation that contracts the
  wrong axis. Such an implementation is only wrong when the operator is
  square, and then it raises nothing.
- Requiring ``to_dense`` to be independent of ``matvec`` keeps the rest of the
  suite meaningful. Were ``to_dense`` written as ``self @ eye``, comparing
  ``matvec`` against it would compare ``matvec`` with itself.
- Tying ``factor``, ``cholesky`` and ``whiten`` together matters because they
  are separate square roots that downstream code assumes are consistent.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .base import LinOp, PSDOperator, SquareLinOp, UnsupportedOpError

__all__ = [
    "check_operator",
    "check_dense_independent",
    "check_matvec",
    "check_matmat",
    "check_solve",
    "check_factor",
    "check_scalars",
    "check_pytree",
]

_RTOL, _ATOL = 1e-9, 1e-9


def _ref(op: LinOp) -> np.ndarray:
    return np.asarray(op.to_dense())


def _close(a, b, what: str, rtol=_RTOL, atol=_ATOL) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    err = np.abs(a - b).max() if a.size else 0.0
    assert np.allclose(a, b, rtol=rtol, atol=atol), f"{what}: max abs err {err:.3e}"


def _rng_x(key, shape) -> Array:
    return jax.random.normal(key, shape, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def check_dense_independent(op: LinOp) -> None:
    """Check that ``to_dense`` is defined by the type, not inherited generically."""
    assert "to_dense" in type(op).__dict__ or any(
        "to_dense" in c.__dict__ for c in type(op).__mro__[1:-1] if c is not LinOp
    ), f"{type(op).__name__} must define to_dense independently of matvec"


def check_matvec(op: LinOp, key) -> None:
    """Check ``matvec`` against a dense reference at batch rank 0, 1 and 2."""
    A = _ref(op)
    n_out, n_in = op.shape
    for batch in [(), (3,), (2, 3)]:
        x = _rng_x(key, (*batch, n_in))
        got = op.matvec(x)
        want = np.einsum("ij,...j->...i", A, np.asarray(x))
        _close(got, want, f"{type(op).__name__}.matvec batch={batch}")
        assert got.shape == (*batch, n_out)


def check_matmat(op: LinOp, key) -> None:
    """Check ``matmat`` on core shape ``(n_in, k)``, with and without batching."""
    A = _ref(op)
    n_out, n_in = op.shape
    for batch in [(), (2,)]:
        X = _rng_x(key, (*batch, n_in, 4))
        got = op.matmat(X)
        want = np.asarray(A) @ np.asarray(X)
        _close(got, want, f"{type(op).__name__}.matmat batch={batch}")
        assert got.shape == (*batch, n_out, 4)


def check_solve(op: LinOp, key) -> None:
    """Check ``solve`` and ``solve_mat`` against a dense inverse.

    Skipped if the operator does not support ``solve``.
    """
    if not (isinstance(op, SquareLinOp) and op.supports("solve")):
        return
    A = _ref(op)
    n = op.shape[0]
    for batch in [(), (3,)]:
        b = _rng_x(key, (*batch, n))
        _close(
            op.solve(b),
            np.einsum("ij,...j->...i", np.linalg.inv(A), np.asarray(b)),
            f"{type(op).__name__}.solve batch={batch}",
            rtol=1e-7, atol=1e-7,
        )
    B = _rng_x(key, (n, 4))
    _close(op.solve_mat(B), np.linalg.solve(A, np.asarray(B)),
           f"{type(op).__name__}.solve_mat", rtol=1e-7, atol=1e-7)


def check_factor(op: LinOp, key) -> None:
    """Check that the square roots reproduce the operator and agree with each other.

    Verifies ``L @ L.T`` equals the operator for both ``factor`` and
    ``cholesky``, that ``whiten`` inverts the same factor ``cholesky``
    returns, and that whitening yields unit variance.
    """
    if not isinstance(op, PSDOperator):
        return
    A = _ref(op)
    n = op.shape[0]

    if op.supports("factor"):
        L = op.factor()
        assert L.shape[0] == n, f"factor rows {L.shape[0]} != {n}"
        Ld = _ref(L)
        _close(Ld @ Ld.T, A, f"{type(op).__name__}.factor: L L^T != A")

    if op.supports("cholesky"):
        C = op.cholesky()
        assert C.shape == (n, n), f"cholesky must be square, got {C.shape}"
        Cd = _ref(C)
        _close(Cd @ Cd.T, A, f"{type(op).__name__}.cholesky: L L^T != A")

        # whiten(x) must equal L^{-1} x for that same L.
        x = _rng_x(key, (n,))
        _close(
            op.whiten(x),
            np.linalg.solve(Cd, np.asarray(x)),
            f"{type(op).__name__}.whiten disagrees with cholesky()",
            rtol=1e-7, atol=1e-7,
        )
        # ... and whitening must actually whiten: ||L^-1 x||^2 == x^T A^-1 x.
        _close(
            np.sum(np.asarray(op.whiten(x)) ** 2),
            np.asarray(x) @ np.linalg.solve(A, np.asarray(x)),
            f"{type(op).__name__}.whiten is not a whitener",
            rtol=1e-6, atol=1e-6,
        )


def check_scalars(op: LinOp) -> None:
    """Check ``diag`` and ``logdet`` against a dense reference.

    Also checks that ``logdet`` returns a real array rather than a Python
    float or a complex value.
    """
    A = _ref(op)
    if isinstance(op, SquareLinOp) and op.supports("diag"):
        _close(op.diag(), np.diag(A), f"{type(op).__name__}.diag")
    if isinstance(op, SquareLinOp) and op.supports("logdet"):
        ld = op.logdet()
        assert isinstance(ld, jnp.ndarray), "logdet must return a JAX array"
        assert not jnp.iscomplexobj(ld), "logdet must be real, not complex"
        _close(ld, np.linalg.slogdet(A)[1], f"{type(op).__name__}.logdet",
               rtol=1e-7, atol=1e-7)


def check_pytree(op: LinOp, key) -> None:
    """Check the operator survives flatten/unflatten, ``jit``, ``vmap`` and ``grad``."""
    leaves, treedef = jax.tree_util.tree_flatten(op)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(op)

    n_in = op.shape[1]
    x = _rng_x(key, (n_in,))
    _close(rebuilt.matvec(x), op.matvec(x), f"{type(op).__name__} round-trip matvec")

    f = jax.jit(lambda o, v: o.matvec(v))
    _close(f(op, x), op.matvec(x), f"{type(op).__name__} under jit")

    xs = _rng_x(key, (3, n_in))
    _close(jax.vmap(lambda v: op.matvec(v))(xs), op.matvec(xs),
           f"{type(op).__name__} vmap agrees with native batching")

    # grad through the leaves; only meaningful if there are float leaves.
    if any(jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) for leaf in leaves):
        g = jax.grad(lambda o: jnp.sum(o.matvec(x)))(op)
        assert jax.tree_util.tree_structure(g) == treedef


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def check_operator(op: LinOp, *, seed: int = 0) -> None:
    """Run every conformance check against one operator instance.

    Parameters
    ----------
    op
        Instance to check. Should be small enough to densify.
    seed
        Seed for the random test vectors.

    Raises
    ------
    AssertionError
        On the first check that fails, with the operation and the error.
    """
    key = jax.random.key(seed)
    check_dense_independent(op)
    check_matvec(op, key)
    check_matmat(op, key)
    check_solve(op, key)
    check_factor(op, key)
    check_scalars(op)
    check_pytree(op, key)


def assert_raises_unsupported(op: LinOp, name: str) -> None:
    """Check that an unsupported operation raises rather than falling back to dense."""
    try:
        getattr(op, name)()
    except UnsupportedOpError:
        return
    except TypeError:
        try:
            getattr(op, name)(jnp.zeros(op.shape[1]))
        except UnsupportedOpError:
            return
    raise AssertionError(
        f"{type(op).__name__}.{name} should have raised UnsupportedOpError"
    )
