"""Conformance checks for operator implementations.

Call :func:`check_operator` on an instance of a new operator type — small
enough to densify — to verify it against the linear operator contract. It
runs the individual checks below, which can also be called on their own.

=================================  ===========================================
function                           checks
=================================  ===========================================
:func:`check_core`                 ``matvec``/``rmatvec`` at batch rank
                                   0, 1, 2; ``matmat``/``rmatmat`` batched
                                   and unbatched
:func:`check_transpose`            ``T`` matches the dense transpose, its
                                   capabilities included
:func:`check_solve`                ``solve``/``solve_mat`` vs. the inverse
:func:`check_factor`               ``factor()`` reproduces the operator
:func:`check_whiten`               ``whiten`` is a fixed valid whitener
:func:`check_scalars`              ``diag`` and ``logdet``
:func:`check_dense_independence`   ``to_dense`` does not route through matvec
:func:`check_capabilities`         ``supports`` is honest in both directions
:func:`check_operand_validation`   wrong core shapes raise ``ValueError``
:func:`check_pytree`               flatten round trip, sentinels, ``jit``,
                                   ``vmap`` over operands and operators,
                                   ``grad``
:func:`check_repr`                 repr is type and shape, no array data
:func:`check_arithmetic`           arithmetic dispatch and guided errors
:func:`check_family`               ``batch_shape`` and family inertness
=================================  ===========================================

Checks skip operations the operator does not claim to support; capability
honesty itself is checked, so the same suite applies to every type.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .base import LinOp, PSDLinOp, SquareLinOp, UnsupportedOpError
from .composite import Product, PSDScaled, Scaled, SquareScaled, Transposed

__all__ = [
    "check_operator",
    "check_core",
    "check_transpose",
    "check_solve",
    "check_factor",
    "check_whiten",
    "check_scalars",
    "check_dense_independence",
    "check_capabilities",
    "check_operand_validation",
    "check_pytree",
    "check_repr",
    "check_arithmetic",
    "check_family",
]

_RTOL, _ATOL = 1e-9, 1e-9
_MISSING = object()


def _ref(op: LinOp) -> np.ndarray:
    return np.asarray(op.to_dense())


def _close(got, want, what: str, rtol=_RTOL, atol=_ATOL) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    err = np.abs(got - want).max() if got.size else 0.0
    assert np.allclose(got, want, rtol=rtol, atol=atol), f"{what}: max abs err {err:.3e}"


def _rand(key, shape) -> Array:
    return jax.random.normal(key, shape, dtype=jnp.float64)


def _expect_raises(exc: type[Exception], fn, what: str) -> Exception:
    try:
        fn()
    except exc as e:
        return e
    raise AssertionError(f"{what} should have raised {exc.__name__}")


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def check_core(op: LinOp, key) -> None:
    """Check application and transposed application against the dense form.

    ``matvec`` and ``rmatvec`` run at leading batch rank 0, 1 and 2 with a
    distinct random operand per rank; ``matmat`` and ``rmatmat`` run batched
    and unbatched. Varying the rank catches implementations that contract
    the wrong axis, which are wrong without raising exactly when the
    operator is square.
    """
    A = _ref(op)
    n_out, n_in = op.shape
    keys = jax.random.split(key, 10)
    for i, batch in enumerate([(), (3,), (2, 3)]):
        x = _rand(keys[i], (*batch, n_in))
        want = np.einsum("ij,...j->...i", A, np.asarray(x))
        _close(op.matvec(x), want, f"{op!r}.matvec batch={batch}")
        y = _rand(keys[3 + i], (*batch, n_out))
        want = np.einsum("ij,...i->...j", A, np.asarray(y))
        _close(op.rmatvec(y), want, f"{op!r}.rmatvec batch={batch}")
    for j, batch in enumerate([(), (2,)]):
        X = _rand(keys[6 + j], (*batch, n_in, 4))
        _close(op.matmat(X), A @ np.asarray(X), f"{op!r}.matmat batch={batch}")
        Y = _rand(keys[8 + j], (*batch, n_out, 4))
        _close(op.rmatmat(Y), A.T @ np.asarray(Y), f"{op!r}.rmatmat batch={batch}")


def check_transpose(op: LinOp, key) -> None:
    """Check ``T``: the dense transpose, full core behaviour, and ``T.T``.

    The transpose is checked as an operator in its own right — core
    behaviour, solve, scalars, and capability honesty — so a structured
    ``T`` override cannot ship a broken ``solve`` or ``logdet`` behind a
    correct dense form.
    """
    A = _ref(op)
    t = op.T
    key_core, key_solve = jax.random.split(key)
    _close(t.to_dense(), A.T, f"{op!r}.T.to_dense")
    check_core(t, key_core)
    check_solve(t, key_solve)
    check_scalars(t)
    check_capabilities(t)
    _close(t.T.to_dense(), A, f"{op!r}.T.T.to_dense")


def check_solve(op: LinOp, key) -> None:
    """Check ``solve`` at batch rank 0, 1, 2 and ``solve_mat``, when claimed."""
    if not (isinstance(op, SquareLinOp) and op.supports("solve")):
        return
    A = _ref(op)
    n = op.shape[0]
    keys = jax.random.split(key, 5)
    inv = np.linalg.inv(A)
    for i, batch in enumerate([(), (3,), (2, 3)]):
        b = _rand(keys[i], (*batch, n))
        want = np.einsum("ij,...j->...i", inv, np.asarray(b))
        _close(op.solve(b), want, f"{op!r}.solve batch={batch}", rtol=1e-7, atol=1e-7)
    for j, batch in enumerate([(), (2,)]):
        B = _rand(keys[3 + j], (*batch, n, 4))
        _close(
            op.solve_mat(B),
            np.linalg.solve(A, np.asarray(B)),
            f"{op!r}.solve_mat batch={batch}",
            rtol=1e-7,
            atol=1e-7,
        )


def check_factor(op: LinOp, key) -> None:
    """Check ``factor()``: ``L L^T`` reproduces the operator, and ``L`` is a
    conforming operator in its own right (including ``rmatvec``)."""
    if not (isinstance(op, PSDLinOp) and op.supports("factor")):
        return
    A = _ref(op)
    L = op.factor()
    assert L.shape[0] == op.shape[0], f"factor rows {L.shape[0]} != {op.shape[0]}"
    Ld = _ref(L)
    _close(Ld @ Ld.T, A, f"{op!r}.factor: L L^T != A", rtol=1e-8, atol=1e-8)
    check_core(L, key)


def check_whiten(op: LinOp, key) -> None:
    """Check that ``whiten`` applies one fixed valid whitener, when claimed.

    Recovers ``W`` by applying ``whiten`` to the columns of the identity,
    then requires ``W A W^T == I`` and *elementwise* agreement of
    ``whiten(x)`` with ``W x`` at batch rank 0, 1 and 2 — which pins
    linearity, per-instance fixedness, and rank behaviour at once.
    ``whiten_mat`` is compared columnwise against ``whiten``, never against
    any particular factorization, which the contract does not promise.
    """
    if not (isinstance(op, PSDLinOp) and op.supports("whiten")):
        return
    A = _ref(op)
    n = op.shape[0]
    keys = jax.random.split(key, 5)
    W = np.asarray(op.whiten(jnp.eye(n))).T  # row i of whiten(I) is W e_i
    _close(W @ A @ W.T, np.eye(n), f"{op!r}.whiten: W A W^T != I", rtol=1e-7, atol=1e-7)
    for i, batch in enumerate([(), (3,), (2, 3)]):
        x = _rand(keys[i], (*batch, n))
        want = np.einsum("ij,...j->...i", W, np.asarray(x))
        _close(
            op.whiten(x), want, f"{op!r}.whiten batch={batch}", rtol=1e-7, atol=1e-7
        )
    for j, batch in enumerate([(), (2,)]):
        X = _rand(keys[3 + j], (*batch, n, 4))
        _close(
            op.whiten_mat(X),
            W @ np.asarray(X),
            f"{op!r}.whiten_mat batch={batch}",
            rtol=1e-7,
            atol=1e-7,
        )


def check_scalars(op: LinOp) -> None:
    """Check ``diag`` and ``logdet`` against the dense reference, when claimed.

    Also checks that ``logdet`` is a real 0-d JAX array, never a Python
    float or a complex value.
    """
    A = _ref(op)
    if isinstance(op, SquareLinOp) and op.supports("diag"):
        _close(op.diag(), np.diag(A), f"{op!r}.diag")
    if isinstance(op, SquareLinOp) and op.supports("logdet"):
        ld = op.logdet()
        assert isinstance(ld, jnp.ndarray), "logdet must return a JAX array"
        assert jnp.ndim(ld) == 0, "logdet must be 0-d"
        assert not jnp.iscomplexobj(ld), "logdet must be real, not complex"
        _close(ld, np.linalg.slogdet(A)[1], f"{op!r}.logdet", rtol=1e-7, atol=1e-7)


def check_dense_independence(op: LinOp) -> None:
    """Check that ``to_dense`` does not route through ``matvec`` or ``matmat``.

    Temporarily replaces the class-level application methods and hooks —
    all eight names, ``matvec``/``rmatvec``/``matmat``/``rmatmat`` and
    their hooks — with raising stubs, on the class of **every** operator
    in the pytree (instances are frozen, so the patch must be on classes;
    stubbing only the outermost class would let a composite route its
    ``to_dense`` through a child's application). A ``to_dense`` written
    via application would make every dense comparison in this suite
    compare application with itself.
    """
    names = (
        "matvec", "_matvec", "matmat", "_matmat",
        "rmatvec", "_rmatvec", "rmatmat", "_rmatmat",
    )

    def _operator_classes(obj: LinOp) -> set[type]:
        classes = {type(obj)}
        for f in dataclasses.fields(obj):
            if f.metadata.get("static", False):
                continue
            value = getattr(obj, f.name)
            items = value if isinstance(value, tuple) else (value,)
            for item in items:
                if isinstance(item, LinOp):
                    classes |= _operator_classes(item)
        return classes

    def _stub(name):
        def raise_(self, *args, **kwargs):
            raise AssertionError(f"to_dense must not route through {name}")

        return raise_

    classes = _operator_classes(op)
    saved = {
        (cls, name): cls.__dict__.get(name, _MISSING)
        for cls in classes
        for name in names
    }
    try:
        for cls, name in saved:
            setattr(cls, name, _stub(name))
        op.to_dense()
    finally:
        for (cls, name), impl in saved.items():
            if impl is _MISSING:
                delattr(cls, name)
            else:
                setattr(cls, name, impl)


_OPERATION_OPERANDS = {
    "matvec": lambda n_out, n_in: (jnp.zeros(n_in),),
    "rmatvec": lambda n_out, n_in: (jnp.zeros(n_out),),
    "matmat": lambda n_out, n_in: (jnp.zeros((n_in, 3)),),
    "rmatmat": lambda n_out, n_in: (jnp.zeros((n_out, 3)),),
    "to_dense": lambda n_out, n_in: (),
    "solve": lambda n_out, n_in: (jnp.zeros(n_out),),
    "solve_mat": lambda n_out, n_in: (jnp.zeros((n_out, 3)),),
    "logdet": lambda n_out, n_in: (),
    "diag": lambda n_out, n_in: (),
    "factor": lambda n_out, n_in: (),
    "whiten": lambda n_out, n_in: (jnp.zeros(n_out),),
    "whiten_mat": lambda n_out, n_in: (jnp.zeros((n_out, 3)),),
}


def check_capabilities(op: LinOp) -> None:
    """Check that ``supports`` is honest in both directions.

    Every supported operation runs without ``UnsupportedOpError``; every
    type-defined operation reported unsupported raises it; operations below
    the operator's level are absent from the type. An unknown name raises
    ``ValueError``.
    """
    n_out, n_in = op.shape
    for name, make_args in _OPERATION_OPERANDS.items():
        supported = op.supports(name)
        defined = hasattr(type(op), name)
        if supported:
            assert defined, f"{op!r} supports {name} but does not define it"
            getattr(op, name)(*make_args(n_out, n_in))  # must not raise
        elif defined:
            _expect_raises(
                UnsupportedOpError,
                lambda n=name, f=make_args: getattr(op, n)(*f(n_out, n_in)),
                f"{op!r}.{name} (unsupported)",
            )
    _expect_raises(ValueError, lambda: op.supports("choleksy"), "unknown name")
    caps = op.capabilities()
    assert all(op.supports(name) for name in caps)

    # Operations below the operator's level must be absent from the type.
    square_names = ("solve", "solve_mat", "logdet", "diag")
    psd_names = ("factor", "whiten", "whiten_mat")
    if not isinstance(op, SquareLinOp):
        for name in square_names + psd_names:
            assert not hasattr(type(op), name), f"{op!r} defines below-level {name}"
    elif not isinstance(op, PSDLinOp):
        for name in psd_names:
            assert not hasattr(type(op), name), f"{op!r} defines below-level {name}"


def check_operand_validation(op: LinOp) -> None:
    """Check that a wrong contracted-axis length or insufficient rank raises.

    The uncontracted ``k`` axis of the matrix methods is unconstrained and
    must not raise, including ``k = 0``.
    """
    n_out, n_in = op.shape
    vec_methods = [("matvec", n_in), ("rmatvec", n_out)]
    mat_methods = [("matmat", n_in), ("rmatmat", n_out)]
    if isinstance(op, SquareLinOp) and op.supports("solve"):
        vec_methods.append(("solve", op.shape[0]))
        mat_methods.append(("solve_mat", op.shape[0]))
    if isinstance(op, PSDLinOp) and op.supports("whiten"):
        vec_methods.append(("whiten", op.shape[0]))
        mat_methods.append(("whiten_mat", op.shape[0]))

    for name, size in vec_methods:
        method = getattr(op, name)
        _expect_raises(
            ValueError,
            lambda m=method, s=size: m(jnp.zeros(s + 1)),
            f"{op!r}.{name} wrong size",
        )
        _expect_raises(
            ValueError, lambda m=method: m(jnp.asarray(0.0)), f"{op!r}.{name} rank 0"
        )
    for name, size in mat_methods:
        method = getattr(op, name)
        _expect_raises(
            ValueError,
            lambda m=method, s=size: m(jnp.zeros((s + 1, 3))),
            f"{op!r}.{name} wrong size",
        )
        _expect_raises(
            ValueError, lambda m=method, s=size: m(jnp.zeros(s)), f"{op!r}.{name} rank 1"
        )
        out = method(jnp.zeros((size, 0)))  # k = 0 is a valid core shape
        assert out.shape[-1] == 0


def check_pytree(op: LinOp, key) -> None:
    """Check pytree behaviour: round trip, sentinel tolerance, ``jit``,
    ``vmap`` over operands and over the operator itself, and ``grad``.

    The operator-batching check flattens the instance, stacks each leaf,
    and reconstructs inside ``jax.vmap`` — exercising the vmap-exit
    reconstruction, which bypasses the constructor. So does the sentinel
    check: ``tree_unflatten`` with bare ``object()`` leaves must succeed
    for every operator type, composites included.
    """
    leaves, treedef = jax.tree_util.tree_flatten(op)
    n_in = op.shape[1]
    keys = jax.random.split(key, 3)

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(op)
    x = _rand(keys[0], (n_in,))
    _close(rebuilt.matvec(x), op.matvec(x), f"{op!r} round-trip matvec")

    # JAX internals may unflatten with placeholder leaves; constructors must
    # tolerate them, composites included.
    jax.tree_util.tree_unflatten(treedef, [object()] * treedef.num_leaves)

    _close(
        jax.jit(lambda o, v: o.matvec(v))(op, x), op.matvec(x), f"{op!r} under jit"
    )

    xs = _rand(keys[1], (3, n_in))
    _close(
        jax.vmap(lambda v: op.matvec(v))(xs),
        op.matvec(xs),
        f"{op!r} vmap over operand agrees with native batching",
    )

    if leaves:
        stacked = [jnp.stack([leaf] * 3) for leaf in leaves]
        family = jax.vmap(
            lambda *ls: jax.tree_util.tree_unflatten(treedef, list(ls))
        )(*stacked)
        got = jax.vmap(lambda o, v: o.matvec(v))(family, xs)
        want = np.stack([np.asarray(op.matvec(xs[i])) for i in range(3)])
        _close(got, want, f"{op!r} vmap over the operator agrees with a loop")

    float_leaves = any(
        jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) for leaf in leaves
    )
    if float_leaves:
        g = jax.grad(lambda o: jnp.sum(o.matvec(x)), allow_int=True)(op)
        assert jax.tree_util.tree_structure(g) == treedef


def check_repr(op: LinOp) -> None:
    """Check that ``repr`` is the type name and shape, with no array data."""
    assert repr(op) == f"{type(op).__name__}{op.shape}", repr(op)


def check_arithmetic(op: LinOp) -> None:
    """Check operator arithmetic: composition, scaling, and guided errors.

    ``@`` composes with another operator and rejects arrays with a guided
    ``TypeError`` — for JAX *and* NumPy left operands, which pins the
    ``__array_ufunc__ = None`` deferral the guided errors depend on. ``*``
    and ``/`` return a level-preserving scaled operator and fold when
    nested; non-scalar factors are rejected.
    """
    A = _ref(op)
    n_out, n_in = op.shape

    composed = op @ op.T
    assert isinstance(composed, Product)
    _close(composed.to_dense(), A @ A.T, f"{op!r} @ {op!r}.T", rtol=1e-8, atol=1e-8)

    for bad in (lambda: op @ jnp.ones(n_in), lambda: jnp.ones(n_out) @ op,
                lambda: np.ones(n_out) @ op):
        e = _expect_raises(TypeError, bad, "array @ dispatch")
        assert "matvec" in str(e), str(e)
    for bad in (lambda: jnp.ones(3) * op, lambda: np.ones(3) * op):
        e = _expect_raises(TypeError, bad, "non-scalar * op")
        assert "scalar" in str(e), str(e)
    _expect_raises(TypeError, lambda: op * op, "op * op")

    expected_cls = (
        PSDScaled
        if isinstance(op, PSDLinOp)
        else SquareScaled
        if isinstance(op, SquareLinOp)
        else Scaled
    )
    for scaled, factor_val in [(2.0 * op, 2.0), (op * np.float64(3.0), 3.0),
                               (op / 2.0, 0.5)]:
        assert type(scaled) is expected_cls, type(scaled)
        _close(scaled.to_dense(), factor_val * A, f"{op!r} scaled by {factor_val}")

    folded = 2.0 * (2.0 * op)
    assert isinstance(folded, Scaled) and not isinstance(folded.op, Scaled)
    _close(folded.to_dense(), 4.0 * A, f"{op!r} folded scaling")

    t = op.T
    if isinstance(op, PSDLinOp):
        assert t is op
    elif type(op).T is LinOp.T:
        assert isinstance(t, Transposed) and t.T is op


def check_family(op: LinOp) -> None:
    """Check ``batch_shape`` legibility and family inertness.

    The instance itself must report ``batch_shape == ()``. Its leaves are
    stacked and unflattened into a vmapped family, which must report the
    stacked batch shape, take the ``vmapped(...)`` repr form, refuse every
    type-defined operation with ``ValueError``, and still answer
    introspection (``shape``, ``supports``, ``capabilities``).
    """
    assert op.batch_shape == (), f"{op!r}.batch_shape != ()"
    leaves, treedef = jax.tree_util.tree_flatten(op)
    if not leaves:
        return  # no data leaves: a family of this operator cannot exist
    family = jax.tree_util.tree_unflatten(
        treedef, [jnp.stack([leaf] * 3) for leaf in leaves]
    )
    assert family.batch_shape == (3,), family.batch_shape
    want_repr = f"vmapped({type(op).__name__}{op.shape}, batch=(3,))"
    assert repr(family) == want_repr, repr(family)
    assert family.shape == op.shape
    assert family.supports("matvec")
    family.capabilities()
    assert family.T.batch_shape == (3,), "T must stay a view on a family"

    # Multi-axis batches must be reported in full, not just the last axis.
    family2 = jax.tree_util.tree_unflatten(
        treedef, [jnp.broadcast_to(leaf, (2, 3, *leaf.shape)) for leaf in leaves]
    )
    assert family2.batch_shape == (2, 3), family2.batch_shape

    # Arithmetic is guarded: families are inert on both sides.
    for bad in (lambda: 2.0 * family, lambda: family / 2.0,
                lambda: family @ op, lambda: op @ family):
        e = _expect_raises(ValueError, bad, "family arithmetic")
        assert "vmap" in str(e), str(e)

    n_out, n_in = op.shape
    for name, make_args in _OPERATION_OPERANDS.items():
        if not hasattr(type(op), name):
            continue
        e = _expect_raises(
            ValueError,
            lambda n=name, f=make_args: getattr(family, n)(*f(n_out, n_in)),
            f"family {name}",
        )
        assert "vmap" in str(e), str(e)


def check_operator(op: LinOp, *, seed: int = 0) -> None:
    """Run every conformance check against one operator instance.

    Parameters
    ----------
    op
        Instance to check. Should be small enough to densify.
    seed
        Seed for the random test operands; each check draws its own keys.

    Raises
    ------
    AssertionError
        On the first check that fails, with the operation and the error.
    """
    keys = jax.random.split(jax.random.key(seed), 6)
    check_core(op, keys[0])
    check_transpose(op, keys[1])
    check_solve(op, keys[2])
    check_factor(op, keys[3])
    check_whiten(op, keys[4])
    check_scalars(op)
    check_dense_independence(op)
    check_capabilities(op)
    check_operand_validation(op)
    check_pytree(op, keys[5])
    check_repr(op)
    check_arithmetic(op)
    check_family(op)
