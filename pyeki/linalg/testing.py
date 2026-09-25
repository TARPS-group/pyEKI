"""Conformance checks for operator implementations.

Call :func:`check_operator` on an instance of a new operator type — small
enough to densify — to verify it against the linear operator contract. It
runs the individual checks below, which can also be called on their own.

=================================  ===========================================
function                           checks
=================================  ===========================================
:func:`check_core`                 ``matvec``/``rmatvec`` at batch shapes
                                   ``()``, ``(1,)``, ``(n,)``, ``(2, n)``;
                                   ``matmat``/``rmatmat`` at ``k`` of 1,
                                   ``n`` and ``n + 1``
:func:`check_transpose`            ``T`` and ``T.T`` match the dense form,
                                   and are conforming operators themselves
:func:`check_solve`                ``solve``/``solve_mat`` vs. the inverse
:func:`check_factor`               ``factor()`` reproduces the operator and
                                   is a conforming operator itself
:func:`check_whiten`               ``whiten`` is a fixed valid whitener
:func:`check_scalars`              ``diag`` and ``logdet``
:func:`check_dense_independence`   ``to_dense`` does not route through matvec
:func:`check_capabilities`         ``supports`` is honest in both directions,
                                   and a PSD type is symmetric PSD
:func:`check_operand_validation`   wrong core shapes raise ``ValueError``
:func:`check_pytree`               every operation after a flatten round
                                   trip, under ``jit``, and under ``vmap``
                                   over operands and operators; sentinels;
                                   ``grad`` values
:func:`check_repr`                 repr is type and shape, no array data
:func:`check_arithmetic`           arithmetic dispatch and guided errors
:func:`check_family`               ``batch_shape`` and family inertness
=================================  ===========================================

Checks skip operations the operator does not claim to support; capability
honesty itself is checked, so the same suite applies to every type.

Every array an operation returns — ``to_dense`` and ``factor().to_dense()``
included — must be a JAX array of a real floating dtype, and of the dtype of
the dense reference it is compared with: float64 for an operator with
float64 arrays applied to float64 operands. A NumPy array, a Python float or
a single-precision result fails, even when its values are close.
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
_LOGDET_TOL = 1e-10
_MISSING = object()


def _leaf_dtype(op: LinOp) -> np.dtype:
    """The floating dtype an operator's dense form should have: that of its
    inexact array leaves combined, or the default float for none."""
    dtypes = [
        jnp.result_type(leaf)
        for leaf in jax.tree_util.tree_leaves(op)
        if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)
    ]
    return np.dtype(jnp.result_type(*dtypes) if dtypes else jnp.result_type(float))


def _check_output(got, what: str, dtype) -> None:
    """Require a JAX array of a real floating dtype, equal to ``dtype``."""
    assert isinstance(got, jax.Array), (
        f"{what}: returned {type(got).__name__}, not a JAX array"
    )
    assert jnp.issubdtype(got.dtype, jnp.floating), (
        f"{what}: dtype {got.dtype} is not a real floating dtype"
    )
    assert got.dtype == dtype, f"{what}: dtype {got.dtype}, expected {dtype}"


def _ref(op: LinOp) -> np.ndarray:
    """The dense form, checked for type, dtype and shape."""
    dense = op.to_dense()
    _check_output(dense, f"{op!r}.to_dense", _leaf_dtype(op))
    assert dense.shape == op.shape, f"{op!r}.to_dense: shape {dense.shape}"
    return np.asarray(dense)


def _close_np(got, want, what: str, rtol=_RTOL, atol=_ATOL) -> None:
    """Compare two arrays in value and shape only."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, f"{what}: shape {got.shape} != {want.shape}"
    err = np.abs(got - want).max() if got.size else 0.0
    assert np.allclose(got, want, rtol=rtol, atol=atol), f"{what}: max abs err {err:.3e}"


def _close(got, want, what: str, rtol=_RTOL, atol=_ATOL) -> None:
    """Compare an operation's output with a reference: ``got`` must be a JAX
    array of the reference's dtype, and agree with it in shape and value."""
    want = np.asarray(want)
    _check_output(got, what, want.dtype)
    _close_np(got, want, what, rtol=rtol, atol=atol)


def _numpy_rng(key) -> np.random.Generator:
    """A NumPy generator seeded from a JAX key, typed or raw.

    Operands are drawn with NumPy because ``jax.random`` compiles a sampler
    for every new operand shape, and the checks draw many shapes.
    """
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    return np.random.default_rng(np.asarray(key, dtype=np.uint32).ravel().tolist())


def _rand(rng: np.random.Generator, shape) -> Array:
    return jnp.asarray(rng.standard_normal(shape))


def _vec_batches(n: int) -> list[tuple[int, ...]]:
    """Leading batch shapes for a vector operand whose core length is ``n``.

    A batch axis of length ``n`` makes the operand square in its trailing
    two axes, where contracting the wrong axis gives a wrong answer instead
    of a shape error.
    """
    return list(dict.fromkeys([(), (1,), (n,), (2, n)]))


def _mat_cases(n: int) -> list[tuple[tuple[int, ...], int]]:
    """``(batch shape, k)`` pairs for a matrix operand with ``n`` rows.

    ``k = 1`` catches squeezing; ``k = n`` makes the core square, where the
    wrong contraction is silent.
    """
    return list(
        dict.fromkeys((batch, k) for batch in [(), (2,)] for k in (1, n, n + 1))
    )


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

    ``matvec`` and ``rmatvec`` run at leading batch shapes ``()``, ``(1,)``,
    ``(n,)`` and ``(2, n)``, where ``n`` is the operand's core length, with
    a distinct random operand for each. ``matmat`` and ``rmatmat`` run
    batched and unbatched at ``k`` of 1, ``n`` and ``n + 1``. A batch or
    column count equal to ``n`` catches implementations that contract the
    wrong axis, which are wrong without raising exactly when the operand is
    square in its trailing axes; ``k = 1`` catches squeezing.
    """
    A = _ref(op)
    n_out, n_in = op.shape
    rng = _numpy_rng(key)
    for batch in _vec_batches(n_in):
        x = _rand(rng, (*batch, n_in))
        want = np.einsum("ij,...j->...i", A, np.asarray(x))
        _close(op.matvec(x), want, f"{op!r}.matvec batch={batch}")
    for batch in _vec_batches(n_out):
        y = _rand(rng, (*batch, n_out))
        want = np.einsum("ij,...i->...j", A, np.asarray(y))
        _close(op.rmatvec(y), want, f"{op!r}.rmatvec batch={batch}")
    for batch, k in _mat_cases(n_in):
        X = _rand(rng, (*batch, n_in, k))
        _close(op.matmat(X), A @ np.asarray(X), f"{op!r}.matmat batch={batch} k={k}")
    for batch, k in _mat_cases(n_out):
        Y = _rand(rng, (*batch, n_out, k))
        _close(
            op.rmatmat(Y), A.T @ np.asarray(Y), f"{op!r}.rmatmat batch={batch} k={k}"
        )


def check_transpose(op: LinOp, key) -> None:
    """Check ``T`` and ``T.T``: the dense forms, and full behaviour of each.

    ``T`` must densify to the dense transpose and ``T.T`` to the operator's
    own dense form. Each is then checked as an operator in its own right —
    core behaviour, solve, scalars, and capability honesty — so a
    structured ``T`` override cannot ship a broken ``solve`` or ``logdet``
    behind a correct dense form. Either one that is the operator itself (as
    ``T`` is for a PSD operator, and ``T.T`` for a ``Transposed`` view) is
    not checked again here.
    """
    A = _ref(op)
    t = op.T
    tt = t.T
    keys = jax.random.split(key, 4)
    _close(t.to_dense(), A.T, f"{op!r}.T.to_dense")
    _close(tt.to_dense(), A, f"{op!r}.T.T.to_dense")
    views = [t] if tt is t else [t, tt]
    for view, (key_core, key_solve) in zip(views, (keys[:2], keys[2:]), strict=False):
        if view is op:
            continue
        check_core(view, key_core)
        check_solve(view, key_solve)
        check_scalars(view)
        check_capabilities(view)


def check_solve(op: LinOp, key) -> None:
    """Check ``solve`` and ``solve_mat`` against the dense inverse, when claimed.

    Operands take the batch shapes and column counts of :func:`check_core`.
    """
    if not (isinstance(op, SquareLinOp) and op.supports("solve")):
        return
    A = _ref(op)
    n = op.shape[0]
    rng = _numpy_rng(key)
    inv = np.linalg.inv(A)
    for batch in _vec_batches(n):
        b = _rand(rng, (*batch, n))
        want = np.einsum("ij,...j->...i", inv, np.asarray(b))
        _close(op.solve(b), want, f"{op!r}.solve batch={batch}")
    for batch, k in _mat_cases(n):
        B = _rand(rng, (*batch, n, k))
        _close(
            op.solve_mat(B), inv @ np.asarray(B), f"{op!r}.solve_mat batch={batch} k={k}"
        )


def check_factor(op: LinOp, key) -> None:
    """Check ``factor()``, when claimed: ``L L^T`` reproduces the operator,
    and ``L`` passes every other check in this module.

    Those are :func:`check_core`, :func:`check_transpose`,
    :func:`check_solve`, :func:`check_whiten`, :func:`check_scalars`,
    :func:`check_dense_independence`, :func:`check_capabilities`,
    :func:`check_operand_validation` and :func:`check_pytree` — each of
    which skips what ``L`` does not claim. :func:`check_factor` itself is
    not applied to ``L``, since a factor may be its own factor.
    """
    if not (isinstance(op, PSDLinOp) and op.supports("factor")):
        return
    A = _ref(op)
    L = op.factor()
    assert isinstance(L, LinOp), f"{op!r}.factor returned {type(L).__name__}"
    assert L.shape[0] == op.shape[0], f"factor rows {L.shape[0]} != {op.shape[0]}"
    Ld = _ref(L)
    _close_np(Ld @ Ld.T, A, f"{op!r}.factor: L L^T != A", rtol=1e-8, atol=1e-8)
    keys = jax.random.split(key, 5)
    check_core(L, keys[0])
    check_transpose(L, keys[1])
    check_solve(L, keys[2])
    check_whiten(L, keys[3])
    check_scalars(L)
    check_dense_independence(L)
    check_capabilities(L)
    check_operand_validation(L)
    check_pytree(L, keys[4])


def check_whiten(op: LinOp, key) -> None:
    """Check that ``whiten`` applies one fixed valid whitener, when claimed.

    Recovers ``W`` by applying ``whiten`` to the columns of the identity,
    then requires ``W A W^T == I`` and *elementwise* agreement of
    ``whiten(x)`` with ``W x`` at the batch shapes of :func:`check_core` —
    which pins linearity, per-instance fixedness, and batch behaviour at
    once. ``whiten_mat`` is compared columnwise against ``whiten``, never
    against any particular factorization, which the contract does not
    promise.
    """
    if not (isinstance(op, PSDLinOp) and op.supports("whiten")):
        return
    A = _ref(op)
    n = op.shape[0]
    rng = _numpy_rng(key)
    identity_image = op.whiten(jnp.eye(n))
    _check_output(identity_image, f"{op!r}.whiten", A.dtype)
    W = np.asarray(identity_image).T  # row i of whiten(I) is W e_i
    _close_np(W @ A @ W.T, np.eye(n), f"{op!r}.whiten: W A W^T != I")
    for batch in _vec_batches(n):
        x = _rand(rng, (*batch, n))
        want = np.einsum("ij,...j->...i", W, np.asarray(x))
        _close(op.whiten(x), want, f"{op!r}.whiten batch={batch}")
    for batch, k in _mat_cases(n):
        X = _rand(rng, (*batch, n, k))
        _close(
            op.whiten_mat(X), W @ np.asarray(X), f"{op!r}.whiten_mat batch={batch} k={k}"
        )


def check_scalars(op: LinOp) -> None:
    """Check ``diag`` and ``logdet`` against the dense reference, when claimed.

    ``logdet`` must be a real 0-d JAX array, never a Python float or a
    complex value, and agree with the dense log-determinant to a relative
    tolerance of ``1e-10``.
    """
    A = _ref(op)
    if isinstance(op, SquareLinOp) and op.supports("diag"):
        _close(op.diag(), np.diag(A), f"{op!r}.diag")
    if isinstance(op, SquareLinOp) and op.supports("logdet"):
        ld = op.logdet()
        _check_output(ld, f"{op!r}.logdet", A.dtype)
        assert jnp.ndim(ld) == 0, "logdet must be 0-d"
        _close(
            ld,
            np.linalg.slogdet(A)[1],
            f"{op!r}.logdet",
            rtol=_LOGDET_TOL,
            atol=_LOGDET_TOL,
        )


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
    """Check that ``supports`` and the operator's level are honest.

    Every supported operation runs without ``UnsupportedOpError``; every
    type-defined operation reported unsupported raises it; operations below
    the operator's level are absent from the type. An unknown name raises
    ``ValueError``. An operator at the PSD level must have a dense form
    that is symmetric, to ``1e-10`` relative to its spectral norm, with no
    eigenvalue below ``-1e-8`` times that norm.
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
    else:
        A = _ref(op)
        scale = np.linalg.norm(A, 2)
        asymmetry = np.abs(A - A.T).max()
        assert asymmetry <= 1e-10 * scale, (
            f"{op!r} is PSD-level but not symmetric: max |A - A^T| {asymmetry:.3e}"
        )
        lowest = np.linalg.eigvalsh((A + A.T) / 2).min()
        assert lowest >= -1e-8 * scale, (
            f"{op!r} is PSD-level but has eigenvalue {lowest:.3e}"
        )


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


#: Every operation, as a function of the operator and its operands, and the
#: core shape of each operand: "in"/"out" for a vector of length n_in/n_out,
#: "in_mat"/"out_mat" for a matrix with that many rows.
_OPERATIONS = {
    "matvec": (lambda o, x: o.matvec(x), ("in",)),
    "rmatvec": (lambda o, y: o.rmatvec(y), ("out",)),
    "matmat": (lambda o, X: o.matmat(X), ("in_mat",)),
    "rmatmat": (lambda o, Y: o.rmatmat(Y), ("out_mat",)),
    "to_dense": (lambda o: o.to_dense(), ()),
    "solve": (lambda o, b: o.solve(b), ("out",)),
    "solve_mat": (lambda o, B: o.solve_mat(B), ("out_mat",)),
    "logdet": (lambda o: o.logdet(), ()),
    "diag": (lambda o: o.diag(), ()),
    "factor": (lambda o: o.factor().to_dense(), ()),
    "whiten": (lambda o, x: o.whiten(x), ("out",)),
    "whiten_mat": (lambda o, X: o.whiten_mat(X), ("out_mat",)),
}


def _supported_operations(op: LinOp) -> dict:
    return {
        name: entry
        for name, entry in _OPERATIONS.items()
        if hasattr(type(op), name) and op.supports(name)
    }


def _draw_operands(op: LinOp, table: dict, rng: np.random.Generator) -> dict:
    n_out, n_in = op.shape
    shapes = {"in": (n_in,), "out": (n_out,), "in_mat": (n_in, 2), "out_mat": (n_out, 2)}
    return {
        name: tuple(_rand(rng, shapes[spec]) for spec in specs)
        for name, (_, specs) in table.items()
    }


def _apply(o: LinOp, table: dict, operands: dict, what: str) -> dict:
    """Run every operation in ``table``, turning any exception into an
    ``AssertionError`` that names the operation and the context."""
    out = {}
    for name, (fn, _) in table.items():
        try:
            out[name] = fn(o, *operands[name])
        except Exception as e:  # noqa: BLE001 - reported, not swallowed
            raise AssertionError(
                f"{what}: {name} raised {type(e).__name__}: {e}"
            ) from e
    return out


def _compare(got: dict, want: dict, what: str) -> None:
    for name in want:
        _close(got[name], want[name], f"{what}: {name}")


def check_pytree(op: LinOp, key, *, other: LinOp | None = None) -> None:
    """Check pytree behaviour of every supported operation.

    Each operation — the application methods, ``to_dense``, ``solve``,
    ``solve_mat``, ``logdet``, ``diag``, ``factor().to_dense()``,
    ``whiten`` and ``whiten_mat``, as supported — must give the eager
    result on an instance rebuilt by a flatten round trip; under ``jit``,
    with the operator passed as an argument rather than closed over; under
    ``vmap`` over its operands, where it must agree with native batching;
    and under ``vmap`` over a family of two operators, where it must agree
    with a loop over the two. ``tree_unflatten`` with bare ``object()``
    leaves must succeed. The gradients of ``y @ matvec(x)`` and
    ``x @ rmatvec(y)`` with respect to the array leaves must equal the
    gradient of ``y @ to_dense() @ x``, leaf by leaf.

    Parameters
    ----------
    op
        Instance to check.
    key
        JAX random key for the operands.
    other
        A second instance with the same pytree structure and leaf shapes
        but different values, to make up the family with ``op``. If
        omitted, the family is two copies of ``op``, which cannot expose an
        operator that answers from something other than its leaves.

    Raises
    ------
    AssertionError
        On the first disagreement, or if any operation raises.

    Notes
    -----
    The family is built by stacking leaves and reconstructing inside
    ``jax.vmap``, which exercises the vmap-exit reconstruction; like the
    sentinel check, that bypasses the constructor.
    """
    leaves, treedef = jax.tree_util.tree_flatten(op)
    rng = _numpy_rng(key)
    table = _supported_operations(op)
    operands = _draw_operands(op, table, rng)
    eager = _apply(op, table, operands, f"{op!r}")

    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(op)
    _compare(
        _apply(rebuilt, table, operands, f"{op!r} after a flatten round trip"),
        eager,
        f"{op!r} round trip",
    )

    # JAX internals may unflatten with placeholder leaves; constructors must
    # tolerate them, composites included.
    jax.tree_util.tree_unflatten(treedef, [object()] * treedef.num_leaves)

    jitted = jax.jit(lambda o, a: _apply(o, table, a, f"{op!r} under jit"))
    _compare(jitted(op, operands), eager, f"{op!r} under jit")

    with_operands = {name: entry for name, entry in table.items() if entry[1]}
    stacked = {
        name: tuple(_rand(rng, (3, *a.shape)) for a in args)
        for name, args in operands.items()
        if name in with_operands
    }
    what = f"{op!r} under vmap over operands"
    _compare(
        jax.vmap(lambda a: _apply(op, with_operands, a, what))(stacked),
        _apply(op, with_operands, stacked, f"{op!r} with batched operands"),
        f"{what} vs. native batching",
    )

    if leaves:
        second = op if other is None else other
        other_leaves, other_treedef = jax.tree_util.tree_flatten(second)
        assert other_treedef == treedef, (
            f"other ({second!r}) has a different pytree structure or static "
            f"metadata from {op!r}"
        )
        for a, b in zip(leaves, other_leaves, strict=True):
            assert jnp.shape(a) == jnp.shape(b), (
                f"other's leaf shape {jnp.shape(b)} != {jnp.shape(a)}"
            )
        family = jax.vmap(
            lambda *ls: jax.tree_util.tree_unflatten(treedef, list(ls))
        )(*[jnp.stack([a, b]) for a, b in zip(leaves, other_leaves, strict=True)])
        second_operands = _draw_operands(op, table, rng)
        both = jax.tree_util.tree_map(
            lambda a, b: jnp.stack([a, b]), operands, second_operands
        )
        want = jax.tree_util.tree_map(
            lambda a, b: jnp.stack([a, b]),
            eager,
            _apply(second, table, second_operands, f"{second!r} (other)"),
        )
        what = f"{op!r} under vmap over the operator"
        _compare(
            jax.vmap(lambda o, a: _apply(o, table, a, what))(family, both),
            want,
            f"{what} vs. a loop",
        )

    if any(jnp.issubdtype(jnp.result_type(leaf), jnp.inexact) for leaf in leaves):
        n_out, n_in = op.shape
        x, y = _rand(rng, (n_in,)), _rand(rng, (n_out,))
        via_dense = jax.grad(
            lambda o: jnp.dot(y, jnp.dot(o.to_dense(), x)), allow_int=True
        )(op)
        for name, scalar in (
            ("matvec", lambda o: jnp.dot(y, o.matvec(x))),
            ("rmatvec", lambda o: jnp.dot(x, o.rmatvec(y))),
        ):
            g = jax.grad(scalar, allow_int=True)(op)
            assert jax.tree_util.tree_structure(g) == treedef
            for (path, got), want_leaf in zip(
                jax.tree_util.tree_leaves_with_path(g),
                jax.tree_util.tree_leaves(via_dense),
                strict=True,
            ):
                if jnp.issubdtype(got.dtype, jnp.inexact):
                    _close(
                        got,
                        want_leaf,
                        f"{op!r} grad of {name} wrt "
                        f"{jax.tree_util.keystr(path)} vs. via to_dense",
                    )


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


def check_operator(op: LinOp, *, seed: int = 0, other: LinOp | None = None) -> None:
    """Run every conformance check against one operator instance.

    Parameters
    ----------
    op
        Instance to check. Should be small enough to densify.
    seed
        Seed for the random test operands; each check draws its own keys.
    other
        Optional second instance of the same type, with the same pytree
        structure and leaf shapes but different values. :func:`check_pytree`
        then applies a family made of ``op`` and ``other`` under
        ``jax.vmap``, instead of a family of copies of ``op``.

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
    check_pytree(op, keys[5], other=other)
    check_repr(op)
    check_arithmetic(op)
    check_family(op)
