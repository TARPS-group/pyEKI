"""Operators built from other operators.

============================  ==============================================
class                         represents
============================  ==============================================
:class:`Transposed`           the transpose of another operator
:class:`Scaled`               a scalar multiple, ``c * A``
:class:`SquareScaled`         a scalar multiple of a square operator
:class:`PSDScaled`            a positive scalar multiple of a PSD operator
:class:`Product`              a chain of operators applied in sequence
:class:`HStack`               blocks placed side by side, ``[A_1 ... A_m]``
:class:`BlockDiag`            a block-diagonal matrix of arbitrary blocks
:class:`PSDBlockDiag`         a block-diagonal matrix of PSD blocks
:class:`PSDDiagCongruence`    ``diag(s) A diag(s)`` for a PSD ``A``
============================  ==============================================

Construct through the factory functions — :func:`block_diag`,
:func:`product`, :func:`hstack`, :func:`diag_congruence` — or through
operator arithmetic (``A @ B``, ``c * A``, ``A / c``, ``A.T``), which pick
the most capable class for the ingredients. Constructing a class directly
is allowed but never upgrades to a more capable one.

See :mod:`pyeki.linalg.base` for the shape convention shared by all
operators.

Notes
-----
This layer does not track definiteness through composition: a
:class:`Product` of PSD operators is not PSD in general, so ``Product`` and
``HStack`` are plain :class:`~.base.LinOp`. An operator family that is
closed under a composition gets its own class — :class:`PSDDiagCongruence`
is the diagonal congruence, which maps PSD operators to PSD operators.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .base import (
    LinOp,
    PSDLinOp,
    SquareLinOp,
    _broadcast_batch,
    _check_core_rank,
    linop,
    value_check,
)
from .elementary import PSDDiagonal

__all__ = [
    "Transposed",
    "Scaled",
    "SquareScaled",
    "PSDScaled",
    "Product",
    "HStack",
    "BlockDiag",
    "PSDBlockDiag",
    "PSDDiagCongruence",
    "block_diag",
    "product",
    "hstack",
    "diag_congruence",
]


def _check_ops_tuple(cls_name: str, field_name: str, ops, required=LinOp) -> None:
    """Structural check for a tuple-of-operators field."""
    if not isinstance(ops, tuple):
        raise TypeError(f"{cls_name}.{field_name} must be a tuple of operators")
    if not ops:
        raise ValueError(f"{cls_name} needs at least one operator")
    for op in ops:
        if not isinstance(op, required):
            raise TypeError(
                f"{cls_name} blocks must be {required.__name__}, "
                f"got {type(op).__name__}"
            )


# ---------------------------------------------------------------------------
# views: transpose and scalar scaling
# ---------------------------------------------------------------------------


@linop
class Transposed(LinOp):
    """The transpose of another operator, as a view.

    What :attr:`~pyeki.linalg.LinOp.T` returns by default. A plain ``LinOp``
    regardless of the wrapped operator's level: transposition preserves
    solvability but not the layer's knowledge of it, and operators whose
    transpose supports more override ``T`` with a structured result instead.

    Parameters
    ----------
    op
        The operator to transpose.
    """

    op: LinOp

    def __post_init__(self) -> None:
        if not isinstance(self.op, LinOp):
            raise TypeError(
                f"Transposed wraps a LinOp, got {type(self.op).__name__}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        n_out, n_in = self.op.shape
        return (n_in, n_out)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.op.batch_shape

    @property
    def T(self) -> LinOp:  # noqa: N802 - mirrors the NumPy attribute
        """The original operator, since transposing a view unwraps it."""
        return self.op

    def _matvec(self, x: Array) -> Array:
        return self.op._rmatvec(x)

    def _rmatvec(self, x: Array) -> Array:
        return self.op._matvec(x)

    def _matmat(self, X: Array) -> Array:
        return self.op._rmatmat(X)

    def _rmatmat(self, X: Array) -> Array:
        return self.op._matmat(X)

    def _to_dense(self) -> Array:
        return self.op._to_dense().swapaxes(-1, -2)


@linop
class Scaled(LinOp):
    """A scalar multiple of another operator, ``c * A``.

    Built by the arithmetic ``c * op``, ``op * c`` and ``op / c``, which
    select this class or one of its level-preserving subclasses
    (:class:`SquareScaled`, :class:`PSDScaled`) to match the operand, and
    fold nested scalings into a single wrapper. Every capability delegates
    to the base operator with the scalar folded in.

    Parameters
    ----------
    op
        The operator to scale.
    c
        Scalar array (0-d). Held as an array so a traced value — a
        tempering increment chosen inside a ``jit``-ed step — flows through
        without rebuilding the base operator.
    """

    op: LinOp
    c: Array

    def __post_init__(self) -> None:
        if not isinstance(self.op, LinOp):
            raise TypeError(f"Scaled wraps a LinOp, got {type(self.op).__name__}")
        _check_core_rank("Scaled", "c", self.c, 0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.op.shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch(
            type(self).__name__, self.op.batch_shape, self.c.shape
        )

    def supports(self, name: str) -> bool:
        return super().supports(name) and self.op.supports(name)

    def _matvec(self, x: Array) -> Array:
        return self.c * self.op._matvec(x)

    def _rmatvec(self, x: Array) -> Array:
        return self.c * self.op._rmatvec(x)

    def _matmat(self, X: Array) -> Array:
        return self.c * self.op._matmat(X)

    def _rmatmat(self, X: Array) -> Array:
        return self.c * self.op._rmatmat(X)

    def _to_dense(self) -> Array:
        return self.c * self.op._to_dense()


@linop
class SquareScaled(Scaled, SquareLinOp):
    """A scalar multiple of a square operator; adds the square operations."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.op, SquareLinOp):
            raise TypeError(
                f"SquareScaled wraps a SquareLinOp, got {type(self.op).__name__}"
            )
        value_check(
            self.c,
            lambda c: bool(jnp.all(c != 0)),
            "scaling a square operator requires a nonzero scalar",
        )

    def _solve(self, b: Array) -> Array:
        return self.op._solve(b) / self.c

    def _solve_mat(self, B: Array) -> Array:
        return self.op._solve_mat(B) / self.c

    def _logdet(self) -> Array:
        return self.n * jnp.log(jnp.abs(self.c)) + self.op._logdet()

    def _diag(self) -> Array:
        return self.c * self.op._diag()


@linop
class PSDScaled(SquareScaled, PSDLinOp):
    """A positive scalar multiple of a PSD operator; adds the PSD operations.

    The scalar must be strictly positive, or the result is not PSD; this is
    a value precondition, checked only in debug mode.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.op, PSDLinOp):
            raise TypeError(
                f"PSDScaled wraps a PSDLinOp, got {type(self.op).__name__}"
            )
        value_check(
            self.c,
            lambda c: bool(jnp.all(c > 0)),
            "scaling a PSD operator requires a strictly positive scalar",
        )

    def _factor(self) -> LinOp:
        return self.op._factor() * jnp.sqrt(self.c)

    def _whiten(self, x: Array) -> Array:
        return self.op._whiten(x) / jnp.sqrt(self.c)

    def _whiten_mat(self, X: Array) -> Array:
        return self.op._whiten_mat(X) / jnp.sqrt(self.c)


# ---------------------------------------------------------------------------
# composition and stacking
# ---------------------------------------------------------------------------


@linop
class Product(LinOp):
    """A chain of operators, applied right to left.

    Parameters
    ----------
    ops
        Operators to compose, with adjacent shapes agreeing. The last is
        applied first, like matrix multiplication.
    """

    ops: tuple[LinOp, ...]

    def __post_init__(self) -> None:
        _check_ops_tuple("Product", "ops", self.ops)
        for a, b in zip(self.ops[:-1], self.ops[1:], strict=True):
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"shape mismatch in Product: {a!r} @ {b!r}")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ops[0].shape[0], self.ops[-1].shape[1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch("Product", *[op.batch_shape for op in self.ops])

    def _matvec(self, x: Array) -> Array:
        for op in reversed(self.ops):
            x = op.matvec(x)
        return x

    def _rmatvec(self, x: Array) -> Array:
        for op in self.ops:
            x = op.rmatvec(x)
        return x

    def _matmat(self, X: Array) -> Array:
        for op in reversed(self.ops):
            X = op.matmat(X)
        return X

    def _rmatmat(self, X: Array) -> Array:
        for op in self.ops:
            X = op.rmatmat(X)
        return X

    def _to_dense(self) -> Array:
        out = self.ops[-1].to_dense()
        for op in reversed(self.ops[:-1]):
            out = op.to_dense() @ out
        return out


@linop
class HStack(LinOp):
    """Blocks placed side by side, ``[A_1  A_2  ...  A_m]``.

    A block *row*: the operand is split along its trailing axis, one piece
    per block, and the blocks' outputs are summed, so that
    ``[A_1 A_2] @ [x_1; x_2] == A_1 x_1 + A_2 x_2``. The transpose,
    ``hstack(...).T``, is the corresponding block column.

    Parameters
    ----------
    ops
        Blocks, all with the same number of rows. Column counts may differ.
    """

    ops: tuple[LinOp, ...]

    def __post_init__(self) -> None:
        _check_ops_tuple("HStack", "ops", self.ops)
        rows = {op.shape[0] for op in self.ops}
        if len(rows) != 1:
            raise ValueError(
                f"HStack blocks must share a row count, got "
                f"{[op.shape for op in self.ops]}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ops[0].shape[0], sum(op.shape[1] for op in self.ops))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch("HStack", *[op.batch_shape for op in self.ops])

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """The blocks' shapes, in order."""
        return tuple(op.shape for op in self.ops)

    @property
    def _splits(self) -> list[int]:
        out, acc = [], 0
        for op in self.ops[:-1]:
            acc += op.shape[1]
            out.append(acc)
        return out

    def _matvec(self, x: Array) -> Array:
        chunks = jnp.split(x, self._splits, axis=-1)
        total = self.ops[0].matvec(chunks[0])
        for op, chunk in zip(self.ops[1:], chunks[1:], strict=True):
            total = total + op.matvec(chunk)
        return total

    def _rmatvec(self, x: Array) -> Array:
        return jnp.concatenate([op.rmatvec(x) for op in self.ops], axis=-1)

    def _to_dense(self) -> Array:
        return jnp.concatenate([op.to_dense() for op in self.ops], axis=-1)


# ---------------------------------------------------------------------------
# block diagonals
# ---------------------------------------------------------------------------


@linop
class BlockDiag(LinOp):
    """A block-diagonal matrix whose blocks may be rectangular.

    What ``PSDBlockDiag.factor()`` returns, since per-block factors need
    not be square. Provides application and transposition only; use
    :class:`PSDBlockDiag` when every block is PSD.

    Parameters
    ----------
    blocks
        Operators in order along the diagonal.
    """

    blocks: tuple[LinOp, ...]

    def __post_init__(self) -> None:
        _check_ops_tuple("BlockDiag", "blocks", self.blocks)

    @property
    def shape(self) -> tuple[int, int]:
        return (
            sum(b.shape[0] for b in self.blocks),
            sum(b.shape[1] for b in self.blocks),
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch("BlockDiag", *[b.batch_shape for b in self.blocks])

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """The blocks' shapes, in order along the diagonal."""
        return tuple(b.shape for b in self.blocks)

    def _apply_blockwise(self, x: Array, method: str, axis: int) -> Array:
        splits, acc = [], 0
        for b in self.blocks[:-1]:
            acc += b.shape[axis]
            splits.append(acc)
        chunks = jnp.split(x, splits, axis=-1)
        return jnp.concatenate(
            [
                getattr(b, method)(chunk)
                for b, chunk in zip(self.blocks, chunks, strict=True)
            ],
            axis=-1,
        )

    def _matvec(self, x: Array) -> Array:
        return self._apply_blockwise(x, "matvec", axis=1)

    def _rmatvec(self, x: Array) -> Array:
        return self._apply_blockwise(x, "rmatvec", axis=0)

    def _to_dense(self) -> Array:
        return jax.scipy.linalg.block_diag(*[b.to_dense() for b in self.blocks])


@linop
class PSDBlockDiag(PSDLinOp):
    """A block-diagonal matrix whose blocks are all PSD.

    Every operation is applied blockwise, so cost is the sum over blocks
    rather than cubic in the total size. Which operations are available
    depends on the blocks: this operator supports ``solve``, for example,
    only if all of its blocks do — check with ``op.supports(name)``.

    Parameters
    ----------
    blocks
        Square PSD operators, in order along the diagonal.
    """

    blocks: tuple[PSDLinOp, ...]

    def __post_init__(self) -> None:
        _check_ops_tuple("PSDBlockDiag", "blocks", self.blocks, required=PSDLinOp)

    @property
    def shape(self) -> tuple[int, int]:
        n = sum(b.shape[0] for b in self.blocks)
        return (n, n)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch(
            "PSDBlockDiag", *[b.batch_shape for b in self.blocks]
        )

    @property
    def block_shapes(self) -> tuple[tuple[int, int], ...]:
        """The blocks' shapes, in order along the diagonal."""
        return tuple(b.shape for b in self.blocks)

    def supports(self, name: str) -> bool:
        return super().supports(name) and all(
            b.supports(name) for b in self.blocks
        )

    @property
    def _splits(self) -> list[int]:
        out, acc = [], 0
        for b in self.blocks[:-1]:
            acc += b.shape[0]
            out.append(acc)
        return out

    def _apply_blockwise(self, x: Array, method: str) -> Array:
        chunks = jnp.split(x, self._splits, axis=-1)
        return jnp.concatenate(
            [
                getattr(b, method)(chunk)
                for b, chunk in zip(self.blocks, chunks, strict=True)
            ],
            axis=-1,
        )

    def _matvec(self, x: Array) -> Array:
        return self._apply_blockwise(x, "matvec")

    def _solve(self, b: Array) -> Array:
        return self._apply_blockwise(b, "solve")

    def _whiten(self, x: Array) -> Array:
        return self._apply_blockwise(x, "whiten")

    def _logdet(self) -> Array:
        return sum(b.logdet() for b in self.blocks)

    def _diag(self) -> Array:
        return jnp.concatenate([b.diag() for b in self.blocks], axis=-1)

    def _factor(self) -> LinOp:
        return BlockDiag(tuple(b.factor() for b in self.blocks))

    def _to_dense(self) -> Array:
        return jax.scipy.linalg.block_diag(*[b.to_dense() for b in self.blocks])


# ---------------------------------------------------------------------------
# diagonal congruence
# ---------------------------------------------------------------------------


@linop
class PSDDiagCongruence(PSDLinOp):
    """``diag(s) A diag(s)`` for a PSD operator ``A`` and positive ``s``.

    A diagonal congruence of a PSD operator is PSD, and every capability
    delegates to the base operator with the diagonal folded in, so per-entry
    rescaling — inflating observation noise by a reciprocal taper, say —
    costs no more than the original operator.

    Parameters
    ----------
    op
        The PSD operator to rescale.
    scale
        Per-coordinate scale vector of length ``op.n``, strictly positive.
    """

    op: PSDLinOp
    scale: Array

    def __post_init__(self) -> None:
        if not isinstance(self.op, PSDLinOp):
            raise TypeError(
                f"PSDDiagCongruence wraps a PSDLinOp, got {type(self.op).__name__}"
            )
        _check_core_rank("PSDDiagCongruence", "scale", self.scale, 1)
        if getattr(self.scale, "ndim", None) is not None:
            if self.scale.shape[-1] != self.op.shape[0]:
                raise ValueError(
                    f"PSDDiagCongruence: scale length {self.scale.shape[-1]} does not "
                    f"match operator side {self.op.shape[0]}"
                )
        value_check(
            self.scale,
            lambda s: bool(jnp.all(s > 0)),
            "PSDDiagCongruence.scale must be strictly positive",
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.op.shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return _broadcast_batch(
            "PSDDiagCongruence", self.op.batch_shape, self.scale.shape[:-1]
        )

    def supports(self, name: str) -> bool:
        return super().supports(name) and self.op.supports(name)

    def _matvec(self, x: Array) -> Array:
        return self.scale * self.op._matvec(self.scale * x)

    def _solve(self, b: Array) -> Array:
        return self.op._solve(b / self.scale) / self.scale

    def _logdet(self) -> Array:
        return 2.0 * jnp.sum(jnp.log(self.scale), axis=-1) + self.op._logdet()

    def _diag(self) -> Array:
        return self.scale * self.scale * self.op._diag()

    def _factor(self) -> LinOp:
        return Product((PSDDiagonal(self.scale), self.op._factor()))

    def _whiten(self, x: Array) -> Array:
        return self.op._whiten(x / self.scale)

    def _to_dense(self) -> Array:
        D = self.op._to_dense()
        return self.scale[..., :, None] * D * self.scale[..., None, :]


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------


def _check_factory_args(name: str, ops: tuple) -> None:
    if not ops:
        raise ValueError(f"{name}() needs at least one operator")
    for op in ops:
        if not isinstance(op, LinOp):
            raise TypeError(
                f"{name}() arguments must be operators, got {type(op).__name__}"
            )


def block_diag(*blocks: LinOp) -> LinOp:
    """Build a block-diagonal operator, choosing the most capable class.

    Returns :class:`PSDBlockDiag` when every block is a
    :class:`~.base.PSDLinOp` and :class:`BlockDiag` otherwise. A single
    block is returned unchanged.
    """
    _check_factory_args("block_diag", blocks)
    if len(blocks) == 1:
        return blocks[0]
    if all(isinstance(b, PSDLinOp) for b in blocks):
        return PSDBlockDiag(tuple(blocks))
    return BlockDiag(tuple(blocks))


def product(*ops: LinOp) -> LinOp:
    """Compose operators right to left; what ``A @ B`` builds.

    A single operator is returned unchanged.

    Notes
    -----
    A structured square variant will be added when a consumer needs
    ``solve`` or ``logdet`` through a product.
    """
    _check_factory_args("product", ops)
    if len(ops) == 1:
        return ops[0]
    return Product(tuple(ops))


def hstack(*ops: LinOp) -> LinOp:
    """Place blocks side by side as a block row.

    A single operator is returned unchanged.
    """
    _check_factory_args("hstack", ops)
    if len(ops) == 1:
        return ops[0]
    return HStack(tuple(ops))


def diag_congruence(op: PSDLinOp, scale) -> PSDDiagCongruence:
    """Build ``diag(s) op diag(s)`` for a PSD operator and positive ``s``.

    Parameters
    ----------
    op
        The PSD operator to rescale.
    scale
        Per-coordinate scale vector of length ``op.n``, strictly positive.
    """
    if not isinstance(op, PSDLinOp):
        raise TypeError(
            f"diag_congruence() requires a PSDLinOp, got {type(op).__name__}"
        )
    scale = jnp.asarray(scale)
    if scale.ndim != 1 or scale.shape[0] != op.shape[0]:
        raise ValueError(
            f"diag_congruence(): expected a scale vector of shape "
            f"({op.shape[0]},), got {scale.shape}"
        )
    return PSDDiagCongruence(op, scale)
