"""Operators built from other operators.

============================  ==============================================
class                         represents
============================  ==============================================
:class:`Product`              a chain of operators applied in sequence
:class:`HStack`               blocks placed side by side, ``[A_1 ... A_m]``
:class:`BlockDiag`            a block-diagonal matrix of PSD blocks
:class:`BlockDiagGeneral`     a block-diagonal matrix of arbitrary blocks
:class:`Kron`                 a Kronecker product of two PSD factors
:class:`KronGeneral`          a Kronecker product of two arbitrary factors
============================  ==============================================

See :mod:`pyeki.linalg.base` for the shape convention shared by
all operators.

``Product``, ``HStack`` and ``KronGeneral`` are what ``factor()`` returns for
composed operators, since a square root distributes over composition:

- the factor of a sum is its factors side by side, an ``HStack``;
- the factor of ``D A D`` is ``D`` times the factor of ``A``, a ``Product``;
- the factor of ``A (x) B`` is ``factor(A) (x) factor(B)``, a ``KronGeneral``.

Notes
-----
``Product`` and ``HStack`` are plain ``LinOp``, not ``PSDOperator``, because
composing PSD operators does not generally give a PSD result and this layer
does not track definiteness through composition. An operator that stays PSD
under composition, such as a symmetric congruence ``diag(d) A diag(d)``, needs
its own class rather than being expressed as a ``Product``.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from .base import LinOp, PSDOperator, operator

__all__ = [
    "Product",
    "HStack",
    "BlockDiag",
    "BlockDiagGeneral",
    "Kron",
    "KronGeneral",
]


@operator
class Product(LinOp):
    """A chain of operators, applied right to left.

    Parameters
    ----------
    ops
        Operators to compose, with adjacent shapes agreeing. The last is
        applied first.
    """

    ops: tuple[LinOp, ...]

    def __post_init__(self) -> None:
        if not self.ops:
            raise ValueError("Product needs at least one operator")
        for a, b in zip(self.ops[:-1], self.ops[1:], strict=True):
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"shape mismatch in Product: {a.shape} @ {b.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ops[0].shape[0], self.ops[-1].shape[1])

    def matvec(self, x: Array) -> Array:
        for op in reversed(self.ops):
            x = op.matvec(x)
        return x

    def to_dense(self) -> Array:
        out = self.ops[-1].to_dense()
        for op in reversed(self.ops[:-1]):
            out = op.to_dense() @ out
        return out


@operator
class HStack(LinOp):
    """Blocks placed side by side, ``[A_1  A_2  ...  A_m]``.

    The input is split along its trailing axis, one piece per block, and the
    blocks' outputs are summed, so that
    ``[A_1 A_2] @ [x_1; x_2] == A_1 x_1 + A_2 x_2``. This is a block *row*;
    it does not apply each block to the whole input.

    Parameters
    ----------
    ops
        Blocks, all with the same number of rows. Column counts may differ.
    """

    ops: tuple[LinOp, ...]

    def __post_init__(self) -> None:
        if not self.ops:
            raise ValueError("HStack needs at least one operator")
        rows = {op.shape[0] for op in self.ops}
        if len(rows) != 1:
            raise ValueError(
                f"HStack blocks must share a row count, got {[o.shape for o in self.ops]}"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ops[0].shape[0], sum(op.shape[1] for op in self.ops))

    @property
    def _splits(self) -> list[int]:
        out, acc = [], 0
        for op in self.ops[:-1]:
            acc += op.shape[1]
            out.append(acc)
        return out

    def matvec(self, x: Array) -> Array:
        chunks = jnp.split(x, self._splits, axis=-1)
        total = self.ops[0].matvec(chunks[0])
        for op, chunk in zip(self.ops[1:], chunks[1:], strict=True):
            total = total + op.matvec(chunk)
        return total

    def to_dense(self) -> Array:
        return jnp.concatenate([op.to_dense() for op in self.ops], axis=-1)


@operator
class BlockDiag(PSDOperator):
    """A block-diagonal matrix whose blocks are all PSD.

    Every operation is applied blockwise, so cost is the sum over blocks
    rather than cubic in the total size.

    Which operations are available depends on the blocks: this operator
    supports ``solve`` only if all of its blocks do. Check with
    ``op.supports(name)`` before calling.

    Parameters
    ----------
    blocks
        Square PSD operators, in order along the diagonal.
    """

    blocks: tuple[PSDOperator, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("BlockDiag needs at least one block")
        for b in self.blocks:
            if not isinstance(b, PSDOperator):
                raise TypeError(
                    f"BlockDiag blocks must be PSDOperator, got {type(b).__name__}"
                )

    @property
    def shape(self) -> tuple[int, int]:
        n = sum(b.shape[0] for b in self.blocks)
        return (n, n)

    @property
    def _splits(self) -> list[int]:
        out, acc = [], 0
        for b in self.blocks[:-1]:
            acc += b.shape[0]
            out.append(acc)
        return out

    def supports(self, name: str) -> bool:
        return super().supports(name) and all(b.supports(name) for b in self.blocks)

    def _blockwise(self, x: Array, method: str) -> Array:
        chunks = jnp.split(x, self._splits, axis=-1)
        return jnp.concatenate(
            [getattr(b, method)(c) for b, c in zip(self.blocks, chunks, strict=True)],
            axis=-1,
        )

    def matvec(self, x: Array) -> Array:
        return self._blockwise(x, "matvec")

    def solve(self, b: Array) -> Array:
        return self._blockwise(b, "solve")

    def whiten(self, x: Array) -> Array:
        return self._blockwise(x, "whiten")

    def factor(self) -> LinOp:
        return BlockDiagGeneral(tuple(b.factor() for b in self.blocks))

    def cholesky(self) -> LinOp:
        return BlockDiagGeneral(tuple(b.cholesky() for b in self.blocks))

    def diag(self) -> Array:
        return jnp.concatenate([b.diag() for b in self.blocks], axis=-1)

    def logdet(self) -> Array:
        return sum(b.logdet() for b in self.blocks)

    def to_dense(self) -> Array:
        return _dense_block_diag([b.to_dense() for b in self.blocks])


@operator
class BlockDiagGeneral(LinOp):
    """A block-diagonal matrix whose blocks may be rectangular.

    Returned by :meth:`BlockDiag.factor` and :meth:`BlockDiag.cholesky`, whose
    per-block factors need not be square.

    Parameters
    ----------
    blocks
        Operators in order along the diagonal.
    """

    blocks: tuple[LinOp, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (
            sum(b.shape[0] for b in self.blocks),
            sum(b.shape[1] for b in self.blocks),
        )

    def matvec(self, x: Array) -> Array:
        splits, acc = [], 0
        for b in self.blocks[:-1]:
            acc += b.shape[1]
            splits.append(acc)
        chunks = jnp.split(x, splits, axis=-1)
        return jnp.concatenate(
            [b.matvec(c) for b, c in zip(self.blocks, chunks, strict=True)], axis=-1
        )

    def to_dense(self) -> Array:
        return _dense_block_diag([b.to_dense() for b in self.blocks])


@operator
class Kron(PSDOperator):
    """A Kronecker product of two square PSD factors.

    The entry at row ``i * n_B + k`` and column ``j * n_B + l`` is
    ``A[i, j] * B[k, l]``, so block ``(i, j)`` of the result is ``A[i, j] * B``
    and the first factor's index varies most slowly. This is the ordering
    ``numpy.kron`` uses.

    Every operation is carried out on the two factors separately, so nothing of
    side ``n_A * n_B`` is ever formed or factorized. Which operations are
    available depends on the factors: this operator supports ``solve`` only if
    both factors do. Check with ``op.supports(name)`` before calling.

    Parameters
    ----------
    A
        Square PSD operator, the factor whose index varies most slowly.
    B
        Square PSD operator, the factor whose index varies fastest.

    Raises
    ------
    TypeError
        If either factor is not a :class:`~.base.PSDOperator`.
    ValueError
        If either factor is not square.

    Notes
    -----
    The ordering above is stated explicitly because reversing it is silent
    rather than loud. ``B (x) A`` is a different matrix, but it is positive
    definite whenever ``A (x) B`` is, and it has the same shape when the two
    factors have the same size, so a reversed implementation returns wrong
    numbers without raising.
    """

    A: PSDOperator
    B: PSDOperator

    def __post_init__(self) -> None:
        for name, f in (("A", self.A), ("B", self.B)):
            if not isinstance(f, PSDOperator):
                raise TypeError(
                    f"Kron factor {name} must be PSDOperator, got {type(f).__name__}"
                )
            if f.shape[0] != f.shape[1]:
                raise ValueError(
                    f"Kron factor {name} must be square, got shape {f.shape}"
                )

    @property
    def shape(self) -> tuple[int, int]:
        n = self.A.shape[0] * self.B.shape[0]
        return (n, n)

    @property
    def _core(self) -> tuple[int, int]:
        """Core shape the trailing axis is reshaped to, ``(n_A, n_B)``."""
        return (self.A.shape[0], self.B.shape[0])

    def supports(self, name: str) -> bool:
        return (
            super().supports(name)
            and self.A.supports(name)
            and self.B.supports(name)
        )

    def matvec(self, x: Array) -> Array:
        return _kron_apply(x, self.A.matvec, self.B.matvec, self._core, self._core)

    def solve(self, b: Array) -> Array:
        # (A (x) B)^-1 == A^-1 (x) B^-1.
        return _kron_apply(b, self.A.solve, self.B.solve, self._core, self._core)

    def whiten(self, x: Array) -> Array:
        # (L_A (x) L_B)^-1 == L_A^-1 (x) L_B^-1, applied factor by factor rather
        # than by solving against the assembled factor.
        return _kron_apply(x, self.A.whiten, self.B.whiten, self._core, self._core)

    def factor(self) -> LinOp:
        return KronGeneral(self.A.factor(), self.B.factor())

    def cholesky(self) -> LinOp:
        return KronGeneral(self.A.cholesky(), self.B.cholesky())

    def diag(self) -> Array:
        return jnp.kron(self.A.diag(), self.B.diag())

    def logdet(self) -> Array:
        n_A, n_B = self._core
        # det(A (x) B) == det(A)^n_B * det(B)^n_A: each factor's contribution is
        # scaled by the size of the *other* factor. Swapping the two is silent
        # whenever n_A == n_B.
        return n_B * self.A.logdet() + n_A * self.B.logdet()

    def to_dense(self) -> Array:
        return jnp.kron(self.A.to_dense(), self.B.to_dense())


@operator
class KronGeneral(LinOp):
    """A Kronecker product of two operators, either of which may be rectangular.

    Ordering follows :class:`Kron`: the entry at row ``i * n_B + k`` and column
    ``j * k_B + l`` is ``A[i, j] * B[k, l]``.

    Returned by :meth:`Kron.factor` and :meth:`Kron.cholesky`, because a square
    root of a Kronecker product is the Kronecker product of the factors' square
    roots, and those need not be square. Provides ``matvec``, ``matmat`` and
    ``to_dense`` only.

    Parameters
    ----------
    A
        The slow factor, of shape ``(n_A, k_A)``.
    B
        The fast factor, of shape ``(n_B, k_B)``.

    Notes
    -----
    The operand is reshaped to ``(k_A, k_B)`` and the result to
    ``(n_A, n_B)``. These differ whenever either factor is rectangular, with a
    mixed intermediate of shape ``(k_A, n_B)``, so an implementation that
    reshapes operand and result alike is correct only in the square case.
    """

    A: LinOp
    B: LinOp

    @property
    def shape(self) -> tuple[int, int]:
        return (
            self.A.shape[0] * self.B.shape[0],
            self.A.shape[1] * self.B.shape[1],
        )

    def matvec(self, x: Array) -> Array:
        return _kron_apply(
            x,
            self.A.matvec,
            self.B.matvec,
            (self.A.shape[1], self.B.shape[1]),
            (self.A.shape[0], self.B.shape[0]),
        )

    def to_dense(self) -> Array:
        return jnp.kron(self.A.to_dense(), self.B.to_dense())



def _dense_block_diag(mats: list[Array]) -> Array:
    """Assemble a dense block-diagonal array from dense blocks."""
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = jnp.zeros((rows, cols), dtype=mats[0].dtype)
    r = c = 0
    for m in mats:
        out = out.at[r : r + m.shape[0], c : c + m.shape[1]].set(m)
        r, c = r + m.shape[0], c + m.shape[1]
    return out


def _kron_apply(
    x: Array,
    left: Callable[[Array], Array],
    right: Callable[[Array], Array],
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
) -> Array:
    """Apply a Kronecker product to the trailing axis of ``x``.

    Reshapes the trailing axis of ``x`` to ``in_shape``, applies ``right``
    along the resulting trailing axis and ``left`` along the axis before it,
    then flattens the ``out_shape`` result back to a single trailing axis.

    Parameters
    ----------
    x
        Operand whose trailing axis has length ``in_shape[0] * in_shape[1]``,
        preceded by any number of batch axes.
    left, right
        Callables applying the slow and fast factors respectively, each
        contracting the trailing axis of its own argument.
    in_shape, out_shape
        Core shapes of the operand and of the result.

    Notes
    -----
    The two shapes are separate parameters because a rectangular Kronecker
    product reshapes its operand and its result differently, passing through a
    mixed intermediate of shape ``(in_shape[0], out_shape[1])``. Reusing one
    shape for both is correct only when both factors are square.
    """
    batch = x.shape[:-1]
    X = x.reshape(*batch, *in_shape)
    Y = right(X)                                     # (..., in_shape[0], out_shape[1])
    Z = left(Y.swapaxes(-1, -2)).swapaxes(-1, -2)    # (..., out_shape[0], out_shape[1])
    return Z.reshape(*batch, out_shape[0] * out_shape[1])
