"""Structured linear operators.

Operators represent matrices implicitly, by how they act on vectors, so that
known structure can be exploited instead of storing or factorizing dense
arrays. A block-diagonal covariance, for example, is solved block by block at a
cost that is the sum over blocks rather than cubic in the total size.

This is a lean layer aimed at what Ensemble Kalman Inversion needs — applying
operators, solving against them, and taking square roots to sample and whiten —
rather than a general-purpose linear algebra library.

- :mod:`~pyeki.linalg.base` defines the class hierarchy, the array-shape
  convention, and how to add a new operator.
- :mod:`~pyeki.linalg.leaves` holds operators defined by their own arrays.
- :mod:`~pyeki.linalg.composite` holds operators built from other operators.
- :mod:`~pyeki.linalg.testing` holds conformance checks for new operator types.

Import :mod:`pyeki` before creating any array, so that float64 is enabled
first.
"""
from .base import (
    LinOp,
    PSDOperator,
    SquareLinOp,
    UnsupportedOpError,
    dense_matvec,
    densify,
    operator,
    static_field,
    tri_solve,
)
from .composite import BlockDiag, BlockDiagGeneral, HStack, Product
from .leaves import Dense, DensePSD, Diagonal, Identity, ScaledIdentity, Triangular

__all__ = [
    # base
    "LinOp",
    "SquareLinOp",
    "PSDOperator",
    "UnsupportedOpError",
    "densify",
    "operator",
    "static_field",
    "dense_matvec",
    "tri_solve",
    # leaves
    "Identity",
    "ScaledIdentity",
    "Diagonal",
    "Dense",
    "Triangular",
    "DensePSD",
    # composites
    "Product",
    "HStack",
    "BlockDiag",
    "BlockDiagGeneral",
]
