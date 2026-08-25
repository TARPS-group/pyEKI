"""Structured linear operators.

Operators represent matrices implicitly, by how they act on vectors, so that
known structure is exploited instead of storing or factorizing dense arrays.
A block-diagonal covariance, for example, is solved block by block at a cost
that is the sum over blocks rather than cubic in the total size.

This is a lean layer aimed at what Ensemble Kalman Inversion needs — applying
operators and their transposes, solving against them, and taking square roots
to sample and whiten — rather than a general-purpose linear algebra library.
Its behaviour is specified by the "Linear operator contract" page of the
documentation.

- :mod:`~pyeki.linalg.base` defines the class hierarchy, the array-shape
  convention, and how to add a new operator.
- :mod:`~pyeki.linalg.elementary` holds operators defined by their own
  arrays.
- :mod:`~pyeki.linalg.composite` holds operators built from other operators,
  and the factory functions that construct them.
- :mod:`~pyeki.linalg.testing` holds conformance checks for new operator
  types.

Import :mod:`pyeki` before creating any array, so that float64 is enabled
first.
"""
from .base import (
    LinOp,
    PSDLinOp,
    SquareLinOp,
    UnsupportedOpError,
    debug_checks,
    dense_matvec,
    densify,
    linop,
    set_debug_checks,
    static_field,
    tri_solve,
    value_check,
)
from .composite import (
    BlockDiag,
    DiagCongruence,
    HStack,
    Product,
    PSDBlockDiag,
    PSDScaled,
    Scaled,
    SquareScaled,
    Transposed,
    block_diag,
    diag_congruence,
    hstack,
    product,
)
from .elementary import (
    Dense,
    DensePSD,
    DenseSquare,
    Identity,
    PSDDiagonal,
    Triangular,
)

__all__ = [
    # hierarchy and machinery
    "LinOp",
    "SquareLinOp",
    "PSDLinOp",
    "UnsupportedOpError",
    "densify",
    "linop",
    "static_field",
    "dense_matvec",
    "tri_solve",
    "set_debug_checks",
    "debug_checks",
    "value_check",
    # elementary operators
    "Identity",
    "PSDDiagonal",
    "Dense",
    "DenseSquare",
    "Triangular",
    "DensePSD",
    # composites
    "Transposed",
    "Scaled",
    "SquareScaled",
    "PSDScaled",
    "Product",
    "HStack",
    "BlockDiag",
    "PSDBlockDiag",
    "DiagCongruence",
    # factories
    "block_diag",
    "product",
    "hstack",
    "diag_congruence",
]
