# API reference

```{eval-rst}
.. currentmodule:: pyeki
```

## pyeki.linalg

Structured linear operators. See {doc}`../user-guide/operators` for the
catalogue with costs, {doc}`../user-guide/writing-an-operator` for adding a
new structure, and {doc}`../linop-contract` for the full behavioural
contract.

### Base classes

```{eval-rst}
.. autoclass:: pyeki.linalg.LinOp
   :members:

.. autoclass:: pyeki.linalg.SquareLinOp
   :members:

.. autoclass:: pyeki.linalg.PSDLinOp
   :members:
```

### Operators defined by their own arrays

```{eval-rst}
.. autoclass:: pyeki.linalg.Identity
.. autoclass:: pyeki.linalg.PSDDiagonal
.. autoclass:: pyeki.linalg.Dense
.. autoclass:: pyeki.linalg.DenseSquare
   :members: from_matrix
.. autoclass:: pyeki.linalg.Triangular
.. autoclass:: pyeki.linalg.DensePSD
   :members: from_matrix
.. autoclass:: pyeki.linalg.PSDLowRank
```

### Operators built from other operators

```{eval-rst}
.. autoclass:: pyeki.linalg.Transposed
.. autoclass:: pyeki.linalg.Scaled
.. autoclass:: pyeki.linalg.SquareScaled
.. autoclass:: pyeki.linalg.PSDScaled
.. autoclass:: pyeki.linalg.Product
.. autoclass:: pyeki.linalg.HStack
.. autoclass:: pyeki.linalg.BlockDiag
.. autoclass:: pyeki.linalg.PSDBlockDiag
.. autoclass:: pyeki.linalg.PSDDiagCongruence
```

### Factory functions

```{eval-rst}
.. autofunction:: pyeki.linalg.block_diag
.. autofunction:: pyeki.linalg.product
.. autofunction:: pyeki.linalg.hstack
.. autofunction:: pyeki.linalg.diag_congruence
```

### Helpers for defining operators

```{eval-rst}
.. autofunction:: pyeki.linalg.linop
.. autofunction:: pyeki.linalg.static_field
.. autofunction:: pyeki.linalg.dense_matvec
.. autofunction:: pyeki.linalg.tri_solve
.. autofunction:: pyeki.linalg.densify
.. autofunction:: pyeki.linalg.set_debug_checks
.. autofunction:: pyeki.linalg.debug_checks
.. autofunction:: pyeki.linalg.value_check
.. autoexception:: pyeki.linalg.UnsupportedOpError
```

### Conformance testing

```{eval-rst}
.. automodule:: pyeki.linalg.testing
   :members: check_operator, check_core, check_transpose, check_solve,
             check_factor, check_whiten, check_scalars,
             check_dense_independence, check_capabilities,
             check_operand_validation, check_pytree, check_repr,
             check_arithmetic, check_family
```
