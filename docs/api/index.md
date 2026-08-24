# API reference

```{eval-rst}
.. currentmodule:: pyeki
```

## pyeki.linalg

Structured linear operators. See {doc}`../user-guide/operators` for the
catalogue with costs, and {doc}`../user-guide/writing-an-operator` for adding a
new structure.

### Base classes

```{eval-rst}
.. autoclass:: pyeki.linalg.LinOp
   :members:

.. autoclass:: pyeki.linalg.SquareLinOp
   :members:

.. autoclass:: pyeki.linalg.PSDOperator
   :members:
```

### Operators defined by their own arrays

```{eval-rst}
.. autoclass:: pyeki.linalg.Identity
.. autoclass:: pyeki.linalg.ScaledIdentity
.. autoclass:: pyeki.linalg.Diagonal
.. autoclass:: pyeki.linalg.Dense
.. autoclass:: pyeki.linalg.Triangular
.. autoclass:: pyeki.linalg.DensePSD
   :members: from_matrix
```

### Operators built from other operators

```{eval-rst}
.. autoclass:: pyeki.linalg.Product
.. autoclass:: pyeki.linalg.HStack
.. autoclass:: pyeki.linalg.BlockDiag
.. autoclass:: pyeki.linalg.BlockDiagGeneral
```

### Helpers for defining operators

```{eval-rst}
.. autofunction:: pyeki.linalg.operator
.. autofunction:: pyeki.linalg.static_field
.. autofunction:: pyeki.linalg.dense_matvec
.. autofunction:: pyeki.linalg.tri_solve
.. autofunction:: pyeki.linalg.densify
.. autoexception:: pyeki.linalg.UnsupportedOpError
```

### Conformance testing

```{eval-rst}
.. automodule:: pyeki.linalg.testing
   :members: check_operator, check_matvec, check_matmat, check_solve,
             check_factor, check_scalars, check_pytree, check_dense_independent
```
