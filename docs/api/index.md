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

## pyeki.gauss

Joint Gaussian distributions and conditioning. See
{doc}`../user-guide/conditioning` for when to use each piece, and
{doc}`../gaussian-contract` for the full behavioural contract.

```{eval-rst}
.. automodule:: pyeki.gauss
   :no-members:
```

### Distributions

```{eval-rst}
.. autoclass:: pyeki.gauss.Gaussian
   :members: from_samples, dim, batch_shape, sample, log_density

.. autoclass:: pyeki.gauss.GaussianJoint
   :members: from_linear_map, from_samples, from_factors, u_dim, v_dim,
             latent_dim, batch_shape, u_marginal, v_marginal, condition,
             pathwise

.. autoclass:: pyeki.gauss.EmpiricalJoint
   :members: n_samples, u_dim, v_dim, batch_shape, u_mean, v_mean,
             u_anomalies, v_anomalies, to_gaussian_joint, transform_update,
             pathwise_update
```

### Conditioning primitives

```{eval-rst}
.. autofunction:: pyeki.gauss.gain_weights
.. autofunction:: pyeki.gauss.sqrt_transform
```

## pyeki.eki

Ensemble Kalman Inversion: the ladder, the policies that shape it, and the
run. See {doc}`../user-guide/running-an-inversion` for when to use each piece,
and {doc}`../eki-contract` for the full behavioural contract.

```{eval-rst}
.. automodule:: pyeki.eki
   :no-members:
```

### Value classes

```{eval-rst}
.. autoclass:: pyeki.eki.EKIState
   :members: from_prior, restart, n_members, u_dim, mean, batch_shape

.. autoclass:: pyeki.eki.Evaluation
   :members: misfits, centre_misfit, n_members, u_dim, v_dim, batch_shape

.. autoclass:: pyeki.eki.HistoryRecord
   :members: batch_shape

.. autoclass:: pyeki.eki.EKIResult
   :members: ensemble, beta, mean, n_evaluations, n_completed_steps,
             min_n_valid, stacked, stop_fired, budget_complete
```

### The three axes, as protocols

```{eval-rst}
.. autoclass:: pyeki.eki.EnsembleUpdate
   :members: __call__

.. autoclass:: pyeki.eki.Schedule
   :members: next_increment

.. autoclass:: pyeki.eki.StoppingRule
   :members: __call__

.. autoclass:: pyeki.eki.Inflation
   :members: __call__
```

### Update rules

```{eval-rst}
.. autoclass:: pyeki.eki.TransformUpdate
   :members: __call__
.. autoclass:: pyeki.eki.PathwiseUpdate
   :members: __call__
```

### Schedules and stopping rules

```{eval-rst}
.. autoclass:: pyeki.eki.FixedSchedule
   :members: uniform, constant, n_steps, beta_target, next_increment
.. autoclass:: pyeki.eki.AdaptiveESSSchedule
   :members: n_steps, next_increment
.. autoclass:: pyeki.eki.AdaptiveMisfitSchedule
   :members: n_steps, next_increment
.. autoclass:: pyeki.eki.DiscrepancyStop
   :members: __call__
```

### Inflation

```{eval-rst}
.. autoclass:: pyeki.eki.MultiplicativeInflation
   :members: __call__, batch_shape
.. autoclass:: pyeki.eki.AdditiveInflation
   :members: __call__
```

### The driver, and one step

```{eval-rst}
.. autofunction:: pyeki.eki.run
.. autofunction:: pyeki.eki.iterate
.. autofunction:: pyeki.eki.evaluate
.. autofunction:: pyeki.eki.assimilate
.. autofunction:: pyeki.eki.advance
```

### Helpers, status constants, and the exception

```{eval-rst}
.. autofunction:: pyeki.eki.misfits
.. autofunction:: pyeki.eki.effective_sample_size
.. autofunction:: pyeki.eki.repair_failed_members

.. autodata:: pyeki.eki.SCHEDULE_EXHAUSTED
.. autodata:: pyeki.eki.STOPPING_RULE
.. autodata:: pyeki.eki.INTERRUPTED

.. autoexception:: pyeki.eki.EKIError
```

### Conformance testing

```{eval-rst}
.. automodule:: pyeki.eki.testing
   :members: check_schedule, check_update, check_inflation,
             check_stopping_rule, synthetic_evaluation
```
