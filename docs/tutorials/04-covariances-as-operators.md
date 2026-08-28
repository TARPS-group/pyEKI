# 4. Covariances as operators

:::{admonition} Stub
:class: note
Not yet written. The scope below is settled; the prose is not.
:::

## Goal

The reader can express the error and prior structure their own problem actually
has, instead of the diagonal one the earlier tutorials assumed.

## Prerequisites

Tutorials 1 to 3.

## What this page covers

Lead with the need, not the abstraction. The reader arrives here because their
observations are correlated, or their prior is not diagonal, or their parameter
dimension is too large for a dense covariance to fit in memory.

- What an operator is: a matrix represented by how it acts on vectors, so
  structure is used rather than materialized. A `PSDDiagonal` of length one
  million applies in linear time; the dense equivalent does not fit in memory.
- The catalogue as a reader needs it, not exhaustively: `PSDDiagonal`,
  `DensePSD`, `Identity`, `PSDLowRank`, and `block_diag` for the common
  independent-plus-correlated case.
- The batch-axis rule — leading batch axes, core operand shape trailing — and
  that an ensemble stored `(J, n)` is just a batch.
- Whitening: what `whiten` is for, and that a noise covariance in this library
  needs only `whiten`, never `solve` or `logdet`. This is why a structured
  noise operator is cheap to use.
- `supports()` and `capabilities()`, and that `UnsupportedOpError` is raised
  rather than silently falling back to dense linear algebra.
- Tempering a structured operator: `noise / dbeta` preserves structure and
  capabilities, and the scalar may be traced.
- One worked change: take the tutorial's running problem, replace the diagonal
  noise with `block_diag` of an independent and a correlated block, and show
  that nothing else in the call changes.

## Deliberately not covered

- writing a new operator → {doc}`../user-guide/writing-an-operator`
- the full cost table → {doc}`../user-guide/operators`
- `vmap` over operators, and the operator/operand batch distinction
- the hierarchy's three levels as a design topic → {doc}`../linop-contract`

## API exercised

`pyeki.linalg`: `PSDDiagonal`, `DensePSD`, `Identity`, `PSDLowRank`,
`block_diag`, `whiten`, `factor`, `supports`, `capabilities`,
`UnsupportedOpError`, scalar division.

## Notes for the writer

:::{important}
{doc}`../user-guide/quickstart` currently covers most of this material, and was
written as the package's entry point before this series existed. Decide
explicitly whether this tutorial **absorbs** that page — with the quickstart
retired and its inbound links repointed — or whether the quickstart stays as
the operator layer's standalone reference and this page stays shorter and more
motivated. Do not silently duplicate it; that is how the two drift apart.
:::

Never write `M @ x` when demonstrating application: for arrays of two or more
dimensions it contracts the second-to-last axis and silently returns a wrong
answer when the operator is square. Use `matvec`, or
`pyeki.linalg.dense_matvec`. `op @ x` on an array raises an error saying so,
which is worth showing.
