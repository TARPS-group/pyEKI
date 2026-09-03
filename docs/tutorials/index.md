# Tutorials

A series to read in order. Each one builds on the last and ends where the next
begins.

The first tutorial runs a complete inversion and assumes nothing — no
familiarity with EKI, and no knowledge of anything else in pyEKI. Structured
operators, tempering schedules and ensemble diagnostics are each introduced at
the point where the problem needs them, not before.

The series works the problems in {doc}`../user-guide/toy-models`, so a
tutorial can be followed with nothing of your own to hand and the same problem
can be carried from one page to the next. Those models are for learning and
for tests, not for production.

If you already know what you are looking for, the {doc}`../user-guide/quickstart`
and the rest of the user guide are organized by question rather than by
sequence, and the three contracts in the reference section specify behaviour
exactly.

```{toctree}
:maxdepth: 1

01-first-inversion
02-reading-a-run
03-sampling-or-optimizing
04-tempering-schedules
05-transform-or-pathwise
06-covariances-as-operators
07-small-ensembles
08-when-the-model-fails
09-your-own-policy
```

## What each one covers

| tutorial | you will be able to |
| --- | --- |
| {doc}`01-first-inversion` | run an inversion on your own forward model |
| {doc}`02-reading-a-run` | tell whether the answer is trustworthy |
| {doc}`03-sampling-or-optimizing` | choose between the two forms of EKI |
| {doc}`04-tempering-schedules` | choose how gradually to assimilate the data |
| {doc}`05-transform-or-pathwise` | choose between the two ensemble updates |
| {doc}`06-covariances-as-operators` | express correlated and structured error |
| {doc}`07-small-ensembles` | recognize and mitigate the ensemble-size limit |
| {doc}`08-when-the-model-fails` | survive forward models that crash or return `nan` |
| {doc}`09-your-own-policy` | write and validate a custom schedule or update |

:::{admonition} Being written
:class: note

Tutorials 1 to 3 are written. Tutorials 4 to 9 are stubs: each states its
scope, its prerequisites and the API it exercises, so the series can be
written in order without re-deciding the structure.
:::
