# Writing a forward model

pyEKI supplies everything in a run except the forward model. This page is what
that callable must satisfy; {ref}`eki-failures` is the normative statement.
There is nothing to subclass and nothing to register: pyEKI ships no forward
models and defines no base class or protocol for one.

## The interface

```python
import pyeki  # enables float64; import before creating arrays
import jax.numpy as jnp

times = jnp.array([0.5, 1.0, 2.0])

def forward(ensemble):                    # (J, 2) in
    return ensemble[:, 0:1] * jnp.exp(-ensemble[:, 1:2] * times)   # (J, 3) out
```

That is a complete forward model.

:::{important}
**`forward` receives the whole ensemble, not one member.** It is called once
per step with a `(J, P)` array and returns `(J, N)` — one prediction per
member, in the same order. A function written for a single parameter vector is
the most common mistake here, and it fails badly: depending on the arithmetic
it either raises from deep inside JAX without mentioning the ensemble, or
broadcasts to a plausible wrong shape.
:::

If your model is naturally per-member, wrap it:

```python
import jax

def one_member(u):                        # (P,) -> (N,)
    return u[0] * jnp.exp(-u[1] * times)

forward = jax.vmap(one_member)            # (J, P) -> (J, N)
```

`jax.vmap` works there because that model is JAX. For one that is not — a
subprocess, a scheduler submission, a legacy binary — the wrapper is an
ordinary Python loop, which is equally legal.

## The whole obligation

| what | requirement |
| --- | --- |
| **argument** | one positional argument: a concrete `jax.Array`, shape exactly `(J, P)`, in the run's working dtype (`float64` under the package default) |
| **return** | any array-like of shape `(J, N)`: a `jax.Array`, a NumPy array, or a nested Python list |
| **dtype** | a real floating dtype; a narrower one than the run's is promoted, with a warning |
| **rows** | row `j` of the return depends only on row `j` of the argument |
| **failure** | signalled by a **non-finite row**, never by an exception |
| **exceptions** | the callable owns its own; anything that escapes stops the run |

Nothing else — see [what is not required](#what-is-not-required).

## The argument

It is a **concrete** `jax.Array`, never a tracer. The driver loop is ordinary
Python rather than `jax.lax.scan` precisely so this holds, and it is what makes
an external model legal: you can branch on the values, print them, write them
to disk, block on a process that reads them.

It is exactly two-dimensional — members down the leading axis, parameters
across the trailing one — and never has a further axis in front, since a run
binds one ensemble. Its dtype is the run's working dtype, `float64` unless you
have disabled JAX's x64 mode.

Being a `jax.Array`, it must be converted for any library that does not speak
JAX:

```python
import numpy as np

members = np.asarray(ensemble)            # zero-copy, and read-only
```

:::{warning}
`np.asarray` returns a **read-only view**, not a copy. Assigning into it raises
`ValueError: assignment destination is read-only` from wherever you wrote,
not at the conversion. Copy with `np.array` if you need to modify.
:::

With an inflation configured, the argument is the **inflated** ensemble — the
members actually evaluated, and the ones carried on the resulting `Evaluation`.
A wrapper caching evaluations by parameter value should key on what it was
handed.

## The return

Any array-like of shape `(J, N)`. A `jax.Array`, a NumPy array and a nested
Python list of the same numbers are equivalent and give identical runs, so a
wrapper that builds rows in Python or reads them from a file with NumPy need
not convert.

The dtype must be a real floating one; an integer or complex return is a
`ValueError` rather than a silent conversion. A dtype **narrower** than the
run's — in practice `float32` — is promoted to the run's working dtype and
warns once per run.

pyEKI enables `float64` because ensemble anomalies are formed by subtraction
and lose digits to cancellation, and a model that computes or reports in single
precision has already lost them before the array arrives; promoting prevents a
*second* loss in the conditioning arithmetic and nothing more. A `float32`
return costs about `7e-5` relative error in the posterior mean where
predictions have a mean-to-spread ratio of `1e4`, and promotion halves it.
Return `float64` where you can. Where you cannot, the run is still legitimate
and the warning is telling you the price.

## Signalling failure

A member is *failed* when its prediction row contains any non-finite entry.
That is the entire signal, and it puts one real obligation on the wrapper:

:::{important}
**A model that may crash, time out, exit non-zero, or lose a worker must catch
that itself and return a non-finite row** for the affected members. A one-array
interface can express a failed member but not a raised one, so an exception
escaping the callable propagates out of the driver and stops the run — worse
than a `nan` row, which the layer knows how to handle.
:::

Failed members are repaired to the valid members' centre by default;
`on_failure="raise"` turns any failure into an `EKIError`. Either way, fewer
than two valid members raises.

The signal cannot see finite nonsense — zeros, an initial condition, a sentinel
such as `-9999`. The sentinel is the dangerous case: it is finite, so the member
counts as valid, and its enormous misfit reads to an adaptive schedule as
genuine ensemble disagreement, so the run stalls rather than flagging it. Map
those to non-finite rows in your wrapper, where the information exists.

## Row independence

**Row `j` of the return must depend only on row `j` of the argument.** The
layer fits a Gaussian to the pairs and conditions with the resulting
cross-covariance, so a model that normalizes across the ensemble, or shares a
mutable accumulator between rows, returns something that is not a sample of
the joint law at all. Nothing detects it: the shapes are right and the numbers
are finite.

This is the only requirement beyond the shapes and the failure signal.

(what-is-not-required)=
## What is not required

Jittability, traceability, `vmap`-ability, pure JAX, or differentiability. The
model is never traced and never inspected; NumPy, SciPy, a C extension, a
subprocess, an HTTP call to a cluster queue are all legal.

**Determinism is not required either.** A stochastic simulator is a legitimate
forward model. It costs three things, none of which raises: the layer's
exactness result becomes a statement about the model *including* its noise; the
extra prediction spread damps the gain, so the run under-fits; and both adaptive
schedules read that spread as disagreement and shorten their increments, so it
costs more evaluations. Side effects — scratch files, job submissions, a process
pool — are likewise fine.

## A worked example: an external executable

A solver invoked as a subprocess, one member at a time, which sometimes fails.
The first block stands in for the external code — deliberately plain, since
the thing it represents is not a Python library. Substitute your own.

```python
import pathlib, subprocess, sys, tempfile
import numpy as np

WORKDIR = pathlib.Path(tempfile.mkdtemp())
SOLVER = WORKDIR / "solver.py"
SOLVER.write_text(
    "import sys, math\n"
    "u = [float(x) for x in open(sys.argv[1])]\n"
    "if u[1] < 0.0:\n"
    "    sys.exit('solver diverged: negative decay rate')\n"
    "with open(sys.argv[2], 'w') as out:\n"
    "    for t in (0.5, 1.0, 2.0):\n"
    "        out.write(repr(u[0] * math.exp(-u[1] * t)) + '\\n')\n"
)
```

The wrapper:

```python
V_DIM = 3

def forward(ensemble):
    """Evaluate the external solver once per member."""
    members = np.asarray(ensemble)                  # read-only view; only read
    predictions = np.full((members.shape[0], V_DIM), np.nan)

    for j, member in enumerate(members):
        member_in = WORKDIR / f"in_{j}.txt"
        member_out = WORKDIR / f"out_{j}.txt"
        member_out.unlink(missing_ok=True)          # never read a stale result
        np.savetxt(member_in, member)
        try:
            subprocess.run(
                [sys.executable, str(SOLVER), str(member_in), str(member_out)],
                check=True, capture_output=True, timeout=60,
            )
            row = np.loadtxt(member_out)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError, ValueError):
            continue                                # leave the row non-finite
        if row.shape == (V_DIM,):
            predictions[j] = row

    return predictions
```

Three details matter. **The result starts as `nan`**, so a member is
valid only if something wrote over its row — every path that produces no
prediction is already a correctly signalled failure, which is what keeps the
`except` clause short enough to be right. **The `except` names its own
failures** rather than catching everything, so a bug in the wrapper still
reaches you. **Stale outputs are deleted first**, so a solver that exits zero
without writing cannot return a previous step's answer.

Driving it is unremarkable:

```python
import jax, jax.numpy as jnp
from pyeki.eki import AdaptiveESSSchedule, EKIState, run
from pyeki.gauss import Gaussian
from pyeki.linalg import PSDDiagonal

times = jnp.array([0.5, 1.0, 2.0])
truth = jnp.array([2.0, 0.7])
y = truth[0] * jnp.exp(-truth[1] * times) + jnp.array([0.02, -0.01, 0.015])
noise = PSDDiagonal(jnp.full(V_DIM, 0.01))
prior = Gaussian(mean=jnp.array([1.0, 1.0]), cov=PSDDiagonal(jnp.array([1.0, 0.5])))

state = EKIState.from_prior(jax.random.key(0), prior, n_members=32)
result = run(state, forward, y, noise, schedule=AdaptiveESSSchedule())

result.mean          # [2.0798, 0.7406]  against a truth of [2.0, 0.7]
result.min_n_valid   # 29 of 32 — the prior puts mass on negative rates
```

`min_n_valid` of 29 is the wrapper working: those three members drew a negative
decay rate, the solver exited non-zero, and the wrapper turned each into a `nan`
row the driver repaired.

## Common mistakes

| symptom | cause |
| --- | --- |
| a shape error from inside JAX naming neither `J` nor `P` | `forward` written for one member; wrap it with `jax.vmap` |
| `the forward model returned shape (N,)` | returning one prediction vector rather than `(J, N)` |
| `assignment destination is read-only` | writing into `np.asarray(ensemble)`; copy with `np.array` |
| a `UserWarning` about `float32` | the model reports in single precision |
| the run stops with a traceback from your solver | an exception escaped the callable; catch it, return a non-finite row |
| the run stalls at tiny increments, every member valid | a sentinel fill value such as `-9999`; map it to `nan` |
| a wrong answer with nothing raised | rows coupled across members, or the return's row order not matching the argument's |

The last row's other cause is a missing transpose. With members in rows the
contraction is `ensemble @ G.T`; writing `ensemble @ G` raises for a
rectangular `G`, but for a **square** one it silently returns the transposed
model's predictions, right shape and no error.
