# Writing a forward model

The forward model is the one part of a run pyEKI does not supply and cannot
inspect, so what it must satisfy is the package's most important external
interface. This page states that obligation completely, in one place.

pyEKI ships no forward models and defines no base class or protocol for one.
There is nothing to subclass and nothing to register: a forward model is a
callable, and the contract below is a contract on its behaviour rather than on
its type. {ref}`eki-failures` is the normative statement; this page is the same
thing written for the person implementing one.

## The interface

```python
import pyeki  # enables float64; import before creating arrays
import jax.numpy as jnp

times = jnp.array([0.5, 1.0, 2.0])

def forward(ensemble):            # (J, 2) in
    a = ensemble[:, 0:1]
    rate = ensemble[:, 1:2]
    return a * jnp.exp(-rate * times)     # (J, 3) out
```

That is a complete forward model. `run(state, forward, y, noise_cov, ...)`
calls it once per rung.

:::{important}
**`forward` receives the whole ensemble, not one member.** It is called once
per rung with a `(J, P)` array and must return `(J, N)` — `J` predictions, one
per member, in the same order. A function written for a single parameter
vector is the most common mistake on this page, and it does not fail cleanly:
depending on the arithmetic it either raises from deep inside JAX with no
mention of the ensemble, or broadcasts to a plausible wrong shape.
:::

If your model is naturally per-member, wrap it rather than rewriting it:

```python
import jax

def one_member(u):                # (P,) -> (N,)
    return u[0] * jnp.exp(-u[1] * times)

forward = jax.vmap(one_member)    # (J, P) -> (J, N)
```

`jax.vmap` is available because that model happens to be JAX. For one that is
not — a subprocess, a scheduler submission, a legacy binary — the wrapper is
an ordinary Python loop, and that is equally legal. See
[the worked example](#a-worked-example-an-external-executable) below.

## The complete obligation

| | |
| --- | --- |
| **argument** | one positional argument: a concrete `jax.Array` of shape exactly `(J, P)`, dtype `float64` by default |
| **return** | any array-like of shape `(J, N)` — a `jax.Array`, a NumPy array, or a nested Python list |
| **dtype** | a real floating dtype. A narrower one than the run's is promoted, with a warning |
| **rows** | row `j` of the return must depend only on row `j` of the argument |
| **failure** | a member that failed is signalled by a **non-finite row**, not by an exception |
| **exceptions** | the callable owns its own: anything that escapes stops the run |

Nothing else. In particular the model need not be jittable, traceable,
`vmap`-able, pure JAX, differentiable, or deterministic — see
[What is not required](#what-is-not-required).

## What the callable receives

The argument is a **concrete** `jax.Array` — never a tracer. The driver loop is
ordinary Python rather than `jax.lax.scan` precisely so this is true, and it is
what makes an external model legal: you can branch on the values, print them,
write them to disk, and block on a process that reads them.

It is exactly two-dimensional. The leading axis indexes ensemble members and
the trailing axis parameters; there is never a further axis in front, because a
run binds one ensemble and a batched `EKIState` is rejected at the call.

Being a `jax.Array` rather than a NumPy one, it must be converted before it
reaches a library that does not speak JAX:

```python
import numpy as np

members = np.asarray(ensemble)    # zero-copy, and read-only
```

:::{warning}
`np.asarray` on the argument returns a **read-only view**, not a copy.
Assigning into it raises `ValueError: assignment destination is read-only`,
from wherever you happened to write, rather than at the conversion. If you need
to scale, clip, or otherwise modify the parameters in place, copy first with
`np.array(ensemble)`.
:::

With an inflation configured, the argument is the **inflated** ensemble, not
`state.ensemble` — the members that were actually evaluated, and the ones
carried on the resulting `Evaluation`. A wrapper that caches evaluations by
parameter value should key on what it was handed.

## What the callable may return

Any array-like of shape `(J, N)`. A `jax.Array`, a NumPy array, and a nested
Python list of the same numbers are equivalent and produce identical runs, so a
wrapper that assembles rows in Python or reads them from a file with NumPy need
not convert anything.

The dtype must be a real floating one. An integer or complex return is a
`ValueError` rather than a silent conversion, because a model returning
integers is far more likely to have a bug than a precision preference.

A dtype **narrower** than the run's — in practice `float32` — is promoted to
the run's working dtype, and warns once per run:

> the forward model returned float32 predictions, promoted to the run's working
> dtype float64 …

The promotion is not a fix, which is why you are told about it. pyEKI enables
`float64` because ensemble anomalies are formed by subtraction and lose digits
to cancellation; a model that computes or reports in single precision has
already lost them before the array arrives, and no dtype chosen afterwards
recovers them. Promoting avoids a *second* dose in the conditioning arithmetic
and keeps the run's precision independent of the model, and that is all it
does. Return `float64` where you can; where you cannot — a single-precision
solver, an output file written as `REAL*4` — the run is still legitimate, and
the warning is telling you about a real cost rather than a defect.

## Signalling failure

A member is *failed* when its prediction row contains any non-finite entry.
That is the entire failure signal, and it puts one real obligation on the
wrapper:

:::{important}
**A model that may crash, time out, return a non-zero exit code, or lose a
worker must catch that itself and return a non-finite row** for the affected
members. A one-array interface can express a failed member but not a raised
one, so an exception that escapes the callable propagates out of the driver and
stops the run — a worse outcome than a `nan` row, which the layer knows how to
handle.
:::

By default failed members are repaired to the valid members' centre and the run
continues; `on_failure="raise"` turns any failure into an `EKIError` instead.
Either way fewer than two valid members raises.

What the signal cannot see is finite nonsense — a solver returning zeros, its
initial condition, or a sentinel such as `-9999`. The sentinel is the dangerous
case: it is finite, so the member counts as valid, and its enormous misfit
reads to an adaptive schedule as genuine ensemble disagreement, so the run
stalls instead of flagging it. Map those to non-finite rows in your wrapper,
where the information exists.

## What is not required

- **Jittability, traceability, `vmap`-ability.** The model is never traced and
  never inspected.
- **Pure JAX.** NumPy, SciPy, a C extension, a subprocess, an HTTP call to a
  cluster queue — all legal.
- **Differentiability.** EKI is derivative-free; this is the point of it.
- **Determinism.** A stochastic simulator is a legitimate forward model. What
  it costs is that the layer's exactness result is about the model *including*
  its noise, and that the extra prediction spread damps the gain and shortens
  the adaptive schedules' increments — so a noisy model converges more slowly
  and costs more evaluations. Nothing raises.
- **Purity of effect.** Writing scratch files, submitting jobs and holding a
  process pool are the ordinary way to reach an external code.

The one thing that *is* required beyond the shapes is **row independence**: row
`j` of the return depends only on row `j` of the argument. A model that
normalizes across the ensemble, or shares a mutable accumulator between rows,
breaks the pairing the ensemble update is built on. Nothing detects it.

## A worked example: an external executable

The case the interface is shaped for: a solver invoked as a subprocess, one
member at a time, which sometimes fails. The wrapper converts its input,
catches its own failures, and returns non-finite rows for the members that
failed.

The first block stands in for the external code. Substitute your own.

```python
import pathlib, subprocess, sys, tempfile
import numpy as np

WORKDIR = pathlib.Path(tempfile.mkdtemp())
SOLVER = WORKDIR / "solver.py"
SOLVER.write_text(
    "import sys, numpy as np\n"
    "u = np.loadtxt(sys.argv[1])\n"
    "if u[1] < 0.0:\n"
    "    sys.exit('solver diverged: negative decay rate')\n"
    "np.savetxt(sys.argv[2], u[0] * np.exp(-u[1] * np.array([0.5, 1.0, 2.0])))\n"
)
```

The wrapper:

```python
N_OBS = 3

def forward(ensemble):
    """Evaluate the external solver once per member."""
    members = np.asarray(ensemble)                  # read-only view; only read
    predictions = np.full((members.shape[0], N_OBS), np.nan)

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
        if row.shape == (N_OBS,):
            predictions[j] = row

    return predictions
```

Four things in it are the point:

- **`np.asarray(ensemble)` once, at the top.** The loop then works in NumPy.
- **The result array starts as `nan`.** A member is valid only if something
  wrote over its row, so every path out of the loop that does not produce a
  prediction — a crash, a timeout, an unreadable file, a wrong-shaped one — is
  already a correctly signalled failure. Failing *closed* is what makes the
  `except` clause short enough to be right.
- **The `except` names its own failures** rather than catching everything: a
  bug in the wrapper should still reach you.
- **Stale outputs are deleted before the call**, so a solver that exits zero
  without writing cannot yield a previous rung's answer.

Driving it is unremarkable:

```python
import jax, jax.numpy as jnp
from pyeki.eki import AdaptiveESSSchedule, EKIState, run
from pyeki.gauss import Gaussian
from pyeki.linalg import PSDDiagonal

times = jnp.array([0.5, 1.0, 2.0])
truth = jnp.array([2.0, 0.7])
y = truth[0] * jnp.exp(-truth[1] * times) + jnp.array([0.02, -0.01, 0.015])
noise = PSDDiagonal(jnp.full(N_OBS, 0.01))
prior = Gaussian(mean=jnp.array([1.0, 1.0]), cov=PSDDiagonal(jnp.array([1.0, 0.5])))

state = EKIState.from_prior(jax.random.key(0), prior, n_members=32)
result = run(state, forward, y, noise, schedule=AdaptiveESSSchedule())

result.mean          # [2.0798, 0.7406]  against a truth of [2.0, 0.7]
result.min_n_valid   # 29 of 32 — the prior puts mass on negative rates
```

`min_n_valid` being 29 is the wrapper working: the prior puts some members at a
negative decay rate, the solver exits non-zero on those, and the wrapper turns
each into a `nan` row that the driver repairs. The run also warns once, and
logs at `WARNING` on each affected step.

## Common mistakes

| symptom | cause |
| --- | --- |
| a shape error from inside JAX naming neither `J` nor `P` | `forward` written for one member; wrap it with `jax.vmap` |
| `the forward model returned shape (N,)` | returning one prediction vector rather than `(J, N)` |
| `assignment destination is read-only` | writing into `np.asarray(ensemble)`; copy with `np.array` |
| a `UserWarning` about `float32` | the model reports in single precision; see [above](#what-the-callable-may-return) |
| the run stops with a traceback from your solver | an exception escaped the callable; catch it and return a non-finite row |
| the run stalls at tiny increments, every member valid | a sentinel fill value such as `-9999`; map it to `nan` in the wrapper |
| a wrong answer with nothing raised | rows coupled across members, or the return's row order not matching the argument's |

The last row deserves a line of its own, because an affine model is where it
bites. With members in rows, the contraction is `ensemble @ G.T` for a `G` of
shape `(N, P)`. Writing `ensemble @ G` instead raises when `G` is rectangular —
but when it is **square** it returns the transposed model's predictions, with
the right shape and no error at all. `G @ ensemble` raises either way, so it is
the harmless mistake; the silent one is the missing `.T`.
