# Examples

Runnable notebooks that work a problem end to end — model, prior, inversion,
diagnostics and plots — rather than teaching one idea at a time. The
{doc}`../tutorials/index` teach; these demonstrate.

Every notebook here runs against the shipped toy models, so it can be executed
from a clean checkout with no data and no domain code.

:::{admonition} Being written
:class: note

The notebooks below are planned, not yet written, and the build wiring for
executing them is not yet in place. See *Build wiring* at the bottom of this
page for the decisions that must be made first.
:::

## Planned notebooks

`01_linear_gaussian.ipynb` — **EKI against a closed form.**
: A linear forward model and a Gaussian prior, where the posterior is known
  exactly. Run the sampling form and compare the ensemble's mean and covariance
  against the analytic posterior, showing convergence in $J$ at the Monte Carlo
  rate. This is the notebook that establishes the library computes the right
  answer, so the comparison should be against the closed form rather than
  against a tolerance.

`02_nonlinear.ipynb` — **A nonlinear model, and what the ladder buys.**
: A mildly nonlinear model where a single unit step and an adaptive ladder give
  visibly different answers. Show the ensemble at each iteration, the misfit
  trajectory, and the ESS. Make the point that exactness is claimed for the
  affine-Gaussian case only, and that this is an approximation with no
  consistency guarantee.

`03_sampling_vs_optimizing.ipynb` — **The two forms, on one problem.**
: The same model and data under `AdaptiveESSSchedule` to $\beta = 1$ and under
  `FixedSchedule` with `DiscrepancyStop`. Show the difference in ensemble
  spread, and show what happens to the optimization form with the stopping rule
  removed — overfitting made visible.

`04_structured_covariances.ipynb` — **When error is correlated.**
: A problem with two observation streams, one independent and one correlated,
  built with `block_diag`. Show that the inversion call is unchanged, and show
  what the wrong (diagonal) noise assumption does to the answer.

`05_failing_model.ipynb` — **A model that fails.**
: A forward model with a controllable failure rate. Sweep the rate, show
  `n_valid` in the history and the damping it implies, and show where repair
  stops being adequate.

`06_ensemble_size.ipynb` — **The subspace bound.**
: A high-dimensional problem, $P \gg J$. Show the rank of the reachable
  subspace, sweep $J$, and show what inflation does and does not fix. This is
  the notebook that will be revisited when `pyeki.localize` lands.

## Build wiring

Three decisions, none yet made:

**1. Which extension.** `nbsphinx` is currently listed in the dev dependency
group but is **not** in `docs/conf.py`'s extension list, so no notebook would
render today. The recommendation is `myst-nb` instead: the docs already use
`myst-parser` for Markdown, and `myst-nb` supersedes it — it registers both
`.md` and `.ipynb` and would replace `myst_parser` in the extension list rather
than sit alongside it. Loading both is an error.

**2. Execution policy.** Executing at build time catches examples that have
rotted against the API, and makes them fail the docs CI job, which already runs
with warnings as errors. Committing outputs instead is faster but lets examples
break silently. The recommendation is to execute, with `nb_execution_cache`
enabled so unchanged notebooks are not re-run, and to keep each notebook small
enough that the docs job stays cheap.

**3. Whether notebooks are the source form.** A `.ipynb` under version control
produces noisy diffs. MyST notebooks — `.md` with a code-cell syntax — diff
cleanly and are executed identically by `myst-nb`. Worth choosing before the
first one is written, because converting later means rewriting them.
