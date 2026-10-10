# Handoff

Written 2026-08-24, updated 2026-08-25 after the operator layer was reworked
against the normative contract, 2026-08-27 after `pyeki.eki` shipped,
2026-08-28 after the forward-model contract was specified and the layer
vocabulary was fixed, 2026-09-02 after the joint was split into a
Gaussian and a sample container, 2026-10-03 when the EnsKit redesign was
adopted, the same day after PR 1's renames, again after PR 2's linalg
additions, and 2026-10-04 after PR 3's distribution contract, PR 10's
Kronecker operators, PR 4's `enskit.distribution`, the fix for #60, PR 5's
`enskit.maps`, PR 6's `enskit.kalman`, PR 9's localization and PR 7's
`enskit.algorithms.eki`, and 2026-10-05 after PR 8's `enskit.algorithms.enkf`,
PR 11's documentation and PR 12's release, and 2026-10-09 after
statistical linearization (#85) and the divisor rule (#91).
Read
`CLAUDE.md` first for conventions, including the layer rules, which the
redesign replaced; then the two sections below; then the rest of this file,
which describes the code as it stands before the redesign lands. That
description is historical: where it names `pyeki.<module>`, the module is now
`enskit.<module>`.

## 2026-10-09: one divisor rule for particle covariances (#91)

**What landed.** A keyword-only `unbiased: bool = True` on `Ensemble.cov`,
`Ensemble.project` and `maps.statistical_linearization`. `False` divides by
$J$, or $1$ when weighted: the empirical distribution's own moments, for
particles that are a deterministic rule's points rather than samples. The
default, and every number, is unchanged. `EnsembleGaussian` gains the static
field `unbiased` (a constructor keyword) and the property `divisor`, and
`MatheronMap` gains the static attribute `divisor`. The rule is stated once,
in the new section `(dist-divisor)` of the distribution contract and the
user guide's `(guide-divisor)`; the rest refers to it. The contract's
*Deliberately excluded* entry "a configurable anomaly divisor" is gone.
Tests: `tests/test_divisor.py`, which is distribution obligation 24 and
kalman obligation 27.

**Decided with the maintainer** (comment on #91): the default stays
unbiased; the flag is named `unbiased`; it is offered only at those three
functions, and `kalman.gaussian_approximation`, `exact_moment_ensemble` and
the algorithms' diagnostics stay at $J - 1$. **An unweighted projection is
always an `EnsembleGaussian`, and carries its divisor**: a plain-`Gaussian`
result for the empirical divisor was rejected as unintuitive. Weighted
projections stay plain `Gaussian`s; whether they should be aligned too is
#93.

**Every reader of an `EnsembleGaussian`'s factor rows uses `divisor`.**
That covers `realize_particles`, `square_root_map`,
`MatheronMap.particle_coefficients`, and localization's three
$\sqrt{J-1}$s. The rules are public and accept any approximation, so they
honor an empirical approximation passed through `approximation=`. Each
seam's regression test was checked by reverting that seam to $J - 1$: every
reversion fails at least one test.

**Things the implementation found.**

- **`maps` rebuilds an aligned Gaussian through the public constructor**
  (`maps/_common.py::with_block`), which would have dropped the divisor
  silently. It passes `unbiased` now. A new place that builds an
  `EnsembleGaussian` must pass it too. `c.build` bypasses the constructor,
  so the field deliberately has no class-level default: a build that forgets
  it raises `AttributeError` at the first read of `divisor`, rather than
  silently reading $J - 1$.
- **Third-party rules are not checked against the divisor (#94).** The
  protocol's alignment clause now says $\sqrt{\delta}$, but
  `check_update_rule` builds only through `gaussian_approximation`
  ($\delta = J - 1$). A user rule that hard-codes $\sqrt{J-1}$ passes
  conformance. The adversarial review found it; #94 has the options.
- **The adversarial review's trivial findings were fixed here.** The
  Matheron "unbiased over the key" claims now say they need $\delta = J - 1$
  (with $\delta = J$, the noise term comes out $(J-1)/J$ short). The
  weighted unbiased claim is limited to fixed weights. "Sigma points" became
  rules with nonnegative weights, since the unscented transform's negative
  $W_0$ does not fit `log_weights`. The `particle_coefficients` test now
  compares against the actual particles rather than the realized ones. jit
  and vmap tests were added. The bool messages now say "Python bool", since
  NumPy 2's `np.bool_` is also named `bool`.
- **`statistical_linearization` needed no special case.** A and b do not
  depend on the divisor, and an empirical projection is still aligned, so
  minimum norm works for both divisors.
- **The square-root update is invariant to a consistent rescaling.** The
  divisor $J$ with noise $cR$, $c = (J-1)/J$, gives exactly the particles
  that $J - 1$ with $R$ does. This is the test that pins the square-root
  paths to an independent reference.

**For #88.** Its step 6 (`residual_cov * (J - 1) / J`) becomes
`statistical_linearization(..., unbiased=False)`, and D3 is settled.
`sigma_points` can say that `project(unbiased=False)` reproduces the
Gaussian's covariance exactly, and the "$\frac{J}{J-1}$ caveat" in its user
guide plan becomes a pointer to `(guide-divisor)`.

## 2026-10-09: statistical linearization (#85)

**What landed.** `Gaussian.regression(target, *, given, min_norm=False)`
in `enskit.distribution` (new private module `_regression.py`, new public
`Regression`), and `maps.statistical_linearization(dist, *, inputs, output,
min_norm=False)` with its `NamedTuple` result `Linearization`, in
`enskit.maps` (`_linearization.py`). Contracts: a new `regression` section
in the distribution contract (and rule 1 of *The objects* now admits a
`Regression`), a new `statistical_linearization` section in the maps
contract. User guide: a section in each of `distributions.md` and `maps.md`.
Example 16. Tests: `tests/test_linearization.py`.

**Decided with the maintainer** (issue #85, Q1 to Q4): the function takes
input–output pairs already in the distribution, never a callable (push
first); minimum norm is opt-in, and the non-unique case raises naming both
remedies; both layers ship; $\Omega$ uses the divisor $J - 1$.

**Things the implementation found.**

- **$\Omega$ as an independent term breaks `cov`.** `AdditiveNoise(Omega)`
  pushes fine, but `Omega` is a `PSDLowRank` with no `whiten`, so the
  result's `cov(y)` cannot build its `LowRankUpdate` and raises. The docs say
  to `absorb` it into the factor, or use `LowRankUpdate(R, Omega.factor())`.
- **The ridge coefficients cannot use the stable form.** $A^{-1}SW$ is
  stable read off the SVD, but forming it as an operator needs $W^\top$,
  which operators do not provide. The code computes
  $(I+SS^\top)^{-1}F_c^\top D_c^{-1}$ with `solve`, which loses relative
  accuracy like $\varepsilon(1+\sigma_{\max}^2)$; documented in the
  docstring and contract. So a noisy given term needs `whiten` *and*
  `solve`.
- **Weighted minimum norm is unsupported.** A weighted projection's null
  vector is $\sqrt w$, not $\mathbf 1$; `statistical_linearization`
  refuses a weighted ensemble with more input coordinates than $J - 1$.
  `Gaussian.regression` on such a plain Gaussian relies on the debug rank
  check.
- **Rank is assumed, not measured (#86).** The case is chosen from the
  type and the sizes; outside debug mode, given rows of lower rank
  (duplicates after `resample`, a constant coordinate, an ensemble
  conditioned exactly on other blocks, zero weights) give wrong or `nan`
  coefficients without raising. Documented; the options are in #86.
- **The adversarial review's trivial findings were fixed here**: the
  $\mathbf 1^\perp$ basis was a host-side NumPy QR of a $J \times J$ matrix
  (6 s at $J = 2000$), now a Householder reflector applied in $O(NJ)$; the
  "update equals the linear-Gaussian update" claim is now limited to the
  deterministic square-root rule; a given block without a factor row no
  longer needs `whiten` and `solve`; a vacuous test assertion; coverage of
  mixed and structured cases; vocabulary in the distribution docs.
- **This PR also bumps the version** to `0.2.0.dev0` and opens the 0.2.0
  changelog entry, as PR #84 does; whichever merges second resolves the
  `CHANGELOG.md` conflict by keeping both entries' lines.

**Natural follow-ups** (not filed; see the PR body for the reasoning): an
SLR `StructuredMap` that pushes a Gaussian through a nonlinear map with a
deterministic point set (sigma points), and with it a posterior
linearization filter; per-step linearizations exposed from an EKI run's
history; minimum norm for weighted ensembles; and a held-out estimate of
$\Omega$ for the regime $d_x \ge J - 1$, where in-sample residuals vanish.

## 2026-10-05: PR 12, release 0.1.0

**Decided with the maintainer:** 0.1.0 is a **tagged GitHub release, not a
PyPI upload**; PyPI is held off, and is #80. The five
issues still open in the "EnsKit 0.1" milestone (#47, #68, #69, #72, #73)
moved to a new "EnsKit 0.2" milestone; #51, #54, #68, #69 and #76 are the
changelog's known limitations.

**The release procedure** is in `CLAUDE.md` under *Releases*. In short: the
version is written once, `enskit.__version__` (hatch reads it, `docs/conf.py`
imports it); `CHANGELOG.md` is new, included in the docs as
`docs/changelog.md`; `tests/test_release.py` checks the changelog heading
against the version, every `git+...@vX.Y.Z` install command against the
newest released tag, the installation page's stated floors against
`pyproject.toml`, and runs the README's three blocks, which nothing ran
before. Pushing a tag `v*` runs `.github/workflows/release.yml`: tag against
version, build, the full default suite against the installed wheel with the
source directory deleted, and `gh release create` with the changelog entry
and both distributions.

**After this merges**, the maintainer tags the merge commit `v0.1.0` and
pushes the tag. Then the **first PR after the release** sets `__version__` to
`0.2.0.dev0` and opens `## 0.2.0 (unreleased)` in the changelog;
`test_release.py` accepts exactly that state.

**Dependency floors were measured, not guessed.** `jax>=0.4`/`numpy>=1.24`
had never been tested. Now `jax>=0.10.1`, `numpy>=2.0` (JAX 0.10.1 itself
requires NumPy 2.0), checked by CI's new `minimum-versions` job, which
installs with `uv pip install --resolution lowest-direct --upgrade -e .` on
Python 3.11. Without `--upgrade` uv keeps the locked versions, which satisfy
the bounds, and the job silently tests the lock.
Below 0.10.1 every number still agrees; what fails:

| JAX | fails |
| --- | --- |
| 0.4.38 | the two pinned-draw snapshots, the systematic-resampling regression, and two `check_simulator` cases |
| 0.5.3, 0.6.0 | `check_simulator` reports the symmetric coupling through its permutation check, not the subset one, so the test's message match fails (the defect is still rejected) |
| up to 0.8.0 | both count tests: `jax.monitoring.unregister_event_duration_listener` does not exist yet |
| 0.9.0 to 0.10.0 | `test_enkf.py::test_12`: the first new length compiles 9 operations and the next 7, against an asserted equal increment of at most 8 |

Lowering the floor means relaxing those count tests, which is a decision
about what the count tests promise, not a packaging fix.

**The sdist** carries `enskit/`, `tests/`, `docs/` (without the redesign's
working material), the README, changelog and license: 0.7 MB, down from
1.2 MB, and it no longer carries `memory/`, `HANDOFF.md`, `CLAUDE.md` or
`uv.lock`. Unpacked on its own, it builds the docs with `-W` and passes the
default suite. `notebooks/README.md`, which listed three planned notebooks
that the examples gallery replaced, is deleted.

**For later PRs.**

- **PyPI** is #80. A README bound for PyPI needs absolute links (its
  `CHANGELOG.md` link is relative), and the install commands change.
- Every PR that changes what a user sees adds a changelog line (`CLAUDE.md`,
  *Finishing a PR*).
- **#81**, from the adversarial review: the published docs deploy from `main`
  and so will describe unreleased interfaces while the install commands name
  `v0.1.0`; there is no branch for a patch release; and `main` reports
  `0.1.0` until the version-bump PR merges. Settle it before or with that PR.
- **The adversarial review's trivial findings were fixed here**: the
  `minimum-versions` job tested the lock (now `--upgrade`, and it asserts the
  installed versions equal the floors); the installation page misstated what
  `uv add` writes; the tag regex missed `.git@` and `rev =` spellings; the
  status lines named the version; `release.yml` did not check that the tag is
  on `main`; the float32 limitation said "tempered" though every EKI run
  scales its noise; #7, #63 and #64 joined the known limitations; and the
  README test now fails on a warning instead of hiding it.

## 2026-10-05: PR 11, the documentation

**Notebook wiring, decided with the maintainer:** `myst-nb` replaces
`myst_parser` in `docs/conf.py` (it parses every `.md` page with the same
MyST parser, and the build was unchanged by the swap); `nbsphinx` is gone
from the dev group, since it needs pandoc. `nb_execution_mode = "off"`: the
docs build never executes a notebook.

**The examples gallery** is `docs/examples/`. The fifteen design examples
were copied from the frozen `docs/redesign/notebooks/src/` into
`docs/examples/src/` (percent-format `.py`, the reviewed source) and ported.
`docs/examples/build.py` executes a source in a fresh interpreter and writes
`docs/examples/<name>.ipynb` with stored outputs; never edit a notebook by
hand. `docs/examples/references.py` is the bibliography, copied from the
redesign folder plus `kitagawa1996`. `tests/test_examples.py` checks fast that
every stored notebook's cells are its source's (so a changed source fails
until rebuilt), and, marked `slow`, executes every source (about 2 minutes);
`pytest` deselects `slow` by default, and CI's new `examples` job runs
`pytest -m slow`. Every number an example's text states is asserted in its
final `# checks` cell; the Lorenz-96 ones (2, 11–14) state approximate values
and assert bands measured over seeds 0–5, since their stored outputs differ by
platform.

**Porting the examples changed more than names.** Their Setup sections
described the prototype's toys (ten times on [0.1, 2], noise 0.05,
u† = (1.5, 1.2); a full prior covariance for `linear_gaussian`), so every
setup was rewritten against the shipped toy and every number recomputed. A
`Gaussian` is pushed through `maps.Linear(problem.G)`, never through
`problem.forward`. Example 7 checks against `LinearStateSpace.exact_filter()`
and a dense recursion; example 10's reference became grid quadrature (the
prototype's prior importance sampler had an ESS of 277 and more error than
the estimate it judged); example 13 inflates by 1.05; example 14 keeps 1.02
(over seeds 0–5 its error is 0.27–0.31 at 1.02, 0.33–0.35 at 1.05).

**The user guide is organized by level** (`docs/user-guide/index.md`):
running an algorithm, one update, forecast and update separately,
probabilistic operations, operators. One new page,
`forecast-and-update.md`: the hand-written cycle, its Gaussian version (which
is `exact_filter`), and the drivers' halves equal to it bit for bit. The
quickstart is titled "Operator quickstart" and sits at the operators level.
`tests/test_docs.py` runs every block of the fragment pages (`updates`,
`distributions`, `operators`, `quickstart`, `writing-a-forward-model`,
`writing-an-operator`) after a setup that builds what each fragment assumes,
and checks that every user-guide page is either tested there or by its
layer's test module.

**Tutorials.** 1 to 3 revised: vocabulary (particles, steps), the new rule
names, and the four claims that stopped holding in PR 7 rewritten from the
measured numbers and pinned (`test_8_*` in `tests/test_tutorials.py`; the
section-5 xfails are gone, and the one-step band test asserts the claim, a
rate overspread above 15 at both 64 and 2048 particles). Stubs 4 to 9 are on
the new API, still stubs; writing them is #77, which carries #24 (tutorial
7's stub now drops the step count and plans the variance histogram). #26 (the
ablation study) is deferred to its own PR as an example notebook with stored
outputs, which this wiring now supports.

**Dark figures (#27).** `docs/figures.py` draws every figure twice, from
`THEMES["light"]` and `THEMES["dark"]` (`_use_theme` rebinds the module's
color names, so a figure function must read colors at call time, never in a
default argument). A transform registered by the extension,
`add_dark_variants`, gives each generated image Furo's `only-light` class and
inserts its `-dark.png` sibling, so a page names only the light figure.

**Also here:** the repository is `TARPS-group/EnsKit`, so the URLs in
`pyproject.toml`, `docs/conf.py`, the README and `docs/installation.md` say
so, and the docs URL is `tarps-group.github.io/EnsKit/`. The package
docstring, README and landing page describe both algorithms.

**Found here:** #76, a composite operator's `shape` is exponential in its
nesting depth (each uncompressed `condition` doubles the `Product.shape`
calls; correct results, exploding time). Shipped paths compress, so they
avoid it. #78, from the adversarial review: nothing compares a stored
notebook's outputs with a fresh run, so a rendered number inside an
example's assertion band can go stale silently.

**The adversarial review** found no wrong numbers. Its trivial findings were
fixed here: two misdescriptions in the gallery index and example 9, an
assertion that could not fail (the builder now records execution counts), a
pin on a rounding boundary in example 15, the dark-variant transform
restricted to HTML and to `.png` figures, the one-step band test made to
compare the two ensemble sizes rather than fit a band, the docs harness
counting the blocks it skips, and example 14 showing `check_update_rule`.

**For later PRs.**

- **PR 12 (release).** The docs deploy from `main` to GitHub Pages under the
  renamed repository; check the Pages URL once after the first deploy.
- **Changing an example** means editing its source, running
  `uv run python docs/examples/build.py exNN`, and committing both. A change
  to the package that moves an example's numbers fails the slow test, not the
  default one; run `uv run pytest -m slow` before a release.

## 2026-10-05: PR 8, `enskit.algorithms.enkf` and the state-space toys

The ensemble Kalman filter is `enskit.algorithms.enkf` (private modules
`_filter` and `_values`): `forecast`, `analysis` (returning the analysis
ensemble and the one-step log evidence), `filter`, `FilterResult`,
`EnKFError` and `PREDICTION`. Its normative page is `docs/enkf-contract.md`,
written for this PR from the stub; its *Departures from the design* lists
everything that moved. `enskit.toy` gained `lorenz96`, `lorenz96_step`,
`Lorenz96`, `linear_state_space` and `LinearStateSpace` (with
`exact_filter()`, the reference for the exactness obligation). The user-guide
page is `docs/user-guide/filtering.md`, whose blocks
`tests/test_enkf.py::test_18_*` runs and whose numbers it pins;
`toy-models.md`, the landing page, the README and the API reference were
updated. `tests/test_enkf.py` follows the contract's 18 obligations by number.

**Decided here, with the maintainer** (each is in the contract):

- **The key is keyword-only and optional** (`key=`) in all three functions,
  per `CLAUDE.md`'s sometimes-random rule; the stub had it first and
  positional. `filter` splits it four ways, `(next, forecast, inflate,
  analysis)`, at every time; `forecast` splits it into `(transition, noise)`.
- **#71 is settled by keeping it**: the relaxation's `prior` is the
  background, the inflated forecast, in both drivers, and both contracts and
  the user guide say the two compound. #71 can be closed when this merges.
- **The toys are two classes** with the module's existing argument names
  (`state_dim`, `data_dim`, `n_times`, `noise_std`), not one
  `StateSpaceProblem` with `dim`/`obs_dim`/`n_steps`/`noise_sd`.
- `EnKFError` carries `time`, `key` and `result` (a `FilterResult` of the
  times before), so a caught error resumes bit for bit. There is no
  `on_failure="repair"`.
- `analysis` requires `inputs`; `filter` defaults it to `(state,)`.
- **`filter` takes `start_time`** (from the adversarial review): without it a
  resumed filter restarted `time` at 0, so a policy reading `time` diverged
  from the uninterrupted run although the draws agreed. The resume recipe is
  `filter(exc.result.ensemble, ys[exc.time:], key=exc.key,
  start_time=exc.time, ...)`.

**The adversarial review** found no wrong numbers for valid inputs. Its
trivial findings were fixed here (contract departures 9 and 10): `filter`
now checks `transition`/`observe` and transition noise without a key before
the first transition call; a non-finite *prediction* is named as the
observation model's, not blamed on the update rule; the log evidence is cast
to the ensemble's dtype (float64 observations with float32 particles gave a
float64 evidence); and the tests gained the cases it showed missing.

**Things not to rediscover.**

- **`fold_in(k, i)` is `split(k, n)[i]`** for every `n > i` under JAX's
  partitionable threefry (the default in 0.10). So #73's suggested fix,
  deriving toy keys with `fold_in`, does not work: a first version of
  `lorenz96` keyed its observation noise exactly as
  `initial.sample(key(0), ...)` keyed its draw. The new toys split from
  `fold_in(key(seed), 0x746F79)` instead (`toy._TOY_STREAM`); the old three
  are unchanged, so #73 stays open for them.
- **A few operations compile once per length of `observations`** (the
  finiteness check, row indexing, stacking the means), so "a 30-time filter
  compiles nothing a 3-time one has not" is false by construction; the
  compile test compares the 3→30 and 30→60 increments instead, and slices
  outside the counted region (a no-op slice compiles nothing, and threw the
  first version off by one).
- **Example 13's inflation of 1.02 is seed-sensitive** on the new toy truth:
  it lost the truth at two of seeds 0 to 3 (errors 2.8, 2.3); 1.05 tracked
  all four (0.31 to 0.39). The test and the user guide use 1.05.
- **Lorenz-96 numbers are not portable.** The 1000-step spin-up turns a
  $10^{-15}$ change in the start into an $O(10)$ change in $x_0$, so CI's
  Linux filters a different trajectory from macOS at the same seed. The
  first CI run failed on two pinned digits (the guide's error, 0.336 against
  0.308; localized stochastic 0.54 against a bound of 0.5). Lorenz-96 tests
  now check bands measured over seeds 0 to 5, and the guide says its numbers
  are approximate. PR 11's stored Lorenz-96 notebook outputs will differ by
  platform for the same reason.
- **Float32 filters work** with both rules and every policy: the filter
  never scales the noise covariance, so #68 does not arise here.
- `kalman.update`'s shipped rules already refuse an approximation that drops
  a block, with a less direct message; `analysis` checks the approximation's
  blocks first, inside the capturing wrapper it passes as
  `approximation=`, which is also how the log evidence is read from the very
  object the update conditioned.

**Measured.** Lorenz-96 at the defaults ($d = 40$, $T = 300$, half observed),
$J = 40$, inflation 1.05: time-averaged error 0.31 with `Matheron` (3 s),
0.32 with `SymmetricSquareRoot` (1.5 s). At $J = 10$: global 3.61, localized
ETKF 0.36, localized stochastic 0.40, hybrid 0.90 against plain 4.86. The
linear exactness holds to $5\times10^{-16}$ in the means and
$3\times10^{-15}$ in the log evidence.

**For later PRs.**

- **PR 11 (documentation).** The design's Examples 2, 7, 11, 12 and 13 run on
  the new API with the renames above (`toy.lorenz96(state_dim=..., n_times=...,
  noise_std=...)`, `toy.lorenz96_step`, `key=` keyword-only, `inputs="x"` on
  `analysis`); their quoted numbers change with the new toy truth (see
  *Measured*), and Example 13 should inflate by 1.05. Example 7's loop is
  `LinearStateSpace.exact_filter()`.
- `docs/user-guide/filtering.md` sits after `running-an-inversion` in the
  toctree; the level-of-abstraction reorganization is PR 11's.

## 2026-10-04: PR 7, `enskit.algorithms.eki`; `gauss` and `eki` deleted

The EKI driver is ported onto the new layers as `enskit.algorithms.eki`
(private modules `_values`, `_schedules`, `_driver`, `_helpers`), and
`enskit.algorithms` holds the inflation and relaxation policies every driver
shares (`_policies`, `_common`). `docs/eki-contract.md` was revised, not
rewritten from the stubs: its *Departures from the design* and *Changes from
the previous contract* sections list everything that moved, and its *Ported
regression tests* table maps the old `tests/test_eki.py` onto the new one.
`enskit.gauss`, `enskit.eki`, `tests/test_gauss.py`,
`docs/gaussian-contract.md` and `docs/user-guide/conditioning.md` are gone;
`enskit.toy` was ported (priors are `distribution.Gaussian`s over block
`"u"`; `u_dim`/`v_dim` became `parameter_dim`/`data_dim`); the import-linter
contract names `enskit.algorithms` without parentheses and no longer lists
the old modules. `enskit.testing` gained `check_schedule`, `check_inflation`,
`check_relaxation` and `check_stopping_rule`.

**Decided here** (each is in the contract):

- **The key splits four ways**, `(next, inflate, evaluate, update)`: a
  simulator declaring `needs_key` gets `key_evaluate` through `pushforward`.
- **`update_rule` is required**; `on_failure` defaults to `"raise"`, so a
  repair is opted into. The update rule no longer sees `step`, `beta` or the
  increment (it is a `kalman.UpdateRule`); a ladder-dependent behavior is a
  relaxation, which receives `step` and `beta`.
- **Relaxation** (`RelaxToPriorSpread`, `RelaxToPriorPerturbations`) is a new
  axis, applied after each update with `prior` the evaluated particles; the
  old contract excluded it.
- **`assimilate` returns the state only** (the stub's signature), so
  `HistoryRecord.from_evaluation(evaluation, increment=None)` is public: it is
  how the driver builds every record, and how a hand loop gets the same ones.
  `iterate` still yields `(state, record, evaluation)`.
- **The promotion warning is issued at every promoting evaluation** (the
  maps layer's rule), not once per run as the old contract promised. A first
  version recorded each evaluation's warnings and re-issued them; the
  adversarial review showed that this replaced a failing run's `EKIError` by
  the warning under an `error` filter, broke module filters, and defeated
  Python's once-per-location (every `catch_warnings` invalidates the
  registries). Python's default filter shows the warning once.
- **The adaptive schedules absorb a round-off remainder** below
  $10^{-9}\beta_{\text{target}}$ into the step before it, and the entry
  budget check counts with that tolerance (`_steps_needed`). Without it,
  $10^5$ floor-sized steps, or a budget a hair above a multiple of the floor,
  passed the entry check and then exhausted `max_steps`.
- **The update runs eagerly**, about 3 ms per step at $J = 64$; the debug
  checks of the layers below therefore run. Test 14 counts compilations with
  `jax.monitoring` across whole runs.

**For later PRs.**

- **PR 8 (EnKF).** Reuse `enskit.algorithms.Inflation`/`Relaxation` with
  context `time=` (the protocols say a driver passes what it has), and
  `algorithms._common.pytree_class` and `check_policy_output`. The contract's
  *Inflation and relaxation* section is the normative home of the shared
  policies; the EnKF contract should refer to it rather than restate it.
- **PR 9 (localization).** A localized rule plugs into `update_rule=` with no
  driver change. Obligation 25's matrix should gain it.
- **PR 11 (documentation).** The tutorials' code is on the new API; their
  prose is not. Every printed value and every number in prose that states an
  output was recomputed (the prior draw changed, since `from_prior` and
  `Gaussian.sample` each split their key, and tutorial 1 runs
  `kalman.Matheron()` where it ran `PathwiseUpdate`). Five claims no longer
  hold, each pinned as a strict expected failure in section 5 of
  `tests/test_tutorials.py` naming its sentence: tutorial 1's "tracks the
  distributions fairly well" and "the second distribution presents a
  challenge" (the rate spread at the fifth level is now 0.73 of the exact
  one, the second level's 1.29), the one-step error band at 64 particles
  (now `[1.55, 19.51]`; at 2048 it is still in band), and tutorial 3's
  "`tau=2` stops one step earlier" and "within 13% of the target" (both
  rules now stop at beta = 2). Prose still naming old classes:
  tutorial 1 lines ~297 and ~331 (`PathwiseUpdate`, `TransformUpdate`),
  tutorial 2 line 10 ("the library's default update rule"), tutorial 3 line
  ~225, and the stubs 4, 5, 7, 8 (`on_failure='repair'` as the default) and 9
  (`enskit.eki.testing`, `check_update`, `synthetic_evaluation`).
  Tutorial 7's numbers changed most: the high-dimensional run's spread is
  0.302 against an exact 0.990 (3.3 times too small, over about 30 steps),
  not 0.014 and seventy times; the old figure was a key coincidence, the old
  particles reproducing the rows of `G` (see the toy note below).
  `running-an-inversion.md` and `writing-a-forward-model.md` were rewritten
  for the new API, and tests run every block of the first and of the landing
  page; `writing-a-forward-model.md` could fold into `maps.md`.
  `docs/joint-factor.md` is kept, with a status note mapping the old class
  names to the new ones.
- **Toy keys.** `toy.linear_gaussian(seed=s)` draws `G` from
  `split(key(s), 3)[0]`, which equals `split(key(s), 2)[0]`. The old
  `from_prior(key(s))` drew its particles from that same key, so at seed 0 the
  particles were multiples of the rows of `G`. The new `Gaussian.sample`
  splits once more and avoids it by accident; deriving the toy's keys with
  `fold_in` would make it robust but changes every toy number.
- **Open from this PR**: #71 (relaxation relaxes toward the inflated
  particles, so the two compound; PR 8 should decide), #72 (an update rule
  cannot see the increment, so the ensemble Kalman sampler is a hand loop),
  #73 (the toy keys, above).
- **Float32 runs** raise at the first update with either rule (#68: scaling
  an operator promotes it to float64). A strict expected failure,
  `tests/test_eki.py::test_29_a_float32_run_stays_float32`, flips when #68 is
  fixed; remove the contract's sentence then.

## 2026-10-04: PR 9, localization

`enskit.kalman` gained `LocalizedUpdateRule`, `DomainLocalization` and
`gaspari_cohn`, in the private module `_localization.py`, implementing the
contract's new section *Localization*, which was written first. The user-guide
page is `docs/user-guide/localization.md` (its code blocks run in
`tests/test_localization.py`, which also covers the contract's obligations 20
to 26); the API reference has a section. `check_update_rule` takes
`diagonal_noise=True` for a rule that accepts only row-local noise.

**How it works.** The neighborhoods (the $K$ nearest given coordinates, ties
to the lower index) and the taper weights are computed once, when the
`DomainLocalization` is constructed. The build whitens the aligned
approximation's given factor rows once ($J$ vectors per given block), and for
each located target coordinate builds one `IdentityPlusGram` of the
$J \times K$ local whitened factor under `jax.vmap`, keeping the local gain row
$S_p^\top A_p^{-1} f_p$ (and, around `SymmetricSquareRoot`, the local
anomalies). A call whitens one vector per given block and only gathers and
contracts. Targets named with `None` get the wrapped rule's global update,
computed the same way with one global `IdentityPlusGram`; around `Matheron`
every local and global update shares one draw, `normal(k_noise, (J, N))` with
`Matheron`'s key split, so with no localization the result equals `Matheron`'s
for the same key.

**Decided here** (the contract's departures 10 to 14):

- Every target block must be named in `target_coords`, `None` for global. An
  omitted block raising beats a forgotten one going global silently.
- The stochastic local residual is $\sqrt\rho\,W(y^* - g_j) - \varepsilon_j$,
  so each local problem gets noise of its own variance $r/\rho$. The design
  prototype scaled $\varepsilon$ by $\sqrt\rho$ as well, which under-disperses
  each coordinate by $\kappa^\top\operatorname{diag}(1-\rho)\kappa$; a
  regression test measures the variance and can tell the two apart.
- Only row-local noise (`PSDDiagonal` or `Identity`, under any number of
  `PSDScaled`), only aligned approximations, no target terms. The exclusions
  section of the contract lists what was left out and why (correlated
  blocks, a localized general path for hybrids, grouping coordinates that
  share a location).
- The taper's positive-definiteness hazard of issue #14 is covariance
  localization's, not this one's: any taper with values in $[0, 1]$ gives a
  valid update, and a boxcar is tested exact. A weight above 1 raises in
  debug mode.

**Things not to rediscover.** `gaspari_cohn`'s outer branch rounds to
$-10^{-6}$ in float32 near $z = 2$, so it is clamped at zero; the square root
of a weight is taken with a zero derivative at zero, or a derivative in the
radius is `nan` wherever a weight is exactly zero. The wide-localization
comparison with `Matheron` agrees to about $10^{-14}$, not $10^{-15}$: the two
paths contract in different orders.

**Opened by the adversarial review**: #68, a scalar scale promotes a float32
operator to float64, so a float32 tempered run fails with every rule (the
localized build now refuses it by name; `SymmetricSquareRoot`'s built update
returns float64 silently, though `update` catches it), which PR 7 must settle
before float32 runs work; and #69, localization's construction and build
memory, $d_x N$ and $d_x J K$, unguarded. The review's trivial findings were
fixed here and are the contract's *Changes made while implementing* 7 to 11.

**Settled from the old open decisions.** The validity mask on `Evaluation`
is not needed by localization: updates take unweighted, finite particles, so
a failed particle is repaired or dropped before any rule sees it. Issue #11's
accessor items concern the old `gauss` module; localization needed only the
public `Gaussian.factor`, `block_cov(...).whiten` and `IdentityPlusGram`.

**For later PRs.**

- **PR 7 (EKI).** A localized rule goes in as `update_rule`; the tempered
  noise `R * (1 / delta)` of a `PSDDiagonal` is row-local, and the bare
  `given_coords` form exists for the driver's internal prediction block.
  Issue #14's demonstration (a $P = 2000$, $J = 40$ linear inversion whose
  global run stays in a 39-dimensional subspace) can become a driver test or
  example once the driver exists; the user guide has the one-update version.
- **PR 8 (EnKF).** Example 12 of the design (Lorenz-96, localized ETKF and
  stochastic EnKF) should run against `LocalizedUpdateRule` unchanged; its
  `problem.coords` and `problem.obs_coords` are what `toy.lorenz96` must
  provide.

## 2026-10-04: PR 6, `enskit.kalman`

The layer exists, implementing `docs/kalman-contract.md`, which was written
first: `update`, `gaussian_approximation`, the `UpdateRule` and
`ParticleUpdate` protocols, `SymmetricSquareRoot`, `Matheron`, and the four
functions `inflate_multiplicative`, `inflate_additive`,
`relax_to_prior_spread` and `relax_to_prior_perturbations`. The modules are
private (`_common`, `_update`, `_rules`, `_inflation`). `enskit.testing`
gained `check_update_rule` and `check_conditional_map`, as private modules
beside PR 5's `_simulator.py`. The user-guide page is
`docs/user-guide/updates.md`; the API reference has a section; the
import-linter contract names `enskit.kalman` without parentheses; and the
fresh-interpreter test in `tests/test_toy.py` checks that no layer loads
`enskit.testing` either.

**Decided here** (each is in the contract's *Departures from the design*):

- **Failed particles must be repaired or dropped before an update** (#60's
  open question). Updates refuse weighted ensembles, so a zero-weight failed
  particle can never reach an update, and the maps' every-row check needs no
  change. Dropping is `resample(key, reweight(ens, where(all_finite, 0,
  -inf)))`; `update` and both builds say so in debug mode.
- **Alignment is a checked precondition.** An `EnsembleGaussian` with the
  particles' count is trusted to realize those particles; in debug mode both
  rules check it (the *alignment check*, tolerance
  $\sqrt\varepsilon\max|a| + 64J\varepsilon\max|x|$). Without it, an
  `EnsembleGaussian` from other particles, or with a rescaled factor, gives
  wrong numbers silently on both rules. A modified approximation goes in as a
  plain `Gaussian`; `Matheron` takes its general path and
  `SymmetricSquareRoot` refuses it.
- **`update` checks its rule's result** statically (type, weights, count,
  blocks, dtype), the old EKI contract's dtype guard carried over.
- `Matheron` splits its key into `(targets, noise)` always, so adding a
  target term never changes the given blocks' noise draw. It draws the
  targets' independent terms itself; the review's 0.044-vs-4.04 regression
  is a test.
- `SymmetricSquareRoot`'s exact-value path factorizes per call, since the
  distribution layer offers exact conditioning only through `condition`.
- The rule classes are pytrees with no fields, and built updates are
  pytrees (`kalman._common.pytree_class`, explicit children, so a field may
  be a distribution or a map); both cross `jit` as arguments.

**Things not to rediscover.** `check_update_rule`'s stochastic test needs
256 replicates at five standard errors: at 64 and six, a bias of a tenth of
the posterior spread in a `Matheron` update passed. The aligned and general
`Matheron` paths agree to about $10^{-15}$, so only the whitening count
(`tests/test_kalman.py::test_7_*`) catches a regression from one to the
other; four mutations (dropping target draws, forcing the general path,
removing the alignment check, unweighted centering in `inflate_additive`)
each fail at least one test. The adversarial review found, and this PR fixed
(the contract's *Changes made while implementing*): a float64 term on a
float32 approximation gave a result with blocks of two dtypes, since
operators carry no dtype (#33) and results are assembled without the
`Ensemble` constructor, so `update` now checks every block and `Matheron`
refuses it; the alignment check's magnitude term must take the largest
magnitude over *all* blocks, because a block built by a linear map of
another rounds at the source's scale (float32, offset $10^5$, differencing:
gap $6\times10^{-3}$); and a target's independent term never enters the gain,
so a hybrid's static covariance must be absorbed into the shared factor, or
`Matheron` turns it into additive inflation.

**For later PRs.**

- **PR 7 (EKI).** Call `kalman.update(ens, {prediction: y},
  noise={prediction: R * (1 / delta)}, update_rule=..., approximation=...,
  key=...)`; repair failed particles before it. The policies wrap the four
  inflation functions. The update and inflation regressions of
  `tests/test_eki.py` are already ported (the contract's table); the driver's
  port with PR 7.
- **PR 8 (EnKF).** The hybrid approximation of Example 11 is a plain
  `Gaussian`, so only `Matheron` (general path, $k + J$ whitened vectors)
  accepts it. Its static covariance must be absorbed before the linear map;
  `maps.pushforward` through `maps.Linear` does that, as the user guide's
  example shows.
- **PR 9 (localization).** Add `LocalizedUpdateRule`, `DomainLocalization`
  and `gaspari_cohn` to `enskit.kalman`, and a section to the contract. Reuse
  `_common.check_build_arguments`, `check_particles_finite` and
  `check_alignment`; run each new rule through `check_update_rule`.

## 2026-10-04: PR 5, `enskit.maps` and the simulator contract

The layer exists, implementing the new normative page
`docs/maps-contract.md`: `pushforward`, the `StructuredMap` protocol,
`Linear`, `AdditiveNoise` and `BlackBox`. Beside it, `enskit.testing` now
exists, with `check_simulator`. Both are packages with private modules
(`enskit/maps/_pushforward.py`, `_structured.py`, `_blackbox.py`,
`_common.py`; `enskit/testing/_simulator.py`), and the import-linter
contract names both without parentheses. The user-guide page is
`docs/user-guide/maps.md`, whose code blocks a test executes and checks; the
API reference has two new sections. `tests/test_maps.py` works through the
contract's 17 obligations, then the ported regressions.

**The simulator contract moved here.** It is the EKI forward-model contract
generalized to several inputs and outputs (a tuple, mapping or `NamedTuple`
from one call), with randomness declared by `needs_key = True` on any
callable. **#19 is settled**: a return wider than the ensemble's dtype
raises, naming the simulator; integer, boolean, complex and incomparable
dtypes (float16 against bfloat16) raise too; narrower is promoted with one
warning per call. The old `enskit.eki` driver keeps its own rule until PR 7;
`docs/eki-contract.md` and `docs/user-guide/writing-a-forward-model.md` now
point at the new page.

**Departures from the stub**, each listed in the contract's *Departures*
section: `needs_key` is a declaration any callable makes, not a `BlackBox`
argument only; `BlackBox` takes a tuple `output_dim` for several outputs,
checks the host's return inside the callback (the prototype's
`np.asarray(..., dtype=...)` silently truncated an integer or wider return),
and gets its zero derivative from `stop_gradient` on its inputs rather than a
custom JVP; a `StructuredMap` pushes one output; `Linear` accepts arrays;
`check_simulator(f, input_dims, output_dims, ...)` runs through `pushforward`.

**Measured on JAX 0.10.2**, and pinned by tests:

- An exception inside a `pure_callback`, including `BlackBox`'s own checks,
  surfaces as `jax.errors.JaxRuntimeError` carrying the original message,
  eagerly too. Warnings raised inside it do reach the caller's filters.
- `stop_gradient` on the callback's inputs gives exact zeros under `grad`,
  `jvp` and `jacfwd`, with no derivative rule on `pure_callback` needed.
- Operators refuse leading axes at construction, so a vmapped family of
  maps or operators comes only from stacking a pytree's leaves; the maps
  refuse one outside `vmap`, as the distributions do.

**Open from this PR**, from its adversarial review, each with measurements
and options:

- **#63**: the bit-exact permutation check of `check_simulator` rejects a
  row-independent NumPy simulator at odd `n_particles` (BLAS kernels depend
  on a row's position). It was inherited from `check_forward_model`; the
  default `n_particles = 6` hides it on the macOS BLAS.
- **#64**: `check_simulator` cannot check a float32 simulator.
- **#65**: its subset check uses one fixed subset, and a `min`-coupled
  simulator passes at 6 of 15 seeds.

The review's defects were fixed here. The worst was a wider NumPy return
demoted silently by `jnp.asarray` with x64 off; the dtype is now read before
any conversion.

**For later PRs.**

- **PR 6 (kalman).** `enskit.testing` is a package: add `check_update_rule`
  and `check_conditional_map` as private modules beside `_simulator.py`,
  and to its index table. `kalman` never imports `maps`.
- **PR 7 (EKI).** Evaluate with `maps.pushforward(ens, forward, inputs="u",
  output="g")` and read `all_finite`. The promotion warning is **per
  pushforward call**, where the old contract promised once per run: the
  driver must either deduplicate it or the EKI contract must change.
  Delete `check_forward_model`, its section of the EKI contract, and its
  tests in `tests/test_toy.py`; the ported copies are in `tests/test_maps.py`
  under the maps contract's *Ported regression tests* table.
- **PR 8 (EnKF).** The forecast is a replacement plus a noise pushforward:
  `pushforward(ens, transition, inputs="x", output="x")` then
  `AdditiveNoise(Q)` with a key, or exactly on a projection.
- **PR 11.** `writing-a-forward-model.md` describes the old driver's
  callable; fold what it still says into `maps.md` when the old driver goes.

## 2026-10-04: failed particles in `enskit.distribution` (#60)

A failed particle, a row with a non-finite entry, is now a valid state of an
`Ensemble`. The contract's new subsection *Failed particles* states the
rules, and its *Changes made while implementing* records the change as item 8.

- **Construction and `assign` accept non-finite particles**, in debug mode
  too. The finiteness check moved to `project` and `cov`, for particles of
  positive weight only, and names the block and the remedy.
- **The weighted sums leave out particles of weight zero**, and a weighted
  ensemble's reference particle is one of largest log weight
  (`Ensemble._reference`). A `nan` particle at log weight $-\infty$ affects
  no mean, covariance, projection, other particle's anomaly or derivative,
  and a finite outlier at weight zero
  no longer costs $10^{-4}$ in the mean. Unweighted ensembles compute
  exactly as before.
- The mask goes on the operand, not the product: `w * where(w > 0, d, 0)`.
  `where(w > 0, w * d, 0)` gives the right values and a `nan` derivative
  with respect to the log weights. A regression test covers each.
- Dropping failed particles without resampling is
  `reweight(ens, jnp.where(ens.all_finite, 0.0, -jnp.inf))`, now in the
  user guide's *Weights* section.
- **Still open, for PR 6 and PR 7:** the conditional maps' tier-4 check on
  samples covers every row, so in debug mode `MatheronMap` refuses an
  ensemble with a failed particle even at weight zero. Since the update
  rules refuse weighted ensembles anyway, failed particles must be repaired
  before an update; commented on #40 and #41.

## 2026-10-04: PR 4, `enskit.distribution`

The layer exists, implementing `docs/distribution-contract.md`: `Ensemble`,
`Gaussian`, `EnsembleGaussian`, the `ConditionalMap` protocol, `MatheronMap`,
`SquareRootMap`, `reweight`, `effective_sample_size`, `resample` and
`exact_moment_ensemble`. The modules are private (`_ensemble`, `_gaussian`,
`_maps`, `_weights`, `_common`); everything public comes from
`enskit.distribution`. The user-guide page is
`docs/user-guide/distributions.md`, and the API reference has a section. The
import-linter contract now names `enskit.distribution` without parentheses.
`tests/test_distribution.py` works through the contract's obligations, then
the ported regressions under their old names, then the new ones.

**The pytree machinery.** `linalg.base._pytree_dataclass` gained
`allow_none=True`: fields annotated `X | None`, or tuples of them, are data,
and a `None` entry is tree structure. `linop` keeps it off.
`enskit.distribution._common.distribution_class` is the partial. No class
holds a distribution as a field, so that half of the issue's extension was not
needed.

**The contract was corrected where implementing it showed it wrong.** Every
change is listed in its new section *Changes made while implementing*:

- **The accuracy claim was narrowed.** The contract said the conditional mean
  was accurate to a few $\varepsilon$ at every $\sigma_{\max}$. That holds
  only where the SVD of $S$ is exact, as for axis-aligned given rows, which is
  what had been measured. For rows in general directions the exact
  conditional moves by about $\varepsilon\sigma_{\max}$ when $F_c$ is
  perturbed by one rounding: $7.5\times10^{-9}$ at $10^8$, and order 1 past
  $1/\varepsilon$. No normwise-stable algorithm can beat that, and the layer
  matches it; obligation 17 tests both statements.
- **Systematic resampling could select a zero-weight particle**, with
  probability about $J\varepsilon$ per call, because the cumulative sum
  rounds below 1. The pinned formula changed; the regression test uses a
  float32 key that hits it.
- `EnsembleGaussian`'s `factors` and `block_covs` are keyword-only.
- `compress` drops the latent space when no block has a row, and its guard
  runs only when it would allocate.
- The weighted divisor's log-sum-exps use `log1p`, which a plain
  `logsumexp` does not; without it the divisor loses about $10^{-10}$ at an
  ESS of $1 + 10^{-6}$.
- A given block without a factor row whitens only its residual.

**Open from this PR:**

- **#60**, since fixed (see the section above).
- **#33**, commented: operators carry no dtype, so the contract's "mixed
  dtypes raise" is enforced for arrays only. A float64 operator row or term on
  float32 means is accepted, and its results are promoted.

**For later PRs.**

- **PR 5 (maps).** Build Gaussians through the public constructor, with
  keyword-only `factors=` and `block_covs=` (both classes), and
  `EnsembleGaussian(..., n_particles=J)` for a result on the same latent
  space. A pushforward may write failed rows into an `Ensemble` directly;
  construction accepts them in debug mode too (#60).
- **PR 6 (kalman).** `MatheronMap` passes the targets' independent terms
  through unsampled, so `kalman.Matheron` must add those draws itself, as
  the contract's consumer section says. `particle_coefficients(values,
  key=...)` is the $J + 1$ route, and it returns coefficients, not particles.
  A weighted ensemble projects to a plain `Gaussian`, which has no
  `square_root_map`. The update rules' message should say to resample first.
- **PR 7.** Delete `tests/test_gauss.py` with `enskit.gauss`. Its regression
  tests are ported in `tests/test_distribution.py` under the same names,
  prefixed `test_regression_`.

## 2026-10-04: PR 10, the Kronecker operators

`enskit.linalg` gained the module `enskit/linalg/kronecker.py`: three
classes for $A \otimes B$ and the factory `kron(A, B)`, which chooses
among them by the operands' levels, as `c * op` chooses among the scaled
operators. They are specified in `docs/linop-contract.md`, *Kronecker
products*, and catalogued in `docs/user-guide/operators.md`.

- **`Kronecker(A, B)`**: any two operators, application and transposition
  only. `T` is `A.T ⊗ B.T`, built without the constructor so that it works
  on a family.
- **`SquareKronecker(A, B)`**: two square operators; adds `solve`,
  `logdet` ($n_B\log\lvert\det A\rvert + n_A\log\lvert\det B\rvert$) and
  `diag`, each supported when both operands support it.
- **`PSDKronecker(A, B)`**: two PSD operators; adds `factor`
  (`kron(A.factor(), B.factor())`) and a primitive `whiten`
  ($W_A \otimes W_B$).

The first operand's index is the slow one, as in `numpy.kron`. Every
operation reshapes the operand's trailing axis and applies $B$ (a vector
method) and then $A$ (the matching matrix method). A count test checks that
no operation forms an array of side $n_A n_B$.

**Scope, settled with the maintainer.** PR 10 is `Kronecker` alone.
`KroneckerPlusNugget` and `KroneckerLMC` moved to issue #55 with #15's
notes and warnings, to be built when a method needs them. `LowRankPlus` was
dropped because `LowRankUpdate` already provides it. Issue #1, closed by
the contract rework, needed nothing further here: `factor()` of two
`DensePSD` operands is a `SquareKronecker` of their `Triangular` factors,
which solves, and `whiten` applies the operands' whiteners directly.

**The conformance suite cannot see orientation.** It compares each
operation with `to_dense`. An implementation reversed in every method at
once passes `check_operator` at any sides, square or rectangular, while
its `matvec` is off by tens. `tests/test_kronecker.py` therefore pins
`matvec`, `rmatvec`, `solve`, `whiten`, `diag`, `logdet` and `to_dense` to
`numpy.kron` separately, and under that mutation 15 of its tests fail. Any later Kronecker-structured
operator needs the same tests, until issue #57 lets `check_operator` take
an independent dense reference.

**For later PRs.**

- PR 4: a prior given as `kron(A, B)` can be a `Gaussian`'s independent
  term, since it whitens; `LowRankUpdate(kron(...), F)` is in the
  conformance list.
- PR 11: the operator page has a *Kronecker products* section. The
  user-guide level "operators" should keep it.

## 2026-10-04: PR 3, the distribution contract

`docs/distribution-contract.md` specifies `enskit.distribution` before any
code exists. It absorbs `docs/gaussian-contract.md`, which keeps governing
`enskit.gauss` until PR 7 and now says so in its status box. The page lists
its departures from the stubs in its section *Departures from the design*;
PR 4 implements the page, not the stubs, wherever they differ.

**Settled here:**

- **#10.** Every conditioning path runs the same two debug-mode checks:
  a *whitening check* right after whitening (it names the given block and
  the likely cause, since `IdentityPlusGram` does not check `S`), and a
  *result check* last. The whitening counts are a table in the page's
  *Cost* section and are count-tested (obligation 15).
- **#25.** Numerically singular noise that still whitens is not a silent
  wrong answer: it approximates the limit (measured agreement $5\times10^{-9}$
  for a rank-7-of-8 dense term). Exactly singular noise that whitens to
  `inf` is caught by the whitening check in debug mode and is `nan`
  otherwise. The covariance's loss of accuracy at large $\sigma_{\max}$ is
  real only for a thin basis ($k > N$): about $\varepsilon\sigma_{\max}$
  relative, of either sign. Past $\sigma_{\max}\approx1/\varepsilon$ the
  variance is exactly $0$ along a coordinate axis and overstated by orders of
  magnitude along a random direction. It is recorded in the page and opened
  as **#54** against `IdentityPlusGram`, with options.
- **Differentiability** is stated as PR 2 measured it: first derivatives
  degenerate-safe, second derivatives exact only for the log-determinant
  parts (#52).

**Departures PR 4 and PR 6 need to know about:**

- `log_density`'s quadratic term goes through `LowRankUpdate(D_c,
  F_c).whiten`. The obvious $\lVert b\rVert^2 - \langle Sb, A^{-1}Sb\rangle$
  cancels catastrophically, and is wrong by a factor of several at
  $\sigma_{\max} = 10^8$. The adversarial review found this; the page has
  the measurements.
- `Gaussian(means, *, factors=None, block_covs=None, latent_dim=None)`: the
  two mappings are keyword-only, since a covariance passed as a factor row is
  silent.
- Absent factor rows and terms, and an unweighted ensemble's `log_weights`,
  are `None` and so pytree structure: a `lax.scan` carry cannot change them
  (**#56**).
- `MatheronMap.particle_coefficients(values, *, key)` is new, and its key is
  required. It is the
  public route to the $J+1$ whitenings that `kalman.Matheron`'s aligned path
  needs; the prototype reached into the map's private `S` instead. It returns
  coefficients, not particles, so the caller adds $F_x w$ to its own
  particles and a no-op stays bit-exact.
- Value checks follow the tiers: `project`'s concentrated-weights check and
  `reweight`'s `nan`/`+inf` check are debug-mode only (the stubs had them
  eager).
- Every block has a factor row or an independent term; `square_root_map`
  and `conditional_map` take only blocks with independent terms; `absorb`
  raises on a block without one; `compress` always returns a plain
  `Gaussian`; `sample` needs `n_particles >= 2`.
- The pinned draws, including split arities, are a table in the page's
  *Randomness* section. Snapshot them in PR 4.
- The page's *Conformance* section ends with a table mapping each regression
  test in `tests/test_gauss.py` to its port. PR 4 should port them under
  those names and keep the old docstrings' reasoning.

## 2026-10-03: PR 2, the linalg additions

`enskit.linalg` gained the conditioning core and three smaller pieces, all
specified in `docs/linop-contract.md` and catalogued in
`docs/user-guide/operators.md`:

- **`IdentityPlusGram(S)`**, in the new module `enskit/linalg/gram.py`:
  $I + SS^\top$ for a `(k, N)` array, one thin SVD stored at construction,
  with `solve_factor` ($A^{-1}Sb$) and `inverse_sqrt()` (an
  `IdentityPlusGramInverseSqrt`, a `PSDLinOp` with `matvec` and
  `to_dense`) beside the usual PSD operations. Its custom JVP rules are the
  contract's section *The conditioning core*.
- **`Zero(n_out, n_in, *, dtype=None)`**, storing no array; `HStack` and
  `Product` skip it when applying.
- **`LowRankUpdate(base, F)`**, $D + FF^\top$, built on `IdentityPlusGram`
  of $S = (W_D F)^\top$.
- **`dense_fallback(max_n=2048)`**, opt-in, in `enskit/linalg/base.py`.
  `UnsupportedOpError` takes an optional fourth argument, `detail`.

`enskit.gauss` now routes every conditioning through `IdentityPlusGram`, so
its updates differentiate finitely at exactly collapsed ensembles and
zero-padded columns. Its behavior is otherwise unchanged; the one test that
pinned the old `nan` gradient was ported to its positive form (see the
test's docstring). `uv run lint-imports` checks the layer rules in CI.

**Departures from the stubs, all recorded in the contract:**

- `LowRankUpdate`'s second parameter is `F`, not `factor`: a field named
  `factor` would shadow the `factor()` method.
- `Zero`'s sizes are at least 1, not 0: the operator contract rejects empty
  operators everywhere, and a Gaussian with no factor row has `None`
  rather than a width-0 row.
- `product` and `hstack` still return `Product` and `HStack` when given a
  `Zero`; the skipping happens when they are applied. Returning a `Zero`
  from `product` broke the conformance suite's arithmetic check, which
  requires `op @ op.T` to be a `Product`.
- `LowRankUpdate.whiten` is degenerate-safe too, not only `solve` and
  `logdet`: it uses the inverse square root of $I + S^\top S$ from the same
  stored decomposition.
- `IdentityPlusGram` does not check that `S` is finite, even in debug mode,
  so the layer that built `S` reports a non-finite result with its own
  message and cause (the gauss contract requires this, and a test pins it).

**Derivatives, as measured.** First derivatives are finite and correct at
every $S$, in forward and reverse mode, under `jit` and `vmap` (forward
mode raises under `jax_debug_nans` at degenerate spectra, from the SVD's
discarded tangents; issue #52). Second derivatives are promised for
`logdet` only. The others are exact at well-separated singular values, lose
accuracy like $\varepsilon/\delta$ near a tie of gap $\delta$ (finite and
wrong, once with the wrong sign), and `jax.hessian` of `solve_factor` is
`nan` at an exact tie — linearization inlines the rules' own arithmetic,
which reads `U` and `sigma`. The decomposition is deliberately *not*
stop-gradiented, which would make them wrong at well-separated spectra too.
The design's §9.1 promise of "degenerate-safe" derivatives therefore holds
for first derivatives only, and the distribution contract (PR 3) should say
so.

**For later PRs.**

- PR 3 and PR 4: build conditioning on `IdentityPlusGram(S)` with
  `S = (W F_y)^T`; check results for finiteness at the distribution layer,
  with the cause in the message, as gauss does. `Gaussian.cov(a)` returns
  `LowRankUpdate(D_a, F_a)`, whose `base` must support `whiten`.
- PR 4 (and each PR that adds a layer): remove the parentheses around the
  new layer's name in `pyproject.toml`'s import-linter contract, so a
  misnamed module fails the contract instead of being skipped as optional.
- PR 7: delete the `enskit.eki` and `enskit.gauss` lines from that contract
  when the modules go.
- PR 10: every Kronecker class goes through `check_operator`; `Zero` and
  `LowRankUpdate` are in `tests/test_conformance.py` as models of paired
  instances.

## 2026-10-03: PR 1, the mechanical renames

The package is **`enskit`**. PR 1 (#35) changed names and nothing else; the
test count is unchanged at 537. What moved:

- `pyeki` → `enskit` everywhere outside `docs/redesign/`, including the
  distribution name, the Sphinx title, the CI coverage flag and the figure
  override, now `ENSKIT_DOCS_FIGURES=force`. Prose says "EnsKit".
- American spelling throughout. The public names it changed are the
  `center_misfit` fields of `Evaluation` and `HistoryRecord` (were
  `centre_misfit`); test names changed with them. Two quoted paper titles keep their British spelling, in the EKI
  contract's references table.
- `SquareLinOp.n` → `SquareLinOp.dim` (#20). The Gaussian contract now
  states one naming rule for both layers instead of recording a departure.
- Every module lists its public classes and functions first, in its index
  table's order, with private helpers below; in modules divided by section
  banners, the helpers' banners say `private`.
  `enskit.linalg.base` has no index table, so it follows its docstring's
  sections: the three levels, then `linop` and `static_field`, then
  `UnsupportedOpError` and `densify`. `enskit.linalg.testing` puts
  `check_operator` first, since its docstring opens with it. Module
  constants go at the top, after `__all__`; the one exception is
  `_DERIVED_DEFAULTS` in `enskit.linalg.base`, which refers to the classes
  and so follows them. New modules should be written in this order from the
  start.

**Still named `pyEKI`, deliberately:** the GitHub URLs in `pyproject.toml`,
`docs/conf.py` and `README.md`, and the clone instructions in
`docs/installation.md` (`git clone …/pyEKI.git`, `cd pyEKI`, and the
`../pyEKI` path of a local checkout). They change when the repository is
renamed; GitHub redirects the old URLs in the meantime. The prototype in
`docs/redesign/prototype/` still imports `pyeki.linalg`, so it runs only on a
checkout from before this PR, such as `b1d829c`.

**Left for PR 11:** the package docstring and the README and landing-page
taglines still describe EKI alone ("Ensemble Kalman Inversion for
derivative-free Bayesian calibration"). Renaming did not change what the
package does, so they were not rewritten here; `pyproject.toml`'s description
already reads "Building blocks for ensemble Kalman methods".

## 2026-10-03: the EnsKit redesign is adopted

pyEKI is being redesigned as **EnsKit** (`enskit`), a toolkit of ensemble
Kalman building blocks with EKI and the EnKF shipped as algorithms built from
them. The design was settled over four rounds of review and is recorded in
`docs/redesign/index.md`; the full document, with every public API written out
as docstrings and fifteen worked examples, is `docs/redesign/design.html`.
`CLAUDE.md` was rewritten for it: new layers, vocabulary ("particle" for an
element of an `Ensemble`, "sample" for any draw) and four new rules
(mathematics in `.. math::` blocks, American English, public API first,
citations).

**The plan is thirteen pull requests**, numbered 0 to 12 in
`docs/redesign/index.md`, each tracked by an issue in the "EnsKit 0.1"
milestone. This is PR 0. **PR 1 is next**: the mechanical renames (`pyeki` →
`enskit`, American spelling, public-API-first ordering, `SquareLinOp.n` →
`dim`) with no behavior change.

Three things are waiting on Andrew before PR 1: reserving `enskit` on PyPI,
renaming the GitHub repository when PR 1 merges, and deciding whether to keep
or retire the nine tutorials.

**What is in `docs/redesign/`, and how to use it:**

- **`stubs/`**: the docstrings each layer starts from. They are a design, not
  a contract. Each layer's normative contract is written and reviewed before
  its code (PRs 3 and 6, and the EKI contract's revision in 7), and it may
  refine them.
- **`notebooks/`**: the fifteen examples as notebook sources, their builder,
  the executed notebooks and the bibliography. Every entry in
  `references.py` was checked against the publisher or arXiv record, so cite
  from it.
- **`prototype/`**: the throwaway implementation the examples ran against. It
  imports `pyeki.linalg`, so it only runs on a tree from before the rename. It
  skips validation and densifies, so treat it as a record, not as code to
  copy. Its examples, and `review_checks.py`, become acceptance tests.

Until PR 7 the old `gauss` and `eki` modules stay in place, under their own
contracts. Do not extend them, and do not make the new layers depend on them.

## Where things stand

**Done.** `pyeki.linalg` is implemented to the specification in
`docs/linop-contract.md` — the normative reference for the layer's behavior,
written and adversarially reviewed before this implementation. Three-level
hierarchy (`LinOp`/`SquareLinOp`/`PSDLinOp`) with template methods (public
methods gate and validate; authors implement `_`-prefixed hooks), transposes
(`rmatvec`, `T`), operator arithmetic (`@` composition, scalar `*`/`/`),
six elementary operators, nine composites with factory functions, a debug
mode for value
preconditions, and a 14-check conformance harness; the full test suite passes.
Documentation builds with zero warnings: landing page, installation,
quickstart, three user-guide pages plus a guide to writing an operator, the
three normative contracts, design notes, and an API reference.

`pyeki.gauss` is implemented to `docs/gaussian-contract.md`: `Gaussian`,
`GaussianJoint`, `EmpiricalJoint` and the two array-level conditioning
primitives, all routed through the whitened-SVD kernel. `PSDLowRank`, the
operator it needed, is in `pyeki.linalg`.

As of 2026-09-02 the joint is **two** classes. `GaussianJoint` holds a joint
Gaussian as a *joint factor* — one factor of the block covariance, cut into
the row blocks that drive both blocks from a shared latent vector — and owns
`condition` and the pathwise (Matheron) map. `EmpiricalJoint` holds paired
samples, offers `to_gaussian_joint()`, and keeps the two updates that return
samples. `condition` is gone from it: conditioning samples means conditioning
a Gaussian fitted to them, and that fit is now written at the call site.

The point of the split is `GaussianJoint.from_linear_map`, which builds the
joint of $u$ and $Gu$ and so gives closed-form linear-Gaussian posteriors —
previously unreachable, since the only entrance to conditioning was to
present samples. `docs/joint-factor.md` derives the representation and records
why it is a factor rather than three covariance blocks. The square-root update
stayed on `EmpiricalJoint` because its reading of the conditioned factor is
valid only for a centered one, which holding samples makes structural; the
contract and that page both give the measured failure it avoids.

`pyeki.eki` was not touched: both update policies call the same two methods
with the same signatures.

**`pyeki.toy`** shipped on 2026-09-02, and is the module `CLAUDE.md` had been
promising: three toy problems, each a forward model bundled with a prior, a
noise covariance, synthetic data and the parameters that generated it.
`linear_gaussian` at any pair of dimensions, whose `posterior(level)`
delegates to `GaussianJoint.from_linear_map(...).condition(y, R / level)` —
two lines, and the reason the split above had to land first; the
`exponential_decay` problem, tuned until a unit step and an adaptive ladder
differ reliably rather than coincidentally; and `restricted_decay`, the same
model with a valid domain, so a member whose rate is not positive returns a
non-finite row. Its user-guide page is `docs/user-guide/toy-models.md`.

The problems are frozen dataclasses of plain values and are deliberately
**not callable**: `run` takes the triple, and the contract excludes a
container accepted in its place. They are not pytrees, hold no mutable state
and count no calls — the recorder in `tests/test_eki.py`'s `_AffineProblem` is
a test instrument and stays there. The existing local closures in `tests/`
were **not** migrated: several are instrumented and several write their
reference locally on purpose, which is what makes them regression tests for
the layer.

`pyeki.eki.testing` gained **`check_forward_model`**, which checks a user's
own model from outside a run: shape at two ensemble sizes, dtype, determinism,
and row independence — twice, because permuting the members is bit-exact and
catches order-dependent coupling while a symmetric coupling survives it and
needs a subset re-evaluation, to a tolerance. The contract's "nothing detects
this" about row coupling is now "no *run* detects this", in both the contract
and the guide.

`pyeki.eki` is implemented to `docs/eki-contract.md`: the four value classes,
the three policy protocols with eight shipped implementations, the two public
phases of a step, `run` and `iterate`, the three array-level helpers, and
the `pyeki.eki.testing` conformance harness for user-written policies. Its
user-guide page is `docs/user-guide/running-an-inversion.md`.

The **forward-model contract** is specified in one place as of 2026-08-28, in
the contract's *Forward models and failed members* and in the user-guide page
`docs/user-guide/writing-a-forward-model.md`. Three properties the driver had
been deciding on its own are now stated and tested (obligations 27-30): what
the callable receives, what it may return, and what it must be. It landed on
its own branch, deliberately ahead of the toy forward models and the first
tutorial, both of which consume it — a session writing the contract *and* the
models satisfying it is under pressure to bend the first toward the second.

The **layer vocabulary** was unified at the same time, and the rules are now
in `CLAUDE.md`. One word per concept: a *run* contains *steps*, each step has
two *phases* (`evaluate` and `assimilate`) made of numbered *operations*, and
each step is preceded by one *evaluation* of the forward model. "Rung" and
"iteration" as a countable noun are retired. Vocabulary flows downward only:
`pyeki.gauss` has samples, not members, and `pyeki.linalg` speaks of neither
Gaussians nor conditioning. The renames that followed: `EnsembleJoint` ->
`EmpiricalJoint`, its `n_members` -> `n_samples`, `apply` -> `assimilate`,
`EKIResult.n_steps` -> `n_evaluations` plus a new `n_completed_steps`, and
`n_obs` -> `v_dim`. `docs/user-guide/conditioning.md` was rewritten in the
gauss layer's own vocabulary at the same time; it was the last prose describing
that layer in EKI's words.

Two things came out of the adversarial review of that branch. An **inflation's
output dtype is now checked**, as an update's already was — the inflated
members are what the forward model is called on, so an inflation returning
`int64` used to hand the model an integer ensemble with nothing raised.
And **a forward model returning a dtype *wider* than the run is not demoted**,
so it fails at the update's dtype check with an error naming the update rule;
that is deliberate for now and recorded as issue #19.

**Tutorial 1** shipped on 2026-09-04, complete and revised, with the figure
machinery the rest of the series needs. Tutorials 2 and 3 are drafted, tested
and building, but have **not** been through a revision pass — they are marked
as unreviewed drafts on the series index, and they run the library's default
update rule rather than the pathwise one tutorial 1 selects. Revising them is
its own PR.

The series was also restructured: tutorial 3 was carrying four lessons, so it
split into three — sampling against optimizing (the destination), tempering
schedules (the path), and the two update rules — which pushed the four
remaining stubs down by two. The series is now nine pages, of which 4 to 9 are
stubs.

**Tutorial 1 runs `PathwiseUpdate`, not the default.** On
`exponential_decay`, `TransformUpdate` reproduces the target's covariance but
leaves the ensemble strung along one direction: the across-ridge projection
has a kurtosis of 19 against a Gaussian's 3, and at 64 members two members
carry 72% of that variance. It is the nonlinearity, not sampling error —
present after a single step, and no better at 4096 members, while on
`linear_gaussian` the same rule leaves the ensemble perfectly Gaussian at any
number of steps. `PathwiseUpdate` draws a perturbation per member and refills
the cloud, at about one extra forward evaluation. Issue #29 asks whether the
package default should change; if it does, the explicit argument and its
explanation come out of the page.

Three things the first three pages settled, none of which had a precedent:

- **Figures are generated at build time.** `docs/figures.py` is both the figure
  module and a Sphinx extension: a `builder-inited` handler writes every figure
  into `docs/_generated/figures`, which is gitignored. Nothing is committed, so
  a figure cannot disagree with the code that made it. Regeneration is skipped
  when every output postdates both that module and every source file of the
  package — 6 s to regenerate all of them, 1.6 s cached — and
  `ENSKIT_DOCS_FIGURES=force` overrides. That alone catches only a figure whose
  code *raises*, so each figure function also returns the numbers it plotted
  and `tests/test_tutorials.py` pins them; a figure drawing the wrong array
  fails the test rather than merely looking wrong. Pixels are deliberately not
  compared, and the module records why.
- **The quickstart stays.** Tutorial 6 (was 4) does not absorb it: four pages
  link to it, and it serves a reader who came for `pyeki.linalg` alone.
  Tutorial 6 is short and problem-led instead, and its stub now says so.
- **Notebook wiring is still undecided**, deliberately. `myst-nb` would
  *replace* `myst_parser` rather than join it, which changes how all 23 existing
  pages are parsed, and that does not belong in a tutorials branch. The three
  sub-decisions at the bottom of `docs/examples/index.md` stand as written, and
  the figure machinery above forecloses none of them.

The nonlinear problem has no closed form, so the pages compare against grid
quadrature — `figures._tempered_moments`, which **refines its box**, because
one grid does not serve every level: a box wide enough for the prior resolves
the $\beta = 1$ target with about five points across its width and reports a
mean wrong in the fourth decimal, while a box sized for the target puts the
prior's mean at `[1.48, 1.31]` instead of `[1, 1]`. After two refinements the
moments agree to six digits across three resolutions and recover the prior
exactly at $\beta = 0$. Both failure modes are silent and invisible in a
contour plot; `tests/test_tutorials.py` asserts against each.

**Not started.** `pyeki.localize`, and the Kronecker family of operators. The
design background for both is in `docs/design.md`. Tutorials 4 to 9 and the
example notebooks are unwritten. Tutorial 7's stub (was 5) carries re-measured
numbers from the shipped problem; issue #24 covers what its page still owes.
Issue #26 proposes a fixed-budget ablation study and leaves open whether it is
a tutorial or a notebook.

**Origin.** This package was extracted from a research repository where the
operator layer was first written. That repository keeps the domain-specific
work — forward models, priors, experiment configuration — and will depend on
pyEKI. Nothing domain-specific should come back across.

## Next steps

Follow the plan in `docs/redesign/index.md`, one pull request at a time, in
a fresh session for each, as described in "Pull requests and handoffs" in
`CLAUDE.md`. PRs 1 (#35), 2 (#36), 3 (#37), 4 (#38), 5 (#39), 6 (#40),
7 (#41), 8 (#42), 9 (#43), 10 (#44), 11 (#45) and 12 (#46) are done: the
plan is complete. Further work is tracked in the "EnsKit 0.2" milestone and
the open issues.
#54's fix would change the regressions pinned in
obligation 17, by design. The notes below, written before the redesign,
still apply to the pull requests they name; their `pyeki` modules are now
`enskit` modules.

## Open decisions

Deferred deliberately, with enough context to settle later:

**Operator addition dispatch.** There is no `__add__` on operators and no
registry of simplification rules. With the current type list a registry would
carry about two rules. When one is added it needs: a walk over the method
resolution order rather than exact type lookup; an n-ary flattened sum rather
than binary nesting; and a way for a rule to decline. A reasonable alternative
is not to simplify on addition at all, and instead dispatch on structure inside
`solve` and `logdet`.

(Three decisions previously listed here were settled. Capability declaration
and whitening versus triangularity went to the operator contract: `supports()`
is defined by hook presence with derived-dependency resolution, and
`cholesky()` was removed in favor of `factor()` plus a primitive `whiten()`.
`AdditiveInflation`'s supposed per-step refactorization turned out not to
exist: every shipped PSD operator factorizes at construction, so `factor()`
returns a stored factor and the update path contains no Cholesky at all.)

## Things not to rediscover

Each of these cost real effort to find and produces wrong numbers rather than
errors. Each is recorded in `docs/design.md` or in the contract for its
layer; this is the index.

| finding | consequence |
| --- | --- |
| Circulant embedding gives `matvec` and sampling but **not** `solve` or `logdet` on a restricted grid | a spectral log-determinant would be silently wrong |
| For exponential correlation, the *whitener* is bidiagonal, not the factor | sampling is a sequential recurrence, not a banded solve |
| A scalar correlation coefficient is wrong for irregular observation times | build the precision from per-interval coefficients |
| A reversed Kronecker orientation is a valid covariance | $B \otimes A$ is PSD whenever $A \otimes B$ is, and always the same shape; nothing raises at any sizes |
| A self-consistent `matvec`/`to_dense` pair passes the whole conformance suite | reversing the orientation in every Kronecker method passes `check_operator`; pin each method to `numpy.kron` separately |
| The Kronecker log-determinant pairs each operand with the *other's* side | the swapped pairing is identical at equal sides, so only unequal sides test it |
| A rectangular Kronecker product reshapes its operand and result differently | `(k_A, k_B)` in, `(n_A, n_B)` out; reshaping both alike is right only for square operands |
| `KronPlusNugget` log-determinant needs an $n\log\det C^l$ term | omitting it is off by a factor, silently |
| Kronecker-plus-nugget needs a strictly positive-definite nugget | a singular one has no simultaneous diagonalization |
| A tapered covariance is PSD only if the taper is a valid PD function | dimension-dependent; use a known family |
| `M @ x` contracts the wrong axis for `ndim >= 2` | wrong answer, no error, whenever the operator is square |
| A `to_dense`-based test does not exercise `matvec` | the obvious guard test is vacuous |
| Lazy factorization caches are discarded inside traces | silent ~10x slowdown |
| Undeclared non-array dataclass fields become tracers | fails later, far from the declaration |
| JAX has no generalized `eigh` | reformulate via Cholesky whitening |
| Per-step noise is $\Sigma/\Delta\beta_t$, never $\Sigma/\beta_t$ | a plausible posterior, wrong by $(T+1)/2$ times the data precision on a uniform $T$-step ladder, growing with ladder length |
| A single-`where` guard sends a `nan` misfit to the `inf` branch | `nan > 0` is `False`, so the schedule silently returns the *largest* allowed step |
| A Python float passed as a `jit` **argument** does not retrace | the retrace-per-step bug is a *static field* on an object crossing the boundary, so never pass an `EKIState` or `Evaluation` whole |
| `Evaluation.center_misfit` is not the mean of `Evaluation.misfits` | they differ by exactly $\tfrac{J-1}{2J}\operatorname{tr}(W \widehat C_{vv} W^\top)$ |
| The repair formula is not bit-exactly the identity when nothing failed | it must be `jnp.where(valid, ensemble, center)` *and* skipped in Python on the synchronized `n_valid` |
| A static field on a `HistoryRecord` makes every record a different pytree | `jax.tree.map` across a history raises instead of stacking |
| `step` is cumulative across runs | chaining a fresh ladder onto a finished state returns unchanged, with nothing raised — use `restart()` |
| `cov.factor()` is free, because operators factorize at construction | "hoisting" it by storing a densified factor turns an $O(P)$ diagonal into a $P \times P$ array |
| Every comparison against `nan` is `False` | a bisection on such a comparison silently returns its lower bracket, and a floor then makes that look like an ordinary step |
| `jnp.mean(x, axis=-2)` without `keepdims` | the subtraction right-aligns against the batch axis, so an operand whose leading axis equals $J$ broadcasts and returns wrong anomalies without raising |
| `np.asarray` on the forward model's argument returns a **read-only view**, not a copy | writing into it raises `assignment destination is read-only` from wherever you wrote, not at the conversion; copy with `np.array` |
| A run's evaluations and its completed steps differ by one whenever it ends on a stopping rule or a `None` increment | a single `n_steps` naming both is how the ambiguity arose; the terminal record is the one with a zero increment, at most one, always last |
| A policy's output needs its **dtype** checked, not only its shape | an inflation returning `int64` handed the forward model an integer ensemble and the run completed silently; the shape check passed |
| A `float32` forward model is promoted and warned about, not rejected | it still costs ~$7\times10^{-5}$ relative in the posterior mean where the prediction mean exceeds the spread by $10^4$; promotion recovers only about half, since the digits are gone before the array arrives |
| `ensemble @ G` instead of `ensemble @ G.T` is silent when $G$ is square | the transposed model's predictions, right shape, no error; `G @ ensemble` raises, so it is the harmless mistake |
| A **symmetric** coupling across ensemble members survives a permutation of them | mean-centering is permutation-equivariant, so a permutation test alone passes a model that normalizes across the ensemble; it takes re-evaluating a *subset* to catch, and that comparison cannot be bit-exact |
| The same members in a differently *sized* batch round differently | a dense contraction picks a different kernel per batch shape, so a subset comparison holds only to round-off while a permutation one is bit-exact |
| The closed form's cost is set by the prior factor's width $k$, not by $P$ | a full-rank prior at $P = 2000$ means a $(2000, 2000)$ posterior factor — 32 MB, 0.07 s — so `LinearGaussian.posterior` guards on $Pk$ and the *run* has no such limit |
| A run at $P \gg J$ reports a spread the exact posterior contradicts | at $P = 2000$, $N = 40$, $J = 40$ the ensemble's mean posterior sd is 0.014 against an exact 0.990, a factor of seventy, with nothing raised and no history field flagging it |
| A deterministic square-root update can get the covariance right and the shape wrong | on a nonlinear problem `TransformUpdate` leaves the least-varying direction with a kurtosis of 19 and two members holding 72% of its variance; it is a linear recombination of existing anomalies, so more members do not help. Only visible in a scatter plot or a shape statistic — no `HistoryRecord` field reports it |
| One grid cannot serve every tempering level | a box wide enough for the prior reports the $\beta = 1$ mean wrong in the fourth decimal; a box sized for the target reports the prior's mean as `[1.48, 1.31]` rather than `[1, 1]`. Both are silent, and invisible in a contour plot |
| A conditional's accuracy at large $\sigma_{\max}$ is set by its own sensitivity, not by the algorithm | for given rows in general directions, perturbing $F_c$ by one rounding moves the exact conditional by about $\varepsilon\sigma_{\max}$; only axis-aligned test rows show "a few $\varepsilon$", so measure against that sensitivity rather than against $\varepsilon$ |
| `cumsum` of normalized weights can end just below 1 | systematic resampling with a clip to `J - 1` then selects the last particle even at weight zero; scale positions by `cumsum[-1]` and bound by the last positive weight |
| A numerically singular noise covariance that still whitens gives a *correct* conditional to about $10^{-9}$, not a wrong one | the conditional exists whenever $F_cF_c^\top + D_c$ is nonsingular; what fails silently is the thin-basis covariance at large $\sigma_{\max}$, which collapses to exactly $0$ (#54) |
| A multi-line `:alt:` value breaks a MyST `{figure}` | the continuation lines are absorbed into the caption, and the build fails with "Figure caption must be a paragraph" pointing at the directive rather than at the option |
| A chaotic toy's numbers are not portable | Lorenz-96's spin-up turns a $10^{-15}$ change into an $O(10)$ change in the state, so a digit pinned on macOS fails on CI's Linux; test claims in bands measured across seeds |
| `jax.random.fold_in(k, i)` equals `jax.random.split(k, n)[i]` for every `n > i` (partitionable threefry, the default) | a key "derived" by folding in a small integer is one a caller's split of the same key also produces, so a toy's draw silently repeats a run's; fold in a large constant, then split |
| The minimum-norm regression of centered outputs on centered inputs is not `pinv([1, X])` | the second penalizes the intercept too; both interpolate the particles, so nothing about the fit tells them apart (differs by $5\times10^{-2}$ at $J = 10$, $d_x = 30$) |
| A singular independent term (a residual covariance $\Omega$) pushes through `AdditiveNoise` without complaint | the result's `cov` of that block raises later, from `LowRankUpdate`; absorb the term first |

## Working agreements

- `uv` for everything. `uv run pytest` before every commit.
- Docstrings follow `CLAUDE.md`. Every user-facing feature gets a user-guide
  entry, not only an API entry.
- Keep the package domain-agnostic. If a docstring wants to mention a specific
  application, that is a sign it belongs in the calling repository.
