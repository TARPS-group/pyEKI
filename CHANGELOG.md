# Changelog

Every release of EnsKit, newest first. Until 1.0, a minor release (0.2, 0.3,
...) may change the interfaces, and its entry lists every such change under
**Changed** or **Removed**; a patch release fixes defects and does not change
them.

## 0.2.0 (unreleased)

### Added

- **Statistical linearization.** `Gaussian.regression(target, given=...)`
  returns the conditional of one block given others as an affine function,
  a `Regression` of coefficient operators, an intercept and a residual
  covariance: least squares without independent terms on the given blocks,
  ridge regression with them, and the minimum-norm solution, with
  `min_norm=True`, when there are more given coordinates than an ensemble's
  rank. `maps.statistical_linearization(dist, inputs=..., output=...)`
  packages it as a `Linear` map with the particles' residuals, a
  `Linearization`. Example 16 works it through.
- **A choice of divisor for covariances computed from particles.**
  `Ensemble.cov`, `Ensemble.project` and `maps.statistical_linearization`
  take `unbiased=True`, the default and the previous behavior ($J - 1$, or
  $1 - \sum_j w_j^2$ weighted); `unbiased=False` divides by $J$, or $1$,
  giving the moments of the empirical distribution itself, as the points of
  a quadrature rule with nonnegative weights need. An `EnsembleGaussian` stores its divisor
  (`unbiased`, and the property `divisor`), and every operation that reads
  particles out of it, the Kalman rules' included, uses that divisor.
  `MatheronMap` gains the attribute `divisor`.

## 0.1.0

The first release. EnsKit is the toolkit of ensemble Kalman building blocks
that the `pyeki` package, never released, was redesigned into; code written
against a `pyeki` checkout does not run unchanged, and the user guide and the
contracts describe the new interfaces.

### Added

- **`enskit.linalg`**: structured linear operators over a three-level
  hierarchy (`LinOp`, `SquareLinOp`, `PSDLinOp`): elementary operators,
  composites (products, stacks, block diagonals, low-rank updates), Kronecker
  products, and `IdentityPlusGram`, the conditioning core, whose derivative
  rules stay finite at repeated and zero singular values. Unsupported
  operations raise rather than densify; `dense_fallback` is the explicit,
  off-by-default exception.
- **`enskit.distribution`**: `Ensemble`, `Gaussian` and `EnsembleGaussian`
  over named blocks, with sampling, marginals, conditioning and projection;
  the conditional maps `MatheronMap` and `SquareRootMap`; weights,
  effective sample size and resampling.
- **`enskit.maps`**: `pushforward` of a distribution through a simulator,
  exactly through `Linear` maps and `AdditiveNoise`, and by particles through
  a `BlackBox`.
- **`enskit.kalman`**: one ensemble Kalman update, `update`, with the
  `SymmetricSquareRoot` and `Matheron` rules; domain localization
  (`LocalizedUpdateRule`, `DomainLocalization`, `gaspari_cohn`); and
  multiplicative and additive inflation and relaxation to the prior.
- **`enskit.algorithms.eki`**: Ensemble Kalman Inversion. `run` drives both
  the approximate-sampling and the optimization form, with fixed and adaptive
  tempering schedules, the discrepancy stopping rule, failed-particle repair,
  and a history of every step; `evaluate`, `assimilate` and `advance` expose
  a step's phases for loops of your own.
- **`enskit.algorithms.enkf`**: the ensemble Kalman filter, as `forecast`,
  `analysis` (which also returns the one-step log evidence) and `filter`, the
  cycle over a sequence of observations.
- **`enskit.algorithms`**: the inflation and relaxation policies both drivers
  share.
- **`enskit.testing`**: conformance checks for a simulator, update rule,
  conditional map, schedule, stopping rule, inflation or relaxation of your
  own, and, in `enskit.linalg.testing`, `check_operator` for an operator.
- **`enskit.toy`**: five problems for trying the library and for its tests:
  `linear_gaussian`, `exponential_decay`, `restricted_decay`, `lorenz96` and
  `linear_state_space`.
- **Documentation**: three tutorials (six more are outlined), a user guide
  organized by level of abstraction, fifteen worked examples, and a normative
  contract for each layer.

### Known limitations

Each has an issue with the measurements and the options.

- A float32 EKI run, in either form, raises `TypeError` at its first update,
  because the noise scaled by the step's increment is promoted to float64
  ([#68](https://github.com/TARPS-group/EnsKit/issues/68)). Float32 filters
  work.
- `check_simulator` can reject a valid NumPy simulator at an odd number of
  particles, reporting it order-dependent across rows
  ([#63](https://github.com/TARPS-group/EnsKit/issues/63)), and cannot check
  a float32 simulator
  ([#64](https://github.com/TARPS-group/EnsKit/issues/64)).
- The message of `UnsupportedOpError` suggests `densify`, which returns `nan`
  for a singular PSD operator such as a thin `PSDLowRank`
  ([#7](https://github.com/TARPS-group/EnsKit/issues/7)).
- Conditioning a thin factor at a very large ratio of signal to noise loses
  accuracy in the conditional variances of the constrained directions, about
  $\varepsilon\,\sigma_{\max}$ relative
  ([#54](https://github.com/TARPS-group/EnsKit/issues/54)).
- `LowRankUpdate.solve` loses accuracy when the low-rank term dominates the
  base ([#51](https://github.com/TARPS-group/EnsKit/issues/51)); its `logdet`
  and `whiten` are unaffected.
- Repeated conditioning without `compress` nests operators, and an operator's
  `shape` then costs time exponential in the depth
  ([#76](https://github.com/TARPS-group/EnsKit/issues/76)). The ensemble
  drivers never nest, and `LinearStateSpace.exact_filter` compresses.
- Domain localization holds every distance and every local factor at once,
  with no size guard ([#69](https://github.com/TARPS-group/EnsKit/issues/69)).
