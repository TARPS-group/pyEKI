# Kalman contract

This page specifies `enskit.kalman`: ensemble Kalman updates, the protocol a
new update rule implements, the two shipped rules, inflation and relaxation,
and the conformance checks `enskit.testing` provides for code written against
the layer. It is normative: an implementation that violates a rule here is
defective even if its tests pass. It is written for contributors implementing
or reviewing the layer, and for users who want a more precise account of an
update than the user guide gives.

*Must* and *never* state requirements, *should* states a strong default that a
documented reason may override, and *may* states a permission. The layer is
built on the distribution layer and refers to {doc}`distribution-contract`
rather than restating it; the conditioning mathematics, the whitening checks
and the validation tiers are that page's, and this one says only how the
updates assemble them.

:::{admonition} Status: specified, then implemented
:class: note

PR 6 of the redesign plan wrote this page first and then implemented it in
`enskit.kalman` and `enskit.testing`, as PR 3 and PR 4 did for the
distribution layer. Where implementing it showed the page wrong or
incomplete, the page was corrected in the same pull request; each change is
listed in {ref}`kalman-implementation-changes`. Where this page departs from
the design's stubs (`docs/redesign/stubs/kalman.py`), it says so in
{ref}`kalman-departures`, and this page wins.

PR 9 added localization (`LocalizedUpdateRule`, `DomainLocalization`,
`gaspari_cohn`) the same way: {ref}`kalman-localization` was written first,
then implemented, and its departures and changes are listed with the rest.
:::

(kalman-scope)=
## Scope

An **ensemble Kalman update** takes the particles of an ensemble over *target*
and *given* blocks and a value $y^*$ for the given blocks, and returns
particles approximating the conditional distribution of the targets at
$y^*$. The layer provides:

- **the update**, in three stages (approximate, build, call) and in one call
  (`update`);
- **the default approximation**, the moment-matching Gaussian with known
  noise added as a covariance (`gaussian_approximation`);
- **two protocols**: `UpdateRule`, which a new update implements, and
  `ParticleUpdate`, what a rule builds;
- **two rules**: the deterministic symmetric square-root rule
  (`SymmetricSquareRoot`) and the stochastic rule (`Matheron`);
- **domain localization** around either rule (`LocalizedUpdateRule`), with
  its geometry (`DomainLocalization`) and the Gaspari–Cohn taper
  (`gaspari_cohn`);
- **four functions on ensembles** that the algorithms apply around an update:
  multiplicative and additive inflation, and relaxation to the prior spread
  or to the prior perturbations.

What the layer is not:

- It knows nothing about where the given values came from: no observations,
  forecasts, steps, schedules or time. An algorithm that tempers its noise
  passes the scaled covariance; the layer sees only a covariance.
- It never imports `enskit.maps`. Pushing particles through a simulator
  happens before the update, in user code or an algorithm.
- It never repairs failed particles ({ref}`kalman-failed`) and never
  resamples.
- It computes no conditioning of its own, with one exception. Every gain,
  transform and whitening of the two rules is a call into
  `enskit.distribution`, so the numerical rules of {doc}`distribution-contract`
  (one thin SVD of the whitened factor, centering before whitening, the
  identity completion) hold here by construction. Localization, whose local
  problems the distribution layer cannot express, whitens the aligned
  approximation's factor rows itself and builds one `IdentityPlusGram` per
  local problem, under the same rules and with the same whitening and result
  checks ({ref}`kalman-localized-rule`).

(kalman-notation)=
## Notation

The symbols of {ref}`dist-notation` carry over: $J$ particles, given blocks
$c$ with total dimension $N$, target blocks $x$, factor rows $F_b$,
independent terms $D_b$, the whitener $W$ of the given blocks' terms, the
whitened factor $S = (WF_c)^\top$, $A = I_k + SS^\top$, and the unit roundoff
$\varepsilon$ of the particles' dtype. In addition:

| symbol | meaning |
| ------ | ------- |
| $x_j$, $g_j$ | particle $j$'s values of a target block and of the given blocks |
| $\bar x$, $a_j = x_j - \bar x$ | a block's sample mean and anomalies ({ref}`dist-ensemble`) |
| $R_c$ | the known noise covariance of given block $c$, the independent term the approximation carries for it |
| $K$ | the gain $\operatorname{cov}(x, y)\operatorname{cov}(y, y)^{-1}$ of the approximation, never formed |
| $L_b$ | `D_b.factor()`, of width $w_b$, for a target block's independent term |
| $\lambda$ | `anomaly_scale` of multiplicative inflation |
| $\alpha$ | the weight of a relaxation |

The approximation is a `Gaussian` over the same blocks as the particles; its
*given* blocks are the ones named when the update is built, and every other
block is a *target*.

(kalman-objects)=
## The objects

| object | is |
| ------ | -- |
| `update` | one ensemble Kalman update, the three stages in one call |
| `gaussian_approximation` | the default joint Gaussian an update conditions |
| `UpdateRule` | protocol: `build(particles, approximation, given)` returns a `ParticleUpdate` |
| `ParticleUpdate` | protocol: called with the given values, returns the updated particles |
| `SymmetricSquareRoot` | the deterministic rule: realize after condition |
| `Matheron` | the stochastic rule: transport the particles through the conditional map |
| `LocalizedUpdateRule` | domain localization around either rule |
| `DomainLocalization` | locations, taper, radius and neighborhood size |
| `gaspari_cohn` | the compactly supported fifth-order taper |
| `inflate_multiplicative` | scale anomalies by $\lambda$ |
| `inflate_additive` | add centered Gaussian draws |
| `relax_to_prior_spread` | RTPS: relax the per-coordinate spread toward the prior's |
| `relax_to_prior_perturbations` | RTPP: blend posterior and prior anomalies |

and, in `enskit.testing`, `check_update_rule` and `check_conditional_map`
({ref}`kalman-testing`).

(kalman-stages)=
## The update in three stages

```python
approx = kalman.gaussian_approximation(ens, {"g": R})   # 1. approximate
built = kalman.Matheron().build(ens, approx, ("g",))     # 2. build, without values
post = built(g=y_star, key=key)                          # 3. call with the values

post = kalman.update(ens, g=y_star, noise={"g": R},
                     update_rule=kalman.Matheron(), key=key)  # the same, in one call
```

1. **Approximate.** A joint Gaussian is fitted to the particles. The default
   is `gaussian_approximation`; the `approximation=` hook of `update` carries
   any other (a hybrid covariance, a shrinkage estimator).
2. **Build.** An update rule builds a particle update from the particles, the
   approximation and the *names* of the given blocks. Everything that does not
   depend on the given values (the whitened factor, its decomposition, the
   square-root transform, the target draws' factors) is computed here, once.
3. **Call.** The particle update is called with the given *values* and
   returns the updated particles. Calling it again with other values reuses
   the build.

`update` runs the three stages and returns exactly what the third returns:
`update(ens, values, update_rule=r, noise=n, approximation=f, key=k)` is
`r.build(ens, f(ens, n), names)(values, key=k)`, with `names` the given
blocks in the ensemble's block order.

(kalman-protocols)=
## The protocols

(kalman-update-rule)=
### `UpdateRule`

```python
class UpdateRule(Protocol):
    def build(self, particles: Ensemble, approximation: Gaussian,
              given: tuple[str, ...]) -> ParticleUpdate: ...
```

Writing a class with this one method is how a new update is added; it is then
accepted everywhere a shipped rule is, `update` and both drivers included.
The contract between the caller and a rule:

- `particles` is an unweighted `Ensemble` containing every block of the
  approximation.
- `approximation` is a `Gaussian` over the same blocks as `particles`, of the
  same dimensions and dtype. Its given blocks carry their known noise as
  independent terms.
- `given` names the given blocks; every other block of the approximation is a
  target.
- **If** the approximation is an `EnsembleGaussian` with the particles'
  count, it is *aligned* with these particles ({ref}`kalman-alignment`), and a
  rule may rely on that for a faster path. `gaussian_approximation`
  guarantees it. A modified approximation is generally a plain `Gaussian`, and
  a rule that needs alignment refuses it.

A rule may refuse an approximation it does not support, at build, with
`ValueError`. The shipped rules accept `given` as a `str` or a sequence of
`str`, and normalize it to the approximation's block order.

(kalman-particle-update)=
### `ParticleUpdate`

```python
class ParticleUpdate(Protocol):
    def __call__(self, values=None, /, *, key=None, **block_values) -> Ensemble: ...
```

The obligations of a built update, which `check_update_rule` checks
({ref}`kalman-testing`):

1. **Values.** The values follow {ref}`dist-block-arguments`: a
   positional-only mapping, keywords, or both, with exactly one `(d_c,)` array
   for each given block. A missing, extra or misshapen value raises.
2. **Result.** An unweighted `Ensemble` with the particles' count, over the
   target blocks in the approximation's block order, of the particles' dtype.
3. **The build is value-free.** One built update called with several values
   gives what a fresh build gives for each.
4. **Keys.** A key is consumed whole, and the result is deterministic given
   it. An update that needs a key and receives none raises `ValueError`.
5. **Tracing.** Building and calling run under `jax.jit` and `jax.vmap` with
   no data-dependent shapes.
6. **Exactness on a linear-Gaussian problem.** With particles whose sample
   moments equal a linear-Gaussian joint's ({ref}`dist-exact-moments`), the
   updated particles' sample mean equals the exact conditional mean: exactly
   for a deterministic update, in expectation over the key for a stochastic
   one. The sample covariance (divisor $J - 1$) likewise, for a rule that
   claims an exact covariance; a rule such as the deterministic EnKF of Sakov
   & Oke (2008) does not.

A built update is bound to the particles, approximation and given names it
was built from. Calling it is the only operation it promises.

(kalman-alignment)=
## Alignment

An `EnsembleGaussian` is **aligned with particles** when it has their count
and its realized particles without independent terms are those particles:
for every block $b$,

$$
m_b + \sqrt{\delta}\,F_b e_j = x^{(b)}_j \qquad (j = 1, \dots, J),
$$

with $\delta$ the approximation's stored divisor (`divisor`,
{ref}`dist-divisor`), $e_j$ the $j$-th unit vector, and a block without a factor row realizing
as its mean (`realize_particles(exclude_block_covs=...)` with every block that
has a term excluded; {ref}`dist-realize`). `Ensemble.project()` returns such a
Gaussian to round-off, and `add_noise` on blocks without terms keeps it one.
So `gaussian_approximation` is aligned with the particles it was given.

Both shipped rules rely on alignment: `SymmetricSquareRoot` returns the
approximation's own conditioned particles, and `Matheron`'s fast path reads
the given blocks' residuals off the approximation's whitened factor instead
of the particles. An `EnsembleGaussian` of the right count built from other
particles, or from these particles with a modified factor, gives silently
wrong numbers on either path. So alignment is a value precondition of both
rules, checked at build in debug mode (tier 4) by the **alignment check**:
for every block $b$, with $\tilde x$ the realized particles, $a$ the
particles' anomalies in that block, and $M$ the largest magnitude of any
particle in *any* block,

$$
\max_{j,i}\big|\tilde x^{(b)}_{ji} - x^{(b)}_{ji}\big|
  \le \sqrt{\varepsilon}\,\max_{j,i}\big|a^{(b)}_{ji}\big| + 64\,J\,\varepsilon\,M .
$$

The second term bounds the round-off of `project` followed by realization,
and of an aligned block built by a linear map of another, whose round-off
scales with the magnitude of the block it came from rather than its own (a
differencing map of float32 particles at an offset of $10^5$ realizes to
within $6 \times 10^{-3}$, above a bound built from the derived block alone).
The first allows other exact constructions. Any particle set that differs at
the scale of its spread is still caught, except in float32 next to a block of
much larger magnitude, where the check is loose by design: it exists to catch
gross misuse, never to fail a correct approximation. On failure the check raises `ValueError` naming the rule, the block
and the remedy: build the approximation with `gaussian_approximation` from
these particles, or pass a modified approximation as a plain `Gaussian`. A
modification meant to inflate the spread is `inflate_multiplicative` on the
particles before the update.

(kalman-gaussian-approximation)=
## `gaussian_approximation(ensemble, noise=None)`

The joint Gaussian an update conditions:

$$
\hat p(x, y) = \mathcal N\big(\bar z,\ \hat C + \operatorname{blockdiag}(0, R)\big),
$$

the Gaussian with the particles' sample mean $\bar z$ and sample covariance
$\hat C$ jointly over every block, with each known noise covariance $R_c$ =
`noise[c]` added to its block *as a covariance*, never as samples. It is
`ensemble.project().add_noise(noise)`, or `ensemble.project()` when `noise`
is `None` or empty, and so always uses the unbiased divisor
({ref}`dist-divisor`); it takes no `unbiased` argument. An approximation
projected with the empirical divisor is passed through the `approximation=`
hook, and every shipped rule honors its divisor.

- `noise` is a mapping from block name to `PSDLinOp` (`TypeError` otherwise),
  each of its block's side (`ValueError`), naming blocks of the ensemble
  (`KeyError`). It is used only through what later operations ask of it,
  which for conditioning is `whiten`.
- For an unweighted ensemble the result is an `EnsembleGaussian` aligned with
  the particles. A weighted ensemble projects to a plain `Gaussian`
  ({ref}`dist-project`); the result is valid, for densities for instance, but
  no shipped rule accepts it.
- The finiteness and concentrated-weight checks are `project`'s.

Adding the noise as a covariance rather than as sampled perturbations is the
lower-variance estimate of the same joint, and it is the default of every
`approximation=` hook, so the noise reaches the update through `whiten` alone.

(kalman-update)=
## `update(ensemble, given=None, /, *, update_rule, noise=None, approximation=None, key=None, **given_values)`

One ensemble Kalman update. With `noise`, each given value $y^*_c$ is
understood as a realization of $x_c + e_c$, $e_c \sim \mathcal N(0, R_c)$,
while `ensemble[c]` holds the noise-free $x_c$; without it, the given blocks
are conditioned on exactly.

- `ensemble`: an unweighted `Ensemble` holding every given block and at least
  one other block. A weighted one raises `ValueError` saying to resample
  first, with `enskit.distribution.resample`.
- The given values follow {ref}`dist-block-arguments` (`given` positional
  only, or keywords), at least one, each naming a block of `ensemble`. A
  block whose name is one of this function's parameters must go in `given`.
  The given names are taken in the ensemble's block order.
- `update_rule`: keyword-only and required; any object with a `build`
  method (`TypeError` otherwise). There is no default: which rule to use is
  a modeling choice ({ref}`kalman-square-root`, *Notes*).
- `noise`: keyword-only, a mapping from given block to `PSDLinOp`
  (`TypeError` otherwise, whatever the approximation). A name
  that is not a block raises `KeyError`; a block that is not given raises
  `ValueError`. Noise on a target is not the known error of a given value, and adding
  it is `add_noise` on a custom approximation.
- `approximation`: keyword-only, a callable `(ensemble, noise) -> Gaussian`,
  with `noise` always a `dict` (empty when none was given);
  `gaussian_approximation` by default. A result that is not a `Gaussian`
  raises `TypeError`.
- `key`: keyword-only, passed whole to the built update.

The result is checked before it is returned, statically and always: it must
be an unweighted `Ensemble` with the ensemble's particle count, over exactly
the target blocks in the approximation's order, each of the ensemble's
dimension and dtype. A result that is not raises (`TypeError` for
the type or dtype, `ValueError` otherwise) naming the rule. A rule whose
result changes the dtype would otherwise demote every later update of an
algorithm silently.

In debug mode `update` checks, before calling the approximation, that every
particle is finite ({ref}`kalman-failed`).

(kalman-square-root)=
## `SymmetricSquareRoot`

The deterministic rule: realize after condition. For noisy given blocks it
builds `approximation.square_root_map(given)` ({ref}`dist-square-root-map`)
and returns that map as the particle update. Called with $y^*$ it returns,
for each target block $x$,

$$
x_j' = m_x + F_x w + \sqrt{\delta}\,F_x T e_j \;\big[+\, L_x \eta^{(x)}_j\big],
\qquad w = A^{-1} S\, W(y^* - m_c), \qquad T = A^{-1/2},
$$

with $\delta$ the approximation's divisor and the bracketed draw present
for a target with an independent term. $T$
and the realized conditional anomalies do not depend on $y^*$ and are
computed at build. For particles whose moments equal a linear-Gaussian
joint's and targets without independent terms, the output's sample mean and
covariance equal the exact conditional's (Bishop et al., 2001; Hunt et al.,
2007). $T$ is symmetric, which is what preserves the mean (Wang et al., 2004);
"symmetric" in the name refers to that choice of square root.

**Exact values.** When no given block has an independent term (`update`
without `noise`), the build returns a particle update that, called with
$y^*$, returns `approximation.condition(values).realize_particles(key=key)`
({ref}`dist-condition`). Its factorization depends on nothing but the build,
but the distribution layer offers it only through `condition`, so this path
factorizes at every call. It needs $N \le J - 1$, which the build checks
(`ValueError`), and a given block's factor rows of full row rank, which
`condition` checks in debug mode.

**Build** checks, in order: the arguments of {ref}`kalman-validation`; that
the approximation is an `EnsembleGaussian` with the particles' count
(`ValueError` naming `gaussian_approximation`); that the given blocks are all
noisy or all exact (`ValueError`); the exact size condition; capabilities
(`whiten` of each given term, `factor` of each target term); then, in debug
mode, the finiteness of the particles and the alignment check.

**Call**: the key is required exactly when a target block has an independent
term, which is then sampled (`ValueError` when missing); otherwise it is
ignored.

**Notes.** Exact moments do not mean the right shape. On a nonlinear problem
the square-root update keeps the particles' arrangement, rotated and scaled,
and can leave a few particles carrying most of the spread; the stochastic
rule redraws a perturbation for every particle. That is why `update` has no
default rule.

(kalman-matheron)=
## `Matheron`

The stochastic rule: transport the particles through the approximation's
conditional map, with perturbed values (Burgers et al., 1998; Houtekamer &
Mitchell, 1998), which is Matheron's rule applied to particles (Wilson et
al., 2021). Called with $y^*$, it moves each particle by

$$
x_j' = x_j + e_{x,j} + K\big(y^* - g_j - e_j\big) = x_j + e_{x,j} + F_x w_j,
\qquad w_j = A^{-1} S\, W\big(y^* - g_j - e_j\big),
$$

for every target block $x$, where $e_j \sim \mathcal N(0, R)$ is the given
blocks' noise, drawn in whitened coordinates, $W e_j = \varepsilon_j$ with
$\varepsilon_j$ standard normal, so the noise covariances need only `whiten`;
and $e_{x,j} = L_x \eta^{(x)}_j$ is a draw from the target's independent term
when it has one, and zero otherwise. A target without a factor row is not
moved by the gain. A target's independent term is variation the particles do
not carry and the given blocks do not see, so it never enters the gain; a
covariance meant to enter the gain (the static part of a hybrid covariance)
belongs in the shared factor, which `absorb` puts it in. The draw $e_{x,j}$
is the rule's: `MatheronMap` passes the
targets' independent terms through unsampled ({ref}`dist-matheron`), and
without it the updated spread omits $D_x$ entirely. (The design review's
prototype did that, and reported a variance of 0.044 where the conditional
has 4.04.)

It works with any approximation, by one of two paths chosen at build from the
approximation's type, never from values:

- **aligned** (an `EnsembleGaussian` with the particles' count): the
  coefficients come from `MatheronMap.particle_coefficients`, which reads the
  whitened residuals off the approximation's own factor,
  $W(y^* - g_j) = W(y^* - m_c) - \sqrt{\delta}\,S_{j\cdot}^\top$, and the rule
  adds $F_x w_j$, with $F_x$ = `approximation.factor(x)`, to its own
  particles. The build whitens $k = J$ vectors per given block and the call
  one: $J + 1$ in all.
- **general** (any other `Gaussian`, a hybrid covariance for instance): the
  conditional map is called on the particles, with the key, so that it draws
  the noise and whitens each particle's residual. The build whitens $k$
  vectors per given block and the call $J$: $2J$ when $k = J$.

The two paths agree to round-off, not bit-exactly, for the same key on an
aligned approximation. Particles whose given anomalies are exactly zero come
back from the aligned path bit-identical (plus any target draw), since the
coefficients are exact zeros and are added to the particles themselves.

For a linear-Gaussian problem and an approximation with divisor $J - 1$, as
`gaussian_approximation` builds, the updated particles' sample mean and
covariance (divisor $J - 1$) are unbiased, over the key, for the conditional
moments of the *fitted* Gaussian; their sampling error is of order
$1/\sqrt{J}$. Given the particles, the sample mean's variance over the key is
$K R K^\top / J$ plus $D_x / J$ for a target with a term.

**Build** checks, in order: the arguments of {ref}`kalman-validation`; that
every given block has an independent term (`ValueError`: the perturbations
are draws of that noise; for exact values use `SymmetricSquareRoot`);
capabilities (`whiten` of each given term, `factor` of each target term);
then, in debug mode, the finiteness of the particles and, on the aligned
path, the alignment check.

**Call**: the key is required (`ValueError` when missing). It is split as in
{ref}`kalman-prng`.

(kalman-localization)=
## Localization

With $J$ particles, every update above moves each target block within the
span of its own anomalies, a space of dimension at most $J - 1$, and its gain
is built from sample covariances whose entries between unrelated coordinates
are noise of order $1/\sqrt{J}$. When the targets and the given blocks are
spatially extended and $J$ is much smaller than their dimensions, both limit
the update. **Domain localization** (Ott et al., 2004; Hunt et al., 2007)
replaces the one global update by one local update per target coordinate,
which sees only the given coordinates near it, each with its noise inflated
by the reciprocal of a taper of its distance. Each local update has its own
weights in $\mathbb R^J$, so the corrections taken together are not confined
to the anomalies' span.

The layer implements domain localization, not covariance localization
(Houtekamer & Mitchell, 2001; Hamill et al., 2001), which tapers the sample
covariance entrywise: the entrywise product destroys the factor's low rank,
and with it the whitened factor and its one thin SVD that every update here
is built on. The two are closely related (Sakov & Bertino, 2011).

(kalman-domain-localization)=
### `DomainLocalization(target_coords, given_coords, *, radius, max_neighbors, taper=None, distance=None)`

The geometry of a localized update, which the caller supplies: the layer
never constructs locations or distances. Notation for this section:

| symbol | meaning |
| ------ | ------- |
| $c_p \in \mathbb R^q$ | the location of coordinate $p$ of a located target block, row $p$ of `target_coords[x]` |
| $c_i \in \mathbb R^q$ | the location of given coordinate $i$, $i = 1, \dots, N$, over the given blocks concatenated |
| $d(c_p, c_i)$ | the distance, `distance(c_p, points)`; Euclidean by default |
| $L$ | `radius`, the taper's support |
| $\rho$ | `taper`, a function of $r = d/L$ |
| $K$ | `max_neighbors`, the neighborhood size |
| $\mathcal N_p = (i_1, \dots, i_K)$ | the neighborhood of $c_p$: the $K$ given coordinates nearest it |
| $\rho_{pk}$ | the weight of neighbor $k$, $\rho\big(d(c_p, c_{i_k})/L\big)$ |

The neighborhood of $c_p$ is the $K$ given coordinates with the smallest
distances to it, nearest first, ties broken by the lower index in the order of
`given_coords`. Every neighborhood has the same size $K$, so the local updates
vectorize with no data-dependent shapes; a neighbor beyond the radius has
$\rho_{pk} = 0$ and no influence, and $\rho_{pk} > 0$ is the validity mask. A
given coordinate within the radius but outside the $K$ nearest has no
influence either: $K$ truncates the taper, and $K = N$ never does.

- `target_coords`: a mapping from target block name to a `(d_x, q)` array of
  locations, one row per coordinate, or to `None` for a block without a
  location (a global parameter, say), which gets the global update
  ({ref}`kalman-localized-rule`). At least one block has locations.
- `given_coords`: a mapping from given block name to a `(d_c, q)` array, or
  one bare `(N, q)` array, accepted when exactly one block is given. The bare
  form spares the caller of an algorithm from naming the algorithm's internal
  block. The mapping's order sets the concatenation order, and so the
  tie-breaking.
- Every array is real and exactly 2-D with at least one row, all with the
  same $q \ge 1$. Integer locations are converted to the default floating
  dtype.
- `radius`: a real scalar (a Python number or a 0-d array, which may be
  traced); in debug mode finite and positive.
- `max_neighbors`: an integer (anything `operator.index` accepts, but not a
  `bool`), $1 \le K \le N$.
- `taper`: `None` (`gaspari_cohn`), or a callable from an array of
  $r \ge 0$ to an array of the same shape, of values in $[0, 1]$.
- `distance`: `None` (Euclidean), or a callable `(point (q,), points (m, q))
  -> (m,)`, applied under `jax.vmap` over the points $c_p$. A periodic domain
  passes a periodic distance.

The neighborhoods and weights are computed **at construction**, in the
locations' floating dtype, and stored, so
a build does no geometry and a `DomainLocalization` built once serves every
update of a run. They are readable as `neighbors[x]`, a `(d_x, K)` integer
array of indices into the concatenated given coordinates, and `weights[x]`,
the `(d_x, K)` array of $\rho_{pk}$, for each located block $x$. The radius
enters only the weights, so a localization built inside a traced function is
differentiable in it.

Any taper with values in $[0, 1]$ gives a valid update: a weight only scales
a noise variance, $r_i \mapsto r_i / \rho_{pk}$, and the local problem stays a
Gaussian one. That a taper must be a positive-definite function, which depends
on the dimension $q$ (Gaspari & Cohn, 1999), is a requirement of covariance
localization, where the taper multiplies a covariance; it is not one here. A
weight above 1 would shrink a noise variance below the known one, and is
refused in debug mode.

**Construction** checks, in order: argument types (`TypeError`: a
`target_coords` that is not a mapping, a name that is not a `str`, a `None`
in `given_coords`, a `max_neighbors` that is not an integer or is a `bool`, a
`taper` or `distance` that is not callable, a complex or boolean array); an
empty `given_coords` (`ValueError`); shapes and sizes
(`ValueError`: an array not 2-D or with no rows, differing $q$, no located
block, a `radius` that is not a scalar, $K$ outside
$[1, N]$, a distance or taper result of the wrong shape); then, in debug mode,
that every location and the radius are finite, the radius positive, every
distance finite and nonnegative, and every weight finite and in $[0, 1]$
(`ValueError`).

(kalman-gaspari-cohn)=
### `gaspari_cohn(r)`

The fifth-order piecewise rational function of Gaspari & Cohn (1999,
eq. 4.10), on $r = d/L$, with $z = 2|r|$:

$$
\rho(r) = \begin{cases}
  -\tfrac14 z^5 + \tfrac12 z^4 + \tfrac58 z^3 - \tfrac53 z^2 + 1,
    & 0 \le z \le 1,\\[2pt]
  \tfrac1{12} z^5 - \tfrac12 z^4 + \tfrac58 z^3 + \tfrac53 z^2 - 5z + 4
    - \tfrac{2}{3z}, & 1 < z \le 2,\\[2pt]
  0, & z > 2 .
\end{cases}
$$

It equals 1 at $r = 0$, decreases to exactly 0 at $|r| = 1$ and stays there,
and is a positive-definite correlation function in up to three dimensions.
Elementwise on an array of any shape, keeping its floating dtype (an integer
array is converted to the default floating dtype), with a finite derivative
everywhere (the $1/z$ term is never evaluated at $z \le 1$). The second
branch rounds to slightly negative values near $z = 2$ (about $-10^{-6}$ in
float32), so the result is clamped at zero.

(kalman-localized-rule)=
### `LocalizedUpdateRule(update_rule, localization)`

An `UpdateRule` performing domain localization around `update_rule`, a
`SymmetricSquareRoot` or a `Matheron` (`TypeError` otherwise), with the
geometry `localization`, a `DomainLocalization` (`TypeError` otherwise).

**The local problem.** For coordinate $p$ of a located target block $x$, the
local problem is the update of that one coordinate given the coordinates
$\mathcal N_p$ of the given blocks, with noise variance $r_{i_k}/\rho_{pk}$ in
place of $r_{i_k}$, a neighbor with $\rho_{pk} = 0$ being absent. **The
localized update of coordinate $p$ is the wrapped rule's update on its local
problem**, for the stochastic rule with one noise draw shared by every local
problem. That is the normative definition; the formulas below compute it.

With $S = (WF_c)^\top \in \mathbb R^{J \times N}$ the whitened factor of
the given blocks ({ref}`dist-notation`), $f_p \in \mathbb R^J$ row $p$ of
$x$'s factor row $F_x$, and $m_{x,p}$ entry $p$ of its mean, the local
whitened factor and its operator are

$$
S_p = \Big[\sqrt{\rho_{p1}}\, S_{\cdot i_1}\ \cdots\ \sqrt{\rho_{pK}}\, S_{\cdot i_K}\Big]
  \in \mathbb R^{J \times K},
\qquad A_p = I_J + S_p S_p^\top ,
$$

since for row-local noise the local whitener is $W$ restricted to
$\mathcal N_p$ and multiplied by $\sqrt{\rho_{pk}}$. Then, for particle $j$:

- around **`SymmetricSquareRoot`**,

  $$
  x'_{jp} = m_{x,p} + f_p^\top A_p^{-1} S_p\, b_p
    + \sqrt{\delta}\,\big(A_p^{-1/2} f_p\big)_j ,
  \qquad (b_p)_k = \sqrt{\rho_{pk}}\,\big(W(y^* - m_c)\big)_{i_k} ;
  $$

- around **`Matheron`**, on the aligned path of {ref}`kalman-matheron`,

  $$
  x'_{jp} = x_{jp} + f_p^\top A_p^{-1} S_p\, b_{pj},
  \qquad (b_{pj})_k = \sqrt{\rho_{pk}}\,\big(W(y^* - g_j)\big)_{i_k} - \varepsilon_{j i_k},
  $$

  with $W(y^* - g_j) = W(y^* - m_c) - \sqrt{\delta}\,S_{j\cdot}^\top$,
  $\delta$ the approximation's divisor, and
  $\varepsilon$ the one `normal(k_noise, (J, N))` draw of `Matheron`
  ({ref}`kalman-prng`). In the local problem's whitened coordinates
  $\varepsilon_{ji_k}$ is a draw of noise of variance $r_{i_k}/\rho_{pk}$, so
  the local spread is the local problem's.

A target block named in `target_coords` with `None` gets the wrapped rule's
**global** update, with $S$ itself, every given coordinate at weight 1, and
for `Matheron` the same $\varepsilon$. (Every target has a factor row: an
aligned approximation refuses a block with neither a row nor a term, and a
target with a term is refused.) Neither $A_p$ nor anything else here depends on
$y^*$, so the build computes, for each located coordinate, the **local gain
row** $\kappa_p = S_p^\top A_p^{-1} f_p \in \mathbb R^K$ and, around
`SymmetricSquareRoot`, the **local anomalies** $\sqrt{\delta}\,A_p^{-1/2} f_p$;
a call only gathers and contracts.

Two consequences, both tested:

- **No localization, no change.** With $K = N$ and every weight 1, every
  $A_p$ is $A$ with the columns of $S$ permuted, so the localized update
  equals the wrapped rule's global update to round-off, for the same key.
  This is the check that catches a mis-indexed neighborhood.
- A zero weight gives a zero column of $S_p$, and so an exactly zero singular
  value; `IdentityPlusGram`'s derivative rules keep first derivatives finite
  there. The square root of a weight is taken with a zero derivative at zero,
  not an infinite one, so **the derivative with respect to a weight that is
  exactly zero is zero by convention**. For a taper that reaches zero with
  zero slope, `gaspari_cohn` among them, a derivative in the radius is
  therefore exact; one that reaches zero with nonzero slope (a linear ramp)
  loses that one-sided term. The Euclidean distance has a zero derivative,
  not `nan`, where two locations coincide.

**Restrictions**, each refused at build:

- **Row-local noise.** Every given block has an independent term that is a
  `PSDDiagonal` or an `Identity`, possibly wrapped in `PSDScaled` any number
  of times (as an algorithm's tempered `R * (1 / delta)` is); `TypeError`
  otherwise, and `ValueError` for a given block with no term. A local problem
  needs the noise covariance restricted to a neighborhood, and a principal
  submatrix of a correlated block is not an operator-layer operation; whitening
  the whole block and then selecting rows would condition on the wrong
  problem silently.
- **The particles' dtype.** Each noise whitens to the particles' dtype
  (`TypeError`). Operators carry no dtype, and a scalar scale of a float32
  operator promotes it to float64 (issue #68), which
  would otherwise give a float64 result silently.
- **Alignment.** The approximation is an `EnsembleGaussian` with the
  particles' count (`ValueError`), and aligned with them (the alignment check,
  in debug mode): both local formulas read the whitened residuals off $S$.
- **No target terms.** No target block has an independent term
  (`ValueError`).
- **The localization fits the problem.** The given blocks are those of
  `given_coords`, as a set, or exactly one with the bare form (`ValueError`);
  every name in `target_coords` is a block (`KeyError`) and a target
  (`ValueError`); **every target block is named in `target_coords`**, with
  `None` for a global update (`ValueError`: a forgotten block would otherwise
  get the global update silently); and each array has its block's dimension
  in rows (`ValueError`).

**Build** checks, in order: the arguments of {ref}`kalman-validation`; the
approximation's type; the localization against the problem, in the order
listed; the targets' terms; the given blocks' terms (present and row-local);
then, in debug mode, the finiteness of the particles and the alignment check.
The work follows: the dtype check and the whitening check on $S$ right after
it is computed.

**Call**: the values as for {ref}`kalman-particle-update`; around `Matheron`
the key is required (`ValueError`) and split as `Matheron` splits it; around
`SymmetricSquareRoot` a key is type-checked and otherwise ignored. In debug
mode the call checks that the values are finite, runs the whitening check on
$W(y^* - m_c)$, and checks every updated block (the result check).

**Cost.** The build whitens $J$ vectors per given block, the factor rows
once, shared by every local problem; a call whitens one. There is one SVD of
a $J \times K$ array per located coordinate with a factor row, computed at
build under `jax.vmap`, so one batched `svd` per located block, and one SVD of
the $J \times N$ array $S$ if a target named with `None` has a factor row. A
call computes no SVD. The build holds every local factor of a block at once,
$d_x J K$ numbers; a call around `Matheron` gathers $J d_x K$. Constructing a
`DomainLocalization` computes every distance from each located block's
coordinates to the given ones, a $d_x \times N$ array, and so is
$O(d_x N q)$ in time and memory, with no size guard yet (issue #69).

(kalman-failed)=
## Failed and weighted particles

Updates take **unweighted, finite** particles. A failed particle (a row with
a non-finite entry, {ref}`dist-failed-particles`) must be repaired or dropped
before an update:

- The distribution layer lets a failed particle be dropped by giving it weight
  zero. An update cannot take that ensemble, because updates refuse weighted
  ensembles: their projection is a plain `Gaussian`, not aligned with the
  particles, and a weighted square-root reading is a different estimator.
  Dropping failed particles before an update is therefore
  `resample(key, reweight(ens, jnp.where(ens.all_finite, 0.0, -jnp.inf)))`,
  which returns an unweighted ensemble of finite particles, or the
  algorithms' repair, which replaces the failed rows.
- So the conditional maps' tier-4 check, which covers every row
  ({ref}`dist-failed-particles`), needs no change for this layer: no
  ensemble with a zero-weight failed particle reaches it from an update.
- In debug mode `update` and both rules' builds check every particle for
  finiteness, before any approximation is computed, and raise `ValueError`
  naming the call, the block, the number of failed particles and both
  remedies. Outside debug mode a failed particle makes the projection's mean
  `nan`, and so every updated particle; nothing raises.

A weighted ensemble raises `ValueError` from `update` and from both rules'
builds, always (it is static): resample first.

(kalman-inflation)=
## Inflation and relaxation

Four functions on ensembles, applied around an update by the algorithms or by
hand. Each returns a new `Ensemble` with the named blocks replaced, every
other block unchanged and in its position, and the weights kept. `names` is a
block name or a sequence of them (`KeyError` for one that is not a block,
`ValueError` for one repeated), every block by default. In debug mode each
checks that every particle of positive weight in the named blocks is finite
(the remedies are those of {ref}`kalman-failed`): with one, the mean is `nan`
and so is every particle.

(kalman-inflate-multiplicative)=
### `inflate_multiplicative(ensemble, anomaly_scale, names=None)`

$$
x_j \mapsto \bar x + \lambda\,(x_j - \bar x), \qquad
\hat C \mapsto \lambda^2 \hat C ,
$$

with $\lambda$ = `anomaly_scale`, the factor on the *anomalies* (Anderson &
Anderson, 1999). The covariance grows by $\lambda^2$: an intended variance
inflation of 1.2 is `anomaly_scale=math.sqrt(1.2)`, and the argument is named
for what it multiplies so that the two conventions cannot be confused. The
mean, weighted when the ensemble is, is preserved, and computed with
`Ensemble.mean` and `Ensemble.anomalies`, so identical particles stay
identical at any magnitude.

`anomaly_scale` is a real scalar: a Python number or a 0-d array, which may be
traced. Any other shape raises `ValueError`: a `(d,)` array would broadcast
and inflate each coordinate by a different factor, with nothing else wrong. It
is converted to the ensemble's dtype, so a Python float never promotes a
float32 ensemble. In debug mode it must be finite and positive.

(kalman-inflate-additive)=
### `inflate_additive(key, ensemble, covs=None, /, **block_covs)`

$$
x_j^{(b)} \mapsto x_j^{(b)} + \varepsilon_j^{(b)} - \bar\varepsilon^{(b)}, \qquad
\varepsilon_j^{(b)} = L_b \eta_j^{(b)} \sim \mathcal N(0, Q_b),
\qquad \bar\varepsilon^{(b)} = \sum_j w_j \varepsilon_j^{(b)},
$$

for each block $b$ with $Q_b$ = `covs[b]` and $L_b$ = `Q_b.factor()`, with
the ensemble's weights $w_j$ ($1/J$ when unweighted). The covariances follow
{ref}`dist-block-arguments`: at least one, each a `PSDLinOp` of its block's
side. The (weighted) mean is unchanged to round-off and the covariance grows
by $Q_b$ in expectation, since

$$
\mathbb E\Big[\sum_j w_j (\varepsilon_j - \bar\varepsilon)(\varepsilon_j - \bar\varepsilon)^\top\Big]
  = \Big(1 - \sum_j w_j^2\Big) Q_b
$$

and the covariance divisor is $1 - \sum_j w_j^2$. Unlike an update, which
moves particles within the span of their anomalies, this adds variance in new
directions (Hamill & Whitaker, 2005). Always random, so the key comes first;
each $Q_b$ must support `factor` (`UnsupportedOpError` before any work). The
draw is pinned ({ref}`kalman-prng`).

(kalman-rtps)=
### `relax_to_prior_spread(prior, posterior, alpha, names=None)`

RTPS (Whitaker & Hamill, 2012): relax the posterior spread toward the prior's,
coordinate by coordinate,

$$
a^{\text{post}}_j \mapsto a^{\text{post}}_j\Big(1 + \alpha\,
  \frac{s^{\text{prior}} - s^{\text{post}}}{s^{\text{post}}}\Big),
\qquad x^{\text{post}}_j \mapsto \bar x^{\text{post}} + a^{\text{post}}_j ,
$$

with $a$ the anomalies and $s$ the per-coordinate sample standard deviations,
$s_i = \big(\sum_j a_{ji}^2 / (J-1)\big)^{1/2}$, all elementwise. A
coordinate whose posterior spread is exactly zero has zero anomalies and is
left unchanged (the formula is $0/0$ there), with a finite derivative.

`prior` is the ensemble the update started from and `posterior` the update's
result: both unweighted (`ValueError`: they are an update's input and output),
with the same particle count (`ValueError`) and dtype (`TypeError`), particle
$j$ of one corresponding to particle $j$ of the other. The default `names`
are the posterior's blocks; every named block must be in both, with the same
dimension. The result is `posterior` with the named blocks replaced.
`alpha` is a real scalar (`ValueError` otherwise), in debug mode finite and
in $[0, 1]$.

(kalman-rtpp)=
### `relax_to_prior_perturbations(prior, posterior, alpha, names=None)`

RTPP (Zhang et al., 2004): blend the anomalies particle by particle,

$$
a^{\text{post}}_j \mapsto (1 - \alpha)\, a^{\text{post}}_j + \alpha\, a^{\text{prior}}_j,
\qquad x^{\text{post}}_j \mapsto \bar x^{\text{post}} + a^{\text{post}}_j ,
$$

with the arguments and checks of `relax_to_prior_spread`.

(kalman-cost)=
## Cost

Counted in whitened vectors per given block, as in {ref}`dist-cost`, for
given blocks with factor rows, with $k$ the approximation's latent width
($k = J$ for an aligned one):

| operation | build | call | SVDs |
| --------- | ----- | ---- | ---- |
| `SymmetricSquareRoot`, noisy | $k$ | $1$ | 1, at build |
| `SymmetricSquareRoot`, exact | $0$ | $0$ | 0 (one thin QR per call) |
| `Matheron`, aligned | $J$ | $1$ | 1, at build |
| `Matheron`, general | $k$ | $J$ | 1, at build |
| `LocalizedUpdateRule`, either rule | $J$ | $1$ | one $J \times K$ per located coordinate, and one $J \times N$ if a global target has a factor row, at build |

So one aligned stochastic update costs $J + 1$ whitened vectors and one
square-root update $J + 1$; a general stochastic update on an approximation of
width $k$ costs $k + J$. These are normative and count-tested
({ref}`kalman-conformance`, obligation 7): the general path agrees with the
aligned one to round-off, so only a count catches a regression from one to the
other. `gaussian_approximation` whitens nothing, and the alignment check
whitens nothing.

(kalman-prng)=
## Randomness

The distribution layer's rules hold ({ref}`dist-prng`): typed keys only (a
raw `uint32` key raises `TypeError`), consumed whole, never stored. Always
random functions take the key first (`inflate_additive`); sometimes-random
ones take a keyword-only `key=` (`update`, the particle updates). The draws
are pinned and snapshotted by the tests:

| call | draw |
| ---- | ---- |
| `Matheron`'s update | `k_targets, k_noise = split(key)`, always. With $n_T$ the number of target blocks with an independent term: `keys = split(k_targets, n_T)` and $e_{x} = L_x$ `normal(keys[i], (J, w_x))` for the $i$-th such block in the approximation's block order (no split when $n_T = 0$). The given blocks' noise is the conditional map's draw with `k_noise`: `normal(k_noise, (J, N))`, its columns the given blocks' whitened coordinates in block order, on both paths |
| `LocalizedUpdateRule(Matheron())`'s update | `k_targets, k_noise = split(key)`, as `Matheron`; $\varepsilon$ = `normal(k_noise, (J, N))`, its columns the given blocks' whitened coordinates in block order, shared by every local and global update. `k_targets` is unused, since targets have no terms |
| `LocalizedUpdateRule(SymmetricSquareRoot())`'s update | none |
| `SymmetricSquareRoot`'s update | the key whole to the `SquareRootMap`, or on the exact path to `realize_particles`: the draw of `realize_particles` over the target blocks |
| `update` | the key whole to the built update |
| `inflate_additive` | `keys = split(key, n)`, $n$ the number of named blocks, in the ensemble's block order; $\varepsilon = L_b$ `normal(keys[i], (J, w_b))`; centered by `mean(eps, axis=0, keepdims=True)` when unweighted, `sum(weights[:, None] * eps, axis=0, keepdims=True)` when weighted |

`Matheron` splits into two whatever the targets hold, so adding an independent
term to a target never changes the given blocks' noise. All normal draws have
the particles' dtype.

(kalman-validation)=
## Validation and errors

The four tiers of {ref}`dist-validation` apply: static conditions always,
values only in debug mode on concrete arrays, through the operator layer's
`value_check`, so no check reads a value inside a trace.

**The build arguments**, checked by both shipped rules' `build`, in order:

1. the family guard on `particles` and `approximation` ({ref}`dist-jax`);
2. `particles` is an `Ensemble` and `approximation` a `Gaussian`
   (`TypeError`);
3. `given`: a `str` or a sequence of `str`, at least one, each a block of
   the approximation (`KeyError`), none repeated (`ValueError`);
4. `particles` unweighted (`ValueError`, saying to resample);
5. the approximation's blocks are the particles' blocks, as a set, with the
   same dimensions (`ValueError`) and dtype (`TypeError`);
6. at least one target (`ValueError`);
7. the rule's own structural conditions and capabilities
   ({ref}`kalman-square-root`, {ref}`kalman-matheron`);
8. tier 4: the particles' finiteness, then the alignment check.

`update` checks, in order: the family guard; `ensemble` an `Ensemble`; the
values' names (duplicates `TypeError`, none `ValueError`, unknown
`KeyError`); `update_rule` has `build`; `noise` (a mapping, names known, each
given); `approximation` callable; the key's type; unweighted; a target
remains; tier 4 on the particles; then the approximation, the build and the
call, and the result check of {ref}`kalman-update`.

The inflation functions check: the family guard; argument types; names; the
scalar's shape; the covariances' types and sides; the key; the covariances'
capabilities; the pair's agreement (relaxations); then tier 4.

| condition | raises |
| --------- | ------ |
| an argument of the wrong type (`particles` not an `Ensemble`, a rule without `build`, a covariance not a `PSDLinOp`) | `TypeError` |
| a block name that is not a block | `KeyError` |
| a name repeated, or a value given twice | `ValueError` / `TypeError` |
| a weighted ensemble given to an update, a rule or a relaxation | `ValueError` |
| an approximation over other blocks or dimensions than the particles | `ValueError` |
| an approximation of another dtype | `TypeError` |
| no target block | `ValueError` |
| `Matheron` with a given block without an independent term | `ValueError` |
| `SymmetricSquareRoot` or `LocalizedUpdateRule` on a plain `Gaussian`, or an `EnsembleGaussian` of another count | `ValueError` |
| `LocalizedUpdateRule` wrapping another rule, or given a geometry that is not a `DomainLocalization` | `TypeError` |
| `LocalizedUpdateRule` with a given block's noise not row-local | `TypeError` |
| `LocalizedUpdateRule` with a given block without noise, a target with a term, a target missing from `target_coords`, given blocks other than the localization's, or locations of the wrong size | `ValueError` |
| a `DomainLocalization` argument of the wrong type, shape or size | `TypeError` / `ValueError` ({ref}`kalman-domain-localization`) |
| mixed noisy and exact given blocks | `ValueError` |
| exact values with $N > J - 1$ | `ValueError` |
| `noise` on a block that is not given | `ValueError` |
| a rule's result that is not an unweighted `Ensemble` over the targets, in order, with the particles' count and dimensions | `ValueError` (`TypeError` for its type or a block's dtype) |
| a noise covariance, or an operator of the approximation, of another dtype than the particles, found when the result is assembled | `TypeError` |
| `anomaly_scale` or `alpha` not a real scalar | `ValueError` |
| a prior and posterior of different counts or dimensions | `ValueError` |
| a needed key missing | `ValueError` |
| a raw `uint32` key | `TypeError` |
| a capability missing (`whiten`, `factor`) | `UnsupportedOpError`, from the operator layer |
| any call on a vmapped family | `ValueError` |
| a failed particle, a misaligned approximation, a non-finite or non-positive scale, an `alpha` outside $[0, 1]$, a non-finite location, distance or radius, a taper weight outside $[0, 1]$, a non-finite result | `ValueError` in debug mode; `nan` or a wrong finite result otherwise |

The layer defines no exception types. The whitening and result checks of the
conditional maps ({ref}`dist-accuracy`) run inside every update, so a
singular noise covariance is reported in debug mode by the map, naming the
given block. `Matheron` adds its own result check, on each updated block.

(kalman-derivatives)=
## Differentiability

Every function here is differentiable with respect to every array it reads
(particles, given values, operator parameters inside noise covariances,
`anomaly_scale`, `alpha`), and inherits the distribution layer's guarantees
({ref}`dist-derivatives`): first derivatives of both rules are finite and
correct at every input, including a collapsed ensemble and exactly repeated
or zero singular values of $S$, in forward and reverse mode, under `jit` and
`vmap`; second derivatives are exact only at well-separated spectra. The
stochastic rule and additive inflation are reparameterized: for a fixed key
they are deterministic functions of their inputs, and since `Matheron` draws
the given blocks' noise in whitened coordinates, a noise covariance's
parameters enter through $W$ and are differentiated exactly. The relaxations'
square roots are guarded, so a coordinate of zero spread has a finite
derivative.

(kalman-jax)=
## JAX integration

- `SymmetricSquareRoot` and `Matheron` are frozen pytrees with no fields, so
  they cross `jax.jit` as arguments. They compare by identity and are never
  `static_argnums`.
- A built update is a pytree: the `SquareRootMap` itself, or for `Matheron`
  a private class whose data are the conditional map, the particles, the
  targets' factor rows and the factors of their independent terms, and whose
  static fields are the given and target names, the particle count and which
  path it takes. For the exact square-root path the data are the
  approximation and the static fields the names. Their unflattening bypasses
  any constructor.
- `LocalizedUpdateRule` is a pytree whose children are the wrapped rule and
  the localization. `DomainLocalization` is a pytree whose data are the
  locations, the radius, the neighbors and the weights, and whose static
  fields are the block names, $K$ and the taper's name, so that a fresh
  `lambda` passed as taper or distance does not make `jit` retrace. A localized
  built update is a private class whose data are the whitened factor $S$, the
  given terms and means, the local gain rows, neighbors and anomalies, the
  global operator and factor rows, and (around `Matheron`) the particles.
- Building and calling are `jit`- and `vmap`-safe, with no data-dependent
  shapes. Under `vmap` over the given values one build serves every value.
- A vmapped family of ensembles, distributions or built updates is refused by
  every function, before any other check, with the family message of
  {ref}`dist-jax`.

(kalman-repr)=
## `repr`

```text
Matheron()
SymmetricSquareRoot()
MatheronUpdate(given=('g',), targets=('u',), n_particles=32, aligned=True)
ExactSquareRootUpdate(given=('g',), targets=('u',), n_particles=32)
DomainLocalization(target_dims={'x': 40, 'theta': None}, given_dims={'g': 20}, coord_dim=1, max_neighbors=10, taper=gaspari_cohn)
LocalizedUpdateRule(Matheron(), DomainLocalization(...))
LocalizedUpdate(rule=Matheron(), given=('g',), targets=('x', 'theta'), n_particles=10, located=('x',))
```

With a bare `given_coords`, `given_dims` is the number of rows. The rule's
repr abbreviates its localization; neither shows the radius, which may be an
array.

`SymmetricSquareRoot`'s noisy build returns the `SquareRootMap`, with its own
repr. A vmapped built update wraps its form in `vmapped(..., batch=...)`.

(kalman-surface)=
## Public surface

`enskit.kalman` exports exactly: `update`, `gaussian_approximation`,
`UpdateRule`, `ParticleUpdate`, `SymmetricSquareRoot`, `Matheron`,
`inflate_multiplicative`, `inflate_additive`, `relax_to_prior_spread` and
`relax_to_prior_perturbations`, `LocalizedUpdateRule`, `DomainLocalization`
and `gaspari_cohn`. Its private modules are imported only
from inside `enskit.kalman`; it imports `enskit.distribution` and
`enskit.linalg` and nothing else of the package.

(kalman-testing)=
## Conformance checks: `enskit.testing`

`enskit.testing` holds conformance checks for code written against the
layers. It may import any layer and is imported by none. This PR adds two.
Each raises `AssertionError` with a message naming the obligation that
failed, and returns `None` when every check passes.

**`check_update_rule(rule, *, key=None, exact_covariance=True, diagonal_noise=False)`** checks a
user's `UpdateRule` against {ref}`kalman-particle-update`, on a fixture it
builds itself: a linear-Gaussian joint with two target blocks and a noisy
given block, $J = 12$ exact-moment particles, `gaussian_approximation`, and
the exact conditional from `Gaussian.condition`. It checks, in order:

1. the build returns a callable, and its result is an unweighted `Ensemble`
   over the targets in the approximation's order, with $J$ particles, the
   particles' dtype, finite values; a float32 fixture stays float32; a
   missing value, a value for a block that is not given and a misshapen value
   each raise;
2. whether the update is stochastic: two different keys giving different
   results. A stochastic update must be deterministic given its key, and
   raise `ValueError` when called without one;
3. value-freeness: one build called at two values equals two fresh builds;
4. the sample mean against the exact conditional mean, and the sample
   covariance when `exact_covariance`: to round-off for a deterministic
   update; for a stochastic one, the average over 256 keys within five of its
   estimated standard errors, plus round-off;
5. `jax.jit` of build and call agrees with the eager result, and `jax.vmap`
   over two values agrees with a loop.

`key` seeds the stochastic checks (a fixed key by default, so the check is
reproducible). A rule that claims no exact covariance, such as the
deterministic EnKF, passes `exact_covariance=False`. A rule that accepts only
row-local noise, such as `LocalizedUpdateRule`, passes `diagonal_noise=True`:
the fixture's noise is then the `PSDDiagonal` of its covariance's diagonal,
everything else unchanged. The blocks are `"u"` (dimension 3) and `"v"`
(dimension 2), targets, and `"g"` (dimension 4), given, so a localization for
the fixture can be built; one with $K = N$ and every weight 1 makes a
localized rule exact. The fixture has one given
block and no target with an independent term, so the check says nothing about
several given blocks or a target's draw: a rule's own tests cover those.

**`check_conditional_map(cmap, joint, given, *, tol=1e-8, keyed_noise_free=False)`** checks a
`ConditionalMap` against `joint.condition`, on samples of `joint` whose
moments are exact ({ref}`dist-conditional-map`, obligations 1 to 5):

1. `cmap.given` names the blocks of `given` and `cmap.targets` is disjoint
   from it, all blocks of `joint`;
2. a missing value, a value for a block that is not given, and a misshapen
   value each raise;
3. the result over samples carrying an extra block and log weights has the
   targets replaced, the given blocks dropped, the extra block unchanged and
   in position, the weights kept, and the samples' count; a sample set
   missing a target raises;
4. without a key, permuting the samples permutes the result exactly;
5. the same key twice gives the same result;
6. on unweighted exact-moment joint samples (the given blocks with their
   noise), the result's sample mean and covariance of every target, and
   cross-covariances between targets, match `joint.condition(values)` within
   `tol` relative to their scale. An affine map is exact to round-off there;
   an approximate map passes a looser `tol`;
7. only with `keyed_noise_free`: called with a key on samples of the joint
   *without* the given blocks' noise, the result's sample moments averaged
   over 256 keys match the conditional's within five standard errors plus
   `tol`. That is how `MatheronMap` reads a key, and what `Matheron`'s
   general path relies on; the protocol does not require it of other maps.

(kalman-consumers)=
## How the layers above consume this one

Not normative here, but the layer was shaped against these call sites.

**The EKI driver** (PR 7) evaluates the forward model through `pushforward`,
repairs failed particles, and calls
`update(ens, {prediction: y}, update_rule=rule, noise={prediction: R * (1/delta)},
approximation=approximation, key=k)`. The scaled covariance is the operator
layer's `PSDScaled`, so a traced increment flows through a 0-d field and
nothing refactorizes. Its inflation and relaxation policies wrap the four
functions here and pass their own keys.

**The EnKF driver** (PR 8) calls the same `update` after its forecast, with
the hybrid covariance of Example 11 reaching it through `approximation=`; that
approximation is a plain `Gaussian`, so `Matheron` takes its general path and
`SymmetricSquareRoot` refuses it.

**Localization** is consumed by passing `LocalizedUpdateRule(rule,
localization)` as either driver's `update_rule`; neither driver knows about
it. The EKI driver's tempered noise `R * (1 / delta)` of a `PSDDiagonal` is
row-local. Its given block is internal to the driver, which is what the bare
`given_coords` form is for.

(kalman-conformance)=
## Conformance

Obligations on `tests/test_kalman.py` and `tests/test_testing.py`. The dense
reference is hand-written in the tests from the arrays a problem was built
from, never routed through `enskit.distribution` or `IdentityPlusGram`, and
exactness is checked against closed forms at a few $\varepsilon$ times the
quantity's scale.

1. **The approximation.** `gaussian_approximation` equals `project` plus
   `add_noise` (every mean, covariance and cross-covariance against dense),
   is an `EnsembleGaussian` aligned with the particles, carries the noise as
   an independent term and not as samples, and is a plain `Gaussian` for a
   weighted ensemble.
2. **Square-root exactness.** On exact-moment particles of a linear-Gaussian
   joint, with one and with two target blocks and one and several given
   blocks, `SymmetricSquareRoot`'s sample mean and covariance (and
   cross-covariance between targets) equal the dense conditional's; the same
   for exact values; a target with an independent term needs a key and adds
   its draw.
3. **Matheron against dense.** Each updated particle equals the dense
   $x_j + e_{x,j} + K(y^* - g_j - W^{-1}\varepsilon_j)$, with the draws
   recomputed from the pinned split, on both paths; the aligned and general
   paths agree to round-off for the same key; the sample mean and covariance
   averaged over many keys converge to the dense conditional's within their
   standard errors, a target term included.
4. **The one-call form.** `update` equals the three stages bitwise, passes the
   noise and the key through, calls the approximation hook with a `dict`, and
   returns the targets in order.
5. **Value-free builds.** One build called at several values equals fresh
   builds at each, for both rules and both paths.
6. **Inflation and relaxation.** Each function against its formula, written
   out by hand, on weighted and unweighted ensembles where accepted: the
   multiplicative mean preserved and covariance scaled by $\lambda^2$; the
   additive draw elementwise against its pinned definition, the mean
   preserved, the covariance's growth averaging to $Q$; RTPS and RTPP
   elementwise, RTPS leaving a zero-spread coordinate unchanged with a
   finite derivative.
7. **Counts.** Whitened vectors per update, by instrumented operators: $J + 1$
   for an aligned `Matheron` update, $k + J$ on the general path, $J + 1$ for
   `SymmetricSquareRoot`; one SVD per build and none per call.
8. **Alignment.** The alignment check passes on `gaussian_approximation`
   (including at magnitudes of $10^{8}$ and in float32), and raises in debug
   mode on an `EnsembleGaussian` built from other particles of the same count
   and on one with an inflated factor, for both rules; it never fires under a
   trace.
9. **Failed and weighted particles.** In debug mode `update` and both builds
   raise on a failed particle, naming the block and the remedies; outside
   debug mode the result is `nan` without raising; the documented remedy
   (`reweight` then `resample`) updates cleanly; a weighted ensemble raises
   with the resample message.
10. **Validation.** Every tier-2 and tier-3 rule of {ref}`kalman-validation`
    raises as specified, the build checks' order is pinned with two
    simultaneous violations, and the result check of `update` catches a rule
    that demotes the dtype or drops a target.
11. **Capabilities.** A given term without `whiten`, a target term or an
    additive covariance without `factor` raise `UnsupportedOpError` before
    any work.
12. **Diagnostics.** A singular noise covariance is reported in debug mode
    through every rule, naming the given block, and gives `nan` without
    raising outside it.
13. **Derivatives.** First derivatives of both rules with respect to the
    given values, the particles and a noise scale agree with finite
    differences, are finite at a collapsed ensemble, and run under `jit`.
14. **JAX.** The rules and every built update round-trip through flatten and
    unflatten; build and call run under `jit`; `vmap` over values agrees with
    a loop; a vmapped family is refused.
15. **Dtypes.** Every function keeps a float32 ensemble float32, a Python
    float `anomaly_scale` included.
16. **Reproducibility.** Same key, same output; different keys differ; the
    `Matheron` and `inflate_additive` draws are snapshotted.
17. **`repr`** matches {ref}`kalman-repr` and contains no array data.
18. **The checks check.** `check_update_rule` passes both shipped rules and
    the deterministic EnKF of Example 14 (with `exact_covariance=False`), and
    fails on rules that depend on the values at build, return the targets out
    of order or drop one, demote the dtype, bias the mean (deterministic or
    stochastic), omit the noise draw, or cannot be traced.
    `check_conditional_map` passes `MatheronMap` (with `keyed_noise_free`
    too) and fails on maps that couple samples, drop an extra block, lose the
    weights, accept bad values, condition wrongly, or condition wrongly only
    when given a key.
19. **Examples.** Examples 3 and 14 of the design run against the layer and
    their final checks pass.

The obligations of {ref}`kalman-localization`, on `tests/test_localization.py`,
with the dense reference for a local problem written out in Hunt et al.'s
(2007) form, from the neighborhoods found by sorting distances in NumPy:

20. **`gaspari_cohn`** against its formula, continuous at $z = 1$ and $z = 2$,
    exactly zero beyond, even, with a finite derivative everywhere, keeping
    float32.
21. **The geometry.** Neighbors and weights against a NumPy sort, with ties
    broken by the lower index, a periodic distance, a bare `given_coords`, and
    every construction check of {ref}`kalman-domain-localization`, the
    debug-mode ones included.
22. **Exactness, no localization.** With $K = N$ and weight 1, both localized
    rules equal their wrapped rule's global update to round-off, `Matheron`
    for the same key, with located and global targets, and both pass
    `check_update_rule` with `diagonal_noise=True`.
23. **Exactness, local problems.** Each updated coordinate equals the dense
    update of its local problem: `SymmetricSquareRoot`'s mean and anomalies,
    and `Matheron`'s every particle with the draw recomputed, for a taper
    that zeroes some neighbors, two given blocks whose `given_coords` order
    differs from the approximation's, a scaled diagonal noise, and a global
    target beside a located one.
24. **Hazards.** A global target is updated exactly as the wrapped rule
    updates it, so its pooling over every given coordinate is kept; a target
    left out of `target_coords` raises; a taper that is not positive definite
    (a boxcar) still gives each local problem's exact update; a weight above
    1 raises in debug mode; a correlated noise block raises.
25. **The span.** With $J$ much smaller than the target's dimension, the
    localized increments of the mean leave the span of the prior anomalies,
    and the global ones do not.
26. **Counts, derivatives, JAX, `repr`.** $J + 1$ whitened vectors per
    update; one batched `svd` per located block plus one for global targets,
    none per call; first derivatives finite, at zero weights included, and
    agreeing with finite differences, in the values, the particles and the
    radius; pytree round trips, `jit`, `vmap` over values, a refused family;
    float32 kept; the reprs of {ref}`kalman-repr`.
27. **The divisor** (`tests/test_divisor.py`). With an approximation
    projected with the empirical divisor $J$: `SymmetricSquareRoot` with
    noise $cR$, $c = (J-1)/J$, equals the default approximation's update with
    $R$; `Matheron`'s aligned path equals its general path; both localized
    rules without tapering equal their wrapped rule. Each fails when the
    rule reads $J - 1$ instead of the stored divisor.

### Ported regression tests

The update policies of `enskit.eki` are reimplemented by this layer, so the
regression tests of `tests/test_eki.py` that guard the update and inflation
port here; those that guard the driver port with it, in PR 7.

| `tests/test_eki.py` | becomes, for this layer |
| ------------------- | ----------------------- |
| `test_regression_inflation_scales_the_anomalies_not_the_covariance` | `inflate_multiplicative` multiplies the covariance by `anomaly_scale` squared, and the argument's name pins the convention |
| `test_regression_a_vmapped_inflation_refuses_rather_than_broadcasting` | a non-scalar `anomaly_scale` or `alpha` raises rather than inflating each coordinate by a different factor, and a vmapped ensemble is refused |
| `test_regression_a_float32_update_cannot_quietly_demote_a_run` | `update`'s result check refuses a rule that demotes the dtype, and `check_update_rule` fails it |
| `test_10_multiplicative_inflation_stays_in_the_span_and_additive_leaves_it` | the same two statements for the two inflation functions |

New regression tests join them: the design review's dropped target noise
(Matheron's spread with a target term of variance 4, against a conditional of
4.04); the aligned and general paths agreeing for the same key; a Python
float `anomaly_scale` not promoting a float32 ensemble; RTPS at a coordinate
of zero posterior spread; and the alignment check catching an
`EnsembleGaussian` of the right count built from other particles.

(kalman-departures)=
## Departures from the design

The stubs in `docs/redesign/stubs/kalman.py` are the design; where this page
differs, it governs.

1. **Alignment is a checked precondition.** The stubs let a rule trust any
   `EnsembleGaussian` of the right count. That trust is now defined
   ({ref}`kalman-alignment`) and checked in debug mode, since violating it
   gives wrong numbers silently on both rules.
2. **Failed particles must be repaired or dropped before an update**, and the
   messages say how ({ref}`kalman-failed`). The stubs said only that `update`
   raises on a non-finite particle in debug mode; issue #60's follow-up asked
   this PR to decide between that and maps that skip zero-weight rows, and
   since updates refuse weighted ensembles the second would serve nothing.
3. **`update` checks its rule's result** (type, weights, count, blocks,
   dtype), statically. The stubs did not; the old EKI contract's dtype check
   on its update's output is the precedent.
4. **`noise` names only given blocks** in `update`.
5. **`SymmetricSquareRoot`'s exact path factorizes per call**, since the
   distribution layer offers exact conditioning only through `condition`. The
   stubs said a built update reuses everything that does not depend on the
   values.
6. **The inflation functions accept weighted ensembles** (weighted mean and
   covariance), and the relaxations refuse them. `inflate_additive` takes its
   covariances as a mapping, keywords or both, following
   {ref}`dist-block-arguments`.
7. **RTPS leaves a coordinate of zero posterior spread unchanged**; the
   stub's formula is $0/0$ there.
8. **The scalars are validated**: a non-scalar `anomaly_scale` or `alpha`
   raises, and in debug mode $\lambda > 0$ and $\alpha \in [0, 1]$.
9. **`check_update_rule` takes `exact_covariance`**, since the stub's
   covariance check would reject the deterministic EnKF of the design's own
   Example 14. **`check_conditional_map`** arrives here, as the distribution
   contract says, and takes a `tol`.

Localization's departures, from PR 9:

10. **Every target block is named in `target_coords`**, with `None` for a
    global update. The stubs let an omitted block mean `None`, so a block
    forgotten or misspelled at a call site got the global update with nothing
    raised.
11. **The shared draw is a draw of each local problem's noise.** The
    stochastic local residual is $\sqrt{\rho}\,W(y^* - g_j) - \varepsilon_j$,
    not the design prototype's $\sqrt{\rho}\,\big(W(y^* - g_j) -
    \varepsilon_j\big)$. The prototype's perturbation has the untapered
    variance $r_i$ where the local problem's noise has $r_i/\rho_{pk}$, so
    each coordinate's spread fell short of its local problem's conditional
    by $\kappa_p^\top \operatorname{diag}(1 - \rho_{pk})\,\kappa_p$, with
    the stub's "the wrapped rule's update on the local problem" no longer
    true. The two agree at weight 1.
12. **The neighborhoods are computed at construction** and stored, not at
    every build; the stub did not say when.
13. **`DomainLocalization` validates its arguments**, and `Identity` noise is
    row-local beside `PSDDiagonal`. A taper weight outside $[0, 1]$ raises in
    debug mode.
14. **`check_update_rule` takes `diagonal_noise`**, since its fixture's
    correlated noise is refused by a localized rule.

(kalman-implementation-changes)=
## Changes made while implementing

The page was written before the code, in the same pull request. Implementing
it, and its adversarial review, changed it as follows:

1. **`check_update_rule` averages 256 keys, at five standard errors**, not 64
   at six: measured, the weaker test passed a `Matheron` update whose mean was
   biased by a tenth of the posterior spread.
2. **The alignment check's magnitude term takes the largest magnitude over
   every block** ({ref}`kalman-alignment`). Built from each block alone, it
   rejected a correctly aligned approximation whose block was a linear map of
   another at a large offset.
3. **`update`'s result check covers every block's dtype, the order and the
   dimensions**, and `Matheron` and `inflate_additive` refuse an operator of
   another dtype. Operators carry no dtype (issue #33), so a float64 term on a
   float32 approximation gave a result with blocks of two dtypes, which the
   next call refused far from the cause.
4. **`update` type-checks `noise` itself**, rather than relying on the
   default approximation to.
5. **`check_update_rule` checks value validation and the missing key**, and
   says what its fixture does not cover; **`check_conditional_map` can check
   the keyed path** (`keyed_noise_free`).
6. **A target's independent term is stated never to enter the gain**
   ({ref}`kalman-matheron`), and the user guide's hybrid example pushes
   through `maps.Linear`, which absorbs its static covariance first.

PR 9's localization section was changed by its adversarial review:

7. **The localized build checks that each noise whitens to the particles'
   dtype.** A scaled float32 noise gave a float64 result, since the result's
   dtype followed the whitened factor's.
8. **The derivative at a zero weight is stated as a convention**, and the
   Euclidean distance takes a safe square root: a target located exactly at
   a given coordinate gave `nan` derivatives in the locations.
9. **The taper and the distance are no longer static fields** of
   `DomainLocalization`; only the taper's name is kept, for `repr`. Fresh
   callables made `jit` retrace every time.
10. **`max_neighbors` takes any integer**, a NumPy integer included; a `None`
    in `given_coords` raises `TypeError`, and the empty-mapping check follows
    the type checks.
11. **The construction cost is stated** ({ref}`kalman-localized-rule`, *Cost*).

(kalman-excluded)=
## Deliberately excluded

**A default update rule.** The square-root rule is exact in moments and the
stochastic rule keeps the shape of a nonlinear posterior better; neither is
right everywhere, so `update_rule` is required.

**Weighted updates.** A weighted ensemble's projection is not aligned, and a
weighted square-root reading is a different estimator with its own
literature. Resample first.

**Repairing failed particles.** Replacing a failed row changes the estimator;
it is an algorithm-level policy.

**Other square roots.** The left-multiplied adjustment of Anderson (2001) and
random rotations of the symmetric transform are members of the same family;
each would be a separate rule, written against `UpdateRule`.

**Perturbations drawn through `factor()`.** The stochastic rule draws the
given blocks' noise only in whitened coordinates, inside the conditional map;
drawing it through a factor as well would corrupt the joint law
({ref}`dist-matheron`). The rule never accepts or returns its perturbations.

**Covariance localization.** It tapers the sample covariance entrywise, which
destroys the low rank that every update here is built on
({ref}`kalman-localization`).

**Localization of correlated noise.** A neighborhood aligned to the blocks of
a block-diagonal noise covariance could be supported; one that cuts across a
correlated block cannot, by an operator-layer operation. Only row-local noise
is accepted, which covers the common diagonal case.

**Localization of a plain `Gaussian`.** A hybrid covariance's approximation
is not aligned. The stochastic rule's general path could be localized, at
$k \times K$ per local problem, and issue #47 records another option:
defining alignment by the first $J$ latent columns.

**Grouping coordinates that share a location.** Coordinates at one location
have the same neighborhood and weights, and so the same $A_p$; the layer
still computes one SVD per coordinate.

**A configurable divisor** in the relaxations' spreads: $J - 1$, as in
every computation of this layer.

## References

- Anderson, J. L. (2001). An ensemble adjustment Kalman filter for data
  assimilation. *Monthly Weather Review*, 129(12), 2884–2903.
- Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo implementation of
  the nonlinear filtering problem to produce ensemble assimilations and
  forecasts. *Monthly Weather Review*, 127(12), 2741–2758.
- Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
  with the ensemble transform Kalman filter. Part I: Theoretical aspects.
  *Monthly Weather Review*, 129(3), 420–436.
- Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Gaspari, G. & Cohn, S. E. (1999). Construction of correlation functions in
  two and three dimensions. *Quarterly Journal of the Royal Meteorological
  Society*, 125(554), 723–757.
- Hamill, T. M., Whitaker, J. S. & Snyder, C. (2001). Distance-dependent
  filtering of background error covariance estimates in an ensemble Kalman
  filter. *Monthly Weather Review*, 129(11), 2776–2790.
- Hamill, T. M. & Whitaker, J. S. (2005). Accounting for the error due to
  unresolved scales in ensemble data assimilation: a comparison of different
  approaches. *Monthly Weather Review*, 133(11), 3132–3147.
- Houtekamer, P. L. & Mitchell, H. L. (1998). Data assimilation using an
  ensemble Kalman filter technique. *Monthly Weather Review*, 126(3),
  796–811.
- Houtekamer, P. L. & Mitchell, H. L. (2001). A sequential ensemble Kalman
  filter for atmospheric data assimilation. *Monthly Weather Review*, 129(1),
  123–137.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Ott, E., Hunt, B. R., Szunyogh, I., Zimin, A. V., Kostelich, E. J.,
  Corazza, M., Kalnay, E., Patil, D. J. & Yorke, J. A. (2004). A local
  ensemble Kalman filter for atmospheric data assimilation. *Tellus A*,
  56(5), 415–428.
- Sakov, P. & Bertino, L. (2011). Relation between two common localisation
  methods for the EnKF. *Computational Geosciences*, 15(2), 225–237.
- Sakov, P. & Oke, P. R. (2008). A deterministic formulation of the ensemble
  Kalman filter: an alternative to ensemble square root filters. *Tellus A*,
  60(2), 361–371.
- Wang, X., Bishop, C. H. & Julier, S. J. (2004). Which is better, an
  ensemble of positive–negative pairs or a centered spherical simplex
  ensemble? *Monthly Weather Review*, 132(7), 1590–1605.
- Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for
  system errors in ensemble data assimilation. *Monthly Weather Review*,
  140(9), 3078–3089.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
  M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
  Machine Learning Research*, 22(105), 1–47.
- Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial estimate and
  observation availability on convective-scale data assimilation with an
  ensemble Kalman filter. *Monthly Weather Review*, 132(5), 1238–1253.
