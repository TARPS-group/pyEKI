# Maps contract

This page specifies `enskit.maps`: pushing a distribution through a map, the
three map classes the layer ships, the protocol for maps with structure, and
the **simulator contract**, which states what any callable used as a map must
satisfy. It also specifies `enskit.testing.check_simulator`, which checks a
simulator against that contract from outside. It is normative: an
implementation that violates a rule here is defective even if its tests pass.

Throughout, *must* and *never* state requirements, *should* states a strong
default that a documented reason may override, and *may* states a
permission. The layer is built on the distribution layer and the operator
layer, and this page refers to {doc}`distribution-contract` and
{doc}`linop-contract` rather than restating their rules.

:::{admonition} Status: implemented
:class: note

PR 5 of the redesign plan wrote this page and implemented it in
`enskit.maps` and `enskit.testing`. The simulator contract replaces, for the
new layers, the forward-model section of the old EKI contract; PR 7 deleted
the old `enskit.eki` and made {doc}`eki-contract` point here. Where this page
departs from the design's stubs (`docs/redesign/stubs/maps.py`), it says so
in {ref}`maps-departures`, and this page wins.
:::

(maps-scope)=
## Scope

A *map* takes the values of some blocks of a distribution to the value of a
new block, or a replacement for one of them. The layer provides:

- **`pushforward`**: the distribution of $f(\text{inputs})$, added to a
  distribution as a block, for an `Ensemble` and any callable, or for a
  `Gaussian` and a map with structure;
- **the simulator contract**: what a plain callable used as a map receives,
  may return and must be ({ref}`maps-simulators`);
- **`StructuredMap`**: the protocol for maps that also act exactly on a
  `Gaussian`, with two shipped instances, `Linear` and `AdditiveNoise`;
- **`BlackBox`**: a wrapper that makes a host-side simulator callable inside
  `jax.jit`, `jax.vmap` and `jax.grad`, with a zero derivative;
- in `enskit.testing`, **`check_simulator`**, which checks a simulator from
  outside.

What the layer is not:

- It ships **no simulators**. A simulator is any callable from a batch of
  inputs to a batch of outputs; the wrapper classes carry structure, not
  behavior. Toy simulators for tests and documentation live in
  `enskit.toy`, which nothing in the layers imports.
- It has no notion of an update, a gain, an observation, a step or time.
  Updating a distribution given values of some of its blocks is
  `enskit.kalman`; the two layers never import each other.
- It never repairs a failed particle and never decides what a failure means.
  It writes what the simulator returned ({ref}`maps-failure`).
- It does not distribute work. Fanning calls out across processes or
  machines is the simulator's business, done inside the callable.

(maps-notation)=
## Notation and conventions

| symbol | meaning |
| ------ | ------- |
| $x_b$ | the value of input block $b$; $x$ when there is one input |
| $y$ | the value of the output block |
| $d_b$, $d_{\text{out}}$ | the dimension of input block $b$ and of the output |
| $J$ | the number of particles, `n_particles` |
| $A$, $A_b$ | a linear map's operator, `(d_out, d_b)` |
| $c$ | a linear map's shift, `(d_out,)` |
| $R$ | an additive noise's covariance, `(d, d)` |
| $m_b$, $F_b$, $D_b$, $k$ | a `Gaussian`'s mean, factor row, independent term and latent width ({ref}`dist-representation`) |

The distribution layer's conventions hold here unchanged
({ref}`dist-notation`): blocks are named vectors, an ensemble block is a
`(J, d)` array whose row $j$ is particle $j$, the particle axis is a batch
axis to the operator layer, keys are typed and consumed whole, and scalars
are 0-d arrays.

(maps-objects)=
## The objects

| object | is | acts on |
| ------ | -- | ------- |
| `pushforward(dist, f, *, output, inputs=None, key=None)` | the distribution of `f(inputs)`, added as block `output` | `Ensemble`, `Gaussian` |
| any callable | a simulator ({ref}`maps-simulators`) | `Ensemble` |
| `StructuredMap` | protocol: a map that also pushes a `Gaussian` exactly | `Ensemble`, `Gaussian` |
| `Linear(op, shift=None)` | $x \mapsto Ax + c$, or $\sum_b A_bx_b + c$ | both; also a simulator |
| `AdditiveNoise(cov)` | $x \mapsto x + e$, $e \sim \mathcal N(0, R)$ independent of everything | both; not a simulator |
| `BlackBox(f, output_dim, *, dtype=None, needs_key=False)` | a host-side simulator, traceable, with a zero derivative | `Ensemble`; a simulator |
| `statistical_linearization(dist, *, inputs, output, min_norm=False, unbiased=True)` | the affine map $x \mapsto Ax + b$ that best predicts `output` from `inputs`, with its residuals, as a `Linearization` | `Ensemble`, `Gaussian` |
| `enskit.testing.check_simulator` | checks a simulator against the contract | — |

Three rules govern the set:

1. **Any callable is a map.** Nothing needs wrapping, subclassing or
   registering. The classes add structure (`Linear`, `AdditiveNoise`) or
   traceability (`BlackBox`), and never behavior a plain function could not
   have.
2. **Only structure acts on a Gaussian.** A Gaussian pushed through a
   nonlinear map is not Gaussian, so `pushforward` on a `Gaussian` accepts
   only a `StructuredMap`. Approximating a nonlinear pushforward is written
   at the call site: sample, push the particles, project.
3. **Distributions in, distributions out.** `pushforward` returns the type
   it was given, with every block it did not replace kept as it was.

(maps-pushforward)=
## `pushforward(dist, f, *, output, inputs=None, key=None)`

The distribution of $f(\text{inputs})$, as block `output` of the result:

$$
(x_1, \dots, x_n) \sim p \quad\Longrightarrow\quad
\big(x_1, \dots, x_n,\ f(x_{b_1}, \dots, x_{b_m})\big),
$$

where $b_1, \dots, b_m$ are the input blocks. The joint relation between
the inputs and the output is kept: particle by particle on an ensemble, and
in the shared factor on a Gaussian.

**Arguments.**

- `dist`: an `Ensemble` or a `Gaussian` (an `EnsembleGaussian` included).
  Anything else is a `TypeError`.
- `f`: on an `Ensemble`, any callable or `StructuredMap`; on a `Gaussian`,
  a `StructuredMap` only (`TypeError` otherwise, with a message that says to
  sample, push and project).
- `output`: keyword-only. A `str` names one output, which `f` returns as one
  array. A sequence of `str` names several, which `f` returns from one call
  ({ref}`maps-several-outputs`); a sequence of length one is a container of
  one. Names must be distinct (`ValueError`). Only a simulator has several
  outputs: several names with a `StructuredMap` are a `ValueError`.
- `inputs`: keyword-only. A `str` or a sequence of `str`: the blocks passed
  to `f`, positionally, in this order. `None`, the default, means every block
  of `dist`, in block order. At least one; each must be a block (`KeyError`
  listing the blocks), and none repeated (`ValueError`).
- `key`: keyword-only, a typed key ({ref}`maps-prng`). Required when the map
  draws randomness on an ensemble: an `AdditiveNoise`, or a callable that
  declares `needs_key`. Otherwise it is type-checked and ignored.

**Outputs replace inputs or are new.** Each output name must be either one
of `inputs`, whose block it then replaces (an in-place map
$x \mapsto M(x)$), or a name `dist` does not have, which is appended. An
output naming a block that is not an input is a `ValueError`: drop the block
first. The result's blocks are `dist`'s, in order, with each replaced block
in its own position and the new outputs appended in the order given. A
replacement may have a different dimension from the block it replaces.

**Returns** the type of `dist`. On an ensemble, the weights are kept,
`n_particles` is unchanged and every block that is not an output is the same
array. On a Gaussian the kind is kept when the latent space is: an
`EnsembleGaussian` stays one with the same `n_particles`, and becomes a plain
`Gaussian` when an independent term had to be absorbed ({ref}`maps-linear`,
{ref}`maps-additive-noise`).

**Dispatch.**

| `dist` | `f` | what happens |
| ------ | --- | ------------ |
| `Ensemble` | a `StructuredMap` | `f.push_ensemble(dist, inputs, output, key)` |
| `Ensemble` | any other callable | `f` is called once with every particle, under the simulator contract |
| `Gaussian` | a `StructuredMap` | `f.push_gaussian(dist, inputs, output)` |
| `Gaussian` | anything else | `TypeError` |

A `StructuredMap` is recognized by its two methods
({ref}`maps-structured`), so `Linear`, which is also callable, takes the
first row. `pushforward` checks the names before it dispatches, and checks
that a structured map's result has the type of `dist` and the block names
the rules above give (`TypeError` and `ValueError`, naming the map), so a
user-written structured map that drops or reorders a block fails loudly.

(maps-simulators)=
## The simulator contract

A **simulator** is any callable used as a map on an ensemble. It is the
layer's one interface to outside code, and this section states it
completely: what the callable receives, what it may return, and what it must
be. {doc}`user-guide/maps` is the same obligation written for the person
implementing one.

### What the callable receives

- **One positional argument per input block**, in the order of `inputs`:
  the `(J, d_b)` array of that block. With `needs_key`, a typed key comes
  first ({ref}`maps-prng`).
- **Every particle, in one call.** The callable is called **once per
  `pushforward`**, never once per particle. A per-particle function is
  wrapped with `jax.vmap`, or with a loop, by the caller. Particles of
  weight zero and failed particles are passed like any other: the layer does
  not select rows.
- **`jax.Array`s of the ensemble's dtype**, exactly two-dimensional. Outside
  any trace they are concrete, so a callable may branch on values, write
  files, spawn processes or block. Inside `jax.jit` or `jax.vmap` they are
  tracers, which a host-side callable cannot read: wrap it in `BlackBox`.
  `np.asarray` on an argument returns a **read-only view**; a callable that
  writes into its input must copy with `np.array`.

### What the callable may return

**Any array-like of shape `(J, d_out)`**, with $d_{\text{out}} \ge 1$: a
`jax.Array`, a NumPy array or a nested Python sequence. The same values in
any of the three give bit-identical results. That is a promise, not a
tolerance. The return's dtype is read **before** any conversion, from the
array as returned, so that no conversion can hide a wider return
(`jnp.asarray` demotes a float64 NumPy array silently when x64 is off). A
nested Python sequence carries no precision of its own: its floats are read
in the ensemble's dtype, and integers or booleans in it are refused. A return that is not exactly
two-dimensional, or whose leading axis is not $J$, is a `ValueError` naming
the simulator, the output, the shape and the shape expected; one value per
particle is `(J, 1)`, not `(J,)`.

**The dtype rule** compares the return's dtype with the ensemble's:

| return | result |
| ------ | ------ |
| the ensemble's dtype | written as is |
| a narrower real floating dtype (one that promotes to the ensemble's) | promoted to the ensemble's, with **one `UserWarning` per call** naming the simulator, the outputs and both dtypes |
| any other real floating dtype: wider, or not comparable | `ValueError` naming the simulator and both dtypes |
| integer, boolean or complex | `ValueError` |

A wider return raises because writing it would demote it silently, and
keeping it would make the ensemble's blocks disagree on dtype, which the
distribution layer refuses with a message that names neither the simulator
nor the cause. This settles issue #19, in the direction it preferred: the
error points at the simulator. A narrower return is promoted rather than
refused because single precision is a real constraint (a GPU solver, a file
written in single precision, worker processes started without
`JAX_ENABLE_X64=1`), and warned about because promotion does not recover the
digits the simulator never computed. Under `jax.jit` the warning is issued
when the function is traced, not at every call. The warning points at the
first frame outside EnsKit, so it names the caller's line whether
`pushforward` was called directly or through `pipe`.

(maps-several-outputs)=
### Several outputs from one call

With `output` a sequence of names, the callable returns one array per name,
as

- a **tuple** (or list) of arrays, in the order of `output`; its length must
  equal the number of names;
- a **mapping** keyed by exactly the names of `output`; or
- a **`NamedTuple`** whose fields are exactly the names of `output`, read by
  name, not by position.

Any other return, a length or a set of keys that differs, is a `ValueError`
naming the simulator, what it returned and the names expected. Each array
then follows the rules above independently, and the call issues at most one
promotion warning, listing every output it promoted. A simulator that
computes several quantities in one expensive run returns them this way
rather than being called once per quantity.

### What the callable must be

**Row $j$ of every output depends only on row $j$ of the inputs.** The
pushforward of an empirical distribution is the empirical distribution of
the images, $\sum_j w_j \delta_{(x_j,\, f(x_j))}$, only if the image of
particle $j$ is $f$ applied to particle $j$. A callable that couples its
rows (normalizing across the batch, writing into a shared accumulator)
returns something that is not a sample of the joint law of $(x, f(x))$ at
all. **Nothing inside `pushforward` detects this**: the shapes are right and
the numbers finite. From outside it is detectable, and `check_simulator`
does it by permuting the particles and by calling on a subset of them
({ref}`maps-check-simulator`).

**Determinism is not required, and side effects are permitted.** A
stochastic simulator is legitimate: the pushforward of particle $j$ is then a
draw of $f(x_j, \eta_j)$ for the simulator's internal randomness $\eta_j$,
and the result is a sample of the joint law of $(x, f(x, \eta))$. A layer
above that treats the output as a function of the inputs sees the spread of
$\eta$ as part of the relation between them. A simulator that draws its
randomness from a JAX key declares `needs_key` and receives one, which makes
it deterministic given the key ({ref}`maps-prng`).

(maps-failure)=
### Failed particles

**A failed particle is signaled by a non-finite row.** A particle whose
output row contains any `nan` or `inf` has failed; `Ensemble.all_finite`
finds it. `pushforward` writes the row as returned, in debug mode too, and
the result is a valid `Ensemble` ({ref}`dist-failed-particles`). Deciding
what to do about failed particles (repairing them, weighting them to zero,
raising) belongs to the layers above.

:::{important}
**The callable owns its own exceptions.** An array can express a failed
particle but not a raised exception, so a simulator that may crash, time
out, return a non-zero exit code or lose a worker **must catch that itself
and return a non-finite row** for the particles affected. An exception that
escapes the callable propagates out of `pushforward`, and every particle's
result is lost with it.
:::

Non-finiteness catches the failures that announce themselves, and nothing
more. Named, so that their absence reads as a decision:

- **Plausible but wrong output**, such as zeros, an initial condition, or a
  sentinel fill value like `-9999`, is finite and so is a valid particle. A
  fill value is the dangerous case, since it is far from every other
  particle and dominates any moment computed from them.
- **Finite but meaningless output**, such as a diverged solve that did not
  overflow, cannot be told from a poor input at this layer.
- **A call that never returns** is not caught. A per-particle timeout
  belongs to the callable that owns the process.

A simulator that can tell these cases apart should map them to non-finite
rows itself, where the information exists.

(maps-structured)=
## `StructuredMap`

The protocol for maps that also act exactly on a `Gaussian`. It is the
layer's extension point: implement both methods and `pushforward`
dispatches to them. Linearizations and sigma-point rules are future
implementations of it.

```python
class StructuredMap(Protocol):
    def push_ensemble(self, ensemble, inputs, output, key) -> Ensemble: ...
    def push_gaussian(self, gaussian, inputs, output) -> Gaussian: ...
```

`pushforward` calls them with `inputs` a non-empty tuple of distinct block
names of the distribution, `output` a single `str` that is either one of
`inputs` or not a block, and `key` a typed key or `None`. An implementation
must:

1. return the distribution's type, with every block that is not `output`
   kept, in its position, and `output` replaced in place or appended last;
2. raise `ValueError` for inputs it cannot take (a count or dimension it
   does not fit), `ValueError` for a missing key it needs, and the operator
   layer's `UnsupportedOpError`, unmodified, for an operator capability it
   lacks, all before any work;
3. be exact on a `Gaussian`, or say in its own documentation what it
   approximates;
4. agree in distribution between the two methods: pushing a Gaussian's
   samples through `push_ensemble` draws from the distribution
   `push_gaussian` returns;
5. be a pytree when it holds arrays or operators, so that it can cross a
   `jax.jit` boundary as data.

`isinstance(f, StructuredMap)` checks only that both methods exist.

(maps-linear)=
## `Linear(op, shift=None)`

An affine map with a structured operator:

$$
x \mapsto Ax + c, \qquad\text{or, with several inputs,}\qquad
(x_1, \dots, x_m) \mapsto \sum_{b=1}^{m} A_b x_b + c .
$$

**Construction.**

- `op`: a `LinOp` or a 2-D real floating array (wrapped in `Dense`), of
  shape `(d_out, d_in)`; or a non-empty mapping from input block name to
  such an operator, all with one `d_out`, for several inputs. Names are
  `str`. An operator must not be a vmapped family (`ValueError`).
- `shift`: `None` (no shift), or a `(d_out,)` real floating array.

The mapping's order is the order of the inputs. Called directly, a
mapping-form `Linear` takes the inputs positionally in that order; under
`pushforward`, `inputs` must name the mapping's keys in the same order
(`ValueError` otherwise), which catches a pair of operators swapped between
two inputs of equal dimension.

**Introspection.** `output_dim` (`int`), `input_names` (the mapping's keys,
or `None` for a single operator), `op` (the operator, or a `dict` of them in
order) and `shift`.

**As a simulator,** `Linear(...)(*xs)` returns
$\sum_b$ `A_b.matvec(x_b)` $+\ c$, `(..., d_out)` for `(..., d_b)` inputs with
one leading shape, so `pushforward` of an ensemble applies it to every
particle. On an ensemble, `push_ensemble` is that call, under the simulator
contract's shape and dtype rules, with the map named `Linear` in errors.

**On a Gaussian**, `push_gaussian` is exact. First, every input block that
has an independent term has it absorbed into the shared factor
({ref}`dist-absorb`), so that the output stays correlated with it; that
needs `D_b.factor()` of each such term, checked before any work, and makes
the result a plain `Gaussian`. After absorbing, every input has a factor row
$F_b$, and the output block gets

$$
m_y = \sum_b A_b m_b + c, \qquad F_y = \sum_b A_b F_b, \qquad D_y = 0,
$$

so that $\operatorname{cov}(y, y) = \sum_{a,b} A_a\operatorname{cov}(x_a,
x_b) A_b^\top$ and $\operatorname{cov}(y, x_b) = \sum_a
A_a\operatorname{cov}(x_a, x_b)$ for every input $b$, and the output's
covariance with any other block $z$ is $\sum_a A_a F_a F_z^\top$. $F_y$ is
the product operator `product(A, F_x)` for one input, and
`product(hstack(A_1, ..., A_m), hstack(F_1.T, ..., F_m.T).T)` for several:
a structured $A_b$ or $F_b$ is never densified. When nothing was absorbed,
the latent space is unchanged, and an `EnsembleGaussian` stays one with the
same `n_particles` and divisor ({ref}`dist-divisor`):
$F_y\mathbf 1 = \sum_b A_bF_b\mathbf 1 = 0$, so the factor stays centered,
and $\sqrt\delta\,F_ye_j = \sum_b A_b x^{(b)}_j$ minus the mean, so the
particles it realizes are the mapped particles.

On a prior with $k = 0$, `pushforward(prior, Linear(A), inputs="u",
output="g")` is the linear-Gaussian joint of $u$ and $g = Au + c$, with
factor rows $L$ and $AL$ for $LL^\top = C$, the prior's covariance.

**Dtype.** Operators carry no dtype, so `A_b.matvec(m_b) + c` may come out
wider than the distribution's (a float64 operator on a float32 Gaussian).
That is a `ValueError` naming `Linear` and both dtypes, on either kind of
distribution, as for a simulator, and the message says to give the
operators and the shift the distribution's dtype. Only the mean is checked
on a Gaussian: a factor row of a wider dtype is accepted, as the
distribution layer accepts operators of any dtype (issue #33).

**Why the structured route matters.** For an ensemble's projection, pushing
the particles through $A$ and projecting gives exactly the same moments as
pushing the projection, since sample covariances are linear in the
particles. The structured route pays off once the Gaussian is not a sample
covariance: a prior, a static or hybrid covariance, or the result of
earlier exact operations.

(maps-additive-noise)=
## `AdditiveNoise(cov)`

Independent additive Gaussian noise:

$$
x \mapsto x + e, \qquad e \sim \mathcal N(0, R) \ \text{independent of
everything else,}
$$

with $R$ = `cov`, a `PSDLinOp` that is not a vmapped family. It takes
exactly one input, of dimension `R.dim` (`ValueError` otherwise), and has
`dim` and `cov` as attributes. It is not callable: it has no meaning as a
deterministic function of its input.

**On a Gaussian**, exact, with no sampling:

- *Replacing the input* (`output` equal to the input) is
  `gaussian.add_noise({x: R})` ({ref}`dist-add-noise`).
- *A new output* copies the input and gives the copy the noise. When the
  input has an independent term it is absorbed first (so the copy is
  correlated with the input through it, and the result is a plain
  `Gaussian`), which needs `D_x.factor()`. The output block then has

  $$
  m_y = m_x, \qquad F_y = F_x, \qquad D_y = R,
  $$

  so $\operatorname{cov}(y, y) = \operatorname{cov}(x, x) + R$ and
  $\operatorname{cov}(y, x) = \operatorname{cov}(x, x)$. When the input has
  no independent term the latent space is unchanged, and an
  `EnsembleGaussian` stays one. $R$ is used afterward only through what
  later operations ask of it (`whiten`, to condition on $y$).

**On an ensemble**, it draws the noise: with $L$ = `R.factor()` of width $w$,

$$
y_j = x_j + L\,\eta_j, \qquad \eta_j \sim \mathcal N(0, I_w), \quad
j = 1, \dots, J,
$$

which needs a key (`ValueError` when missing) and `factor` (the operator
layer's `UnsupportedOpError` when unsupported, before drawing). The draw is
pinned ({ref}`maps-prng`). A covariance of a wider dtype than the ensemble's
makes $y$ wider, which is a `ValueError` saying to give $R$ the ensemble's
dtype. On a Gaussian, $R$ becomes an independent term and is not checked,
as the distribution layer does not check operators' dtypes (issue #33).

**The order of operations chooses the estimator.** Pushing an ensemble
through `AdditiveNoise` and projecting estimates $\operatorname{cov}(y, y)$
from sampled noise; projecting first and pushing the Gaussian adds $R$
exactly. Both estimate the same joint; the second has lower variance, and
needs only `whiten` of $R$ to condition on $y$, not `factor`.

(maps-blackbox)=
## `BlackBox(f, output_dim, *, dtype=None, needs_key=False)`

A host-side simulator made callable inside `jax.jit`, `jax.vmap` and
`jax.grad`. It wraps `f` in `jax.pure_callback`, so a NumPy code, a
subprocess or a scheduler submission can appear inside a traced function,
and stops the gradient at its inputs, so that differentiation treats its
outputs as constants: the derivative of a `BlackBox` is zero by declaration.
Outside any trace a plain callable needs no wrapper.

**Construction.**

- `f`: a callable satisfying the simulator contract, which receives NumPy
  arrays.
- `output_dim`: a Python `int` $\ge 1$, the output's $d_{\text{out}}$, or a
  tuple of them for a simulator with several outputs, which `f` then
  returns as a tuple in that order. The callback must declare its result
  shapes before calling `f`, so they are given here.
- `dtype`: keyword-only. The result dtype, a real floating dtype; `None`
  means the dtype of the first input at each call.
- `needs_key`: keyword-only, a `bool`. When `True`, the `BlackBox` is
  called as `bb(key, *inputs)` with a typed key and calls
  `f(key_data, *inputs)`, with `key_data` the key's `jax.random.key_data`, a
  `uint32` array; `pushforward` then requires a key.

**Calling it.** `bb(*inputs)` takes one or more arrays of one leading
dimension $n$ and returns `(n, d_out)` (a tuple of them, with a tuple
`output_dim`) of the declared dtype. Inside the callback, `f`'s return is
converted with `np.asarray` and checked against the simulator contract's
rules: the shape, a tuple of the right length when there are several
outputs, and the dtype rule against the declared dtype (a narrower return
is promoted; wider, integer or complex is an error). A promotion issues one
`UserWarning` per call of `f`, listing every output promoted. Because it is
raised inside the callback, it is issued **each time `f` runs**, under
`jit` too, not once at tracing as `pushforward`'s is.

**Under `jax.vmap`** the callback uses `vmap_method="sequential"`: `f` is
called once per member of the mapped family, each time with ordinary
`(n, d_in)` arrays, so the simulator contract holds per call.

**Errors inside the callback**, including the checks above and any
exception `f` lets escape, are raised by JAX as `jax.errors.JaxRuntimeError`
carrying the original message, eagerly as well as under `jit`.

**A zero derivative is a statement about the gradient being computed, not
about the simulator.** A gradient through a computation that calls a
`BlackBox` at points depending on the differentiated parameter is a partial
derivative with those outputs held fixed. That is the derivative wanted when
the parameter acts elsewhere (a noise scale, a prior's hyperparameter) and
the simulator's outputs are data; it is not the derivative of the
simulator.

(maps-linearization)=
## `statistical_linearization(dist, *, inputs, output, min_norm=False, unbiased=True)`

The statistical linear regression of block $y$ = `output` on the blocks
$x$ = `inputs` under `dist`:

$$
A = C_{yx}C_{xx}^{-1}, \qquad b = m_y - Am_x, \qquad
\Omega = \operatorname{cov}(y - Ax - b) = C_{yy} - AC_{xy},
$$

the affine map minimizing $\mathbb E\lVert y - Ax - b\rVert^2$. When
$y = f(x)$ it is a linearization of $f$ with respect to the distribution of
$x$ (Lefebvre et al., 2002); for $x \sim \mathcal N(m, C)$ and
differentiable $f$, Stein's lemma gives $A = \mathbb E[Df(x)]$ (Stein,
1981). It does not call a simulator: `dist` already holds $y$, so a
simulator is linearized by pushing first,

```python
ens = maps.pushforward(ens, f, inputs="x", output="y")
fit = maps.statistical_linearization(ens, inputs="x", output="y")
```

and an ensemble whose outputs a run already computed is linearized without
calling the simulator again.

**The computation** is `g.regression(output, given=inputs,
min_norm=min_norm)` ({ref}`dist-regression`), with
`g = dist.project(unbiased=unbiased)` for an `Ensemble` and `g = dist` for a
`Gaussian`. `unbiased` selects the divisor of {ref}`dist-divisor`: $A$ and
$b$ do not depend on it, and $\Omega$ is proportional to its reciprocal, so
`unbiased=False` gives $\Omega$ times $(J - 1)/J$ unweighted, or
$1 - \sum_j w_j^2$ weighted. A `Gaussian`'s covariance is used as it is (an
`EnsembleGaussian` keeps the divisor it was projected with), so
`unbiased=False` with a `Gaussian` raises `ValueError`. Its three cases, its
validation and its accuracy apply unchanged. A ridge regression is a
`Gaussian` argument, `ens.project().add_noise(x=Lam)`; there is no
regularization argument.

**The result** is a `Linearization`, a `NamedTuple` and so a pytree:

| field | is |
| --- | --- |
| `map` | `Linear` of the coefficients and the intercept: one operator for one input; for several, a mapping in the order `inputs` was written, which is the order `map` takes its arguments and `pushforward` must name them |
| `residual_cov` | $\Omega$, the regression's `residual_cov` |
| `residuals` | for an `Ensemble`, the `(J, d_y)` array `dist[output] - map(*(dist[n] for n in inputs))`; `None` for a `Gaussian` |

**Rules.**

- `inputs` is required (a `str` or a sequence of `str`, at least one,
  distinct); `output` is a `str` naming a block that is not an input;
  `min_norm` and `unbiased` are `bool`s. A vmapped family raises, as in `pushforward`.
- **A weighted ensemble whose inputs' total dimension exceeds $J - 1$
  raises `ValueError`** before projecting, naming `resample` and ridge. Its
  projection's null vector is $\sqrt w$, not $\mathbf 1$, so the
  minimum-norm computation of the distribution layer does not apply to it.
- **The residuals depend on the regime**, which the docstring states: when
  the inputs' dimension is at least $J - 1$, the fit interpolates and the
  residuals and $\Omega$ vanish to round-off; below it, $\Omega$ has the
  projection's divisor $\delta$ and, for a linear model, has expectation
  $(J - 1 - d_x)/\delta$ times the error covariance.
- **Consistency.** Without regularization, pushing the inputs' projection
  through `map` and then `AdditiveNoise(residual_cov)` reproduces the
  projected joint. $\Omega$ is singular in general and has no `whiten`: as
  an independent term it must be absorbed into the factor before `cov` or
  conditioning reads it, or added to a noise covariance as
  `LowRankUpdate(R, residual_cov.factor())`.
- **Ridge and residuals.** For a `Gaussian` argument there are no particles,
  hence no `residuals`; $\Omega$ is then the regression's, the covariance of
  the output given the noisy inputs.

(maps-prng)=
## Randomness

Keys are typed keys (`jax.random.key`); a raw `uint32` key is a
`TypeError`. A key is consumed whole: nothing stores, advances or splits
one beyond the call.

**A simulator declares that it needs a key** with an attribute `needs_key`
whose value is `True`; `pushforward` then calls it as `f(key, *inputs)` and
raises `ValueError` when no key is given. Any other value, or no attribute,
means no key. `BlackBox` sets the attribute from its argument; a JAX
function can set it directly (`f.needs_key = True`). The declaration is read
off the callable because a callable has nowhere else to put it.

The draws are pinned. Same key, same arguments and same representation give
identical arrays across EnsKit releases, for a fixed JAX version:

| call | draw |
| ---- | ---- |
| `AdditiveNoise` on an ensemble | `normal(key, (J, w), dtype)` for $\eta$, with $w$ the width of `R.factor()` and `dtype` the ensemble's; $y = x +$ `L.matvec(eta)` |
| a callable with `needs_key` | the key itself, passed whole; what the simulator draws is its own |
| `BlackBox` with `needs_key` | `jax.random.key_data(key)`, passed to `f` |

(maps-validation)=
## Validation and errors

The four tiers of {ref}`contract-validation` apply as in
{ref}`dist-validation`: everything static is checked always, values only in
debug mode. This layer adds no tier-4 checks of its own: a simulator's
output may be non-finite by contract, and the distributions it builds run
their own.

**Order of checks** in `pushforward`, so that when two things are wrong the
same one is reported:

1. the type of `dist`, then the family guard (a vmapped family is a
   `ValueError`, as in {ref}`dist-jax`);
2. `output`, then `inputs`: types, emptiness, repeats, unknown blocks, and
   the replace-or-new rule;
3. the key's type;
4. the kind of `f` against the kind of `dist` and the number of outputs;
5. inside the map: its fit to the inputs (count, names, dimensions), the
   key's presence, operator capabilities, then the work;
6. the outputs: container, shapes, dtypes; then the result's structure.

The layer defines **no exception types**. `UnsupportedOpError` comes only
from the operator layer, unmodified. Error messages name the call, the map
or simulator (by `__name__`, or `repr`), the block, the expectation and
what was found.

| condition | raises |
| --------- | ------ |
| `dist` not an `Ensemble` or `Gaussian` | `TypeError` |
| `dist` a vmapped family | `ValueError` |
| `output` or `inputs` not a `str` or a sequence of `str` | `TypeError` |
| no output, no input, or a name repeated in either | `ValueError` |
| an input that is not a block | `KeyError` |
| an output naming a block that is not an input | `ValueError` |
| a raw `uint32` key, or a key that is not a key | `TypeError` |
| a `Gaussian` with a map that is not a `StructuredMap` | `TypeError` |
| several outputs with a `StructuredMap` | `ValueError` |
| `f` neither callable nor a `StructuredMap` | `TypeError` |
| a needed key missing | `ValueError` |
| a map's fit to its inputs (count, names, dimensions) | `ValueError` |
| a capability missing (`factor`) | `UnsupportedOpError`, from the operator layer |
| a return that is not array-like, or not the declared container | `ValueError` |
| an output's shape not `(J, d_out)` with `d_out >= 1` | `ValueError` |
| an output's dtype integer, boolean, complex, wider or not comparable | `ValueError` |
| an output's dtype narrower | promoted, one `UserWarning` per call |
| a structured map's result of the wrong type or blocks | `TypeError` / `ValueError` |
| a constructor argument of the wrong type | `TypeError` |
| a constructor argument of the wrong shape, size or value | `ValueError` |
| anything raised inside a `BlackBox`'s callback | `jax.errors.JaxRuntimeError` |

(maps-derivatives)=
## Differentiability

- **A traceable simulator is differentiated by JAX** like any other
  function: the derivative of `pushforward` of an ensemble through a JAX
  callable is the derivative of the callable, row by row.
- **A `BlackBox` has a zero derivative** with respect to its inputs, in
  forward and reverse mode, under `jit` and `vmap` ({ref}`maps-blackbox`).
  A host-side callable used outside any trace returns arrays that are
  constants for any gradient computed afterward, which is the same thing.
- **`Linear` and `AdditiveNoise` on a Gaussian are differentiable** with
  respect to the Gaussian's arrays, the operators' parameters and the shift,
  since they only build means, product operators and independent terms from
  them; the derivatives of what is computed from the result are those of the
  distribution layer ({ref}`dist-derivatives`).
- **`AdditiveNoise` on an ensemble is reparameterized**: for a fixed key the
  result is $x + L\eta$ with $\eta$ fixed, and its derivative with respect
  to the parameters of $R$ is the pathwise derivative through `factor`.

(maps-jax)=
## JAX integration

`Linear`, `AdditiveNoise` and `BlackBox` are frozen pytrees, declared with
the distribution layer's machinery: construction validates, unflattening
bypasses the constructor, objects compare and hash by identity and are
never `static_argnums`. Their fields:

| class | data | static |
| ----- | ---- | ------ |
| `Linear` | the operators (a tuple, in input order); the shift, or `None` | the input names, or `None` |
| `AdditiveNoise` | the covariance | — |
| `BlackBox` | — | `f`, `output_dim`, `dtype`, `needs_key` |

A `Linear` or `AdditiveNoise` passed into a `jit`-compiled function crosses
as data, so new operator values do not recompile. Whether a `Linear` has a
shift is tree structure, as `None` entries are in the distribution layer.
A `BlackBox` has no leaves; passed into a `jit`-compiled function it is
static, and one with a different `f` compiles again.

`pushforward` is `jit`- and `vmap`-safe when its map is: a JAX callable,
`Linear`, `AdditiveNoise` or a `BlackBox`. Every shape is static, and every
check above is a check of structure, shapes or dtypes, so none of them reads
a traced value.

(maps-repr)=
## `repr`

Type name and static sizes, never array contents:

```text
Linear(shape=(6, 4), shift=True)
Linear(inputs={'x': 4, 'z': 2}, output_dim=6, shift=False)
AdditiveNoise(dim=6)
BlackBox(f=simulate, output_dim=10, needs_key=False)
```

(maps-check-simulator)=
## `enskit.testing.check_simulator`

`check_simulator(f, input_dims, output_dims, *, n_particles=6, seed=0, stochastic=False)`
checks a simulator against {ref}`maps-simulators` from outside, by calling
it through `pushforward` on ensembles of pseudo-random inputs. It raises
`AssertionError` naming the violated obligation, and returns `None`.

- `input_dims`: an `int`, or a non-empty sequence of `int`, the dimension of
  each positional input.
- `output_dims`: an `int` for a simulator returning one array, or a mapping
  from output name to dimension for one returning several (in any of the
  containers of {ref}`maps-several-outputs`). Each output's dimension is
  checked against it.
- `n_particles`: a Python `int` $\ge 3$ (`ValueError` below): both
  row-independence comparisons need a proper subset and a permutation that
  is not the identity, and at $J < 3$ neither exists.
- `stochastic`: declares a simulator that is not deterministic, which skips
  the determinism and row-independence checks.

| checked | how | calls |
| ------- | --- | ----- |
| the containers, shapes and dtypes | at `n_particles` and at `n_particles + 1`, since a simulator whose rows are independent cannot depend on how many there are; a narrower dtype fails the check, although `pushforward` promotes it | 2 |
| determinism | calling twice on the same inputs (and key), comparing bit-exactly | 1 |
| row independence, order | permuting the particles (never the identity permutation) and comparing **bit-exactly**: the rows are the same set either way. Catches a coupling that depends on order, such as a running total | 1 |
| row independence, company | calling on two of the particles without the others and comparing to a **per-element tolerance** of $10^{-8}\max(1, \lvert v\rvert)$ for each output value $v$: a differently shaped batch may legitimately round differently, while a coupling changes the answer by $O(1)$. Catches a symmetric coupling, such as normalizing across the batch, which survives a permutation | 1 |

Comparisons treat two `nan`s as equal and compare the pattern of non-finite
entries exactly: a non-finite row is legal, and a failure that moves
depending on the other particles is a coupling like any other. The
tolerance is per element, not one scale set by the largest output, so a
coupling in a small output beside a large one is not hidden.

A simulator that declares `needs_key` is called with a fixed key; its
determinism given the key is checked, and the row-independence checks are
skipped, since its draws are laid out by row. With `stochastic=True` the
check makes two calls, and with `needs_key` three. **Otherwise it makes
five**, which for a real simulator is five runs of the expensive thing:
check a cheap configuration of it.

Both row-independence comparisons are necessary conditions, not sufficient
ones. The check does not test failure signaling on the simulator's own
error paths, cost, side effects, or whether a non-finite row should have
been produced.

(maps-surface)=
## Public surface

`enskit.maps` exports exactly `pushforward`, `StructuredMap`, `Linear`,
`AdditiveNoise`, `BlackBox`, `statistical_linearization` and
`Linearization`. `enskit.testing` exports `check_simulator`;
later pull requests add `check_update_rule` and `check_conditional_map`
there. The layers never import `enskit.testing`. The modules of both
packages are private, and imported only from inside their own package.

(maps-consumers)=
## How the layers above consume this one

Not normative for this layer, but its shape was set by these call sites.

**The algorithms** push every particle through the user's simulator with one
call per step, and read `all_finite` on the result to find failed particles:

```python
ens = maps.pushforward(ens, forward, inputs="u", output="g")
```

The filter's forecast is a replacement, and its noise a second pushforward:

```python
ens = (ens.pipe(maps.pushforward, transition, inputs="x", output="x")
          .pipe(maps.pushforward, maps.AdditiveNoise(Q), inputs="x", output="x",
                key=key))
```

**Building a joint for exact Gaussian work** is two pushforwards of a
prior; conditioning the result on `y` is the linear-Gaussian posterior:

```python
joint = (prior.pipe(maps.pushforward, maps.Linear(H), inputs="x", output="g")
              .pipe(maps.pushforward, maps.AdditiveNoise(R), inputs="g", output="y"))
```

**`enskit.kalman`** never imports this layer. A user adds known noise to a
projection with `Gaussian.add_noise`, or with `AdditiveNoise` through
`pipe`, before handing it to an update.

(maps-conformance)=
## Conformance

Obligations on `tests/test_maps.py`, under the package's rules: exactness
against closed forms written in the tests by hand, with dense linear
algebra, never routed through the layer. The suite must verify at least:

1. **Names and structure.** The replace-or-new rule; result block order;
   weights, `n_particles` and untouched blocks kept; `inputs` defaulting to
   every block; a `str` against a sequence of one.
2. **The call.** One call per `pushforward`, counted; one positional
   `(J, d_b)` `jax.Array` per input, in the order of `inputs`, concrete when
   eager, read-only under `np.asarray`; every particle passed, failed and
   zero-weight ones included.
3. **The containers are a promise.** A `jax.Array`, a NumPy array and a
   nested list give bit-identical ensembles.
4. **Several outputs**: a tuple, a mapping and a `NamedTuple` (read by name,
   in a field order different from `output`) give the same result; a wrong
   length or set of keys raises.
5. **The dtype rule**, each row of its table, and exactly one warning per
   call when two outputs are promoted, pointing at the caller's line directly
   and through `pipe` (#19). The dtype is read before conversion, so a wider
   NumPy return raises with x64 off too; a nested list is read in the
   ensemble's dtype; the structured maps' messages name what to change.
6. **Failed rows are written through**, in debug mode too, and the result's
   `all_finite` finds them.
7. **`needs_key`**: the key is passed whole and first, a missing key
   raises, and the result is deterministic given the key.
8. **`Linear` on an ensemble** is exactly $Ax_j + c$ per particle, and
   pushing the particles then projecting equals projecting then pushing, in
   mean, covariance and cross-covariance, to round-off.
9. **`Linear` on a Gaussian** matches the dense closed form for the mean,
   $\operatorname{cov}(y)$, $\operatorname{cov}(y, x)$ and the covariance
   with a third block: with and without an independent term on the input,
   with several inputs, and replacing its input. The type is kept exactly
   when nothing is absorbed. The new row is a product of the operators, not
   a dense array. On a prior, the joint conditioned on $g$ is the
   linear-Gaussian posterior.
10. **`AdditiveNoise` on a Gaussian** replacing its input equals
    `add_noise`; to a new block it matches the closed form, with and
    without the input's independent term, and keeps the type exactly when
    nothing is absorbed.
11. **`AdditiveNoise` on an ensemble** reproduces the pinned draw; a missing
    key and a term without `factor` raise before drawing.
12. **`BlackBox`** matches its plain callable eagerly, under `jit` and under
    `vmap` (with `f` called once per member, on 2-D arrays); its derivative
    is exactly zero in forward and reverse mode; `needs_key` passes the key
    data; a tuple `output_dim` returns several outputs; the callback's
    checks raise.
13. **Differentiability**: the derivative through a JAX simulator, through
    `Linear` on a Gaussian (with respect to the operator and the shift) and
    through `AdditiveNoise` on an ensemble (with respect to $R$), each
    against a closed form.
14. **JAX**: `pushforward` under `jit` and `vmap`; a `Linear` passed as a
    `jit` argument with new values does not recompile, counted.
15. **A user-written `StructuredMap`** is dispatched to, and a result of the
    wrong blocks is refused.
16. **Validation**: every row of the table in {ref}`maps-validation`.
17. **`check_simulator`** passes every toy simulator and a NumPy one and a
    failing one; rejects each defect it claims to catch (an order-dependent
    coupling, a symmetric one, a per-particle function, a narrow, an
    integer, and a nondeterministic return, a wrong size at $J + 1$, a
    moving failure, a coupling in a small output); makes the number of
    calls it promises; refuses $J < 3$; never uses the identity permutation;
    honors `stochastic` and `needs_key` while still running every other
    check; and checks a `BlackBox` as it checks a function, raising
    `AssertionError` for a failure inside the callback.
18. **`statistical_linearization`** (`tests/test_linearization.py`): the
    recipe calls the simulator once and recovers a linear one; residuals
    and $\Omega$ against least squares by hand; several inputs keep the
    order written, and the map pushes with them; the consistency identity
    in both exact cases; a `SymmetricSquareRoot` update of the ensemble
    equals the linear-Gaussian update with model $A$ and noise
    $R + \Omega$; on an
    antithetic ensemble a quadratic's fit is exactly its average Jacobian;
    a `Gaussian` gives ridge and no residuals; a wide weighted ensemble and
    each validation rule raise; `jit` agrees and `vmap` gives a family.

### Ported regression tests

The simulator contract and its checker moved here from `enskit.eki`; the
checker's tests moved with them, under the same claims. PR 7 deleted the
originals from `tests/test_toy.py` with `enskit.eki`.

| old test (`tests/test_toy.py`) | new test (`tests/test_maps.py`) |
| ------------------------------ | ------------------------------- |
| `test_4_every_model_passes_check_forward_model` | `test_17_every_toy_simulator_passes_check_simulator` |
| `test_4_check_forward_model_rejects_the_defects_it_claims_to_catch` | `test_17_check_simulator_rejects_the_defects_it_claims_to_catch` |
| `test_4_check_forward_model_checks_the_second_ensemble_size_and_the_argument` | `test_17_check_simulator_checks_the_second_size_and_the_argument` |
| `test_4_check_forward_model_compares_where_the_failures_are` | `test_17_check_simulator_compares_where_the_failures_are` |
| `test_4_check_forward_model_accepts_a_failing_model_and_a_numpy_one` | `test_17_check_simulator_accepts_a_failing_simulator_and_a_numpy_one` |
| `test_12_regression_the_checker_rejects_a_coupling_in_a_small_observable` | `test_regression_check_simulator_rejects_a_coupling_in_a_small_output` |
| `test_12_regression_the_checker_refuses_an_ensemble_too_small_to_check` | `test_regression_check_simulator_refuses_too_few_particles` |
| `test_12_regression_the_permutation_is_never_the_identity` | `test_regression_check_simulator_permutation_is_never_the_identity` |

From `tests/test_eki.py`, `test_28_the_accepted_containers_are_a_promise_not_a_tolerance`
became `test_3_the_containers_are_a_promise_not_a_tolerance`, and
`test_29_promotion_only_ever_widens`, which pinned the behavior #19
questioned, became `test_5_a_wider_output_raises_naming_the_simulator`.

(maps-departures)=
## Departures from the design

The stub `docs/redesign/stubs/maps.py` is the design; where this page
differs, this page governs. Each departure, and why:

1. **`needs_key` is a declaration any callable may make**, not only a
   `BlackBox` argument. The stub had `pushforward` pass a key to "a
   stochastic callable declared with `needs_key=True` on `BlackBox`", which
   left a pure-JAX stochastic simulator, which needs no `BlackBox`, without
   a way to receive one. `BlackBox`'s argument sets the same attribute.
2. **`BlackBox`'s `output_dim` may be a tuple**, for a host-side simulator
   with several outputs, which the stub's single `int` could not trace.
3. **`BlackBox` checks its simulator's return inside the callback**, under
   the simulator contract's rules. `pure_callback` checks only that the
   result matches the declared shape and dtype, and an `np.asarray(...,
   dtype=...)` conversion, as in the prototype, would have truncated an
   integer or wider return silently.
4. **`BlackBox`'s zero derivative is a `stop_gradient` of its inputs**, not
   a custom JVP rule; the derivatives are the same, zero, and no rule has to
   be kept linear in its tangents.
5. **The `StructuredMap` protocol pushes one output.** Several outputs are a
   simulator feature; a `StructuredMap` is called with a single output name.
6. **`Linear` accepts arrays** for its operators, wrapped in `Dense`, as the
   distribution layer accepts factor rows.
7. **`pushforward` requires at least one input**, and checks a structured
   map's result, which the stub did not say.
8. **`AdditiveNoise` checks that it has exactly one input** and states
   which representation it produces when copying a block, which the stub
   left to the prototype.
9. **The checker is `check_simulator(f, input_dims, output_dims, ...)`**,
   with several inputs and outputs, and runs through `pushforward`, so that
   it checks the callable as the layer will call it. Its predecessor,
   `enskit.eki.testing.check_forward_model`, took one input and one output.

(maps-excluded)=
## Deliberately excluded

Recorded so their absence reads as a decision.

**Forward models.** The layer ships none, and defines no base class,
protocol or registry for a simulator beyond the contract above.

**A `Gaussian` through a nonlinear map.** Linearization, sigma points and
other Gaussian approximations of a nonlinear pushforward are future
`StructuredMap` implementations, each stating what it approximates. Until
one exists, the approximation is written at the call site: sample, push,
project, and, for an affine approximation, `statistical_linearization` of
the result. That function fits a `Linear` map; it is not itself a
`StructuredMap`, since `push_gaussian` takes no key with which to draw the
points a nonlinear map would be evaluated at.

**Simulators that see weights.** A simulator receives particles, not
weights, so it cannot couple its rows through them either.

**Demoting a wider return.** It would discard precision the simulator
computed, silently, which is the loss the promotion warning exists to
report.

**Per-particle calls.** The layer calls once per pushforward. A simulator
that is naturally per particle is a `jax.vmap` or a loop, written by the
caller, who knows whether the particles can run in parallel.

**Block-valued simulators.** A simulator receives arrays, not a
distribution or a mapping of blocks, so it cannot read a block it was not
given.

## References

`statistical_linearization` implements statistical linear regression:

- Lefebvre, T., Bruyninckx, H. & De Schutter, J. (2002). Comment on "A new
  method for the nonlinear transformation of means and covariances in
  filters and estimators". *IEEE Transactions on Automatic Control*, 47(8),
  1406–1409.
- Stein, C. M. (1981). Estimation of the mean of a multivariate normal
  distribution. *The Annals of Statistics*, 9(6), 1135–1151.

Otherwise the layer implements no method from the literature. The order-of-operations
point of {ref}`maps-additive-noise` is the one the ensemble Kalman
literature makes about perturbed observations, and the distribution layer
cites it ({doc}`distribution-contract`, *References*): Burgers, van Leeuwen
& Evensen (1998).
