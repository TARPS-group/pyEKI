"""Helpers shared by the maps layer's modules.

Private to :mod:`enskit.maps`: the class decorator, keys, the simulator
contract's checks on a returned array, and the constructor path new Gaussians
are built through.
"""
from __future__ import annotations

import functools
import os
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from ..distribution import EnsembleGaussian, Gaussian
from ..linalg.base import _pytree_dataclass

#: The class decorator of this layer: a frozen dataclass and a pytree whose
#: data fields may be ``None``, as the distribution layer's are.
map_class = functools.partial(_pytree_dataclass, allow_none=True)

#: The directory of the package, whose frames a warning skips.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep

#: The verdicts of :func:`dtype_rule`.
SAME, PROMOTE, REFUSE = "same", "promote", "refuse"


def describe(f) -> str:
    """How a map or simulator is named in messages: its ``__name__``, or ``repr``."""
    name = getattr(f, "__name__", None)
    return name if isinstance(name, str) else repr(f)


def check_key(where: str, key, *, required: bool, why: str = "") -> None:
    """Check a key's type, then its presence where one is needed."""
    if key is not None and not (
        isinstance(key, jax.Array) and jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
    ):
        raise TypeError(
            f"{where}: key must be a typed key from jax.random.key, got "
            f"{getattr(key, 'dtype', type(key).__name__)}. A raw uint32 key from "
            f"jax.random.PRNGKey is refused: its trailing shape makes a family "
            f"of keys ambiguous."
        )
    if key is None and required:
        raise ValueError(f"{where}: a key is required{why}")


def needs_key(f) -> bool:
    """Whether ``f`` declares that it takes a key first."""
    return getattr(f, "needs_key", False) is True


def dtype_rule(got, want) -> str:
    """The simulator contract's verdict on a returned dtype against the working one.

    ``SAME`` writes it as is, ``PROMOTE`` widens it with a warning, and
    ``REFUSE`` raises: integer, boolean, complex, wider, or not comparable.
    """
    if not jnp.issubdtype(got, jnp.floating):
        return REFUSE
    if got == want:
        return SAME
    if jnp.promote_types(got, want) == want:
        return PROMOTE
    return REFUSE


def refusal(where: str, who: str, what: str, got, want, remedy: str = "") -> str:
    """The message for a dtype that :func:`dtype_rule` refuses.

    ``what`` names the output (``"output 'g'"``); ``remedy`` replaces the
    default advice for a wider dtype.
    """
    if not jnp.issubdtype(got, jnp.floating):
        why = "which is not a real floating dtype"
    elif jnp.promote_types(got, want) == got:
        remedy = remedy or (
            f"Writing it would discard precision the simulator computed, "
            f"silently; return {want}, or work in {got} throughout"
        )
        why = f"which is wider than the working dtype {want}. {remedy}"
    else:
        why = f"which does not promote to the working dtype {want}"
    return f"{where}: {who} returned dtype {got} for {what}, {why}"


def output_array(
    where: str, who: str, name: str, value, n_particles: int, dtype, remedy: str = ""
):
    """Check one returned output against the simulator contract.

    The dtype is read from the value as returned, before any conversion, so
    that a conversion cannot hide a wider return (``jnp.asarray`` demotes a
    float64 NumPy array when x64 is off). A nested Python sequence carries no
    precision of its own, and its floats are read in ``dtype``.

    Returns ``(array, promoted_from)``: the output in ``dtype``, and the dtype
    it arrived in when that needed a promotion, else ``None``.
    """
    what = f"output {name!r}"
    if isinstance(value, (jax.Array, np.ndarray)):
        arr, got = value, value.dtype
    elif any(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(value)):
        # A sequence of JAX arrays, possibly tracers: they carry their dtype.
        arr = jnp.asarray(value)
        got = arr.dtype
    else:
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError) as exc:
            arr, cause = None, exc
        else:
            cause = None
        if arr is None or arr.dtype == object:
            raise ValueError(
                f"{where}: {who} returned {type(value).__name__} for {what}, which "
                f"is not array-like; return a jax.Array, a NumPy array or a nested "
                f"Python sequence"
            ) from cause
        got = dtype if np.issubdtype(arr.dtype, np.floating) else arr.dtype
    if arr.ndim != 2 or arr.shape[0] != n_particles or arr.shape[1] < 1:
        raise ValueError(
            f"{where}: {who} returned shape {arr.shape} for {what}, "
            f"expected ({n_particles}, d_out) with d_out >= 1. A simulator is "
            f"called once with every particle, not once per particle, and one "
            f"value per particle is ({n_particles}, 1)"
        )
    verdict = dtype_rule(got, dtype)
    if verdict == REFUSE:
        raise ValueError(refusal(where, who, what, got, dtype, remedy))
    out = jnp.asarray(arr, dtype=dtype)
    return out, (got if verdict == PROMOTE else None)


def warn_promoted(where: str, who: str, promoted: dict, dtype) -> None:
    """One warning for every output of one call that had to be promoted.

    The warning points at the first frame outside this package, so that it
    names the caller's line whether ``pushforward`` was called directly or
    through ``pipe``.
    """
    listed = ", ".join(f"{name!r} from {was}" for name, was in promoted.items())
    warnings.warn(
        f"{where}: {who} returned a narrower dtype than the working dtype {dtype}, "
        f"so its output was promoted ({listed}). Promotion does not recover the "
        f"digits the simulator did not compute.",
        UserWarning,
        stacklevel=_outside_package(),
    )


def with_block(g: Gaussian, name: str, mean, factor, cov, *, aligned: bool) -> Gaussian:
    """``g`` with block ``name`` replaced in place, or appended.

    Built through the public constructor.

    ``aligned`` keeps an :class:`EnsembleGaussian` one, with its divisor;
    the caller decides, since only it knows whether the latent space changed.
    """
    means = {n: g.mean(n) for n in g.names}
    factors = {n: g.factor(n) for n in g.names}
    covs = {n: g.block_cov(n) for n in g.names}
    means[name], factors[name], covs[name] = mean, factor, cov
    factors = {n: F for n, F in factors.items() if F is not None}
    covs = {n: D for n, D in covs.items() if D is not None}
    if aligned:
        return EnsembleGaussian(
            means,
            factors=factors,
            block_covs=covs,
            n_particles=g.n_particles,
            unbiased=g.unbiased,
        )
    return Gaussian(means, factors=factors, block_covs=covs, latent_dim=g.latent_dim)


# -- private -----------------------------------------------------------------------


def _outside_package() -> int:
    """The ``stacklevel``, for a ``warn`` in the caller, of the first frame outside it."""
    frame, level = sys._getframe(1), 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame, level = frame.f_back, level + 1
    return level
