"""Helpers shared by the distribution layer's modules.

Private to :mod:`enskit.distribution`: block arguments, keys, the family
guard, the two debug-mode checks of the conditioning paths, and the
constructor-bypassing path results are built through.
"""
from __future__ import annotations

import functools
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from ..linalg import value_check
from ..linalg.base import _broadcast_batch, _pytree_dataclass

#: The class decorator of this layer: a frozen dataclass and a pytree whose
#: data fields may be ``None`` or tuples with ``None`` entries.
distribution_class = functools.partial(_pytree_dataclass, allow_none=True)

broadcast_batch = _broadcast_batch

#: What the whitening check says, after naming the method and the block.
WHITENING_CAUSE = (
    "The likeliest cause is an independent term that is singular or cannot "
    "whiten these values, or values so large relative to it that whitening "
    "overflows."
)

#: What the result check says, after naming the method and the quantity.
RESULT_CAUSE = (
    "The whitened inputs were finite, so this means overflow (applying a "
    "target factor row, or squaring a whitened residual) or a decomposition "
    "that failed."
)


def build(cls: type, **fields):
    """Build an instance without running its constructor.

    The path pytree unflattening takes. Results whose fields are known valid
    are built this way, after their checks have run, so that no
    constructor's own value check fires first and names the wrong call.
    """
    obj = object.__new__(cls)
    for name, value in fields.items():
        object.__setattr__(obj, name, value)
    return obj


def guard(obj, method: str) -> None:
    """Refuse a method on a vmapped family, before any other check."""
    batch = obj.batch_shape
    if batch != ():
        raise ValueError(
            f"{obj!r}.{method}: this is a vmapped family with batch shape "
            f"{batch}, which cannot be used directly; apply it under jax.vmap, "
            f"one member at a time."
        )


def merge_blocks(where: str, mapping, kwargs: dict) -> dict:
    """Merge a positional mapping and keywords into one ordered dict.

    Mapping entries come first, in their order, then keywords in the order
    written. A name given in both raises ``TypeError``, as a repeated keyword
    does in Python.
    """
    out: dict = {}
    if mapping is not None:
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"{where}: the positional argument must be a mapping from block "
                f"name to value, got {type(mapping).__name__}"
            )
        for name, value in mapping.items():
            if not isinstance(name, str):
                raise TypeError(
                    f"{where}: block names must be str, got {type(name).__name__}"
                )
            out[name] = value
    for name, value in kwargs.items():
        if name in out:
            raise TypeError(
                f"{where}: block {name!r} is given twice, in the mapping and as "
                f"a keyword"
            )
        out[name] = value
    return out


def check_known(where: str, names: tuple[str, ...], given) -> None:
    """Raise ``KeyError`` for any name in ``given`` that is not a block."""
    for name in given:
        if name not in names:
            raise KeyError(f"{where}: no block named {name!r}; the blocks are {names}")


def check_name_list(where: str, names: tuple[str, ...], selected) -> tuple[str, ...]:
    """Validate positional block names: each a block, none repeated."""
    selected = tuple(selected)
    for name in selected:
        if not isinstance(name, str):
            raise TypeError(
                f"{where}: block names must be str, got {type(name).__name__}"
            )
    check_known(where, names, selected)
    seen = set()
    for name in selected:
        if name in seen:
            raise ValueError(f"{where}: block {name!r} is named more than once")
        seen.add(name)
    return selected


def given_names(where: str, names: tuple[str, ...], given) -> tuple[str, ...]:
    """Normalize a ``given`` argument (a ``str`` or sequence of ``str``).

    Returns the names in the distribution's block order.
    """
    if isinstance(given, str):
        given = (given,)
    try:
        given = tuple(given)
    except TypeError:
        raise TypeError(
            f"{where}: given must be a block name or a sequence of block names, "
            f"got {type(given).__name__}"
        ) from None
    if not given:
        raise ValueError(f"{where}: at least one given block is required")
    selected = check_name_list(where, names, given)
    return tuple(n for n in names if n in selected)


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


def require(op, *operations: str) -> None:
    """Demand operations of an operator, in order, before any work.

    The ``UnsupportedOpError`` is the operator layer's, raised unmodified;
    inside :func:`~enskit.linalg.dense_fallback` a covered operator passes.
    """
    for name in operations:
        op._require(name)


def check_n_particles(where: str, n_particles, name: str = "n_particles") -> None:
    """A Python ``int`` (exact type) of at least 2."""
    if type(n_particles) is not int:
        raise TypeError(
            f"{where}: {name} must be a Python int, got {type(n_particles).__name__}. "
            f"It determines an output shape, so it can never be traced."
        )
    if n_particles < 2:
        raise ValueError(
            f"{where}: {name} must be at least 2, got {n_particles}. A single "
            f"particle has no anomalies."
        )


def check_unbiased(where: str, unbiased) -> None:
    """A Python ``bool``: it selects a divisor, and so a static structure."""
    if not isinstance(unbiased, bool):
        raise TypeError(
            f"{where}: unbiased must be a Python bool, got "
            f"{type(unbiased).__module__}.{type(unbiased).__name__}"
        )


def particle_divisor(n_particles: int, unbiased: bool) -> int:
    """The divisor of an unweighted covariance: ``J - 1`` if unbiased, else ``J``."""
    return n_particles - 1 if unbiased else n_particles


def check_value(where: str, name: str, value, dim: int, dtype):
    """A block value: exactly ``(dim,)``, real, converted to ``dtype``."""
    arr = jnp.asarray(value)
    check_real(where, name, arr)
    if arr.shape != (dim,):
        raise ValueError(
            f"{where}: the value of block {name!r} must have shape ({dim},), got "
            f"{arr.shape}. A family of values is a jax.vmap over this call."
        )
    return arr.astype(dtype)


def check_real(where: str, name: str, arr) -> None:
    """Refuse a complex or boolean array."""
    if not (
        jnp.issubdtype(arr.dtype, jnp.floating) or jnp.issubdtype(arr.dtype, jnp.integer)
    ):
        raise TypeError(f"{where}: {name} must be real, got dtype {arr.dtype}")


def check_finite(where: str, what: str, value, hint: str = "") -> None:
    """Debug-mode check that an array is finite; skipped under a trace."""
    message = f"{where}: {what} must be finite"
    value_check(
        value,
        lambda a: bool(jnp.all(jnp.isfinite(a))),
        f"{message}. {hint}" if hint else message,
    )


def lazy_value_check(x, predicate, message) -> None:
    """:func:`~enskit.linalg.value_check` with a message built only on failure.

    For messages that read array values themselves, which must not be
    computed under a trace. ``predicate`` returns an array comparison.
    """
    failed = []

    def check(a):
        outcome = predicate(a)
        if isinstance(outcome, jax.core.Tracer):
            return outcome
        if not bool(outcome):
            failed.append(True)
        return True

    value_check(x, check, "")
    if failed:
        raise ValueError(message())


def whitening_check(where: str, block: str, value) -> None:
    """The whitening check: the whitened factor and residuals are finite."""
    value_check(
        value,
        lambda a: bool(jnp.all(jnp.isfinite(a))),
        f"{where}: the whitened factor or residual of given block {block!r} is "
        f"not finite. {WHITENING_CAUSE}",
    )


def result_check(where: str, what: str, value) -> None:
    """The result check: a conditioning result is finite."""
    value_check(
        value,
        lambda a: bool(jnp.all(jnp.isfinite(a))),
        f"{where}: the {what} must be finite. {RESULT_CAUSE}",
    )


def safe_repr(build_text, type_name: str, batch_shape) -> str:
    """Render a repr, wrapping a family; a marker form if sizes are unreadable."""
    try:
        base = build_text()
        batch = batch_shape()
    except Exception:
        return f"<{type_name} (unprintable leaves)>"
    return f"vmapped({base}, batch={batch})" if batch != () else base
