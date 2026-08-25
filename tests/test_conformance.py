"""Conformance tests: every operator type, through the full contract suite.

One instance per operator class (plus variants whose capabilities differ)
runs through :func:`pyeki.linalg.testing.check_operator`, which verifies the
thirteen checks of the linear operator contract.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pyeki  # noqa: F401  -- enables x64 before any array exists
from pyeki.linalg import (
    BlockDiag,
    Dense,
    DensePSD,
    DenseSquare,
    Diagonal,
    Identity,
    LinOp,
    ScaledIdentity,
    Triangular,
    block_diag,
    diag_congruence,
    hstack,
    product,
)
from pyeki.linalg.testing import check_operator

RNG = np.random.default_rng(0)


def _psd(n: int) -> jnp.ndarray:
    M = RNG.normal(size=(n, n))
    return jnp.asarray(M @ M.T + n * np.eye(n))


def _instances() -> list[LinOp]:
    d = jnp.asarray(RNG.uniform(0.5, 3.0, 6))
    well_conditioned = jnp.asarray(RNG.normal(size=(5, 5)) + 5 * np.eye(5))
    return [
        Identity(6),
        ScaledIdentity(jnp.asarray(2.5), 6),
        Diagonal(d),
        Dense(jnp.asarray(RNG.normal(size=(4, 6)))),
        DenseSquare.from_matrix(well_conditioned),
        Triangular(jnp.linalg.cholesky(_psd(5)), lower=True),
        Triangular(jnp.linalg.cholesky(_psd(4)).T, lower=False),
        DensePSD.from_matrix(_psd(5)),
        # composites
        product(Diagonal(d), Dense(jnp.asarray(RNG.normal(size=(6, 4))))),
        hstack(
            Dense(jnp.asarray(RNG.normal(size=(5, 2)))), DensePSD.from_matrix(_psd(5))
        ),
        BlockDiag(
            (
                Dense(jnp.asarray(RNG.normal(size=(2, 3)))),
                Dense(jnp.asarray(RNG.normal(size=(3, 3)))),
            )
        ),
        block_diag(Diagonal(d), DensePSD.from_matrix(_psd(3))),
        block_diag(Identity(2), ScaledIdentity(jnp.asarray(4.0), 3)),
        diag_congruence(
            DensePSD.from_matrix(_psd(4)), jnp.asarray(RNG.uniform(0.5, 2, 4))
        ),
        # arithmetic-built composites
        2.0 * DensePSD.from_matrix(_psd(4)),
        3.0 * DenseSquare.from_matrix(well_conditioned),
        1.5 * Dense(jnp.asarray(RNG.normal(size=(3, 5)))),
        Dense(jnp.asarray(RNG.normal(size=(3, 5)))).T,
    ]


@pytest.mark.parametrize("op", _instances(), ids=lambda o: type(o).__name__)
def test_conformance(op):
    check_operator(op)
