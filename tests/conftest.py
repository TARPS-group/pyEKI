"""Helpers shared across the test suite.

Import them by name — ``from conftest import prints_as``. Only what more than
one test module needs belongs here; a reference implementation a single module
checks against stays in that module, where the package's conformance rules
want it.
"""
from __future__ import annotations

import numpy as np


def prints_as(got, want, decimals: int = 4) -> None:
    """Assert a value's printed digits, which is what a documentation block shows.

    Pinning with a tolerance invites a window nobody measured: an ``atol`` of
    half the printed precision leaves whatever slack the rounding happens to
    give, which for one value in the toy-models page was 0.25% of the stated
    window. Rounding to the printed precision and comparing exactly says what
    the documentation actually claims.

    Parameters
    ----------
    got
        The value the code produces. Converted with :func:`numpy.asarray`.
    want
        The digits the documentation shows, at ``decimals`` places.
    decimals
        The number of decimal places the documentation prints.

    Raises
    ------
    AssertionError
        If the shapes differ, or any rounded value differs.
    """
    got = np.round(np.asarray(got, dtype=float), decimals)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    assert np.array_equal(got, want), f"prints as {got}, documented as {want}"
