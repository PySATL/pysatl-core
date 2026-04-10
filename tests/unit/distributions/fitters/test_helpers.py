"""
Tests for shared fitter helpers: resolve, collect_support, build_tail_table,
estimate_support_bounds, maybe_unwrap_scalar.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
import pytest

from pysatl_core.distributions.fitters.helpers import (
    collect_support,
    estimate_support_bounds,
    maybe_unwrap_scalar,
)
from pysatl_core.distributions.support import (
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)

# ===================================================================
# maybe_unwrap_scalar
# ===================================================================


class TestMaybeUnwrapScalar:
    """Tests for the maybe_unwrap_scalar helper."""

    def test_scalar_from_single_element(self) -> None:
        arr = np.array([42.0])
        result = maybe_unwrap_scalar(arr)
        assert np.ndim(result) == 0
        assert float(result) == 42.0

    def test_array_from_multi_element(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = maybe_unwrap_scalar(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_empty_array_returned_as_is(self) -> None:
        arr = np.array([], dtype=float)
        result = maybe_unwrap_scalar(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)


# ===================================================================
# collect_support
# ===================================================================


class TestCollectSupport:
    """Tests for the collect_support helper."""

    def test_explicit_table_support(self) -> None:
        support = ExplicitTableDiscreteSupport([3.0, 1.0, 2.0])
        xs = collect_support(support)
        np.testing.assert_array_equal(xs, [1.0, 2.0, 3.0])
        assert xs.dtype == float

    def test_integer_lattice_bounded(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=5, modulus=1, residue=0)
        xs = collect_support(support)
        np.testing.assert_array_equal(xs, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_integer_lattice_with_modulus(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=10, modulus=2, residue=0)
        xs = collect_support(support)
        np.testing.assert_array_equal(xs, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    def test_left_unbounded_lattice_raises(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=5, modulus=1, residue=0)
        with pytest.raises(RuntimeError, match="Left-unbounded"):
            collect_support(support)

    def test_fully_unbounded_lattice_raises(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=None, modulus=1, residue=0)
        with pytest.raises(RuntimeError, match="unbounded"):
            collect_support(support)


# ===================================================================
# estimate_support_bounds
# ===================================================================


class TestEstimateSupportBounds:
    """Tests for the estimate_support_bounds helper."""

    def test_standard_normal_cdf_bounds(self) -> None:
        """Bounds for a standard normal CDF should be roughly [-6, 6]."""
        from scipy.stats import norm

        def cdf_func(x: np.ndarray, **kwargs: Any) -> np.ndarray:
            return norm.cdf(x)

        lo, hi = estimate_support_bounds(cdf_func, eps=1e-6, x0=0.0)
        assert lo < -4.0
        assert hi > 4.0

    def test_uniform_cdf_bounds(self) -> None:
        """Bounds for Uniform[0,1] CDF should bracket [0, 1]."""

        def cdf_func(x: np.ndarray, **kwargs: Any) -> np.ndarray:
            return np.clip(x, 0.0, 1.0)

        lo, hi = estimate_support_bounds(cdf_func, eps=1e-6, x0=0.5)
        assert lo <= 0.0
        assert hi >= 1.0

    def test_custom_starting_point(self) -> None:
        """Starting from x0=10 should still find bounds for standard normal."""
        from scipy.stats import norm

        def cdf_func(x: np.ndarray, **kwargs: Any) -> np.ndarray:
            return norm.cdf(x)

        lo, hi = estimate_support_bounds(cdf_func, eps=1e-6, x0=10.0)
        assert lo < -4.0
        assert hi >= 10.0
