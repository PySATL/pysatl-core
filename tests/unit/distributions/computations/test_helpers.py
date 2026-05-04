"""
Tests for shared fitter helpers: resolve, collect_discrete_support, build_tail_table,
estimate_support_bounds.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
import pytest

from pysatl_core.distributions.computations._utils import (
    build_head_table,
    build_tail_table,
    collect_discrete_support,
    estimate_support_bounds,
)
from pysatl_core.distributions.support import (
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)

# ===================================================================
# collect_discrete_support
# ===================================================================


class TestCollectSupport:
    """Tests for the collect_discrete_support helper."""

    def test_explicit_table_support(self) -> None:
        support = ExplicitTableDiscreteSupport([3.0, 1.0, 2.0])
        xs = collect_discrete_support(support)
        np.testing.assert_array_equal(xs, [1.0, 2.0, 3.0])
        assert xs.dtype == float

    def test_integer_lattice_bounded(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=5, modulus=1, residue=0)
        xs = collect_discrete_support(support)
        np.testing.assert_array_equal(xs, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    def test_integer_lattice_with_modulus(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=10, modulus=2, residue=0)
        xs = collect_discrete_support(support)
        np.testing.assert_array_equal(xs, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    def test_left_unbounded_lattice_raises(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=5, modulus=1, residue=0)
        with pytest.raises(RuntimeError, match="Left-unbounded"):
            collect_discrete_support(support)

    def test_right_unbounded_lattice_raises(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=1, residue=0)
        with pytest.raises(RuntimeError, match="Right-unbounded"):
            collect_discrete_support(support)

    def test_fully_unbounded_lattice_raises(self) -> None:
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=None, modulus=1, residue=0)
        with pytest.raises(RuntimeError, match="unbounded"):
            collect_discrete_support(support)


# ===================================================================
# build_tail_table
# ===================================================================


class TestBuildTailTable:
    """Tests for the build_tail_table helper."""

    def test_uniform_pmf_tail_probabilities(self) -> None:
        """Uniform PMF on [0, 4]: tail_from[i] = (5 - i) / 5."""
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=4, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 0.2)

        xs, tail_from = build_tail_table(support, pmf)

        np.testing.assert_array_equal(xs, [0.0, 1.0, 2.0, 3.0, 4.0])
        assert tail_from.shape == (6,)
        assert float(tail_from[0]) == pytest.approx(1.0)
        assert float(tail_from[-1]) == pytest.approx(0.0)
        # Monotonically non-increasing
        assert np.all(np.diff(tail_from) <= 0)

    def test_tail_from_first_element_is_one(self) -> None:
        """tail_from[0] must always equal 1.0 (normalisation)."""
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=3, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            # Unnormalised: sum = 4 * 2.0 = 8.0
            return np.full_like(x, 2.0)

        _, tail_from = build_tail_table(support, pmf)
        assert float(tail_from[0]) == pytest.approx(1.0)

    def test_zero_pmf_returns_valid_tail(self) -> None:
        """When PMF is identically zero the tail table must still be valid
        (all values in [0, 1]) and not raise."""
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=2, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x)

        xs, tail_from = build_tail_table(support, pmf, eps=1e-12, max_batches=4)
        assert np.all(tail_from >= 0.0)
        assert np.all(tail_from <= 1.0)

    def test_left_unbounded_mass_is_captured(self) -> None:
        """
        The key correctness test: a PMF that places mass far to the left of
        ``max_k`` must be captured by the downward walk.

        Distribution: P(X = k) = 0.5 for k in {-10, 10}, 0 elsewhere.
        CDF(x) should be 0.5 for x in [-10, 9] and 1.0 for x >= 10.
        The old algorithm (starting from residue=0) would miss the mass at -10
        and return CDF(x) = 0 for x < 0.
        """
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=10, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            result[np.isclose(x, -10.0)] = 0.5
            result[np.isclose(x, 10.0)] = 0.5
            return result

        xs, tail_from = build_tail_table(support, pmf)

        # tail_from[i] = P(X >= xs[i])
        # P(X >= -10) = 1.0, P(X >= 10) = 0.5, P(X >= 11) = 0.0
        idx_minus10 = int(np.searchsorted(xs, -10.0))
        idx_10 = int(np.searchsorted(xs, 10.0))

        assert float(tail_from[idx_minus10]) == pytest.approx(1.0, abs=1e-9)
        assert float(tail_from[idx_10]) == pytest.approx(0.5, abs=1e-9)
        assert float(tail_from[-1]) == pytest.approx(0.0, abs=1e-9)

    def test_single_point_lattice(self) -> None:
        """Single-point lattice: xs = [residue], tail_from = [1.0, 0.0]."""
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=3, modulus=1, residue=3)

        def pmf(x: np.ndarray) -> np.ndarray:
            return np.ones_like(x)

        xs, tail_from = build_tail_table(support, pmf)
        np.testing.assert_array_equal(xs, [3.0])
        assert float(tail_from[0]) == pytest.approx(1.0)
        assert float(tail_from[-1]) == pytest.approx(0.0)

    def test_modulus_greater_than_one(self) -> None:
        """Lattice with modulus=2: xs = [0, 2, 4]."""
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=4, modulus=2, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 1.0 / 3.0)

        xs, tail_from = build_tail_table(support, pmf)
        np.testing.assert_array_equal(xs, [0.0, 2.0, 4.0])
        assert float(tail_from[0]) == pytest.approx(1.0)
        assert float(tail_from[-1]) == pytest.approx(0.0)

    def test_negative_pmf_values_clipped_to_zero(self) -> None:
        """Negative PMF values must be clipped to 0 before summation."""
        support = IntegerLatticeDiscreteSupport(min_k=None, max_k=2, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            # Returns -1 for all points — should be treated as 0
            return np.full_like(x, -1.0)

        xs, tail_from = build_tail_table(support, pmf)
        # All PMF values are 0 after clipping → total == 0 → no normalisation
        # tail_from should be all zeros (clipped)
        assert np.all(tail_from >= 0.0)
        assert np.all(tail_from <= 1.0)


# ===================================================================
# build_head_table
# ===================================================================


class TestBuildHeadTable:
    """Tests for the build_head_table helper (left-bounded, right-unbounded lattice)."""

    def test_uniform_pmf_cdf_values(self) -> None:
        """Uniform PMF on [0, 4]: cdf_at[i] = (i + 1) / 5."""
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            mask = (x >= 0) & (x <= 4)
            result[mask] = 0.2
            return result

        xs, cdf_at = build_head_table(support, pmf)

        np.testing.assert_array_equal(xs, [0.0, 1.0, 2.0, 3.0, 4.0])
        assert cdf_at.shape == (5,)
        assert float(cdf_at[0]) == pytest.approx(0.2, abs=1e-9)
        assert float(cdf_at[-1]) == pytest.approx(1.0, abs=1e-9)
        # Monotonically non-decreasing
        assert np.all(np.diff(cdf_at) >= 0)

    def test_cdf_at_last_element_is_one(self) -> None:
        """cdf_at[-1] must equal 1.0 after normalisation."""
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            mask = (x >= 0) & (x <= 3)
            result[mask] = 2.0  # unnormalised
            return result

        _, cdf_at = build_head_table(support, pmf)
        assert float(cdf_at[-1]) == pytest.approx(1.0)

    def test_right_unbounded_mass_is_captured(self) -> None:
        """
        The key correctness test: a PMF that places mass far to the right of
        ``min_k`` must be captured by the upward walk.

        Distribution: P(X = k) = 0.5 for k in {0, 100}, 0 elsewhere.
        CDF(x) should be 0.5 for x in [0, 99] and 1.0 for x >= 100.
        """
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            result[np.isclose(x, 0.0)] = 0.5
            result[np.isclose(x, 100.0)] = 0.5
            return result

        xs, cdf_at = build_head_table(support, pmf)

        idx_0 = int(np.searchsorted(xs, 0.0))
        idx_100 = int(np.searchsorted(xs, 100.0))

        assert float(cdf_at[idx_0]) == pytest.approx(0.5, abs=1e-9)
        assert float(cdf_at[idx_100]) == pytest.approx(1.0, abs=1e-9)

    def test_modulus_greater_than_one(self) -> None:
        """Lattice with modulus=2, min_k=0: xs = [0, 2, 4]."""
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=2, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            mask = (x >= 0) & (x <= 4)
            result[mask] = 1.0 / 3.0
            return result

        xs, cdf_at = build_head_table(support, pmf)
        np.testing.assert_array_equal(xs, [0.0, 2.0, 4.0])
        assert float(cdf_at[0]) == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert float(cdf_at[-1]) == pytest.approx(1.0, abs=1e-9)

    def test_zero_pmf_returns_valid_cdf(self) -> None:
        """When PMF is identically zero the CDF table must still be valid."""
        support = IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=1, residue=0)

        def pmf(x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x)

        xs, cdf_at = build_head_table(support, pmf, eps=1e-12, max_batches=4)
        assert np.all(cdf_at >= 0.0)
        assert np.all(cdf_at <= 1.0)

    def test_symmetric_with_tail_table(self) -> None:
        """
        For a symmetric PMF around 0, build_head_table on [0, ∞) and
        build_tail_table on (-∞, 0] should produce mirror-image results.
        """
        # PMF: 0.5 at 0, 0.5 at 5 (right-unbounded from 0)
        support_head = IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=1, residue=0)
        # Mirror: PMF: 0.5 at 0, 0.5 at -5 (left-unbounded up to 0)
        support_tail = IntegerLatticeDiscreteSupport(min_k=None, max_k=0, modulus=1, residue=0)

        def pmf_head(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            result[np.isclose(x, 0.0)] = 0.5
            result[np.isclose(x, 5.0)] = 0.5
            return result

        def pmf_tail(x: np.ndarray) -> np.ndarray:
            result = np.zeros_like(x, dtype=float)
            result[np.isclose(x, 0.0)] = 0.5
            result[np.isclose(x, -5.0)] = 0.5
            return result

        xs_h, cdf_h = build_head_table(support_head, pmf_head)
        xs_t, tail_t = build_tail_table(support_tail, pmf_tail)

        # cdf_h[i] = P(X <= xs_h[i]); tail_t[i] = P(X' >= xs_t[i])
        # For the mirrored distribution: P(X <= k) = P(X' >= -k)
        # xs_h = [0, 5], cdf_h = [0.5, 1.0]
        # xs_t = [-5, 0], tail_t = [1.0, 0.5, 0.0]
        assert float(cdf_h[0]) == pytest.approx(float(tail_t[1]), abs=1e-9)  # P(X<=0) = P(X'>=-0)
        assert float(cdf_h[-1]) == pytest.approx(1.0, abs=1e-9)


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
