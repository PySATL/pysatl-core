"""
Unit tests for pysatl_core.sampling.unuran.core._unuran_sampler.domain

Tests the UnuranDomain class which derives integer and continuous bounds
from a distribution's support for use during UNU.RAN setup.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from math import inf
from unittest.mock import MagicMock

import numpy as np
import pytest

from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.sampling.unuran.core._unuran_sampler.domain import UnuranDomain


class TestUnuranDomainDiscreteInteger:
    """Tests for determine_discrete_domain with integer-valued supports."""

    def test_returns_none_when_support_is_none(self) -> None:
        """When the distribution has no support, discrete domain is None."""
        domain = UnuranDomain(None)

        result = domain.determine_discrete_domain()

        assert result is None

    def test_explicit_integer_support_returns_index_domain(self) -> None:
        """ExplicitTableDiscreteSupport with integer points returns index domain (0, n-1)."""
        support = ExplicitTableDiscreteSupport([1, 2, 3])
        domain = UnuranDomain(support)

        result = domain.determine_discrete_domain()

        assert result == (0, 2)

    def test_explicit_integer_support_single_point_returns_zero_zero(self) -> None:
        """ExplicitTableDiscreteSupport with a single point returns index domain (0, 0)."""
        support = ExplicitTableDiscreteSupport([5])
        domain = UnuranDomain(support)

        result = domain.determine_discrete_domain()

        assert result == (0, 0)

    def test_explicit_float_support_returns_index_domain(self) -> None:
        """ExplicitTableDiscreteSupport backed by floats returns index domain (0, n-1)."""
        points = np.array([0.1, 0.5, 0.9], dtype=float)
        mock_support = MagicMock(spec=ExplicitTableDiscreteSupport)
        mock_support.points = points
        domain = UnuranDomain(mock_support)

        result = domain.determine_discrete_domain()

        assert result == (0, 2)

    def test_explicit_empty_support_returns_none(self) -> None:
        """ExplicitTableDiscreteSupport with an empty point array returns None."""
        points = np.array([], dtype=int)
        mock_support = MagicMock(spec=ExplicitTableDiscreteSupport)
        mock_support.points = points
        domain = UnuranDomain(mock_support)

        result = domain.determine_discrete_domain()

        assert result is None

    def test_integer_lattice_with_both_bounds_returns_first_and_last(self) -> None:
        """IntegerLatticeDiscreteSupport with both bounds returns (first, last)."""
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=1, min_k=0, max_k=10)
        domain = UnuranDomain(support)

        result = domain.determine_discrete_domain()

        assert result == (0, 10)

    def test_integer_lattice_with_only_left_bound_returns_none(self) -> None:
        """IntegerLatticeDiscreteSupport with only left bound (unbounded right) returns None."""
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=1, min_k=5)
        domain = UnuranDomain(support)

        result = domain.determine_discrete_domain()

        assert result is None

    def test_integer_lattice_unbounded_returns_none(self) -> None:
        """IntegerLatticeDiscreteSupport with no bounds returns None."""
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=1)
        domain = UnuranDomain(support)

        result = domain.determine_discrete_domain()

        assert result is None

    def test_integer_lattice_only_right_bound_returns_none(self) -> None:
        """IntegerLatticeDiscreteSupport with only right bound returns None (no first)."""
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=1, max_k=10)
        domain = UnuranDomain(support)

        result = domain.determine_discrete_domain()

        assert result is None

    def test_unknown_support_type_returns_none(self) -> None:
        """An unrecognized support type returns None (no domain can be inferred)."""
        domain = UnuranDomain(MagicMock())

        result = domain.determine_discrete_domain()

        assert result is None


class TestUnuranDomainExplicitTablePoints:
    """Tests for explicit_table_points helper."""

    def test_returns_points_for_explicit_table_support(self) -> None:
        """Returns the points array for ExplicitTableDiscreteSupport."""
        support = ExplicitTableDiscreteSupport([1, 5, 10])
        domain = UnuranDomain(support)

        result = domain.explicit_table_points()

        assert result is not None
        np.testing.assert_array_equal(result, [1, 5, 10])

    def test_returns_points_for_float_support(self) -> None:
        """Returns float points array for non-integer ExplicitTableDiscreteSupport."""
        points = np.array([0.1, 0.5, 0.9])
        mock_support = MagicMock(spec=ExplicitTableDiscreteSupport)
        mock_support.points = points
        domain = UnuranDomain(mock_support)

        result = domain.explicit_table_points()

        assert result is not None
        np.testing.assert_array_equal(result, points)

    def test_returns_none_for_empty_explicit_table(self) -> None:
        """Returns None when ExplicitTableDiscreteSupport has no points."""
        points = np.array([], dtype=int)
        mock_support = MagicMock(spec=ExplicitTableDiscreteSupport)
        mock_support.points = points
        domain = UnuranDomain(mock_support)

        assert domain.explicit_table_points() is None

    def test_returns_none_for_non_explicit_table_support(self) -> None:
        """Returns None for IntegerLatticeDiscreteSupport."""
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=1, min_k=0, max_k=5)
        domain = UnuranDomain(support)

        assert domain.explicit_table_points() is None

    def test_returns_none_when_support_is_none(self) -> None:
        """Returns None when support is None."""
        domain = UnuranDomain(None)

        assert domain.explicit_table_points() is None


class TestUnuranDomainContinuous:
    """Tests for determine_continuous_domain with continuous supports."""

    def test_returns_none_when_support_is_none(self) -> None:
        """When the distribution has no support, continuous domain is None."""
        domain = UnuranDomain(None)

        result = domain.determine_continuous_domain()

        assert result is None

    def test_bounded_continuous_support_returns_left_and_right(self) -> None:
        """ContinuousSupport with finite bounds returns those bounds as floats."""
        support = ContinuousSupport(left=0.0, right=1.0)
        domain = UnuranDomain(support)

        result = domain.determine_continuous_domain()

        assert result == (0.0, 1.0)

    def test_unbounded_continuous_support_returns_inf_bounds(self) -> None:
        """ContinuousSupport() (real line) returns (-inf, inf) since both are floats."""
        support = ContinuousSupport()
        domain = UnuranDomain(support)

        result = domain.determine_continuous_domain()

        assert result == (-inf, inf)

    def test_support_without_left_or_right_returns_none(self) -> None:
        """Support without 'left'/'right' attributes returns None."""
        mock_support = MagicMock()
        del mock_support.left
        del mock_support.right
        domain = UnuranDomain(mock_support)

        result = domain.determine_continuous_domain()

        assert result is None

    def test_support_with_nonnumeric_bounds_returns_none(self) -> None:
        """Support with non-numeric bounds (e.g. strings) returns None."""
        mock_support = MagicMock()
        mock_support.left = "a"
        mock_support.right = "b"
        domain = UnuranDomain(mock_support)

        result = domain.determine_continuous_domain()

        assert result is None

    def test_integer_bounds_are_cast_to_float(self) -> None:
        """Integer left/right bounds are accepted and returned as floats."""
        mock_support = MagicMock()
        mock_support.left = 0
        mock_support.right = 5
        domain = UnuranDomain(mock_support)

        result = domain.determine_continuous_domain()

        assert result == (0.0, 5.0)
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_numpy_float64_bounds_are_accepted(self) -> None:
        """np.float64 bounds pass the numeric check (np.float64 is a subclass of float)."""
        import numpy as np

        mock_support = MagicMock()
        mock_support.left = np.float64(-1.5)
        mock_support.right = np.float64(3.5)
        domain = UnuranDomain(mock_support)

        result = domain.determine_continuous_domain()

        assert result == (-1.5, 3.5)
        assert isinstance(result[0], float)

    def test_numpy_int64_bounds_are_accepted(self) -> None:
        """np.int64 bounds pass the numeric check via numbers.Real ABC."""
        import numpy as np

        mock_support = MagicMock()
        mock_support.left = np.int64(0)
        mock_support.right = np.int64(10)
        domain = UnuranDomain(mock_support)

        result = domain.determine_continuous_domain()

        assert result == (0.0, 10.0)
        assert isinstance(result[0], float)

    @pytest.mark.parametrize("left,right", [(-3.5, 7.2), (0.0, 100.0), (-inf, 0.0)])
    def test_various_numeric_bounds_are_returned_correctly(self, left: float, right: float) -> None:
        """Various numeric bound combinations are returned unchanged (as floats)."""
        mock_support = MagicMock()
        mock_support.left = left
        mock_support.right = right
        domain = UnuranDomain(mock_support)

        result = domain.determine_continuous_domain()

        assert result == (float(left), float(right))
