"""
Unit tests for pysatl_core.sampling.unuran.core._unuran_sampler.dgt

Tests the DGTSetup class that prepares UNU.RAN DGT (Discrete Generation Table)
method requirements: domain inference, PMF normalization, and PV construction.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import contextlib
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import (
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.sampling.unuran.core._unuran_sampler.dgt import DGTSetup
from pysatl_core.sampling.unuran.core._unuran_sampler.domain import UnuranDomain
from pysatl_core.types import CharacteristicName, Kind
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


class DGTTestBase:
    @staticmethod
    def _make_discrete_distr_with_pmf(
        pmf_values: dict[int, float],
        support: Any = None,
    ) -> StandaloneEuclideanUnivariateDistribution:
        """Create a discrete distribution with a PMF defined by a dictionary of {k: probability}."""

        def pmf(x: float) -> float:
            return pmf_values.get(int(x), 0.0)

        from collections.abc import Callable
        from typing import Any

        from mypy_extensions import KwArg

        pmf_func = cast(Callable[[float, KwArg(Any)], float], pmf)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations=[
                AnalyticalComputation[float, float](target=CharacteristicName.PMF, func=pmf_func)
            ],
            support=support,
        )

    @staticmethod
    def _make_dgt_setup(
        distr: StandaloneEuclideanUnivariateDistribution,
        lib_overrides: dict[str, Any] | None = None,
    ) -> tuple[DGTSetup, MagicMock, MagicMock]:
        """Build a DGTSetup with mock CFFI handles."""

        class _NULL:
            """Sentinel NULL handle."""

        mock_lib = MagicMock()
        mock_lib.unur_distr_discr_set_domain.return_value = 0
        mock_lib.unur_distr_discr_make_pv.return_value = 5
        mock_lib.unur_get_errno.return_value = 0
        if lib_overrides:
            for attr, val in lib_overrides.items():
                setattr(mock_lib, attr, val)

        mock_ffi = MagicMock()
        mock_ffi.NULL = _NULL()
        mock_ffi.string.return_value = b"error"

        unuran_distr = object()
        domain = UnuranDomain(distr.support)
        pmf = None
        with contextlib.suppress(RuntimeError):
            pmf = distr.query_method(CharacteristicName.PMF)
        setup = DGTSetup(mock_lib, mock_ffi, domain, unuran_distr, distr.support, pmf)
        return setup, mock_lib, mock_ffi


class TestRequireAttr(DGTTestBase):
    """Tests for DGTSetup._require_attr."""

    def test_present_attribute_does_not_raise(self) -> None:
        """_require_attr passes silently when the library exposes the required function."""
        masses = {0: 0.5, 1: 0.5}
        support = ExplicitTableDiscreteSupport([0, 1])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr)

        mock_lib.some_function = MagicMock()

        # Should not raise
        setup._require_attr("some_function")

    def test_missing_attribute_raises_runtime_error(self) -> None:
        """_require_attr raises RuntimeError when the required function is absent."""
        masses = {0: 1.0}
        support = ExplicitTableDiscreteSupport([0])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr)

        # Remove the attribute from mock
        del mock_lib.nonexistent_function

        with pytest.raises(RuntimeError, match="nonexistent_function"):
            setup._require_attr("nonexistent_function")


class TestDomainPoints(DGTTestBase):
    """Tests for DGTSetup._domain_points."""

    def test_dense_range_for_plain_support(self) -> None:
        """A plain (non-sparse) support produces a dense integer range."""
        masses = dict.fromkeys(range(5), 0.1)
        distr = self._make_discrete_distr_with_pmf(masses)
        setup, _, _ = self._make_dgt_setup(distr)

        result = list(setup._domain_points(0, 4))

        assert result == [0, 1, 2, 3, 4]

    def test_explicit_table_support_returns_all_points(self) -> None:
        """ExplicitTableDiscreteSupport returns all support points regardless of domain bounds."""
        masses = dict.fromkeys([0, 2, 5, 7, 10], 0.1)
        support = ExplicitTableDiscreteSupport([0, 2, 5, 7, 10])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, _, _ = self._make_dgt_setup(distr)

        result = list(setup._domain_points(0, 4))  # index domain [0, n-1]

        assert list(result) == [0, 2, 5, 7, 10]

    def test_integer_lattice_with_modulus_gt_1_steps_correctly(self) -> None:
        """Lattice support with modulus > 1 steps by the modulus, starting at residue."""
        masses = dict.fromkeys([0, 2, 4, 6, 8], 1 / 5)
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=2, min_k=0, max_k=8)
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, _, _ = self._make_dgt_setup(distr)

        result = list(setup._domain_points(0, 8))

        assert result == [0, 2, 4, 6, 8]

    def test_explicit_table_always_returns_all_points_ignoring_bounds(self) -> None:
        """ExplicitTableDiscreteSupport ignores domain bounds and returns all points."""
        masses = dict.fromkeys([10, 20], 0.5)
        support = ExplicitTableDiscreteSupport([10, 20])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, _, _ = self._make_dgt_setup(distr)

        result = list(setup._domain_points(0, 1))  # index domain [0, n-1]

        assert list(result) == [10, 20]


class TestCalculatePmfSum(DGTTestBase):
    """Tests for DGTSetup._calculate_pmf_sum."""

    def test_pmf_sum_is_correct_for_simple_distribution(self) -> None:
        """PMF sum over [0, 2] equals the sum of the defined masses."""
        masses = {0: 0.2, 1: 0.5, 2: 0.3}
        support = ExplicitTableDiscreteSupport([0, 1, 2])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, _, _ = self._make_dgt_setup(distr)

        result = setup._calculate_pmf_sum(0, 2)

        assert result == pytest.approx(1.0)

    def test_pmf_sum_excludes_zero_probability_points(self) -> None:
        """Points outside the support contribute zero to the PMF sum."""
        masses = {0: 0.4, 1: 0.6}
        support = ExplicitTableDiscreteSupport([0, 1])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, _, _ = self._make_dgt_setup(distr)

        # Range [0, 5] but only 0 and 1 have nonzero mass
        result = setup._calculate_pmf_sum(0, 1)

        assert result == pytest.approx(1.0)

    def test_raises_if_pmf_unavailable(self) -> None:
        """_calculate_pmf_sum raises RuntimeError when PMF is not registered."""
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations=[],  # No PMF
            support=None,
        )
        setup, _, _ = self._make_dgt_setup(distr)

        with pytest.raises(RuntimeError, match="PMF is unavailable"):
            setup._calculate_pmf_sum(0, 5)


class TestSetupDgtMethod(DGTTestBase):
    """Tests for DGTSetup.setup_dgt_method."""

    def test_setup_succeeds_with_explicit_integer_support(self) -> None:
        """setup_dgt_method completes without error for a proper bounded integer support."""
        masses = {0: 0.2, 1: 0.5, 2: 0.3}
        support = ExplicitTableDiscreteSupport([0, 1, 2])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr)

        # Should not raise
        setup.setup_dgt_method()

        mock_lib.unur_distr_discr_set_domain.assert_called_once_with(setup._unuran_distr, 0, 2)

    def test_raises_when_domain_cannot_be_determined(self) -> None:
        """setup_dgt_method raises RuntimeError when the distribution has no discrete domain."""
        masses = {0: 1.0}
        distr = self._make_discrete_distr_with_pmf(masses, support=None)
        setup, _, _ = self._make_dgt_setup(distr)

        with pytest.raises(RuntimeError, match="domain"):
            setup.setup_dgt_method()

    def test_raises_when_domain_has_no_right_bound(self) -> None:
        """setup_dgt_method raises RuntimeError when the domain right bound is None."""
        masses = dict.fromkeys(range(10), 0.1)
        # Only left bound, no right bound
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=1, min_k=0)
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, _, _ = self._make_dgt_setup(distr)

        with pytest.raises(RuntimeError, match="domain"):
            setup.setup_dgt_method()

    def test_raises_when_set_domain_returns_nonzero(self) -> None:
        """setup_dgt_method raises RuntimeError if unur_distr_discr_set_domain fails."""
        masses = {0: 0.5, 1: 0.5}
        support = ExplicitTableDiscreteSupport([0, 1])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr, lib_overrides={})
        mock_lib.unur_distr_discr_set_domain.return_value = -1

        with pytest.raises(RuntimeError, match="domain"):
            setup.setup_dgt_method()

    def test_raises_when_make_pv_returns_nonpositive(self) -> None:
        """setup_dgt_method raises RuntimeError if make_pv returns a non-positive value."""
        masses = {0: 0.5, 1: 0.5}
        support = ExplicitTableDiscreteSupport([0, 1])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr)
        mock_lib.unur_distr_discr_make_pv.return_value = 0

        with pytest.raises(RuntimeError, match="PV"):
            setup.setup_dgt_method()

    def test_raises_when_required_cffi_function_missing(self) -> None:
        """setup_dgt_method raises RuntimeError if a required CFFI binding is absent."""
        masses = {0: 1.0}
        support = ExplicitTableDiscreteSupport([0])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr)
        del mock_lib.unur_distr_discr_set_domain

        with pytest.raises(RuntimeError, match="unur_distr_discr_set_domain"):
            setup.setup_dgt_method()

    def test_make_pv_called_after_set_domain(self) -> None:
        """unur_distr_discr_make_pv is invoked after a successful set_domain call."""
        masses = {0: 0.4, 1: 0.6}
        support = ExplicitTableDiscreteSupport([0, 1])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr)

        setup.setup_dgt_method()

        mock_lib.unur_distr_discr_make_pv.assert_called_once_with(setup._unuran_distr)

    def test_pmf_sum_is_set_when_function_available(self) -> None:
        """PMF sum is registered via unur_distr_discr_set_pmfsum when available."""
        masses = {0: 0.3, 1: 0.7}
        support = ExplicitTableDiscreteSupport([0, 1])
        distr = self._make_discrete_distr_with_pmf(masses, support=support)
        setup, mock_lib, _ = self._make_dgt_setup(distr)
        mock_lib.unur_distr_discr_set_pmfsum = MagicMock()

        setup.setup_dgt_method()

        mock_lib.unur_distr_discr_set_pmfsum.assert_called_once()
