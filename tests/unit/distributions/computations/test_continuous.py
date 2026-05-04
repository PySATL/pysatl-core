"""
Tests for continuous-distribution fitters (1C).

Each test verifies:
- Correctness against a known analytical distribution (Uniform[0,1] or Logistic).
- Array semantics: any input → array out.
- Option effects (e.g. changing ``limit`` or ``h``).
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest

from pysatl_core.distributions.computations.continuous import (
    _build_cdf_to_pdf_1C,
    _build_cdf_to_ppf_1C,
    _build_pdf_to_cdf_1C,
    _build_ppf_to_cdf_1C,
    _fit_cdf_to_pdf_1C,
    _fit_cdf_to_ppf_1C,
    _fit_pdf_to_cdf_1C,
    _fit_ppf_to_cdf_1C,
)
from pysatl_core.types import CharacteristicName
from tests.unit.distributions.test_basic import DistributionTestBase


class TestFitPdfToCdf1C(DistributionTestBase):
    """Tests for _fit_pdf_to_cdf_1C (segment-wise integration)."""

    def test_uniform_cdf_correctness(self) -> None:
        """CDF of Uniform[0,1] should be x on [0,1]."""
        distr = self.make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(result, xs, atol=1e-4)  # type: ignore[arg-type]

    def test_scalar_in_array_out(self) -> None:
        distr = self.make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        result = fitted.func(np.float64(0.5))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_array_in_array_out(self) -> None:
        distr = self.make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        xs = np.array([0.1, 0.5, 0.9])
        result = fitted.func(xs)  # type: ignore[call-arg]
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_monotonicity(self) -> None:
        """CDF must be non-decreasing."""
        distr = self.make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        xs = np.linspace(0.0, 1.0, 20)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(np.diff(result) >= -1e-10)

    def test_cdf_bounds(self) -> None:
        """CDF values must be in [0, 1]."""
        distr = self.make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        xs = np.linspace(-1.0, 2.0, 30)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(result >= 0.0)  # type: ignore[operator]
        assert np.all(result <= 1.0)  # type: ignore[operator]

    def test_segment_wise_unsorted_input(self) -> None:
        """Segment-wise integration should handle unsorted input correctly."""
        distr = self.make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        # Unsorted input
        xs_unsorted = np.array([0.9, 0.1, 0.5, 0.3, 0.7])
        result_unsorted = np.asarray(fitted.func(xs_unsorted), dtype=float)  # type: ignore[call-arg,type-var]
        # Sorted input for reference
        xs_sorted = np.sort(xs_unsorted)
        result_sorted = np.asarray(fitted.func(xs_sorted), dtype=float)  # type: ignore[call-arg,type-var]
        # Results should match when reordered
        order = np.argsort(xs_unsorted)
        np.testing.assert_allclose(result_unsorted[order], result_sorted, atol=1e-6)  # type: ignore[arg-type]

    def test_descriptor_metadata(self) -> None:
        desc = _build_pdf_to_cdf_1C()
        assert desc.target == CharacteristicName.CDF
        assert desc.sources == [CharacteristicName.PDF]
        assert "continuous" in desc.constraint_tags
        assert desc.option_names() == ("limit",)

    def test_fitted_method_metadata(self) -> None:
        distr = self.make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        assert fitted.target == CharacteristicName.CDF
        assert fitted.sources == [CharacteristicName.PDF]


class TestFitCdfToPdf1C(DistributionTestBase):
    """Tests for _fit_cdf_to_pdf_1C (five-point finite difference)."""

    def test_logistic_pdf_correctness(self) -> None:
        """PDF of logistic CDF should match analytical logistic PDF."""
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_pdf_1C(distr)
        xs = np.linspace(-3.0, 3.0, 15)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        # Analytical logistic PDF: e^(-x) / (1 + e^(-x))^2
        expected = np.exp(-xs) / (1.0 + np.exp(-xs)) ** 2
        np.testing.assert_allclose(result, expected, atol=1e-4)  # type: ignore[arg-type]

    def test_scalar_in_array_out(self) -> None:
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_pdf_1C(distr)
        result = fitted.func(np.float64(0.0))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_array_in_array_out(self) -> None:
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_pdf_1C(distr)
        xs = np.array([-1.0, 0.0, 1.0])
        result = fitted.func(xs)  # type: ignore[call-arg]
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_pdf_non_negative(self) -> None:
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_pdf_1C(distr)
        xs = np.linspace(-5.0, 5.0, 50)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(result >= 0.0)  # type: ignore[operator]

    def test_descriptor_metadata(self) -> None:
        desc = _build_cdf_to_pdf_1C()
        assert desc.target == CharacteristicName.PDF
        assert desc.sources == [CharacteristicName.CDF]
        assert desc.option_names() == ("h",)


class TestFitCdfToPpf1C(DistributionTestBase):
    """Tests for _fit_cdf_to_ppf_1C (vectorised bisection)."""

    def test_logistic_ppf_correctness(self) -> None:
        """PPF of logistic CDF should match analytical logistic PPF."""
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_ppf_1C(distr)
        qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        result = np.asarray(fitted.func(qs), dtype=float)  # type: ignore[call-arg,type-var]
        # Analytical logistic PPF: log(q / (1 - q))
        expected = np.log(qs / (1.0 - qs))
        np.testing.assert_allclose(result, expected, atol=1e-3)  # type: ignore[arg-type]

    def test_scalar_in_array_out(self) -> None:
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_ppf_1C(distr)
        result = fitted.func(np.float64(0.5))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_boundary_values(self) -> None:
        """PPF(0) = -inf, PPF(1) = +inf."""
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_ppf_1C(distr)
        qs = np.array([0.0, 1.0])
        result = np.asarray(fitted.func(qs), dtype=float)  # type: ignore[call-arg,type-var]
        assert result[0] == -np.inf
        assert result[1] == np.inf

    def test_monotonicity(self) -> None:
        distr = self.make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_ppf_1C(distr)
        qs = np.linspace(0.01, 0.99, 20)
        result = np.asarray(fitted.func(qs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(np.diff(result) >= 0.0)

    def test_descriptor_metadata(self) -> None:
        desc = _build_cdf_to_ppf_1C()
        assert desc.target == CharacteristicName.PPF
        assert desc.sources == [CharacteristicName.CDF]
        assert set(desc.option_names()) == {"max_iter", "x_tol", "eps", "x0"}


class TestFitPpfToCdf1C(DistributionTestBase):
    """Tests for _fit_ppf_to_cdf_1C (root inversion via brentq)."""

    def test_uniform_cdf_from_ppf(self) -> None:
        """CDF from PPF of Uniform[0,1] should be x on [0,1]."""
        distr = self.make_uniform_ppf_distribution()
        fitted = _fit_ppf_to_cdf_1C(distr)
        xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(result, xs, atol=1e-3)  # type: ignore[arg-type]

    def test_scalar_in_array_out(self) -> None:
        distr = self.make_uniform_ppf_distribution()
        fitted = _fit_ppf_to_cdf_1C(distr)
        result = fitted.func(np.float64(0.5))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_cdf_interior_bounds(self) -> None:
        """CDF values for interior points must be in [0, 1]."""
        distr = self.make_uniform_ppf_distribution()
        fitted = _fit_ppf_to_cdf_1C(distr)
        # Only test interior points where brentq can find a root
        xs = np.linspace(0.01, 0.99, 20)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(result >= 0.0)  # type: ignore[operator]
        assert np.all(result <= 1.0)  # type: ignore[operator]

    def test_no_nan_at_support_bounds(self) -> None:
        """CDF must return 0.0 at the left support bound and 1.0 at the right, not nan."""
        distr = self.make_uniform_ppf_distribution()
        fitted = _fit_ppf_to_cdf_1C(distr)
        # Uniform[0,1]: ppf(q) = q, so support is [0, 1].
        # x=0 is at/below ppf(q_lowest≈0), x=1 is at/above ppf(q_highest≈1).
        xs = np.array([0.0, 1.0])
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert not np.any(np.isnan(result)), f"Got nan at support bounds: {result}"
        assert result[0] == pytest.approx(0.0, abs=1e-3)
        assert result[1] == pytest.approx(1.0, abs=1e-3)

    def test_no_nan_outside_support(self) -> None:
        """CDF must return 0.0 below support and 1.0 above support, not nan."""
        distr = self.make_uniform_ppf_distribution()
        fitted = _fit_ppf_to_cdf_1C(distr)
        xs = np.array([-1.0, -0.5, 1.5, 2.0])
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert not np.any(np.isnan(result)), f"Got nan outside support: {result}"
        # Below support → 0.0; above support → 1.0
        np.testing.assert_array_equal(result[:2], 0.0)
        np.testing.assert_array_equal(result[2:], 1.0)

    def test_cdf_full_range_no_nan(self) -> None:
        """CDF must never return nan for any finite input."""
        distr = self.make_uniform_ppf_distribution()
        fitted = _fit_ppf_to_cdf_1C(distr)
        xs = np.linspace(-2.0, 3.0, 50)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert not np.any(np.isnan(result)), f"Got nan values: {result[np.isnan(result)]}"
        assert np.all(result >= 0.0)  # type: ignore[operator]
        assert np.all(result <= 1.0)  # type: ignore[operator]

    def test_descriptor_metadata(self) -> None:
        desc = _build_ppf_to_cdf_1C()
        assert desc.target == CharacteristicName.CDF
        assert desc.sources == [CharacteristicName.PPF]
        assert set(desc.option_names()) == {"q_lowest", "q_highest", "max_iter"}


class TestContinuousRoundtrip(DistributionTestBase):
    """Roundtrip tests: CDF(PPF(q)) ≈ q and PPF(CDF(x)) ≈ x."""

    def test_cdf_ppf_roundtrip(self) -> None:
        """CDF(PPF(q)) ≈ q for logistic distribution."""
        distr = self.make_logistic_cdf_distribution()
        ppf_fitted = _fit_cdf_to_ppf_1C(distr)
        qs = np.linspace(0.05, 0.95, 10)
        xs = np.asarray(ppf_fitted.func(qs), dtype=float)  # type: ignore[call-arg,type-var]
        # Now evaluate CDF at those xs using array semantics
        cdf_func = distr.query_method(CharacteristicName.CDF)
        cdf_vals = np.asarray(cdf_func(xs), dtype=float)
        np.testing.assert_allclose(cdf_vals, qs, atol=1e-3)
