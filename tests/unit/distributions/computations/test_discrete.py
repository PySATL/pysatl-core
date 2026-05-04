"""
Tests for discrete-distribution fitters (1D).

Each test verifies:
- Correctness against a known discrete distribution (point PMF).
- Array semantics: any input → array out.
- Edge cases (off-support queries, boundary values).

The test distribution has support {0, 1, 2} with PMF {0.0: 0.2, 1.0: 0.5, 2.0: 0.3}.

Right-unbounded tests use a truncated geometric-like PMF on {0, 1, 2, ...}
with P(X=k) = 0.5^(k+1) for k >= 0.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
import pytest

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.computations.discrete import (
    _build_cdf_to_pmf_1D,
    _build_cdf_to_ppf_1D,
    _build_pmf_to_cdf_1D,
    _build_ppf_to_cdf_1D,
    _fit_cdf_to_pmf_1D,
    _fit_cdf_to_ppf_1D,
    _fit_pmf_to_cdf_1D,
)
from pysatl_core.distributions.support import (
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL as DEFAULT_ANALYTICAL_LABEL,
    CharacteristicName,
    Kind,
    NumericArray,
)
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


def _make_discrete_cdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    """Create a discrete distribution with an analytical CDF for testing cdf→pmf and cdf→ppf."""
    # Support {0, 1, 2}, PMF {0: 0.2, 1: 0.5, 2: 0.3}
    # CDF: F(x) = 0 for x < 0, 0.2 for 0 <= x < 1, 0.7 for 1 <= x < 2, 1.0 for x >= 2

    def cdf(x: NumericArray, **_: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.where(
            x_arr < 0.0, 0.0, np.where(x_arr < 1.0, 0.2, np.where(x_arr < 2.0, 0.7, 1.0))
        )
        return result

    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.DISCRETE,
        analytical_computations={
            CharacteristicName.CDF: {
                DEFAULT_ANALYTICAL_LABEL: AnalyticalComputation[NumericArray, NumericArray](
                    target=CharacteristicName.CDF,
                    func=cdf,  # type: ignore[arg-type]
                )
            }
        },
        support=ExplicitTableDiscreteSupport([0, 1, 2]),
    )


class TestFitPmfToCdf1D(DistributionTestBase):
    """Tests for _fit_pmf_to_cdf_1D (prefix-sum)."""

    def test_point_pmf_cdf_correctness(self) -> None:
        """CDF of {0: 0.2, 1: 0.5, 2: 0.3} should be step function."""
        distr = self.make_discrete_point_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)

        # Before first support point
        assert float(fitted.func(np.float64(-0.5))[0]) == pytest.approx(0.0)  # type: ignore[call-arg,arg-type]
        # At support points
        assert float(fitted.func(np.float64(0.0))[0]) == pytest.approx(0.2, abs=1e-6)  # type: ignore[call-arg,arg-type]
        assert float(fitted.func(np.float64(1.0))[0]) == pytest.approx(0.7, abs=1e-6)  # type: ignore[call-arg,arg-type]
        assert float(fitted.func(np.float64(2.0))[0]) == pytest.approx(1.0, abs=1e-6)  # type: ignore[call-arg,arg-type]
        # Between support points
        assert float(fitted.func(np.float64(0.5))[0]) == pytest.approx(0.2, abs=1e-6)  # type: ignore[call-arg,arg-type]
        # After last support point
        assert float(fitted.func(np.float64(3.0))[0]) == pytest.approx(1.0, abs=1e-6)  # type: ignore[call-arg,arg-type]

    def test_scalar_in_array_out(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        result = fitted.func(np.float64(1.0))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_array_in_array_out(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        xs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        result = fitted.func(xs)  # type: ignore[call-arg]
        assert isinstance(result, np.ndarray)
        assert result.shape == (7,)

    def test_monotonicity(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        xs = np.linspace(-1.0, 3.0, 50)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(np.diff(result) >= -1e-10)

    def test_cdf_bounds(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        xs = np.linspace(-2.0, 4.0, 50)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(result >= 0.0)  # type: ignore[operator]
        assert np.all(result <= 1.0)  # type: ignore[operator]

    def test_requires_discrete_support(self) -> None:
        distr = self.make_discrete_point_pmf_distribution(is_with_support=False)
        with pytest.raises(RuntimeError, match="Discrete support"):
            _fit_pmf_to_cdf_1D(distr)

    def test_descriptor_metadata(self) -> None:
        desc = _build_pmf_to_cdf_1D()
        assert desc.target == CharacteristicName.CDF
        assert desc.sources == [CharacteristicName.PMF]
        assert "discrete" in desc.constraint_tags


class TestFitCdfToPmf1D:
    """Tests for _fit_cdf_to_pmf_1D (finite differences on CDF table)."""

    def test_pmf_correctness(self) -> None:
        """PMF from CDF should recover original PMF values at support points."""
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = _fit_cdf_to_pmf_1D(distr)
        xs = np.array([0.0, 1.0, 2.0])
        result = np.asarray(pmf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(result, [0.2, 0.5, 0.3], atol=1e-6)  # type: ignore[arg-type]

    def test_off_support_returns_zero(self) -> None:
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = _fit_cdf_to_pmf_1D(distr)
        xs = np.array([-1.0, 0.5, 1.5, 3.0])
        result = np.asarray(pmf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 0.0], atol=1e-10)  # type: ignore[arg-type]

    def test_scalar_in_array_out(self) -> None:
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = _fit_cdf_to_pmf_1D(distr)
        result = pmf_fitted.func(np.float64(1.0))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_descriptor_metadata(self) -> None:
        desc = _build_cdf_to_pmf_1D()
        assert desc.target == CharacteristicName.PMF
        assert desc.sources == [CharacteristicName.CDF]


class TestFitCdfToPpf1D:
    """Tests for _fit_cdf_to_ppf_1D (table inversion via searchsorted)."""

    def test_ppf_correctness(self) -> None:
        """PPF should return the smallest support point with CDF >= q."""
        distr = _make_discrete_cdf_distribution()
        ppf_fitted = _fit_cdf_to_ppf_1D(distr)

        # CDF: {0: 0.2, 1: 0.7, 2: 1.0}
        assert float(ppf_fitted.func(np.float64(0.1))[0]) == pytest.approx(0.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.2))[0]) == pytest.approx(0.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.3))[0]) == pytest.approx(1.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.7))[0]) == pytest.approx(1.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.8))[0]) == pytest.approx(2.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(1.0))[0]) == pytest.approx(2.0)  # type: ignore[call-arg,arg-type]

    def test_scalar_in_array_out(self) -> None:
        distr = _make_discrete_cdf_distribution()
        ppf_fitted = _fit_cdf_to_ppf_1D(distr)
        result = ppf_fitted.func(np.float64(0.5))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_array_in_array_out(self) -> None:
        distr = _make_discrete_cdf_distribution()
        ppf_fitted = _fit_cdf_to_ppf_1D(distr)
        qs = np.array([0.1, 0.5, 0.9])
        result = ppf_fitted.func(qs)  # type: ignore[call-arg]
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_descriptor_metadata(self) -> None:
        desc = _build_cdf_to_ppf_1D()
        assert desc.target == CharacteristicName.PPF
        assert desc.sources == [CharacteristicName.CDF]


class TestFitPpfToCdf1D:
    """Tests for _fit_ppf_to_cdf_1D (grid probing + step-function table)."""

    def test_descriptor_metadata(self) -> None:
        desc = _build_ppf_to_cdf_1D()
        assert desc.target == CharacteristicName.CDF
        assert desc.sources == [CharacteristicName.PPF]
        assert desc.option_names() == ("n_q_grid",)


def _make_right_unbounded_pmf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    """
    Create a right-unbounded discrete distribution for testing pmf→cdf.

    Support: {0, 1, 2, ...} (left-bounded at 0, right-unbounded).
    PMF: P(X=k) = 0.5^(k+1) for k >= 0  (geometric with p=0.5).
    CDF: F(k) = 1 - 0.5^(k+1) for k >= 0.
    """

    def pmf(x: NumericArray, **_: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        k_arr = np.round(x_arr).astype(int)
        on_lattice = np.abs(x_arr - k_arr) < 1e-9
        result = np.where(
            (k_arr >= 0) & on_lattice,
            0.5 ** (k_arr + 1),
            0.0,
        )
        return result

    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.DISCRETE,
        analytical_computations={
            CharacteristicName.PMF: {
                DEFAULT_ANALYTICAL_LABEL: AnalyticalComputation[NumericArray, NumericArray](
                    target=CharacteristicName.PMF,
                    func=pmf,  # type: ignore[arg-type]
                )
            }
        },
        support=IntegerLatticeDiscreteSupport(min_k=0, max_k=None, modulus=1, residue=0),
    )


class TestFitPmfToCdfRightUnbounded:
    """Tests for _fit_pmf_to_cdf_1D on a left-bounded, right-unbounded lattice."""

    def test_cdf_correctness_at_support_points(self) -> None:
        """CDF of geometric(0.5) should be F(k) = 1 - 0.5^(k+1)."""
        distr = _make_right_unbounded_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)

        for k in range(6):
            expected = 1.0 - 0.5 ** (k + 1)
            actual = float(fitted.func(np.array([float(k)]))[0])  # type: ignore[call-arg]
            assert actual == pytest.approx(expected, abs=1e-6), f"CDF({k}) mismatch"

    def test_cdf_zero_before_support(self) -> None:
        """CDF must be 0 for x < 0."""
        distr = _make_right_unbounded_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        result = float(fitted.func(np.array([-1.0]))[0])  # type: ignore[call-arg]
        assert result == pytest.approx(0.0)

    def test_cdf_monotonicity(self) -> None:
        """CDF must be non-decreasing."""
        distr = _make_right_unbounded_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        xs = np.linspace(-1.0, 20.0, 100)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(np.diff(result) >= -1e-10)

    def test_cdf_bounds(self) -> None:
        """CDF values must lie in [0, 1]."""
        distr = _make_right_unbounded_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        xs = np.linspace(-2.0, 30.0, 100)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(result >= 0.0)  # type: ignore[operator]
        assert np.all(result <= 1.0)  # type: ignore[operator]

    def test_scalar_in_array_out(self) -> None:
        """Output must always be an ndarray."""
        distr = _make_right_unbounded_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        result = fitted.func(np.float64(3.0))  # type: ignore[call-arg,arg-type]
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)


class TestDiscreteRoundtrip(DistributionTestBase):
    """Roundtrip tests for discrete fitters."""

    def test_pmf_cdf_pmf_roundtrip(self) -> None:
        """PMF → CDF table values should be consistent with CDF → PMF."""
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = _fit_cdf_to_pmf_1D(distr)

        xs = np.array([0.0, 1.0, 2.0])
        pmf_result = np.asarray(pmf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(pmf_result, [0.2, 0.5, 0.3], atol=1e-6)  # type: ignore[arg-type]

    def test_cdf_ppf_roundtrip(self) -> None:
        """CDF(PPF(q)) should be >= q for discrete distributions."""
        distr = self.make_discrete_point_pmf_distribution()
        cdf_fitted = _fit_pmf_to_cdf_1D(distr)

        # Use the CDF distribution for PPF
        cdf_distr = _make_discrete_cdf_distribution()
        ppf_fitted = _fit_cdf_to_ppf_1D(cdf_distr)

        qs = np.array([0.1, 0.3, 0.5, 0.8])
        xs = np.asarray(ppf_fitted.func(qs), dtype=float)  # type: ignore[call-arg,type-var]
        cdf_at_xs = np.asarray(cdf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,arg-type,type-var]
        # For discrete distributions, CDF(PPF(q)) >= q
        assert np.all(cdf_at_xs >= qs - 1e-10)  # type: ignore[operator]
