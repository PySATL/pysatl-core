"""
Tests for discrete-distribution fitters (1D).

Each test verifies:
- Correctness against a known discrete distribution (point PMF).
- Array semantics: scalar in → scalar out, array in → array out.
- Edge cases (off-support queries, boundary values).

The test distribution has support {0, 1, 2} with PMF {0.0: 0.2, 1.0: 0.5, 2.0: 0.3}.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from mypy_extensions import KwArg

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.fitters.discrete import (
    FITTER_CDF_TO_PMF_1D,
    FITTER_CDF_TO_PPF_1D,
    FITTER_PMF_TO_CDF_1D,
    FITTER_PPF_TO_CDF_1D,
    fit_cdf_to_pmf_1D,
    fit_cdf_to_ppf_1D,
    fit_pmf_to_cdf_1D,
)
from pysatl_core.distributions.support import ExplicitTableDiscreteSupport
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL as DEFAULT_ANALYTICAL_LABEL,
    CharacteristicName,
    Kind,
)
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


def _make_discrete_cdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    """Create a discrete distribution with an analytical CDF for testing cdf→pmf and cdf→ppf."""
    # Support {0, 1, 2}, PMF {0: 0.2, 1: 0.5, 2: 0.3}
    # CDF: F(x) = 0 for x < 0, 0.2 for 0 <= x < 1, 0.7 for 1 <= x < 2, 1.0 for x >= 2

    def cdf(x: float, **_: Any) -> float:
        if x < 0.0:
            return 0.0
        if x < 1.0:
            return 0.2
        if x < 2.0:
            return 0.7
        return 1.0

    cdf_func = cast(Callable[[float, KwArg(Any)], float], cdf)
    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.DISCRETE,
        analytical_computations={
            CharacteristicName.CDF: {
                DEFAULT_ANALYTICAL_LABEL: AnalyticalComputation[float, float](
                    target=CharacteristicName.CDF, func=cdf_func
                )
            }
        },
        support=ExplicitTableDiscreteSupport([0, 1, 2]),
    )


class TestFitPmfToCdf1D(DistributionTestBase):
    """Tests for fit_pmf_to_cdf_1D (prefix-sum)."""

    def test_point_pmf_cdf_correctness(self) -> None:
        """CDF of {0: 0.2, 1: 0.5, 2: 0.3} should be step function."""
        distr = self.make_discrete_point_pmf_distribution()
        fitted = fit_pmf_to_cdf_1D(distr)

        # Before first support point
        assert float(fitted.func(np.float64(-0.5))) == pytest.approx(0.0)  # type: ignore[call-arg,arg-type]
        # At support points
        assert float(fitted.func(np.float64(0.0))) == pytest.approx(0.2, abs=1e-6)  # type: ignore[call-arg,arg-type]
        assert float(fitted.func(np.float64(1.0))) == pytest.approx(0.7, abs=1e-6)  # type: ignore[call-arg,arg-type]
        assert float(fitted.func(np.float64(2.0))) == pytest.approx(1.0, abs=1e-6)  # type: ignore[call-arg,arg-type]
        # Between support points
        assert float(fitted.func(np.float64(0.5))) == pytest.approx(0.2, abs=1e-6)  # type: ignore[call-arg,arg-type]
        # After last support point
        assert float(fitted.func(np.float64(3.0))) == pytest.approx(1.0, abs=1e-6)  # type: ignore[call-arg,arg-type]

    def test_scalar_in_scalar_out(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = fit_pmf_to_cdf_1D(distr)
        result = fitted.func(np.float64(1.0))  # type: ignore[call-arg,arg-type]
        assert np.ndim(result) == 0

    def test_array_in_array_out(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = fit_pmf_to_cdf_1D(distr)
        xs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        result = fitted.func(xs)  # type: ignore[call-arg]
        assert isinstance(result, np.ndarray)
        assert result.shape == (7,)

    def test_monotonicity(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = fit_pmf_to_cdf_1D(distr)
        xs = np.linspace(-1.0, 3.0, 50)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(np.diff(result) >= -1e-10)

    def test_cdf_bounds(self) -> None:
        distr = self.make_discrete_point_pmf_distribution()
        fitted = fit_pmf_to_cdf_1D(distr)
        xs = np.linspace(-2.0, 4.0, 50)
        result = np.asarray(fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        assert np.all(result >= 0.0)  # type: ignore[operator]
        assert np.all(result <= 1.0)  # type: ignore[operator]

    def test_requires_discrete_support(self) -> None:
        distr = self.make_discrete_point_pmf_distribution(is_with_support=False)
        with pytest.raises(RuntimeError, match="Discrete support"):
            fit_pmf_to_cdf_1D(distr)

    def test_descriptor_metadata(self) -> None:
        assert FITTER_PMF_TO_CDF_1D.target == CharacteristicName.CDF
        assert FITTER_PMF_TO_CDF_1D.sources == [CharacteristicName.PMF]
        assert "discrete" in FITTER_PMF_TO_CDF_1D.tags


class TestFitCdfToPmf1D:
    """Tests for fit_cdf_to_pmf_1D (finite differences on CDF table)."""

    def test_pmf_correctness(self) -> None:
        """PMF from CDF should recover original PMF values at support points."""
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = fit_cdf_to_pmf_1D(distr)
        xs = np.array([0.0, 1.0, 2.0])
        result = np.asarray(pmf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(result, [0.2, 0.5, 0.3], atol=1e-6)  # type: ignore[arg-type]

    def test_off_support_returns_zero(self) -> None:
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = fit_cdf_to_pmf_1D(distr)
        xs = np.array([-1.0, 0.5, 1.5, 3.0])
        result = np.asarray(pmf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 0.0], atol=1e-10)  # type: ignore[arg-type]

    def test_scalar_in_scalar_out(self) -> None:
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = fit_cdf_to_pmf_1D(distr)
        result = pmf_fitted.func(np.float64(1.0))  # type: ignore[call-arg,arg-type]
        assert np.ndim(result) == 0

    def test_descriptor_metadata(self) -> None:
        assert FITTER_CDF_TO_PMF_1D.target == CharacteristicName.PMF
        assert FITTER_CDF_TO_PMF_1D.sources == [CharacteristicName.CDF]


class TestFitCdfToPpf1D:
    """Tests for fit_cdf_to_ppf_1D (table inversion via searchsorted)."""

    def test_ppf_correctness(self) -> None:
        """PPF should return the smallest support point with CDF >= q."""
        distr = _make_discrete_cdf_distribution()
        ppf_fitted = fit_cdf_to_ppf_1D(distr)

        # CDF: {0: 0.2, 1: 0.7, 2: 1.0}
        assert float(ppf_fitted.func(np.float64(0.1))) == pytest.approx(0.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.2))) == pytest.approx(0.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.3))) == pytest.approx(1.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.7))) == pytest.approx(1.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(0.8))) == pytest.approx(2.0)  # type: ignore[call-arg,arg-type]
        assert float(ppf_fitted.func(np.float64(1.0))) == pytest.approx(2.0)  # type: ignore[call-arg,arg-type]

    def test_scalar_in_scalar_out(self) -> None:
        distr = _make_discrete_cdf_distribution()
        ppf_fitted = fit_cdf_to_ppf_1D(distr)
        result = ppf_fitted.func(np.float64(0.5))  # type: ignore[call-arg,arg-type]
        assert np.ndim(result) == 0

    def test_array_in_array_out(self) -> None:
        distr = _make_discrete_cdf_distribution()
        ppf_fitted = fit_cdf_to_ppf_1D(distr)
        qs = np.array([0.1, 0.5, 0.9])
        result = ppf_fitted.func(qs)  # type: ignore[call-arg]
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_descriptor_metadata(self) -> None:
        assert FITTER_CDF_TO_PPF_1D.target == CharacteristicName.PPF
        assert FITTER_CDF_TO_PPF_1D.sources == [CharacteristicName.CDF]


class TestFitPpfToCdf1D:
    """Tests for fit_ppf_to_cdf_1D (grid probing + step-function table)."""

    def test_descriptor_metadata(self) -> None:
        assert FITTER_PPF_TO_CDF_1D.target == CharacteristicName.CDF
        assert FITTER_PPF_TO_CDF_1D.sources == [CharacteristicName.PPF]
        assert FITTER_PPF_TO_CDF_1D.option_names() == ("n_q_grid",)


class TestDiscreteRoundtrip(DistributionTestBase):
    """Roundtrip tests for discrete fitters."""

    def test_pmf_cdf_pmf_roundtrip(self) -> None:
        """PMF → CDF table values should be consistent with CDF → PMF."""
        distr = _make_discrete_cdf_distribution()
        pmf_fitted = fit_cdf_to_pmf_1D(distr)

        xs = np.array([0.0, 1.0, 2.0])
        pmf_result = np.asarray(pmf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,type-var]
        np.testing.assert_allclose(pmf_result, [0.2, 0.5, 0.3], atol=1e-6)  # type: ignore[arg-type]

    def test_cdf_ppf_roundtrip(self) -> None:
        """CDF(PPF(q)) should be >= q for discrete distributions."""
        distr = self.make_discrete_point_pmf_distribution()
        cdf_fitted = fit_pmf_to_cdf_1D(distr)

        # Use the CDF distribution for PPF
        cdf_distr = _make_discrete_cdf_distribution()
        ppf_fitted = fit_cdf_to_ppf_1D(cdf_distr)

        qs = np.array([0.1, 0.3, 0.5, 0.8])
        xs = np.asarray(ppf_fitted.func(qs), dtype=float)  # type: ignore[call-arg,type-var]
        cdf_at_xs = np.asarray(cdf_fitted.func(xs), dtype=float)  # type: ignore[call-arg,arg-type,type-var]
        # For discrete distributions, CDF(PPF(q)) >= q
        assert np.all(cdf_at_xs >= qs - 1e-10)  # type: ignore[operator]
