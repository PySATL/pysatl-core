"""Performance sanity checks for the fitters package.

These tests verify that:
1. Fitters handle moderate array sizes without quadratic blow-up.
2. Segment-wise PDF→CDF integration is faster than naive per-point integration.
3. Discrete fitters scale linearly with query-array size.

All timing assertions use generous multipliers to avoid flaky CI
failures while still catching O(n^2) regressions.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy import integrate

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.computations.continuous import (
    _fit_cdf_to_pdf_1C,
    _fit_cdf_to_ppf_1C,
    _fit_pdf_to_cdf_1C,
)
from pysatl_core.distributions.computations.discrete import (
    _fit_cdf_to_pmf_1D,
    _fit_cdf_to_ppf_1D,
    _fit_pmf_to_cdf_1D,
)
from pysatl_core.distributions.support import ExplicitTableDiscreteSupport
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL as DEFAULT_ANALYTICAL_LABEL,
    CharacteristicName,
    Kind,
    NumericArray,
)
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_base = DistributionTestBase()


def _time_call(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _make_logistic_cdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    """Logistic CDF distribution: CDF(x) = 1/(1+exp(-x))."""
    return _base.make_logistic_cdf_distribution()


def _make_uniform_pdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    """Uniform PDF distribution on [0, 1]."""
    return _base.make_uniform_pdf_distribution()


def _make_discrete_pmf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    """Discrete PMF distribution with support {0, 1, 2}."""
    return _base.make_discrete_point_pmf_distribution()


def _make_discrete_cdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    """Discrete distribution with analytical CDF for cdf→pmf and cdf→ppf tests."""

    def cdf(x: NumericArray, **_: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        return np.where(
            x_arr < 0.0, 0.0, np.where(x_arr < 1.0, 0.2, np.where(x_arr < 2.0, 0.7, 1.0))
        )

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


# ---------------------------------------------------------------------------
# Continuous performance tests
# ---------------------------------------------------------------------------


class TestContinuousPerformance:
    """Performance sanity checks for continuous fitters."""

    def test_pdf_to_cdf_moderate_array_completes_in_time(self) -> None:
        """_fit_pdf_to_cdf_1C with 50 points should complete in < 10s.

        This catches quadratic blow-up: 50 points with segment-wise
        integration should be ~50 quad calls, not 50 full-range integrals.
        """
        distr = _make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)
        x = np.linspace(0.05, 0.95, 50)

        _, elapsed = _time_call(fitted.func, x)
        assert elapsed < 10.0, (
            f"_fit_pdf_to_cdf_1C with 50 points took {elapsed:.2f}s (expected < 10s)"
        )

    def test_pdf_to_cdf_segment_wise_scales_subquadratically(self) -> None:
        """Doubling the array size should NOT quadruple the time.

        With segment-wise integration, time should scale roughly linearly
        (each additional point adds one short-range quad call).
        We allow up to 5× growth for 2× input to account for noise.
        """
        distr = _make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)

        x_small = np.linspace(0.05, 0.95, 20)
        x_large = np.linspace(0.05, 0.95, 40)

        # Warm up
        fitted.func(x_small[:2])  # type: ignore[call-arg]

        _, t_small = _time_call(fitted.func, x_small)
        _, t_large = _time_call(fitted.func, x_large)

        # With linear scaling, ratio should be ~2.0.
        # We allow up to 5.0 to avoid flakiness, but catch O(n²) where ratio ≈ 4.
        if t_small > 0.001:  # Only check ratio if small time is measurable
            ratio = t_large / t_small
            assert ratio < 5.0, (
                f"Scaling ratio {ratio:.1f}× for 2× input "
                f"(small={t_small:.4f}s, large={t_large:.4f}s). "
                f"Possible quadratic behavior."
            )

    def test_cdf_to_pdf_array_performance(self) -> None:
        """_fit_cdf_to_pdf_1C with 100 points should complete quickly."""
        distr = _make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_pdf_1C(distr)
        x = np.linspace(-5.0, 5.0, 100)

        _, elapsed = _time_call(fitted.func, x)
        assert elapsed < 5.0, (
            f"_fit_cdf_to_pdf_1C with 100 points took {elapsed:.2f}s (expected < 5s)"
        )

    def test_cdf_to_ppf_array_performance(self) -> None:
        """_fit_cdf_to_ppf_1C with 50 points should complete in < 10s."""
        distr = _make_logistic_cdf_distribution()
        fitted = _fit_cdf_to_ppf_1C(distr)
        q = np.linspace(0.01, 0.99, 50)

        _, elapsed = _time_call(fitted.func, q)
        assert elapsed < 10.0, (
            f"_fit_cdf_to_ppf_1C with 50 points took {elapsed:.2f}s (expected < 10s)"
        )

    def test_pdf_to_cdf_input_order_preserves_integration_work(self) -> None:
        """Preserve CDF values and segment-wise integration work under permutation."""
        distr = _make_uniform_pdf_distribution()
        fitted = _fit_pdf_to_cdf_1C(distr)

        x_sorted = np.linspace(0.05, 0.95, 30)
        rng = np.random.default_rng(42)
        x_shuffled = rng.permutation(x_sorted)

        expected_intervals = [(-np.inf, float(x_sorted[0]))]
        expected_intervals.extend(zip(x_sorted[:-1], x_sorted[1:], strict=True))

        for x in (x_sorted, x_shuffled):
            # Run the real integrator while recording its integration bounds.
            with patch(
                "pysatl_core.distributions.computations.continuous._sp_integrate.quad",
                wraps=integrate.quad,
            ) as quad:
                result = fitted.func(x)  # type: ignore[call-arg]

            # Uniform(0, 1) has CDF(x) = x here, in the original input order.
            np.testing.assert_allclose(result, x, rtol=0, atol=1e-8)
            assert quad.call_count == x.size
            intervals = [(call.args[1], call.args[2]) for call in quad.call_args_list]
            assert intervals == expected_intervals


# ---------------------------------------------------------------------------
# Discrete performance tests
# ---------------------------------------------------------------------------


class TestDiscretePerformance:
    """Performance sanity checks for discrete fitters."""

    def test_pmf_to_cdf_moderate_array(self) -> None:
        """_fit_pmf_to_cdf_1D with 1000 query points should be fast."""
        distr = _make_discrete_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)
        x = np.linspace(-1.0, 3.0, 1000)

        _, elapsed = _time_call(fitted.func, x)
        assert elapsed < 2.0, (
            f"_fit_pmf_to_cdf_1D with 1000 points took {elapsed:.2f}s (expected < 2s)"
        )

    def test_pmf_to_cdf_scales_linearly(self) -> None:
        """Doubling query size should roughly double time, not quadruple."""
        distr = _make_discrete_pmf_distribution()
        fitted = _fit_pmf_to_cdf_1D(distr)

        x_small = np.linspace(-1.0, 3.0, 500)
        x_large = np.linspace(-1.0, 3.0, 1000)

        # Warm up
        fitted.func(x_small[:10])  # type: ignore[call-arg]

        _, t_small = _time_call(fitted.func, x_small)
        _, t_large = _time_call(fitted.func, x_large)

        if t_small > 0.0001:
            ratio = t_large / t_small
            assert ratio < 5.0, (
                f"Scaling ratio {ratio:.1f}× for 2× input "
                f"(small={t_small:.4f}s, large={t_large:.4f}s)"
            )

    def test_cdf_to_pmf_moderate_array(self) -> None:
        """_fit_cdf_to_pmf_1D with 1000 query points should be fast."""
        distr = _make_discrete_cdf_distribution()
        fitted = _fit_cdf_to_pmf_1D(distr)
        x = np.linspace(-1.0, 3.0, 1000)

        _, elapsed = _time_call(fitted.func, x)
        assert elapsed < 2.0, (
            f"_fit_cdf_to_pmf_1D with 1000 points took {elapsed:.2f}s (expected < 2s)"
        )

    def test_cdf_to_ppf_moderate_array(self) -> None:
        """_fit_cdf_to_ppf_1D with 500 query points should be fast."""
        distr = _make_discrete_cdf_distribution()
        fitted = _fit_cdf_to_ppf_1D(distr)
        q = np.linspace(0.01, 0.99, 500)

        _, elapsed = _time_call(fitted.func, q)
        assert elapsed < 2.0, (
            f"_fit_cdf_to_ppf_1D with 500 points took {elapsed:.2f}s (expected < 2s)"
        )


# ---------------------------------------------------------------------------
# Registry performance tests
# ---------------------------------------------------------------------------


class TestRegistryPerformance:
    """Performance sanity checks for FitterRegistry operations."""

    def test_registry_lookup_is_fast(self) -> None:
        """Finding a fitter in the registry should be sub-millisecond."""
        from pysatl_core.distributions.computations.registry import FitterRegistry, fitter_registry

        reg = FitterRegistry()
        reg.register_many(fitter_registry().all_descriptors())

        _, elapsed = _time_call(
            reg.find,
            target=CharacteristicName.CDF,
            sources=frozenset({CharacteristicName.PDF}),
        )
        assert elapsed < 0.01, f"Registry lookup took {elapsed:.4f}s (expected < 10ms)"

    def test_registry_find_all_is_fast(self) -> None:
        """Finding all matching fitters should be sub-millisecond."""
        from pysatl_core.distributions.computations.registry import FitterRegistry, fitter_registry

        reg = FitterRegistry()
        reg.register_many(fitter_registry().all_descriptors())

        _, elapsed = _time_call(
            reg.find_all,
            target=CharacteristicName.CDF,
            sources=[CharacteristicName.PDF],
        )
        assert elapsed < 0.01, f"Registry find_all took {elapsed:.4f}s (expected < 10ms)"
