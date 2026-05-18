"""
Unit tests for Bootstrap, BootstrapResult, ClassicalResampling, SmoothResampling.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.typing import NDArray

from pysatl_core.inference import (
    Bootstrap,
    BootstrapResult,
    ClassicalResampling,
    SmoothResampling,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def data(rng: np.random.Generator) -> NDArray[np.float64]:
    return rng.normal(0.0, 1.0, 100)


class TestClassicalResampling:
    def test_returns_correct_size(self, data: NDArray[np.float64]) -> None:
        result = ClassicalResampling().resample(data, len(data))
        assert result.shape == (len(data),)

    def test_custom_size(self, data: NDArray[np.float64]) -> None:
        result = ClassicalResampling().resample(data, 50)
        assert result.shape == (50,)

    def test_samples_only_from_original_data(self, data: NDArray[np.float64]) -> None:
        result = ClassicalResampling().resample(data, len(data))
        assert all(v in data for v in result)

    def test_can_repeat_elements(self) -> None:
        tiny = np.array([1.0, 2.0])
        method = ClassicalResampling()
        seen_repeats = False
        for _ in range(50):
            sample = method.resample(tiny, 10)
            if len(set(sample.tolist())) < len(sample):
                seen_repeats = True
                break
        assert seen_repeats

    def test_reproducible_with_same_seed(self, data: NDArray[np.float64]) -> None:
        r1 = ClassicalResampling(rng=np.random.default_rng(0)).resample(data, len(data))
        r2 = ClassicalResampling(rng=np.random.default_rng(0)).resample(data, len(data))
        assert np.array_equal(r1, r2)

    def test_different_seeds_give_different_results(self, data: NDArray[np.float64]) -> None:
        r1 = ClassicalResampling(rng=np.random.default_rng(0)).resample(data, len(data))
        r2 = ClassicalResampling(rng=np.random.default_rng(1)).resample(data, len(data))
        assert not np.array_equal(r1, r2)


class TestSmoothResampling:
    def test_returns_correct_size(self, data: NDArray[np.float64]) -> None:
        result = SmoothResampling().resample(data, len(data))
        assert result.shape == (len(data),)

    def test_custom_size(self, data: NDArray[np.float64]) -> None:
        result = SmoothResampling().resample(data, 30)
        assert result.shape == (30,)

    def test_caches_distribution_for_same_data(self, data: NDArray[np.float64]) -> None:
        method = SmoothResampling()
        method.resample(data, 10)
        distr_first = method._distr
        method.resample(data, 10)
        assert method._distr is distr_first

    def test_refits_when_data_changes(self, data: NDArray[np.float64]) -> None:
        method = SmoothResampling()
        method.resample(data, 10)
        other = data * 2.0
        method.resample(other, 10)
        assert method._distr is not None
        assert method._distr.data is other


class TestBootstrapResult:
    @pytest.fixture
    def result(self) -> BootstrapResult:
        replicates = np.random.default_rng(0).normal(3.0, 0.5, 1000)
        return BootstrapResult(observed=3.1, replicates=replicates)

    def test_standard_error_matches_std(self, result: BootstrapResult) -> None:
        assert result.standard_error() == pytest.approx(result.replicates.std())

    def test_bias_formula(self, result: BootstrapResult) -> None:
        expected = float(result.replicates.mean()) - result.observed
        assert result.bias() == pytest.approx(expected)

    def test_confidence_interval_lower_less_than_upper(self, result: BootstrapResult) -> None:
        lo, hi = result.confidence_interval()
        assert lo < hi

    def test_confidence_interval_matches_percentiles(self, result: BootstrapResult) -> None:
        lo, hi = result.confidence_interval(level=0.95)
        assert lo == pytest.approx(float(np.percentile(result.replicates, 2.5)))
        assert hi == pytest.approx(float(np.percentile(result.replicates, 97.5)))

    def test_confidence_interval_custom_level(self, result: BootstrapResult) -> None:
        lo, hi = result.confidence_interval(level=0.90)
        assert lo == pytest.approx(float(np.percentile(result.replicates, 5.0)))
        assert hi == pytest.approx(float(np.percentile(result.replicates, 95.0)))

    def test_bias_zero_when_replicates_centered_on_observed(self) -> None:
        result = BootstrapResult(observed=2.0, replicates=np.array([1.0, 2.0, 3.0]))
        assert result.bias() == pytest.approx(0.0)


class TestBootstrap:
    def test_run_returns_bootstrap_result(self, data: NDArray[np.float64], rng: np.random.Generator) -> None:
        assert isinstance(Bootstrap(data, B=100, rng=rng).run(np.mean), BootstrapResult)

    def test_observed_is_functional_on_original_data(
        self, data: NDArray[np.float64], rng: np.random.Generator
    ) -> None:
        result = Bootstrap(data, B=100, rng=rng).run(np.mean)
        assert result.observed == pytest.approx(float(np.mean(data)))

    def test_replicates_shape(self, data: NDArray[np.float64], rng: np.random.Generator) -> None:
        result = Bootstrap(data, B=250, rng=rng).run(np.mean)
        assert result.replicates.shape == (250,)

    def test_works_with_std(self, data: NDArray[np.float64], rng: np.random.Generator) -> None:
        result = Bootstrap(data, B=100, rng=rng).run(np.std)
        assert result.observed == pytest.approx(float(np.std(data)))

    def test_works_with_median(self, data: NDArray[np.float64], rng: np.random.Generator) -> None:
        result = Bootstrap(data, B=100, rng=rng).run(np.median)
        assert result.observed == pytest.approx(float(np.median(data)))

    def test_works_with_custom_functional(
        self, data: NDArray[np.float64], rng: np.random.Generator
    ) -> None:
        p90 = lambda x: float(np.percentile(x, 90))
        result = Bootstrap(data, B=100, rng=rng).run(p90)
        assert result.observed == pytest.approx(float(np.percentile(data, 90)))

    def test_reproducible_with_same_seed(self, data: NDArray[np.float64]) -> None:
        r1 = Bootstrap(data, B=200, rng=np.random.default_rng(7)).run(np.mean)
        r2 = Bootstrap(data, B=200, rng=np.random.default_rng(7)).run(np.mean)
        assert np.array_equal(r1.replicates, r2.replicates)

    def test_default_method_uses_bootstrap_rng(self, data: NDArray[np.float64]) -> None:
        r1 = Bootstrap(data, B=50, rng=np.random.default_rng(99)).run(np.mean)
        r2 = Bootstrap(data, B=50, rng=np.random.default_rng(99)).run(np.mean)
        assert np.array_equal(r1.replicates, r2.replicates)

    def test_smooth_resampling_produces_correct_shape(self, data: NDArray[np.float64]) -> None:
        result = Bootstrap(data, B=50, method=SmoothResampling()).run(np.mean)
        assert result.replicates.shape == (50,)

    def test_smooth_resampling_observed_equals_classical(
        self, data: NDArray[np.float64], rng: np.random.Generator
    ) -> None:
        r_classical = Bootstrap(data, B=50, rng=rng).run(np.mean)
        r_smooth = Bootstrap(data, B=50, method=SmoothResampling()).run(np.mean)
        assert r_classical.observed == pytest.approx(r_smooth.observed)
