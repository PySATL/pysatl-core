"""
Unit tests for EmpiricalDistribution: with_method / set_method / indirection.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from pysatl_core.distributions.empirical import (
    EmpiricalComputationStrategy,
    EmpiricalDistribution,
    FittedEmpirical,
    ScipyGaussianKde,
)
from pysatl_core.types import CharacteristicName


class _ConstantFittedEmpirical:
    def __init__(self, pdf_value: float, cdf_value: float) -> None:
        self._pdf_value = pdf_value
        self._cdf_value = cdf_value
        self.fit_calls = 0

    def pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.full_like(x, self._pdf_value)

    def cdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.full_like(x, self._cdf_value)


class _ConstantMethod:
    def __init__(self, pdf_value: float, cdf_value: float) -> None:
        self._pdf_value = pdf_value
        self._cdf_value = cdf_value
        self.fit_calls = 0

    def fit(self, sample: NDArray[np.float64]) -> FittedEmpirical:
        self.fit_calls += 1
        return _ConstantFittedEmpirical(self._pdf_value, self._cdf_value)


@pytest.fixture
def sample() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, 300)


@pytest.fixture
def distr(sample: NDArray[np.float64]) -> EmpiricalDistribution:
    return EmpiricalDistribution(sample)


class TestIndirection:
    def test_pdf_indirection_reads_current_estimator(self, distr: EmpiricalDistribution) -> None:
        distr._estimator = _ConstantFittedEmpirical(pdf_value=42.0, cdf_value=0.5)
        result = distr.calculate_characteristic(CharacteristicName.PDF, np.array([0.0, 1.0]))
        assert np.allclose(result, 42.0)

    def test_cdf_indirection_reads_current_estimator(self, distr: EmpiricalDistribution) -> None:
        distr._estimator = _ConstantFittedEmpirical(pdf_value=1.0, cdf_value=0.7)
        result = distr.calculate_characteristic(CharacteristicName.CDF, np.array([0.0, 2.0]))
        assert np.allclose(result, 0.7)


class TestDefaultStrategy:
    def test_default_computation_strategy_is_empirical(self, distr: EmpiricalDistribution) -> None:
        assert isinstance(distr.computation_strategy, EmpiricalComputationStrategy)


class TestWithMethod:
    def test_returns_new_instance(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_method(ScipyGaussianKde(bandwidth="silverman"))
        assert clone is not distr
        assert isinstance(clone, EmpiricalDistribution)

    def test_shares_sample(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_method(ScipyGaussianKde(bandwidth="silverman"))
        assert clone.data is distr.data

    def test_changes_pdf(self, sample: NDArray[np.float64]) -> None:
        scott = EmpiricalDistribution(sample, method=ScipyGaussianKde(bandwidth="scott"))
        silverman = scott.with_method(ScipyGaussianKde(bandwidth="silverman"))

        x = np.array([0.0, 0.5, 1.0])
        pdf_scott = scott.calculate_characteristic(CharacteristicName.PDF, x)
        pdf_silverman = silverman.calculate_characteristic(CharacteristicName.PDF, x)

        assert not np.allclose(pdf_scott, pdf_silverman)

    def test_preserves_original(self, distr: EmpiricalDistribution) -> None:
        x = np.array([0.0, 1.0])
        pdf_before = distr.calculate_characteristic(CharacteristicName.PDF, x)

        distr.with_method(ScipyGaussianKde(bandwidth="silverman"))

        pdf_after = distr.calculate_characteristic(CharacteristicName.PDF, x)
        assert np.allclose(pdf_before, pdf_after)

    def test_independent_strategies(self, distr: EmpiricalDistribution) -> None:
        clone = distr.with_method(ScipyGaussianKde(bandwidth="silverman"))
        assert clone.sampling_strategy is not distr.sampling_strategy
        assert clone.computation_strategy is not distr.computation_strategy

    def test_method_property_reflects_new_method(self, distr: EmpiricalDistribution) -> None:
        new_method = ScipyGaussianKde(bandwidth="silverman")
        clone = distr.with_method(new_method)
        assert clone.method is new_method
        assert distr.method is not new_method

    def test_calls_fit_exactly_once_on_new_method(self, sample: NDArray[np.float64]) -> None:
        method = _ConstantMethod(pdf_value=1.0, cdf_value=0.5)
        distr = EmpiricalDistribution(sample)

        clone = distr.with_method(method)

        assert method.fit_calls == 1
        assert np.allclose(
            clone.calculate_characteristic(CharacteristicName.PDF, np.array([0.0])),
            1.0,
        )

    def test_composes_with_sampling_strategy_override(self, distr: EmpiricalDistribution) -> None:
        from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import (
            DefaultUnuranSamplingStrategy,
        )

        new_sampling = DefaultUnuranSamplingStrategy()
        chained = distr.with_method(ScipyGaussianKde(bandwidth="silverman")).with_sampling_strategy(
            new_sampling
        )
        assert chained.sampling_strategy is new_sampling
        assert chained.method is not distr.method


class TestSetMethod:
    def test_mutates_in_place_returns_none(self, distr: EmpiricalDistribution) -> None:
        original_id = id(distr)
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        assert id(distr) == original_id

    def test_method_property_reflects_new_method(self, distr: EmpiricalDistribution) -> None:
        new_method = ScipyGaussianKde(bandwidth="silverman")
        distr.set_method(new_method)
        assert distr.method is new_method

    def test_pdf_changes_after_swap(self, distr: EmpiricalDistribution) -> None:
        x = np.array([0.0, 0.5, 1.0])
        pdf_before = distr.calculate_characteristic(CharacteristicName.PDF, x)
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        pdf_after = distr.calculate_characteristic(CharacteristicName.PDF, x)
        assert not np.allclose(pdf_before, pdf_after)

    def test_estimator_object_replaced(self, distr: EmpiricalDistribution) -> None:
        old_estimator = distr._estimator
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        assert distr._estimator is not old_estimator

    def test_calls_fit_exactly_once(self, distr: EmpiricalDistribution) -> None:
        method = _ConstantMethod(pdf_value=7.0, cdf_value=0.3)
        distr.set_method(method)
        assert method.fit_calls == 1
        assert np.allclose(
            distr.calculate_characteristic(CharacteristicName.PDF, np.array([0.0])),
            7.0,
        )

    def test_invalidates_sampling_sampler(
        self, distr: EmpiricalDistribution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _StubSampler:
            def __init__(self, distr: Any, config: Any) -> None: ...

            def sample(self, n: int) -> NDArray[np.float64]:
                return np.zeros(n)

        monkeypatch.setattr(
            "pysatl_core.sampling.unuran.core.unuran_sampling_strategy.DefaultUnuranSampler",
            _StubSampler,
        )
        _ = distr.sample(10)
        from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import (
            DefaultUnuranSamplingStrategy,
        )

        sampler_strategy = cast(DefaultUnuranSamplingStrategy, distr.sampling_strategy)
        assert sampler_strategy._sampler is not None
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        assert sampler_strategy._sampler is None

    def test_computation_cache_cleared_via_estimator_tracking(
        self, sample: NDArray[np.float64]
    ) -> None:
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        d = EmpiricalDistribution(sample, computation_strategy=strategy)

        from pysatl_core.distributions.computations.computation import FittedComputationMethod

        strategy.query_method(CharacteristicName.PDF, d)
        strategy._ppf_cache[id(d)] = FittedComputationMethod[Any, Any](
            target=CharacteristicName.PPF,
            sources=(CharacteristicName.CDF,),
            func=lambda *a, **kw: 0.0,
        )
        assert id(d) in strategy._ppf_cache

        d.set_method(ScipyGaussianKde(bandwidth="silverman"))

        strategy.query_method(CharacteristicName.PDF, d)
        assert id(d) not in strategy._ppf_cache

    def test_sample_after_swap_rebuilds_sampler(
        self, distr: EmpiricalDistribution, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[Any] = []

        class _StubSampler:
            def __init__(self, distr: Any, config: Any) -> None:
                built.append(self)

            def sample(self, n: int) -> NDArray[np.float64]:
                return np.zeros(n)

        monkeypatch.setattr(
            "pysatl_core.sampling.unuran.core.unuran_sampling_strategy.DefaultUnuranSampler",
            _StubSampler,
        )
        _ = distr.sample(5)
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        out = distr.sample(50)

        assert out.shape[0] == 50
        assert len(built) == 2
        assert built[0] is not built[1]

    def test_works_when_strategies_lack_invalidate(self, sample: NDArray[np.float64]) -> None:
        class _BareSampling:
            def sample(self, n: int, distr: Any, **options: Any) -> NDArray[np.float64]:
                return np.zeros(n)

        class _BareComputation:
            def query_method(self, state: Any, distr: Any, **options: Any) -> Any:
                return distr.analytical_computations[CharacteristicName.PDF][
                    next(iter(distr.analytical_computations[CharacteristicName.PDF]))
                ]

            def explain_computation_path(self, state: Any, distr: Any) -> Any:
                raise NotImplementedError

        d = EmpiricalDistribution(
            sample,
            sampling_strategy=_BareSampling(),
            computation_strategy=_BareComputation(),  # type: ignore[arg-type]
        )

        d.set_method(ScipyGaussianKde(bandwidth="silverman"))
