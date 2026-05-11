"""
Unit tests for EmpiricalComputationStrategy.

Covers:
  - estimator-identity-aware cache invalidation
  - degenerate behaviour for distributions without an _estimator attribute
  - inheritance contract with DefaultComputationStrategy
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions.computations.computation import FittedComputationMethod
from pysatl_core.distributions.empirical import EmpiricalComputationStrategy
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.types import CharacteristicName


class _FakeDistr:
    def __init__(self, estimator: object | None) -> None:
        self._estimator = estimator


def _put_dummy_ppf_cache_entry(strategy: EmpiricalComputationStrategy, fake_id: int = 0) -> None:
    dummy = FittedComputationMethod[Any, Any](
        target=cast(CharacteristicName, "ppf"),
        sources=(),
        func=lambda *a, **kw: None,
    )
    strategy._ppf_cache[fake_id] = dummy


class TestInheritance:
    def test_is_default_strategy_subclass(self) -> None:
        assert issubclass(EmpiricalComputationStrategy, DefaultComputationStrategy)
        assert isinstance(EmpiricalComputationStrategy(), DefaultComputationStrategy)

    def test_caching_disabled_by_default(self) -> None:
        strategy = EmpiricalComputationStrategy()
        assert strategy.is_caching_enabled is False

    def test_caching_can_be_enabled(self) -> None:
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        assert strategy.is_caching_enabled is True


class TestMaybeInvalidate:
    def test_first_query_records_estimator_without_clearing(self) -> None:
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        estimator = object()
        distr = cast(Any, _FakeDistr(estimator))

        strategy._maybe_invalidate(distr)

        assert strategy._tracked_estimator is estimator
        assert strategy._cache == {}
        assert strategy._ppf_cache == {}

    def test_same_estimator_keeps_cache(self) -> None:
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        estimator = object()
        distr = cast(Any, _FakeDistr(estimator))

        strategy._maybe_invalidate(distr)
        _put_dummy_ppf_cache_entry(strategy, fake_id=1)
        strategy._maybe_invalidate(distr)

        assert 1 in strategy._ppf_cache
        assert strategy._tracked_estimator is estimator

    def test_swapped_estimator_clears_cache(self) -> None:
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        first = object()
        second = object()

        strategy._maybe_invalidate(cast(Any, _FakeDistr(first)))
        _put_dummy_ppf_cache_entry(strategy, fake_id=1)
        _put_dummy_ppf_cache_entry(strategy, fake_id=2)

        strategy._maybe_invalidate(cast(Any, _FakeDistr(second)))

        assert strategy._ppf_cache == {}
        assert strategy._tracked_estimator is second

    def test_distribution_without_estimator_attribute_does_not_crash(self) -> None:
        strategy = EmpiricalComputationStrategy(enable_caching=True)

        class _Bare:
            pass

        strategy._maybe_invalidate(cast(Any, _Bare()))
        assert strategy._tracked_estimator is None

        _put_dummy_ppf_cache_entry(strategy, fake_id=0)
        strategy._maybe_invalidate(cast(Any, _Bare()))
        assert 0 in strategy._ppf_cache


class TestVectorizedPpf:
    def _make_distr(self, sample_size: int = 300) -> Any:
        np = pytest.importorskip("numpy")
        from pysatl_core.distributions.empirical import EmpiricalDistribution

        rng = np.random.default_rng(0)
        sample = rng.normal(0.0, 1.0, sample_size)
        return EmpiricalDistribution(sample)

    def test_array_input_returns_array(self) -> None:
        np = pytest.importorskip("numpy")
        distr = self._make_distr()
        q = np.linspace(0.05, 0.95, 50)
        result = distr.calculate_characteristic(CharacteristicName.PPF, q)
        assert result.shape == (50,)

    def test_result_is_monotonic_nondecreasing(self) -> None:
        np = pytest.importorskip("numpy")
        distr = self._make_distr()
        q = np.linspace(0.01, 0.99, 200)
        result = distr.calculate_characteristic(CharacteristicName.PPF, q)
        assert np.all(np.diff(result) >= -1e-10)

    def test_scalar_input_returns_scalar(self) -> None:
        distr = self._make_distr()
        result = distr.calculate_characteristic(CharacteristicName.PPF, 0.5)
        assert np.ndim(result) == 0

    def test_matches_root_finding_within_tolerance(self) -> None:
        np = pytest.importorskip("numpy")
        from pysatl_core.distributions.empirical import EmpiricalDistribution
        from pysatl_core.distributions.strategies import DefaultComputationStrategy

        rng = np.random.default_rng(7)
        sample = rng.normal(0.0, 1.0, 400)

        fast = EmpiricalDistribution(sample)
        slow = EmpiricalDistribution(sample, computation_strategy=DefaultComputationStrategy())
        quantiles = np.linspace(0.1, 0.9, 9)
        fast_vals = fast.calculate_characteristic(CharacteristicName.PPF, quantiles)
        slow_vals = np.array(
            [
                float(np.squeeze(slow.calculate_characteristic(CharacteristicName.PPF, float(q))))
                for q in quantiles
            ]
        )
        assert np.allclose(fast_vals, slow_vals, atol=5e-2)

    def test_handles_extreme_quantiles_without_nan(self) -> None:
        np = pytest.importorskip("numpy")
        distr = self._make_distr()
        q = np.array([1e-7, 0.5, 1.0 - 1e-7])
        result = distr.calculate_characteristic(CharacteristicName.PPF, q)
        assert not np.any(np.isnan(result))

    def test_ppf_recomputed_after_estimator_swap(self) -> None:
        np = pytest.importorskip("numpy")
        from pysatl_core.distributions.empirical import (
            EmpiricalDistribution,
            ScipyGaussianKde,
        )

        rng = np.random.default_rng(3)
        sample = rng.normal(0.0, 1.0, 300)
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        distr = EmpiricalDistribution(sample, computation_strategy=strategy)

        first = distr.calculate_characteristic(CharacteristicName.PPF, 0.5)
        cached_after_first = strategy._ppf_cache.get(id(distr))
        assert cached_after_first is not None

        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        second = distr.calculate_characteristic(CharacteristicName.PPF, 0.5)
        cached_after_swap = strategy._ppf_cache.get(id(distr))

        assert cached_after_swap is not None
        assert cached_after_swap is not cached_after_first
        assert np.isfinite(first)
        assert np.isfinite(second)


class TestQueryMethodIntegration:
    def test_estimator_swap_clears_fitted_cache_via_query_method(self) -> None:
        np = pytest.importorskip("numpy")

        from pysatl_core.distributions.empirical import (
            EmpiricalDistribution,
            ScipyGaussianKde,
        )

        rng = np.random.default_rng(0)
        sample = rng.normal(0, 1, 200)
        strategy = EmpiricalComputationStrategy(enable_caching=True)
        distr = EmpiricalDistribution(sample, computation_strategy=strategy)

        strategy.query_method(CharacteristicName.PDF, distr)
        _put_dummy_ppf_cache_entry(strategy, fake_id=99999)
        assert 99999 in strategy._ppf_cache

        new_estimator = ScipyGaussianKde(bandwidth="silverman").fit(sample)
        distr._estimator = new_estimator

        strategy.query_method(CharacteristicName.PDF, distr)

        assert 99999 not in strategy._ppf_cache
        assert strategy._tracked_estimator is new_estimator
