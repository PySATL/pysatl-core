from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.sampling import ArraySample
from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig, UnuranSampler
from pysatl_core.stats._unuran.bindings._core.unuran_sampling_strategy import (
    DefaultUnuranSamplingStrategy,
)


class DummySampler(UnuranSampler):
    """Minimal sampler stub recording instantiations and sample calls."""

    instantiations: list[DummySampler] = []

    def __init__(self, distr: Distribution, config: UnuranMethodConfig, **options: Any) -> None:
        self.distr = distr
        self.config = config
        self.options = options
        self.sample_calls: list[int] = []
        self._method = options.get("method_override", UnuranMethod.PINV)
        self._is_initialized = True
        DummySampler.instantiations.append(self)

    def sample(self, n: int) -> npt.NDArray[np.float64]:
        self.sample_calls.append(n)
        offset = float(self.options.get("offset", 0.0))
        return np.arange(n, dtype=float) + offset

    def reset(self, seed: int | None = None) -> None:  # pragma: no cover - simple stub
        self._is_initialized = True

    @property
    def method(self) -> UnuranMethod:
        return self._method

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized


@pytest.fixture(autouse=True)
def reset_dummy_sampler() -> Generator[None, None, None]:
    """Reset sampler instantiation tracking between tests."""
    DummySampler.instantiations.clear()
    yield
    DummySampler.instantiations.clear()


@dataclass
class DummyDistribution:
    """Simple distribution stub with identifier-based equality."""

    identifier: int
    eq_raises: bool = False

    def __eq__(self, other: object) -> bool:
        if self.eq_raises:
            raise TypeError("Equality not supported")
        if not isinstance(other, DummyDistribution):
            return NotImplemented
        return self.identifier == other.identifier

    def __deepcopy__(self, memo: dict[int, object]) -> DummyDistribution:
        return DummyDistribution(self.identifier, self.eq_raises)


class NoDeepcopyDistribution(DummyDistribution):
    """Distribution stub whose deepcopy operation fails."""

    def __deepcopy__(self, memo: dict[int, object]) -> NoDeepcopyDistribution:
        raise TypeError("Cannot deepcopy")


DummySamplerType = cast(type[UnuranSampler], DummySampler)


def as_distribution(distr: DummyDistribution) -> Distribution:
    """Helper to satisfy typing when interacting with strategy APIs."""
    return cast(Distribution, distr)


class TestDefaultUnuranSamplingStrategy:
    """Tests covering sampling workflow, caching helpers, and configuration."""

    def test_sample_returns_array_sample_with_reshaped_data(self) -> None:
        """sample() should reshape 1D sampler output into ArraySample column vector."""
        strategy = DefaultUnuranSamplingStrategy(sampler_class=DummySamplerType, use_cache=False)
        distr = DummyDistribution(identifier=1)

        result = strategy.sample(3, as_distribution(distr), offset=1.5)

        assert isinstance(result, ArraySample)
        expected = (np.arange(3, dtype=float) + 1.5).reshape(-1, 1)
        np.testing.assert_array_equal(result.array, expected)
        assert DummySampler.instantiations[0].options["offset"] == 1.5

    def test_sample_negative_n_raises_value_error(self) -> None:
        """sampling negative number of observations must raise ValueError."""
        strategy = DefaultUnuranSamplingStrategy(sampler_class=DummySamplerType)
        with pytest.raises(ValueError, match="non-negative"):
            strategy.sample(-1, as_distribution(DummyDistribution(identifier=1)))

    def test_sample_without_cache_creates_new_sampler_each_time(self) -> None:
        """When caching disabled, each call should instantiate a fresh sampler."""
        strategy = DefaultUnuranSamplingStrategy(sampler_class=DummySamplerType, use_cache=False)
        distr = DummyDistribution(identifier=1)

        strategy.sample(2, as_distribution(distr))
        strategy.sample(2, as_distribution(distr))

        assert len(DummySampler.instantiations) == 2

    def test_sample_with_cache_reuses_cached_sampler(self) -> None:
        """Caching enabled should reuse sampler for repeated sampling of same distribution."""
        strategy = DefaultUnuranSamplingStrategy(sampler_class=DummySamplerType, use_cache=True)
        distr = DummyDistribution(identifier=2)

        strategy.sample(3, as_distribution(distr))
        strategy.sample(2, as_distribution(distr))

        assert len(DummySampler.instantiations) == 1
        assert DummySampler.instantiations[0].sample_calls == [3, 2]

    def test_maybe_get_cached_sampler_uses_distribution_copy_equality(self) -> None:
        """Cache lookup should fallback to deepcopy comparison when object differs."""
        strategy = DefaultUnuranSamplingStrategy(sampler_class=DummySamplerType, use_cache=True)
        original = DummyDistribution(identifier=3)
        strategy._create_and_cache_sampler(as_distribution(original))

        equivalent = DummyDistribution(identifier=3)
        cached = strategy._maybe_get_cached_sampler(as_distribution(equivalent))

        assert cached is strategy._cached_sampler

    def test_maybe_get_cached_sampler_returns_none_on_equality_error(self) -> None:
        """Equality errors while checking cached copy should be swallowed and return None."""
        strategy = DefaultUnuranSamplingStrategy(sampler_class=DummySamplerType, use_cache=True)
        strategy._cached_sampler = DummySampler(
            as_distribution(DummyDistribution(1)), UnuranMethodConfig()
        )
        strategy._cached_distribution = as_distribution(DummyDistribution(identifier=4))
        strategy._cached_distribution_copy = as_distribution(
            DummyDistribution(identifier=5, eq_raises=True)
        )

        uncached_distribution = as_distribution(DummyDistribution(identifier=6))
        result = strategy._maybe_get_cached_sampler(uncached_distribution)

        assert result is None

    def test_create_and_cache_sampler_handles_deepcopy_errors(self) -> None:
        """_create_and_cache_sampler should tolerate deepcopy failures and still cache sampler."""
        strategy = DefaultUnuranSamplingStrategy(sampler_class=DummySamplerType, use_cache=True)
        distr = NoDeepcopyDistribution(identifier=7)

        sampler = strategy._create_and_cache_sampler(as_distribution(distr))

        assert sampler is strategy._cached_sampler
        assert strategy._cached_distribution is distr
        assert strategy._cached_distribution_copy is None

    def test_default_config_property_returns_same_object(self) -> None:
        """default_config property must expose the original configuration object."""
        config = UnuranMethodConfig()
        strategy = DefaultUnuranSamplingStrategy(default_config=config)

        assert strategy.default_config is config
