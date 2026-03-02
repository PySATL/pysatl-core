"""
Unit tests for pysatl_core.sampling.unuran.core.unuran_sampling_strategy

Tests DefaultUnuranSamplingStrategy:
  - initialization with default and custom parameters
  - config property
  - sample delegation to the sampler
  - fallback to DefaultSamplingUnivariateStrategy on sampler creation failure
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.sampling.unuran.core.unuran_sampling_strategy import (
    DefaultUnuranSamplingStrategy,
)
from pysatl_core.sampling.unuran.method_config import UnuranMethod, UnuranMethodConfig
from pysatl_core.types import CharacteristicName, Kind
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


def _make_continuous_distr() -> StandaloneEuclideanUnivariateDistribution:
    """Build a minimal continuous distribution with a constant PDF."""
    from collections.abc import Callable
    from typing import Any

    from mypy_extensions import KwArg

    pdf_func = cast(Callable[[float, KwArg(Any)], float], lambda x, **_: 1.0)
    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=[
            AnalyticalComputation[float, float](target=CharacteristicName.PDF, func=pdf_func)
        ],
        support=ContinuousSupport(0.0, 1.0),
    )


class _StubSampler:
    """Minimal sampler stub that records sample calls."""

    def __init__(self, distr, config):
        self._n_called = 0
        self._fixed_value = 0.5

    def sample(self, n: int) -> np.ndarray:
        self._n_called += 1
        return np.full(n, self._fixed_value)

    @property
    def method(self) -> UnuranMethod:
        return UnuranMethod.PINV


class _FailingSampler:
    """Sampler stub whose __init__ always raises RuntimeError."""

    def __init__(self, distr, config):
        raise RuntimeError("sampler init failed")

    def sample(self, n: int) -> np.ndarray:  # pragma: no cover
        return np.array([])

    @property
    def method(self) -> UnuranMethod:  # pragma: no cover
        return UnuranMethod.PINV


class TestDefaultUnuranSamplingStrategyInit:
    """Tests for DefaultUnuranSamplingStrategy.__init__."""

    def test_default_config_is_auto_method(self) -> None:
        """Without explicit config, the strategy uses AUTO method configuration."""
        strategy = DefaultUnuranSamplingStrategy()

        assert strategy.config.method == UnuranMethod.AUTO

    def test_custom_config_is_stored(self) -> None:
        """A supplied config is stored and exposed via the config property."""
        config = UnuranMethodConfig(method=UnuranMethod.PINV, use_pdf=False)
        strategy = DefaultUnuranSamplingStrategy(config=config)

        assert strategy.config is config
        assert strategy.config.method == UnuranMethod.PINV

    def test_sampler_is_not_created_until_first_sample_call(self) -> None:
        """No sampler instance is created in __init__ — lazy creation on first call."""
        strategy = DefaultUnuranSamplingStrategy()

        assert strategy._sampler is None


class TestConfigProperty:
    """Tests for DefaultUnuranSamplingStrategy.config property."""

    def test_config_returns_same_object_on_repeated_access(self) -> None:
        """The config property is immutable — it returns the same object each time."""
        config = UnuranMethodConfig(method=UnuranMethod.NINV)
        strategy = DefaultUnuranSamplingStrategy(config=config)

        assert strategy.config is strategy.config

    def test_config_method_params_defaults_to_empty_dict(self) -> None:
        """method_params defaults to {} when None is passed to UnuranMethodConfig."""
        config = UnuranMethodConfig(method_params=None)
        strategy = DefaultUnuranSamplingStrategy(config=config)

        assert strategy.config.method_params == {}


_SAMPLER_MODULE = "pysatl_core.sampling.unuran.core.unuran_sampling_strategy.DefaultUnuranSampler"


class TestSampleMethod:
    """Tests for DefaultUnuranSamplingStrategy.sample."""

    def test_negative_n_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Requesting a negative number of samples raises ValueError."""
        monkeypatch.setattr(_SAMPLER_MODULE, _StubSampler)
        strategy = DefaultUnuranSamplingStrategy()
        distr = _make_continuous_distr()

        with pytest.raises(ValueError, match="non-negative"):
            strategy.sample(-1, distr)

    def test_zero_samples_returns_empty_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Requesting 0 samples returns an empty array."""
        monkeypatch.setattr(_SAMPLER_MODULE, _StubSampler)
        strategy = DefaultUnuranSamplingStrategy()
        distr = _make_continuous_distr()

        result = strategy.sample(0, distr)

        assert len(result) == 0

    def test_samples_are_generated_by_sampler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """sample() delegates to the underlying sampler and returns its output."""
        monkeypatch.setattr(_SAMPLER_MODULE, _StubSampler)
        strategy = DefaultUnuranSamplingStrategy()
        distr = _make_continuous_distr()

        result = strategy.sample(5, distr)

        assert len(result) == 5

    def test_sampler_is_created_on_first_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sampler is lazily created on the first sample() call."""
        monkeypatch.setattr(_SAMPLER_MODULE, _StubSampler)
        strategy = DefaultUnuranSamplingStrategy()
        distr = _make_continuous_distr()

        assert strategy._sampler is None
        strategy.sample(3, distr)
        assert strategy._sampler is not None

    def test_sampler_is_reused_across_multiple_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same sampler instance is reused for subsequent sample() calls."""
        monkeypatch.setattr(_SAMPLER_MODULE, _StubSampler)
        strategy = DefaultUnuranSamplingStrategy()
        distr = _make_continuous_distr()

        strategy.sample(2, distr)
        sampler_first = strategy._sampler

        strategy.sample(2, distr)
        sampler_second = strategy._sampler

        assert sampler_first is sampler_second

    def test_falls_back_to_default_strategy_when_sampler_init_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to DefaultSamplingUnivariateStrategy when sampler construction raises."""
        monkeypatch.setattr(_SAMPLER_MODULE, _FailingSampler)
        strategy = DefaultUnuranSamplingStrategy()

        # The distribution has PPF-compatible fallback (inverse-transform sampling)
        # Provide a PPF for the fallback to work
        from collections.abc import Callable
        from typing import Any

        from mypy_extensions import KwArg

        ppf_func = cast(Callable[[float, KwArg(Any)], float], lambda q, **_: q)
        distr_with_ppf = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](target=CharacteristicName.PPF, func=ppf_func)
            ],
            support=ContinuousSupport(0.0, 1.0),
        )

        # Should not raise — falls back to DefaultSamplingUnivariateStrategy
        result = strategy.sample(10, distr_with_ppf)

        assert len(result) == 10
