"""
UNU.RAN Default Sampling Strategy
=================================

This module provides the default UNU.RAN sampling strategy implementation that
creates UNU.RAN samplers for distributions and converts the output to the
standard Sample format. The strategy supports caching of samplers to improve
performance with repeated sampling from the same distribution.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import copy
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.sampling import ArraySample
from pysatl_core.stats._unuran.api import UnuranMethodConfig, UnuranSamplingStrategy
from pysatl_core.stats._unuran.bindings._core.unuran_sampler import DefaultUnuranSampler

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.sampling import Sample
    from pysatl_core.stats._unuran.api import UnuranSampler


class DefaultUnuranSamplingStrategy(UnuranSamplingStrategy):
    """
    Default UNU.RAN sampling strategy implementation.

    This strategy creates UNU.RAN samplers for distributions and converts
    the output to the standard Sample format.

    Notes
    -----
    - Supports caching of samplers to improve performance with repeated sampling
    - By default, caching is enabled (use_cache=True)
    - When caching is enabled, the last sampler and a copy of the distribution
      are stored and reused if the same distribution object is used again
    """

    def __init__(
        self,
        default_config: UnuranMethodConfig | None = None,
        sampler_class: type[UnuranSampler] | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize the sampling strategy.

        Parameters
        ----------
        default_config : UnuranMethodConfig | None, optional
            Default method configuration. If None, uses UnuranMethodConfig()
            with default values (AUTO method selection).
        sampler_class : Type[UnuranSampler] | None, optional
            Class to use for creating samplers. If None, uses DefaultUnuranSampler.
        use_cache : bool, optional
            Whether to cache samplers for reuse. Default is True.
            When True, the last sampler and distribution are cached and reused
            if the same distribution object is used in subsequent calls.
        """
        self._default_config = default_config or UnuranMethodConfig()
        self._sampler_class: type[UnuranSampler] = sampler_class or DefaultUnuranSampler
        self._use_cache = use_cache
        self._cached_sampler: UnuranSampler | None = None
        self._cached_distribution: Distribution | None = None
        self._cached_distribution_copy: Distribution | None = None

    def sample(self, n: int, distr: Distribution, **options: Any) -> Sample:
        """
        Generate a sample from the distribution using UNU.RAN.

        Parameters
        ----------
        n : int
            Number of observations to draw.
        distr : Distribution
            The distribution to sample from.
        **options : Any
            Additional options that may override the default configuration:
            - ``method``: override the sampling method
            - ``seed``: override the random seed
            - Other method-specific parameters

        Returns
        -------
        Sample
            A 2D sample of shape ``(n, 1)`` for univariate distributions.

        Raises
        ------
        RuntimeError
            If the distribution type is not supported, or if UNU.RAN
            cannot create a sampler with the available characteristics.
        ValueError
            If the configuration is invalid.
        """
        if n < 0:
            raise ValueError(f"Number of samples must be non-negative, got {n}")

        if self._use_cache:
            sampler = self._maybe_get_cached_sampler(distr)
            if sampler is None:
                sampler = self._create_and_cache_sampler(distr, **options)
        else:
            sampler = self._sampler_class(distr, self._default_config, **options)

        samples_1d = sampler.sample(n)

        samples_2d = samples_1d.reshape(-1, 1)

        return ArraySample(samples_2d)

    def _maybe_get_cached_sampler(self, distr: Distribution) -> UnuranSampler | None:
        """Return cached sampler if present and distribution matches."""
        cached_sampler = self._cached_sampler
        cached_distribution = self._cached_distribution

        if cached_sampler is None or cached_distribution is None:
            return None

        if distr is cached_distribution:
            return cached_sampler

        cached_distribution_copy = self._cached_distribution_copy
        if cached_distribution_copy is not None:
            try:
                if distr == cached_distribution_copy:
                    return cached_sampler
            except (TypeError, ValueError, AttributeError):
                return None

        return None

    def _create_and_cache_sampler(self, distr: Distribution, **options: Any) -> UnuranSampler:
        """Create sampler instance and populate cache."""
        sampler = self._sampler_class(distr, self._default_config, **options)
        self._cached_sampler = sampler
        self._cached_distribution = distr
        try:
            self._cached_distribution_copy = copy.deepcopy(distr)
        except (TypeError, ValueError, AttributeError):
            self._cached_distribution_copy = None
        return sampler

    @property
    def default_config(self) -> UnuranMethodConfig:
        """Default method configuration."""
        return self._default_config
