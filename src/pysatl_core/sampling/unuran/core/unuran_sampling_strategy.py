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

from typing import TYPE_CHECKING, Any, Final

from pysatl_core.sampling.default import DefaultSamplingUnivariateStrategy
from pysatl_core.sampling.unuran.core.unuran_sampler import DefaultUnuranSampler
from pysatl_core.sampling.unuran.method_config import (
    UnuranMethodConfig,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import NumericArray


class DefaultUnuranSamplingStrategy:
    """
    Default UNU.RAN sampling strategy implementation.

    This strategy creates UNU.RAN samplers for distributions and converts
    the output to the standard Sample format.

    Notes
    -----
    - Supports caching of samplers to improve performance with repeated sampling
    """

    def __init__(
        self,
        config: UnuranMethodConfig | None = None,
    ):
        """
        Initialize the sampling strategy.

        Parameters
        ----------
        config : UnuranMethodConfig | None, optional
            Method configuration. If None, uses UnuranMethodConfig()
            with default values (AUTO method selection).
        """
        self._config_value: Final[UnuranMethodConfig] = config or UnuranMethodConfig()
        self._sampler: DefaultUnuranSampler | None = None

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray:
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

        if self._sampler is None:
            try:
                self._sampler = DefaultUnuranSampler(distr, self.config)
            except RuntimeError:
                return DefaultSamplingUnivariateStrategy().sample(n, distr, **options)

        return self._sampler.sample(n)

    @property
    def config(self) -> UnuranMethodConfig:
        """Method configuration."""
        return self._config_value
