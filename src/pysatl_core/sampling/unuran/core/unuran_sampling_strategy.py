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

    def __deepcopy__(self, memo: dict[int, Any]) -> DefaultUnuranSamplingStrategy:
        """
        Return an uninitialised copy that shares only the configuration.

        A cached sampler cannot be copied and must not be shared.  It holds the
        CFFI handles for the UNU.RAN generator (``_ffi``, ``_lib`` and raw
        ``_CDataBase`` pointers), which ``deepcopy`` cannot duplicate -- it
        falls back to the pickle protocol and fails with ``TypeError: cannot
        pickle '_cffi_backend.FFI' object``.  That failure is the lesser evil:
        two Python objects owning one C generator would run its teardown twice.

        Dropping the sampler is also the right semantics rather than a
        workaround.  A clone is made because something about the distribution
        is changing (see ``Distribution.with_strategies``), so a generator
        built on the original's characteristics would be stale anyway; the
        clone builds its own on first :meth:`sample`.

        Notes
        -----
        The configuration object is shared, not copied.  It is a frozen
        dataclass, so its fields cannot be rebound, but ``method_params`` is a
        plain dict and mutating it in place is visible from both strategies.
        Nothing in the library mutates it today.
        """
        new = DefaultUnuranSamplingStrategy(config=self._config_value)
        memo[id(self)] = new
        return new

    def invalidate(self) -> None:
        """
        Drop the cached UNURAN sampler.

        The next call to :meth:`sample` will rebuild a fresh sampler from the
        distribution's current characteristics. Use this when the underlying
        distribution state has changed (e.g. an empirical method has been
        swapped) and the existing UNURAN generator was built on stale data —
        keeping it would silently produce samples from the old distribution.

        Notes
        -----
        Releases the only strong reference held by the strategy. Any external
        code that captured ``self._sampler`` directly will keep its old sampler
        alive and continue sampling from the previous distribution; such
        references must be re-acquired by the caller.
        """
        self._sampler = None

    @property
    def config(self) -> UnuranMethodConfig:
        """Method configuration."""
        return self._config_value
