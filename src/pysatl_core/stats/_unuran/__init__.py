"""
UNU.RAN Integration API
=======================

This module defines the API for UNU.RAN (Universal Non-Uniform Random Number
Generator) integration. UNU.RAN provides efficient methods for generating
non-uniform random variates from various distributions.

The API described here will be implemented through C bindings to the UNU.RAN
library in the future. Currently, this module only provides the interface
specification for integration with the distribution sampling system.

Notes
-----
- UNU.RAN supports multiple sampling methods (inversion, rejection, etc.)
- Methods can be selected automatically or specified explicitly
- The API integrates with the :class:`~pysatl_core.distributions.strategies.SamplingStrategy`
  protocol
- Currently, only univariate distributions are supported

Examples
--------
Create a UNU.RAN sampling strategy for a distribution:

    >>> from pysatl_core.stats._unuran import (
    >>>     create_unuran_strategy, UnuranMethodConfig, UnuranMethod
    >>> )
    >>> config = UnuranMethodConfig(method=UnuranMethod.AUTO)
    >>> strategy = create_unuran_strategy(config)
    >>> sample = strategy.sample(n=1000, distr=my_distribution)
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.stats._unuran.api import (
    UnuranMethod,
    UnuranMethodConfig,
    UnuranSampler,
    UnuranSamplingStrategy,
)
from pysatl_core.stats._unuran.bindings import (
    DefaultUnuranSampler,
    DefaultUnuranSamplingStrategy,
)

__all__ = [
    "UnuranMethod",
    "UnuranMethodConfig",
    "UnuranSampler",
    "UnuranSamplingStrategy",
    "DefaultUnuranSampler",
    "DefaultUnuranSamplingStrategy",
]
