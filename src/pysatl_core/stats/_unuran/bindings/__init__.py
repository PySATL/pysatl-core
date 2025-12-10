"""
UNU.RAN C Bindings
==================

This module provides C bindings to the UNU.RAN library.

The actual C extension module will be implemented here. For now, this module
provides stub implementations that raise NotImplementedError.

In the final implementation, this module will contain:
- C extension module compiled from Cython or ctypes bindings
- Low-level functions for creating UNU.RAN generators
- Functions for sampling from distributions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.stats._unuran.api import (
        UnuranMethodConfig,
        UnuranSampler,
        UnuranSamplingStrategy,
    )

# Import core binding functions
from pysatl_core.stats._unuran.bindings._core import (
    create_sampler_impl,
    create_strategy_impl,
)

__all__ = [
    "create_sampler_impl",
    "create_strategy_impl",
]

