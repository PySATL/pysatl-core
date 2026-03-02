"""Core UNU.RAN sampler and sampling strategy implementations.

Exposes :class:`DefaultUnuranSampler`, which wraps the UNU.RAN C library via
CFFI and automatically selects a sampling method (PINV, NINV, DGT, …) based on
the available distribution characteristics, and
:class:`DefaultUnuranSamplingStrategy`, the high-level strategy that integrates
the sampler with the PySATL distribution protocol.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .unuran_sampler import DefaultUnuranSampler
from .unuran_sampling_strategy import DefaultUnuranSamplingStrategy

__all__ = [
    "DefaultUnuranSampler",
    "DefaultUnuranSamplingStrategy",
]
