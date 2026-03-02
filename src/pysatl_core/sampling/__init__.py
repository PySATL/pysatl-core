"""
Public sampling interface re-exporting UNURAN-based defaults.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .unuran import (
    DefaultUnuranSampler,
    DefaultUnuranSamplingStrategy,
    UnuranMethod,
    UnuranMethodConfig,
)

SamplingMethod = UnuranMethod
SamplingMethodConfig = UnuranMethodConfig
DefaultSampler = DefaultUnuranSampler
DefaultSamplingStrategy = DefaultUnuranSamplingStrategy

__all__ = [
    "SamplingMethod",
    "SamplingMethodConfig",
    "DefaultSampler",
    "DefaultSamplingStrategy",
]
