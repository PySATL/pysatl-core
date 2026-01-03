"""
Core abstractions and implementations for probability distributions
used throughout PySATL.

This package provides:
 - Distribution protocol and computational primitives
 - Characteristic graph registry with constraint-based filtering
 - Sampling interfaces and strategies
 - Support structures for distribution domains
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .computation import (
    AnalyticalComputation,
    ComputationMethod,
    FittedComputationMethod,
)
from .distribution import Distribution
from .registry import *
from .registry import __all__ as _registry_all
from .sampling import ArraySample, Sample
from .strategies import (
    ComputationStrategy,
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
    SamplingStrategy,
)
from .support import *
from .support import __all__ as _support_all

__all__ = [
    # computation primitives
    "AnalyticalComputation",
    "ComputationMethod",
    "FittedComputationMethod",
    # distribution
    "Distribution",
    # sampling
    "Sample",
    "ArraySample",
    # strategies
    "ComputationStrategy",
    "DefaultComputationStrategy",
    "SamplingStrategy",
    "DefaultSamplingUnivariateStrategy",
    # registry
    *_registry_all,
    # support
    *_support_all,
]

del _registry_all
del _support_all
