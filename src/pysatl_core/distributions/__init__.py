"""
Distributions subpackage

Interfaces and default implementations for probability distributions used by
PySATL Core:

- distribution protocol (:mod:`.distribution`);
- numerical fitters (:mod:`.fitters`);
- characteristic graph registry (:mod:`.registry`);
- sampling protocol and array-backed samples (:mod:`.sampling`);
- pluggable strategies (:mod:`.strategies`).
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"
from .computation import (
    AnalyticalComputation,
    ComputationMethod,
    FittedComputationMethod,
)
from .distribution import Distribution
from .registry import DEFAULT_COMPUTATION_KEY
from .sampling import ArraySample, Sample
from .strategies import (
    ComputationStrategy,
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
    SamplingStrategy,
)

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
    "DEFAULT_COMPUTATION_KEY",
]
