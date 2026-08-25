"""
Empirical distribution and its computation strategy.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.distributions.empirical.distribution import (
    EmpiricalDistribution,
    EmpiricalMethod,
    FittedEmpirical,
    ScipyGaussianKde,
)
from pysatl_core.distributions.empirical.strategy import EmpiricalComputationStrategy

__all__ = [
    "EmpiricalComputationStrategy",
    "EmpiricalDistribution",
    "EmpiricalMethod",
    "FittedEmpirical",
    "ScipyGaussianKde",
]
