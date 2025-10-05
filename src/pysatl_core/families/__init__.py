"""
Parametric Families module for working with statistical distribution families.

This package provides a comprehensive framework for defining, managing, and
working with parametric families of statistical distributions. It supports
multiple parameterizations, constraint validation, and automatic conversion
between different parameter formats.
"""

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    ParametrizationConstraint,
    constraint,
    parametrization,
)
from pysatl_core.families.registry import ParametricFamilyRegister

__all__ = [
    "ParametricFamilyRegister",
    "ParametrizationConstraint",
    "Parametrization",
    "ParametricFamily",
    "ParametricFamilyDistribution",
    "constraint",
    "parametrization",
]
