"""
Parametric Families module for working with statistical distribution families.

This package provides a comprehensive framework for defining, managing, and
working with parametric families of statistical distributions.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov, Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from .configuration import configure_families_register
from .distribution import ParametricFamilyDistribution
from .parametric_family import ParametricFamily
from .parametrizations import (
    Parametrization,
    ParametrizationConstraint,
    constraint,
    parametrization,
)
from .registry import ParametricFamilyRegister

__all__ = [
    "ParametricFamilyRegister",
    "ParametrizationConstraint",
    "Parametrization",
    "ParametricFamily",
    "ParametricFamilyDistribution",
    "constraint",
    "parametrization",
    "configure_families_register",
]
