"""
Approximation utilities for transformed distributions.

This subpackage contains approximation interfaces and concrete
approximators that can materialize analytical characteristics for
complex transformation trees.
"""

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .approximation import CharacteristicApproximationMethod
from .linear_interpolations import (
    CDFMonotoneSplineApproximation,
    PDFLinearInterpolationApproximation,
    PPFMonotoneSplineApproximation,
)

__all__ = [
    "CharacteristicApproximationMethod",
    "PDFLinearInterpolationApproximation",
    "CDFMonotoneSplineApproximation",
    "PPFMonotoneSplineApproximation",
]
