"""
Interpolation-based approximation methods for specific characteristics.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .cdf import CDFMonotoneSplineApproximation
from .pdf import PDFLinearInterpolationApproximation
from .ppf import PPFMonotoneSplineApproximation

__all__ = [
    "PDFLinearInterpolationApproximation",
    "CDFMonotoneSplineApproximation",
    "PPFMonotoneSplineApproximation",
]
