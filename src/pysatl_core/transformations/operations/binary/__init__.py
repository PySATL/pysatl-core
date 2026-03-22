"""
Binary transformations for probability distributions.

This package provides pairwise transformed distributions such as
``X + Y``, ``X - Y``, ``X * Y``, and ``X / Y``.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.types import BinaryOperationName

from .base import BinaryDistribution, binary
from .division import DivisionBinaryDistribution
from .linear import LinearBinaryDistribution
from .multiplication import MultiplicationBinaryDistribution

__all__ = [
    "BinaryDistribution",
    "BinaryOperationName",
    "DivisionBinaryDistribution",
    "LinearBinaryDistribution",
    "MultiplicationBinaryDistribution",
    "binary",
]
