"""
Parameter estimation for parametric families.

For now this package holds only the exception taxonomy that the built-in
families raise from their closed-form maximum likelihood solutions.  The
estimator itself is added on top of it.
"""

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core.estimation.errors import (
    FitDataError,
    InsufficientDataError,
    MLEError,
)

__all__ = [
    "MLEError",
    "FitDataError",
    "InsufficientDataError",
]
