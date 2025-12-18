"""
Common fixtures and utilities for continuous distribution tests.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import math
from typing import Any

import numpy as np


class BaseDistributionTest:
    """Base class for all distribution families' tests"""

    # Precision for floating point comparisons
    CALCULATION_PRECISION = 1e-10

    @staticmethod
    def assert_arrays_almost_equal(
        actual: np.ndarray[Any, Any], expected: np.ndarray[Any, Any], precision: float | None = None
    ) -> None:
        """Helper method to assert arrays are almost equal."""
        if precision is None:
            precision = BaseDistributionTest.CALCULATION_PRECISION

        np.testing.assert_array_almost_equal(actual, expected, decimal=int(-math.log10(precision)))
