"""
Monotone-spline approximation for PPF.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.interpolate import PchipInterpolator

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.transformations.approximations.linear_interpolations._common import (
    evaluate_on_grid,
    get_analytical_method,
    validate_univariate_continuous,
)
from pysatl_core.types import CharacteristicName, ComputationFunc, NumericArray

if TYPE_CHECKING:
    from pysatl_core.transformations.distribution import DerivedDistribution


class PPFMonotoneSplineApproximation:
    """
    Approximate PPF on a finite probability interval with monotone splines.

    Parameters
    ----------
    n_grid : int, default=513
        Number of interpolation nodes.
    lower_limit : float, default=0.0
        Left bound of the probability interval used for interpolation.
    upper_limit : float, default=1.0
        Right bound of the probability interval used for interpolation.
    """

    def __init__(
        self,
        *,
        n_grid: int = 513,
        lower_limit: float = 0.0,
        upper_limit: float = 1.0,
    ) -> None:
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")
        if not (0.0 <= lower_limit < upper_limit <= 1.0):
            raise ValueError("PPF limits must satisfy 0 <= lower_limit < upper_limit <= 1.")

        self._n_grid = int(n_grid)
        self._lower_limit = float(lower_limit)
        self._upper_limit = float(upper_limit)

    def approximate(
        self,
        distribution: DerivedDistribution,
        **options: Any,
    ) -> AnalyticalComputation[Any, Any]:
        """
        Approximate PPF for a distribution.
        """
        validate_univariate_continuous(distribution)
        source_method = get_analytical_method(distribution, CharacteristicName.PPF)

        grid = np.linspace(self._lower_limit, self._upper_limit, self._n_grid, dtype=float)
        values = evaluate_on_grid(
            source_method,
            grid,
            characteristic_name=CharacteristicName.PPF,
            **options,
        )
        values = self._regularize_monotone_ppf(values)
        spline = PchipInterpolator(grid, values, extrapolate=False)
        lower_value = float(values[0])
        upper_value = float(values[-1])

        def _ppf(data: Any, /, **_kwargs: Any) -> Any:
            array = np.asarray(data, dtype=float)
            clipped = np.clip(array, self._lower_limit, self._upper_limit)
            result = np.asarray(spline(clipped), dtype=float)
            result = np.where(array <= self._lower_limit, lower_value, result)
            result = np.where(array >= self._upper_limit, upper_value, result)
            return cast(NumericArray, result)

        return AnalyticalComputation(
            target=CharacteristicName.PPF,
            func=cast(ComputationFunc[Any, Any], _ppf),
        )

    @staticmethod
    def _regularize_monotone_ppf(values: NumericArray) -> NumericArray:
        """
        Regularize PPF samples to finite monotone values.
        """
        regularized = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(regularized)
        if not np.any(finite_mask):
            raise ValueError("Could not build PPF approximation: all grid values are non-finite.")

        if not np.all(finite_mask):
            indices = np.arange(regularized.size, dtype=float)
            regularized = np.interp(indices, indices[finite_mask], regularized[finite_mask])

        regularized = np.maximum.accumulate(regularized)
        return cast(NumericArray, regularized)


__all__ = [
    "PPFMonotoneSplineApproximation",
]
