"""
Monotone-spline approximation for CDF.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.interpolate import PchipInterpolator

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.transformations.approximations.linear_interpolations._common import (
    evaluate_on_grid,
    get_analytical_method,
    validate_univariate_continuous,
)
from pysatl_core.types import CharacteristicName, ComputationFunc, NumericArray

if TYPE_CHECKING:
    from pysatl_core.transformations.distribution import DerivedDistribution


class CDFMonotoneSplineApproximation:
    """
    Approximate CDF on a finite domain with monotone cubic splines.

    Parameters
    ----------
    n_grid : int, default=513
        Number of interpolation nodes.
    lower_limit_prob : float, default=1e-6
        Left-tail probability used to determine finite lower domain
        boundary ``x_left`` such that ``CDF(x_left) <= lower_limit_prob``.
    upper_limit_prob : float, default=1e-6
        Right-tail probability used to determine finite upper domain
        boundary ``x_right`` such that ``CDF(x_right) >= 1 - upper_limit_prob``.
    max_search_steps : int, default=80
        Maximum number of geometric expansion steps when searching for
        finite domain boundaries.
    """

    def __init__(
        self,
        *,
        n_grid: int = 513,
        lower_limit_prob: float = 1e-6,
        upper_limit_prob: float = 1e-6,
        max_search_steps: int = 80,
    ) -> None:
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")
        if not (0.0 <= lower_limit_prob < 0.5):
            raise ValueError("lower_limit_prob must satisfy 0 <= p < 0.5.")
        if not (0.0 <= upper_limit_prob < 0.5):
            raise ValueError("upper_limit_prob must satisfy 0 <= p < 0.5.")
        if max_search_steps < 1:
            raise ValueError("max_search_steps must be positive.")

        self._n_grid = int(n_grid)
        self._lower_limit_prob = float(lower_limit_prob)
        self._upper_limit_prob = float(upper_limit_prob)
        self._max_search_steps = int(max_search_steps)

    def approximate(
        self,
        distribution: DerivedDistribution,
        **options: Any,
    ) -> AnalyticalComputation[Any, Any]:
        """
        Approximate CDF for a distribution.
        """
        validate_univariate_continuous(distribution)
        source_method = get_analytical_method(distribution, CharacteristicName.CDF)
        lower_limit, upper_limit = self._resolve_limits(distribution, source_method, **options)

        grid = np.linspace(lower_limit, upper_limit, self._n_grid, dtype=float)
        values = evaluate_on_grid(
            source_method,
            grid,
            characteristic_name=CharacteristicName.CDF,
            **options,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = np.clip(values, 0.0, 1.0)
        values = np.maximum.accumulate(values)
        if float(values[-1] - values[0]) <= 0.0:
            raise ValueError("Could not build monotone CDF approximation: degenerate grid values.")

        spline = PchipInterpolator(grid, values, extrapolate=False)

        def _cdf(data: Any, /, **_kwargs: Any) -> Any:
            array = np.asarray(data, dtype=float)
            result = np.asarray(spline(array), dtype=float)
            result = np.where(array < lower_limit, 0.0, result)
            result = np.where(array > upper_limit, 1.0, result)
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            return cast(NumericArray, result)

        return AnalyticalComputation(
            target=CharacteristicName.CDF,
            func=cast(ComputationFunc[Any, Any], _cdf),
        )

    def _resolve_limits(
        self,
        distribution: DerivedDistribution,
        cdf_method: AnalyticalComputation[Any, Any],
        **options: Any,
    ) -> tuple[float, float]:
        """
        Resolve finite interpolation limits using support and tail probabilities.
        """
        support = distribution.support
        left: float | None = None
        right: float | None = None

        if isinstance(support, ContinuousSupport):
            left = float(support.left) if np.isfinite(support.left) else None
            right = float(support.right) if np.isfinite(support.right) else None

        if left is None:
            left = self._search_lower_limit(cdf_method, **options)
        if right is None:
            right = self._search_upper_limit(cdf_method, **options)

        if not np.isfinite(left) or not np.isfinite(right) or left >= right:
            raise ValueError("Failed to resolve finite CDF interpolation limits.")

        return left, right

    def _search_lower_limit(
        self, cdf_method: AnalyticalComputation[Any, Any], **options: Any
    ) -> float:
        """
        Find finite lower bound satisfying tail-probability constraint.
        """
        magnitudes = np.power(2.0, np.arange(self._max_search_steps, dtype=float))
        candidates = -magnitudes
        values = evaluate_on_grid(
            cdf_method,
            cast(NumericArray, candidates),
            characteristic_name=CharacteristicName.CDF,
            **options,
        )
        mask = values <= self._lower_limit_prob
        if not np.any(mask):
            raise ValueError(
                "Failed to infer lower interpolation limit from CDF and lower_limit_prob."
            )
        return float(candidates[int(np.argmax(mask))])

    def _search_upper_limit(
        self, cdf_method: AnalyticalComputation[Any, Any], **options: Any
    ) -> float:
        """
        Find finite upper bound satisfying tail-probability constraint.
        """
        target = 1.0 - self._upper_limit_prob
        candidates = np.power(2.0, np.arange(self._max_search_steps, dtype=float))
        values = evaluate_on_grid(
            cdf_method,
            cast(NumericArray, candidates),
            characteristic_name=CharacteristicName.CDF,
            **options,
        )
        mask = values >= target
        if not np.any(mask):
            raise ValueError(
                "Failed to infer upper interpolation limit from CDF and upper_limit_prob."
            )
        return float(candidates[int(np.argmax(mask))])


__all__ = [
    "CDFMonotoneSplineApproximation",
]
