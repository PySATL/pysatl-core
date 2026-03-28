"""
Linear interpolation approximation for PDF.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np

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


class PDFLinearInterpolationApproximation:
    """
    Approximate PDF with piecewise-linear interpolation on a finite grid.

    Parameters
    ----------
    n_grid : int, default=513
        Number of interpolation nodes.
    lower_limit : float or None, optional
        Left interpolation bound. If ``None``, it is inferred from finite
        support when available, otherwise a large negative constant is used.
    upper_limit : float or None, optional
        Right interpolation bound. If ``None``, it is inferred from finite
        support when available, otherwise a large positive constant is used.
    """

    _AUTO_LOWER_LIMIT = -1_000.0
    _AUTO_UPPER_LIMIT = 1_000.0

    def __init__(
        self,
        *,
        n_grid: int = 513,
        lower_limit: float | None = None,
        upper_limit: float | None = None,
    ) -> None:
        if n_grid < 2:
            raise ValueError("n_grid must be at least 2.")
        self._n_grid = int(n_grid)
        self._lower_limit = lower_limit
        self._upper_limit = upper_limit

    def approximate(
        self,
        distribution: DerivedDistribution,
        **options: Any,
    ) -> AnalyticalComputation[Any, Any]:
        """
        Approximate PDF for a distribution.
        """
        validate_univariate_continuous(distribution)

        lower_limit, upper_limit = self._resolve_limits(distribution)
        grid = np.linspace(lower_limit, upper_limit, self._n_grid, dtype=float)
        source_method = get_analytical_method(distribution, CharacteristicName.PDF)
        values = evaluate_on_grid(
            source_method,
            grid,
            characteristic_name=CharacteristicName.PDF,
            **options,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.clip(values, 0.0, None)

        def _pdf(data: Any, /, **_kwargs: Any) -> Any:
            array = np.asarray(data, dtype=float)
            result = np.interp(array, grid, values, left=0.0, right=0.0)
            return cast(NumericArray, np.asarray(result, dtype=float))

        return AnalyticalComputation(
            target=CharacteristicName.PDF,
            func=cast(ComputationFunc[Any, Any], _pdf),
        )

    def _resolve_limits(self, distribution: DerivedDistribution) -> tuple[float, float]:
        """
        Resolve interpolation limits.
        """
        lower = self._lower_limit
        upper = self._upper_limit

        support = distribution.support
        if isinstance(support, ContinuousSupport):
            if lower is None and np.isfinite(support.left):
                lower = float(support.left)
            if upper is None and np.isfinite(support.right):
                upper = float(support.right)

        lower = self._AUTO_LOWER_LIMIT if lower is None else float(lower)
        upper = self._AUTO_UPPER_LIMIT if upper is None else float(upper)

        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "Interpolation limits must be finite and satisfy lower_limit < upper_limit."
            )

        return lower, upper


__all__ = [
    "PDFLinearInterpolationApproximation",
]
