"""Discrete empirical CDF estimator."""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.support import ExplicitTableDiscreteSupport
from pysatl_core.types import (
    CharacteristicName,
    ComputationFunc,
    DistributionType,
    GenericCharacteristicName,
    UnivariateDiscrete,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support


class EmpiricalCdf:
    """Estimate a discrete distribution assigning each observation mass ``1 / n``."""

    @property
    def distribution_type(self) -> DistributionType:
        return UnivariateDiscrete

    def resolve_support(self, sample: NDArray[np.float64], support: Support | None) -> Support:
        observed = ExplicitTableDiscreteSupport(sample)
        if support is None:
            return observed
        if not isinstance(support, ExplicitTableDiscreteSupport) or not np.array_equal(
            support.points, observed.points
        ):
            raise ValueError("EmpiricalCdf support must contain exactly the observed values.")
        return support

    def fit(
        self, sample: NDArray[np.float64]
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        values, counts = np.unique(sample, return_counts=True)
        masses = counts / sample.size
        cumulative = np.concatenate(([0.0], np.cumsum(masses)))
        cumulative[-1] = 1.0

        def cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
            points = np.asarray(x, dtype=float)
            positions = np.searchsorted(values, points, side="right")
            return cast(
                NDArray[np.float64], np.where(np.isnan(points), np.nan, cumulative[positions])
            )

        def pmf(x: NDArray[np.float64]) -> NDArray[np.float64]:
            points = np.asarray(x, dtype=float)
            positions = np.searchsorted(values, points, side="left")
            clipped = np.minimum(positions, values.size - 1)
            found = (positions < values.size) & (values[clipped] == points)
            return cast(
                NDArray[np.float64],
                np.where(np.isnan(points), np.nan, np.where(found, masses[clipped], 0.0)),
            )

        return {
            CharacteristicName.CDF: AnalyticalComputation(
                CharacteristicName.CDF, cast(ComputationFunc[Any, Any], cdf)
            ),
            CharacteristicName.PMF: AnalyticalComputation(
                CharacteristicName.PMF, cast(ComputationFunc[Any, Any], pmf)
            ),
        }
