from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pysatl_core.distributions import (
    AnalyticalComputation,
    ComputationStrategy,
    Distribution,
    SamplingStrategy,
)
from pysatl_core.distributions.distribution import _KEEP
from pysatl_core.distributions.support import Support
from pysatl_core.types import (
    EuclideanDistributionType,
    GenericCharacteristicName,
    Kind,
    NumericArray,
)


class MockSamplingStrategy(SamplingStrategy):
    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray:
        return np.random.random((n, 1))


@dataclass(slots=True)
class StandaloneEuclideanUnivariateDistribution(Distribution):
    """
    Minimal standalone univariate Euclidean distribution.

    Notes
    -----
    - Dimension is fixed to 1.
    - Default strategies are attached: computation and univariate sampling.
    """

    _distribution_type: EuclideanDistributionType
    _analytical_computations: dict[GenericCharacteristicName, AnalyticalComputation[Any, Any]]
    _support: Support | None

    def __init__(
        self,
        kind: Kind,
        analytical_computations: (
            Iterable[AnalyticalComputation[Any, Any]]
            | Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]
        ) = (),
        support: Support | None = None,
    ) -> None:
        super(StandaloneEuclideanUnivariateDistribution, self).__init__(
            distribution_type=EuclideanDistributionType(kind, 1),
            analytical_computations=analytical_computations,
            support=support,
        )

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> StandaloneEuclideanUnivariateDistribution:
        # Actually a stub
        return StandaloneEuclideanUnivariateDistribution(
            Kind.CONTINUOUS,
            analytical_computations=self.analytical_computations,
        )
