from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence
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
    CharacteristicName,
    EuclideanDistributionType,
    GenericCharacteristicName,
    Kind,
    LabelName,
    NumericArray,
)

type MockAnalyticalComputations = Mapping[
    GenericCharacteristicName,
    AnalyticalComputation[Any, Any] | Mapping[LabelName, AnalyticalComputation[Any, Any]],
]


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
    _analytical_computations: dict[
        GenericCharacteristicName, dict[LabelName, AnalyticalComputation[Any, Any]]
    ]
    _support: Support | None

    def __init__(
        self,
        kind: Kind,
        analytical_computations: MockAnalyticalComputations
        | Sequence[AnalyticalComputation[Any, Any]],
        support: Support | None = None,
    ) -> None:
        force_empty_analytical = False
        if isinstance(analytical_computations, Mapping):
            normalized_analytical: MockAnalyticalComputations = analytical_computations
        else:
            normalized_analytical = {comp.target: comp for comp in analytical_computations}
            if not normalized_analytical:
                # Keep backward compatibility with legacy tests that passed an empty list.
                normalized_analytical = {
                    CharacteristicName.MEAN: AnalyticalComputation[Any, float](
                        target=CharacteristicName.MEAN,
                        func=lambda **_kwargs: 0.0,
                    )
                }
                force_empty_analytical = True

        super(StandaloneEuclideanUnivariateDistribution, self).__init__(
            distribution_type=EuclideanDistributionType(kind, 1),
            analytical_computations=normalized_analytical,
            support=support,
        )
        if force_empty_analytical:
            self._analytical_computations = {}

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
