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
    EuclideanDistributionType,
    GenericCharacteristicName,
    Kind,
    LabelName,
    NumericArray,
)

type MockEntryComputations = Mapping[
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
    _support: Support | None

    def __init__(
        self,
        kind: Kind,
        entry_computations: MockEntryComputations | Sequence[AnalyticalComputation[Any, Any]],
        support: Support | None = None,
    ) -> None:
        if isinstance(entry_computations, Mapping):
            normalized_entries: MockEntryComputations = entry_computations
        else:
            normalized_entries = {comp.target: comp for comp in entry_computations}

        super(StandaloneEuclideanUnivariateDistribution, self).__init__(
            distribution_type=EuclideanDistributionType(kind, 1),
            entry_computations=normalized_entries,
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
            entry_computations=self.entry_computations,
        )
