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
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
    Distribution,
    SamplingStrategy,
)
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
    _analytical: dict[GenericCharacteristicName, AnalyticalComputation[Any, Any]]
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
        self._distribution_type = EuclideanDistributionType(kind, 1)
        self._support = support
        if isinstance(analytical_computations, Mapping):
            self._analytical = dict(analytical_computations)
        else:
            self._analytical = {ac.target: ac for ac in analytical_computations}

    @property
    def distribution_type(self) -> EuclideanDistributionType:
        """Distribution type descriptor (kind and dimension)."""
        return self._distribution_type

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """Mapping from characteristic name to analytical callable."""
        return self._analytical

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Sampling strategy instance."""
        return DefaultSamplingUnivariateStrategy()

    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]:
        """Computation strategy instance."""
        return DefaultComputationStrategy()

    @property
    def support(self):
        return self._support
