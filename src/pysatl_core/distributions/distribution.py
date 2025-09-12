from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.sampling import Sample
from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    DefaultComputationStrategy,
    DefaultSamplingStrategy,
    SamplingStrategy,
)
from pysatl_core.types import (
    DistributionType,
    EuclidianDistributionType,
    GenericCharacteristicName,
)


@runtime_checkable
class Distribution(Protocol):
    @property
    def distribution_type(self) -> DistributionType: ...

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]: ...

    @property
    def sampling_strategy(self) -> SamplingStrategy: ...
    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]: ...

    def sample(self, n: int, **options: Any) -> Sample: ...
    def log_likelihood(self, batch: Sample) -> float: ...


@dataclass(slots=True)
class StandaloneEuclidianDistribution:
    _distribution_type: EuclidianDistributionType
    _analytical: dict[GenericCharacteristicName, AnalyticalComputation[Any, Any]]

    def __init__(
        self,
        distribution_type: EuclidianDistributionType,
        analytical_computations: Iterable[AnalyticalComputation[Any, Any]]
        | Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]] = (),
    ):
        self._distribution_type = distribution_type
        if isinstance(analytical_computations, Mapping):
            self._analytical = dict(analytical_computations)
        else:
            self._analytical = {ac.target: ac for ac in analytical_computations}

    @property
    def distribution_type(self) -> EuclidianDistributionType:
        return self._distribution_type

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        return self._analytical

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        return DefaultSamplingStrategy()

    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]:
        return DefaultComputationStrategy()

    def sample(self, n: int, **options: Any) -> Sample:
        return self.sampling_strategy.sample(n, d=self.distribution_type.dimension, **options)

    def log_likelihood(self, batch: Sample) -> float:
        # TODO: Не ноль)
        return 0.0
