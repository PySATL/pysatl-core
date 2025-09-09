from typing import Any, Protocol

import numpy as np

from pysatl_core.distributions.computation import (
    AnalyticalComputation,
    ComputationMethod,
    FittedComputationMethod,
)
from pysatl_core.types import (
    GenericCharacteristicName,
)

from .distribution import Distribution
from .registry import distribution_type_register
from .sampling import ArraySample, Sample

type Method[In, Out] = (
    ComputationMethod[In, Out] | FittedComputationMethod[In, Out] | AnalyticalComputation[In, Out]
)


class ComputationStrategy[In, Out](Protocol):
    enable_caching: bool = False
    cached_computations: dict[GenericCharacteristicName, FittedComputationMethod[In, Out]] = {}

    def query_method(
        self, state: GenericCharacteristicName, distr: Distribution
    ) -> Method[In, Out]: ...


class DefaultComputationStrategy[In, Out](ComputationStrategy[In, Out]):
    enable_caching: bool = False
    cached_computations: dict[GenericCharacteristicName, FittedComputationMethod[In, Out]] = {}

    def query_method(
        self, state: GenericCharacteristicName, distr: Distribution
    ) -> Method[In, Out]:
        if state in distr.analytical_computations:
            return distr.analytical_computations[state]

        if self.enable_caching and state in self.cached_computations:
            return self.cached_computations[state]
        method = (
            distribution_type_register()
            .get(distr.dimension, distr.kind)
            .get_available_definitive_characteristics()[state][0]
            .fit(distr)
        )
        if self.enable_caching:
            self.cached_computations[state] = method

        return method


class SamplingStrategy(Protocol):
    def sample(self, n: int, **options: Any) -> Sample: ...


class DefaultSamplingStrategy(SamplingStrategy):
    def sample(self, n: int, **options: Any) -> ArraySample:
        d = int(options.get("d", 0))
        if d <= 0:
            raise ValueError("DefaultSamplingMethodND.sample requires positive 'd' option")
        return ArraySample(np.zeros((n, d), dtype=np.float64))
