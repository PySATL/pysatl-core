"""
Distribution Interfaces and a Standalone Univariate Implementation
==================================================================

This module defines the public :class:`Distribution` protocol and a concrete
standalone univariate Euclidean distribution:

- :class:`Distribution` protocol – abstract interface used throughout the core.
- :class:`StandaloneEuclideanUnivariateDistribution` – a minimal implementation
  that plugs default computation and sampling strategies.

Notes
-----
- The univariate sampling strategy draws from the distribution's ``ppf``.
- Log-likelihood is computed element-wise using ``pdf`` (continuous) or ``pmf``
  (discrete), assuming scalar characteristic functions (``float -> float``).
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.sampling import Sample
from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    Method,
    SamplingStrategy,
)
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
)


@runtime_checkable
class Distribution(Protocol):
    """Public distribution interface used by strategies and fitters."""

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

    def query_method(
        self, characteristic_name: GenericCharacteristicName, **options: Any
    ) -> Method[Any, Any]:
        return self.computation_strategy.query_method(characteristic_name, self, **options)

    def calculate_characteristic(
        self, characteristic_name: GenericCharacteristicName, value: Any, **options: Any
    ) -> Any:
        return self.query_method(characteristic_name, **options)(value)

    def sample(self, n: int, **options: Any) -> Sample:
        return self.sampling_strategy.sample(n, distr=self, **options)

    def log_likelihood(self, batch: Sample) -> float: ...
