"""
Distribution Interface

This module defines the public Distribution protocol that serves as the
abstract interface for all probability distributions in the system.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self, cast

from pysatl_core.types import NumericArray

_KEEP: object = object()


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from pysatl_core.distributions.computation import AnalyticalComputation
    from pysatl_core.distributions.strategies import (
        ComputationStrategy,
        SamplingStrategy,
    )
    from pysatl_core.distributions.support import Support
    from pysatl_core.types import (
        DistributionType,
        GenericCharacteristicName,
        Method,
    )


class Distribution(ABC):
    """
    Protocol defining the interface for probability distributions.

    This protocol is the central abstraction used throughout the system.
    Concrete distribution implementations must provide the properties and
    methods defined here.

    Attributes
    ----------
    distribution_type : DistributionType
        Type information about the distribution (kind, dimension, etc.).
    analytical_computations : Mapping[str, AnalyticalComputation]
        Direct analytical computations provided by the distribution.
    sampling_strategy : SamplingStrategy
        Strategy for generating random samples.
    computation_strategy : ComputationStrategy
        Strategy for computing characteristics and conversions.
    support : Support or None
        Support of the distribution, if defined.
    """

    @property
    @abstractmethod
    def distribution_type(self) -> DistributionType: ...

    @property
    @abstractmethod
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]: ...

    @property
    @abstractmethod
    def sampling_strategy(self) -> SamplingStrategy: ...

    @property
    @abstractmethod
    def computation_strategy(self) -> ComputationStrategy: ...

    @property
    @abstractmethod
    def support(self) -> Support | None: ...

    @abstractmethod
    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> Distribution: ...

    def with_sampling_strategy(self, sampling_strategy: SamplingStrategy | None) -> Self:
        """Return a copy of this distribution with an updated sampling strategy."""
        return cast(Self, self._clone_with_strategies(sampling_strategy=sampling_strategy))

    def with_computation_strategy(self, computation_strategy: ComputationStrategy | None) -> Self:
        """Return a copy of this distribution with an updated computation strategy."""
        return cast(
            Self,
            self._clone_with_strategies(computation_strategy=computation_strategy),
        )

    def with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> Self:
        """Return a copy of this distribution with updated strategies."""
        return cast(
            Self,
            self._clone_with_strategies(
                sampling_strategy=sampling_strategy,
                computation_strategy=computation_strategy,
            ),
        )

    def query_method(
        self, characteristic_name: GenericCharacteristicName, **options: Any
    ) -> Method[Any, Any]:
        """
        Query a computation method for a specific characteristic.

        Parameters
        ----------
        characteristic_name : str
            Name of the characteristic to compute (e.g., "pdf", "cdf").
        **options : Any
            Additional options for the computation.

        Returns
        -------
        Method
            Callable method that computes the characteristic.
        """
        return self.computation_strategy.query_method(characteristic_name, self, **options)

    def calculate_characteristic(
        self, characteristic_name: GenericCharacteristicName, value: Any, **options: Any
    ) -> Any:
        """
        Calculate a characteristic at the given value.

        Parameters
        ----------
        characteristic_name : str
            Name of the characteristic to compute.
        value : Any
            Point(s) at which to evaluate the characteristic.
        **options : Any
            Additional computation options.

        Returns
        -------
        Any
            Value of the characteristic at the given point(s).
        """
        return self.query_method(characteristic_name, **options)(value)

    def sample(self, n: int, **options: Any) -> NumericArray:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        **options : Any
            Additional sampling options forwarded to the underlying
            sampling strategy.

        Returns
        -------
        NumericArray
            NumPy array containing ``n`` generated samples.
            The exact array shape depends on the distribution and
            the sampling strategy.
        """
        return cast(NumericArray, self.sampling_strategy.sample(n, distr=self, **options))
