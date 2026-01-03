"""
Distribution Interface

This module defines the public Distribution protocol that serves as the
abstract interface for all probability distributions in the system.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from pysatl_core.types import NumericArray

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from pysatl_core.distributions.computation import AnalyticalComputation
    from pysatl_core.distributions.strategies import (
        ComputationStrategy,
        Method,
        SamplingStrategy,
    )
    from pysatl_core.distributions.support import Support
    from pysatl_core.types import (
        DistributionType,
        GenericCharacteristicName,
    )


@runtime_checkable
class Distribution(Protocol):
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
    def distribution_type(self) -> DistributionType: ...

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]: ...

    @property
    def sampling_strategy(self) -> SamplingStrategy: ...

    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]: ...

    @property
    def support(self) -> Support | None: ...

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
