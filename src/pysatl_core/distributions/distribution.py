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
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Self, cast

from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    SamplingStrategy,
)
from pysatl_core.types import DEFAULT_ANALYTICAL_COMPUTATION_LABEL, NumericArray

_KEEP: object = object()


if TYPE_CHECKING:
    from typing import Any

    from pysatl_core.distributions.computation import AnalyticalComputation
    from pysatl_core.distributions.support import Support
    from pysatl_core.types import (
        DistributionType,
        GenericCharacteristicName,
        LabelName,
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
    analytical_computations : Mapping[
        GenericCharacteristicName,
        (
            AnalyticalComputation[Any, Any]
            | Mapping[LabelName, AnalyticalComputation[Any, Any]]
        ),
    ]
        Direct analytical computations provided by the distribution.
    sampling_strategy : SamplingStrategy
        Strategy for generating random samples.
    computation_strategy : ComputationStrategy
        Strategy for computing characteristics and conversions.
    support : Support or None
        Support of the distribution, if defined.
    """

    def __init__(
        self,
        distribution_type: DistributionType,
        analytical_computations: Mapping[
            GenericCharacteristicName,
            (AnalyticalComputation[Any, Any] | Mapping[LabelName, AnalyticalComputation[Any, Any]]),
        ],
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        """
        Initialize common distribution state.

        Parameters
        ----------
        distribution_type : DistributionType
            Type information about the distribution (kind, dimension, etc.).
        analytical_computations :
            Mapping[
                GenericCharacteristicName,
                (
                    AnalyticalComputation[Any, Any]
                    | Mapping[LabelName, AnalyticalComputation[Any, Any]]
                ),
            ]
            Analytical computations provided by the distribution.
        support : Support or None, default=None
            Support of the distribution.
        sampling_strategy : SamplingStrategy or None, default=None
            Sampling strategy instance. If omitted, univariate default is used.
        computation_strategy : ComputationStrategy or None, default=None
            Computation strategy instance. If omitted, default strategy is used.
        """
        from pysatl_core.distributions.strategies import (
            DefaultComputationStrategy,
            DefaultSamplingUnivariateStrategy,
        )

        self._distribution_type = distribution_type
        normalized_analytical: dict[
            GenericCharacteristicName, dict[LabelName, AnalyticalComputation[Any, Any]]
        ] = {}
        for characteristic_name, methods in analytical_computations.items():
            if isinstance(methods, Mapping):
                normalized_analytical[characteristic_name] = dict(methods)
            else:
                normalized_analytical[characteristic_name] = {
                    DEFAULT_ANALYTICAL_COMPUTATION_LABEL: methods
                }

        if not normalized_analytical:
            raise ValueError("Distribution requires at least one analytical computation.")

        for characteristic_name, labeled_methods in normalized_analytical.items():
            if not labeled_methods:
                raise ValueError(
                    f"Characteristic '{characteristic_name}' must provide at least one "
                    "analytical computation."
                )

        self._analytical_computations = normalized_analytical
        self._support = support
        self._sampling_strategy = sampling_strategy or DefaultSamplingUnivariateStrategy()
        self._computation_strategy = computation_strategy or DefaultComputationStrategy()

    @property
    def distribution_type(self) -> DistributionType:
        """Return type metadata of the distribution (kind, dimension, etc.)."""
        return self._distribution_type

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, Mapping[LabelName, AnalyticalComputation[Any, Any]]]:
        """Return analytical computations provided directly by this distribution."""
        return self._analytical_computations

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Return the currently attached sampling strategy."""
        return self._sampling_strategy

    @property
    def computation_strategy(self) -> ComputationStrategy:
        """Return the currently attached computation strategy."""
        return self._computation_strategy

    @property
    def support(self) -> Support | None:
        """Return the support of the distribution, if it is defined."""
        return self._support

    @abstractmethod
    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> Distribution:
        """
        Return a cloned distribution with updated strategies.

        The ``_KEEP`` sentinel means the existing strategy should be preserved
        for that side.
        """
        ...

    def _new_sampling_strategy(
        self,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
    ) -> SamplingStrategy | None:
        """
        Resolve sampling strategy for cloning.

        When ``sampling_strategy`` is ``_KEEP``, returns a deep copy of the
        current sampling strategy.
        """
        return cast(
            SamplingStrategy | None,
            deepcopy(self._sampling_strategy) if sampling_strategy is _KEEP else sampling_strategy,
        )

    def _new_computation_strategy(
        self,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> ComputationStrategy | None:
        """
        Resolve computation strategy for cloning.

        When ``computation_strategy`` is ``_KEEP``, returns a deep copy of the
        current computation strategy.
        """
        return cast(
            ComputationStrategy | None,
            deepcopy(self._computation_strategy)
            if computation_strategy is _KEEP
            else computation_strategy,
        )

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
