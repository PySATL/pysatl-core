"""
Concrete distribution instances with specific parameter values.

This module provides the implementation for individual distribution instances
created from parametric families.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, cast

from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.strategies import (
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import NumericArray

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from pysatl_core.distributions.computation import AnalyticalComputation
    from pysatl_core.distributions.strategies import (
        ComputationStrategy,
        SamplingStrategy,
    )
    from pysatl_core.distributions.support import Support
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import (
        Parametrization,
        ParametrizationConstraint,
    )
    from pysatl_core.types import (
        DistributionType,
        GenericCharacteristicName,
        ParametrizationName,
    )


class ParametricFamilyDistribution(Distribution):
    """
    A specific distribution instance from a parametric family.

    Represents a concrete distribution with specific parameter values,
    providing methods for computation and sampling.

    Parameters
    ----------
    family_name : str
        Name of the distribution family.
    distribution_type : DistributionType
        Type of this distribution.
    parametrization : Parametrization
        Parameter values for this distribution.
    support : Support or None
        Support of this distribution.
    sampling_strategy : SamplingStrategy
        Strategy for generating random samples.
        Such an object is unique for each distribution.
    computation_strategy : ComputationStrategy
        Strategy for computing characteristics and conversions.
        Such an object is unique for each distribution.
    """

    def __init__(
        self,
        family_name: str,
        distribution_type: DistributionType,
        parametrization: Parametrization,
        support: Support | None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ):
        self._distribution_type = distribution_type
        self._family_name = family_name
        self._parametrization = parametrization
        self._support = support

        self._computation_strategy = computation_strategy or DefaultComputationStrategy()
        self._sampling_strategy = sampling_strategy or DefaultSamplingUnivariateStrategy()

        self._analytical_cache_key: tuple[int, GenericCharacteristicName] | None = None
        self._analytical_cache_val: (
            Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]] | None
        ) = None

    @property
    def family_name(self) -> str:
        "Get the name of the family this distribution belongs to."
        return self._family_name

    @property
    def distribution_type(self) -> DistributionType:
        """Get the distribution type."""
        return self._distribution_type

    @property
    def parametrization(self) -> Parametrization:
        """
        Get the parametrization object containing distribution parameters.

        Returns
        -------
        Parametrization
            Parametrization instance with all parameter values.
        """
        return self._parametrization

    @property
    def parameters(self) -> dict[str, Any]:
        """
        Get distribution parameters as a dictionary (shortcut).

        Returns
        -------
        dict[str, Any]
            Dictionary mapping parameter names to their values.
        """
        return self._parametrization.parameters

    @property
    def parametrization_name(self) -> ParametrizationName:
        """
        Get the name of the parametrization used by this distribution.

        Returns
        -------
        ParametrizationName
            Name of the parameterization format (e.g., 'meanStd').
        """
        return self._parametrization.name

    @property
    def parameters_constraints(self) -> list[ParametrizationConstraint]:
        """
        Get constraints that apply to the distribution's parameters.

        Returns
        -------
        list[ParametrizationConstraint]
            List of parameter constraints for validation.
        """
        return self._parametrization.constraints

    @property
    def family(self) -> ParametricFamily:
        """
        Get the parametric family this distribution belongs to.

        Returns
        -------
        ParametricFamily
            The parametric family of this distribution.
        """
        return ParametricFamilyRegister.get(self.family_name)

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """
        Get analytical computations for this distribution.

        Lazily computed and cached per instance. Cache invalidates when
        parametrization object or name changes.
        """
        key = (id(self.parametrization), self.parametrization_name)

        if self._analytical_cache_key != key or self._analytical_cache_val is None:
            self._analytical_cache_val = self.family.build_analytical_computations(
                self.parametrization
            )
            self._analytical_cache_key = key

        return self._analytical_cache_val

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Get the sampling strategy for this distribution."""
        return self._sampling_strategy

    @property
    def computation_strategy(self) -> ComputationStrategy:
        """Get the computation strategy for this distribution."""
        return self._computation_strategy

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> ParametricFamilyDistribution:
        """Return a copy of this distribution with updated strategies."""
        return ParametricFamilyDistribution(
            family_name=self._family_name,
            distribution_type=self._distribution_type,
            parametrization=self._parametrization,
            support=self._support,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

    @property
    def support(self) -> Support | None:
        """Get the support of this distribution."""
        return self._support

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
