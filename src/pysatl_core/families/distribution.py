"""
Concrete distribution instances with specific parameter values.

This module provides the implementation for individual distribution instances
created from parametric families.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pysatl_core.distributions.distribution import Distribution
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


@dataclass(slots=True)
class ParametricFamilyDistribution(Distribution):
    """
    A specific distribution instance from a parametric family.

    Represents a concrete distribution with specific parameter values,
    providing methods for computation and sampling.

    Parameters
    ----------
    family_name : str
        Name of the distribution family.
    _distribution_type : DistributionType
        Type of this distribution.
    _parametrization : Parametrization
        Parameter values for this distribution.
    _support : Support or None
        Support of this distribution.
    """

    family_name: str
    _distribution_type: DistributionType
    _parametrization: Parametrization
    _support: Support | None

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
        cache_key = getattr(self, "_analytical_cache_key", None)
        cache_val = getattr(self, "_analytical_cache_val", None)

        if cache_key != key or cache_val is None:
            cache_val = self.family._build_analytical_computations(self.parametrization)
            self._analytical_cache_key = key
            self._analytical_cache_val = cache_val

        return cache_val

    @property
    def sampling_strategy(self) -> SamplingStrategy:
        """Get the sampling strategy for this distribution."""
        return self.family.sampling_strategy

    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]:
        """Get the computation strategy for this distribution."""
        return self.family.computation_strategy

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
