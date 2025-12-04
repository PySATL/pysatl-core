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
from typing import TYPE_CHECKING

from pysatl_core.distributions.distribution import Distribution
from pysatl_core.families.registry import ParametricFamilyRegister

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from pysatl_core.distributions.computation import AnalyticalComputation
    from pysatl_core.distributions.sampling import Sample
    from pysatl_core.distributions.strategies import ComputationStrategy, SamplingStrategy
    from pysatl_core.distributions.support import Support
    from pysatl_core.families.parametric_family import ParametricFamily
    from pysatl_core.families.parametrizations import Parametrization
    from pysatl_core.types import (
        DistributionType,
        GenericCharacteristicName,
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
    parameters : Parametrization
        Parameter values for this distribution.
    _support : Support or None
        Support of this distribution.
    """

    family_name: str
    _distribution_type: DistributionType
    parameters: Parametrization
    _support: Support | None

    @property
    def distribution_type(self) -> DistributionType:
        """Get the distribution type."""
        return self._distribution_type

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
        key = (id(self.parameters), self.parameters.name)
        cache_key = getattr(self, "_analytical_cache_key", None)
        cache_val = getattr(self, "_analytical_cache_val", None)

        if cache_key != key or cache_val is None:
            cache_val = self.family._build_analytical_computations(self.parameters)
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

    def sample(self, n: int, **options: Any) -> Sample:
        """
        Generate samples from this distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        **options : Any
            Additional options for sampling.

        Returns
        -------
        Sample
            Generated samples.
        """
        return self.sampling_strategy.sample(n, distr=self, **options)
