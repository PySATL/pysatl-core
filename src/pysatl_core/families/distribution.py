"""
Concrete distribution instances with specific parameter values.

This module provides the implementation for individual distribution instances
created from parametric families. It handles distribution characteristics
computation, sampling, and provides access to analytical methods for
specific parameter sets.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions import (
    AnalyticalComputation,
    ComputationStrategy,
    Distribution,
    Sample,
    SamplingStrategy,
)
from pysatl_core.families.parametrizations import Parametrization
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.families.support import Support
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
)

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily


@dataclass(slots=True)
class ParametricFamilyDistribution(Distribution):
    """
    A specific distribution instance from a parametric family.

    This class represents a concrete distribution with specific parameter
    values, providing methods for computation and sampling.

    Attributes
    ----------
    family_name : str
        Name of the distribution family.
    _distribution_type : DistributionType
        Type of this distribution.
    parameters : Parametrization
        Parameter values for this distribution.
    """

    family_name: str
    _distribution_type: DistributionType
    parameters: Parametrization

    @property
    def distribution_type(self) -> DistributionType:
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
        """Lazily computed analytical computations for this distribution instance.

        Delegates construction to the parent family (precomputed plan) and
        caches the result per-instance. The cache auto-invalidates when either
        the **parametrization object** changes (by identity) or the
        **parametrization name** changes.

        *If you mutate numeric fields of the same parametrization object*,
        the callables see fresh values because they close over that object.
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
        """
        Get the sampling strategy for this distribution.

        Returns
        -------
        SamplingStrategy
            Strategy for sampling from this distribution.
        """
        return self.family.sampling_strategy

    @property
    def computation_strategy(self) -> ComputationStrategy[Any, Any]:
        """
        Get the computation strategy for this distribution.

        Returns
        -------
        ComputationStrategy
            Strategy for computing characteristics of this distribution.
        """
        return self.family.computation_strategy

    @property
    def _support(self) -> Support:
        return self.family._support_resolver(self.parameters)

    def log_likelihood(self, batch: Sample) -> float:
        raise NotImplementedError

    def sample(self, n: int, **options: Any) -> Sample:
        """
        Generate samples from this distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        **options : Any
            Additional options for the sampling algorithm.

        Returns
        -------
        Sample
            The generated samples.
        """
        return self.sampling_strategy.sample(n, distr=self, **options)
