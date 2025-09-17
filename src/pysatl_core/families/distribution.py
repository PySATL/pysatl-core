"""
Concrete distribution instances with specific parameter values.

This module provides the implementation for individual distribution instances
created from parametric families. It handles distribution characteristics
computation, sampling, and provides access to analytical methods for
specific parameter sets.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions import (
    AnalyticalComputation,
    ComputationStrategy,
    Sample,
    SamplingStrategy,
)
from pysatl_core.families.parametrizations import Parametrization
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
)

if TYPE_CHECKING:
    from pysatl_core.families.parametric_family import ParametricFamily


@dataclass
class ParametricFamilyDistribution:
    """
    A specific distribution instance from a parametric family.

    This class represents a concrete distribution with specific parameter
    values, providing methods for computation and sampling.

    Attributes
    ----------
    distr_name : str
        Name of the distribution family.
    distribution_type : DistributionType
        Type of this distribution.
    parameters : Parametrization
        Parameter values for this distribution.
    """

    distr_name: str
    distribution_type: DistributionType
    parameters: Parametrization

    @property
    def family(self) -> ParametricFamily:
        """
        Get the parametric family this distribution belongs to.

        Returns
        -------
        ParametricFamily
            The parametric family of this distribution.
        """
        return ParametricFamilyRegister.get(self.distr_name)

    @property
    def analytical_computations(
        self,
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """
        Get analytical computation functions for this distribution.

        Returns
        -------
        Mapping[GenericCharacteristicName, AnalyticalComputation]
            Mapping from characteristic names to computation functions.
        """
        analytical_computations = {}

        # First form list of all characteristics, available from current parametrization
        for characteristic, forms in self.family.distr_characteristics.items():
            if self.parameters.name in forms:
                analytical_computations[characteristic] = AnalyticalComputation(
                    target=characteristic,
                    func=partial(forms[self.parameters.name], self.parameters),
                )
        # TODO: Second, apply rule set, for, e.g. approximations

        # Finally, fill other chacteristics
        base_name = self.family.parametrizations.base_parametrization_name
        base_parameters = self.family.parametrizations.get_base_parameters(self.parameters)
        for characteristic, forms in self.family.distr_characteristics.items():
            if characteristic in analytical_computations:
                continue
            if base_name in forms:
                analytical_computations[characteristic] = AnalyticalComputation(
                    target=characteristic, func=partial(forms[base_name], base_parameters)
                )

        return analytical_computations

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
