"""
Parametric family definitions and management infrastructure.

This module contains the main class for defining parametric families of
distributions, including support for multiple parameterizations, distribution
characteristics, sampling strategies, and computation methods. It serves as
the central definition point for statistical distribution families.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from typing import Any

from pysatl_core.distributions import (
    ComputationStrategy,
    SamplingStrategy,
)
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.families.parametrizations import Parametrization, ParametrizationSpec
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
    ParametrizationName,
)

type ParametrizedFunction = Callable[[Parametrization, Any], Any]


class ParametricFamily:
    """
    A family of distributions with multiple parametrizations.

    This class represents a parametric family of distributions, such as
    the normal or lognormal family, which can be parameterized in different
    ways (e.g., mean-variance or canonical parametrization).

    Attributes
    ----------
    name : str
        Name of the distribution family.
    distr_type : DistributionType | Callable[[Parametrization] | DistributionType]
        Type of distributions in this family.
    parametrizations : ParametrizationSpec
        Specification of available parametrizations.
    distr_parametrizations :

    distr_characteristics : Dict[GenericCharacteristicName, Callable[[Any, Any], Any]]
        Mapping from characteristic names to computation functions.
    sampling_strategy : SamplingStrategy
        Strategy for sampling from distributions in this family.
    computation_strategy : ComputationStrategy
        Strategy for computing distribution characteristics.
    """

    def __init__(
        self,
        name: str,
        distr_type: DistributionType | Callable[[Parametrization], DistributionType],
        distr_parametrizations: list[ParametrizationName],
        distr_characteristics: dict[
            GenericCharacteristicName,
            dict[ParametrizationName, ParametrizedFunction] | ParametrizedFunction,
        ],
        sampling_strategy: SamplingStrategy,
        computation_strategy: ComputationStrategy[Any, Any],
    ):
        """
        Initialize a new parametric family.

        Parameters
        ----------
        name : str
            Name of the distribution family.

        distr_type : DistributionType | Callable[[Parametrization], DistributionType]
            Type of distributions in this family or, if type is parameter-depended, function
            that takes as input *base* parametrization and inferes type based on it.

        distr_parametrizations : List[ParametrizationName]
            List of parametrizations for this distribution. *First parametrization is always
            base parametrization*.

        distr_characteristics:
            Mapping from characteristics names to computation functions or dictionary of those,
            if for multiple parametrizations same characteristic available.

        sampling_strategy : SamplingStrategy
            Strategy for sampling from distributions in this family.

        computation_strategy : ComputationStrategy
            Strategy for computing distribution characteristics.
        """
        self._name = name
        self._distr_type: Callable[[Parametrization], DistributionType] = (
            (lambda params: distr_type) if isinstance(distr_type, DistributionType) else distr_type
        )

        # Parametrizations must be built by user
        self.parametrization_names = distr_parametrizations
        self.parametrizations = ParametrizationSpec(self.parametrization_names[0])

        self.sampling_strategy = sampling_strategy
        self.computation_strategy = computation_strategy

        def _process_char_val(
            value: dict[ParametrizationName, ParametrizedFunction] | ParametrizedFunction,
        ) -> dict[ParametrizationName, ParametrizedFunction]:
            return value if isinstance(value, dict) else {self.parametrization_names[0]: value}

        self.distr_characteristics = {
            key: _process_char_val(value) for key, value in distr_characteristics.items()
        }

    @property
    def name(self) -> str:
        return self._name

    def distribution(
        self, parametrization_name: str | None = None, **parameters_values: Any
    ) -> ParametricFamilyDistribution:
        """
        Create a distribution instance with the given parameters.

        Parameters
        ----------
        parametrization_name : str | None, optional
            Name of the parametrization to use, or None for base parametrization.
        **parameters_values
            Parameter values for the distribution.

        Returns
        -------
        ParametricFamilyDistribution
            A distribution instance with the specified parameters.

        Raises
        ------
        ValueError
            If the parameters don't satisfy the parametrization constraints.
        """
        if parametrization_name is None:
            parametrization_class = self.parametrizations.base
        else:
            parametrization_class = self.parametrizations.parametrizations[parametrization_name]

        parameters = parametrization_class(**parameters_values)
        base_parameters = self.parametrizations.get_base_parameters(parameters)
        parameters.validate()
        distribution_type = self._distr_type(base_parameters)
        return ParametricFamilyDistribution(self.name, distribution_type, parameters)

    __call__ = distribution
