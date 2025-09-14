from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pysatl_core.distributions import (
    ComputationStrategy,
    SamplingStrategy,
)
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.families.parametrizations import Parametrization, ParametrizationSpec
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
)


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
        distr_characteristics: dict[
            GenericCharacteristicName, Callable[[Parametrization, Any], Any]
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
        distr_type : DistributionType
            Type of distributions in this family.
        distr_characteristics : Dict[GenericCharacteristicName, Callable[[Any, Any], Any]]
            Mapping from characteristic names to computation functions.
        sampling_strategy : SamplingStrategy
            Strategy for sampling from distributions in this family.
        computation_strategy : ComputationStrategy
            Strategy for computing distribution characteristics.
        """
        self.name = name
        self.distr_type: Callable[[Parametrization], DistributionType] = (
            (lambda params: distr_type) if isinstance(distr_type, DistributionType) else distr_type
        )

        self.parametrizations = ParametrizationSpec()
        self.distr_characteristics = distr_characteristics
        self.sampling_strategy = sampling_strategy
        self.computation_strategy = computation_strategy

        ParametricFamilyRegister.register(self)

    def __call__(
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
        return ParametricFamilyDistribution(self.name, self.distr_type(base_parameters), parameters)
