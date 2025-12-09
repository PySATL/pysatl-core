"""
Parametric family definitions and management infrastructure.

This module contains the main class for defining parametric families of
distributions, including support for multiple parameterizations, distribution
characteristics, sampling strategies, and computation methods.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from functools import partial
from typing import TYPE_CHECKING, dataclass_transform

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.strategies import (
    DefaultComputationStrategy,
    DefaultSamplingUnivariateStrategy,
)
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.types import DistributionType

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from pysatl_core.distributions.strategies import ComputationStrategy, SamplingStrategy
    from pysatl_core.distributions.support import Support
    from pysatl_core.families.parametrizations import (
        Parametrization,
    )
    from pysatl_core.types import (
        GenericCharacteristicName,
        ParametrizationName,
    )

    type ParametrizedFunction = Callable[[Parametrization, Any], Any]
    type SupportArg = Callable[[Parametrization], Support | None] | None
    type SupportResolver = Callable[[Parametrization], Support | None]


class ParametricFamily:
    """
    A family of distributions with multiple parametrizations.

    Represents a parametric family of distributions (e.g., normal, lognormal)
    that can be parameterized in different ways. Manages parametrizations,
    distribution characteristics, and provides factory methods for creating
    distribution instances.

    Parameters
    ----------
    name : str
        Name of the distribution family.
    distr_type : DistributionType or Callable[[Parametrization], DistributionType]
        Distribution type or function that infers type from base parametrization.
    distr_parametrizations : list[ParametrizationName]
        List of parametrization names (first is base parametrization).
    distr_characteristics : dict[str, dict[str, Callable] or Callable]
        Mapping from characteristic names to computation functions.
        Single functions are treated as defined for the base parametrization.
    sampling_strategy : SamplingStrategy, optional
        Strategy for sampling from distributions.
    computation_strategy : ComputationStrategy, optional
        Strategy for computing distribution characteristics.
    support_by_parametrization : Callable or None, optional
        Function that returns support for given parameters.
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
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy[Any, Any] | None = None,
        support_by_parametrization: SupportArg = None,
    ):
        self._name = name
        self._distr_type: Callable[[Parametrization], DistributionType] = (
            (lambda params: distr_type) if isinstance(distr_type, DistributionType) else distr_type
        )

        self.computation_strategy = (
            DefaultComputationStrategy() if computation_strategy is None else computation_strategy
        )

        if support_by_parametrization is None:
            self._support_resolver: SupportResolver
            self._support_resolver = lambda _params: None
        else:
            self._support_resolver = support_by_parametrization

        # Ordered names; the first one is the base parametrization name
        self.parametrization_names: list[ParametrizationName] = distr_parametrizations
        self.base_parametrization_name: ParametrizationName = self.parametrization_names[0]

        # Runtime registry of parametrization classes
        self._parametrizations: dict[ParametrizationName, type[Parametrization]] = {}

        self.sampling_strategy = (
            DefaultSamplingUnivariateStrategy() if sampling_strategy is None else sampling_strategy
        )

        def _process_char_val(
            value: dict[ParametrizationName, ParametrizedFunction] | ParametrizedFunction,
        ) -> dict[ParametrizationName, ParametrizedFunction]:
            return value if isinstance(value, dict) else {self.parametrization_names[0]: value}

        self.distr_characteristics: dict[
            GenericCharacteristicName, dict[ParametrizationName, ParametrizedFunction]
        ] = {key: _process_char_val(val) for key, val in distr_characteristics.items()}

        # Precompute analytical plan
        self._analytical_plan: dict[
            ParametrizationName, dict[GenericCharacteristicName, ParametrizationName]
        ] = {}
        base_name = self.base_parametrization_name
        for pname in self.parametrization_names:
            plan_for_p: dict[GenericCharacteristicName, ParametrizationName] = {}
            for characteristic, forms in self.distr_characteristics.items():
                if pname in forms:
                    plan_for_p[characteristic] = pname
                elif base_name in forms:
                    plan_for_p[characteristic] = base_name
            self._analytical_plan[pname] = plan_for_p

    @property
    def name(self) -> str:
        """Get the family name."""
        return self._name

    @property
    def parametrizations(self) -> dict[ParametrizationName, type[Parametrization]]:
        """Get mapping from parametrization names to classes."""
        return self._parametrizations

    @property
    def base(self) -> type[Parametrization]:
        """
        Get the base parametrization class.

        Raises
        ------
        ValueError
            If base parametrization is not registered.
        """
        try:
            return self._parametrizations[self.base_parametrization_name]
        except KeyError as exc:
            raise ValueError(
                f"Base parametrization '{self.base_parametrization_name}' is not registered."
            ) from exc

    @property
    def support_resolver(self) -> SupportResolver:
        """Get the support resolver function."""
        return self._support_resolver

    def register_parametrization(
        self,
        name: ParametrizationName,
        parametrization_class: type[Parametrization],
    ) -> None:
        """
        Register a parametrization class.

        Parameters
        ----------
        name : ParametrizationName
            Unique parametrization name.
        parametrization_class : type[Parametrization]
            Parametrization class to register.

        Raises
        ------
        ValueError
            If name is already registered.
        """
        if name in self._parametrizations:
            raise ValueError(f"Parametrization '{name}' is already registered.")
        self._parametrizations[name] = parametrization_class

    def get_parametrization(self, name: ParametrizationName) -> type[Parametrization]:
        """
        Fetch a parametrization class by name.

        Raises
        ------
        KeyError
            If name is not registered.
        """
        return self._parametrizations[name]

    def to_base(self, parameters: Parametrization) -> Parametrization:
        """
        Convert parameters to the base parametrization.

        Parameters
        ----------
        parameters : Parametrization
            Parameters in any parametrization.

        Returns
        -------
        Parametrization
            Equivalent parameters in base parametrization.
        """
        if parameters.name == self.base_parametrization_name:
            return parameters
        return parameters.transform_to_base_parametrization()

    def _build_analytical_computations(
        self, parameters: Parametrization
    ) -> dict[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """
        Build analytical computations for given parameters.

        Uses precomputed provider plan for efficient computation.
        """
        plan = self._analytical_plan.get(parameters.name, {})
        result: dict[GenericCharacteristicName, AnalyticalComputation[Any, Any]] = {}
        base_params: Parametrization | None = None

        for characteristic, provider_name in plan.items():
            if provider_name == parameters.name:
                params_obj = parameters
            else:
                if base_params is None:
                    base_params = self.to_base(parameters)
                params_obj = base_params

            func_factory = self.distr_characteristics[characteristic][provider_name]
            result[characteristic] = AnalyticalComputation(
                target=characteristic,
                func=partial(func_factory, params_obj),
            )

        return result

    def distribution(
        self,
        parametrization_name: str | None = None,
        **parameters_values: Any,
    ) -> ParametricFamilyDistribution:
        """
        Create a distribution instance with given parameters.

        Parameters
        ----------
        parametrization_name : str, optional
            Name of parametrization to use (defaults to base).
        **parameters_values
            Parameter values for the distribution.

        Returns
        -------
        ParametricFamilyDistribution
            Distribution instance with specified parameters.

        Raises
        ------
        KeyError
            If parametrization name is not registered.
        ValueError
            If parameters don't satisfy constraints.
        """
        if parametrization_name is None:
            parametrization_class = self.base
        else:
            parametrization_class = self._parametrizations[parametrization_name]

        parameters = parametrization_class(**parameters_values)
        parameters.validate()
        base_parameters = self.to_base(parameters)
        distribution_type = self._distr_type(base_parameters)
        return ParametricFamilyDistribution(
            self.name, distribution_type, parameters, self.support_resolver(parameters)
        )

    @dataclass_transform()
    def parametrization(
        self, *, name: str
    ) -> Callable[[type[Parametrization]], type[Parametrization]]:
        """
        Create a class decorator that registers a parametrization.

        If you want to use this syntax and so that Mypy doesn't swear,
        you should mark your class as a dataclass.
        At the moment, Mypy cannot identify dataclass_transform if the decorator is a class method.

        Parameters
        ----------
        name : str
            Name of the parametrization.

        Returns
        -------
        Callable[[type[Parametrization]], type[Parametrization]]
            Class decorator for registering parametrizations.
        """
        from pysatl_core.families.parametrizations import parametrization as _param_deco

        return _param_deco(family=self, name=name)

    __call__ = distribution
