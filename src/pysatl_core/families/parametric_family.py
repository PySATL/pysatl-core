"""
Parametric family definitions and management infrastructure.

This module contains the main class for defining parametric families of
distributions, including support for multiple parameterizations, distribution
characteristics, sampling strategies, and computation methods. It serves as
the central definition point for statistical distribution families.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov, Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from functools import partial
from typing import Any

from pysatl_core.distributions import (
    ComputationStrategy,
    DefaultComputationStrategy,
    SamplingStrategy,
)
from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.families.parametrizations import Parametrization
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
    ways (e.g., mean-variance or canonical parametrization). The family
    owns the registry of parametrizations and enforces invariants such as
    unique names and the existence of a single base parametrization.

    Attributes
    ----------
    name : str
        Name of the distribution family.
    distr_type : DistributionType | Callable[[Parametrization], DistributionType]
        Type of distributions in this family or a function that infers the
        type from *base* parametrization values.
    parametrization_names : list[ParametrizationName]
        Ordered list of parametrization names. The first name is considered
        the base parametrization name.
    distr_characteristics : Dict[GenericCharacteristicName, \
    Dict[ParametrizationName, ParametrizedFunction]]
        Mapping from characteristic names to computation functions by parametrization.
        If a single function is provided, it is assumed to be defined for the base
        parametrization.
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
        computation_strategy: ComputationStrategy[Any, Any] = DefaultComputationStrategy(),
    ):
        """
        Initialize a new parametric family.

        Parameters
        ----------
        name : str
            Name of the distribution family.
        distr_type : DistributionType | Callable[[Parametrization], DistributionType]
            Type of distributions in this family or, if the type is parameter-dependent,
            a function that takes *base* parametrization and infers the type.
        distr_parametrizations : list[ParametrizationName]
            List of parametrizations for this distribution. The first element is
            always the base parametrization name.
        distr_characteristics : dict[GenericCharacteristicName, \
        dict[ParametrizationName, ParametrizedFunction] | ParametrizedFunction]
            Mapping from characteristic names to computation functions. A single function
            value is treated as defined for the base parametrization.
        sampling_strategy : SamplingStrategy
            Strategy for sampling from distributions in this family.
        computation_strategy : ComputationStrategy
            Strategy for computing distribution characteristics.
        """
        self._name = name
        self._distr_type: Callable[[Parametrization], DistributionType] = (
            (lambda params: distr_type) if isinstance(distr_type, DistributionType) else distr_type
        )

        # Ordered names; the first one is the base parametrization name
        self.parametrization_names: list[ParametrizationName] = distr_parametrizations
        self.base_parametrization_name: ParametrizationName = self.parametrization_names[0]

        # Runtime registry of parametrization classes (formerly in ParametrizationSpec)
        self._parametrizations: dict[ParametrizationName, type[Parametrization]] = {}

        self.sampling_strategy = sampling_strategy
        self.computation_strategy = computation_strategy

        def _process_char_val(
            value: dict[ParametrizationName, ParametrizedFunction] | ParametrizedFunction,
        ) -> dict[ParametrizationName, ParametrizedFunction]:
            return value if isinstance(value, dict) else {self.parametrization_names[0]: value}

        self.distr_characteristics: dict[
            GenericCharacteristicName, dict[ParametrizationName, ParametrizedFunction]
        ] = {key: _process_char_val(val) for key, val in distr_characteristics.items()}

        # Precompute analytical plan: for each parametrization, choose provider parametrization
        # (either itself or base) for every characteristic that is available.
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
        """
        Family name.

        Returns
        -------
        str
            The family name.
        """
        return self._name

    @property
    def parametrizations(self) -> dict[ParametrizationName, type[Parametrization]]:
        """
        Mapping from parametrization names to parametrization classes.

        Returns
        -------
        dict[ParametrizationName, type[Parametrization]]
            A live mapping of registered parametrizations.
        """
        return self._parametrizations

    @property
    def base(self) -> type[Parametrization]:
        """
        Base parametrization class for this family.

        Returns
        -------
        type[Parametrization]
            The class registered under ``base_parametrization_name``.

        Raises
        ------
        ValueError
            If the base parametrization has not been registered yet.
        """
        try:
            return self._parametrizations[self.base_parametrization_name]
        except KeyError as exc:
            raise ValueError(
                f"Base parametrization '{self.base_parametrization_name}' is not registered."
            ) from exc

    def register_parametrization(
        self,
        name: ParametrizationName,
        parametrization_class: type[Parametrization],
    ) -> None:
        """
        Register a parametrization class under ``name``.

        Parameters
        ----------
        name : ParametrizationName
            Unique parametrization name within the family.
        parametrization_class : type[Parametrization]
            The class that implements the parametrization.

        Raises
        ------
        RuntimeError
            If the family is frozen and no further registration is allowed.
        ValueError
            If ``name`` is already registered.
        """
        if name in self._parametrizations:
            raise ValueError(f"Parametrization '{name}' is already registered.")
        self._parametrizations[name] = parametrization_class

    def get_parametrization(self, name: ParametrizationName) -> type[Parametrization]:
        """
        Fetch a parametrization class by name.

        Parameters
        ----------
        name : ParametrizationName
            Parametrization name.

        Returns
        -------
        type[Parametrization]
            The registered parametrization class.

        Raises
        ------
        KeyError
            If the name is not registered.
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
            Equivalent parameters in the base parametrization.
        """
        if parameters.name == self.base_parametrization_name:
            return parameters
        return parameters.transform_to_base_parametrization()

    def _build_analytical_computations(
        self, parameters: Parametrization
    ) -> dict[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """
        Build analytical computations mapping for the given parameter instance.

        This uses a precomputed provider plan so runtime work is reduced to:
        - (Optionally) converting parameters to base once,
        - binding callables with :func:`functools.partial`.

        Parameters
        ----------
        parameters : Parametrization
            Parameters in any registered parametrization.

        Returns
        -------
        dict[GenericCharacteristicName, AnalyticalComputation]
            Mapping from characteristic name to analytical computation callable.
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
        Create a distribution instance with the given parameters.

        Parameters
        ----------
        parametrization_name : str | None, optional
            Name of the parametrization to use, or ``None`` for the base parametrization.
        **parameters_values
            Parameter values for the distribution.

        Returns
        -------
        ParametricFamilyDistribution
            A distribution instance with the specified parameters.

        Raises
        ------
        KeyError
            If the parametrization name is not registered.
        ValueError
            If the parameters don't satisfy the parametrization constraints.
        """
        if parametrization_name is None:
            parametrization_class = self.base
        else:
            parametrization_class = self._parametrizations[parametrization_name]

        parameters = parametrization_class(**parameters_values)
        parameters.validate()
        base_parameters = self.to_base(parameters)
        distribution_type = self._distr_type(base_parameters)
        return ParametricFamilyDistribution(self.name, distribution_type, parameters)

    def parametrization(
        self,
        name: str,
    ) -> Callable[[type[Parametrization]], type[Parametrization]]:
        """
        Create a class decorator that registers a parametrization in this family.

        Parameters
        ----------
        name : str
            Name of the parametrization.

        Returns
        -------
        Callable[[type[Parametrization]], type[Parametrization]]
            Class decorator that registers the parametrization and returns it.
        """
        from pysatl_core.families.parametrizations import parametrization as _param_deco

        return _param_deco(family=self, name=name)

    __call__ = distribution
