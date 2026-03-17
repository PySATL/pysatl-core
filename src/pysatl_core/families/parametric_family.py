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

import inspect
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, cast, dataclass_transform

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    ComputationFunc,
    DistributionType,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pysatl_core.distributions.strategies import ComputationStrategy, SamplingStrategy
    from pysatl_core.distributions.support import Support
    from pysatl_core.families.parametrizations import (
        Parametrization,
    )
    from pysatl_core.types import (
        GenericCharacteristicName,
        LabelName,
        ParametrizationName,
    )

    type SupportArg = Callable[[Parametrization], Support | None] | None
    type SupportResolver = Callable[[Parametrization], Support | None]
    type LabeledCharacteristicProvider = (
        Mapping[LabelName, CharacteristicFunction[Any, Any]] | CharacteristicFunction[Any, Any]
    )
    type CharacteristicProvider = (
        Mapping[ParametrizationName, LabeledCharacteristicProvider]
        | CharacteristicFunction[Any, Any]
    )
    type CharacteristicsMap = Mapping[GenericCharacteristicName, CharacteristicProvider]
    type NonParametrizedCharacteristic[In, Out] = Callable[[], Out]
    type CharacteristicFunction[In, Out] = (
        NonParametrizedCharacteristic[In, Out] | ParametrizedCharacteristic[In, Out]
    )


type ParametrizedCharacteristic[In, Out] = (
    Callable[[Parametrization, In], Out] | Callable[[Parametrization], Out]
)


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
    distr_characteristics : CharacteristicsMap
        Mapping from characteristic names to analytical provider callables.

        Each provider callable may accept a parametrization instance as the first argument.
        The remaining signature is characteristic-specific:

        - nullary characteristics (e.g., mean, var): provider(params, **kwargs) -> Any
        - pointwise characteristics (e.g., pdf, cdf, ppf): provider(params, x, **kwargs) -> Any

        Providers are grouped by parametrization and may define multiple labeled methods.
        If a single callable is provided, it is treated as the base-parametrization method
        under ``DEFAULT_ANALYTICAL_COMPUTATION_LABEL``.
    support_by_parametrization : Callable or None, optional
        Function that returns support for given parameters.
    """

    def __init__(
        self,
        name: str,
        distr_type: DistributionType | Callable[[Parametrization], DistributionType],
        distr_parametrizations: list[ParametrizationName],
        distr_characteristics: CharacteristicsMap,
        support_by_parametrization: SupportArg = None,
    ):
        if not distr_parametrizations:
            raise ValueError(
                "distr_parametrizations must be non-empty (base parametrization is required)."
            )

        self._name = name
        # Ordered names; the first one is the base parametrization name
        self.parametrization_names = distr_parametrizations
        self.base_parametrization_name = self.parametrization_names[0]
        self._distr_type: Callable[[Parametrization], DistributionType] = (
            (lambda params: distr_type) if isinstance(distr_type, DistributionType) else distr_type
        )

        self._support_resolver: SupportResolver = support_by_parametrization or (lambda _p: None)

        # Runtime registry of parametrization classes
        self._parametrizations: dict[ParametrizationName, type[Parametrization]] = {}

        def _normalize_labeled_provider(
            characteristic_name: GenericCharacteristicName,
            parametrization_name: ParametrizationName,
            provider: LabeledCharacteristicProvider,
        ) -> dict[LabelName, CharacteristicFunction[Any, Any]]:
            normalized = (
                dict(provider)
                if isinstance(provider, Mapping)
                else {DEFAULT_ANALYTICAL_COMPUTATION_LABEL: provider}
            )
            if not normalized:
                raise ValueError(
                    f"Characteristic '{characteristic_name}' has no labeled providers for "
                    f"parametrization '{parametrization_name}'."
                )
            return normalized

        def _normalize_characteristic(
            characteristic_name: GenericCharacteristicName,
            value: CharacteristicProvider,
        ) -> dict[ParametrizationName, dict[LabelName, CharacteristicFunction[Any, Any]]]:
            if not isinstance(value, Mapping):
                base_name = self.base_parametrization_name
                return {
                    base_name: _normalize_labeled_provider(characteristic_name, base_name, value)
                }

            normalized_by_parametrization: dict[
                ParametrizationName, dict[LabelName, CharacteristicFunction[Any, Any]]
            ] = {}
            for parametrization_name, provider in value.items():
                normalized_by_parametrization[parametrization_name] = _normalize_labeled_provider(
                    characteristic_name,
                    parametrization_name,
                    provider,
                )
            return normalized_by_parametrization

        self.distr_characteristics: dict[
            GenericCharacteristicName,
            dict[ParametrizationName, dict[LabelName, CharacteristicFunction[Any, Any]]],
        ] = {
            characteristic_name: _normalize_characteristic(characteristic_name, provider)
            for characteristic_name, provider in distr_characteristics.items()
        }

        # Validate characteristic providers
        valid_names = set(self.parametrization_names)
        for char_name, forms in self.distr_characteristics.items():
            if not forms:
                raise ValueError(f"Characteristic '{char_name}' has no providers.")
            unknown = set(forms) - valid_names
            if unknown:
                raise ValueError(
                    f"Characteristic '{char_name}' has providers for unknown parametrizations: "
                    f"{sorted(unknown)}."
                )

        # Precompute analytical plan: for each parametrization pick provider (self or base)
        self._analytical_plan: dict[
            ParametrizationName, dict[GenericCharacteristicName, ParametrizationName]
        ] = {}
        base = self.base_parametrization_name
        for pname in self.parametrization_names:
            plan: dict[GenericCharacteristicName, ParametrizationName] = {}
            for characteristic, forms in self.distr_characteristics.items():
                if pname in forms:
                    plan[characteristic] = pname
                elif base in forms:
                    plan[characteristic] = base
            self._analytical_plan[pname] = plan

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
        """Support resolver callable."""
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

    @staticmethod
    def _bind_parametrization[In, Out](
        func: CharacteristicFunction[In, Out], params_obj: Parametrization
    ) -> ComputationFunc[In, Out]:
        """Bind ``params_obj`` to ``func`` only when ``func`` can accept positional arguments.

        This allows parametrization-independent analytical providers to be written without
        a dummy first argument (e.g. ``def skew_func()`` or ``def kurt_func(*, excess=False)``),
        while still supporting the usual ``def f(parameters, ...)`` style.

        It means that we will always make any other analytical_computation params like
        ``excess`` as keyword-only
        """

        sig = inspect.signature(func)

        params = list(sig.parameters.values())
        accepts_first_positional = bool(params) and params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )

        return cast(
            ComputationFunc[In, Out],
            partial(cast(ParametrizedCharacteristic[In, Out], func), params_obj)
            if accepts_first_positional
            else func,
        )

    def _build_analytical_computations(
        self, parameters: Parametrization
    ) -> dict[GenericCharacteristicName, dict[LabelName, AnalyticalComputation[Any, Any]]]:
        """
        Build analytical computations for given parameters.

        Uses precomputed provider plan for efficient computation.
        """
        plan = self._analytical_plan.get(parameters.name, {})
        result: dict[
            GenericCharacteristicName, dict[LabelName, AnalyticalComputation[Any, Any]]
        ] = {}
        base_params: Parametrization | None = None

        for characteristic, provider_name in plan.items():
            if provider_name == parameters.name:
                params_obj = parameters
            else:
                base_params = base_params or self.to_base(parameters)
                params_obj = base_params

            labeled_providers = self.distr_characteristics[characteristic][provider_name]
            result[characteristic] = {
                label_name: AnalyticalComputation(
                    target=characteristic,
                    func=self._bind_parametrization(func_factory, params_obj),
                )
                for label_name, func_factory in labeled_providers.items()
            }

        return result

    def distribution(
        self,
        parametrization_name: ParametrizationName | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
        **parameters_values: Any,
    ) -> ParametricFamilyDistribution:
        """
        Create a distribution instance with given parameters.

        Parameters
        ----------
        parametrization_name : ParametrizationName | None, optional
            Name of parametrization to use (defaults to base).
        sampling_strategy : SamplingStrategy
            Strategy for generating random samples. Such an object is unique for each distribution.
        computation_strategy : ComputationStrategy
            Strategy for computing characteristics and conversions.
            Such an object is unique for each distribution.
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
        parametrization_class = (
            self.base
            if parametrization_name is None
            else self._parametrizations[parametrization_name]
        )

        parameters = parametrization_class(**parameters_values)
        parameters.validate()
        base_parameters = self.to_base(parameters)
        distribution_type = self._distr_type(base_parameters)
        analytical_computations = self._build_analytical_computations(parameters)
        return ParametricFamilyDistribution(
            family_name=self.name,
            distribution_type=distribution_type,
            analytical_computations=analytical_computations,
            parametrization=parameters,
            support=self.support_resolver(parameters),
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
        )

    @dataclass_transform()
    def parametrization(
        self, *, name: ParametrizationName
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
