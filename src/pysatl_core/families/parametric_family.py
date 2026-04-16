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
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, dataclass_transform, overload

import numpy as np

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.families.parametrizations import Parametrization
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    ComputationFunc,
    DistributionType,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pysatl_core.distributions.strategies import ComputationStrategy, SamplingStrategy
    from pysatl_core.distributions.support import Support
    from pysatl_core.types import (
        GenericCharacteristicName,
        LabelName,
        NumericArray,
        ParametrizationName,
    )

    type SupportArg = Callable[[Parametrization], Support | None] | None
    type SupportResolver = Callable[[Parametrization], Support | None]
    type LabeledCharacteristicProvider = (
        Mapping[LabelName, ParametricFamilyCharacteristic[Any, Any]]
        | ParametricFamilyCharacteristic[Any, Any]
    )
    type CharacteristicProvider = (
        Mapping[ParametrizationName, LabeledCharacteristicProvider]
        | ParametricFamilyCharacteristic[Any, Any]
    )
    type CharacteristicsMap = Mapping[GenericCharacteristicName, CharacteristicProvider]
    type NonParametrizedCharacteristic[In, Out] = Callable[[], Out]
    type ParametricFamilyCharacteristic[In, Out] = (
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
        base_score: Callable[[Parametrization, NumericArray], NumericArray] | None = None,
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
        self._base_score = base_score

        # Runtime registry of parametrization classes
        self._parametrizations: dict[ParametrizationName, type[Parametrization]] = {}

        def _normalize_labeled_provider(
            characteristic_name: GenericCharacteristicName,
            parametrization_name: ParametrizationName,
            provider: LabeledCharacteristicProvider,
        ) -> dict[LabelName, ParametricFamilyCharacteristic[Any, Any]]:
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
        ) -> dict[ParametrizationName, dict[LabelName, ParametricFamilyCharacteristic[Any, Any]]]:
            if not isinstance(value, Mapping):
                base_name = self.base_parametrization_name
                return {
                    base_name: _normalize_labeled_provider(characteristic_name, base_name, value)
                }

            normalized_by_parametrization: dict[
                ParametrizationName, dict[LabelName, ParametricFamilyCharacteristic[Any, Any]]
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
            dict[ParametrizationName, dict[LabelName, ParametricFamilyCharacteristic[Any, Any]]],
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
        func: ParametricFamilyCharacteristic[In, Out], params_obj: Parametrization
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

    def score(self, parameters: Parametrization, x: NumericArray) -> NumericArray:
        """
        Compute the score (gradient of log‑PDF) for the given parametrization.

        Parameters
        ----------
        parameters : Parametrization
            Parametrization instance of the family.
        x : NumericArray
            Points at which to evaluate the gradient.

        Returns
        -------
        NumericArray
            Gradient with respect to the parameters of the given parametrization.
            Shape is (..., d), where d is the number of parameters of the parametrization.
        """
        if self._base_score is None:
            raise ValueError(
                f"Family '{self.name}' does not provide score (gradient) method. "
                "Please pass '_base_score' to the constructor."
            )
        x_arr = np.atleast_1d(x)

        base_params = parameters.transform_to_base_parametrization()
        base_grad = self._base_score(base_params, x_arr)
        return parameters.gradient_transform(base_grad)

    def view(
        self,
        *,
        parametrization_name: str | None = None,
        **fixed_params: Any,
    ) -> PartialParametricFamily:
        """
        Create a view of this family with partially fixed parameters.

        Parameters
        ----------
        parametrization_name : str, optional
            Name of the parametrization in which the fixed parameters are given.
            If not provided, the base parametrization of the family is used.
        **fixed_params : Any
            Parameter names and values to fix.

        Returns
        -------
        PartialParametricFamily
            A view that behaves like the original family but with the specified
            parameters fixed.

        Examples
        --------
        >>> uniform = ParametricFamilyRegister.get("uniform")
        >>> uniform_lower0 = uniform.view(lower_bound=0)
        >>> dist = uniform_lower0.distribution(upper_bound=1)  # Uniform(0,1)
        """
        if parametrization_name is not None and parametrization_name not in self.parametrizations:
            raise ValueError(
                f"Unknown parametrization '{parametrization_name}' for family '{self.name}'"
            )
        return PartialParametricFamily(
            base_family=self,
            fixed_params=fixed_params,
            parametrization_name=parametrization_name,
        )

    __call__ = distribution


class PartialParametricFamily(ParametricFamily):
    """
    View on a parametric family with partially fixed parameters.

    This class represents a parametric family where some parameters have been
    fixed to specific values. It inherits all behaviour from `ParametricFamily`
    but restricts the available parametrization to the one in which parameters
    are fixed. All analytical characteristics are preserved via delegation to
    the base parametrization of the original family.

    Parameters
    ----------
    base_family : ParametricFamily
        The original parametric family.
    fixed_params : dict[str, Any]
        Dictionary of fixed parameter names and their values.
    parametrization_name : str, optional
        Name of the parametrization in which the fixed parameters are specified.
        If not provided, the base parametrization of the family is used.

    Raises
    ------
    ValueError
        If all parameters of the chosen parametrization are fixed (use `.distribution()` directly),
        or if any fixed parameter name is unknown for that parametrization,
        or if the parametrization name is not registered in the family.
    """

    def __init__(
        self,
        base_family: ParametricFamily,
        fixed_params: dict[str, Any],
        parametrization_name: str | None = None,
    ) -> None:
        self._fixed_in_param = parametrization_name or base_family.base_parametrization_name
        self._base_family = base_family
        self._param_class = base_family.get_parametrization(self._fixed_in_param)
        self._fixed_params = fixed_params.copy()

        required_fields = set(getattr(self._param_class, "__dataclass_fields__", {}).keys())
        # Validate that fixed parameters exist
        unknown = set(self._fixed_params) - required_fields
        if unknown:
            raise ValueError(
                f"Unknown parameters for parametrization '{self._fixed_in_param}': {unknown}"
            )

        # Check completeness: if all parameters are fixed, raise an error
        if required_fields.issubset(fixed_params):
            raise ValueError(
                f"All parameters of parametrization '{self._fixed_in_param}' are already fixed. "
                "Use `.distribution()` directly."
            )

        # Generate lightweight parametrization with only free fields
        self._free_param_class = self._create_free_param_class()
        # Assign __param_name__ and __family__ so that instances have .name and .family
        self._free_param_class.__param_name__ = self._fixed_in_param
        self._free_param_class.__family__ = self

        self._free_parameter_names = tuple(
            name
            for name in getattr(self._param_class, "__dataclass_fields__", {})
            if name not in self._fixed_params
        )

        def _view_distr_type(params: Parametrization) -> DistributionType:
            canonical = base_family.to_base(params)
            return base_family._distr_type(canonical)

        view_chars = self._build_view_characteristics(base_family)

        super().__init__(
            name=base_family._name,
            distr_type=_view_distr_type,
            distr_parametrizations=[self._fixed_in_param],
            distr_characteristics=view_chars,
            support_by_parametrization=base_family._support_resolver,
            base_score=base_family._base_score,
        )

        # Register the parametrization (needed for parent methods)
        self.register_parametrization(self._fixed_in_param, self._free_param_class)

    def _create_free_param_class(self) -> type[Parametrization]:
        """Create a parametrization class containing only the free (unfixed) parameters.

        The generated class exposes only the fields that were *not* fixed via
        :meth:`view`.

        Its :meth:`transform_to_base_parametrization` automatically injects the
        fixed values and delegates to the conversion logic of the
        original parametrization class.

        The :meth:`validate` method substitutes
        the fixed values before running the original validation.

        The :meth:`gradient_transform` method maps a base-parametrization gradient to
        the subspace of free parameters only.

        Returns
        -------
        type[Parametrization]
            A lightweight parametrization class with only the unfixed fields.
        """
        fixed_params = self._fixed_params
        original_class = self._param_class
        all_fields = getattr(original_class, "__dataclass_fields__", {})
        free_field_names = [name for name in all_fields if name not in fixed_params]

        def __init__(self: Parametrization, **kwargs: Any) -> None:
            unexpected = set(kwargs) - set(free_field_names)
            if unexpected:
                raise TypeError(
                    f"__init__() got unexpected keyword arguments: "
                    f"{', '.join(repr(u) for u in unexpected)}"
                )
            missing = set(free_field_names) - set(kwargs)
            if missing:
                raise TypeError(
                    f"__init__() missing required keyword arguments: "
                    f"{', '.join(repr(m) for m in missing)}"
                )
            for name in free_field_names:
                object.__setattr__(self, name, kwargs[name])

        def transform_to_base(self: Parametrization) -> Parametrization:
            """Substitute fixed values and delegate to the original parametrization."""
            combined = {
                **fixed_params,
                **{f: getattr(self, f) for f in free_field_names},
            }
            original_instance = original_class(**combined)
            return original_instance.transform_to_base_parametrization()

        def validate(self: Parametrization) -> None:
            """Validate by combining fixed and free parameters, then delegating."""
            combined = {
                **fixed_params,
                **{f: getattr(self, f) for f in free_field_names},
            }
            original_class(**combined).validate()

        def gradient_transform(self: Parametrization, base_grad: NumericArray) -> NumericArray:
            """Map a gradient from the base parametrization to free-parameter space.

            The base gradient is first transformed into the original (full)
            parametrization.  Components that correspond to fixed parameters are
            then discarded, keeping only the directions of the free parameters.
            """
            combined = {
                **fixed_params,
                **{f: getattr(self, f) for f in free_field_names},
            }
            full_instance = original_class(**combined)
            full_grad = full_instance.gradient_transform(base_grad)
            all_field_names = list(all_fields.keys())
            free_indices = [i for i, name in enumerate(all_field_names) if name in free_field_names]
            return full_grad[..., free_indices]

        new_class = type(
            f"{original_class.__name__}Free",
            (Parametrization,),
            {
                "__init__": __init__,
                "transform_to_base_parametrization": transform_to_base,
                "validate": validate,
                "gradient_transform": gradient_transform,
                "__dataclass_fields__": {name: all_fields[name] for name in free_field_names},
                "__annotations__": {name: all_fields[name].type for name in free_field_names},
            },
        )
        return new_class

    @property
    def parametrizations(self) -> dict[str, type[Parametrization]]:
        """Return a dictionary containing only the fixed (free‑parameter) parametrization."""
        return {self._fixed_in_param: self._free_param_class}

    @property
    def parent_family(self) -> ParametricFamily:
        """Original family this view was created from."""
        return self._base_family

    @property
    def fixed_parameters(self) -> Mapping[str, Any]:
        """Fixed parameter values."""
        return MappingProxyType(self._fixed_params)

    @property
    def fixed_parameter_names(self) -> frozenset[str]:
        """Names of fixed parameters."""
        return frozenset(self._fixed_params)

    @property
    def free_parameter_names(self) -> tuple[str, ...]:
        """Names of parameters that remain free in this view."""
        return self._free_parameter_names

    @overload
    def get_parametrization(self) -> type[Parametrization]: ...

    @overload
    def get_parametrization(self, name: ParametrizationName) -> type[Parametrization]: ...

    def get_parametrization(self, name: ParametrizationName | None = None) -> type[Parametrization]:
        """Return the lightweight parametrization class with only free parameters.

        If `name` is omitted, returns the fixed parametrization class.
        If `name` is given, it must match the fixed parametrization name.

        Raises KeyError for any other name.
        """
        if name is None:
            return self._free_param_class
        if name != self._fixed_in_param:
            raise KeyError(
                f"Parametrization '{name}' is not available in this view. "
                f"Only '{self._fixed_in_param}' is available."
            )
        return self._free_param_class

    @property
    def base(self) -> type[Parametrization]:
        """Return a lightweight parametrization class with only free parameters.

        Its ``transform_to_base_parametrization`` substitutes fixed values and
        delegates to the original parametrization.
        """
        return self._free_param_class

    def to_base(self, parameters: Parametrization) -> Parametrization:
        """Convert view parameters to the original family's base parametrization.
        The view's own base is the lightweight class, but the true base is the
        original family's base. We always transform through the full parametrization.
        """
        return parameters.transform_to_base_parametrization()

    def _build_view_characteristics(self, base_family: ParametricFamily) -> dict[str, Any]:
        view_chars = {}
        original_base = base_family.base_parametrization_name

        def wrap_provider(
            provider: ParametricFamilyCharacteristic[Any, Any],
        ) -> ParametricFamilyCharacteristic[Any, Any]:
            def wrapped(params: Parametrization, *args: Any, **kwargs: Any) -> Any:
                base_params = base_family.to_base(params)
                bound = ParametricFamily._bind_parametrization(provider, base_params)
                return bound(*args, **kwargs)

            return wrapped

        def wrap_fixed_provider(
            provider: ParametricFamilyCharacteristic[Any, Any],
        ) -> ParametricFamilyCharacteristic[Any, Any]:
            def wrapped(params: Parametrization, *args: Any, **kwargs: Any) -> Any:
                combined = {
                    **self._fixed_params,
                    **{f: getattr(params, f) for f in self.free_parameter_names},
                }
                full_params = self._param_class(**combined)
                bound = ParametricFamily._bind_parametrization(provider, full_params)
                return bound(*args, **kwargs)

            return wrapped

        for char_name, char_map in base_family.distr_characteristics.items():
            if self._fixed_in_param in char_map:
                providers = char_map[self._fixed_in_param]
                wrapped_providers = {
                    label: wrap_fixed_provider(provider) if callable(provider) else provider
                    for label, provider in providers.items()
                }
                view_chars[char_name] = {self._fixed_in_param: wrapped_providers}
            elif original_base in char_map:
                original_provider = char_map[original_base]
                wrapped = {
                    label: wrap_provider(provider) if callable(provider) else provider
                    for label, provider in original_provider.items()
                }
                view_chars[char_name] = {self._fixed_in_param: wrapped}

        return view_chars

    def distribution(
        self,
        parametrization_name: str | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
        **kwargs: Any,
    ) -> ParametricFamilyDistribution:
        target = parametrization_name or self._fixed_in_param
        if target != self._fixed_in_param:
            raise ValueError(
                f"Only parametrization '{self._fixed_in_param}' is available in this view. "
                "Please omit 'parametrization_name' or use the fixed one."
            )
        for key, fixed_val in self._fixed_params.items():
            if key in kwargs and kwargs[key] != fixed_val:
                raise ValueError(
                    f"Parameter '{key}' is fixed to {fixed_val}, but got {kwargs[key]}"
                )
        return super().distribution(
            parametrization_name=target,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
            **kwargs,
        )

    def view(
        self,
        *,
        parametrization_name: str | None = None,
        **additional_params: Any,
    ) -> PartialParametricFamily:
        if parametrization_name is not None and parametrization_name != self._fixed_in_param:
            raise ValueError(
                f"Cannot change parametrization. Current fixed parametrization is "
                f"'{self._fixed_in_param}'. Use the same or omit the argument."
            )
        for key, fixed_val in self._fixed_params.items():
            if key in additional_params and additional_params[key] != fixed_val:
                raise ValueError(
                    f"Parameter '{key}' is fixed to {fixed_val}, but got {additional_params[key]}"
                )
        new_fixed = {**self._fixed_params, **additional_params}
        return PartialParametricFamily(self._base_family, new_fixed, self._fixed_in_param)

    __call__ = distribution
