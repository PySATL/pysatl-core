from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.differentiate import jacobian
from scipy.integrate import nquad
from scipy.linalg import det

from pysatl_core.distributions import (
    SamplingStrategy,
)
from pysatl_core.distributions.support import (
    ContinuousSupport,
    SupportByPredicate,
)
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import Parametrization, parametrization
from pysatl_core.types import (
    CharacteristicName,
    DistributionType,
    GenericCharacteristicName,
    ParametrizationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support
    from pysatl_core.types import Number, NumericArray

    type ParametrizedFunction = Callable[[Parametrization, Any], Any]
    type SupportArg = Callable[[Parametrization], Support | None] | None
    type NumberParameter = Number | NumericArray


@dataclass
class ExponentialFamilyParametrization(Parametrization):
    """
    Standard parametrization of Exponential Family.
    """

    theta: NumberParameter

    def transform_to_base_parametrization(self) -> ExponentialFamilyParametrization:
        return self


@dataclass
class ExponentialConjugateHyperparameters:
    effective_suff_stat_value: NumberParameter
    effective_sample_size: int


class ContinuousExponentialClassFamily(ParametricFamily):
    """
    Representation of exponential class with density = h(x) * exp(<n(t), T(x)> + A(t)),
    where canonical parametrization is that, when n = t

    Usage of this class:
    - you can use method transform_to_another to replace x to smth else, for example, into
    """

    def __init__(
        self,
        *,
        log_partition: Callable[[NumberParameter], NumberParameter],
        sufficient_statistics: Callable[[NumberParameter], NumberParameter],
        normalization_constant: Callable[[NumberParameter], NumberParameter],
        support: SupportByPredicate,
        parameter_space: SupportByPredicate,
        sufficient_statistics_values: SupportByPredicate,
        name: str = "ExponentialFamily",
        distr_type: DistributionType | Callable[[Parametrization], DistributionType],
        distr_parametrizations: list[ParametrizationName],
        sampling_strategy: SamplingStrategy,
        support_by_parametrization: SupportArg = None,
    ):
        self._sufficient = sufficient_statistics
        self._log_partition = log_partition
        self._normalization = normalization_constant

        self._support = support
        self._parameter_space = parameter_space
        self._sufficient_statistics_values = sufficient_statistics_values

        distr_characteristics: dict[
            GenericCharacteristicName,
            dict[ParametrizationName, ParametrizedFunction] | ParametrizedFunction,
        ] = {
            CharacteristicName.PDF: self.density,
            CharacteristicName.MEAN: self._mean,
            CharacteristicName.VAR: self._var,
        }

        ParametricFamily.__init__(
            self,
            name=name,
            distr_type=distr_type,
            distr_parametrizations=distr_parametrizations,
            distr_characteristics=distr_characteristics,
            sampling_strategy=sampling_strategy,
            support_by_parametrization=support_by_parametrization,
        )
        parametrization(family=self, name="theta")(ExponentialFamilyParametrization)

    @property
    def log_density(self) -> ParametrizedFunction:
        def log_density_func(parametrization: Parametrization, x: NumberParameter) -> Number:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            parametrization = parametrization.transform_to_base_parametrization()
            if x not in self._support:
                return -np.inf

            theta = parametrization.theta
            sufficient = self._sufficient(x)
            dot = np.dot(theta, sufficient)
            if hasattr(dot, "__len__"):
                dot = dot[0]

            result = np.log(self._normalization(x)) + dot + self._log_partition(theta)
            return cast(np.floating, result.item())

        return log_density_func

    @property
    def density(self) -> ParametrizedFunction:
        return lambda parametrization, x: np.exp(self.log_density(parametrization, x))

    @property
    def conjugate_prior_family(self) -> ContinuousExponentialClassFamily:
        def conjugate_sufficient(
            theta: NumberParameter,
        ) -> NumberParameter:
            if not hasattr(theta, "__len__"):
                theta = np.array([theta])

            if theta not in self._parameter_space:
                return np.full(len(theta) + 1, float("-inf"))
            # print("Current sufficient:", theta, self._log_partition(theta))
            return np.append(theta, self._log_partition(theta))

        def conjugate_log_partition(
            parametrization: NumberParameter,
        ) -> NumberParameter:
            def pdf(theta: NumberParameter) -> NumberParameter:
                if not hasattr(theta, "__len__"):
                    theta = np.array([theta])
                return cast(
                    np.floating,
                    np.exp(
                        np.dot(
                            conjugate_sufficient(theta),
                            parametrization,
                        )
                    ).item(),
                )

            all_value = nquad(
                lambda x: pdf(x) if x in self._parameter_space else 0,  # type: ignore[arg-type]
                [(float("-inf"), float("+inf"))],
            )[0]
            return cast(np.float64, -np.log(all_value))

        def conjugate_sufficient_accepts(
            theta: NumericArray,
        ) -> bool:
            xi = theta[:-1]
            nu = theta[-1]

            return xi in self._sufficient_statistics_values and nu in ContinuousSupport(0, np.inf)

        return ContinuousExponentialClassFamily(
            log_partition=conjugate_log_partition,
            sufficient_statistics=conjugate_sufficient,
            normalization_constant=lambda _: 1,
            support=self._parameter_space,
            sufficient_statistics_values=self._parameter_space,  # TODO: write convex hull for this
            parameter_space=SupportByPredicate(conjugate_sufficient_accepts),  # type: ignore[arg-type]
            name=self.name,
            sampling_strategy=self.sampling_strategy,
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
            support_by_parametrization=self.support_resolver,
        )

    def transform(
        self,
        transform_function: Callable[[Any], Any],
    ) -> ContinuousExponentialClassFamily:
        def calculate_jacobian(x: Any) -> Any:
            if type(x) is not list:
                x = np.array([x])

            return np.abs(det(jacobian(transform_function, x).df))

        def new_support(x: Any) -> bool:
            return transform_function(x) in self._support

        def new_sufficient(x: Any) -> Any:
            return self._sufficient(transform_function(x))

        def new_normalization(x: Any) -> Any:
            return self._normalization(x) * calculate_jacobian(x)

        return ContinuousExponentialClassFamily(
            log_partition=self._log_partition,
            sufficient_statistics=new_sufficient,
            normalization_constant=new_normalization,
            support=SupportByPredicate(new_support),
            parameter_space=self._parameter_space,
            sufficient_statistics_values=self._sufficient_statistics_values,
            name=f"Transformed{self._name}",
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
            sampling_strategy=self.sampling_strategy,
            support_by_parametrization=self.support_resolver,
        )

    @property
    def _mean(self) -> ParametrizedFunction:
        def mean_func(parametrization: Parametrization, x: Any) -> Any:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            dimension_size = 1
            if hasattr(x, "__len__"):
                dimension_size = len(x)
            return nquad(
                lambda x: (  # type: ignore[arg-type]
                    np.dot(x, self.density(parametrization, x)) if x in self._support else 0
                ),
                [(float("-inf"), float("inf"))] * dimension_size,
            )[0]

        return mean_func

    @property
    def _second_moment(self) -> ParametrizedFunction:
        def func(parametrization: Parametrization, x: Any) -> Any:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            dimension_size = 1
            if hasattr(x, "__len__"):
                dimension_size = len(x)
            return nquad(
                lambda x: (  # type: ignore[arg-type]
                    x**2 * self.density(parametrization, x) if x in self._support else 0
                ),
                [(float("-inf"), float("inf"))] * dimension_size,
            )[0]

        return func

    @property
    def _var(self) -> ParametrizedFunction:
        def func(parametrization: Parametrization, x: Any) -> Any:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            return self._second_moment(parametrization, x) - self._mean(parametrization, x) ** 2

        return func

    def posterior_hyperparameters(
        self, prior_hyper: ExponentialConjugateHyperparameters, sample: list[Any]
    ) -> ExponentialConjugateHyperparameters:
        posterior_effective_suff_stat_value = prior_hyper.effective_suff_stat_value
        posterior_effective_sample_size = prior_hyper.effective_sample_size
        if hasattr(sample, "__iter__") and not isinstance(sample, str):
            posterior_effective_suff_stat_value += np.sum(
                [self._sufficient(x) for x in sample],  # type: ignore[arg-type]
                axis=0,
            )
            posterior_effective_sample_size += len(sample)
        else:
            posterior_effective_suff_stat_value += self._sufficient(sample)  # type: ignore[arg-type]
            posterior_effective_sample_size += 1

        return ExponentialConjugateHyperparameters(
            effective_suff_stat_value=posterior_effective_suff_stat_value,
            effective_sample_size=posterior_effective_sample_size,
        )
