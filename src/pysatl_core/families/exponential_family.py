from __future__ import annotations

__author__ = "Vinogradov Ilya"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.differentiate import jacobian
from scipy.integrate import nquad
from scipy.linalg import det

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
    UnivariateContinuous,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support
    from pysatl_core.types import Number, NumberParameter, NumericArray

    type ParametrizedFunction = Callable[[Parametrization, Any], Any]
    type SupportArg = Callable[[Parametrization], Support | None] | None


@dataclass
class ExponentialFamilyParametrization(Parametrization):
    """
    Standard parametrization of Exponential Family.
    """

    theta: NumberParameter

    def transform_to_base_parametrization(self) -> ExponentialFamilyParametrization:
        return self


@dataclass
class ExponentialConjugateHyperparameters(Parametrization):
    effective_suff_stat_value: NumericArray
    effective_sample_size: Number

    def transform_to_base_parametrization(self) -> ExponentialFamilyParametrization:
        return ExponentialFamilyParametrization(
            np.append(self.effective_suff_stat_value, self.effective_sample_size)
        )


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
        support_by_parametrization: SupportArg = None,
        base_score: Callable[[Parametrization, NumericArray], NumericArray] | None = None,
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
            CharacteristicName.MEAN_DEFAULT: self._mean,
            CharacteristicName.VAR_DEFAULT: self._var,
        }

        ParametricFamily.__init__(
            self,
            name=name,
            distr_type=distr_type,
            distr_parametrizations=distr_parametrizations,
            distr_characteristics=distr_characteristics,
            support_by_parametrization=support_by_parametrization,
            base_score=base_score,
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
            parameter_space=SupportByPredicate(predicate=conjugate_sufficient_accepts),  # type: ignore[arg-type]
            name=self.name,
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
            support_by_parametrization=self.support_resolver,
        )

    def transform(
        self,
        transform_function: Callable[[NumberParameter], NumberParameter],
    ) -> ContinuousExponentialClassFamily:
        def calculate_jacobian(x: NumberParameter) -> NumberParameter:
            if not isinstance(x, Iterable):
                x = np.array([x])

            return np.abs(det(jacobian(transform_function, x).df))

        def new_support(x: NumberParameter) -> bool:
            return transform_function(x) in self._support

        def new_sufficient(x: NumberParameter) -> NumberParameter:
            return self._sufficient(transform_function(x))

        def new_normalization(x: NumberParameter) -> NumberParameter:
            return self._normalization(x) * calculate_jacobian(x)

        return ContinuousExponentialClassFamily(
            log_partition=self._log_partition,
            sufficient_statistics=new_sufficient,
            normalization_constant=new_normalization,
            support=SupportByPredicate(predicate=new_support),
            parameter_space=self._parameter_space,
            sufficient_statistics_values=self._sufficient_statistics_values,
            name=f"Transformed{self._name}",
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
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
        self, parametrizaiton: ExponentialConjugateHyperparameters, sample: list[Any]
    ) -> ExponentialConjugateHyperparameters:
        posterior_effective_suff_stat_value = parametrizaiton.effective_suff_stat_value
        posterior_effective_sample_size = parametrizaiton.effective_sample_size
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

    @property
    def posterior_predictive(self) -> ParametricFamily:
        def conjugate_log_partition(
            parametrization: ExponentialConjugateHyperparameters,
        ) -> NumberParameter:
            conjugate_value = self.conjugate_prior_family._log_partition(
                parametrization.transform_to_base_parametrization().theta
            )
            return np.exp(conjugate_value)

        def posterior_density(parametrization: Parametrization, x: NumberParameter) -> Number:
            parametrization = cast(ExponentialConjugateHyperparameters, parametrization)
            return cast(
                np.float32,
                self._normalization(x)
                * conjugate_log_partition(parametrization)
                / conjugate_log_partition(
                    self.posterior_hyperparameters(
                        parametrizaiton=parametrization, sample=[self._sufficient(x)]
                    )
                ),
            )

        family = ParametricFamily(
            name=f"PosteriorPredictive{self.name}",
            distr_type=UnivariateContinuous,
            distr_characteristics={CharacteristicName.PDF: posterior_density},
            distr_parametrizations=["posterior"],
            support_by_parametrization=lambda _: ContinuousSupport(),
        )
        parametrization(family=family, name="posterior")(ExponentialConjugateHyperparameters)
        return family
