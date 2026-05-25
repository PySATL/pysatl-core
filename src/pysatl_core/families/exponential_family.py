"""
Exponential family distributions in continuous spaces.

This module implements the continuous exponential family of probability distributions,
their conjugate priors, posterior inference, and posterior predictive distributions.
"""

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
    PredicateSupport,
)
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import Parametrization, constraint, parametrization
from pysatl_core.types import (
    CharacteristicName,
    DistributionType,
    Number,
    NumericArray,
    ParametrizationName,
    UnivariateContinuous,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support
    from pysatl_core.families.parametric_family import (
        CharacteristicsMap,
        ParametricFamilyCharacteristic,
    )

    type SupportArg = Callable[[Parametrization], Support | None] | None


@dataclass
class ExponentialFamilyParametrization(Parametrization):
    """
    Standard parametrization of an exponential family distribution.

    This parametrization uses the natural (canonical) parameter vector `theta`
    The density is expressed as:
        f(x|θ) = h(x) * exp(θᵀ T(x) - A(θ))

    Attributes:
        theta (NumericArray): Natural parameter vector (can be a scalar or array)
    """

    theta: NumericArray

    def transform_to_base_parametrization(self) -> ExponentialFamilyParametrization:
        """Return the base parametrization (identity transform for canonical form)."""
        return self


@dataclass
class ExponentialConjugateHyperparameters(Parametrization):
    """
    Hyperparameters for the conjugate prior of an exponential family

    For a prior of the form:
        p(θ) ∝ exp(ν₀ᵀ T(θ) + n₀ A(θ))
    the hyperparameters are:
        effective_suff_stat_value = ν₀
        effective_sample_size = n₀

    Attributes:
        effective_suff_stat_value (NumericArray): Pseudo‑sufficient statistic ν₀
        effective_sample_size (Number): Pseudo‑sample size n₀ (a non‑negative scalar)
    """

    effective_suff_stat_value: NumericArray
    effective_sample_size: Number

    def transform_to_base_parametrization(self) -> ExponentialFamilyParametrization:
        """
        Convert hyperparameters to a canonical parametrization.

        The resulting parameter vector is [ν₀, n₀] concatenated.
        """
        return ExponentialFamilyParametrization(
            np.append(self.effective_suff_stat_value, self.effective_sample_size)
        )


class ContinuousExponentialClassFamily(ParametricFamily):
    """
    Representation of a continuous exponential family distribution.

    The density is given by:
        f(x|θ) = h(x) * exp(θᵀ T(x) + A(θ))

    where:
        - θ is the natural parameter,
        - T(x) is the sufficient statistic vector,
        - h(x) is the base measure (the `normalization_constant`),
        - A(θ) is the log‑partition function.

    This class supports:
        - Canonical parametrization (θ) via `ExponentialFamilyParametrization`.
        - Conjugate prior families.
        - Posterior updates and posterior predictive distributions.
        - Transformation of the random variable (change of variable with Jacobian).

    The user must supply functions for the log‑partition `log_partition`,
    sufficient statistics `sufficient_statistics`,
    base measure `normalization_constant`, as well as the support of the distribution,
    the natural parameter space and the range of the sufficient statistic.
    """

    def __init__(
        self,
        *,
        log_partition: Callable[[NumericArray], NumericArray],
        sufficient_statistics: Callable[[NumericArray], NumericArray],
        normalization_constant: Callable[[NumericArray], Number],
        support: Support,
        parameter_space: Support,
        sufficient_statistics_values: Support,
        name: str,
        distr_type: DistributionType | Callable[[Parametrization], DistributionType],
        distr_parametrizations: list[ParametrizationName],
        distr_characteristics: CharacteristicsMap | None = None,
        support_by_parametrization: SupportArg = None,
        base_score: Callable[[Parametrization, NumericArray], NumericArray] | None = None,
    ):
        """
        Initialize a continuous exponential family distribution.

        Args:
            log_partition: Function A(θ) – the log‑partition function.
            sufficient_statistics: Function T(x) – the sufficient statistic vector.
            normalization_constant: Function h(x) – the base measure.
            support: Predicate defining the support of the distribution.
            parameter_space: Predicate defining the natural parameter space.
            sufficient_statistics_values: Predicate defining the range of T(x).
            name: Name of the family.
            distr_type: Type of distribution or a callable returning it.
            distr_parametrizations: List of parametrization names this family supports.
            distr_characteristics: Additional analytical characteristics to register.
            support_by_parametrization: Callable that returns the support given a parametrization.
            base_score: Optional base score function.
        """
        self._sufficient = sufficient_statistics
        self._log_partition = log_partition
        self._normalization = normalization_constant

        self._support = support
        self._parameter_space = parameter_space
        self._sufficient_statistics_values = sufficient_statistics_values

        family_characteristics: CharacteristicsMap = {
            CharacteristicName.PDF: self.density,
            CharacteristicName.MEAN: self._mean,
            CharacteristicName.VAR: self._var,
        }
        merged_characteristics = dict(distr_characteristics or {})
        merged_characteristics.update(family_characteristics)

        ParametricFamily.__init__(
            self,
            name=name,
            distr_type=distr_type,
            distr_parametrizations=distr_parametrizations,
            distr_characteristics=merged_characteristics,
            support_by_parametrization=support_by_parametrization,
            base_score=base_score,
        )

        @parametrization(family=self, name="theta")
        class ThetaParametrization(ExponentialFamilyParametrization):
            @constraint(description="theta belongs to parameter_space")
            def check_theta_in_parameter_space(self) -> bool:
                theta = np.atleast_1d(np.asarray(self.theta, dtype=float))
                return bool(self.__family__._parameter_space.contains(theta))  # type: ignore[attr-defined]

    @property
    def log_density(self) -> ParametricFamilyCharacteristic[NumericArray, Number]:
        """
        Log‑density function for the exponential family.

        The function takes a parametrization (must be `ExponentialFamilyParametrization`)
        and a point `x`, and returns log f(x|θ). Returns -inf for x outside the support.

        Returns:
            Callable[[Parametrization, NumericArray], Number]
        """

        def log_density_func(parametrization: Parametrization, x: NumericArray) -> Number:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            parametrization = parametrization.transform_to_base_parametrization()
            if not self._support.contains(np.array([x])):
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
    def density(self) -> ParametricFamilyCharacteristic[NumericArray, Number]:
        """
        Density function (exponentiated log‑density).

        Returns:
            Callable[[Parametrization, NumericArray], Number]
        """
        log_density = cast(Callable[[Parametrization, NumericArray], Number], self.log_density)

        def density_func(parametrization: Parametrization, x: NumericArray) -> Number:
            return cast(Number, np.exp(log_density(parametrization, x)))

        return density_func

    @property
    def conjugate_prior_family(self) -> ContinuousExponentialClassFamily:
        """
        Build the conjugate prior family for this exponential family.

        The conjugate prior is an exponential family in the natural parameter θ,
        with sufficient statistic [θ, A(θ)] and base measure 1. The resulting
        family has its own [log_partition, sufficient_statistics, ...] such that
        the posterior updates are given by adding the observed sufficient statistics.

        Returns:
            ContinuousExponentialClassFamily: The conjugate prior family.
        """

        def conjugate_sufficient(
            theta: NumericArray,
        ) -> NumericArray:
            if not hasattr(theta, "__len__"):
                theta = np.array([theta])

            if not self._parameter_space.contains(theta):
                return np.full(len(theta) + 1, float("-inf"))
            return np.append(theta, self._log_partition(theta))

        def conjugate_log_partition(
            parametrization: NumericArray,
        ) -> NumericArray:
            def pdf(theta: NumericArray) -> Number:
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

            def integrand(x: float) -> float:
                theta = np.asarray([x], dtype=float)
                if not self._parameter_space.contains(theta):
                    return 0.0
                return float(pdf(theta))

            all_value = nquad(integrand, [(float("-inf"), float("+inf"))])[0]
            return np.array([cast(np.float64, -np.log(all_value))])

        def conjugate_sufficient_accepts(
            theta: NumericArray,
        ) -> bool:
            xi = theta[:-1]
            nu = theta[-1]

            return bool(self._sufficient_statistics_values.contains(xi)) and bool(
                ContinuousSupport(0, np.inf).contains(np.array([nu]))
            )

        return ContinuousExponentialClassFamily(
            log_partition=conjugate_log_partition,
            sufficient_statistics=conjugate_sufficient,
            normalization_constant=lambda _: 1,
            support=self._parameter_space,
            sufficient_statistics_values=self._parameter_space,
            parameter_space=PredicateSupport(predicate=conjugate_sufficient_accepts),
            name=self.name,
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
            support_by_parametrization=self.support_resolver,
        )

    def transform(
        self,
        transform_function: Callable[[NumericArray], NumericArray],
    ) -> ContinuousExponentialClassFamily:
        """
        Transform the random variable by a monotonic, differentiable function.

        The new density is obtained via the change‑of‑variable formula.
        The sufficient statistic becomes T(transform(x)) and the base measure
        gains the Jacobian factor.

        Args:
            transform_function: Invertible, differentiable function g(y) such that
                x = g(y). Must be defined on the original support.

        Returns:
            ContinuousExponentialClassFamily: A new family for the transformed variable.
        """

        def calculate_jacobian(x: NumericArray) -> NumericArray:
            if not isinstance(x, Iterable):
                x = np.array([x], dtype=float)
            else:
                x = np.atleast_1d(np.asarray(x, dtype=float))

            return np.abs(det(jacobian(transform_function, x).df))

        def new_support(x: NumericArray) -> bool:
            return bool(self._support.contains(transform_function(x)))

        def new_sufficient(x: NumericArray) -> NumericArray:
            return self._sufficient(transform_function(x))

        def new_normalization(x: NumericArray) -> Number:
            return cast(np.float64, self._normalization(x) * calculate_jacobian(x))

        return ContinuousExponentialClassFamily(
            log_partition=self._log_partition,
            sufficient_statistics=new_sufficient,
            normalization_constant=new_normalization,
            support=PredicateSupport(predicate=new_support),
            parameter_space=self._parameter_space,
            sufficient_statistics_values=self._sufficient_statistics_values,
            name=f"Transformed{self._name}",
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
            support_by_parametrization=self.support_resolver,
        )

    @property
    def _mean(self) -> ParametricFamilyCharacteristic[Any, Any]:
        """Compute the mean E[X] by numerical integration over the density."""

        def mean_func(parametrization: Parametrization) -> Any:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            density = cast(Callable[[Parametrization, NumericArray], Number], self.density)
            return nquad(
                lambda x: (
                    np.dot(x, density(parametrization, x))
                    if self._support.contains(np.array([x]))
                    else 0
                ),
                [(float("-inf"), float("inf"))],
            )[0]

        return mean_func

    @property
    def _second_moment(self) -> ParametricFamilyCharacteristic[Any, Any]:
        """Compute the second moment E[X²] by numerical integration."""

        def func(parametrization: Parametrization) -> Any:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            density = cast(Callable[[Parametrization, NumericArray], Number], self.density)
            return nquad(
                lambda x: (
                    x**2 * density(parametrization, x)
                    if self._support.contains(np.array([x]))
                    else 0
                ),
                [(float("-inf"), float("inf"))],
            )[0]

        return func

    @property
    def _var(self) -> ParametricFamilyCharacteristic[Any, Any]:
        """Compute the variance Var[X] = E[X²] - (E[X])²."""

        def func(parametrization: Parametrization) -> Any:
            parametrization = cast(ExponentialFamilyParametrization, parametrization)
            second_moment = cast(Callable[[Parametrization], Any], self._second_moment)
            mean = cast(Callable[[Parametrization], Any], self._mean)
            return second_moment(parametrization) - mean(parametrization) ** 2

        return func

    def posterior_hyperparameters(
        self, parametrizaiton: ExponentialConjugateHyperparameters, sample: list[Any]
    ) -> ExponentialConjugateHyperparameters:
        """
        Update the conjugate prior hyperparameters given observed data.

        For a conjugate prior with hyperparameters (ν₀, n₀), the posterior
        hyperparameters become:
            ν = ν₀ + Σ_{i} T(x_i)
            n = n₀ + N

        Args:
            parametrizaiton: Current conjugate hyperparameters.
            sample: List of observations (each can be scalar or array).

        Returns:
            ExponentialConjugateHyperparameters:
                Updated hyperparameters after incorporating the sample.
        """

        if hasattr(sample, "__iter__") and not isinstance(sample, str):
            posterior_effective_suff_stat_value = np.array(
                parametrizaiton.effective_suff_stat_value
            ) + np.sum(
                [self._sufficient(x) for x in sample],
                axis=0,
            )
            posterior_effective_sample_size = parametrizaiton.effective_sample_size + len(sample)
        else:
            posterior_effective_suff_stat_value = np.array(
                parametrizaiton.effective_suff_stat_value,
            ) + np.asarray(self._sufficient(sample))  # type: ignore[arg-type]
            posterior_effective_sample_size = parametrizaiton.effective_sample_size + 1

        return ExponentialConjugateHyperparameters(
            effective_suff_stat_value=posterior_effective_suff_stat_value,
            effective_sample_size=posterior_effective_sample_size,
        )

    @property
    def posterior_predictive(self) -> ParametricFamily:
        """
        Construct the posterior predictive distribution.

        For a conjugate prior, the posterior predictive density of a new observation x
        given hyperparameters (ν, n) is:
            p(x | ν, n) = h(x) * exp( A(ν) - A(ν + T(x)) )
        where A(·) is the log‑partition function of the conjugate prior family.

        Returns:
            ParametricFamily: A family with parametrization `ExponentialConjugateHyperparameters`
                and a `pdf` method implementing the posterior predictive density.
        """

        def conjugate_log_partition(
            parametrization: ExponentialConjugateHyperparameters,
        ) -> NumericArray:
            conjugate_value = self.conjugate_prior_family._log_partition(
                parametrization.transform_to_base_parametrization().theta
            )
            return np.exp(conjugate_value)

        def posterior_density(parametrization: Parametrization, x: NumericArray) -> Number:
            parametrization = cast(ExponentialConjugateHyperparameters, parametrization)
            return cast(
                np.float32,
                self._normalization(x)
                * conjugate_log_partition(parametrization)
                / conjugate_log_partition(
                    self.posterior_hyperparameters(parametrizaiton=parametrization, sample=[x])
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
