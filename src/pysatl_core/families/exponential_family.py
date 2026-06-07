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

    This parametrization uses the natural parameter vector ``theta``. In this
    module the density is written as

    ``f(x | theta) = h(x) exp(theta^T T(x) + B(theta))``,

    where ``B(theta)`` is the log-normalizing term supplied as
    ``log_partition``.

    Attributes
    ----------
    theta : NumericArray
        Natural parameter vector.
    """

    theta: NumericArray

    def transform_to_base_parametrization(self) -> ExponentialFamilyParametrization:
        """Return the base parametrization (identity transform for canonical form)."""
        return self


@dataclass
class ExponentialConjugateHyperparameters(Parametrization):
    """
    Hyperparameters for the conjugate prior of an exponential family.

    For this module's sign convention, the conjugate prior over ``theta`` is
    proportional to

    ``exp(nu_0^T theta + n_0 B(theta))``,

    where ``B(theta)`` is the base family's log-normalizing term.

    Attributes
    ----------
    effective_suff_stat_value : NumericArray
        Pseudo-sufficient statistic value ``nu_0``.
    effective_sample_size : Number
        Pseudo-sample size ``n_0``.
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

    The density is given by

    ``f(x | theta) = h(x) exp(theta^T T(x) + B(theta))``.

    Here ``theta`` is the natural parameter, ``T(x)`` is the sufficient
    statistic, ``h(x)`` is the base measure supplied as
    ``normalization_constant``, and ``B(theta)`` is the log-normalizing term
    supplied as ``log_partition``. With the usual convention
    ``h(x) exp(theta^T T(x) - A(theta))``, this means ``B(theta) = -A(theta)``.

    The family provides canonical parametrization by ``theta``, conjugate prior
    construction, posterior hyperparameter updates, posterior predictive
    densities, and monotone differentiable transformations.
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

        Parameters
        ----------
        log_partition : Callable[[NumericArray], NumericArray]
            Function ``B(theta)`` used in
            ``f(x | theta) = h(x) exp(theta^T T(x) + B(theta))``.
        sufficient_statistics : Callable[[NumericArray], NumericArray]
            Function ``T(x)`` returning the sufficient statistic.
        normalization_constant : Callable[[NumericArray], Number]
            Function ``h(x)`` returning the base measure.
        support : Support
            Support of the observation variable ``x``.
        parameter_space : Support
            Support of the natural parameter ``theta``.
        sufficient_statistics_values : Support
            Support of possible sufficient-statistic values.
        name : str
            Family name.
        distr_type : DistributionType or Callable[[Parametrization], DistributionType]
            Distribution type, or a resolver from base parametrization to type.
        distr_parametrizations : list[ParametrizationName]
            Parametrization names supported by this family.
        distr_characteristics : CharacteristicsMap, optional
            Additional analytical characteristics to register.
        support_by_parametrization : Callable[[Parametrization], Support | None], optional
            Resolver for the distribution support of a concrete parametrization.
        base_score : Callable[[Parametrization, NumericArray], NumericArray], optional
            Score function in the base parametrization.
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
        distr_characteristics = dict(distr_characteristics or {})
        merged_characteristics = dict(family_characteristics)
        merged_characteristics.update(distr_characteristics)

        ParametricFamily.__init__(
            self,
            name=name,
            distr_type=distr_type,
            distr_parametrizations=distr_parametrizations,
            distr_characteristics=merged_characteristics,
            support_by_parametrization=support_by_parametrization,
            base_score=base_score,
        )

        family = self

        @parametrization(family=family, name="theta")
        class ThetaParametrization(ExponentialFamilyParametrization):
            @constraint(description="theta belongs to parameter_space")
            def check_theta_in_parameter_space(self) -> bool:
                theta = np.atleast_1d(np.asarray(self.theta, dtype=float))
                return family._parameter_space_contains(theta)

    def _parameter_space_contains(self, theta: NumericArray) -> bool:
        is_contains = self._parameter_space.contains(theta)
        if isinstance(is_contains, np.ndarray):
            return bool(np.all(is_contains))
        return bool(is_contains)

    @property
    def log_density(self) -> ParametricFamilyCharacteristic[NumericArray, Number]:
        """
        Return the log-density characteristic.

        The returned callable evaluates

        ``log h(x) + theta^T T(x) + B(theta)``.

        Points outside the observation support return ``-np.inf``.

        Returns
        -------
        ParametricFamilyCharacteristic[NumericArray, Number]
            Callable accepting a parametrization and observation.
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
        Return the density characteristic.

        Returns
        -------
        ParametricFamilyCharacteristic[NumericArray, Number]
            Callable evaluating ``exp(log_density(parametrization, x))``.
        """
        log_density = cast(Callable[[Parametrization, NumericArray], Number], self.log_density)

        def density_func(parametrization: Parametrization, x: NumericArray) -> Number:
            return cast(Number, np.exp(log_density(parametrization, x)))

        return density_func

    @property
    def conjugate_prior_family(self) -> ContinuousExponentialClassFamily:
        """
        Build the conjugate prior family for this exponential family.

        The conjugate prior is an exponential family over ``theta`` with
        sufficient statistic ``[theta, B(theta)]`` and base measure ``1``.
        Its natural parameter is ``[nu, n]``, matching
        :class:`ExponentialConjugateHyperparameters`.

        Returns
        -------
        ContinuousExponentialClassFamily
            Conjugate prior family for the natural parameter.
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

        ``transform_function`` is interpreted as the inverse map ``x = g(y)``.
        The transformed density is

        ``f_Y(y | theta) = h(g(y)) exp(theta^T T(g(y)) + B(theta)) |J_g(y)|``.

        Parameters
        ----------
        transform_function : Callable[[NumericArray], NumericArray]
            Invertible differentiable function ``g`` mapping transformed
            observations back to the original observation support.

        Returns
        -------
        ContinuousExponentialClassFamily
            Family for the transformed random variable.
        """

        def calculate_jacobian(x: NumericArray) -> NumericArray:
            if not isinstance(x, Iterable):
                x = np.array([x], dtype=float)
            else:
                x = np.atleast_1d(np.asarray(x, dtype=float))

            return np.abs(det(jacobian(transform_function, x).df))

        def new_support(x: NumericArray) -> bool:
            return bool(self._support.contains(np.asarray(transform_function(x))))

        def new_sufficient(x: NumericArray) -> NumericArray:
            return self._sufficient(transform_function(x))

        def new_normalization(x: NumericArray) -> Number:
            return cast(
                np.float64,
                self._normalization(transform_function(x)) * calculate_jacobian(x),
            )

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
        self,
        parametrization: ExponentialConjugateHyperparameters,
        sample: list[Any] | Any,
    ) -> ExponentialConjugateHyperparameters:
        """
        Update the conjugate prior hyperparameters given observed data.

        For prior hyperparameters ``(nu_0, n_0)`` and observations ``x_i``,
        the posterior hyperparameters are

        ``nu = nu_0 + sum_i T(x_i)`` and ``n = n_0 + N``.

        Parameters
        ----------
        parametrization : ExponentialConjugateHyperparameters
            Current conjugate hyperparameters.
        sample : list[Any] or Any
            Observations used for the update. A non-string iterable is treated
            as a sample; any other value is treated as one observation.

        Returns
        -------
        ExponentialConjugateHyperparameters
            Updated hyperparameters after incorporating ``sample``.
        """
        if hasattr(sample, "__iter__") and not isinstance(sample, str):
            posterior_effective_suff_stat_value = np.array(
                parametrization.effective_suff_stat_value
            ) + np.sum(
                [self._sufficient(x) for x in sample],
                axis=0,
            )
            posterior_effective_sample_size = parametrization.effective_sample_size + len(sample)
        else:
            posterior_effective_suff_stat_value = np.array(
                parametrization.effective_suff_stat_value,
            ) + np.asarray(self._sufficient(sample))  # type: ignore[arg-type]
            posterior_effective_sample_size = parametrization.effective_sample_size + 1

        return ExponentialConjugateHyperparameters(
            effective_suff_stat_value=posterior_effective_suff_stat_value,
            effective_sample_size=posterior_effective_sample_size,
        )

    @property
    def posterior_predictive(self) -> ParametricFamily:
        """
        Construct the posterior predictive distribution.

        For conjugate hyperparameters ``(nu, n)``, the predictive density of a
        new observation ``x`` is

        ``p(x | nu, n) = h(x) Z_c(nu, n) / Z_c(nu + T(x), n + 1)``,

        where ``Z_c`` is the conjugate-prior normalizing integral. The density
        is zero outside the original observation support.

        Returns
        -------
        ParametricFamily
            Family with ``ExponentialConjugateHyperparameters`` parametrization
            and a ``pdf`` characteristic for the posterior predictive density.
        """

        def conjugate_log_partition(
            parametrization: ExponentialConjugateHyperparameters,
        ) -> Number:
            conjugate_value = self.conjugate_prior_family._log_partition(
                parametrization.transform_to_base_parametrization().theta
            )
            return cast(Number, np.exp(conjugate_value).item())

        def posterior_density(parametrization: Parametrization, x: NumericArray) -> Number:
            parametrization = cast(ExponentialConjugateHyperparameters, parametrization)
            if not self._support.contains(np.asarray(x)):
                return cast(np.float32, 0.0)
            return cast(
                np.float32,
                self._normalization(x)
                * conjugate_log_partition(parametrization)
                / conjugate_log_partition(
                    self.posterior_hyperparameters(parametrization=parametrization, sample=[x])
                ),
            )

        family = ParametricFamily(
            name=f"PosteriorPredictive{self.name}",
            distr_type=UnivariateContinuous,
            distr_characteristics={CharacteristicName.PDF: posterior_density},
            distr_parametrizations=["posterior"],
            support_by_parametrization=lambda _: self._support,
        )
        parametrization(family=family, name="posterior")(ExponentialConjugateHyperparameters)
        return family
