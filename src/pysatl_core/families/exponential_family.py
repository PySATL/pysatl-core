from __future__ import annotations
from collections.abc import Callable
from typing import Any, cast, TYPE_CHECKING
from scipy.integrate import nquad, quad
import numpy as np

from pysatl_core.distributions.fitters import _ppf_brentq_from_cdf
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import Parametrization, parametrization
from pysatl_core.types import (
    DistributionType,
    ParametrizationName,
)
from pysatl_core.distributions import (
    SamplingStrategy,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support

    type ParametrizedFunction = Callable[[Parametrization, Any], Any]
    type SupportArg = Callable[[Parametrization], Support | None] | None


PDF = "pdf"
CDF = "cdf"
PPF = "ppf"
CF = "char_func"
MEAN = "mean"
VAR = "var"
SKEW = "skewness"
KURT = "kurtosis"


class ExponentialFamilyParametrization(Parametrization):
    """
    Standard parametrization of Exponential Family.
    """

    theta: list[Callable[[float], float]]  # TODO: mb more clever


class ExponentialConjugateHyperparameters:
    def __init__(self, alpha: Any, beta: int):
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return f"alpha={self.alpha}, beta={self.beta}"


def doesAccept(x, support):
    if not hasattr(x, "__len__"):
        x = [x]

    def accept_1D(x, borders):
        left, right = borders
        if abs(x) == 0 and (abs(left) == 0 or abs(right) == 0):
            return False
        return left <= x <= right

    return all(accept_1D(x_i, border) for x_i, border in zip(x, support))


class SpacePredicate:
    def __init__(self, predicate: Callable[[Any], bool]):
        self._predicate = predicate

    def accepts(self, x: Any) -> bool:
        return self._predicate(x)


class SpacePredicateArray(SpacePredicate):
    def __init__(self, space: list[tuple[float, float]]):
        SpacePredicate.__init__(self, lambda x: doesAccept(x, space))
        self._space = space


class NaturalExponentialFamily(ParametricFamily):
    def __init__(
        self,
        *,
        log_partition: Callable[[ExponentialFamilyParametrization], float],
        sufficient_statistics: Callable[[Any], Any],
        normalization_constant: Callable[[Any], Any],
        support: SpacePredicate,
        parameter_space: SpacePredicate,
        sufficient_statistics_values: SpacePredicate,
        name: str = "NaturalExponentialFamily",
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

        distr_characteristics = {
            PDF: self.density,
            MEAN: self._mean,
            VAR: self._var,
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
        parametrization(family=self, name="theta")((ExponentialFamilyParametrization))

    @property
    def log_density(self) -> ParametrizedFunction:
        def log_density_func(
            parametrization: ExponentialFamilyParametrization, x: Any
        ) -> Any:
            if not self._support.accepts(x):
                return float("-inf")

            params = cast(ExponentialFamilyParametrization, parametrization)
            theta = params.parameters.get("theta")
            sufficient = self._sufficient(x)
            dot = np.dot(theta, sufficient)
            result = float(
                np.log(self._normalization(x))
                + dot
                + self._log_partition(parametrization)
            )
            return result

        return log_density_func

    @property
    def density(self) -> ParametrizedFunction:
        return lambda parametrization, x: np.exp(self.log_density(parametrization, x))

    @property
    def conjugate_prior_family(self):
        def conjugate_sufficient(theta: Any):
            if not self._parameter_space.accepts(theta):
                return [float("-inf"), float("-inf")]

            return [
                theta,
                self._log_partition(ExponentialFamilyParametrization(theta=[theta])),
            ]

        def conjugate_log_partition(parametrization: ExponentialFamilyParametrization):
            alpha = parametrization.theta[0]
            beta = parametrization.theta[1]

            def pdf(theta: Any):
                if not hasattr(theta, "__len__"):
                    theta = [theta]
                parametrization = ExponentialFamilyParametrization(
                    theta=theta,
                )
                return np.exp(
                    np.dot(theta, alpha) + beta * self._log_partition(parametrization)
                )[0]

            all_value = nquad(
                lambda x: pdf(x) if self._parameter_space.accepts(x) else 0,
                [(float("-inf"), float("+inf"))],
            )[0]
            return -np.log(all_value)

        # TODO: remove hardcoding - Done, all hardcoding is only on user's hands
        # 1. pr with prototype/draft - in progress
        # 2. write instruction about to add distributions as member of exponential family - not started
        # 3. parametrization's spaces (передавать в конструктор) - maybe impossible, discuss this with desiment on meeting

        def conjugate_sufficient_accepts(
            parametrization: ExponentialFamilyParametrization,
        ):
            parametrization = cast(parametrization, ExponentialFamilyParametrization)
            theta = parametrization.parameters.get("theta")
            xi = theta[:-1]
            nu = theta[-1]

            return self._sufficient_statistics_values(xi) and SpacePredicateArray(
                [(0, float("+inf"))]
            ).accepts(nu)

        return NaturalExponentialFamily(
            log_partition=conjugate_log_partition,
            sufficient_statistics=conjugate_sufficient,
            normalization_constant=lambda _: 1,
            support=self._parameter_space,
            sufficient_statistics_values=self._parameter_space,  # TODO: write convex hull for this
            parameter_space=SpacePredicate(conjugate_sufficient_accepts),
            sampling_strategy=self.sampling_strategy,
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
            support_by_parametrization=self.support_resolver,
        )

    @property
    def _mean(self) -> ParametrizedFunction:
        def mean_func(parametrization: Parametrization, x: Any) -> Any:
            dimension_size = 1
            if hasattr(x, "__len__"):
                dimension_size = len(x)
            return nquad(
                lambda x: (
                    np.dot(x, self.density(parametrization, x))
                    if self._support.accepts(x)
                    else 0
                ),
                [(float("-inf"), float("inf"))] * dimension_size,
            )[0]

        return mean_func

    @property
    def _second_moment(self) -> ParametrizedFunction:
        def func(parametrization: Parametrization, x: Any) -> Any:
            dimension_size = 1
            if hasattr(x, "__len__"):
                dimension_size = len(x)
            return nquad(
                lambda x: (
                    x**2 * self.density(parametrization, x)
                    if self._support.accepts(x)
                    else 0
                ),
                [(float("-inf"), float("inf"))] * dimension_size,
            )[0]

        return func

    @property
    def _var(self):
        def func(parametrization, x: Any):
            return (
                self._second_moment(parametrization, x)
                - self._mean(parametrization, x) ** 2
            )

        return func

    def posterior_hyperparameters(
        self, prior_hyper: ExponentialConjugateHyperparameters, sample
    ):
        alpha = prior_hyper.alpha
        beta = prior_hyper.beta

        alpha_post = None
        beta_post = None
        if hasattr(sample, "__iter__") and not isinstance(sample, str):
            alpha_post = np.sum([self._sufficient(x) for x in sample], axis=0)
            beta_post = len(sample)
        else:
            alpha_post = self._sufficient(sample)
            beta_post = 1

        return ExponentialConjugateHyperparameters(
            alpha=alpha + alpha_post, beta=beta + beta_post
        )


class ExponentialFamily(NaturalExponentialFamily):
    def __init__(
        self,
        *,
        log_partition: Callable[[ExponentialFamilyParametrization], float],
        sufficient_statistics: Callable[[Any], Any],
        normalization_constant: Callable[[Any], Any],
        parameter_from_natural_parameter: Callable[[Any], Any],
        support: SpacePredicate,
        parameter_space: SpacePredicate,
        sufficient_statistics_values: SpacePredicate,
        distr_type: DistributionType | Callable[[Parametrization], DistributionType],
        distr_parametrizations: list[ParametrizationName],
        sampling_strategy: SamplingStrategy,
        name: str = "ExponentialFamily",
        support_by_parametrization: SupportArg = None,
    ):
        def natural_log_partition(eta_parametrizaion: ExponentialFamilyParametrization):
            eta_parametrizaion = cast(
                ExponentialFamilyParametrization, eta_parametrizaion
            )
            eta = eta_parametrizaion.parameters.get("theta")
            theta = parameter_from_natural_parameter(eta)
            return log_partition(ExponentialFamilyParametrization(theta=[theta]))

        natural_sufficient_statistics_values = SpacePredicate(
            lambda eta: sufficient_statistics_values.accepts(
                parameter_from_natural_parameter(eta)
            )
        )
        natural_parameter_space = SpacePredicate(
            lambda eta: parameter_space.accepts(parameter_from_natural_parameter(eta)),
        )

        NaturalExponentialFamily.__init__(
            self,
            log_partition=natural_log_partition,
            sufficient_statistics=sufficient_statistics,
            normalization_constant=normalization_constant,
            support=support,
            parameter_space=natural_parameter_space,
            sufficient_statistics_values=natural_sufficient_statistics_values,
            name=name,
            distr_parametrizations=distr_parametrizations,
            distr_type=distr_type,
            sampling_strategy=sampling_strategy,
            support_by_parametrization=support_by_parametrization,
        )
