from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any, cast
from scipy.integrate import nquad, quad
import numpy as np

from pysatl_core.distributions.fitters import _ppf_brentq_from_cdf
from pysatl_core.families.parametric_family import (
    ParametricFamily,
)
from pysatl_core.families.parametrizations import Parametrization, parametrization
from pysatl_core.types import (
    DistributionType,
    ParametrizationName,
)
from pysatl_core.distributions import (
    SamplingStrategy,
)

PDF = "pdf"
CDF = "cdf"
PPF = "ppf"
CF = "char_func"
MEAN = "mean"
VAR = "var"
SKEW = "skewness"
KURT = "kurtosis"


class ExponentialClassParametrization(Parametrization):
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


def accepts(x, support):
    if not hasattr(x, "__len__"):
        x = [x]

    def accept_1D(x, borders):
        left, right = borders
        return left <= x <= right

    return all(accept_1D(x_i, border) for x_i, border in zip(x, support))


class ExponentialFamily(ParametricFamily):
    def __init__(
        self,
        *,
        A: Callable[[ExponentialClassParametrization], float],
        T: Callable[[Any], Any],
        h: Callable[[Any], float],
        eta: Callable[[Any], Any],
        support: list[tuple[float, float]],
        param_space: list[tuple[float, float]],
        natural_param_space: list[tuple[float, float]],
        name: str = "ExponentialFamily",
        theta_from_eta: Callable[[Any], Any] = None,
        distr_type: DistributionType | Callable[[Parametrization], DistributionType],
        distr_parametrizations: list[ParametrizationName],
        sampling_strategy: SamplingStrategy,
        support_by_parametrization: SupportArg = None,
    ):

        self._A = A
        self._T = T
        self._h = h

        self._eta = eta if eta is not None else (lambda th: th)
        self._theta_from_eta = theta_from_eta
        self._natural_param_space = natural_param_space
        self._param_space = param_space
        self._support = support

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
        parametrization(family=self, name="theta")((ExponentialClassParametrization))

    @property
    def log_density(self) -> ParametrizedFunction:
        def log_density_func(
            parametrization: ExponentialClassParametrization, x: Any
        ) -> Any:
            if not accepts(x, self._support):
                return float("-inf")

            params = cast(ExponentialClassParametrization, parametrization)
            theta = params.parameters.get("theta")
            eta = self._eta(theta)
            sufficient = self._T(x)
            dot = np.dot(eta, sufficient)

            result = float(np.log(self._h(x)) + dot + self._A(parametrization))
            return result

        return log_density_func

    @property
    def density(self) -> ParametrizedFunction:
        return lambda parametrization, x: np.exp(self.log_density(parametrization, x))

    @property
    def conjugate_prior_family(self):
        def conjugate_sufficient(eta: Any):
            theta = [self._theta_from_eta(eta)]
            if not accepts(theta, self._param_space):
                return [float("-inf"), float("-inf")]

            return [eta, self._A(ExponentialClassParametrization(theta=theta))]

        def conjugate_log_partition(parametrization: ExponentialClassParametrization):
            alpha = parametrization.theta[0]
            beta = parametrization.theta[1]

            def pdf(eta: Any):
                theta = self._theta_from_eta(eta)
                if not hasattr(theta, "__len__"):
                    theta = [theta]
                parametrization = ExponentialClassParametrization(
                    theta=theta,
                )
                return np.exp(np.dot(eta, alpha) + beta * self._A(parametrization))

            all_value = nquad(pdf, self._natural_param_space)[0]
            return -np.log(all_value)

        if self._theta_from_eta is None:
            raise RuntimeError("Theta from eta wasn't specified")

        return ExponentialFamily(
            A=conjugate_log_partition,
            T=conjugate_sufficient,
            h=lambda _: 1,
            eta=lambda x: x,
            theta_from_eta=lambda eta: eta,
            support=self._natural_param_space,
            natural_param_space=[(float("-inf"), float("inf"))] * 2,
            param_space=[(float("-inf"), float("inf"))] * 2,
            sampling_strategy=self.sampling_strategy,
            distr_type=self._distr_type,
            distr_parametrizations=self.parametrization_names,
            support_by_parametrization=self.support_resolver,
        )

    @property
    def _mean(self) -> ParametrizedFunction:
        def mean_func(parametrization: Parametrization, x: Any) -> Any:
            if hasattr(x, "__len__"):
                dimension_size = len(x)
            else:
                dimension_size = 1
            print(dimension_size)
            return nquad(
                lambda x: np.dot(x, self.density(parametrization, x)),
                [(float("-inf"), float("inf"))] * dimension_size,
            )[0]

        return mean_func

    @property
    def _second_moment(self) -> ParametrizedFunction:
        def func(parametrization: Parametrization, x: Any) -> Any:
            if hasattr(x, "__len__"):
                dimension_size = len(x)
            else:
                dimension_size = 1
            return nquad(
                lambda x: x**2 * self.density(parametrization, x),
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
            alpha_post = np.sum([self._T(x) for x in sample], axis=0)
            beta_post = len(sample)
        else:
            alpha_post = self.T(sample)
            beta_post = 1

        return ExponentialConjugateHyperparameters(
            alpha=alpha + alpha_post, beta=beta + beta_post
        )
