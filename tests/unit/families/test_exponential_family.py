from typing import Any, cast

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose

from pysatl_core.distributions.strategies import DefaultSamplingUnivariateStrategy
from pysatl_core.families import (
    ExponentialFamily,
    ExponentialFamilyParametrization,
    SpacePredicateArray,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import UnivariateContinuous


def gamma_pdf(alpha: float, beta: float, x: float) -> float:
    return scipy.stats.gamma(a=alpha, scale=1 / beta).pdf(x).item()


@pytest.fixture(scope="function")
def conjugate_for_exponential() -> ExponentialFamily:
    def get_parameter_from_natural_parameter(
        eta_parametrization: ExponentialFamilyParametrization,
    ):
        if hasattr(eta_parametrization, "__len__"):
            if len(eta_parametrization) > 1:
                return list(-1 * np.array(eta_parametrization))
            eta_parametrization = eta_parametrization[0]
        return -eta_parametrization

    def natural_parameter(
        theta_parametrization: Any,
    ) -> Any:
        if type(theta_parametrization) is ExponentialFamilyParametrization:
            theta_parametrization = cast(
                ExponentialFamilyParametrization, theta_parametrization
            )
            eta = -theta_parametrization.theta
            return ExponentialFamilyParametrization(theta=eta)

        return -1 * theta_parametrization

    def transform_function(x: list[Any]) -> list[Any]:
        if type(x) is not list:
            return -x
        return [-x[0]]

    fam = ExponentialFamily(
        log_partition=lambda parametrization: np.log(parametrization.theta[0]),
        sufficient_statistics=lambda x: x,
        normalization_constant=lambda _: 1,
        parameter_from_natural_parameter=get_parameter_from_natural_parameter,
        natural_parameter=natural_parameter,
        parameter_space=SpacePredicateArray([(0, float("+inf"))]),
        sufficient_statistics_values=SpacePredicateArray([(0, float("+inf"))]),
        support=SpacePredicateArray([(0, float("+inf"))]),
        distr_type=UnivariateContinuous,
        distr_parametrizations=["theta"],
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
    )

    conjugate_fam = fam.conjugate_prior_family.transform(transform_function)
    ParametricFamilyRegister().register(conjugate_fam)
    return cast(
        ExponentialFamily,
        ParametricFamilyRegister().get("TransformedExponentialFamily"),
    )


@pytest.mark.parametrize("theta1", range(2, 5))
@pytest.mark.parametrize("theta2", range(2, 5))
def test_exponential_pdf(theta1, theta2, conjugate_for_exponential):
    gamma_family: ExponentialFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(
        theta=np.array([theta1, theta2]), parametrization_name="theta"
    )
    pdf = exponential.computation_strategy.query_method("pdf", distr=exponential)

    x = [i / 10 for i in range(100)]

    assert_allclose(
        [pdf(xx) for xx in x], [gamma_pdf(alpha, beta, xx) for xx in x], rtol=1e-6
    )


@pytest.mark.parametrize("theta1", range(2, 5))
@pytest.mark.parametrize("theta2", range(2, 5))
def test_exponential_mean(theta1, theta2, conjugate_for_exponential):
    gamma_family: ExponentialFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(
        theta=np.array([theta1, theta2]), parametrization_name="theta"
    )
    mean = exponential.computation_strategy.query_method("mean", distr=exponential)
    assert np.isclose(mean(12), alpha / beta, rtol=1e-6)


@pytest.mark.parametrize("theta1", range(2, 5))
@pytest.mark.parametrize("theta2", range(2, 5))
def test_exponential_var(theta1, theta2, conjugate_for_exponential):
    gamma_family: ExponentialFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(
        theta=np.array([theta1, theta2]), parametrization_name="theta"
    )
    var = exponential.computation_strategy.query_method("var", distr=exponential)
    assert np.isclose(var(12), alpha / beta**2, rtol=1e-6)
