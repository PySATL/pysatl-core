from typing import cast

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose

from pysatl_core.distributions.strategies import DefaultSamplingUnivariateStrategy
from pysatl_core.distributions.support import ContinuousNDSupport, SupportByIntervals
from pysatl_core.families import (
    ContinuousExponentialClassFamily,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import Interval1D, UnivariateContinuous


def gamma_pdf(alpha: float, beta: float, x: float) -> float:
    return scipy.stats.gamma(a=alpha, scale=1 / beta).pdf(x).item()  # type: ignore[attr-defined]


@pytest.fixture(scope="function")
def conjugate_for_exponential() -> ContinuousExponentialClassFamily:
    def transform_function(x: list[float] | float) -> list[float] | float:
        if type(x) is list:
            return [-x[0]]
        return -x  # type: ignore[operator]

    support_neg = SupportByIntervals(ContinuousNDSupport(intervals=[Interval1D(-np.inf, 0)]))
    support_pos = SupportByIntervals(ContinuousNDSupport(intervals=[Interval1D(0, np.inf)]))
    fam = ContinuousExponentialClassFamily(
        log_partition=lambda parametrization: np.log(-parametrization),
        sufficient_statistics=lambda x: x,
        normalization_constant=lambda _: 1,
        parameter_space=support_neg,
        sufficient_statistics_values=support_pos,
        support=support_pos,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["theta"],
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
    )

    conjugate_fam = fam.conjugate_prior_family.transform(transform_function)
    ParametricFamilyRegister().register(conjugate_fam)
    return cast(
        ContinuousExponentialClassFamily,
        ParametricFamilyRegister().get("TransformedExponentialFamily"),
    )


@pytest.mark.parametrize("theta1", range(2, 5))
@pytest.mark.parametrize("theta2", range(2, 5))
def test_exponential_pdf(theta1, theta2, conjugate_for_exponential):
    gamma_family: ContinuousExponentialClassFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(theta=np.array([theta1, theta2]), parametrization_name="theta")
    pdf = exponential.computation_strategy.query_method("pdf", distr=exponential)

    x = [i / 10 for i in range(100)]

    assert_allclose([pdf(xx) for xx in x], [gamma_pdf(alpha, beta, xx) for xx in x], rtol=1e-6)


@pytest.mark.parametrize("theta1", range(2, 5))
@pytest.mark.parametrize("theta2", range(2, 5))
def test_exponential_mean(theta1, theta2, conjugate_for_exponential):
    gamma_family: ContinuousExponentialClassFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(theta=np.array([theta1, theta2]), parametrization_name="theta")
    mean = exponential.computation_strategy.query_method("mean", distr=exponential)
    assert np.isclose(mean(12), alpha / beta, rtol=1e-6)


@pytest.mark.parametrize("theta1", range(2, 5))
@pytest.mark.parametrize("theta2", range(2, 5))
def test_exponential_var(theta1, theta2, conjugate_for_exponential):
    gamma_family: ContinuousExponentialClassFamily = conjugate_for_exponential

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(theta=np.array([theta1, theta2]), parametrization_name="theta")
    var = exponential.computation_strategy.query_method("var", distr=exponential)
    assert np.isclose(var(12), alpha / beta**2, rtol=1e-6)
