import numpy as np
import pytest
import scipy
from typing import cast

# from pysatl_core.distributions.computation import PDF
from pysatl_core.distributions.strategies import DefaultSamplingUnivariateStrategy
from pysatl_core.families import (
    ExponentialFamily,
    ExponentialFamilyParametrization,
    SpacePredicateArray,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import UnivariateContinuous


# TODO: WRITE TEEEEEEESTS, MANY TESTS.
def test_exponential():
    # pass
    # fam = NaturalExponentialFamily(
    #     log_partition=lambda parametrization: np.log(-parametrization.theta[0]),
    #     sufficient_statistics=lambda x: x,
    #     normalization_constant=lambda _: 1,
    #     # param_space=SpacePredicateArray([(0, float("+inf"))]),
    #     support=SpacePredicateArray([(0, float("+inf"))]),
    #     parameter_space=SpacePredicateArray([(float("-inf"), 0)]),
    #     sufficient_statistics_values=SpacePredicateArray([(0, float("+inf"))]),
    #     distr_type=UnivariateContinuous,
    #     distr_parametrizations=["theta"],
    #     sampling_strategy=DefaultSamplingUnivariateStrategy(),
    # )

    def get_parameter_from_natural_parameter(
        eta_parametrization: ExponentialFamilyParametrization,
    ):
        if hasattr(eta_parametrization, "__len__"):
            if len(eta_parametrization) > 1:
                return list(-1 * np.array(eta_parametrization))
            eta_parametrization = eta_parametrization[0]
        return -eta_parametrization

    fam = ExponentialFamily(
        log_partition=lambda parametrization: np.log(parametrization.theta[0]),
        sufficient_statistics=lambda x: x,
        normalization_constant=lambda _: 1,
        parameter_from_natural_parameter=get_parameter_from_natural_parameter,
        parameter_space=SpacePredicateArray([(0, float("+inf"))]),
        sufficient_statistics_values=SpacePredicateArray([(0, float("+inf"))]),
        support=SpacePredicateArray([(0, float("+inf"))]),
        distr_type=UnivariateContinuous,
        distr_parametrizations=["theta"],
        sampling_strategy=DefaultSamplingUnivariateStrategy(),
    )

    conjugate_fam = fam
    conjugate_fam = fam.conjugate_prior_family
    ParametricFamilyRegister().register(conjugate_fam)
    # print(
    #     fam.posterior_hyperparameters(
    #         ExponentialConjugateHyperparameters(alpha=10, beta=1), [12]
    #     )
    # )
    gamma_family: ExponentialFamily = cast(
        ExponentialFamily, ParametricFamilyRegister().get("NaturalExponentialFamily")
    )
    print(type(gamma_family))
    # conjugate = gamma_family.conjugate_prior_family
    # exponential = gamma_family(theta=np.array([2]), parametrization_name="theta")
    theta1 = 4
    theta2 = 4

    alpha = theta2 + 1
    beta = theta1

    exponential = gamma_family(
        theta=np.array([theta1, theta2]), parametrization_name="theta"
    )
    pdf = exponential.computation_strategy.query_method("pdf", distr=exponential)

    def gamma_pdf(alpha: float, beta: float, x: float):
        return scipy.stats.gamma(a=alpha, scale=1 / beta).pdf(x).item()

    x = [i / 10 for i in range(-100, 100)]
    # print(pdf(-x))
    import matplotlib.pyplot as plt

    plt.plot(x, [pdf(-xx) for xx in x], label="conjugate")
    plt.plot(
        x,
        [gamma_pdf(alpha, beta, xx) for xx in x],
        label=f"gamma({alpha}, {beta}) test",
    )

    from scipy.integrate import quad

    print(quad(pdf, float("-inf"), float("inf")))
    # mean = exponential.computation_strategy.query_method("mean", distr=exponential)
    # print(mean(12))
    plt.legend()
    plt.savefig("a.png")
    # print(gamma_pdf(alpha, beta, x))
