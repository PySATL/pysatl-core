from typing import cast
import pytest
import numpy as np

# from pysatl_core.distributions.computation import PDF
from pysatl_core.distributions.strategies import DefaultSamplingUnivariateStrategy
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families import (
    ExponentialFamily,
    ExponentialConjugateHyperparameters,
    ExponentialClassParametrization,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import UnivariateContinuous
import math


# TODO: WRITE TEEEEEEESTS
def test_exponential():
    pass
    # fam = ExponentialFamily(
    #     A=lambda parametrization: np.log(parametrization.theta[0]),
    #     T=lambda x: x,
    #     h=lambda _: 1,
    #     eta=lambda theta: -1 * theta,
    #     theta_from_eta=lambda eta: -1 * eta,
    #     param_space=[(0, float("+inf"))],
    #     support=[(0, float("+inf"))],
    #     natural_param_space=[(float("-inf"), 0)],
    #     distr_type=UnivariateContinuous,
    #     distr_parametrizations=["theta"],
    #     sampling_strategy=DefaultSamplingUnivariateStrategy(),
    # )

    # conjugate_fam = fam
    # conjugate_fam = fam.conjugate_prior_family
    # params = ExponentialClassParametrization(theta=np.array([2, -1]))
    # print(conjugate_fam._A(params))
    # ParametricFamilyRegister().register(conjugate_fam)
    # # print(
    # #     fam.posterior_hyperparameters(
    # #         ExponentialConjugateHyperparameters(alpha=10, beta=1), [12]
    # #     )
    # # )
    # gamma_family: ExponentialFamily = cast(
    #     ExponentialFamily, ParametricFamilyRegister().get("ExponentialFamily")
    # )
    # print(type(gamma_family))
    # # conjugate = gamma_family.conjugate_prior_family
    # # exponential = gamma_family(theta=np.array([2]), parametrization_name="theta")
    # exponential = gamma_family(theta=np.array([0, -1]), parametrization_name="theta")
    # pdf = exponential.computation_strategy.query_method("pdf", distr=exponential)
    # print(pdf(-1))
