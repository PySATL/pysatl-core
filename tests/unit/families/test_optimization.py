from typing import cast

import numpy as np
from numpy.testing import assert_allclose

from pysatl_core.families.configuration import configure_families_register
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.families.parametrizations import Parametrization
from pysatl_core.families.registry_graph import BinaryOperationType
from pysatl_core.types import CharacteristicName, FamilyName, Number, NumericArray


def test_parametrization_optimization():
    registry = configure_families_register()
    exponential_fam = registry.get(FamilyName.EXPONENTIAL)
    normal_fam = registry.get(FamilyName.NORMAL)
    uniform_fam = registry.get(FamilyName.CONTINUOUS_UNIFORM)

    def transform_function_exponent(head_param: Parametrization) -> Parametrization:
        head_param = head_param.transform_to_base_parametrization()
        tail_param_type = normal_fam.get_parametrization(normal_fam.base_parametrization_name)
        return tail_param_type(mu=0, sigma=1)  # type: ignore[call-arg]

    def transform_constraint_exponent(head_param: Parametrization) -> bool:
        head_param = head_param.transform_to_base_parametrization()
        return head_param.lambda_ == 1.0  # type: ignore[attr-defined]

    registry.register_parametrization_transformation(
        FamilyName.EXPONENTIAL,
        FamilyName.NORMAL,
        transform_constraint_exponent,
        transform_function_exponent,
    )

    def transform_function_normal(head_param: Parametrization) -> Parametrization:
        tail_param_type = uniform_fam.get_parametrization(uniform_fam.base_parametrization_name)
        return tail_param_type(lower_bound=0.0, upper_bound=1.0)  # type: ignore[call-arg]

    def transform_constraint_normal(head_param: Parametrization) -> bool:
        head_param = head_param.transform_to_base_parametrization()
        return head_param.mu == 0.0 and head_param.sigma == 1.0  # type: ignore[attr-defined]

    registry.register_parametrization_transformation(
        FamilyName.NORMAL,
        FamilyName.CONTINUOUS_UNIFORM,
        transform_constraint_normal,
        transform_function_normal,
    )

    registry._change_family_temperature(FamilyName.CONTINUOUS_UNIFORM, 129)

    exponential = exponential_fam(lambda_=1.0)
    exponential_parametrization = exponential.parametrization

    assert exponential.family_name == FamilyName.CONTINUOUS_UNIFORM
    assert exponential_parametrization.lower_bound == 0  # type: ignore[attr-defined]
    assert exponential_parametrization.upper_bound == 1  # type: ignore[attr-defined]

    pdf = exponential.query_method(CharacteristicName.PDF)

    assert pdf(0.5) == 1
    assert pdf(10) == 0
    registry._reset()


def test_density_optimization():
    registry = configure_families_register()
    registry.get(FamilyName.NORMAL)
    lognormal_fam = registry.get(FamilyName.LOGNORMAL)

    def transform_function(x: Number | NumericArray) -> Number | NumericArray:
        return np.exp(x)

    registry.register_density_transformation(
        FamilyName.NORMAL, FamilyName.LOGNORMAL, transform_function
    )

    def revert_function(x: Number | NumericArray) -> Number | NumericArray:
        return np.log(x)

    registry.register_density_transformation(
        FamilyName.LOGNORMAL, FamilyName.NORMAL, revert_function
    )

    registry._change_family_temperature(FamilyName.LOGNORMAL, 129)

    lognormal_result = registry.get_optimal_density(FamilyName.NORMAL)
    if lognormal_result is None:
        raise ValueError("The family is not in registry")

    lognormal_optimized, transformation = lognormal_result

    assert lognormal_optimized == lognormal_fam

    x = np.array(range(1, 10))
    assert_allclose(
        np.array([transform_function(xx) for xx in x]), np.array([transformation(xx) for xx in x])
    )


def test_transformations():
    registry = configure_families_register()
    normal_fam = registry.get(FamilyName.NORMAL)

    def transform_function(
        left_parametrization: Parametrization, right_parametrization: Parametrization
    ) -> Parametrization:
        param_type = normal_fam.get_parametrization(normal_fam.base_parametrization_name)
        return param_type(  # type: ignore[call-arg]
            mu=left_parametrization.mu + right_parametrization.mu,  # type: ignore[attr-defined]
            sigma=left_parametrization.sigma + right_parametrization.sigma,  # type: ignore[attr-defined]
        )

    registry.add_binary_transformation(
        FamilyName.NORMAL,
        FamilyName.NORMAL,
        FamilyName.NORMAL,
        BinaryOperationType.ADD,
        transform_function,
    )
    distribution_one = normal_fam(mu=1, sigma=1)
    distribution_two = normal_fam(mu=2, sigma=3)

    distribution_add = cast(ParametricFamilyDistribution, distribution_one + distribution_two)
    distribution_sub = distribution_one * distribution_two

    assert distribution_add.parametrization.mu == 3  # type: ignore[attr-defined]
    assert distribution_add.parametrization.sigma == 4  # type: ignore[attr-defined]

    assert isinstance(distribution_add, ParametricFamilyDistribution)
    assert not isinstance(distribution_sub, ParametricFamilyDistribution)
