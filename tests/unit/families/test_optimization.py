from pysatl_core.families.configuration import configure_families_register
from pysatl_core.families.parametrizations import Parametrization
from pysatl_core.types import CharacteristicName, FamilyName


def test_optimization():
    registry = configure_families_register()
    exponential_fam = registry.get(FamilyName.EXPONENTIAL)
    normal_fam = registry.get(FamilyName.NORMAL)
    uniform_fam = registry.get(FamilyName.CONTINUOUS_UNIFORM)

    def transform_function_exponent(head_param: Parametrization) -> Parametrization:
        tail_param_type = normal_fam.get_parametrization(normal_fam.base_parametrization_name)
        return tail_param_type(mu=0, sigma=1)  # type: ignore[call-arg]

    def transform_constraint_exponent(head_param: Parametrization) -> bool:
        return head_param.lambda_ == 1.0  # type: ignore[attr-defined]

    registry.add_optimization_edge(
        FamilyName.EXPONENTIAL,
        FamilyName.NORMAL,
        transform_constraint_exponent,
        transform_function_exponent,
    )

    def transform_function_normal(head_param: Parametrization) -> Parametrization:
        tail_param_type = uniform_fam.get_parametrization(uniform_fam.base_parametrization_name)
        return tail_param_type(lower_bound=0.0, upper_bound=1.0)  # type: ignore[call-arg]

    def transform_constraint_normal(head_param: Parametrization) -> bool:
        return head_param.mu == 0.0 and head_param.sigma == 1.0  # type: ignore[attr-defined]

    registry.add_optimization_edge(
        FamilyName.NORMAL,
        FamilyName.CONTINUOUS_UNIFORM,
        transform_constraint_normal,
        transform_function_normal,
    )

    exponential = exponential_fam(lambda_=1.0)
    exponential_parametrization = exponential.parametrization

    assert exponential.family_name == FamilyName.CONTINUOUS_UNIFORM
    assert exponential_parametrization.lower_bound == 0  # type: ignore[attr-defined]
    assert exponential_parametrization.upper_bound == 1  # type: ignore[attr-defined]

    pdf = exponential.query_method(CharacteristicName.PDF)

    assert pdf(0.5) == 1
    assert pdf(10) == 0
