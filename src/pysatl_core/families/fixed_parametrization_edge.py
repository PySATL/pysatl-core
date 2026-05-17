from collections.abc import Callable

from pysatl_core.families.parametrizations import Parametrization


class EdgeWithFixedParametrization:
    def __init__(
        self,
        tail_name: str,
        transform_constraint: Callable[[Parametrization], bool],
        transform_function: Callable[[Parametrization], Parametrization],
    ):
        self._transform_function = transform_function
        self._transform_constraint = transform_constraint
        self._tail_name = tail_name

    def is_transoform_possible(self, parametrization: Parametrization) -> bool:
        return self._transform_constraint(parametrization)

    def transform_parametrization(self, parametrization: Parametrization) -> Parametrization:
        return self._transform_function(parametrization)

    @property
    def tail_name(self) -> str:
        return self._tail_name
