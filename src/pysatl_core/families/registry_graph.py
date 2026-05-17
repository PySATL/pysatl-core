from __future__ import annotations

from collections.abc import Callable
from queue import Queue
from typing import TYPE_CHECKING, cast

from pysatl_core.families.parametrizations import Parametrization

if TYPE_CHECKING:
    from typing import ClassVar

    from pysatl_core.types import Number, NumericArray


class RegistryEdge:
    def __init__(self, head_name: str, tail_name: str):
        self._head_name = head_name
        self._tail_name = tail_name

    @property
    def tail_name(self) -> str:
        return self._tail_name

    @property
    def head_name(self) -> str:
        return self._head_name


class TransformatedParametrizationEdge(RegistryEdge):
    def __init__(
        self,
        head_name: str,
        tail_name: str,
        transform_constraint: Callable[[Parametrization], bool],
        transform_function: Callable[[Parametrization], Parametrization],
    ):
        self._transform_function = transform_function
        self._transform_constraint = transform_constraint

        RegistryEdge.__init__(self, head_name, tail_name)

    def is_transoform_possible(self, parametrization: Parametrization) -> bool:
        return self._transform_constraint(parametrization)

    def transform_parametrization(self, parametrization: Parametrization) -> Parametrization:
        return self._transform_function(parametrization)


class TransformatedDensityEdge(RegistryEdge):
    def __init__(
        self,
        head_name: str,
        tail_name: str,
        transform_function: Callable[[Number | NumericArray], Number | NumericArray],
    ):
        self._transform_function = transform_function
        RegistryEdge.__init__(self, head_name, tail_name)

    def transform_density(self, argument: Number | NumericArray) -> Number | NumericArray:
        return self._transform_function(argument)


class RegistryGraphTransformations:
    _instance: ClassVar[RegistryGraphTransformations | None] = None
    _registered_families_temperature: dict[str, int]
    _registered_parametrzation_transformations: dict[str, list[TransformatedParametrizationEdge]]
    _registered_transformations: dict[str, list[TransformatedDensityEdge]]

    def __new__(cls) -> RegistryGraphTransformations:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registered_parametrzation_transformations = {}
            cls._registered_families_temperature = {}
            cls._registered_transformations = {}
        return cls._instance

    @classmethod
    def _run_bfs(
        cls,
        start_vertex: str,
        edge_representation: dict[str, list[RegistryEdge]],
        edge_constraint: Callable[[RegistryEdge], bool],
        visit_edge: Callable[[RegistryEdge], None],
        path_callback: Callable[[RegistryEdge], None],
    ) -> None:
        self = cls()

        visited_vertexes = set()
        previous_in_path: dict[str, tuple[str, RegistryEdge] | None] = {}

        visited_vertexes.add(start_vertex)

        previous_in_path[start_vertex] = None
        queue: Queue[str] = Queue()
        queue.put(start_vertex)

        while not queue.empty():
            current_vertex = queue.get()

            for next_vertex_edge in edge_representation.get(current_vertex, []):
                next_name = next_vertex_edge.tail_name
                if next_name not in visited_vertexes and edge_constraint(next_vertex_edge):
                    visited_vertexes.add(next_name)
                    previous_in_path[next_name] = current_vertex, next_vertex_edge
                    queue.put(next_name)
                    visit_edge(next_vertex_edge)

        best_choice = start_vertex
        for visited_vertex in visited_vertexes:
            if (
                self._registered_families_temperature[visited_vertex]
                > self._registered_families_temperature[best_choice]
            ):
                best_choice = visited_vertex

        path = []
        while previous_in_path[best_choice] is not None:
            previous = cast(tuple[str, RegistryEdge], previous_in_path[best_choice])
            path.append(previous[1])
            best_choice = previous[0]

        path = path[::-1]
        for edge in path:
            path_callback(edge)

    @classmethod
    def get_optimal_parametrization(
        cls, current_family: str, current_parametrization: Parametrization
    ) -> tuple[str, Parametrization]:
        self = cls()

        best_choice = current_family
        result_parametrization = current_parametrization

        parametrizations = {}
        parametrizations[current_family] = current_parametrization

        def path_callback(edge: RegistryEdge) -> None:
            nonlocal best_choice
            nonlocal result_parametrization

            edge = cast(TransformatedParametrizationEdge, edge)
            new_parametrization = edge.transform_parametrization(result_parametrization)
            result_parametrization = new_parametrization
            best_choice = edge.tail_name

        def visit_edge(edge: RegistryEdge) -> None:
            nonlocal parametrizations
            edge = cast(TransformatedParametrizationEdge, edge)

            head = edge.head_name
            tail = edge.tail_name
            parametrizations[tail] = edge.transform_parametrization(parametrizations[head])

        def edge_constraint(edge: RegistryEdge) -> bool:
            nonlocal parametrizations
            edge = cast(TransformatedParametrizationEdge, edge)
            return edge.is_transoform_possible(parametrizations[edge.head_name])

        RegistryGraphTransformations._run_bfs(
            start_vertex=current_family,
            edge_representation=self._registered_parametrzation_transformations,  # type: ignore[arg-type]
            edge_constraint=edge_constraint,
            visit_edge=visit_edge,
            path_callback=path_callback,
        )

        return best_choice, result_parametrization

    @classmethod
    def get_optimal_transoformation(
        cls, current_family: str
    ) -> tuple[str, Callable[[Number | NumericArray], Number | NumericArray]]:
        self = cls()

        def collected_function(x: Number | NumericArray) -> NumericArray | Number:
            return x

        best_choice = current_family

        def edge_constraint(edge: RegistryEdge) -> bool:
            return True

        def visit_edge(edge: RegistryEdge) -> None:
            pass

        def path_callback(edge: RegistryEdge) -> None:
            nonlocal collected_function
            nonlocal best_choice

            edge = cast(TransformatedDensityEdge, edge)
            _temp = collected_function

            def collected_function(x: Number | NumericArray) -> Number | NumericArray:
                return edge.transform_density(_temp(x))

            best_choice = edge.tail_name

        RegistryGraphTransformations._run_bfs(
            current_family,
            edge_representation=self._registered_transformations,  # type: ignore[arg-type]
            edge_constraint=edge_constraint,
            visit_edge=visit_edge,
            path_callback=path_callback,
        )

        return best_choice, collected_function

    @classmethod
    def register_parametrization_transformation(
        cls,
        head_name: str,
        tail_name: str,
        transform_constraint: Callable[[Parametrization], bool],
        transform_function: Callable[[Parametrization], Parametrization],
    ) -> None:
        self = cls()
        self._registered_parametrzation_transformations.setdefault(head_name, []).append(
            TransformatedParametrizationEdge(
                head_name, tail_name, transform_constraint, transform_function
            )
        )

    @classmethod
    def register_density_transformation(
        cls,
        head_name: str,
        tail_name: str,
        transform_function: Callable[[Number | NumericArray], Number | NumericArray],
    ) -> None:
        self = cls()
        self._registered_transformations.setdefault(head_name, []).append(
            TransformatedDensityEdge(head_name, tail_name, transform_function)
        )

    @classmethod
    def register_family_temperature(cls, family_name: str, temperature: int) -> None:
        self = cls()
        self._registered_families_temperature[family_name] = temperature
