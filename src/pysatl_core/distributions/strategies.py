from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from pysatl_core.distributions.computation import (
    AnalyticalComputation,
    ComputationMethod,
    FittedComputationMethod,
)
from pysatl_core.types import (
    GenericCharacteristicName,
)

from .registry import distribution_type_register
from .sampling import ArraySample, Sample

if TYPE_CHECKING:
    from .distribution import Distribution

type Method[In, Out] = AnalyticalComputation[In, Out] | FittedComputationMethod[In, Out]


class ComputationStrategy[In, Out](Protocol):
    enable_caching: bool

    def query_method(
        self, state: GenericCharacteristicName, distr: "Distribution"
    ) -> Method[In, Out]: ...


class DefaultComputationStrategy[In, Out]:
    """
    Дефолтный резолвер:

    1) Если распределение имеет аналитическую реализацию `state` — вернуть её.
    2) Иначе, если есть кэш — вернуть kэш.
    3) Иначе:
       - взять реестр для distribution type;
       - выбрать любую аналитическую характеристику распределения как источник;
       - найти путь (source -> ... -> state);
       - зафитить ПЕРВОЕ ребро пути (фиттер сам дёрнет нужные зависимости через стратегию).
    """

    def __init__(self, enable_caching: bool = False) -> None:
        self.enable_caching = enable_caching
        self._cache: dict[GenericCharacteristicName, FittedComputationMethod[In, Out]] = {}
        self._resolving: dict[int, set[GenericCharacteristicName]] = {}

    def _push_guard(self, distr: "Distribution", state: GenericCharacteristicName) -> None:
        key = id(distr)
        seen = self._resolving.setdefault(key, set())
        if state in seen:
            raise RuntimeError(
                f"Cycle detected while resolving '{state}'. "
                "Provide at least one analytical base characteristic in the distribution."
            )
        seen.add(state)

    def _pop_guard(self, distr: "Distribution", state: GenericCharacteristicName) -> None:
        key = id(distr)
        seen = self._resolving.get(key)
        if seen is not None:
            seen.discard(state)
            if not seen:
                self._resolving.pop(key, None)

    def query_method(
        self, state: GenericCharacteristicName, distr: "Distribution"
    ) -> Method[In, Out]:
        if state in distr.analytical_computations:
            return distr.analytical_computations[state]

        if self.enable_caching:
            cached = self._cache.get(state)
            if cached is not None:
                return cached

        if not distr.analytical_computations:
            raise RuntimeError(
                "Distribution provides no analytical computations to ground conversions."
            )

        reg = distribution_type_register().get(distr.distribution_type)

        self._push_guard(distr, state)
        try:
            chosen_edge: ComputationMethod[In, Out] | None = None

            for src in distr.analytical_computations:
                path = reg.find_path(src, state)
                if path:
                    chosen_edge = path[0]
                    break

            if chosen_edge is None:
                raise RuntimeError(
                    f"No conversion path from any analytical characteristic to '{state}'."
                )

            fitted = chosen_edge.fit(distr)

            if self.enable_caching:
                self._cache[state] = fitted

            return fitted
        finally:
            self._pop_guard(distr, state)


class SamplingStrategy(Protocol):
    def sample(self, n: int, **options: Any) -> Sample: ...


class DefaultSamplingStrategy(SamplingStrategy):
    def sample(self, n: int, **options: Any) -> ArraySample:
        d = options.get("d", 0)
        if d <= 0:
            raise ValueError("DefaultSamplingStrategy.sample requires positive 'd' option.")
        return ArraySample(np.zeros((n, d), dtype=np.float64))
