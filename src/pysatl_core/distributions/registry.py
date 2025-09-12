from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pysatl_core.distributions.computation import (
    ComputationMethod,
    FittedComputationMethod,
)
from pysatl_core.types import (
    DistributionType,
    EuclidianDistributionType,
    GenericCharacteristicName,
    Kind,
)

ScalarFunc = Callable[[float], float]

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution

DEFAULT_COMPUTATION_KEY: str = "PySATL_default_computation"


class GraphInvariantError(RuntimeError):
    """Invariant violation in GenericCharacteristicRegister."""


@dataclass(slots=True, frozen=True)
class GenericCharacteristicRegister:
    """
    Ориентированный граф характеристик для фиксированного DistributionType.

    Вершины: GenericCharacteristicName.
    Рёбра: унарные ComputationMethod: один source -> один target.

    Хранение рёбер:
      adjacency[src][dst] = dict[method_name, ComputationMethod]
      Для удобства предусмотрен ключ дефолтного метода: DEFAULT_COMPUTATION_KEY.

    Инварианты:
      1) Существует >= 1 definitive-вершина.
      2) Подграф, индуцированный definitive-вершинами, сильно связен.
      3) Любая вершина, не отмеченная как definitive (т.е. "indefinitive"),
         достижима из хотя бы одной definitive (нет «оторванных»).
      4) Ни из одной indefinitive-вершины нельзя добраться до definitive
         (никакого обратного пути в definitive-компоненту).
    """

    distribution_type: DistributionType

    __adj: dict[
        GenericCharacteristicName,
        dict[GenericCharacteristicName, dict[str, ComputationMethod[Any, Any]]],
    ] = field(default_factory=dict, repr=False)
    __definitive: set[GenericCharacteristicName] = field(default_factory=set, repr=False)

    def _ensure_vertex(self, v: GenericCharacteristicName) -> None:
        self.__adj.setdefault(v, {})

    def _pick_method(
        self, methods: dict[str, ComputationMethod[Any, Any]]
    ) -> ComputationMethod[Any, Any]:
        if DEFAULT_COMPUTATION_KEY in methods:
            return methods[DEFAULT_COMPUTATION_KEY]
        name = next(iter(sorted(methods.keys())))
        return methods[name]

    def _add_edge_unary(
        self,
        src: GenericCharacteristicName,
        dst: GenericCharacteristicName,
        method: ComputationMethod[Any, Any],
        name: str,
    ) -> None:
        sources = getattr(method, "sources", None)
        target = getattr(method, "target", None)
        if not isinstance(sources, list | tuple) or target is None:
            raise TypeError(
                "ComputationMethod must have attributes 'sources: list[...]' and 'target'."
            )
        if len(sources) != 1:
            raise GraphInvariantError(
                "Only unary methods are supported for edges (1 source -> 1 target). "
                "Consider hypergraph if needed."
            )
        if sources[0] != src or target != dst:
            raise GraphInvariantError(
                f"Method endpoints mismatch: expected {src} -> {dst}, got {sources[0]} -> {target}."
            )

        key = name
        self._ensure_vertex(src)
        self._ensure_vertex(dst)
        self.__adj[src].setdefault(dst, {})[key] = method

    def register_definitive(self, name: GenericCharacteristicName) -> None:
        """
        Отметить вершину как definitive без рёбер.
        Сильная связность на единственной вершине выполняется тривиально.

        Обычно используется при добавлении первой вершины типа распределения.
        Повторные вызовы допустимы (если это разные вершины), но любые рёбра,
        которые нарушат инварианты, приведут к исключению при валидации.
        """
        self.__definitive.add(name)
        self._ensure_vertex(name)
        self._validate_invariants()

    def add_bidirectional_definitive(
        self,
        a_to_b: ComputationMethod[Any, Any],
        b_to_a: ComputationMethod[Any, Any],
        name_ab: str,
        name_ba: str,
    ) -> None:
        """
        Добавить двунаправленную связь между двумя definitive-узлами.
        Если узлы ещё не definitive — пометить их таковыми.
        Допускается указать имена методов (ключи в словаре), по умолчанию — DEFAULT_COMPUTATION_KEY.

        Важно: если добавление нарушит сильную связность подграфа definitive, будет исключение.
        """
        a = a_to_b.sources[0]
        b = a_to_b.target
        if b_to_a.sources[0] != b or b_to_a.target != a:
            raise GraphInvariantError(
                "Inverse methods must link the same pair of definitive nodes "
                "in opposite directions."
            )

        self.__definitive.add(a)
        self.__definitive.add(b)
        self._add_edge_unary(a, b, a_to_b, name=name_ab)
        self._add_edge_unary(b, a, b_to_a, name=name_ba)
        self._validate_invariants()

    def add_conversion(self, method: ComputationMethod[Any, Any], *, name: str) -> None:
        """
        Добавить произвольное унарное ребро (source -> target) с именем метода.
        Разрешены связи:
          - definitive -> definitive,
          - definitive -> indefinitive,
          - indefinitive -> indefinitive.
        Запрещены связи по факту инвариантов:
          - любой путь, ведущий из indefinitive к какой-либо definitive.

        Примечание: сам по себе факт добавления ребра допустим, но если из-за него
        нарушится сильная связность definitive-подграфа или появится путь
        из indefinitive в definitive — будет исключение.
        """
        src = method.sources[0]
        dst = method.target
        self._add_edge_unary(src, dst, method, name=name)
        self._validate_invariants()

    def is_definitive(self, name: GenericCharacteristicName) -> bool:
        return name in self.__definitive

    def definitive_nodes(self) -> frozenset[GenericCharacteristicName]:
        return frozenset(self.__definitive)

    def all_nodes(self) -> frozenset[GenericCharacteristicName]:
        verts = set(self.__adj.keys())
        for nbrs in self.__adj.values():
            verts.update(nbrs.keys())
        verts.update(self.__definitive)
        return frozenset(verts)

    def indefinitive_nodes(self) -> frozenset[GenericCharacteristicName]:
        return self.all_nodes() - self.__definitive

    def find_path(
        self,
        src: GenericCharacteristicName,
        dst: GenericCharacteristicName,
    ) -> list[ComputationMethod[Any, Any]] | None:
        """
        Поиск любой цепочки преобразований src -> ... -> dst (BFS).
        Для ребра (v -> w) выбираем метод: сначала DEFAULT_COMPUTATION_KEY,
        иначе — детерминированно по имени.
        """
        if src == dst:
            return []

        visited: set[GenericCharacteristicName] = {src}
        parent: dict[
            GenericCharacteristicName, tuple[GenericCharacteristicName, ComputationMethod[Any, Any]]
        ] = {}
        q: deque[GenericCharacteristicName] = deque([src])

        while q:
            v = q.popleft()
            for w, methods in self.__adj.get(v, {}).items():
                if w not in visited and methods:
                    visited.add(w)
                    parent[w] = (v, self._pick_method(methods))
                    if w == dst:
                        path: list[ComputationMethod[Any, Any]] = []
                        cur = dst
                        while cur != src:
                            pv, m = parent[cur]
                            path.append(m)
                            cur = pv
                        path.reverse()
                        return path
                    q.append(w)
        return None

    def _validate_invariants(self) -> None:
        # (1) хотя бы один definitive
        if not self.__definitive:
            raise GraphInvariantError("There must be at least one definitive characteristic.")

        # (2) сильная связность среди definitive
        if not self._definitive_strongly_connected():
            raise GraphInvariantError("Definitive subgraph must be strongly connected.")

        # (3) каждая indefinitive достижима из некоторой definitive
        if not self._all_indefinitives_reachable_from_definitives():
            raise GraphInvariantError(
                "Every indefinitive node must be reachable from some definitive node."
            )

        # (4) из indefinitive нет пути к какой-либо definitive
        if self._exists_path_from_indefinitive_to_definitive():
            raise GraphInvariantError(
                "No path from any indefinitive node back to a definitive node is allowed."
            )

    def _reachable_from(
        self,
        start: GenericCharacteristicName,
        allowed: set[GenericCharacteristicName] | None = None,
    ) -> set[GenericCharacteristicName]:
        seen: set[GenericCharacteristicName] = {start}
        q: deque[GenericCharacteristicName] = deque([start])
        while q:
            v = q.popleft()
            for w in self.__adj.get(v, {}):
                if allowed is not None and w not in allowed:
                    continue
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        return seen

    def _reverse_adj(
        self,
    ) -> dict[GenericCharacteristicName, dict[GenericCharacteristicName, bool]]:
        rev: dict[GenericCharacteristicName, dict[GenericCharacteristicName, bool]] = {}
        for u, nbrs in self.__adj.items():
            for v, methods in nbrs.items():
                if methods:
                    rev.setdefault(v, {})[u] = True
                rev.setdefault(u, {})
        return rev

    def _definitive_strongly_connected(self) -> bool:
        if len(self.__definitive) <= 1:
            return True

        start = next(iter(self.__definitive))

        reachable = self._reachable_from(start, allowed=self.__definitive)
        if reachable != self.__definitive:
            return False

        rev = self._reverse_adj()
        seen: set[GenericCharacteristicName] = {start}
        q: deque[GenericCharacteristicName] = deque([start])
        while q:
            v = q.popleft()
            for w in rev.get(v, {}):
                if w in self.__definitive and w not in seen:
                    seen.add(w)
                    q.append(w)
        return seen == self.__definitive

    def _all_indefinitives_reachable_from_definitives(self) -> bool:
        indefs = self.indefinitive_nodes()
        if not indefs:
            return True
        total: set[GenericCharacteristicName] = set()
        for d in self.__definitive:
            total |= self._reachable_from(d, allowed=None)
        return indefs.issubset(total)

    def _exists_path_from_indefinitive_to_definitive(self) -> bool:
        for i in self.indefinitive_nodes():
            reach = self._reachable_from(i, allowed=None)
            if reach & self.__definitive:
                return True
        return False


class DistributionTypeRegister:
    _instance: ClassVar[Self | None] = None
    _register_kinds: dict[DistributionType, GenericCharacteristicRegister]

    def __new__(cls) -> Self:
        if cls._instance is None:
            self = super().__new__(cls)
            self._register_kinds = {}
            cls._instance = self
        return cls._instance

    def get(self, distribution_type: DistributionType) -> GenericCharacteristicRegister:
        reg = self._register_kinds.get(distribution_type)
        if reg is None:
            reg = GenericCharacteristicRegister(distribution_type=distribution_type)
            self._register_kinds[distribution_type] = reg
        if reg.distribution_type != distribution_type:
            raise TypeError(
                f"Inconsistent registry under key ({distribution_type}): "
                f"got ({reg.distribution_type}) inside"
            )
        return reg

    __call__ = get


def _configure(reg: DistributionTypeRegister) -> None:
    """
    Дефолтная конфигурация для 1D непрерывного случая:
    definitive-узлы: pdf, cdf, ppf; двунаправленные рёбра между каждой парой.
    """
    from math import isfinite

    import numpy as _np

    try:
        # Пока игнорим, потом мб стабы будем писать, не знаю
        from scipy import (  # type: ignore[import-untyped]
            integrate as _sp_integrate,
            optimize as _sp_optimize,
        )
    except Exception as e:
        raise RuntimeError(
            "SciPy is required for default continuous 1D computations. Please install scipy."
        ) from e

    PDF = GenericCharacteristicName("pdf")
    CDF = GenericCharacteristicName("cdf")
    PPF = GenericCharacteristicName("ppf")

    reg1C = reg.get(EuclidianDistributionType(Kind.CONTINUOUS, 1))

    def _resolve(distribution: Distribution, name: GenericCharacteristicName) -> ScalarFunc:
        try:
            fn = distribution.computation_strategy.query_method(name, distribution)
        except AttributeError as e:
            raise RuntimeError(
                "Distribution must provide computation_strategy.querry_method(name, distribution)."
            ) from e

        def _wrap(x: float) -> float:
            return float(fn(x))

        return _wrap

    def _ppf_brentq_from_cdf(cdf_func: ScalarFunc) -> ScalarFunc:
        def _ppf(q: float) -> float:
            if not (0.0 < q < 1.0):
                if q <= 0.0:
                    return float("-inf")
                if q >= 1.0:
                    return float("inf")
            L, R = -1.0, 1.0
            for _ in range(60):
                FL = float(cdf_func(L))
                FR = float(cdf_func(R))
                if q < FL:
                    L *= 2.0
                elif q > FR:
                    R *= 2.0
                else:
                    break
            else:
                raise RuntimeError("ppf bracketing failed for given cdf.")

            def f(x: float) -> float:
                return float(cdf_func(x) - q)

            return float(_sp_optimize.brentq(f, L, R, maxiter=256))

        return _ppf

    def _num_derivative(f: ScalarFunc, x: float, h: float = 1e-5) -> float:
        if not isfinite(x):
            return float("nan")
        f1 = float(f(x + h))
        f_1 = float(f(x - h))
        f2 = float(f(x + 2 * h))
        f_2 = float(f(x - 2 * h))
        return float((-f2 + 8 * f1 - 8 * f_1 + f_2) / (12.0 * h))

    def _ppf_derivative(ppf_func: ScalarFunc, q: float, h: float = 1e-5) -> float:
        return _num_derivative(ppf_func, q, h=h)

    def _fit_pdf_to_cdf(distribution: Distribution) -> FittedComputationMethod[float, float]:
        pdf_func = _resolve(distribution, PDF)

        def _cdf(x: float) -> float:
            val, _ = _sp_integrate.quad(lambda t: float(pdf_func(t)), float("-inf"), x, limit=200)
            return float(_np.clip(val, 0.0, 1.0))

        return FittedComputationMethod[float, float](target=CDF, sources=[PDF], func=_cdf)

    def _fit_cdf_to_pdf(distribution: Distribution) -> FittedComputationMethod[float, float]:
        cdf_func = _resolve(distribution, CDF)

        def _pdf(x: float) -> float:
            d = _num_derivative(cdf_func, x, h=1e-5)
            return float(max(d, 0.0))

        return FittedComputationMethod[float, float](target=PDF, sources=[CDF], func=_pdf)

    def _fit_cdf_to_ppf(distribution: Distribution) -> FittedComputationMethod[float, float]:
        cdf_func = _resolve(distribution, CDF)
        ppf_func = _ppf_brentq_from_cdf(cdf_func)
        return FittedComputationMethod[float, float](target=PPF, sources=[CDF], func=ppf_func)

    def _fit_ppf_to_cdf(distribution: Distribution) -> FittedComputationMethod[float, float]:
        import numpy as _np

        ppf_func = _resolve(distribution, PPF)

        def _cdf(x: float) -> float:
            if not isfinite(x):
                return 0.0 if x == float("-inf") else 1.0

            def f(q: float) -> float:
                return float(ppf_func(q) - x)

            lo, hi = 1e-12, 1.0 - 1e-12
            flo, fhi = f(lo), f(hi)
            if flo > 0.0:
                return 0.0
            if fhi < 0.0:
                return 1.0
            q = float(_sp_optimize.brentq(f, lo, hi, maxiter=256))
            return float(_np.clip(q, 0.0, 1.0))

        return FittedComputationMethod[float, float](target=CDF, sources=[PPF], func=_cdf)

    def _fit_pdf_to_ppf(distribution: Distribution) -> FittedComputationMethod[float, float]:
        fitted_cdf = _fit_pdf_to_cdf(distribution)
        cdf_func: ScalarFunc = fitted_cdf.func
        ppf_func = _ppf_brentq_from_cdf(cdf_func)
        return FittedComputationMethod[float, float](target=PPF, sources=[PDF], func=ppf_func)

    def _fit_ppf_to_pdf(distribution: Distribution) -> FittedComputationMethod[float, float]:
        ppf_func = _resolve(distribution, PPF)

        def _pdf(x: float) -> float:
            if not isfinite(x):
                return 0.0

            def f(q: float) -> float:
                return float(ppf_func(q) - x)

            lo, hi = 1e-12, 1.0 - 1e-12
            flo, fhi = f(lo), f(hi)
            if flo > 0.0 or fhi < 0.0:
                return 0.0
            q = float(_sp_optimize.brentq(f, lo, hi, maxiter=256))
            dppf = _ppf_derivative(ppf_func, q, h=1e-5)
            if dppf <= 0.0:
                return 0.0
            return float(1.0 / dppf)

        return FittedComputationMethod[float, float](target=PDF, sources=[PPF], func=_pdf)

    pdf_to_cdf = ComputationMethod[float, float](target=CDF, sources=[PDF], fitter=_fit_pdf_to_cdf)
    cdf_to_pdf = ComputationMethod[float, float](target=PDF, sources=[CDF], fitter=_fit_cdf_to_pdf)

    cdf_to_ppf = ComputationMethod[float, float](target=PPF, sources=[CDF], fitter=_fit_cdf_to_ppf)
    ppf_to_cdf = ComputationMethod[float, float](target=CDF, sources=[PPF], fitter=_fit_ppf_to_cdf)

    pdf_to_ppf = ComputationMethod[float, float](target=PPF, sources=[PDF], fitter=_fit_pdf_to_ppf)
    ppf_to_pdf = ComputationMethod[float, float](target=PDF, sources=[PPF], fitter=_fit_ppf_to_pdf)

    reg1C.add_bidirectional_definitive(
        pdf_to_cdf, cdf_to_pdf, name_ab=DEFAULT_COMPUTATION_KEY, name_ba=DEFAULT_COMPUTATION_KEY
    )

    reg1C.add_bidirectional_definitive(
        cdf_to_ppf, ppf_to_cdf, name_ab=DEFAULT_COMPUTATION_KEY, name_ba=DEFAULT_COMPUTATION_KEY
    )

    reg1C.add_bidirectional_definitive(
        pdf_to_ppf, ppf_to_pdf, name_ab=DEFAULT_COMPUTATION_KEY, name_ba=DEFAULT_COMPUTATION_KEY
    )


@lru_cache(maxsize=1)
def distribution_type_register() -> DistributionTypeRegister:
    reg = DistributionTypeRegister()
    _configure(reg)
    return reg


def _reset_distribution_type_register_for_tests() -> None:
    distribution_type_register.cache_clear()
