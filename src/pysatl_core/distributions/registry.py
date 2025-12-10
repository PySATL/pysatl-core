"""
Characteristic Graph Registry
=============================

A directed graph over characteristic names for a fixed
:class:`~pysatl_core.types.DistributionType`.

- characteristics: ``GenericCharacteristicName``.
- Edges: unary :class:`~pysatl_core.distributions.computation.ComputationMethod`
  (``1 source -> 1 target``).

This registry maintains invariants that capture our "definitive vs.
indefinitive" semantics:

Invariants
----------
1. There is at least one *definitive* characteristic.
2. The subgraph induced by the *definitive* characteristics is **strongly connected**.
3. Every *indefinitive* characteristic is reachable from at least one *definitive* characteristic.
4. No path from any *indefinitive* characteristic back to any *definitive* characteristic
is allowed.
1. There is at least one *definitive* node.
2. The subgraph induced by the *definitive* nodes is **strongly connected**.
3. Every *indefinitive* node is reachable from at least one *definitive* node.
4. No path from any *indefinitive* node back to any *definitive* node is allowed.

The module also exposes a singleton-like
:class:`DistributionTypeRegister` with a default configuration for the
univariate continuous case (``pdf``, ``cdf``, ``ppf`` pairwise bidirectional).

Notes
-----
- If you add additional conversions, the invariants are revalidated.
- Only **unary** edges are supported at the moment. For higher arity, consider
  a hypergraph structure.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pysatl_core.distributions.computation import (
    ComputationMethod,
    cdf_to_pdf_1C,
    cdf_to_ppf_1C,
    pdf_to_cdf_1C,
    ppf_to_cdf_1C,
)
from pysatl_core.types import (
    DistributionType,
    EuclideanDistributionType,
    GenericCharacteristicName,
    Kind,
)

if TYPE_CHECKING:
    pass

DEFAULT_COMPUTATION_KEY: str = "PySATL_default_computation"


class GraphInvariantError(RuntimeError):
    """Raised when the characteristic graph invariants are violated."""


@dataclass(slots=True, frozen=True)
class GenericCharacteristicRegister:
    """
    Directed characteristic graph for a fixed :class:`DistributionType`.

    Attributes
    ----------
    distribution_type : DistributionType
        Distribution type the graph is built for.

    Notes
    -----
    Edges are stored as nested mappings:
    ``adjacency[src][dst] = dict[method_name, ComputationMethod]``
    with a reserved key :data:`DEFAULT_COMPUTATION_KEY` for the default method.
    """

    distribution_type: DistributionType

    __adj: dict[
        GenericCharacteristicName,
        dict[GenericCharacteristicName, dict[str, ComputationMethod[Any, Any]]],
    ] = field(default_factory=dict, repr=False)
    __definitive: set[GenericCharacteristicName] = field(default_factory=set, repr=False)

    def _ensure_vertex(self, v: GenericCharacteristicName) -> None:
        """Ensure that node ``v`` exists in the adjacency structure."""
        self.__adj.setdefault(v, {})

    def _pick_method(
        self, methods: dict[str, ComputationMethod[Any, Any]]
    ) -> ComputationMethod[Any, Any]:
        """Pick a deterministic method for an edge (prefer default key)."""
        if DEFAULT_COMPUTATION_KEY in methods:
            return methods[DEFAULT_COMPUTATION_KEY]
        name = next(iter(sorted(methods.keys())))
        return methods[name]

    def _add_edge_unary(
        self,
        method: ComputationMethod[Any, Any],
        name: str,
    ) -> None:
        """
        Add a unary edge to the graph.

        Parameters
        ----------
        method : ComputationMethod
            Unary conversion method (exactly one source).
        name : str
            Method name (edge label).
        """
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

        key = name
        self._ensure_vertex(sources[0])
        self._ensure_vertex(target)
        self.__adj[sources[0]].setdefault(target, {})[key] = method

    def register_definitive(self, name: GenericCharacteristicName) -> None:
        """
        Mark a node as *definitive*.

        Parameters
        ----------
        name : str
            Node to mark as definitive.

        Notes
        -----
        Strong connectivity for a single-node definitive subgraph is trivial.
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
        Add a bidirectional linkage between two *definitive* nodes.

        Parameters
        ----------
        a_to_b, b_to_a : ComputationMethod
            Inverse unary conversions linking the same pair of nodes.
        name_ab, name_ba : str
            Method names (edge labels).

        Raises
        ------
        GraphInvariantError
            If the methods do not form a proper inverse pair or invariants break.
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
        self._add_edge_unary(a_to_b, name=name_ab)
        self._add_edge_unary(b_to_a, name=name_ba)
        self._validate_invariants()

    def add_conversion(self, method: ComputationMethod[Any, Any], *, name: str) -> None:
        """
        Add an arbitrary unary conversion (``source -> target``).

        Parameters
        ----------
        method : ComputationMethod
            Unary conversion method.
        name : str
            Edge label.

        Notes
        -----
        Allowed link types:
          - definitive -> definitive
          - definitive -> indefinitive
          - indefinitive -> indefinitive

        Any link that creates a path from an indefinitive node to a definitive
        node is forbidden by the invariants and will cause validation to fail.
        """
        self._add_edge_unary(method, name=name)
        self._validate_invariants()

    def is_definitive(self, name: GenericCharacteristicName) -> bool:
        """Return ``True`` if ``name`` is definitive."""
        return name in self.__definitive

    def definitive_nodes(self) -> frozenset[GenericCharacteristicName]:
        """Return the set of definitive nodes."""
        return frozenset(self.__definitive)

    def all_nodes(self) -> frozenset[GenericCharacteristicName]:
        """Return the set of all graph nodes."""
        verts = set(self.__adj.keys())
        for nbrs in self.__adj.values():
            verts.update(nbrs.keys())
        verts.update(self.__definitive)
        return frozenset(verts)

    def indefinitive_nodes(self) -> frozenset[GenericCharacteristicName]:
        """Return the set of non-definitive nodes."""
        return self.all_nodes() - self.__definitive

    def find_path(
        self,
        src: GenericCharacteristicName,
        dst: GenericCharacteristicName,
    ) -> list[ComputationMethod[Any, Any]] | None:
        """
        Find any conversion chain ``src -> ... -> dst`` using BFS.

        For each edge ``(v -> w)``, the method is chosen deterministically:
        first try :data:`DEFAULT_COMPUTATION_KEY`, otherwise pick by sorted name.

        Parameters
        ----------
        src, dst : str
            Source and destination nodes.

        Returns
        -------
        list[ComputationMethod] or None
            A list of conversions if a path exists, otherwise ``None``.
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
        """Validate all graph invariants; raise :class:`GraphInvariantError` on failure."""
        # (1) at least one definitive
        if not self.__definitive:
            raise GraphInvariantError("There must be at least one definitive characteristic.")

        # (2) strong connectivity within definitive subgraph
        if not self._definitive_strongly_connected():
            raise GraphInvariantError("Definitive subgraph must be strongly connected.")

        # (3) every indefinitive is reachable from some definitive
        if not self._all_indefinitives_reachable_from_definitives():
            raise GraphInvariantError(
                "Every indefinitive node must be reachable from some definitive node."
            )

        # (4) no path from any indefinitive to any definitive
        if self._exists_path_from_indefinitive_to_definitive():
            raise GraphInvariantError(
                "No path from any indefinitive node back to a definitive node is allowed."
            )

    def _reachable_from(
        self,
        start: GenericCharacteristicName,
        allowed: set[GenericCharacteristicName] | None = None,
    ) -> set[GenericCharacteristicName]:
        """Return nodes reachable from ``start`` (optionally constrained to ``allowed``)."""
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

    def reachable_from(
        self,
        start: GenericCharacteristicName,
        allowed: set[GenericCharacteristicName] | None = None,
    ) -> set[GenericCharacteristicName]:
        """
        Return all nodes reachable from ``start`` via graph traversal.
        """
        return self._reachable_from(start, allowed)

    def _reverse_adj(
        self,
    ) -> dict[GenericCharacteristicName, dict[GenericCharacteristicName, bool]]:
        """Construct a reverse adjacency (for checks in the definitive subgraph)."""
        rev: dict[GenericCharacteristicName, dict[GenericCharacteristicName, bool]] = {}
        for u, nbrs in self.__adj.items():
            for v, methods in nbrs.items():
                if methods:
                    rev.setdefault(v, {})[u] = True
                rev.setdefault(u, {})
        return rev

    def _definitive_strongly_connected(self) -> bool:
        """Check strong connectivity in the definitive subgraph."""
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
        """Check that every non-definitive node is reachable from some definitive node."""
        indefs = self.indefinitive_nodes()
        if not indefs:
            return True
        total: set[GenericCharacteristicName] = set()
        for d in self.__definitive:
            total |= self._reachable_from(d, allowed=None)
        return indefs.issubset(total)

    def _exists_path_from_indefinitive_to_definitive(self) -> bool:
        """Check that there is no path from any indefinitive to any definitive node."""
        for i in self.indefinitive_nodes():
            reach = self._reachable_from(i, allowed=None)
            if reach & self.__definitive:
                return True
        return False


class DistributionTypeRegister:
    """Singleton-like registry that maps :class:`DistributionType` to its graph."""

    _instance: ClassVar[Self | None] = None
    _register_kinds: dict[DistributionType, GenericCharacteristicRegister]

    def __new__(cls) -> Self:
        if cls._instance is None:
            self = super().__new__(cls)
            self._register_kinds = {}
            cls._instance = self
        return cls._instance

    def get(self, distribution_type: DistributionType) -> GenericCharacteristicRegister:
        """
        Get (or create) the :class:`GenericCharacteristicRegister` for a distribution type.
        """
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
    Default configuration for the univariate continuous case.

    Definitive nodes: ``pdf``, ``cdf``, ``ppf``.
    Bidirectional edges between each pair are registered.
    """

    reg1C = reg.get(EuclideanDistributionType(Kind.CONTINUOUS, 1))

    reg1C.add_bidirectional_definitive(
        pdf_to_cdf_1C,
        cdf_to_pdf_1C,
        name_ab=DEFAULT_COMPUTATION_KEY,
        name_ba=DEFAULT_COMPUTATION_KEY,
    )

    reg1C.add_bidirectional_definitive(
        cdf_to_ppf_1C,
        ppf_to_cdf_1C,
        name_ab=DEFAULT_COMPUTATION_KEY,
        name_ba=DEFAULT_COMPUTATION_KEY,
    )


@lru_cache(maxsize=1)
def distribution_type_register() -> DistributionTypeRegister:
    """Return a cached :class:`DistributionTypeRegister` instance configured with defaults."""
    reg = DistributionTypeRegister()
    _configure(reg)
    return reg


def _reset_distribution_type_register_for_tests() -> None:
    """Reset the cached distribution type register (test helper)."""
    distribution_type_register.cache_clear()
