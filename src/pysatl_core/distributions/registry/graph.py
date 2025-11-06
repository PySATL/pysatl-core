"""
Characteristic Graph (global) + View (per distribution profile)
==============================================================

This module defines the global characteristic registry and per-distribution views.

The CharacteristicRegistry maintains a directed graph over characteristic names
(PDF, CDF, PPF, PMF, etc.) with nodes and edges guarded by constraints. Each
distribution profile sees a filtered view of this graph based on its specific
features (kind, dimension, etc.).

Core concepts:
- **Nodes**: Characteristics (PDF, CDF, etc.) with presence and definitiveness rules
- **Edges**: Unary computation methods between characteristics
- **Constraints**: Rules that determine when nodes/edges are applicable
- **View**: A filtered subgraph for a specific distribution
- **Definitive characteristics**: Starting points for computations
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pysatl_core.distributions.registry.constraint import GraphPrimitiveConstraint
from pysatl_core.distributions.registry.graph_primitives import (
    DEFAULT_COMPUTATION_KEY,
    EdgeMeta,
    GraphInvariantError,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.computation import ComputationMethod
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import GenericCharacteristicName


# --------------------------------------------------------------------------- #
# Registry (singleton)
# --------------------------------------------------------------------------- #


class CharacteristicRegistry:
    """
    Global characteristic graph with constraint-guarded nodes and edges.

    This registry maintains the complete graph of characteristics and computation
    methods. It serves as a singleton that can be configured once and then used
    to create filtered views for specific distributions.

    Invariants (enforced per view):
    1. The subgraph induced by definitive characteristics is strongly connected
    2. Every non-definitive characteristic is reachable from at least one definitive
    3. No path exists from any non-definitive characteristic to any definitive

    Methods
    -------
    add_characteristic(name, is_definitive, presence_constraint=None, definitive_constraint=None)
        Declare a characteristic with presence and optional definitiveness rules.
    add_computation(method, label=DEFAULT_COMPUTATION_KEY, constraint=None)
        Add a unary computation edge between declared nodes.
    view(distr)
        Create a filtered view for the given distribution.

    Notes
    -----
    - Nodes must be declared before adding computations
    - Only unary computations (1 source → 1 target) are supported
    - No invariant validation happens during mutation; validation occurs when
      creating a view with view()
    """

    _instance: ClassVar[Self | None] = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            inst = super().__new__(cls)
            cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        # Adjacency: src → dst → label → [EdgeMeta]
        self._adj: dict[
            GenericCharacteristicName,
            dict[GenericCharacteristicName, dict[str, list[EdgeMeta]]],
        ] = {}
        self._all_nodes: set[GenericCharacteristicName] = set()

        # Node constraints
        self._presence_rules: dict[GenericCharacteristicName, GraphPrimitiveConstraint] = {}
        self._def_rules: dict[GenericCharacteristicName, GraphPrimitiveConstraint] = {}

        # Label preference for path finding
        self.label_preference: tuple[str, ...] = (DEFAULT_COMPUTATION_KEY,)

        self._initialized = True

    def __copy__(self) -> Self:
        """Singleton copy returns the same instance."""
        return self

    def __deepcopy__(self, memo: dict[Any, Any]) -> Self:
        """Singleton deepcopy returns the same instance."""
        return self

    def __reduce__(self) -> tuple[type[Self], tuple[()]]:
        """Ensure pickling preserves singleton semantics."""
        return self.__class__, ()

    @classmethod
    def _reset(cls) -> None:
        """Reset the singleton"""
        cls._instance = None

    def _add_node(self, node: GenericCharacteristicName) -> None:
        """Insert node into the registry (idempotent)."""
        self._adj.setdefault(node, {})
        self._all_nodes.add(node)

    def _ensure_node(self, node: GenericCharacteristicName) -> bool:
        """Check if a node has been declared via add_characteristic()."""
        return node in self._all_nodes

    def _add_presence_rule(
        self, name: GenericCharacteristicName, constraint: GraphPrimitiveConstraint | None
    ) -> None:
        """
        Register a presence rule for a node.

        Warns if the node already has a presence rule.
        """
        if name in self._presence_rules:
            warnings.warn(
                f"Node {name} already has a presence rule. New constraint will be ignored.",
                UserWarning,
                stacklevel=3,
            )
            return
        self._presence_rules[name] = (
            constraint if constraint is not None else GraphPrimitiveConstraint()
        )

    def _add_definitive_rule(
        self, name: GenericCharacteristicName, constraint: GraphPrimitiveConstraint | None
    ) -> None:
        """
        Register a definitiveness rule for a node.

        Warns if the node already has a definitiveness rule.
        """
        if name in self._def_rules and constraint is not None:
            warnings.warn(
                f"Node {name} already has a definitiveness rule. New constraint will be ignored.",
                UserWarning,
                stacklevel=3,
            )
            return
        self._def_rules[name] = constraint if constraint is not None else GraphPrimitiveConstraint()

    def add_computation(
        self,
        method: ComputationMethod[Any, Any],
        *,
        label: str = DEFAULT_COMPUTATION_KEY,
        constraint: GraphPrimitiveConstraint | None = None,
    ) -> None:
        """
        Add a labeled unary computation edge.

        Parameters
        ----------
        method : ComputationMethod
            Computation object with exactly one source and one target.
        label : str, default=DEFAULT_COMPUTATION_KEY
            Variant label for the edge.
        constraint : GraphPrimitiveConstraint, optional
            Edge applicability constraint. If None, a pass-through constraint is used.

        Raises
        ------
        ValueError
            If method is not unary, or source/target nodes are not declared.

        Notes
        -----
        - Multiple edges with different labels can exist between the same nodes
        - The first matching edge for each label is kept when creating views
        """
        if len(method.sources) != 1:
            raise ValueError("Only unary computations are supported (1 source → 1 target).")

        src = method.sources[0]
        dst = method.target

        if not self._ensure_node(src) or not self._ensure_node(dst):
            raise ValueError("Source characteristic or destination characteristic is invalid.")

        self._adj[src].setdefault(dst, {})
        # TODO: We need to be careful here if some constraint more general and with the same label
        #  than other it can consume it. Actually, the same label methods should not intersect their
        #  constraints
        self._adj[src][dst].setdefault(label, [])
        self._adj[src][dst][label].append(
            EdgeMeta(
                method=method,
                constraint=constraint or GraphPrimitiveConstraint(),
            )
        )

    def add_characteristic(
        self,
        name: GenericCharacteristicName,
        is_definitive: bool,
        *,
        presence_constraint: GraphPrimitiveConstraint | None = None,
        definitive_constraint: GraphPrimitiveConstraint | None = None,
    ) -> None:
        """
        Declare a characteristic with presence and optional definitiveness rules.

        Parameters
        ----------
        name : str
            Characteristic name (e.g., "pdf", "cdf").
        is_definitive : bool
            Whether this characteristic can serve as a starting point for computations.
        presence_constraint : GraphPrimitiveConstraint, optional
            Constraint determining when this characteristic exists for a distribution.
        definitive_constraint : GraphPrimitiveConstraint, optional
            Constraint determining when this characteristic is definitive.
            Ignored if is_definitive is False.

        Notes
        -----
        - If is_definitive is False but definitive_constraint is provided,
          a warning is issued and the constraint is ignored
        - Presence constraints are required; without one, the characteristic
          will never appear in any view
        """
        self._add_node(name)
        self._add_presence_rule(name, presence_constraint)

        if not is_definitive and definitive_constraint is not None:
            warnings.warn(
                f"Node {name} is non-definitive but has a definitive constraint. "
                "Constraint will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if is_definitive:
            self._add_definitive_rule(name, definitive_constraint)

    # --------------------------------------------------------------------- #
    # Views
    # --------------------------------------------------------------------- #

    def _compute_present_nodes(self, distr: Distribution) -> set[GenericCharacteristicName]:
        """
        Compute characteristics present for the given distribution.

        Returns
        -------
        set of str
            Characteristics whose presence constraints allow this distribution.
        """
        present: set[GenericCharacteristicName] = set()
        for name, constraint in self._presence_rules.items():
            if constraint.allows(distr):
                present.add(name)
        return present

    def _compute_definitive_nodes(self, distr: Distribution) -> set[GenericCharacteristicName]:
        """
        Compute definitive characteristics for the given distribution.

        Returns
        -------
        set of str
            Characteristics whose definitiveness constraints allow this distribution.
        """
        definitive: set[GenericCharacteristicName] = set()
        for name, constraint in self._def_rules.items():
            if constraint.allows(distr):
                definitive.add(name)
        return definitive

    def view(self, distr: Distribution) -> RegistryView:
        """
        Create a filtered view of the graph for the given distribution.

        Parameters
        ----------
        distr : Distribution
            Distribution profile to filter for.

        Returns
        -------
        RegistryView
            Filtered view containing only applicable nodes and edges.

        Notes
        -----
        1. Filters edges by their constraints
        2. Removes edges touching absent nodes
        3. Computes definitive nodes from the remaining present nodes
        4. Validates graph invariants
        """
        # 1) Filter edges by applicability
        adj: dict[
            GenericCharacteristicName, dict[GenericCharacteristicName, dict[str, EdgeMeta]]
        ] = {}
        for src, d in self._adj.items():
            for dst, variants in d.items():
                kept: dict[str, EdgeMeta] = {}
                for label, metas in variants.items():
                    for edge in metas:
                        if edge.constraint.allows(distr):
                            kept[label] = edge
                            # TODO: It is possible that there are two edges under the same label
                            #  that fit the same distribution, this should not be the case.
                            #  Taking the first one for now
                            break
                if kept:
                    adj.setdefault(src, {}).setdefault(dst, {}).update(kept)

        # 2) Filter by node presence
        present_nodes = self._compute_present_nodes(distr)
        if present_nodes:
            adj = {
                src: {dst: dict(variants) for dst, variants in d.items() if dst in present_nodes}
                for src, d in adj.items()
                if src in present_nodes
            }
        # Ensure isolated present nodes are preserved
        for node in present_nodes:
            adj.setdefault(node, {})

        # 3) Compute definitive nodes (must be present)
        definitive_nodes = self._compute_definitive_nodes(distr) & present_nodes

        return RegistryView(adj, definitive_nodes, present_nodes)


# --------------------------------------------------------------------------- #
# Registry view
# --------------------------------------------------------------------------- #


class RegistryView:
    """
    Filtered view of the characteristic graph for a specific distribution.

    This view contains only the nodes and edges applicable to a particular
    distribution profile, with all graph invariants validated.

    Parameters
    ----------
    adj : Mapping[src, Mapping[dst, Mapping[label, EdgeMeta]]]
        Filtered adjacency preserving label variants.
    definitive_nodes : set of str
        Definitive characteristics in this view.
    present_nodes : set of str
        All present characteristics in this view.

    Raises
    ------
    GraphInvariantError
        If any graph invariant is violated.

    Attributes
    ----------
    definitive_characteristics : set of str
        Definitive characteristics for this distribution.
    all_characteristics : set of str
        All present characteristics for this distribution.
    """

    def __init__(
        self,
        adj: Mapping[
            GenericCharacteristicName,
            Mapping[GenericCharacteristicName, Mapping[str, EdgeMeta]],
        ],
        definitive_nodes: set[GenericCharacteristicName],
        present_nodes: set[GenericCharacteristicName],
    ) -> None:
        # Deep copy adjacency to ensure immutability
        self._adj: dict[
            GenericCharacteristicName, dict[GenericCharacteristicName, dict[str, EdgeMeta]]
        ] = {}
        for src, d in adj.items():
            self._adj[src] = {dst: dict(variants) for dst, variants in d.items()}

        self.definitive_characteristics: set[GenericCharacteristicName] = set(definitive_nodes)
        self.all_characteristics: set[GenericCharacteristicName] = set(present_nodes)

        # Validate invariants immediately
        self._validate_invariants()

    @property
    def indefinitive_characteristics(self) -> set[GenericCharacteristicName]:
        """
        Present but non-definitive characteristics.

        Returns
        -------
        set of str
            Characteristics that exist but are not definitive.
        """
        return self.all_characteristics - self.definitive_characteristics

    def successors(
        self, v: GenericCharacteristicName
    ) -> Mapping[GenericCharacteristicName, Mapping[str, EdgeMeta]]:
        """
        Get outgoing edges from a characteristic.

        Parameters
        ----------
        v : str
            Source characteristic.

        Returns
        -------
        Mapping[str, Mapping[str, EdgeMeta]]
            Destination → label → edge metadata.
        """
        return self._adj.get(v, {})

    def successors_nodes(self, v: GenericCharacteristicName) -> set[GenericCharacteristicName]:
        """
        Get directly reachable characteristics from v.

        Parameters
        ----------
        v : str
            Source characteristic.

        Returns
        -------
        set of str
            Characteristics directly reachable from v.
        """
        return set(self._adj.get(v, {}).keys())

    def predecessors(self, v: GenericCharacteristicName) -> set[GenericCharacteristicName]:
        """
        Get characteristics with edges to v.

        Parameters
        ----------
        v : str
            Destination characteristic.

        Returns
        -------
        set of str
            Characteristics that can reach v directly.
        """
        res: set[GenericCharacteristicName] = set()
        for src, d in self._adj.items():
            if v in d and d[v]:
                res.add(src)
        return res

    def variants(
        self, src: GenericCharacteristicName, dst: GenericCharacteristicName
    ) -> Mapping[str, EdgeMeta]:
        """
        Get all labeled edges between two characteristics.

        Parameters
        ----------
        src, dst : str
            Edge endpoints.

        Returns
        -------
        Mapping[str, EdgeMeta]
            Label → edge metadata mapping.
        """
        return self._adj.get(src, {}).get(dst, {})

    def find_path(
        self,
        src: GenericCharacteristicName,
        dst: GenericCharacteristicName,
        *,
        prefer_label: str | None = None,
    ) -> list[Any] | None:
        """
        Find a computation path from src to dst using BFS.

        Parameters
        ----------
        src, dst : str
            Source and destination characteristics.
        prefer_label : str, optional
            Preferred edge label to use when multiple options exist.

        Returns
        -------
        list of ComputationMethod or None
            List of computation methods forming the path, or None if no path exists.

        Notes
        -----
        Label selection priority:
        1. prefer_label if present
        2. DEFAULT_COMPUTATION_KEY if present
        3. Lexicographically smallest label
        """
        if src == dst:
            return []

        visited: set[GenericCharacteristicName] = {src}
        parent: dict[GenericCharacteristicName, tuple[GenericCharacteristicName, Any]] = {}
        queue: list[GenericCharacteristicName] = [src]
        qi = 0

        while qi < len(queue):
            v = queue[qi]
            qi += 1
            for w, by_label in self._adj.get(v, {}).items():
                if not by_label or w in visited:
                    continue
                method = self._pick_method(by_label, prefer_label)
                visited.add(w)
                parent[w] = (v, method)
                if w == dst:
                    # Reconstruct path
                    path: list[Any] = []
                    cur = dst
                    while cur != src:
                        pv, m = parent[cur]
                        path.append(m)
                        cur = pv
                    path.reverse()
                    return path
                queue.append(w)
        return None

    def _validate_invariants(self) -> None:
        """
        Validate all graph invariants.

        Raises
        ------
        GraphInvariantError
            If any invariant is violated.
        """
        if not self._definitive_strongly_connected():
            raise GraphInvariantError("Definitive subgraph must be strongly connected.")
        if not self._all_indefinitives_reachable_from_definitives():
            raise GraphInvariantError(
                "Every indefinitive characteristic must be reachable from some definitive."
            )
        if self._exists_path_from_indefinitive_to_definitive():
            raise GraphInvariantError(
                "No path from any indefinitive characteristic back to a definitive is allowed."
            )

    def _definitive_strongly_connected(self) -> bool:
        """
        Check if definitive characteristics form a strongly connected subgraph.

        Returns
        -------
        bool
            True if strongly connected.
        """
        defs = self.definitive_characteristics
        if len(defs) <= 1:
            return True
        start = next(iter(defs))
        fwd = self._reachable_from(start, allowed=defs)
        if fwd != (defs - {start}):
            return False

        # Check reverse reachability
        seen: set[GenericCharacteristicName] = {start}
        stack = [start]
        while stack:
            v = stack.pop()
            for w in self.predecessors(v):
                if w in defs and w not in seen:
                    seen.add(w)
                    stack.append(w)
        return seen == defs

    def _all_indefinitives_reachable_from_definitives(self) -> bool:
        """
        Check that all non-definitive nodes are reachable from definitive nodes.

        Returns
        -------
        bool
            True if all indefinitives are reachable.
        """
        indefs = self.indefinitive_characteristics
        if not indefs:
            return True
        total: set[GenericCharacteristicName] = set()
        for d in self.definitive_characteristics:
            total |= self._reachable_from(d)
        return indefs.issubset(total)

    def _exists_path_from_indefinitive_to_definitive(self) -> bool:
        """
        Check if any non-definitive node can reach a definitive node.

        Returns
        -------
        bool
            True if such a path exists (which would violate invariants).
        """
        defs = self.definitive_characteristics
        return any(self._reachable_from(i) & defs for i in self.indefinitive_characteristics)

    def _reachable_from(
        self,
        src: GenericCharacteristicName,
        *,
        allowed: set[GenericCharacteristicName] | None = None,
    ) -> set[GenericCharacteristicName]:
        """
        Compute forward reachable nodes from src.

        Parameters
        ----------
        src : str
            Starting node.
        allowed : set of str, optional
            Restrict to this set of nodes.

        Returns
        -------
        set of str
            Nodes reachable from src (excluding src itself).
        """
        if allowed is not None and src not in allowed:
            return set()

        visited: set[GenericCharacteristicName] = set()
        stack = [src]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            for w in self.successors_nodes(v):
                if allowed is not None and w not in allowed:
                    continue
                if w not in visited:
                    stack.append(w)
        visited.discard(src)
        return visited

    @staticmethod
    def _pick_method(
        variants: Mapping[str, EdgeMeta],
        prefer_label: str | None,
    ) -> Any:
        """
        Select a method from label variants.

        Parameters
        ----------
        variants : Mapping[str, EdgeMeta]
            Available edge variants.
        prefer_label : str, optional
            Preferred label.

        Returns
        -------
        Any
            Selected computation method.
        """
        if prefer_label and prefer_label in variants:
            return variants[prefer_label].method
        if DEFAULT_COMPUTATION_KEY in variants:
            return variants[DEFAULT_COMPUTATION_KEY].method
        label = sorted(variants.keys())[0]
        return variants[label].method
