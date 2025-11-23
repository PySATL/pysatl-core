"""
Characteristic Graph (global) + View (per distribution profile)
==============================================================

This module defines a *single* global directed graph over characteristic names
(PDF, CDF, etc.) and a per-profile **view** derived from it.

Node rules are guarded by declarative :class: `NodeConstraint`.
Edge rules are guarded by declarative :class: `EdgeConstraint`.
Node applicability is split into two parallel branches:

(A) Presence constraints — whether a node **exists** in a given profile.
(B) Definitiveness constraints — whether a node is **definitive** in a given
    profile (only meaningful if the node is present).

A concrete distribution/profile sees a filtered **view** of this graph; that
view is validated against invariants immediately upon construction.

Invariants (per view)
---------------------
1. The subgraph induced by definitive characteristics is **strongly connected**.
2. Every non-definitive characteristic is reachable from at least one definitive.
3. There is **no path** from any non-definitive characteristic **to** any definitive.

Design notes
------------
* Only **unary** computations are supported (1 source -> 1 target).
* Nodes must be **declared explicitly** via :meth:`CharacteristicRegistry.add_characteristic`.
  Nodes without a presence rule are considered **absent** by design.
* If a node has no definitiveness rule, it is treated as **non-definitive**.
* The registry is a **singleton** (see ``__new__``).
* Label variants are preserved; the view holds adjacency as ``src -> dst -> label``.
* Invariants are enforced **per view** (not during mutation of the registry).
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

    Responsibilities
    ----------------
    * Store nodes (characteristics) with **presence** and **definitiveness** rules.
    * Store labeled unary computations (edges) guarded by constraints.
    * Produce per-profile views (see :class:`RegistryView`).

    Public API
    ----------
    add_characteristic(name, is_definitive, presence_constraint=None, definitive_constraint=None)
        Declare a characteristic node with presence (mandatory) and optional definitiveness.
    add_computation(method, *, label=DEFAULT_COMPUTATION_KEY, constraint=None)
        Add a **unary** computation edge between already-declared nodes.

    Notes
    -----
    * Nodes **must** be declared before adding computations.
    * No invariant validation happens during mutation; invariants are checked when a
      :class:`RegistryView` is constructed (on ``view(...)``).
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
        # adjacency: src -> dst -> label -> EdgeMeta
        self._adj: dict[
            GenericCharacteristicName,
            dict[GenericCharacteristicName, dict[str, list[EdgeMeta]]],
        ] = {}
        # declared nodes
        self._all_nodes: set[GenericCharacteristicName] = set()
        # node rules
        self._presence_rules: dict[GenericCharacteristicName, GraphPrimitiveConstraint] = {}
        self._def_rules: dict[GenericCharacteristicName, GraphPrimitiveConstraint] = {}
        # label preference is retained for external consumers (not used internally here)
        self.label_preference: tuple[str, ...] = (DEFAULT_COMPUTATION_KEY,)
        self._initialized = True

    def __copy__(self) -> Self:
        """Singleton copy returns the same instance."""
        return self

    def __deepcopy__(self, memo: dict[Any, Any]) -> Self:
        """Singleton deepcopy returns the same instance (no duplication)."""
        return self

    def __reduce__(self) -> tuple[type[Self], tuple[()]]:
        """Ensure pickling keeps the singleton semantics."""
        return self.__class__, ()

    @classmethod
    def _reset(cls) -> None:
        """Reset the singleton (test helper)."""
        cls._instance = None

    # --------------------------------------------------------------------- #
    # Registration API
    # --------------------------------------------------------------------- #

    def _add_node(self, node: GenericCharacteristicName) -> None:
        """Insert node into the registry (idempotent)."""
        self._adj.setdefault(node, {})
        self._all_nodes.add(node)

    def _ensure_node(self, node: GenericCharacteristicName) -> bool:
        """Check that a node has been declared via :meth:`add_characteristic`."""
        return node in self._all_nodes

    def _add_presence_rule(
        self, name: GenericCharacteristicName, constraint: GraphPrimitiveConstraint | None
    ) -> None:
        """
        Register a presence rule for a node.

        Parameters
        ----------
        name
            Characteristic name.
        constraint
            Node presence constraint. If ``None``, a pass-through constraint is used.

        Notes
        -----
        * Duplicate calls for the same node are ignored with a warning.
        """
        if name in self._presence_rules:
            warnings.warn(
                f"Node {name} have been already added. Constraint will not be taken into account",
                UserWarning,
                stacklevel=3,
            )
            return
        self._presence_rules.setdefault(
            name, constraint if constraint is not None else GraphPrimitiveConstraint()
        )

    def _add_definitive_rule(
        self, name: GenericCharacteristicName, constraint: GraphPrimitiveConstraint | None
    ) -> None:
        """
        Register a definitiveness rule for a node.

        Parameters
        ----------
        name
            Characteristic name.
        constraint
            Node definitiveness constraint. If ``None``, a pass-through constraint is used.

        Notes
        -----
        * Duplicate calls for the same node are ignored with a warning.
        """
        if name in self._def_rules and constraint is not None:
            warnings.warn(
                f"Node {name} have been already added. Constraint will not be taken into account",
                UserWarning,
                stacklevel=3,
            )
            return
        self._def_rules.setdefault(
            name, constraint if constraint is not None else GraphPrimitiveConstraint()
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def add_computation(
        self,
        method: ComputationMethod[Any, Any],
        *,
        label: str = DEFAULT_COMPUTATION_KEY,
        constraint: GraphPrimitiveConstraint | None = None,
    ) -> None:
        """
        Add a labeled **unary** computation edge.

        Parameters
        ----------
        method
            A computation object with ``sources: list[GenericCharacteristicName]`` of length 1
            and ``target: GenericCharacteristicName``.
        label
            Variant label for the (src -> dst) edge.
        constraint
            Edge-level applicability constraint. If ``None``, a pass-through constraint is used.

        Raises
        ------
        ValueError
            If method is not unary, or source/target nodes have not been declared.

        Notes
        -----
        * Label variants for the same (src, dst) pair are preserved side-by-side.
        * No invariant validation is performed here; it happens in :class:`RegistryView`.
        """
        if len(method.sources) != 1:
            raise ValueError("Only unary computations are supported (1 source -> 1 target).")

        src = method.sources[0]
        dst = method.target

        if not self._ensure_node(src) or not self._ensure_node(dst):
            raise ValueError("Source characteristic or destination characteristic is invalid.")

        self._adj[src].setdefault(dst, {})
        # TODO: We need to be careful here if some constraint more general and with the same label
        #  than other it can consume it
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
        Declare a characteristic node with presence and optional definitiveness.

        Parameters
        ----------
        name
            Characteristic name to declare.
        is_definitive
            If ``True``, also register a definitiveness rule for this node.
        presence_constraint
            Presence constraint for the node. If ``None``, a pass-through constraint is used.
        definitive_constraint
            Definitiveness constraint if ``is_definitive`` is ``True``. Ignored otherwise.

        Notes
        -----
        * This call both **registers** the node and **attaches** its presence/definitiveness rules.
        * If ``is_definitive`` is ``False`` yet a ``definitive_constraint`` is passed, it is ignored
          with a warning.
        """
        self._add_node(name)
        self._add_presence_rule(name, presence_constraint)
        if not is_definitive and definitive_constraint is not None:
            warnings.warn(
                f"Node {name} have been added as indefinitive but definitive constraint is "
                f"provided. Constraint will not be taken into account",
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
        Compute the set of **present** nodes for a given profile.

        Parameters
        ----------
        distr
            A distribution-like object whose features are consulted by constraints.

        Returns
        -------
        set of GenericCharacteristicName
            All nodes whose presence constraints allow the given profile.

        Notes
        -----
        Only nodes with explicitly registered presence rules are considered. Nodes
        that have not been declared via :meth:`add_characteristic` are absent by design.
        """
        present: set[GenericCharacteristicName] = set()
        for name, constraint in self._presence_rules.items():
            if constraint.allows(distr):
                present.add(name)
        return present

    def _compute_definitive_nodes(self, distr: Distribution) -> set[GenericCharacteristicName]:
        """
        Compute the set of **definitive** nodes for a given profile.

        Parameters
        ----------
        distr
            A distribution-like object whose features are consulted by constraints.

        Returns
        -------
        set of GenericCharacteristicName
            All nodes whose definitiveness constraints allow the given profile.
        """
        definitive: set[GenericCharacteristicName] = set()
        for name, constraint in self._def_rules.items():
            if constraint.allows(distr):
                definitive.add(name)
        return definitive

    def view(self, distr: Distribution) -> RegistryView:
        """
        Build a per-profile view that preserves label variants and enforces invariants.

        Parameters
        ----------
        distr
            A distribution-like profile.

        Returns
        -------
        RegistryView
            A filtered adjacency (preserving label variants) together with sets of
            present and definitive characteristics.

        Notes
        -----
        1) Edges are filtered by their constraints (``EdgeConstraint``).
        2) Edges touching **absent** nodes are removed; isolated present nodes are kept.
        3) Definitive nodes are intersected with present nodes.
        4) The constructed view is **validated** immediately (see :class:`RegistryView`).
        """
        # 1) Filter edges by applicability; keep all label variants
        adj: dict[
            GenericCharacteristicName, dict[GenericCharacteristicName, dict[str, EdgeMeta]]
        ] = {}
        for src, d in self._adj.items():
            for dst, variants in d.items():
                kept: dict[str, EdgeMeta] = {}
                for label, metas in variants.items():
                    for edge in metas:
                        if (edge.constraint or GraphPrimitiveConstraint()).allows(distr):
                            kept[label] = edge
                            # TODO: It is possible that there are two edges under the same label
                            #  that fit the same distribution, this should not be the case.
                            #  Taking the first one for now
                            break
                if kept:
                    adj.setdefault(src, {}).setdefault(dst, {}).update(kept)

        # 2) Apply node presence filtering (drop edges touching absent nodes)
        present_nodes = self._compute_present_nodes(distr)
        if present_nodes:
            adj = {
                s: {t: dict(variants) for t, variants in d.items() if t in present_nodes}
                for s, d in adj.items()
                if s in present_nodes
            }
        # ensure isolated present nodes are preserved
        for n in present_nodes:
            adj.setdefault(n, {})

        # 3) Compute definitive nodes and intersect with presence
        definitive_nodes = self._compute_definitive_nodes(distr) & present_nodes

        return RegistryView(adj, definitive_nodes, present_nodes)


# --------------------------------------------------------------------------- #
# Registry view
# --------------------------------------------------------------------------- #


class RegistryView:
    """
    A per-profile filtered view of the global graph that **preserves label variants**.

    Parameters
    ----------
    adj : Mapping[src, Mapping[dst, Mapping[label, EdgeMeta]]]
        Multi-edge adjacency with label variants kept intact.
    definitive_nodes : set of GenericCharacteristicName
        Nodes considered definitive in this view.
    present_nodes : set of GenericCharacteristicName
        Nodes that exist in this view (after presence filtering).

    Notes
    -----
    * Invariants are validated in ``__init__``. A violation raises :class:`GraphInvariantError`.
    * Path search honors label priority (see :meth:`find_path`).
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
        # normalize adjacency to dict-of-dict-of-dicts
        self._adj: dict[
            GenericCharacteristicName, dict[GenericCharacteristicName, dict[str, EdgeMeta]]
        ] = {}
        for s, d in adj.items():
            self._adj[s] = {t: dict(variants) for t, variants in d.items()}

        # characteristics sets
        self.definitive_characteristics: set[GenericCharacteristicName] = set(definitive_nodes)
        self.all_characteristics: set[GenericCharacteristicName] = set(present_nodes)

        # validate invariants immediately; if this fails — the global graph is inconsistent
        self._validate_invariants()

    # ---------------- public convenience ----------------

    @property
    def indefinitive_characteristics(self) -> set[GenericCharacteristicName]:
        """
        Present-but-not-definitive characteristics.

        Returns
        -------
        set of GenericCharacteristicName
        """
        return self.all_characteristics - self.definitive_characteristics

    def successors(
        self, v: GenericCharacteristicName
    ) -> Mapping[GenericCharacteristicName, Mapping[str, EdgeMeta]]:
        """
        Outgoing edges grouped by destination, each with its ``label -> EdgeMeta`` map.

        Parameters
        ----------
        v
            Source characteristic.

        Returns
        -------
        Mapping[GenericCharacteristicName, Mapping[str, EdgeMeta]]
        """
        return self._adj.get(v, {})

    def successors_nodes(self, v: GenericCharacteristicName) -> set[GenericCharacteristicName]:
        """
        Set of reachable neighbors from ``v`` (labels ignored).

        Parameters
        ----------
        v
            Source characteristic.

        Returns
        -------
        set of GenericCharacteristicName
        """
        return set(self._adj.get(v, {}).keys())

    def predecessors(self, v: GenericCharacteristicName) -> set[GenericCharacteristicName]:
        """
        Nodes that have at least one labeled edge into ``v``.

        Parameters
        ----------
        v
            Destination characteristic.

        Returns
        -------
        set of GenericCharacteristicName
        """
        res: set[GenericCharacteristicName] = set()
        for s, d in self._adj.items():
            if v in d and d[v]:
                res.add(s)
        return res

    def variants(
        self, src: GenericCharacteristicName, dst: GenericCharacteristicName
    ) -> Mapping[str, EdgeMeta]:
        """
        Label->EdgeMeta mapping for a fixed pair (src, dst).

        Parameters
        ----------
        src, dst
            Edge endpoints.

        Returns
        -------
        Mapping[str, EdgeMeta]
        """
        return self._adj.get(src, {}).get(dst, {})

    # ---------------- path search (labels honored) ----------------

    def find_path(
        self,
        src: GenericCharacteristicName,
        dst: GenericCharacteristicName,
        *,
        prefer_label: str | None = None,
    ) -> list[Any] | None:
        """
        Find a conversion chain ``src -> ... -> dst`` using BFS.

        Label selection policy per (v -> w):
          1) use ``prefer_label`` if present;
          2) else use :data:`DEFAULT_COMPUTATION_KEY` if present;
          3) else use the lexicographically smallest label (deterministic fallback).

        Parameters
        ----------
        src, dst
            Endpoints of the desired path.
        prefer_label
            Optional preferred label.

        Returns
        -------
        list[Any] | None
            Ordered list of methods (edge payloads) if a path exists, otherwise ``None``.
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
                    # reconstruct
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

    # ---------------- invariants: public trigger ----------------

    def _validate_invariants(self) -> None:
        """
        Validate all graph invariants; raise :class:`GraphInvariantError` on failure.

        Invariants (per view)
        ---------------------
        1) Definitive subgraph is strongly connected.
        2) Every non-definitive is reachable from some definitive.
        3) No path from any non-definitive back to any definitive.
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

    # ---------------- invariants: helpers (no external params) ----------------

    def _definitive_strongly_connected(self) -> bool:
        """
        Check strong connectivity within the definitive subgraph.

        Returns
        -------
        bool
        """
        defs = self.definitive_characteristics
        if len(defs) <= 1:
            return True
        start = next(iter(defs))
        fwd = self._reachable_from(start, allowed=defs)
        if fwd != (defs - {start}):
            return False
        # reverse reachability within defs
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
        Check that every non-definitive node is reachable from some definitive.

        Returns
        -------
        bool
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
        Check that no indefinitive node reaches any definitive node.

        Returns
        -------
        bool
        """
        defs = self.definitive_characteristics
        return any(self._reachable_from(i) & defs for i in self.indefinitive_characteristics)

    # ---------------- reachability (labels ignored) ----------------

    def _reachable_from(
        self,
        src: GenericCharacteristicName,
        *,
        allowed: set[GenericCharacteristicName] | None = None,
    ) -> set[GenericCharacteristicName]:
        """
        Forward reachability from ``src``.

        Parameters
        ----------
        src
            Start node.
        allowed
            Optional restriction to a subset of nodes.

        Returns
        -------
        set of GenericCharacteristicName
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

    # ---------------- label picking ----------------

    @staticmethod
    def _pick_method(
        variants: Mapping[str, EdgeMeta],
        prefer_label: str | None,
    ) -> Any:
        """
        Pick the payload method from a ``label -> EdgeMeta`` mapping.

        Selection priority
        ------------------
        1) ``prefer_label`` if present;
        2) :data:`DEFAULT_COMPUTATION_KEY` if present;
        3) lexicographically smallest label.

        Parameters
        ----------
        variants
            Mapping of label to edge metadata.
        prefer_label
            Optional preferred label.

        Returns
        -------
        Any
            The selected edge payload (usually a computation method).
        """
        if prefer_label and prefer_label in variants:
            return variants[prefer_label].method
        if DEFAULT_COMPUTATION_KEY in variants:
            return variants[DEFAULT_COMPUTATION_KEY].method
        # deterministic fallback
        label = sorted(variants.keys())[0]
        return variants[label].method
