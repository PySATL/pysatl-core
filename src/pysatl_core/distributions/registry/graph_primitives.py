"""
Edge metadata and graph error definitions.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pysatl_core.distributions.computation import (
    AnalyticalComputation,
    ComputationMethod,
)
from pysatl_core.distributions.registry.constraint import GraphPrimitiveConstraint
from pysatl_core.types import LabelName

type EdgeMethod = ComputationMethod[Any, Any] | AnalyticalComputation[Any, Any]

DEFAULT_COMPUTATION_KEY: LabelName = "PySATL_default_computation"
"""Default label for computation edges when no specific label is provided."""


@dataclass(frozen=True, slots=True)
class EdgeMeta(ABC):
    """
    Metadata for a computation edge in the characteristic graph.

    Parameters
    ----------
    method : EdgeMethod
        The computation method that defines the edge.
    constraint : GraphPrimitiveConstraint
        Constraint determining when this edge is applicable to a distribution.
        Defaults to a pass-through constraint that always allows.
    is_analytical : bool
        Whether this edge represents an analytical computation.
    """

    method: EdgeMethod
    constraint: GraphPrimitiveConstraint = field(default_factory=GraphPrimitiveConstraint)
    is_analytical: bool = field(default=False)

    @abstractmethod
    def edge_kind(self) -> str:
        """Return edge kind identifier."""
        ...


@dataclass(frozen=True, slots=True)
class ComputationEdgeMeta(EdgeMeta):
    """
    Edge metadata for conversion computations from the registry graph.
    """

    method: ComputationMethod[Any, Any]
    is_analytical: bool = field(default=False)

    def edge_kind(self) -> str:
        return "computation"


@dataclass(frozen=True, slots=True)
class AnalyticalLoopEdgeMeta(EdgeMeta):
    """
    Edge metadata for self-loop analytical computations from a distribution.
    """

    method: AnalyticalComputation[Any, Any]
    is_analytical: bool = field(default=True)

    def edge_kind(self) -> str:
        return "analytical_loop"


@dataclass(frozen=True, slots=True)
class TransformationLoopEdgeMeta(EdgeMeta):
    """
    Edge metadata for transformation-provided self-loops.

    Such loops are attached from ``analytical_computations`` as regular
    stopping points for the strategy, but they are not considered fully
    analytical by the graph semantics.
    """

    method: AnalyticalComputation[Any, Any]
    is_analytical: bool = field(default=False)

    def edge_kind(self) -> str:
        return "transformation_loop"


class GraphInvariantError(RuntimeError):
    """
    Raised when characteristic graph invariants are violated.

    This error occurs when creating a RegistryView and the filtered graph
    does not satisfy the required invariants (e.g., definitive subgraph
    is not strongly connected).
    """
