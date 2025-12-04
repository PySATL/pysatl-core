"""
Edge metadata and graph error definitions.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pysatl_core.distributions.registry.constraint import GraphPrimitiveConstraint

if TYPE_CHECKING:
    from typing import Any

    from pysatl_core.distributions.computation import ComputationMethod

DEFAULT_COMPUTATION_KEY: str = "PySATL_default_computation"
"""Default label for computation edges when no specific label is provided."""


@dataclass(frozen=True, slots=True)
class EdgeMeta:
    """
    Metadata for a computation edge in the characteristic graph.

    Parameters
    ----------
    method : ComputationMethod
        The computation method that defines the edge.
    constraint : GraphPrimitiveConstraint
        Constraint determining when this edge is applicable to a distribution.
        Defaults to a pass-through constraint that always allows.
    """

    method: ComputationMethod[Any, Any]
    constraint: GraphPrimitiveConstraint = field(default_factory=GraphPrimitiveConstraint)


class GraphInvariantError(RuntimeError):
    """
    Raised when characteristic graph invariants are violated.

    This error occurs when creating a RegistryView and the filtered graph
    does not satisfy the required invariants (e.g., definitive subgraph
    is not strongly connected).
    """
