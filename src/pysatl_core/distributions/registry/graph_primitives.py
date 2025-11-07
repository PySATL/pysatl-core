"""
Edge metadata for characteristic graph.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import Any

from pysatl_core.distributions.computation import ComputationMethod
from pysatl_core.distributions.registry.constraint import _GraphPrimitiveConstraint

DEFAULT_COMPUTATION_KEY: str = "PySATL_default_computation"


@dataclass(frozen=True, slots=True)
class EdgeMeta:
    method: ComputationMethod[Any, Any]
    constraint: _GraphPrimitiveConstraint = field(default_factory=_GraphPrimitiveConstraint)


class GraphInvariantError(RuntimeError):
    """Raised when characteristic graph invariants are violated on a view."""
