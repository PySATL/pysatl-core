"""
Characteristic Registry package

This package provides a graph-based registry for mathematical distribution
characteristics (PDF, CDF, PMF, PPF, etc.) with constraint-based filtering.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .configuration import (
    characteristic_registry,
    reset_characteristic_registry,
)
from .constraint import (
    Constraint,
    GraphPrimitiveConstraint,
    NumericConstraint,
    SetConstraint,
)
from .graph import (
    CharacteristicRegistry,
    RegistryView,
)
from .graph_primitives import (
    DEFAULT_COMPUTATION_KEY,
    EdgeMeta,
    GraphInvariantError,
)

__all__ = [
    # Graph primitives and constants
    "DEFAULT_COMPUTATION_KEY",
    "EdgeMeta",
    "GraphInvariantError",
    # Constraint types
    "Constraint",
    "SetConstraint",
    "NumericConstraint",
    "GraphPrimitiveConstraint",
    # Graph classes
    "CharacteristicRegistry",
    "RegistryView",
    # Factory functions
    "characteristic_registry",
    "reset_characteristic_registry",
]
