"""
Characteristic Registry package.

Exports
-------
DEFAULT_COMPUTATION_KEY
Constraint, SetConstraint, NumericConstraint, EdgeConstraint, NodeConstraint
EdgeMeta, GraphInvariantError
CharacteristicRegistry, RegistryView
characteristic_registry, reset_characteristic_registry
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

# Public factory & reset (lazy configuration happens inside characteristic_registry())
from .configuration import (
    characteristic_registry,
    reset_characteristic_registry,
)

# Constraint layer
from .constraint import (
    Constraint,
    EdgeConstraint,
    NodeConstraint,
    NumericConstraint,
    SetConstraint,
)

# Core graph and view
from .graph import (
    CharacteristicRegistry,
    RegistryView,
)

# Graph primitives and constants
from .graph_primitives import (
    DEFAULT_COMPUTATION_KEY,
    EdgeMeta,
    GraphInvariantError,
)

__all__ = [
    # constants & primitives
    "DEFAULT_COMPUTATION_KEY",
    "EdgeMeta",
    "GraphInvariantError",
    # constraints
    "Constraint",
    "SetConstraint",
    "NumericConstraint",
    "EdgeConstraint",
    "NodeConstraint",
    # graph
    "CharacteristicRegistry",
    "RegistryView",
    # accessors
    "characteristic_registry",
    "reset_characteristic_registry",
]
