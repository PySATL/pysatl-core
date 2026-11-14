"""Scenario stage of the accuracy model: grammar, grids, sanity rules.

The declarative side of the framework: the canonical scenario structures
(``*Spec``), the expand -> load -> validate pipeline of the raw JSON
grammar, grid generation per region, framework-predicted special points
and the optional invariant checker.  Value production belongs to the
sibling subpackage ``computations``; persistence, comparison and report
rendering belong to ``reporting``.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .detection import (
    KIND_PDF_JUMP,
    KIND_SINGULARITY,
    KIND_SUPPORT_BOUNDARY,
    SOURCE_CURATED,
    SOURCE_SUPPORT,
    SpecialPointDetector,
)
from .grid import DENSIFICATION_OFFSETS, TAIL_EXTENSION, Grid, GridGenerator
from .invariants import VIOLATION_CODE, InvariantChecker
from .scenario import DEGENERATE_NOTE, ScenarioExpander, ScenarioLoader, ScenarioValidator
from .spec import (
    AccuracyQuery,
    DistributionSpec,
    Interval,
    PairPolicy,
    ParticipantSpec,
    Region,
    ScenarioSpec,
    StrategySpec,
)

__all__ = [
    "AccuracyQuery",
    "DistributionSpec",
    "Grid",
    "DENSIFICATION_OFFSETS",
    "DEGENERATE_NOTE",
    "GridGenerator",
    "Interval",
    "InvariantChecker",
    "KIND_PDF_JUMP",
    "KIND_SINGULARITY",
    "KIND_SUPPORT_BOUNDARY",
    "PairPolicy",
    "ParticipantSpec",
    "Region",
    "ScenarioExpander",
    "ScenarioLoader",
    "ScenarioSpec",
    "ScenarioValidator",
    "SOURCE_CURATED",
    "SOURCE_SUPPORT",
    "SpecialPointDetector",
    "StrategySpec",
    "TAIL_EXTENSION",
    "VIOLATION_CODE",
]
