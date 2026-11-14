"""Accuracy benchmarking model of pysatl_core_bench.

Layout (three work folders plus the shared kernel):

- ``computations``  — the Compute stage: the ``ValueSource`` protocol and
  registry, the pysatl-core adapter, reference sources (precomputed
  tables, external library functions, on-demand caching), the space
  converter and the per-region error metrics;
- ``scenarios``     — the scenario stage: canonical ``*Spec`` structures,
  expand -> load -> validate of the raw JSON grammar, grid generation,
  framework-predicted special points, the invariant checker;
- ``reporting``     — the Compare/Report stages: the results store, the
  comparator (group table, invariant spread, agreement, sweep curve) and
  the reporter;
- kernel modules here — ``runner`` (the orchestrator), ``dto`` (DTOs
  flowing through the pipeline), ``types`` (the vocabulary),
  ``errors`` (the exception hierarchy), ``canonical`` (canonical forms
  and content hashes), ``core_state`` (core singleton hygiene).

The vocabulary lives in ``pysatl_core_bench.accuracy.types``; contracts
follow the accuracy model documents (issue #124 of pysatl-core).  The
public surface is re-exported here one-to-one from the subpackages.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .canonical import canonical_json, content_hash, grid_hash, values_hash
from .computations import (
    LRE_CAP,
    ExternalFunction,
    ExternalFunctionSource,
    MetricsEvaluator,
    OnDemandComputer,
    PrecomputedTable,
    PreparedSubject,
    PySATLAdapter,
    SourceRegistry,
    SpaceConverter,
    TableLookupSource,
    ValueSource,
)
from .core_state import reset_core_state
from .dto import (
    AccuracyResult,
    Agreement,
    AgreementEntry,
    ComparisonTable,
    MetricRecord,
    PointwiseRef,
    Provenance,
    Report,
    SpecialPoint,
    SupportBounds,
    SupportVerdict,
    SweepCurve,
    SweepPoint,
    ValidatorNote,
    Violation,
)
from .errors import (
    AdapterConfigurationError,
    BenchError,
    FrameworkDefect,
    GridMismatchError,
    ScenarioRejected,
    SourceNotRegistered,
    SourceOperationError,
)
from .reporting import Comparator, Reporter, ResultsStore
from .runner import AccuracyRunner
from .scenarios import (
    DEGENERATE_NOTE,
    DENSIFICATION_OFFSETS,
    TAIL_EXTENSION,
    VIOLATION_CODE,
    AccuracyQuery,
    DistributionSpec,
    Grid,
    GridGenerator,
    Interval,
    InvariantChecker,
    PairPolicy,
    ParticipantSpec,
    Region,
    ScenarioExpander,
    ScenarioLoader,
    ScenarioSpec,
    ScenarioValidator,
    SpecialPointDetector,
    StrategySpec,
)
from .types import (
    SCHEMA_VERSION,
    STATUS_PRIORITY,
    Capability,
    Readiness,
    Role,
    Status,
)

__all__ = [
    "DENSIFICATION_OFFSETS",
    "ExternalFunction",
    "LRE_CAP",
    "SCHEMA_VERSION",
    "STATUS_PRIORITY",
    "TAIL_EXTENSION",
    "VIOLATION_CODE",
    "AccuracyQuery",
    "AccuracyResult",
    "AccuracyRunner",
    "AdapterConfigurationError",
    "Agreement",
    "AgreementEntry",
    "BenchError",
    "Capability",
    "Comparator",
    "ComparisonTable",
    "DEGENERATE_NOTE",
    "DistributionSpec",
    "ExternalFunctionSource",
    "FrameworkDefect",
    "Grid",
    "GridGenerator",
    "GridMismatchError",
    "Interval",
    "InvariantChecker",
    "MetricRecord",
    "MetricsEvaluator",
    "OnDemandComputer",
    "PairPolicy",
    "ParticipantSpec",
    "PointwiseRef",
    "PrecomputedTable",
    "PreparedSubject",
    "Provenance",
    "PySATLAdapter",
    "Readiness",
    "Region",
    "Report",
    "Reporter",
    "ResultsStore",
    "Role",
    "ScenarioExpander",
    "ScenarioLoader",
    "ScenarioRejected",
    "ScenarioSpec",
    "ScenarioValidator",
    "SourceNotRegistered",
    "SourceOperationError",
    "SourceRegistry",
    "SpaceConverter",
    "SpecialPoint",
    "SpecialPointDetector",
    "Status",
    "StrategySpec",
    "SupportBounds",
    "SupportVerdict",
    "SweepCurve",
    "SweepPoint",
    "TableLookupSource",
    "ValidatorNote",
    "ValueSource",
    "Violation",
    "canonical_json",
    "content_hash",
    "grid_hash",
    "reset_core_state",
    "values_hash",
]
