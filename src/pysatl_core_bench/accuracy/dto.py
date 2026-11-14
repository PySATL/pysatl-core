"""Data transfer objects flowing through the accuracy pipeline.

Exactly two result-level DTOs exist by design decision (``AccuracyResult``
for the pair outcome and ``MetricRecord`` for per-region metrics); every
other class of this module is their nested supporting vocabulary.  All DTOs
are frozen dataclasses and JSON-serializable through ``to_json_dict()``.

Stage-owned vocabulary lives with its stage instead: the runtime
:class:`~pysatl_core_bench.accuracy.scenarios.grid.Grid` is defined next
to its generator, and the source-side :class:`PreparedSubject` lives in
``computations.protocol``.  This module depends only on the kernel
(``types``), never on a stage subpackage.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pysatl_core_bench.accuracy.canonical import frozen_mapping
from pysatl_core_bench.accuracy.types import Readiness, Status


@dataclass(frozen=True)
class SupportBounds:
    """Interval bounds of a distribution support (source-agnostic view)."""

    left: float
    right: float
    left_closed: bool = True
    right_closed: bool = True


@dataclass(frozen=True)
class SupportVerdict:
    """Readiness verdict of a source (never a result status)."""

    readiness: Readiness
    reason: str
    missing: tuple[float, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "readiness": self.readiness.value,
            "reason": self.reason,
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class ValidatorNote:
    """A note attached to results (degenerate comparison, invariant verdicts)."""

    code: str
    detail: str

    def to_json_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class Violation:
    """Outcome of an invariant check against one participant."""

    participant_id: str
    code: str
    detail: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "code": self.code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SpecialPoint:
    """A framework-predicted special point of the measured distribution."""

    x: float
    kind: str
    source: str

    def to_json_dict(self) -> dict[str, Any]:
        # REV-19: the predicted manifest travels with the result (see
        # AccuracyResult.special_points) — the report consumer can see what
        # the grid was densified against without replaying detection.
        return {"x": self.x, "kind": self.kind, "source": self.source}


@dataclass(frozen=True)
class MetricRecord:
    """Error metrics of one pair for one (characteristic, space, region).

    ``metrics`` are accuracy numbers against the reference;
    ``diagnostics`` are numbers that involve no reference (currently the
    ``e2_self_consistency`` family of ppf follow-ups) — same floats, same
    record, but never reported as accuracy (решение ред. 46).
    """

    characteristic: str
    space: str
    region: str
    grid_hash: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, float] = field(default_factory=dict)
    n_missing_reference: int = 0
    n_compared: int = 0
    n_excluded: int = 0
    n_invalid_subject: int = 0

    def __post_init__(self) -> None:  # REV-20
        object.__setattr__(self, "metrics", frozen_mapping(self.metrics))
        object.__setattr__(self, "diagnostics", frozen_mapping(self.diagnostics))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "characteristic": self.characteristic,
            "space": self.space,
            "region": self.region,
            "grid_hash": self.grid_hash,
            "metrics": dict(self.metrics),
            "diagnostics": dict(self.diagnostics),
            "n_missing_reference": self.n_missing_reference,
            "n_compared": self.n_compared,
            "n_excluded": self.n_excluded,
            "n_invalid_subject": self.n_invalid_subject,
        }


@dataclass(frozen=True)
class Provenance:
    """Origin passport of one comparison side.

    ``path_dump`` carries the actual computation plan (required for compute
    sources; ``None`` for pure lookup sources, whose identity is the
    ``table_hash``).  Provenance *describes* a side and does not prove it.
    """

    library: str
    library_version: str
    strategy: str | Mapping[str, Any] | None = None
    path_dump: Mapping[str, Any] | None = None
    table_hash: str | None = None
    conversion_facts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:  # REV-20
        if isinstance(self.strategy, Mapping):
            object.__setattr__(self, "strategy", frozen_mapping(self.strategy))
        if self.path_dump is not None:
            object.__setattr__(self, "path_dump", frozen_mapping(self.path_dump))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "library": self.library,
            "library_version": self.library_version,
            "strategy": (
                dict(self.strategy) if isinstance(self.strategy, Mapping) else self.strategy
            ),
            "path_dump": dict(self.path_dump) if self.path_dump is not None else None,
            "table_hash": self.table_hash,
            "conversion_facts": [dict(f) for f in self.conversion_facts],
        }


@dataclass(frozen=True)
class PointwiseSegment:
    """Layout of one (characteristic, region) slice inside a sidecar (D1).

    Pointwise arrays concatenate the per-region value blocks of all queries
    in measurement order; the segment records where each block starts and
    ends — and the question space of that block, because regions of one
    scenario may live in different spaces (REV-5) — so the flat sidecar
    stays joinable with the metric records without replaying grid
    construction (А-11).
    """

    characteristic: str
    region: str
    space: str
    start: int
    stop: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "characteristic": self.characteristic,
            "region": self.region,
            "space": self.space,
            "start": self.start,
            "stop": self.stop,
        }


@dataclass(frozen=True)
class PointwiseRef:
    """Reference to pointwise values kept in the results store sidecars."""

    uri: str
    hash: str
    space: str
    n_points: int
    segments: tuple[PointwiseSegment, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "hash": self.hash,
            "space": self.space,
            "n_points": self.n_points,
            "segments": [segment.to_json_dict() for segment in self.segments],
        }


@dataclass(frozen=True)
class AccuracyResult:
    """The result of exactly one participant pair (identity fully resolved)."""

    schema_version: str
    scenario_id: str
    spec_hash: str
    comparison_group_id: str
    sort_key: float | str | None
    pair: tuple[str, str]
    subject_id: str
    reference: str
    role_labels: Mapping[str, str]
    status: Status
    status_explanation: str
    validator_notes: tuple[ValidatorNote, ...] = ()
    invariant_violations: tuple[Violation, ...] = ()
    metrics: tuple[MetricRecord, ...] = ()
    space_computed: str | None = None
    space_reported: str | None = None
    y_true: tuple[float, ...] | None = None
    pointwise_subject: PointwiseRef | None = None
    pointwise_reference: PointwiseRef | None = None
    reference_provenance: Provenance | None = None
    subject_provenance: Provenance | None = None
    reference_hash: str | None = None
    special_points: tuple[SpecialPoint, ...] = ()
    """REV-19: the framework-predicted special-point manifest of the run.

    Detection happens once per scenario (N1 — the points are predicted by
    the framework, never taken from the measured output); the manifest is
    repeated per pair row so the stored result is self-describing: a
    consumer sees which points the grid was densified against without
    replaying detection or holding a live runner.
    """

    def __post_init__(self) -> None:  # REV-20
        object.__setattr__(self, "role_labels", frozen_mapping(self.role_labels))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "spec_hash": self.spec_hash,
            "comparison_group_id": self.comparison_group_id,
            "sort_key": self.sort_key,
            "pair": list(self.pair),
            "subject_id": self.subject_id,
            "reference": self.reference,
            "role_labels": dict(self.role_labels),
            "status": self.status.value,
            "status_explanation": self.status_explanation,
            "validator_notes": [n.to_json_dict() for n in self.validator_notes],
            "invariant_violations": [v.to_json_dict() for v in self.invariant_violations],
            "metrics": [m.to_json_dict() for m in self.metrics],
            "space_computed": self.space_computed,
            "space_reported": self.space_reported,
            "y_true": list(self.y_true) if self.y_true is not None else None,
            "pointwise_subject": (
                self.pointwise_subject.to_json_dict() if self.pointwise_subject else None
            ),
            "pointwise_reference": (
                self.pointwise_reference.to_json_dict() if self.pointwise_reference else None
            ),
            "subject_provenance": (
                self.subject_provenance.to_json_dict() if self.subject_provenance else None
            ),
            "reference_provenance": (
                self.reference_provenance.to_json_dict() if self.reference_provenance else None
            ),
            "reference_hash": self.reference_hash,
            "special_points": [p.to_json_dict() for p in self.special_points],
        }


@dataclass(frozen=True)
class AgreementEntry:
    """Metric differences of two sides over one matched region record."""

    characteristic: str
    space: str
    region: str
    grid_hash: str
    diffs: Mapping[str, float]

    def __post_init__(self) -> None:  # REV-20
        object.__setattr__(self, "diffs", frozen_mapping(self.diffs))


@dataclass(frozen=True)
class Agreement:
    """Metric differences between two results of the same scenario pair:
    agreement, never accuracy.

    ``side_a``/``side_b`` are the subject-side labels of the two results
    (equal pair, possibly different scenario shells — REV-5).
    ``skipped`` lists records/metrics present on one side only — nothing is
    silently dropped (N6, REV-5).
    """

    side_a: str
    side_b: str
    entries: tuple[AgreementEntry, ...]
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class SweepPoint:
    """One point of a sweep curve: one pair, one metric cell, one axis value."""

    sort_key: float | str
    pair: tuple[str, str]
    characteristic: str
    space: str
    region: str
    status: Status
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:  # REV-20
        object.__setattr__(self, "metrics", frozen_mapping(self.metrics))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "sort_key": self.sort_key,
            "pair": list(self.pair),
            "characteristic": self.characteristic,
            "space": self.space,
            "region": self.region,
            "status": self.status.value,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class SweepCurve:
    """Curve over the sweep axis (К-1): series of (sort_key, metrics) points.

    The axis lists every distinct numeric sort_key of the group (ascending),
    followed by distinct string sort_keys (lexicographic).  The axis is the
    full sweep range; the series carry measured points only — unmeasured
    stages stay visible in the flat table of the report (N6).
    """

    comparison_group_id: str
    axis: tuple[float | str, ...]
    points: tuple[SweepPoint, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "comparison_group_id": self.comparison_group_id,
            "axis": list(self.axis),
            "points": [p.to_json_dict() for p in self.points],
        }


@dataclass(frozen=True)
class ComparisonTable:
    """All results of one comparison group; nothing is silently dropped."""

    comparison_group_id: str
    rows: tuple[AccuracyResult, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "comparison_group_id": self.comparison_group_id,
            "rows": [r.to_json_dict() for r in self.rows],
        }


@dataclass(frozen=True)
class Report:
    """Rendered report of one comparison group; every status reaches it."""

    comparison_group_id: str
    rows: tuple[AccuracyResult, ...]
    summary: Mapping[str, int]
    text: str
    sweep: SweepCurve | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "comparison_group_id": self.comparison_group_id,
            "summary": dict(self.summary),
            "rows": [r.to_json_dict() for r in self.rows],
            "sweep": self.sweep.to_json_dict() if self.sweep is not None else None,
            "text": self.text,
        }


__all__ = [
    "AccuracyResult",
    "Agreement",
    "AgreementEntry",
    "ComparisonTable",
    "MetricRecord",
    "PointwiseRef",
    "Provenance",
    "Report",
    "SpecialPoint",
    "SupportBounds",
    "SupportVerdict",
    "SweepCurve",
    "SweepPoint",
    "ValidatorNote",
    "Violation",
]
