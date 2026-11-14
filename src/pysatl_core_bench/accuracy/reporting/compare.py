"""Comparison stage: the group table and pairwise agreement.

The comparator receives only the ``comparison_group_id`` and *reads* all
results of the group from the store — it never writes (В-4a).  The table
includes every result, whatever its status; nothing is silently dropped
(N6).  Invariant verdicts are spread inside the group only (N7) by the
shared :func:`spread_invariant_verdicts` helper: the runner applies it to
the in-memory results *before* writing them to the store (А-5, REV-16:
store rows agree with the report by construction, not by a re-spread at
read time — the comparator reads stored rows as they are).
``agreement`` compares two results of the *same* scenario pair: it
verifies pair identity and the per-record ``grid_hash`` first and reports
*agreement*, never accuracy (C3/D1).
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import replace

from pysatl_core_bench.accuracy.dto import (
    AccuracyResult,
    Agreement,
    AgreementEntry,
    ComparisonTable,
    SweepCurve,
    SweepPoint,
    ValidatorNote,
    Violation,
)
from pysatl_core_bench.accuracy.errors import GridMismatchError
from pysatl_core_bench.accuracy.reporting.store import ResultsStore
from pysatl_core_bench.accuracy.scenarios.invariants import VIOLATION_CODE
from pysatl_core_bench.accuracy.types import Status

__all__ = ["Comparator", "spread_invariant_verdicts"]


def _sort_key_order(key: float | str) -> tuple[int, float, str]:
    """Numeric keys first (ascending), then string keys (lexicographic)."""
    if isinstance(key, str):
        return (1, 0.0, key)
    return (0, float(key), "")


def spread_invariant_verdicts(results: list[AccuracyResult]) -> list[AccuracyResult]:
    """Spread the group's invariant verdicts over touching pairs (N7).

    A pair whose subject or reference was disproved by the invariant check
    cannot stay ``measured``: its status becomes ``invalid_reference`` with
    an explicit note, inside the comparison group only.  Idempotent: already
    spread rows are no longer ``measured`` and pass through unchanged.
    """
    violated: set[str] = set()
    for result in results:
        for violation in result.invariant_violations:
            if violation.code == VIOLATION_CODE and violation.participant_id:
                violated.add(violation.participant_id)
    if not violated:
        return list(results)
    return [_spread_result(result, violated) for result in results]


class Comparator:
    """Builds the group table and pairwise agreements from stored results."""

    def __init__(self, store: ResultsStore) -> None:
        self._store = store

    def table(self, group_id: str) -> ComparisonTable:
        """All results of the group, exactly as stored (N6).

        The runner spreads invariant verdicts before writing (А-5, REV-16),
        so stored rows already agree with the report; the comparator reads
        them without a re-spread.
        """
        rows = self._store.read_group(group_id)
        return ComparisonTable(comparison_group_id=group_id, rows=tuple(rows))

    def curve(self, group_id: str) -> SweepCurve:
        """Sweep curve over ``sort_key`` for groups of a sweep (К-1).

        Built from the comparator's own table; invariant verdicts were
        spread by the runner before writing (А-5), so verdict-affected rows
        are reflected in the curve the same way as in the table (N7).  The
        axis carries every distinct
        numeric sort_key (ascending) then every distinct string sort_key;
        groups with fewer than two distinct sort_keys produce an empty axis
        (no curve).  Series points are the measured rows only; unmeasured
        stages remain visible in the flat table (N6).
        """
        rows = self.table(group_id).rows
        axis = sorted(
            {row.sort_key for row in rows if row.sort_key is not None}, key=_sort_key_order
        )
        if len(axis) < 2:
            return SweepCurve(comparison_group_id=group_id, axis=(), points=())
        points: list[SweepPoint] = []
        for row in rows:
            if row.sort_key is None or not row.metrics:
                continue
            for metric in row.metrics:
                points.append(
                    SweepPoint(
                        sort_key=row.sort_key,
                        pair=row.pair,
                        characteristic=metric.characteristic,
                        space=metric.space,
                        region=metric.region,
                        status=row.status,
                        metrics=dict(metric.metrics),
                    )
                )
        points.sort(
            key=lambda p: (
                p.characteristic,
                p.space,
                p.region,
                p.pair,
                _sort_key_order(p.sort_key),
            )
        )
        return SweepCurve(comparison_group_id=group_id, axis=tuple(axis), points=tuple(points))

    def agreement(self, a: AccuracyResult, b: AccuracyResult) -> Agreement:
        """Metric differences between two results of the *same* scenario pair.

        Precondition: ``a.pair == b.pair`` — the same subject *and* the same
        reference, i.e. the same pair measured twice (repeat run, determinism
        check) or in two scenario shells (REV-5: the pair key already encodes
        the full canonical identity — source + params + strategy — so the
        shells may differ).  Results of *different* pairs are not comparable
        even when they share a subject or a reference: ``agreement`` of
        ``(S, R1)`` and ``(S, R2)`` is undefined, and the consistency of the
        references themselves is what the *symmetric mode* is for — a
        symmetric group over ``{R1, R2}`` already yields agreement numbers
        by construction.  Violating either precondition — pair identity or
        the per-record ``grid_hash`` below — raises ``GridMismatchError``:
        the operation is defined only on comparable results, a number
        computed otherwise would be a lie.
        Records present on only one side are reported in ``skipped`` —
        never silently dropped (N6).
        """
        if a.pair != b.pair:
            msg = (
                "agreement requires two results of the same scenario pair; "
                f"got {a.pair} vs {b.pair}"
            )
            raise GridMismatchError(msg)
        index_b = {(m.characteristic, m.space, m.region): m for m in b.metrics}
        entries: list[AgreementEntry] = []
        skipped: list[str] = []
        for metric_a in a.metrics:
            metric_b = index_b.get((metric_a.characteristic, metric_a.space, metric_a.region))
            if metric_b is None:
                skipped.append(
                    f"{metric_a.characteristic}/{metric_a.space}/{metric_a.region} "
                    "has no counterpart on side b"
                )
                continue
            if metric_a.grid_hash != metric_b.grid_hash:
                msg = (
                    f"grid_hash mismatch for {metric_a.characteristic}/"
                    f"{metric_a.space}/{metric_a.region}: the two sides were "
                    f"measured on different grids; agreement is defined only "
                    f"on equal grids"
                )
                raise GridMismatchError(msg)
            # Diffs cover the accuracy metrics *and* the diagnostics of the
            # record (ред. 46): both are floats attached to the same
            # (characteristic, space, region) key under the same grid-hash
            # precondition.  Key namespaces are disjoint by construction.
            values_a = {**metric_a.metrics, **metric_a.diagnostics}
            values_b = {**metric_b.metrics, **metric_b.diagnostics}
            common = sorted(set(values_a) & set(values_b))
            skipped.extend(
                f"{metric_a.characteristic}/{metric_a.space}/{metric_a.region}: "
                f"metric {key!r} present on one side only"
                for key in sorted(set(values_a) ^ set(values_b))
            )
            diffs = {key: values_a[key] - values_b[key] for key in common}
            entries.append(
                AgreementEntry(
                    characteristic=metric_a.characteristic,
                    space=metric_a.space,
                    region=metric_a.region,
                    grid_hash=metric_a.grid_hash,
                    diffs=diffs,
                )
            )
        a_keys = {(m.characteristic, m.space, m.region) for m in a.metrics}
        skipped.extend(
            f"{key[0]}/{key[1]}/{key[2]} has no counterpart on side a"
            for key in sorted(set(index_b) - a_keys)
        )
        return Agreement(
            side_a=a.subject_id,
            side_b=b.subject_id,
            entries=tuple(entries),
            skipped=tuple(skipped),
        )

    # -- internals -------------------------------------------------------------


def _spread_result(result: AccuracyResult, violated: set[str]) -> AccuracyResult:
    """Re-evaluate one result against the group's violated participants."""
    hit = violated & set(result.pair)
    if not hit or result.status != Status.MEASURED:
        return result
    notes = (*result.validator_notes, _violation_note(hit))
    violations = tuple(v for v in result.invariant_violations if v.participant_id not in hit)
    violations = violations + tuple(
        Violation(
            participant_id=pid,
            code=VIOLATION_CODE,
            detail="spread from the group invariant verdict",
        )
        for pid in sorted(hit)
        if all(v.participant_id != pid for v in result.invariant_violations)
    )
    return replace(
        result,
        status=Status.INVALID_REFERENCE,
        status_explanation=(
            f"{result.status_explanation}; invariant check disproved participant(s) {sorted(hit)}"
        ),
        validator_notes=notes,
        invariant_violations=violations,
    )


def _violation_note(hit: set[str]) -> ValidatorNote:
    return ValidatorNote(
        code="invariant_violation_spread",
        detail=(
            "invariant verdict spread inside the group to participants: " + ", ".join(sorted(hit))
        ),
    )
