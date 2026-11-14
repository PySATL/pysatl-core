"""Report stage: every status reaches the report (N6).

The reporter takes the comparator's table (never the raw store view), so
the spread invariant verdicts are visible; it renders a deterministic text
report (no timestamps — accuracy must stay bit-identical) and stores it.
The per-row summary column carries the *worst* region (max abs error
policy, min LRE), the same convention as the CLI summary (REV-21:
"abs_err max, LRE min") — a pair is as good as its worst region.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core_bench.accuracy.dto import ComparisonTable, Report, SweepCurve
from pysatl_core_bench.accuracy.reporting.compare import Comparator
from pysatl_core_bench.accuracy.reporting.store import ResultsStore
from pysatl_core_bench.accuracy.types import Status

__all__ = ["Reporter"]

_ROW_FMT = "{:<14} {:<34} {:<24} {:>12}"


class Reporter:
    """Renders the report of one comparison group into the store."""

    def __init__(self, store: ResultsStore) -> None:
        self._store = store

    def report(self, group_id: str) -> Report:
        """Build, render and store the report; every row is included."""
        comparator = Comparator(self._store)
        table: ComparisonTable = comparator.table(group_id)
        curve = comparator.curve(group_id)
        summary = self._summary(table)
        text = self._render(table, summary, curve)
        report = Report(
            comparison_group_id=group_id,
            rows=table.rows,
            summary=summary,
            text=text,
            sweep=curve if curve.axis else None,
        )
        self._store.write_report(report)
        return report

    @staticmethod
    def _summary(table: ComparisonTable) -> dict[str, int]:
        summary = {status.value: 0 for status in Status}
        for row in table.rows:
            summary[row.status.value] += 1
        return summary

    @staticmethod
    def _render(table: ComparisonTable, summary: dict[str, int], curve: SweepCurve | None) -> str:
        lines: list[str] = [
            f"Accuracy report: {table.comparison_group_id}",
            f"results: {len(table.rows)}; statuses: "
            + ", ".join(f"{k}={v}" for k, v in summary.items()),
            "",
            _ROW_FMT.format("scenario", "pair", "status", "lre(min)"),
        ]
        for row in table.rows:
            if row.status == Status.MEASURED and row.metrics:
                lre_values = [m.metrics["lre"] for m in row.metrics if "lre" in m.metrics]
                lre_text = f"{min(lre_values):.3f}" if lre_values else "-"
            else:
                lre_text = "-"
            lines.append(
                _ROW_FMT.format(
                    row.scenario_id[:14],
                    " x ".join(row.pair)[:34],
                    row.status.value[:24],
                    lre_text,
                )
            )
            lines.append(f"    explanation: {row.status_explanation}")
            for note in row.validator_notes:
                lines.append(f"    note[{note.code}]: {note.detail}")
            if row.status != Status.MEASURED:
                continue
            for metric in row.metrics:
                rendered = ", ".join(f"{k}={v:.6g}" for k, v in metric.metrics.items())
                lines.append(
                    f"    {metric.characteristic}/{metric.space}/{metric.region}: "
                    f"{rendered} (n={metric.n_compared}, "
                    f"missing_ref={metric.n_missing_reference}, "
                    f"excluded={metric.n_excluded}, "
                    f"invalid_subj={metric.n_invalid_subject})"
                )
                if metric.diagnostics:
                    # Diagnostics (ред. 46) are rendered on their own line:
                    # same record, never reported as accuracy.
                    diag = ", ".join(f"{k}={v:.6g}" for k, v in metric.diagnostics.items())
                    lines.append(
                        f"    {metric.characteristic}/{metric.space}/{metric.region} "
                        f"diagnostics: {diag}"
                    )
        lines.extend(Reporter._render_curve(curve))
        return "\n".join(lines)

    @staticmethod
    def _render_curve(curve: SweepCurve | None) -> list[str]:
        """Sweep curve over sort_key (К-1): empty for non-sweep groups."""
        if curve is None or not curve.axis:
            return []
        axis_text = " -> ".join(str(key) for key in curve.axis)
        lines = [
            "",
            f"sweep curve over sort_key (axis: {axis_text})",
        ]
        current_series: tuple[str, str, str, tuple[str, str]] | None = None
        for point in curve.points:
            series = (point.characteristic, point.space, point.region, point.pair)
            if series != current_series:
                current_series = series
                lines.append(
                    f"  {point.characteristic}/{point.space}/{point.region} "
                    f"[{' x '.join(point.pair)}]:"
                )
            metrics = ", ".join(f"{k}={v:.6g}" for k, v in point.metrics.items())
            suffix = f" {metrics}" if metrics else ""
            lines.append(f"    {point.sort_key}: {point.status.value}{suffix}")
        return lines
