"""Run accuracy scenarios from ``benchmarks/scenarios`` against pysatl-core.

Every run persists its artifacts (default storage: ``benchmark_results/run-
<timestamp>/`` in the repository root, override with ``--out DIR``) and
renders a human-readable ``SUMMARY.md`` at the storage root — the single
entry point to read a run: one column-aligned table per scenario file, one
row per participant pair.  The console stays a brief progress log; the full
picture lives in the storage.

Two commands (from the repository root; poetry installs both packages):

    # smoke mode — quick pipeline health check with expected outcomes:
    poetry run python benchmarks/run_accuracy.py --smoke

    # run every scenario file in benchmarks/scenarios/:
    poetry run python benchmarks/run_accuracy.py --all

    # additional controls:
    poetry run python benchmarks/run_accuracy.py --smoke --out /tmp/out
    poetry run python benchmarks/run_accuracy.py --all S1.4   # + substring filter
    poetry run python benchmarks/run_accuracy.py --list       # what would run

The registry carries three kinds of reference sides: a runnable scipy
wrapper (``reference_sources.scipy_norm_source``), its on-demand caching
variant, and the mpmath-generated production tables
(``reference_tables/`` loaded as ``PrecomputedTable``).

Smoke mode (08_smoke_scenarios §1): the canonical smoke set — the
repository scenarios plus the drop-in smoke files (S1.0 minimal,
S1.6/S1.7/S3.1, NEG1–NEG5) — is run and *verified against expectations*:
positive files must come out ``measured`` with per-region metrics, each
negative file must show exactly its expected honest status (NEG2 also
carries the ``degenerate_comparison`` validator note).  A file rejected
at expansion/load is a smoke failure (§1.4).  Exit code: 0 — pipeline
healthy; 2 — an expectation is violated (measurement broken).  Without
``--smoke`` the batch prints reports and applies the А-15 exit rule:
``failed``/``invalid_reference`` → 2, honest outcomes → 0.

Minimal scenario files (ред. 35): ``schema_version``, ``comparison_group``
and ``scenario_id`` are optional — a file that declares none of them gets
its file stem injected as the comparison group here, before expansion.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # REV-5: script and module invocations both work
    from benchmarks.reference_sources import mpmath_table_sources, scipy_norm_source
except ModuleNotFoundError:  # pragma: no cover - script invocation
    # As a plain script, sys.path[0] is benchmarks/ itself, so the package
    # prefix is unavailable; resolve the sibling module dynamically instead
    # of duplicating the source catalogue here.
    import importlib

    _reference_sources: Any = importlib.import_module("reference_sources")
    mpmath_table_sources = _reference_sources.mpmath_table_sources
    scipy_norm_source = _reference_sources.scipy_norm_source

from pysatl_core_bench import (
    AccuracyResult,
    AccuracyRunner,
    OnDemandComputer,
    PySATLAdapter,
    ResultsStore,
    SourceRegistry,
    Status,
)
from pysatl_core_bench.accuracy import reset_core_state
from pysatl_core_bench.accuracy.errors import FrameworkDefect, ScenarioRejected

SCENARIO_DIR = Path(__file__).parent / "scenarios"

# Canonical smoke set (08_smoke_scenarios §3) → expected pair status.
# Every file of the set participates in ``--smoke``; the expectation is
# checked against every result row of the file's group.
SMOKE_EXPECTATIONS: dict[str, Status] = {
    "S1.0_minimal.json": Status.MEASURED,
    "S1.1_normal_pdf.json": Status.MEASURED,
    "S1.2_normal_cdf.json": Status.MEASURED,
    "S1.3_normal_pdf_from_cf.json": Status.MEASURED,
    "S1.4_normal_pdf_mpmath_table.json": Status.MEASURED,
    "S1.5_normal_cdf_mpmath_table.json": Status.MEASURED,
    "S1.6_normal_ppf.json": Status.MEASURED,
    "S1.7_normal_cdf_survival.json": Status.MEASURED,
    "S3.1_symmetric_agreement.json": Status.MEASURED,
    "sweep_normal_cdf_npoints.json": Status.MEASURED,
    "chain_normal_cdf_sigma.json": Status.MEASURED,
    "NEG1_unknown_strategy.json": Status.UNSUPPORTED_CONFIGURATION,
    "NEG2_degenerate_selfpair.json": Status.MEASURED,
    "NEG3_table_wrong_characteristic.json": Status.NOT_APPLICABLE,
    "NEG4_missing_reference.json": Status.INVALID_REFERENCE,
    "NEG5_subject_unregistered.json": Status.UNSUPPORTED_CONFIGURATION,
}


def build_registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(PySATLAdapter())
    registry.register(scipy_norm_source())
    registry.register(OnDemandComputer(scipy_norm_source(), source_id="on-demand-scipy-norm"))
    for table_source in mpmath_table_sources():
        if table_source.source_id in registry:
            # REV-5 (no silence): two tables of one characteristic (e.g.
            # sigma=1 and sigma=2 files) would silently replace each other.
            msg = f"duplicate source_id {table_source.source_id!r} in the reference tables"
            raise FrameworkDefect(msg)
        registry.register(table_source)
    return registry


def _check_smoke_expectation(path_name: str, results: list[Any], rejected: str | None) -> list[str]:
    """Verify one smoke file against its expected outcome (§1 criteria)."""
    expected = SMOKE_EXPECTATIONS[path_name]
    problems: list[str] = []
    if rejected is not None:
        problems.append(f"rejected at load (§1.4): {rejected}")
        return problems
    if not results:
        problems.append("no results produced")
        return problems
    for row in results:
        if row.status is not expected:
            problems.append(
                f"{row.scenario_id}/{row.subject_id} vs {row.reference}: "
                f"expected {expected.value}, got {row.status.value}"
            )
        if expected is Status.MEASURED and not row.metrics:
            problems.append(f"{row.scenario_id}: measured but no per-region metrics")
    if path_name == "NEG2_degenerate_selfpair.json":
        codes = {note.code for row in results for note in row.validator_notes}
        if "degenerate_comparison" not in codes:
            problems.append("NEG2: validator note degenerate_comparison is missing")
    return problems


def _pair_brief(result: AccuracyResult) -> tuple[str, str]:
    """Worst-region abs error and LRE of one pair, formatted for the summary."""
    abs_values = [
        float(m["abs_error"])
        for record in result.metrics
        for m in [record.metrics]
        if "abs_error" in m
    ]
    lre_values = [
        float(m["lre"]) for record in result.metrics for m in [record.metrics] if "lre" in m
    ]
    abs_text = f"{max(abs_values):.3e}" if abs_values else "-"
    lre_text = f"{min(lre_values):.2f}" if lre_values else "-"
    return abs_text, lre_text


def _render_md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    """Render one markdown table whose plain-text form is column-aligned.

    The width of every column fits both its header cell ("название") and
    every data cell implementing that name — the largest of them, so the
    cells of one column carry matching sizes and every line of the table
    has the same length.  Renderers see the same table as before; the raw
    text stops being ragged.
    """
    widths = [
        max([len(header), *[len(row[i]) for row in rows]]) for i, header in enumerate(headers)
    ]

    def _line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return "| " + " | ".join(padded) + " |"

    lines = [_line(headers), _line(["-" * width for width in widths])]
    lines.extend(_line(row) for row in rows)
    return lines


def _write_summary_md(
    out_root: Path,
    mode: str,
    file_reports: list[tuple[str, str, list[AccuracyResult]]],
    rejected: list[tuple[str, str]],
) -> Path:
    """Render the human-readable entry point of one run into SUMMARY.md.

    One section per scenario file, one row per participant pair; statuses,
    worst-region metrics and validator notes are all resolved here, so the
    reader never has to open the per-pair JSON artifacts.  Tables are
    column-aligned (ред. 55): header names and the data implementing them
    share the column width, so the summary reads as a table in plain text.
    """
    lines: list[str] = [
        f"# Accuracy run — {mode}",
        "",
        f"- finished: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"- scenario files run: {len(file_reports)}, rejected: {len(rejected)}",
        "",
    ]
    headers = [
        "scenario",
        "subject",
        "reference",
        "status",
        "abs_err (max)",
        "LRE (min)",
        "notes",
    ]
    for file_name, group, results in file_reports:
        lines.append(f"## {file_name} — group `{group}`")
        lines.append("")
        rows: list[list[str]] = []
        for r in results:
            abs_text, lre_text = _pair_brief(r)
            notes = ", ".join(note.code for note in r.validator_notes) or "-"
            rows.append(
                [
                    r.scenario_id,
                    r.subject_id,
                    r.reference,
                    r.status.value,
                    abs_text,
                    lre_text,
                    notes,
                ]
            )
        lines.extend(_render_md_table(headers, rows))
        lines.append("")
    if rejected:
        lines.append("## Rejected files")
        lines.append("")
        for file_name, reason in rejected:
            lines.append(f"- {file_name}: {reason}")
        lines.append("")
    summary_path = out_root / "SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filter",
        nargs="?",
        default="",
        help="substring filter for scenario file names (works with --all/--smoke)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="smoke mode: run the canonical smoke set and verify expected statuses",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="run every scenario file (the default without --smoke)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "directory to persist results (default: benchmark_results/run-<timestamp>; "
            "an explicit --out DIR is used as-is)"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the scenario files that would run and exit",
    )
    args = parser.parse_args()

    if args.smoke:
        files = [
            p
            for p in sorted(SCENARIO_DIR.glob("*.json"))
            if p.name in SMOKE_EXPECTATIONS and args.filter in p.name
        ]
    else:
        files = sorted(p for p in SCENARIO_DIR.glob("*.json") if args.filter in p.name)

    if args.list:
        for path in files:
            mark = "smoke " if path.name in SMOKE_EXPECTATIONS else "extra "
            print(f"{mark}{path.name}")
        print(f"{len(files)} file(s) in {SCENARIO_DIR}")
        return 0
    if not files:
        print(f"no scenarios matching {args.filter!r} in {SCENARIO_DIR}")
        return 1

    # Every run persists: default storage accumulates timestamped run dirs
    # under benchmark_results/, an explicit --out DIR is used as-is.
    if args.out is not None:
        out_root = Path(args.out)
    else:
        out_root = Path("benchmark_results") / f"run-{datetime.now():%Y%m%d-%H%M%S}"
    out_root.mkdir(parents=True, exist_ok=True)

    registry = build_registry()

    # Prepare is orchestrated by the runner (R41-1): the CLI only reads the
    # file, applies the minimal-file naming rule (file stem as the group)
    # and hands the raw dict to run_group_raw — expansion, loading and the
    # whole measurement cycle live inside the runner.  A ScenarioRejected
    # (from expansion or load) is a scenario defect: reported and skipped,
    # it does not abort the batch and does not flip the exit code
    # (08_smoke_scenarios §1.5, REV-5).
    broken = False
    smoke_failures: list[tuple[str, list[str]]] = []
    file_reports: list[tuple[str, str, list[AccuracyResult]]] = []
    rejected_files: list[tuple[str, str]] = []
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"=== {path.name}: REJECTED (not JSON): {exc}")
            if args.smoke:
                smoke_failures.append((path.name, [f"not JSON: {exc}"]))
            rejected_files.append((path.name, f"not JSON: {exc}"))
            continue
        # Minimal-file rule (ред. 35): a file that declares neither
        # scenario_id nor comparison_group is named by its file stem.
        if "scenario_id" not in raw and "comparison_group" not in raw:
            raw["comparison_group"] = path.stem
        # One store (and one runner) per file; the scenarios of the file run
        # as one group (REV-16): the store is written once, after the whole
        # group's invariant spread, so rows and report agree by construction.
        store = ResultsStore(root=out_root)
        runner = AccuracyRunner(registry, store)
        try:
            # R41-1: expansion and loading happen inside the runner; the
            # raw dict never leaves the Prepare pipeline, and there is no
            # buffer of expanded variants between expander and loader.
            results = runner.run_group_raw(raw, before_scenario=reset_core_state)
        except ScenarioRejected as exc:
            print(f"=== {path.name}: REJECTED at expansion/load: {exc}")
            if args.smoke:
                smoke_failures.append((path.name, [f"rejected at expansion/load: {exc}"]))
            rejected_files.append((path.name, f"rejected at expansion/load: {exc}"))
            continue
        group = raw.get("comparison_group") or raw.get("scenario_id") or path.stem
        statuses = ", ".join(
            f"{value}×{count}"
            for value, count in sorted(Counter(r.status.value for r in results).items())
        )
        print(f"=== {path.name}: group {group} — {len(results)} pair(s): {statuses}")
        file_reports.append((path.name, group, results))
        if args.smoke:
            problems = _check_smoke_expectation(path.name, results, None)
            verdict = "SMOKE PASS" if not problems else "SMOKE FAIL"
            print(f"    [{verdict}] expected {SMOKE_EXPECTATIONS[path.name].value}")
            for problem in problems:
                print(f"    - {problem}")
            if problems:
                smoke_failures.append((path.name, problems))
        else:
            broken = broken or any(
                r.status in (Status.FAILED, Status.INVALID_REFERENCE) for r in results
            )
        report = store.read_report(group)
        if report is None:  # pragma: no cover - runner always reports
            msg = f"no report stored for group {group!r}"
            raise FrameworkDefect(msg)

    # The console stays a progress log; the full human-readable picture is
    # the SUMMARY.md at the storage root (REV-21).
    summary_path = _write_summary_md(
        out_root, "smoke" if args.smoke else "all", file_reports, rejected_files
    )

    if args.smoke:
        if smoke_failures:
            print(f"SMOKE: {len(smoke_failures)} file(s) FAILED the smoke criteria:")
            for name, problems in smoke_failures:
                for problem in problems:
                    print(f"  - {name}: {problem}")
            print(f"Results: {out_root}{os.sep}  Summary: {summary_path}")
            return 2
        print(f"SMOKE: OK — {len(files)} file(s), all expected outcomes confirmed.")
        print(f"Results: {out_root}{os.sep}  Summary: {summary_path}")
        return 0
    print(f"Results: {out_root}{os.sep}  Summary: {summary_path}")
    return 2 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
