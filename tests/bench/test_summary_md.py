"""SUMMARY.md table contract (ред. 55): matching sizes of names and data.

The raw text is a primary reading mode of the run summary, so the plain-text
table must not be ragged: the width of every column fits both its header
name («название») and every data cell implementing that name, all cells of
a column carry the same size, and every line of a table has the same
length.  Markdown renderers see the same table as before.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pysatl_core_bench import AccuracyResult, Status
from pysatl_core_bench.accuracy.dto import MetricRecord, ValidatorNote

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # for the ``benchmarks`` namespace import below
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "run_accuracy_under_test", ROOT / "benchmarks" / "run_accuracy.py"
)
assert _spec is not None and _spec.loader is not None  # loader contract of importlib
_run_accuracy: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_accuracy)


def test_wide_data_cell_widens_the_name_column() -> None:
    """A data cell wider than its header name sets the column width."""
    lines = _run_accuracy._render_md_table(["status"], [["unsupported_configuration"]])
    assert lines == [
        "| status                    |",
        "| ------------------------- |",
        "| unsupported_configuration |",
    ]
    assert len({len(line) for line in lines}) == 1


def test_wide_name_cell_widens_the_data_cells() -> None:
    """A header name wider than its data pads the implementing cells."""
    lines = _run_accuracy._render_md_table(["status"], [["measured"]])
    assert lines == ["| status   |", "| -------- |", "| measured |"]


def test_every_line_of_a_table_has_one_length() -> None:
    """Header, delimiter and data rows share the total width."""
    lines = _run_accuracy._render_md_table(["a", "bc"], [["d", "e"]])
    assert lines == ["| a | bc |", "| - | -- |", "| d | e  |"]
    assert len({len(line) for line in lines}) == 1


def test_table_without_rows_keeps_header_widths() -> None:
    """An empty section still renders a valid, aligned table skeleton."""
    lines = _run_accuracy._render_md_table(["scenario", "notes"], [])
    assert lines == ["| scenario | notes |", "| -------- | ----- |"]
    assert len({len(line) for line in lines}) == 1


def _result(
    scenario_id: str,
    notes: tuple[ValidatorNote, ...],
    abs_error: float,
    lre: float,
) -> AccuracyResult:
    """One measured pair of the demo group with worst-region metrics."""
    return AccuracyResult(
        schema_version="1.0",
        scenario_id=scenario_id,
        spec_hash="a" * 12,
        comparison_group_id="demo-group",
        sort_key=None,
        pair=("pysatl-core#aaaaaaaaaaaa", "scipy-norm#bbbbbbbbbbbb"),
        subject_id="pysatl-core#aaaaaaaaaaaa",
        reference="scipy-norm#bbbbbbbbbbbb",
        role_labels={},
        status=Status.MEASURED,
        status_explanation="",
        validator_notes=notes,
        metrics=(
            MetricRecord(
                characteristic="pdf",
                space="probability",
                region="bulk",
                grid_hash="g1",
                metrics={"abs_error": abs_error, "lre": lre},
            ),
        ),
    )


def test_write_summary_md_names_and_data_share_column_width(tmp_path: Path) -> None:
    """The written summary is column-aligned in plain-text form."""
    results = [
        _result("S1.1-normal-pdf", (), 5.551e-17, 15.66),
        _result(
            "NEG2-degenerate-selfpair",
            (ValidatorNote("degenerate_comparison", "identity of both sides matches"),),
            2.220446049250313e-16,
            16.0,
        ),
    ]
    path = _run_accuracy._write_summary_md(
        tmp_path, "smoke", [("S1.1_normal_pdf.json", "demo-group", results)], []
    )
    text = path.read_text(encoding="utf-8")
    table_lines = [line for line in text.splitlines() if line.startswith("|")]
    assert len(table_lines) == 4  # header, delimiter, two data rows
    assert len({len(line) for line in table_lines}) == 1  # one length per table

    cells = [[c.strip() for c in line[1:-1].split("|")] for line in table_lines]
    assert cells[0] == [
        "scenario",
        "subject",
        "reference",
        "status",
        "abs_err (max)",
        "LRE (min)",
        "notes",
    ]
    # delimiter mirrors the whole-column width: header name and every data cell
    assert cells[1] == [
        "-" * max(len(cells[0][i]), len(cells[2][i]), len(cells[3][i])) for i in range(7)
    ]
    assert cells[2] == [
        "S1.1-normal-pdf",
        "pysatl-core#aaaaaaaaaaaa",
        "scipy-norm#bbbbbbbbbbbb",
        "measured",
        "5.551e-17",
        "15.66",
        "-",
    ]
    assert cells[3] == [
        "NEG2-degenerate-selfpair",
        "pysatl-core#aaaaaaaaaaaa",
        "scipy-norm#bbbbbbbbbbbb",
        "measured",
        "2.220e-16",
        "16.00",
        "degenerate_comparison",
    ]
    assert "## Rejected files" not in text


def test_write_summary_md_keeps_rejected_section_bullets(tmp_path: Path) -> None:
    """Rejected files stay a bullet list, outside the aligned tables."""
    path = _run_accuracy._write_summary_md(
        tmp_path, "all", [], [("bad.json", "not JSON:Expecting value")]
    )
    text = path.read_text(encoding="utf-8")
    assert "| " not in text  # no table was rendered
    assert "## Rejected files" in text
    assert "- bad.json: not JSON:Expecting value" in text


@pytest.mark.parametrize(
    ("headers", "rows"),
    [
        (["scenario", "subject", "reference", "status", "abs_err (max)", "LRE (min)", "notes"], []),
        (["scenario"], [["s1"], ["chain-normal-cdf-sigma/sigma-0.5"]]),
    ],
)
def test_render_invariants_hold_on_summary_shape(headers: list[str], rows: list[list[str]]) -> None:
    """Widths follow the widest cell; no line ends or starts ragged."""
    lines = _run_accuracy._render_md_table(headers, rows)
    assert len(lines) == len(rows) + 2  # header and delimiter always present
    assert len({len(line) for line in lines}) == 1
    assert all(line.startswith("| ") and line.endswith(" |") for line in lines)
    widths = [max([len(h), *[len(row[i]) for row in rows]]) for i, h in enumerate(headers)]
    for i, header in enumerate(headers):
        column = [line[1:-1].split("|")[i].strip() for line in lines]
        assert column[0] == header
        assert column[1] == "-" * widths[i]
        assert column[2:] == [row[i] for row in rows]
