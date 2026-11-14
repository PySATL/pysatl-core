"""Optional file persistence of the results store."""

from __future__ import annotations

import json
from pathlib import Path

from pysatl_core_bench import ResultsStore, SourceRegistry
from tests.bench.conftest import cdf_scenario_raw, run_scenario


def test_results_reports_and_sidecars_are_persisted(
    tmp_path: Path, registry_with_table: SourceRegistry
) -> None:
    root = tmp_path / "out"
    results = run_scenario(registry_with_table, cdf_scenario_raw(), ResultsStore(root=root))
    group = results[0].comparison_group_id

    group_dir = root / group
    files = list(group_dir.glob("result-*.json"))
    assert files, "pair results must be mirrored to files"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["spec_hash"] == results[0].spec_hash

    report_file = root / f"report-{group}.json"
    assert report_file.exists()
    assert json.loads(report_file.read_text(encoding="utf-8"))["summary"]["measured"] == 1

    sidecars = list(root.glob("sidecar-*.json"))
    assert sidecars, "pointwise values must be mirrored to sidecar files"
    sidecar = json.loads(sidecars[0].read_text(encoding="utf-8"))
    assert sidecar["space"] == "probability"
    assert len(sidecar["values"]) == len(results[0].y_true or ())


def test_read_group_roundtrip_is_independent_of_files(
    tmp_path: Path, registry_with_table: SourceRegistry
) -> None:
    store = ResultsStore(root=tmp_path / "out2")
    results = run_scenario(registry_with_table, cdf_scenario_raw(), store)
    group = results[0].comparison_group_id
    assert store.read_group(group) == results
