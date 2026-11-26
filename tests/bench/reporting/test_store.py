"""Optional file persistence of the results store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
    assert sidecar["output_scale"] == "probability"
    assert len(sidecar["values"]) == len(results[0].y_true or ())


def test_read_group_roundtrip_is_independent_of_files(
    tmp_path: Path, registry_with_table: SourceRegistry
) -> None:
    store = ResultsStore(root=tmp_path / "out2")
    results = run_scenario(registry_with_table, cdf_scenario_raw(), store)
    group = results[0].comparison_group_id
    assert store.read_group(group) == results


class TestPersistedSidecars:
    """File-backed sidecars are content-addressed and strict-JSON (REV-21)."""

    def test_put_sidecar_persists_a_payload(self, tmp_path: Path) -> None:
        store = ResultsStore(root=tmp_path)
        uri, digest = store.put_sidecar(np.asarray([1.0, 2.0]), "survival")
        assert uri == f"sidecar:{digest}"
        files = list(tmp_path.glob("sidecar-*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["output_scale"] == "survival"
        assert payload["values"] == [1.0, 2.0]

    def test_repeated_put_deduplicates_by_content(self, tmp_path: Path) -> None:
        store = ResultsStore(root=tmp_path)
        first_uri, _ = store.put_sidecar(np.asarray([1.0]), "probability")
        second_uri, _ = store.put_sidecar(np.asarray([1.0]), "probability")
        assert first_uri == second_uri
        assert len(list(tmp_path.glob("sidecar-*.json"))) == 1

    def test_get_sidecar_unknown_uri_is_none(self) -> None:
        store = ResultsStore()
        assert store.get_sidecar("sidecar:deadbeef") is None
        assert store.get_sidecar("not-a-sidecar") is None

    def test_get_sidecar_returns_a_copy(self) -> None:
        store = ResultsStore()
        uri, _ = store.put_sidecar(np.asarray([1.0, 2.0]), "probability")
        arr = store.get_sidecar(uri)
        assert arr is not None
        arr[0] = 100.0
        again = store.get_sidecar(uri)
        assert again is not None
        assert again[0] == 1.0

    def test_read_report_miss_is_none(self) -> None:
        assert ResultsStore().read_report("no-such-group") is None
