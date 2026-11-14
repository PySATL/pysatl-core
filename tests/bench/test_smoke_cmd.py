"""Smoke-command and persistence regressions (r35).

Pins two things that unit tests of single stages do not:

1. the canonical smoke set of ``benchmarks/run_accuracy.py`` covers every
   scenario file shipped in ``benchmarks/scenarios/`` and every smoke
   expectation is confirmed by a real end-to-end run (08_smoke_scenarios §1);
2. a persisted run (``ResultsStore`` with a root) writes plain, strictly
   parseable JSON — REV-20: nested ``mappingproxy`` of frozen-DTO dict
   fields must never leak into the artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from benchmarks.run_accuracy import (
    SCENARIO_DIR,
    SMOKE_EXPECTATIONS,
    _check_smoke_expectation,
    build_registry,
)
from pysatl_core_bench import (
    AccuracyRunner,
    ResultsStore,
)
from pysatl_core_bench.accuracy import reset_core_state


def _scenario_files() -> list[Path]:
    return sorted(SCENARIO_DIR.glob("*.json"))


def test_every_scenario_file_is_in_the_smoke_set() -> None:
    """No orphan files: every shipped scenario participates in ``--smoke``."""
    for path in _scenario_files():
        assert path.name in SMOKE_EXPECTATIONS, (
            f"{path.name} is not in SMOKE_EXPECTATIONS — a scenario shipped "
            "without an expected outcome cannot be smoked"
        )


def test_smoke_set_files_exist() -> None:
    """Every smoke expectation has its file on disk (no stale entries)."""
    on_disk = {p.name for p in _scenario_files()}
    for name in SMOKE_EXPECTATIONS:
        assert name in on_disk, f"{name} is expected by the smoke set but missing"


def _prepare(raw: dict[str, Any], stem: str) -> dict[str, Any]:
    # Minimal-file rule (ред. 35): same injection as run_accuracy.main().
    if "scenario_id" not in raw and "comparison_group" not in raw:
        raw["comparison_group"] = stem
    return raw


def _run_file(path: Path, registry: Any, store: ResultsStore) -> list[Any]:
    # R41-1: the CLI hands the raw dict to the runner; expansion and loading
    # are orchestrated inside run_group_raw (no variant buffer in the CLI).
    raw = json.loads(path.read_text(encoding="utf-8"))
    return AccuracyRunner(registry, store).run_group_raw(
        _prepare(raw, path.stem), before_scenario=reset_core_state
    )


@pytest.fixture()
def smoke_registry() -> Generator[Any, None, None]:
    reset_core_state()
    yield build_registry()
    reset_core_state()


def test_full_smoke_pipeline_confirms_every_expectation(smoke_registry: Any) -> None:
    """End-to-end §1: every smoke file reaches exactly its expected status."""
    for path in _scenario_files():
        store = ResultsStore()
        results = _run_file(path, smoke_registry, store)
        problems = _check_smoke_expectation(path.name, results, None)
        assert not problems, f"{path.name}: {problems}"


def test_persisted_artifacts_are_strict_json(smoke_registry: Any, tmp_path: Path) -> None:
    """REV-20: persisted result/report/sidecar files are plain parseable JSON.

    The PySATL plan dump carries nested ``mappingproxy`` values; a shallow
    ``dict(...)`` in ``to_json_dict`` does not unfold them, so the store's
    serialization boundary must materialize every mapping (the regression
    crashed ``json.dumps`` with ``Object of type mappingproxy is not JSON
    serializable``).
    """
    path = SCENARIO_DIR / "S1.0_minimal.json"
    store = ResultsStore(root=tmp_path)
    results = _run_file(path, smoke_registry, store)
    assert results
    files = sorted(tmp_path.rglob("*.json"))
    assert files, "persistence produced no artifacts"
    for artifact in files:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
