"""Canonical forms and hashes (B1.1/C2.1, D1)."""

from __future__ import annotations

import numpy as np
import pytest

from pysatl_core_bench import ScenarioRejected, canonical_json, content_hash, grid_hash, values_hash


def test_canonical_json_is_key_order_independent() -> None:
    a = {"b": 1, "a": {"d": 2.0, "c": [3, 4]}}
    b = {"a": {"c": [3, 4], "d": 2.0}, "b": 1.0}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_normalizes_int_and_float() -> None:
    assert canonical_json({"x": 1}) == canonical_json({"x": 1.0})


def test_canonical_json_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": float("nan")})


def test_content_hash_stability() -> None:
    h1 = content_hash({"a": [1.0, 2.0], "b": "text"})
    h2 = content_hash({"b": "text", "a": [1, 2.0]})
    assert h1 == h2 and h1.startswith("sha256:")


def test_grid_hash_is_order_independent_and_float_exact() -> None:
    pts = np.linspace(-2.0, 2.0, 33)
    assert grid_hash(pts) == grid_hash(pts[::-1].copy())
    other = np.linspace(-2.0, 2.0, 33)
    other[5] = np.nextafter(other[5], 3.0)
    assert grid_hash(pts) != grid_hash(other)


def test_values_hash_reflects_order() -> None:
    a = values_hash([1.0, 2.0, 3.0])
    b = values_hash([3.0, 2.0, 1.0])
    assert a != b


def test_grid_hash_accepts_missing_markers() -> None:
    assert values_hash([1.0, float("nan")]) == values_hash([1.0, float("nan")])


def test_unknown_scenario_field_is_rejected() -> None:
    from tests.bench.conftest import load

    raw = {
        "schema_version": "1.0",
        "comparison_group": "g",
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "cdf"}],
        "totally_unknown": 1,
    }
    with pytest.raises(ScenarioRejected, match="unknown scenario fields"):
        load(raw)
