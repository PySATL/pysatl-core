"""Canonical forms and hashes (B1.1/C2.1, D1)."""

from __future__ import annotations

import numpy as np
import pytest

from pysatl_core_bench import (
    ScenarioRejected,
    canonical_json,
    content_hash,
    grid_hash,
    values_hash,
)
from pysatl_core_bench.accuracy.canonical import frozen_mapping


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
        "schema_version": "1.1",
        "comparison_group": "g",
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "cdf"}],
        "totally_unknown": 1,
    }
    with pytest.raises(ScenarioRejected, match="unknown scenario fields"):
        load(raw)


class TestCanonicalizeTypes:
    """The canonicalizer must see equivalent payloads as equal hashes.

    Numbers arrive from JSON as int/float but from numpy code paths as
    np.integer/np.floating/np.ndarray; frozen DTO fields arrive as
    ``mappingproxy``.  If any of these hashed differently from their plain
    JSON twin, cache keys and spec hashes would silently diverge.
    """

    def test_numpy_numbers_hash_like_json_numbers(self) -> None:
        assert content_hash({"x": np.int64(3)}) == content_hash({"x": 3.0})
        assert content_hash({"x": np.float64(0.5)}) == content_hash({"x": 0.5})
        # np.float64 subclasses python float, so the np.floating branch is
        # reached only through the narrow float types (float32 etc.)
        assert content_hash({"x": np.float32(0.5)}) == content_hash({"x": 0.5})
        assert content_hash({"x": np.asarray([1.0, 2.0])}) == content_hash({"x": [1.0, 2.0]})

    def test_mappingproxy_hashes_like_plain_dict(self) -> None:
        frozen = frozen_mapping({"a": 1, "b": {"c": 2.0}})
        assert content_hash(frozen) == content_hash({"a": 1, "b": {"c": 2.0}})

    def test_set_hashes_deterministically(self) -> None:
        assert content_hash({"s": {"b", "a"}}) == content_hash({"s": {"a", "b"}})

    def test_non_finite_float_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"x": float("nan")})
        with pytest.raises(ValueError, match="non-finite"):
            canonical_json({"x": float("inf")})

    def test_too_large_integer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="too large"):
            canonical_json({"x": 10**400})

    def test_unsupported_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be canonicalized"):
            canonical_json({"x": object()})

    def test_zero_signs_hash_identically(self) -> None:
        assert content_hash({"x": 0.0}) == content_hash({"x": -0.0})

    def test_grid_hash_is_order_independent_and_nan_stable(self) -> None:
        a = grid_hash(np.asarray([1.0, float("nan"), float("inf"), float("-inf")]))
        b = grid_hash([float("-inf"), float("inf"), float("nan"), 1.0])
        assert a == b

    def test_values_hash_is_nan_stable(self) -> None:
        assert values_hash([float("nan")]) == values_hash([float("nan")])

    def test_values_hash_is_order_sensitive(self) -> None:
        assert values_hash([1.0, 2.0]) != values_hash([2.0, 1.0])

    def test_frozen_mapping_is_mutation_safe(self) -> None:
        frozen = frozen_mapping({"a": {"b": 1.0}})
        with pytest.raises(TypeError):
            frozen["a"]["b"] = 2.0
