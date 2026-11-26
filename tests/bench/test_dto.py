"""JSON serialization shapes of the result DTOs (D2, REV-5/REV-20).

Every DTO claims to be JSON-serializable through ``to_json_dict()``; the
store, the report artifacts and any external consumer rely on that claim.
These tests pin the key layout (a renamed key is a schema change, not an
implementation detail) and the round trip through the store's strict-JSON
boundary.
"""

from __future__ import annotations

import json

import numpy as np

from pysatl_core_bench import (
    AccuracyResult,
    ComparisonTable,
    MetricRecord,
    PointwiseRef,
    Provenance,
    SpecialPoint,
    Status,
    SupportVerdict,
    SweepCurve,
    SweepPoint,
    Violation,
)
from pysatl_core_bench.accuracy.dto import Agreement, AgreementEntry, PointwiseSegment
from pysatl_core_bench.accuracy.types import Readiness


def test_support_verdict_serializes_readiness_and_missing() -> None:
    payload = SupportVerdict(
        readiness=Readiness.NOT_COVERED,
        reason="hole in the table",
        missing=(1.5, 2.5),
    ).to_json_dict()
    assert payload == {
        "readiness": "not_covered",
        "reason": "hole in the table",
        "missing": [1.5, 2.5],
    }


def test_violation_serializes_participant_code_detail() -> None:
    payload = Violation(
        participant_id="p#abc", code="invariant_violation", detail="boom"
    ).to_json_dict()
    assert payload == {
        "participant_id": "p#abc",
        "code": "invariant_violation",
        "detail": "boom",
    }


def test_metric_record_serializes_scale_and_counters() -> None:
    record = MetricRecord(
        characteristic="cdf",
        output_scale="survival",
        region="tail",
        grid_hash="sha256:h",
        metrics={"abs_error": 0.0},
        n_missing_reference=1,
        n_compared=2,
        n_excluded=3,
        n_invalid_subject=4,
    )
    payload = record.to_json_dict()
    assert payload["output_scale"] == "survival"
    assert payload["n_missing_reference"] == 1
    assert payload["n_excluded"] == 3
    assert payload["n_invalid_subject"] == 4
    assert payload["metrics"] == {"abs_error": 0.0}
    assert payload["excluded_points"] == []  # no exclusions: an empty dictionary, not null


def test_excluded_point_serializes_coordinate_and_reason() -> None:
    # AP-1 dictionary entry: (coordinate, reason tag) per inside-support
    # point without a comparable value; the tags name the counters.
    from pysatl_core_bench.accuracy import ExcludedPoint

    payload = ExcludedPoint(x=1.5, reason="n_missing_reference").to_json_dict()
    assert payload == {"x": 1.5, "reason": "n_missing_reference"}

    record = MetricRecord(
        characteristic="cdf",
        output_scale="probability",
        region="bulk",
        grid_hash="sha256:h",
        n_missing_reference=1,
        excluded_points=(ExcludedPoint(x=1.5, reason="n_missing_reference"),),
    )
    payload = record.to_json_dict()
    assert payload["excluded_points"] == [{"x": 1.5, "reason": "n_missing_reference"}]
    assert json.dumps(payload)  # strict-JSON safe


def test_pointwise_ref_serializes_segments() -> None:
    ref = PointwiseRef(
        uri="sidecar:abc",
        hash="sha256:abc",
        output_scale="probability",
        n_points=2,
        segments=(PointwiseSegment("cdf", "bulk", "probability", 0, 2),),
    )
    payload = ref.to_json_dict()
    assert payload["uri"] == "sidecar:abc"
    assert payload["output_scale"] == "probability"
    assert payload["segments"] == [
        {
            "characteristic": "cdf",
            "region": "bulk",
            "output_scale": "probability",
            "start": 0,
            "stop": 2,
        }
    ]


def test_provenance_serializes_optional_fields() -> None:
    payload = Provenance(library="scipy", library_version="1.15").to_json_dict()
    assert payload["library"] == "scipy"
    assert payload["path_dump"] is None
    assert payload["table_hash"] is None
    assert payload["conversion_facts"] == []


def test_special_point_serializes_the_manifest() -> None:
    payload = SpecialPoint(x=0.0, kind="pdf_jump", source="curated_rules").to_json_dict()
    assert payload == {"x": 0.0, "kind": "pdf_jump", "source": "curated_rules"}


def test_sweep_point_and_curve_serialize_axis_and_metrics() -> None:
    point = SweepPoint(
        sort_key=1.0,
        pair=("a", "b"),
        characteristic="cdf",
        output_scale="probability",
        region="bulk",
        status=Status.MEASURED,
        metrics={"lre": 14.0},
    )
    payload = point.to_json_dict()
    assert payload["sort_key"] == 1.0
    assert payload["status"] == "measured"
    assert payload["output_scale"] == "probability"

    curve = SweepCurve(comparison_group_id="g", axis=(0.0, 1.0), points=(point,))
    curve_payload = curve.to_json_dict()
    assert curve_payload["axis"] == [0.0, 1.0]
    assert curve_payload["points"][0]["metrics"] == {"lre": 14.0}


def test_comparison_table_serializes_all_rows() -> None:
    row = AccuracyResult(
        schema_version="1.1",
        scenario_id="s",
        spec_hash="sha256:x",
        comparison_group_id="g",
        sort_key=None,
        pair=("a", "b"),
        subject_id="a",
        reference="b",
        role_labels={"subject": "a", "reference": "b"},
        status=Status.MEASURED,
        status_explanation="ok",
    )
    payload = ComparisonTable(comparison_group_id="g", rows=(row,)).to_json_dict()
    assert payload["comparison_group_id"] == "g"
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["status"] == "measured"
    # strict JSON: the store persists exactly this structure
    json.dumps(payload, allow_nan=False)


def test_agreement_entries_carry_scales_and_skips() -> None:
    entry = AgreementEntry(
        characteristic="cdf",
        output_scale="survival",
        region="tail",
        grid_hash="sha256:h",
        diffs={"abs_error": 0.5},
    )
    agreement = Agreement(side_a="a", side_b="b", entries=(entry,), skipped=("x/lre/y",))
    assert agreement.entries[0].output_scale == "survival"
    assert agreement.skipped == ("x/lre/y",)
    assert np.isfinite(entry.diffs["abs_error"])
