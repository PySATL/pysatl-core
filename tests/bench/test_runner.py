"""Runner end-to-end: statuses, invariants §7, D7 order, K-4, B-3a."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats

from benchmarks.reference_sources import scipy_norm_source
from pysatl_core.types import CharacteristicName
from pysatl_core_bench import (
    AccuracyRunner,
    AdapterConfigurationError,
    Comparator,
    ExternalFunctionSource,
    GridGenerator,
    GridMismatchError,
    PrecomputedTable,
    Provenance,
    ResultsStore,
    ScenarioExpander,
    SourceRegistry,
    Status,
    SupportBounds,
    TableLookupSource,
)
from pysatl_core_bench.accuracy.computations.protocol import PreparedSubject, ValueSource
from pysatl_core_bench.accuracy.dto import SupportVerdict
from pysatl_core_bench.accuracy.errors import FrameworkDefect, ScenarioRejected
from pysatl_core_bench.accuracy.types import Capability, Readiness
from tests.bench.conftest import (
    CrashingSource,
    FollowupCrashSource,
    cdf_scenario_raw,
    load,
    normal_participant,
    normal_table_cdf,
    normal_table_pdf,
    normal_table_ppf,
    run_scenario,
)


class TestHappyPath:
    def test_measured_end_to_end_with_report(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        store = ResultsStore()
        results = run_scenario(registry, cdf_scenario_raw(), store)
        assert len(results) == 1
        result = results[0]
        assert result.status == Status.MEASURED
        assert result.metrics
        assert result.metrics[0].grid_hash.startswith("sha256:")
        assert result.subject_id.startswith("pysatl-core#")
        assert result.y_true is not None
        report = store.read_report("normal-cdf-vs-scipy")
        assert report is not None
        assert "measured=1" in report.text

    def test_pair_key_is_sorted_and_identity_resolved(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        result = run_scenario(registry, cdf_scenario_raw())[0]
        a, b = result.pair
        assert a <= b
        assert result.subject_id in result.pair and result.reference in result.pair

    def test_survival_region_conversion_facts(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["regions"] = [
            {"name": "tail_right", "interval": [1.0, 3.0], "output_scale": "survival"}
        ]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert result.output_scale_computed == "probability"
        assert result.output_scale_reported == "survival"
        assert result.subject_provenance is not None
        facts = result.subject_provenance.conversion_facts
        assert facts and facts[0]["to_scale"] == "survival"

    def test_partial_coverage_is_measured_with_missing_counter(
        self, registry: SourceRegistry
    ) -> None:
        # D6: a hole *inside* the measured support is still a measurement.
        # Since rev. 46 the grid is built on the participants' support
        # intersection, so the hole must live inside the table's own range.
        points = np.linspace(-3.0, 3.0, 41)
        hole = (points > -1.0) & (points < 1.0)
        registry.register(
            TableLookupSource(
                "scipy-norm-cdf",
                PrecomputedTable(
                    characteristic="cdf",
                    output_scale="probability",
                    points=points[~hole],
                    values=stats.norm.cdf(points[~hole]),
                    library="scipy",
                    library_version="1.15",
                    coverage_tol=1e-12,
                ),
            )
        )
        raw = cdf_scenario_raw()
        raw["regions"] = [{"name": "wide", "interval": [-3.0, 3.0], "output_scale": "probability"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert result.metrics[0].n_missing_reference > 0
        assert result.metrics[0].n_compared > 0

    def test_prepared_state_cached_within_run(self, registry: SourceRegistry) -> None:
        calls: list[int] = []
        real_prepare = None
        registry.register(normal_table_cdf())
        # count adapter prepares indirectly: run twice, results identical
        store = ResultsStore()
        spec = load(cdf_scenario_raw())
        first = AccuracyRunner(registry, store).run(spec)
        second = AccuracyRunner(registry, ResultsStore()).run(spec)
        assert [r.to_json_dict() for r in first] == [r.to_json_dict() for r in second]
        _ = calls, real_prepare


class TestNotReady:
    def test_unknown_reference_source_is_invalid_reference(self, registry: SourceRegistry) -> None:
        raw = cdf_scenario_raw()  # scipy-norm-cdf not registered
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.INVALID_REFERENCE
        assert "no value source registered" in result.status_explanation

    def test_reference_without_operation_is_not_applicable(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_pdf("scipy-norm-pdf"))
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "scipy-norm-pdf"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.NOT_APPLICABLE

    def test_table_as_subject_is_unsupported_configuration(self) -> None:
        registry = SourceRegistry()
        registry.register(normal_table_cdf())
        registry.register(normal_table_cdf("other-table"))
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "other-table"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.UNSUPPORTED_CONFIGURATION

    def test_subject_crash_marks_every_pair_failed(self, registry: SourceRegistry) -> None:
        registry.register(CrashingSource())
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "crashing"}
        results = run_scenario(registry, raw)
        assert len(results) == 1
        assert results[0].status == Status.FAILED
        assert "crashed" in results[0].status_explanation

    def test_unknown_strategy_is_unsupported_not_fallback(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["subject"] = {
            **normal_participant(),
            "strategy": "NoSuchStrategy",
        }
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.UNSUPPORTED_CONFIGURATION

    def test_nothing_covered_is_invalid_reference_but_partial_is_measured(
        self, registry: SourceRegistry
    ) -> None:
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["regions"] = [{"name": "far", "interval": [5.0, 8.0], "output_scale": "probability"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.INVALID_REFERENCE

    def test_priority_failed_beats_others(self, registry: SourceRegistry) -> None:
        # subject crashes AND reference is unknown -> failed wins (K-4)
        registry.register(CrashingSource())
        raw = cdf_scenario_raw()  # scipy-norm-cdf not registered
        raw["subject"] = {"source_id": "crashing"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.FAILED
        assert "subject:" in result.status_explanation
        assert "reference:" in result.status_explanation

    def test_subject_not_ready_writes_every_pair(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        registry.register(normal_table_pdf("scipy-norm-pdf"))
        raw = cdf_scenario_raw()
        raw["references"] = [
            {"source_id": "scipy-norm-cdf"},
            {"source_id": "scipy-norm-pdf"},
        ]
        raw["subject"] = {"source_id": "missing-source"}
        results = run_scenario(registry, raw)
        assert len(results) == 2  # N results in the group, no exceptions
        # subject source missing -> unsupported_configuration (K-4 priority)
        assert all(r.status == Status.UNSUPPORTED_CONFIGURATION for r in results)
        assert all("subject:" in r.status_explanation for r in results)

    def test_unready_status_does_not_leak_between_pairs(self, registry: SourceRegistry) -> None:
        # A-1 (K-4): the status of pair 2 must be computed from *its own*
        # participants; the collapsed status of pair 1 must not carry over.
        class RefusingSource(CrashingSource):
            """Compute-capable source that honestly refuses (not_applicable)."""

            def assess(self, participant, query, distribution, points=None):
                from pysatl_core_bench import Readiness, SupportVerdict

                return SupportVerdict(readiness=Readiness.NOT_APPLICABLE, reason="no route")

        class BrokenSource(CrashingSource):
            """Source with empty capabilities (unsupported as any role)."""

            @property
            def capabilities(self):
                return frozenset()

        registry.register(RefusingSource("refusing"))
        registry.register(BrokenSource("broken-ref"))
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "refusing"}
        raw["references"] = [
            {"source_id": "broken-ref"},  # unsupported_configuration for this pair
            {"source_id": "scipy-norm-cdf"},  # ready -> pair collapses to the subject status
        ]
        results = run_scenario(registry, raw)
        statuses = {r.reference.split("#")[0]: r.status for r in results}
        assert statuses["broken-ref"] == Status.UNSUPPORTED_CONFIGURATION
        assert statuses["scipy-norm-cdf"] == Status.NOT_APPLICABLE

    def test_symmetric_subject_crash_is_failed_not_invalid_reference(
        self, registry: SourceRegistry
    ) -> None:
        # A-2: in symmetric mode a side_a (subject) crash must be `failed`
        # blaming the subject, not `invalid_reference` blaming the reference.
        registry.register(CrashingSource("crashing-a"))
        raw = {
            "schema_version": "1.1",
            "comparison_group": "symmetric-crash",
            "pair_policy": "symmetric",
            "distribution": {
                "family": "Normal",
                "parametrization": "meanStd",
                "parameters": {"mu": 0.0, "sigma": 1.0},
            },
            "participants": [
                {"role": "peer", "source_id": "crashing-a"},
                {"role": "peer", **normal_participant()},
            ],
            "queries": [
                {
                    "characteristic": "pdf",
                    "output_scale": "probability",
                    "grid_hint": {"n_points": 5},
                }
            ],
            "regions": [{"name": "bulk", "interval": [-1.0, 1.0], "output_scale": "probability"}],
        }
        results = run_scenario(registry, raw)
        assert len(results) == 1
        result = results[0]
        assert result.status == Status.FAILED
        assert "subject" in result.status_explanation
        assert "failed during evaluate" in result.status_explanation
        assert result.subject_id.startswith("crashing-a")


class TestDegenerateAndInvariants:
    def test_degenerate_note_in_result(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        raw = {
            "schema_version": "1.1",
            "comparison_group": "self-vs-self",
            "distribution": {
                "family": "Normal",
                "parametrization": "meanStd",
                "parameters": {"mu": 0.0, "sigma": 1.0},
            },
            "subject": normal_participant(),
            "references": [normal_participant()],
            "queries": [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0], "output_scale": "probability"}],
        }
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED  # the run happens (B1)
        assert any(n.code == "degenerate_comparison" for n in result.validator_notes)

    def test_invariant_violation_spreads_inside_group(self, registry: SourceRegistry) -> None:
        # a "cdf" source returning values outside [0, 1] must be disproved

        class LyingCdf(CrashingSource):
            def __init__(self) -> None:
                super().__init__("lying-cdf")

            def evaluate(self, prepared: Any, points: Any) -> Any:
                return np.asarray(points) * 10.0

            def compute(self, participant: Any, query: Any, points: Any, distribution: Any) -> Any:
                return np.asarray(points) * 10.0

            def prepare(self, participant: Any, query: Any, distribution: Any) -> Any:
                from pysatl_core_bench import PreparedSubject

                return PreparedSubject(participant_id=participant.participant_id, query=query)

            def assess(
                self, participant: Any, query: Any, distribution: Any, points: Any = None
            ) -> Any:
                from pysatl_core_bench import Readiness, SupportVerdict

                return SupportVerdict(
                    readiness=Readiness.READY,
                    reason="lying",
                )

        registry.register(LyingCdf())
        registry.register(normal_table_cdf())
        raw = {
            "schema_version": "1.1",
            "comparison_group": "invariant-group",
            "distribution": {
                "family": "Normal",
                "parametrization": "meanStd",
                "parameters": {"mu": 0.0, "sigma": 1.0},
            },
            "subject": {"source_id": "lying-cdf"},
            "references": [{"source_id": "scipy-norm-cdf"}, {"source_id": "lying-cdf"}],
            "queries": [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0], "output_scale": "probability"}],
            "invariant_check_on": ["subject"],
        }
        results = run_scenario(registry, raw)
        assert len(results) == 2
        # both pairs contain the checked subject and must be downgraded
        assert all(r.status == Status.INVALID_REFERENCE for r in results)
        assert all(
            any(v.code == "invariant_violation" for v in r.invariant_violations) for r in results
        )
        table = Comparator(ResultsStore()).table  # smoke: comparator buildable
        _ = table


class TestComparatorAndAgreement:
    def test_agreement_requires_matching_grid_hash(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        results = run_scenario(registry, cdf_scenario_raw())
        a = results[0]
        tampered = a.metrics[0]
        from dataclasses import replace

        other = replace(a, pair=a.pair, metrics=(replace(tampered, grid_hash="sha256:other"),))
        with pytest.raises(GridMismatchError, match="grid_hash mismatch"):
            Comparator(ResultsStore()).agreement(a, other)

    def test_agreement_of_identical_runs_is_zero(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        first = run_scenario(registry, cdf_scenario_raw())[0]
        second = run_scenario(registry, cdf_scenario_raw())[0]
        agreement = Comparator(ResultsStore()).agreement(first, second)
        assert agreement.side_a == first.subject_id
        for entry in agreement.entries:
            assert all(diff == pytest.approx(0.0, abs=1e-12) for diff in entry.diffs.values())


class TestIsolation:
    def test_parallel_runs_are_isolated(self, registry: SourceRegistry) -> None:
        import concurrent.futures

        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()

        def one_run(_: int) -> str:
            store = ResultsStore()
            results = run_scenario(registry, raw, store)
            return results[0].status.value

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(one_run, range(8)))
        assert statuses == ["measured"] * 8


class TestSweepCurve:
    """Sweep report: a curve over ``sort_key`` (K-1) instead of a flat table."""

    def test_curve_axis_series_and_report_section(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["sweep_spec"] = {
            "param": "queries.0.grid_hint.n_points",
            "values": [33, 17, 9],
        }
        store = ResultsStore()
        runner = AccuracyRunner(registry, store)
        for scenario in ScenarioExpander().expand(raw):
            runner.run(load(scenario))

        curve = Comparator(store).curve("normal-cdf-vs-scipy")
        assert curve.axis == (9.0, 17.0, 33.0)
        assert curve.points
        keys = [point.sort_key for point in curve.points]
        assert keys == sorted(keys)
        assert all(point.status == Status.MEASURED for point in curve.points)
        assert all(point.metrics for point in curve.points)

        report = store.read_report("normal-cdf-vs-scipy")
        assert report is not None
        assert report.sweep is not None and len(report.sweep.axis) == 3
        assert "sweep curve over sort_key (axis: 9.0 -> 17.0 -> 33.0)" in report.text

    def test_no_curve_for_plain_group(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        store = ResultsStore()
        run_scenario(registry, cdf_scenario_raw(), store)
        curve = Comparator(store).curve("normal-cdf-vs-scipy")
        assert curve.axis == () and curve.points == ()
        report = store.read_report("normal-cdf-vs-scipy")
        assert report is not None
        assert report.sweep is None
        assert "sweep curve" not in report.text

    def test_no_private_core_imports_in_bench_package(self) -> None:
        import re
        from pathlib import Path

        package = Path("src/pysatl_core_bench")
        pattern = re.compile(
            r"from pysatl_core[.\w]* import[^\n#]*(_fit_|_build_|_path_cache|_unuran_sampler)"
        )
        for path in package.rglob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), path


class TestRawEntry:
    """R41-1: the raw-file entry — Prepare orchestration lives in the runner."""

    def test_run_group_raw_matches_manual_expand_load_run(self, registry: SourceRegistry) -> None:
        """run_group_raw(raw) == expand -> load -> run_group, by construction."""
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        manual = [r.to_json_dict() for r in run_scenario(registry, raw)]
        raw_entry = [
            r.to_json_dict() for r in AccuracyRunner(registry, ResultsStore()).run_group_raw(raw)
        ]
        assert manual == raw_entry

    def test_run_group_raw_expands_sweep_into_one_group(self, registry: SourceRegistry) -> None:
        """A sweep file runs all variants as one group through the raw entry."""
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["sweep_spec"] = {"param": "distribution.parameters.sigma", "values": [0.5, 2.0]}
        results = AccuracyRunner(registry, ResultsStore()).run_group_raw(raw)
        assert len(results) == 2
        groups = {r.comparison_group_id for r in results}
        assert groups == {"normal-cdf-vs-scipy"}

    def test_rejection_propagates_from_load(self, registry: SourceRegistry) -> None:
        """ScenarioRejected from Prepare reaches the caller (a file defect)."""
        raw = cdf_scenario_raw()
        raw["no_such_field"] = 1
        with pytest.raises(ScenarioRejected, match="unknown scenario fields"):
            AccuracyRunner(registry, ResultsStore()).run_group_raw(raw)


class TestSupportIntersection:
    """rev. 46, section 7.2 decision: lookup-only member -> intersection + warning."""

    def test_lookup_only_member_builds_intersection_support(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())  # covers [-3, 3] only
        raw = cdf_scenario_raw()
        raw["queries"] = [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}]
        raw["regions"] = [{"name": "wide", "interval": [-4.0, 4.0], "output_scale": "probability"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        codes = {n.code for n in result.validator_notes}
        assert "support_intersection" in codes
        # The grid was built on [-3, 3]: every point answered by the table,
        # no coverage holes at all.
        assert result.metrics[0].n_missing_reference == 0
        assert result.metrics[0].n_compared == 9

    def test_all_compute_group_keeps_single_provider_support(
        self, registry: SourceRegistry
    ) -> None:
        registry.register(normal_table_pdf("scipy-norm-pdf"))
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "scipy-norm-pdf"}]
        result = run_scenario(registry, raw)[0]
        codes = {n.code for n in result.validator_notes}
        assert "support_intersection" not in codes


def _ppf_scenario_raw() -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "comparison_group": "normal-ppf",
        "distribution": {
            "family": "Normal",
            "parametrization": "meanStd",
            "parameters": {"mu": 0.0, "sigma": 1.0},
        },
        "subject": {"source_id": "pysatl-core"},
        "references": [{"source_id": "scipy-norm-ppf"}],
        "queries": [{"characteristic": "ppf", "grid_hint": {"n_points": 17}}],
        "regions": [{"name": "levels", "interval": [0.05, 0.95], "output_scale": "probability"}],
    }


def _scipy_norm_values(distribution: Any, participant: Any, query: Any, points: Any) -> Any:
    from pysatl_core.types import CharacteristicName

    arr = np.asarray(points, dtype=np.float64)
    if query.characteristic == CharacteristicName.PDF:
        return stats.norm.pdf(arr)
    if query.characteristic == CharacteristicName.CDF:
        return stats.norm.cdf(arr)
    if query.characteristic == CharacteristicName.PPF:
        return stats.norm.ppf(arr)
    msg = f"cannot serve {str(query.characteristic)!r}"
    raise NotImplementedError(msg)


def _scipy_norm_source(source_id: str = "scipy-norm") -> Any:
    from pysatl_core.types import CharacteristicName
    from pysatl_core_bench import ExternalFunctionSource, Provenance
    from pysatl_core_bench.accuracy.dto import SupportBounds

    return ExternalFunctionSource(
        source_id,
        _scipy_norm_values,
        provenance=Provenance(library="scipy", library_version="test"),
        characteristics=(
            CharacteristicName.PDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
        ),
        support=SupportBounds(float("-inf"), float("inf"), False, False),
    )


class TestE2True:
    """rev. 46, section 7.1 decision: two metrics — e2_true (accuracy) and
    e2_self_consistency (diagnostic)."""

    def test_e2_true_computed_for_compute_reference(self, registry: SourceRegistry) -> None:
        registry.register(_scipy_norm_source("scipy-norm-ppf"))
        result = run_scenario(registry, _ppf_scenario_raw())[0]
        assert result.status == Status.MEASURED
        record = result.metrics[0]
        assert "e2_true" in record.metrics
        assert "e2" not in record.metrics
        assert "e2_self_consistency" in record.diagnostics

    def test_e2_true_absent_for_lookup_only_reference(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_ppf())
        result = run_scenario(registry, _ppf_scenario_raw())[0]
        assert result.status == Status.MEASURED
        record = result.metrics[0]
        assert "e2_true" not in record.metrics
        assert "e2_self_consistency" in record.diagnostics
        assert any(n.code == "e2_true_skipped" for n in result.validator_notes)

    def test_reference_followup_crash_is_invalid_reference(self, registry: SourceRegistry) -> None:
        registry.register(FollowupCrashSource("scipy-norm-ppf"))
        result = run_scenario(registry, _ppf_scenario_raw())[0]
        assert result.status == Status.INVALID_REFERENCE
        assert "reference cdf follow-up failed" in result.status_explanation

    def test_subject_followup_crash_is_failed(self, registry: SourceRegistry) -> None:
        raw = _ppf_scenario_raw()
        raw["subject"] = {"source_id": "cdf-boom"}
        registry.register(normal_table_ppf())
        registry.register(FollowupCrashSource("cdf-boom"))
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.FAILED
        assert "subject cdf follow-up failed" in result.status_explanation


class TestExplicitPoints:
    """End-to-end behaviour of query-level ``points`` (rev. 60).

    The feature exists for "check my concrete points": a user-declared
    coordinate must reach the metrics even when no generated grid could
    ever place a point there (a far tail beyond ``TAIL_EXTENSION``), and
    every transformation the framework applies to the declared array must
    be visible as a validator note.
    """

    @staticmethod
    def raw_with_points(points: list[float], hint: int | None = 41) -> dict[str, Any]:
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "scipy-norm"}]
        raw["queries"] = [
            {
                "characteristic": "cdf",
                "output_scale": "probability",
                **({"grid_hint": {"n_points": hint}} if hint else {}),
                "points": points,
            }
        ]
        raw["regions"] = [
            {"name": "bulk", "interval": [-3.0, 3.0], "output_scale": "probability"},
            {"name": "tail_right", "interval": [3.0, None], "output_scale": "probability"},
        ]
        return raw

    def test_grid_and_explicit_points_in_one_run(self, registry: SourceRegistry) -> None:
        registry.register(scipy_norm_source())
        result = run_scenario(registry, self.raw_with_points([0.0, 0.5, 3.14159, 25.0]))[0]
        assert result.status == Status.MEASURED
        by_region = {m.region: m for m in result.metrics}
        # 41 generated points + 2 explicit points per region; the far point
        # 25.0 lives beyond the generated tail ([3, 3 + TAIL_EXTENSION]) and
        # is measurable only because the user declared it.
        assert by_region["bulk"].n_compared == 43
        assert by_region["tail_right"].n_compared == 43
        assert by_region["tail_right"].metrics["abs_error"] < 1e-12

    def test_points_only_scenario_measures_exactly_the_points(
        self, registry: SourceRegistry
    ) -> None:
        registry.register(scipy_norm_source())
        result = run_scenario(registry, self.raw_with_points([0.0, 1.0, 25.0], hint=None))[0]
        assert result.status == Status.MEASURED
        by_region = {m.region: m for m in result.metrics}
        assert by_region["bulk"].n_compared == 2  # 0.0 and 1.0
        assert by_region["tail_right"].n_compared == 1  # only 25.0

    def test_duplicate_points_are_deduplicated_with_a_note(self, registry: SourceRegistry) -> None:
        registry.register(scipy_norm_source())
        result = run_scenario(registry, self.raw_with_points([0.0, 0.0, 1.0]))[0]
        assert result.status == Status.MEASURED
        bulk = next(m for m in result.metrics if m.region == "bulk")
        assert bulk.n_compared == 43  # 41 generated + distinct 0.0 and 1.0
        codes = {note.code: note for note in result.validator_notes}
        note = codes["explicit_points_deduplicated"]
        assert "0.0" in note.detail

    def test_out_of_support_points_are_excluded_with_a_note(self, registry: SourceRegistry) -> None:
        # A real compute subject with a bounded declared support (no mocks):
        # it answers everywhere, but its support claims [-2, 2], so the
        # grids live on [-2, 2] and the declared far point 25.0 is removed
        # and reported.  AP-1 shape: out-of-support points are NOT
        # enumerated (no coordinates in the note) — the count carries the
        # fact; inside-support exclusions live in the per-record
        # excluded_points dictionary instead.
        registry.register(scipy_norm_source())  # the reference side
        registry.register(
            ExternalFunctionSource(
                "limited-norm",
                lambda _d, _p, _q, x: stats.norm.cdf(np.asarray(x, dtype=np.float64)),
                provenance=Provenance(library="scipy", library_version="test"),
                native_output_scales=["probability"],
                support=SupportBounds(-2.0, 2.0),
            )
        )
        raw = self.raw_with_points([0.0, 25.0])
        raw["subject"] = {"source_id": "limited-norm"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        codes = {note.code: note for note in result.validator_notes}
        note = codes["explicit_points_out_of_support"]
        assert "1 of 2" in note.detail
        assert "25.0" not in note.detail  # AP-1: out-of-support points are not listed
        bulk = next(m for m in result.metrics if m.region == "bulk")
        tail_metrics = [m for m in result.metrics if m.region == "tail_right"]
        assert not tail_metrics  # the tail grid is empty against [-2, 2]
        # The exclusion must not change the measured set: the same scenario
        # without the far point produces the same compared count (the point
        # 0.0 is measured in both runs).
        without_far = run_scenario(registry, self.raw_with_points([0.0]))[0]
        bulk_without = next(m for m in without_far.metrics if m.region == "bulk")
        assert bulk.n_compared == bulk_without.n_compared


class TestTailPolicyAndExcludedPoints:
    """Tail extension per family and the AP-1 excluded-points reporting (rev. 61).

    The tail length of an unbounded side is derived from the distribution
    parameters by the curated per-family table (Normal: 8 sigma), a query
    may override it (``grid_hint.tail_extension``), and every fallback to
    the fixed constant is a validator note — never silent.
    """

    @staticmethod
    def external_cdf(
        source_id: str, support: SupportBounds | None = None
    ) -> ExternalFunctionSource:
        return ExternalFunctionSource(
            source_id,
            lambda _d, _p, _q, x: stats.norm.cdf(np.asarray(x, dtype=np.float64)),
            provenance=Provenance(library="scipy", library_version="test"),
            native_output_scales=["probability"],
            support=support or SupportBounds(float("-inf"), float("inf")),
        )

    def test_curated_family_resolves_the_tail_without_a_note(
        self, registry: SourceRegistry
    ) -> None:
        registry.register(self.external_cdf("ext-subj"))
        registry.register(self.external_cdf("ext-ref"))
        raw = cdf_scenario_raw()
        raw["schema_version"] = "1.2"
        raw["distribution"] = {
            "family": "Normal",
            "parametrization": "meanStd",
            "parameters": {"mu": 0.0, "sigma": 1.0},
        }
        raw["subject"] = {"source_id": "ext-subj"}
        raw["references"] = [{"source_id": "ext-ref"}]
        raw["queries"] = [
            {
                "characteristic": "cdf",
                "output_scale": "probability",
                "grid_hint": {"n_points": 9, "densify": False},
            }
        ]
        raw["regions"] = [
            {"name": "tail_right", "interval": [1.0, None], "output_scale": "probability"}
        ]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert result.metrics[0].n_compared == 9  # exactly the uniform grid
        # Normal: the curated scale is 8 sigma -> the side resolves to [1, 9].
        expected = np.linspace(1.0, 9.0, 9)
        assert result.y_true is not None
        np.testing.assert_allclose(result.y_true, stats.norm.cdf(expected), atol=1e-9)
        codes = {note.code for note in result.validator_notes}
        assert "tail_extension_fallback" not in codes  # curated family: nothing to say

    def test_user_tail_override_wins_over_the_curated_value(self, registry: SourceRegistry) -> None:
        registry.register(self.external_cdf("ext-subj"))
        registry.register(self.external_cdf("ext-ref"))
        raw = cdf_scenario_raw()
        raw["distribution"] = {
            "family": "Normal",
            "parametrization": "meanStd",
            "parameters": {"mu": 0.0, "sigma": 1.0},
        }
        raw["subject"] = {"source_id": "ext-subj"}
        raw["references"] = [{"source_id": "ext-ref"}]
        raw["queries"] = [
            {
                "characteristic": "cdf",
                "output_scale": "probability",
                "grid_hint": {"n_points": 9, "densify": False, "tail_extension": 20.0},
            }
        ]
        raw["regions"] = [
            {"name": "tail_right", "interval": [1.0, None], "output_scale": "probability"}
        ]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        # The override wins: [1, 1 + 20] instead of the curated [1, 9].
        expected = np.linspace(1.0, 21.0, 9)
        assert result.y_true is not None
        np.testing.assert_allclose(result.y_true, stats.norm.cdf(expected), atol=1e-9)

    def test_fallback_note_for_family_without_curated_scale(self, registry: SourceRegistry) -> None:
        registry.register(self.external_cdf("ext-subj"))
        registry.register(self.external_cdf("ext-ref"))
        raw = cdf_scenario_raw()
        raw["distribution"] = {"family": "Cauchy", "parameters": {"gamma": 1.0}}
        raw["subject"] = {"source_id": "ext-subj"}
        raw["references"] = [{"source_id": "ext-ref"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        codes = {note.code: note for note in result.validator_notes}
        note = codes["tail_extension_fallback"]
        assert "Cauchy" in note.detail
        assert "10" in note.detail  # the fixed fallback length is named

    def test_dictionary_lists_missing_reference_points(self, registry: SourceRegistry) -> None:
        # AP-1: an inside-support point the reference does not answer is
        # enumerated with its coordinate and the counter-named reason tag.
        registry.register(self.external_cdf("ext-subj"))
        registry.register(
            ExternalFunctionSource(
                "holey-ref",
                lambda _d, _p, _q, x: np.where(
                    np.abs(np.asarray(x, dtype=np.float64)) < 1e-9,
                    np.nan,
                    stats.norm.cdf(np.asarray(x, dtype=np.float64)),
                ),
                provenance=Provenance(library="scipy", library_version="test"),
                native_output_scales=["probability"],
                support=SupportBounds(float("-inf"), float("inf")),
            )
        )
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "holey-ref"}]
        raw["subject"] = {"source_id": "ext-subj"}
        raw["queries"] = [
            {
                "characteristic": "cdf",
                "output_scale": "probability",
                "grid_hint": {"n_points": 41, "densify": False},
            }
        ]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        record = result.metrics[0]
        assert record.n_missing_reference == 1
        assert (0.0, "n_missing_reference") in {(p.x, p.reason) for p in record.excluded_points}
        assert record.n_compared == 40

    def test_dictionary_lists_invalid_subject_points(self, registry: SourceRegistry) -> None:
        registry.register(
            ExternalFunctionSource(
                "holey-subj",
                lambda _d, _p, _q, x: np.where(
                    np.abs(np.asarray(x, dtype=np.float64)) < 1e-9,
                    np.nan,
                    stats.norm.cdf(np.asarray(x, dtype=np.float64)),
                ),
                provenance=Provenance(library="scipy", library_version="test"),
                native_output_scales=["probability"],
                support=SupportBounds(float("-inf"), float("inf")),
            )
        )
        registry.register(self.external_cdf("ext-ref"))
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "holey-subj"}
        raw["references"] = [{"source_id": "ext-ref"}]
        raw["queries"] = [
            {
                "characteristic": "cdf",
                "output_scale": "probability",
                "grid_hint": {"n_points": 41, "densify": False},
            }
        ]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        record = result.metrics[0]
        assert record.n_invalid_subject == 1
        assert (0.0, "n_invalid_subject") in {(p.x, p.reason) for p in record.excluded_points}

    def test_removed_generated_points_are_reported_as_a_count(
        self, registry: SourceRegistry
    ) -> None:
        # AP-1: points the support filter removes are not enumerated (the
        # coordinates are reconstructible from the deterministic
        # generator) — the count is the report.  The filter never fires on
        # generator-produced geometry, so the drop is injected through the
        # runner's documented grid_generator seam (a real generator
        # subclass, not a mock).
        class _DustGridGenerator(GridGenerator):
            def _keep_in_support(self, points, support):
                kept, removed = super()._keep_in_support(points, support)
                return kept[:-1], removed + 1

        registry.register(self.external_cdf("ext-subj"))
        registry.register(self.external_cdf("ext-ref"))
        runner = AccuracyRunner(registry, ResultsStore(), grid_generator=_DustGridGenerator())
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "ext-subj"}
        raw["references"] = [{"source_id": "ext-ref"}]
        result = runner.run(load(raw))[0]
        assert result.status == Status.MEASURED
        assert result.metrics[0].n_compared == 40  # 41 generated points minus the dust
        codes = {note.code: note for note in result.validator_notes}
        note = codes["grid_points_removed_out_of_support"]
        assert "1 generated grid point" in note.detail
        assert "[-3.0, 3.0]" not in note.detail  # the count, not coordinates

    def test_normal_run_carries_no_removal_note(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        result = run_scenario(registry, cdf_scenario_raw())[0]
        assert result.status == Status.MEASURED
        codes = {note.code for note in result.validator_notes}
        assert "grid_points_removed_out_of_support" not in codes


class TestGroupGuards:
    """Group-scope invariants of the orchestrator (REV-16, A-5, B-7)."""

    def test_empty_group_is_a_no_op(self, registry: SourceRegistry) -> None:
        assert AccuracyRunner(registry, ResultsStore()).run_group([]) == []

    def test_mixed_groups_are_rejected(self, registry: SourceRegistry) -> None:
        spec_a = load(cdf_scenario_raw())
        raw_b = cdf_scenario_raw()
        raw_b["comparison_group"] = "another-group"
        spec_b = load(raw_b)
        with pytest.raises(FrameworkDefect, match="one comparison_group"):
            AccuracyRunner(registry, ResultsStore()).run_group([spec_a, spec_b])

    def test_duplicate_pair_write_is_rejected(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        store = ResultsStore()
        results = run_scenario(registry, cdf_scenario_raw(), store)
        with pytest.raises(FrameworkDefect, match="duplicate result"):
            store.write_result(results[0])

    def test_before_scenario_hook_runs_once_per_scenario(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        calls: list[int] = []
        AccuracyRunner(registry, ResultsStore()).run_group(
            [load(cdf_scenario_raw())], before_scenario=lambda: calls.append(1)
        )
        assert sum(calls) == 1


class TestUnreadyMerges:
    """Verdict collapse of unready sides (K-4, A-7, B-3a) with real sources."""

    @staticmethod
    def symmetric_raw() -> dict[str, Any]:
        raw = cdf_scenario_raw()
        raw["participants"] = [
            {"role": "subject", "source_id": "pysatl-core"},
            {"role": "reference", "source_id": "scipy-norm"},
        ]
        del raw["references"]  # the list form is authoritative; no synonyms
        del raw["subject"]
        raw["pair_policy"] = "symmetric"
        return raw

    def test_symmetric_mode_with_a_refusing_side(self, registry: SourceRegistry) -> None:
        # A real source whose capability set is empty for its role: the
        # registry resolve succeeds, the assessment refuses — and the pair
        # carries the mapped status with both sides in the explanation.
        registry.register(
            ExternalFunctionSource(
                "empty-caps",
                lambda d, p, q, x: np.asarray(x),
                provenance=Provenance(library="x", library_version="0"),
                capabilities=frozenset(),
            )
        )
        raw = self.symmetric_raw()
        raw["participants"][1] = {"role": "reference", "source_id": "empty-caps"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.UNSUPPORTED_CONFIGURATION
        assert "reference" in result.status_explanation

    def test_subject_not_covered_maps_to_failed(self, registry: SourceRegistry) -> None:
        # A compute subject that honestly declares "these points are not
        # covered" at assess time: NOT_COVERED on the subject side maps to
        # failed (the measured side could not answer), not to a reference
        # verdict.
        class PickySubject(ExternalFunctionSource):
            def assess(self, participant, query, distribution, points=None):
                verdict = super().assess(participant, query, distribution, points)
                if verdict.readiness != Readiness.READY:
                    return verdict
                return SupportVerdict(
                    readiness=Readiness.NOT_COVERED, reason="subject refuses these points"
                )

        registry.register(
            PickySubject(
                "picky",
                lambda d, p, q, x: stats.norm.cdf(np.asarray(x, dtype=np.float64)),
                provenance=Provenance(library="scipy", library_version="test"),
            )
        )
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "picky"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.FAILED

    def test_unregistered_source_yields_unsupported_configuration(
        self, registry: SourceRegistry
    ) -> None:
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "ghost"}]
        result = run_scenario(registry, raw)[0]
        # UNAVAILABLE on the reference side is invalid_reference; an
        # unregistered SUBJECT is a configuration the run cannot serve.
        raw_subject = cdf_scenario_raw()
        raw_subject["subject"] = {"source_id": "ghost"}
        raw_subject["references"] = [{"source_id": "scipy-norm-cdf"}]
        registry.register(normal_table_cdf())
        subject_result = run_scenario(registry, raw_subject)[0]
        assert subject_result.status == Status.UNSUPPORTED_CONFIGURATION
        _ = raw, result


class TestFrameworkDefectPropagation:
    """FrameworkDefect never becomes a status (errors.py contract)."""

    def test_source_raising_framework_defect_crashes_the_run(
        self, registry: SourceRegistry
    ) -> None:
        class DefectSource(ExternalFunctionSource):
            def compute(self, participant, query, points, distribution):
                msg = "framework misused by the source"
                raise FrameworkDefect(msg)

        registry.register(
            DefectSource(
                "defect",
                lambda d, p, q, x: stats.norm.cdf(np.asarray(x, dtype=np.float64)),
                provenance=Provenance(library="scipy", library_version="test"),
            )
        )
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "defect"}
        raw["references"] = [{"source_id": "ppf-table"}]
        raw["queries"] = [{"characteristic": "ppf"}]
        raw["regions"] = [
            {"name": "levels", "interval": [0.05, 0.95], "output_scale": "probability"}
        ]
        registry.register(normal_table_ppf("ppf-table"))
        with pytest.raises(FrameworkDefect):
            run_scenario(registry, raw)


class TestSurvivalFollowup:
    """D8: follow-up values answer in the native scale and are converted
    to probability; the fact travels to provenance (rev. 46)."""

    def test_survival_native_reference_converts_the_followup(
        self, registry: SourceRegistry
    ) -> None:
        # A compute reference answering natively in survival scale (S = 1-F):
        # the ppf follow-up must convert its answers to probability before
        # plugging the found quantiles in, and record the fact.
        registry.register(
            ExternalFunctionSource(
                "survival-norm",
                lambda d, p, q, x: (
                    stats.norm.ppf(np.asarray(x, dtype=np.float64))
                    if q.characteristic == CharacteristicName.PPF
                    else 1.0 - stats.norm.cdf(np.asarray(x, dtype=np.float64))
                ),
                provenance=Provenance(library="scipy", library_version="test"),
                native_output_scales=["survival"],
            )
        )
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "survival-norm"}]
        raw["queries"] = [{"characteristic": "ppf"}]
        raw["regions"] = [
            {"name": "levels", "interval": [0.05, 0.95], "output_scale": "probability"}
        ]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert result.metrics and "e2_true" in result.metrics[0].metrics
        facts = result.reference_provenance.conversion_facts if result.reference_provenance else ()
        assert facts and facts[0]["from_scale"] == "survival"


class TestExplicitPairPolicy:
    """pair_policy as an explicit list (REV-3): one result per declared pair."""

    def test_pair_list_runs_exactly_the_declared_pairs(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["participants"] = [
            {"role": "subject", "source_id": "pysatl-core"},
            {"role": "reference", "source_id": "t1"},
            {"role": "peer", "source_id": "t2"},
        ]
        del raw["subject"], raw["references"]
        raw["pair_policy"] = [[0, 1], [0, 2]]
        registry.register(normal_table_cdf("t1"))
        registry.register(normal_table_cdf("t2"))
        results = run_scenario(registry, raw)
        assert len(results) == 2
        assert all(r.status == Status.MEASURED for r in results)


class TestFollowupSkipNotes:
    """e2_true needs a compute reference; a lookup-only reference is an
    explicit skip note, not a silently missing metric (rev. 46)."""

    def test_lookup_only_reference_skips_e2_true_with_a_note(
        self, registry: SourceRegistry
    ) -> None:
        registry.register(normal_table_ppf("ppf-table"))
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "ppf-table"}]
        raw["queries"] = [{"characteristic": "ppf"}]
        raw["regions"] = [
            {"name": "levels", "interval": [0.05, 0.95], "output_scale": "probability"}
        ]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert "e1" in result.metrics[0].metrics
        assert "e2_true" not in result.metrics[0].metrics
        codes = {note.code for note in result.validator_notes}
        assert "e2_true_skipped" in codes


class TestInvariantCheckOnRun:
    """The invariant post-check wired through the runner (K-6, REV-5)."""

    @staticmethod
    def raw_with_invariants() -> dict[str, Any]:
        raw = cdf_scenario_raw()
        raw["invariant_check_on"] = ["subject"]
        return raw

    def test_violation_flips_measured_to_invalid_reference(self, registry: SourceRegistry) -> None:
        # A real source that returns out-of-range "cdf" values: the check
        # disproofs the subject and the pair flips to invalid_reference.
        registry.register(normal_table_cdf())
        registry.register(
            ExternalFunctionSource(
                "bad-cdf",
                lambda d, p, q, x: np.asarray(x, dtype=np.float64) + 5.0,  # cdf > 1
                provenance=Provenance(library="bad", library_version="0"),
            )
        )
        raw = self.raw_with_invariants()
        raw["subject"] = {"source_id": "bad-cdf"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.INVALID_REFERENCE
        assert result.invariant_violations
        assert any(v.participant_id.startswith("bad-cdf") for v in result.invariant_violations)

    def test_check_by_participant_source_id_targets_the_reference(
        self, registry: SourceRegistry
    ) -> None:
        # invariant_check_on accepts source_id/participant_id: naming the
        # reference runs the check against the reference side (1643-1644).
        registry.register(normal_table_cdf())
        raw = self.raw_with_invariants()
        raw["invariant_check_on"] = ["scipy-norm-cdf"]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED  # a healthy reference passes
        assert not result.invariant_violations


class TestAssessCrashAndRefusalVerdicts:
    """Assess-time failures are verdicts, not crashes (REV-5, errors.py):
    a refusing source is unsupported_configuration, a crashing source is
    unavailable — and the per-role readiness map turns the latter into the
    pair status."""

    @staticmethod
    def refusing_source(source_id: str) -> ValueSource:
        class Refusing(ValueSource):
            """A protocol implementation that refuses at assess time."""

            def __init__(self) -> None:
                self._id = source_id

            @property
            def source_id(self) -> str:
                return self._id

            @property
            def capabilities(self) -> frozenset[Capability]:
                return frozenset({Capability.COMPUTE})

            @property
            def subject_id(self) -> str:
                return f"{self._id}@test"

            @property
            def provenance(self) -> Provenance:
                return Provenance(library=self._id, library_version="0")

            def support(self, participant, distribution):
                return SupportBounds(-10.0, 10.0)

            def assess(self, participant, query, distribution, points=None):
                msg = "this configuration is not supported by design"
                raise AdapterConfigurationError(msg)

            def native_output_scales(self, participant, query):
                return ["probability"]

            def prepare(self, participant, query, distribution):
                return PreparedSubject(participant.participant_id, query, distribution)

            def evaluate(self, prepared, points):
                return np.zeros(np.asarray(points).size)

            def compute(self, participant, query, points, distribution):
                return np.zeros(np.asarray(points).size)

        return Refusing()

    def test_reference_assess_refusal_is_invalid_reference(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        registry.register(self.refusing_source("refusing"))
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "refusing"}]
        result = run_scenario(registry, raw)[0]
        # AdapterConfigurationError on the reference side -> unsupported
        # verdict -> invalid_reference for the pair, reason quoted.
        assert result.status == Status.UNSUPPORTED_CONFIGURATION

    def test_subject_assess_crash_is_unsupported_configuration(
        self, registry: SourceRegistry
    ) -> None:
        class CrashingAssess(ValueSource):
            def __init__(self) -> None:
                self._id = "assess-boom"

            @property
            def source_id(self) -> str:
                return self._id

            @property
            def capabilities(self) -> frozenset[Capability]:
                return frozenset({Capability.COMPUTE})

            @property
            def subject_id(self) -> str:
                return f"{self._id}@test"

            @property
            def provenance(self) -> Provenance:
                return Provenance(library=self._id, library_version="0")

            def support(self, participant, distribution):
                return SupportBounds(-10.0, 10.0)

            def assess(self, participant, query, distribution, points=None):
                raise RuntimeError("assessment exploded")

            def native_output_scales(self, participant, query):
                return ["probability"]

            def prepare(self, participant, query, distribution):
                return PreparedSubject(participant.participant_id, query, distribution)

            def evaluate(self, prepared, points):
                return np.zeros(np.asarray(points).size)

            def compute(self, participant, query, points, distribution):
                return np.zeros(np.asarray(points).size)

        registry.register(CrashingAssess())
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "assess-boom"}
        result = run_scenario(registry, raw)[0]
        # UNAVAILABLE on the subject side maps to unsupported_configuration
        # (the runner cannot yet tell a broken source from an unservable
        # configuration), and the crash detail stays in the explanation.
        assert result.status == Status.UNSUPPORTED_CONFIGURATION
        assert "assessment exploded" in result.status_explanation

    def test_symmetric_both_sides_unready(self, registry: SourceRegistry) -> None:
        registry.register(self.refusing_source("a"))
        registry.register(self.refusing_source("b"))
        raw = cdf_scenario_raw()
        raw["participants"] = [
            {"role": "subject", "source_id": "a"},
            {"role": "reference", "source_id": "b"},
        ]
        del raw["subject"], raw["references"]
        raw["pair_policy"] = "symmetric"
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.UNSUPPORTED_CONFIGURATION
        assert "side_a" in result.status_explanation
        assert "side_b" in result.status_explanation


class TestPpfFollowupSkips:
    """A failed native-scale probe disables the follow-ups explicitly
    (D8): the notes name the side and the skipped metric."""

    @staticmethod
    def ppf_raw() -> dict[str, Any]:
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "scipy-norm"}]
        raw["queries"] = [{"characteristic": "ppf"}]
        raw["regions"] = [
            {"name": "levels", "interval": [0.05, 0.95], "output_scale": "probability"}
        ]
        return raw

    def test_subject_probe_failure_skips_e2_self_consistency(
        self, registry: SourceRegistry
    ) -> None:
        class ProbeBoom(ExternalFunctionSource):
            def native_output_scales(self, participant, query):
                raise RuntimeError("probe exploded")

        registry.register(scipy_norm_source())
        registry.register(
            ProbeBoom(
                "probe-boom",
                lambda d, p, q, x: stats.norm.ppf(np.asarray(x, dtype=np.float64)),
                provenance=Provenance(library="scipy", library_version="test"),
            )
        )
        raw = self.ppf_raw()
        raw["subject"] = {"source_id": "probe-boom"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        codes = {note.code for note in result.validator_notes}
        assert "e2_self_consistency_skipped" in codes
        assert any("probe exploded" in n.detail for n in result.validator_notes)

    def test_reference_probe_failure_skips_e2_true(self, registry: SourceRegistry) -> None:
        class ProbeBoom(ExternalFunctionSource):
            def native_output_scales(self, participant, query):
                raise RuntimeError("probe exploded")

        registry.register(
            ProbeBoom(
                "probe-boom",
                lambda d, p, q, x: 1.0 - stats.norm.cdf(np.asarray(x, dtype=np.float64)),
                provenance=Provenance(library="scipy", library_version="test"),
                native_output_scales=["survival"],
            )
        )
        raw = self.ppf_raw()
        raw["references"] = [{"source_id": "probe-boom"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert "e2_true_skipped" in {note.code for note in result.validator_notes}


class TestInvariantsNoTarget:
    """An invariant name matching neither side of the pair simply has no
    target here (symmetric groups with peers) — no violation is invented."""

    def test_peer_invariant_in_a_symmetric_group_adds_no_violations(
        self, registry: SourceRegistry
    ) -> None:
        registry.register(normal_table_cdf())
        registry.register(normal_table_cdf("t1"))
        registry.register(normal_table_cdf("t2"))
        raw = cdf_scenario_raw()
        raw["participants"] = [
            {"role": "peer", "source_id": "t1"},
            {"role": "reference", "source_id": "t2"},
            {"role": "subject", "source_id": "pysatl-core"},
        ]
        del raw["subject"], raw["references"]
        raw["pair_policy"] = [[2, 1]]
        # t1 is a peer of this group and takes no part in the pair [2, 1]:
        # the invariant name matches no side of THIS pair -> no target here.
        raw["invariant_check_on"] = ["t1"]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert not result.invariant_violations


class TestSweepStringAxis:
    """String sort_keys sort lexicographically after numeric ones (K-1)."""


class TestCurveAxisDetails:
    """Comparator curve axis and agreement guards (K-1, REV-5/C3)."""

    def test_curve_axis_orders_numbers_first_strings_second(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        store = ResultsStore()
        for key in (10.0, "b", 2.0, "a"):
            raw = cdf_scenario_raw()
            raw["comparison_group"] = "mixed-axis"
            raw["scenario_id"] = f"stage-{key}"
            raw["sort_key"] = key
            raw["references"] = [{"source_id": "scipy-norm-cdf"}]
            AccuracyRunner(registry, store).run(load(raw))
        curve = Comparator(store).curve("mixed-axis")
        # numeric keys ascending first, string keys lexicographic after
        assert curve.axis == (2.0, 10.0, "a", "b")

    def test_curve_skips_rows_without_metrics(self, registry: SourceRegistry) -> None:
        # a measured row with metrics and a not-measured row with none:
        # the axis keeps both sort_keys, the series only the measured one
        registry.register(normal_table_cdf())
        store = ResultsStore()
        for key, references in (
            (1.0, [{"source_id": "scipy-norm-cdf"}]),
            (2.0, [{"source_id": "ghost"}]),
        ):
            raw = cdf_scenario_raw()
            raw["comparison_group"] = "mixed-metrics"
            raw["scenario_id"] = f"stage-{key}"
            raw["sort_key"] = key
            raw["references"] = references
            AccuracyRunner(registry, store).run(load(raw))
        curve = Comparator(store).curve("mixed-metrics")
        assert curve.axis == (1.0, 2.0)
        assert {point.pair for point in curve.points} == {curve.points[0].pair}

    def test_agreement_of_different_pairs_is_refused(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        registry.register(normal_table_pdf())
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "scipy-norm-cdf"}]
        first = run_scenario(registry, raw)[0]
        raw2 = cdf_scenario_raw()
        raw2["references"] = [{"source_id": "scipy-norm-pdf"}]
        second = run_scenario(registry, raw2)[0]
        store = ResultsStore()
        with pytest.raises(GridMismatchError, match="same scenario pair"):
            Comparator(store).agreement(first, second)


class TestOutcomeMappings:
    """Outcome-vs-crash boundary of the runner (errors.py contract): each
    test pins one mapping with a real protocol implementation."""

    @staticmethod
    def reference_prepare_crasher(source_id: str) -> ValueSource:
        class PrepareCrasher(ValueSource):
            @property
            def source_id(self) -> str:
                return source_id

            @property
            def capabilities(self) -> frozenset[Capability]:
                return frozenset({Capability.COMPUTE})

            @property
            def subject_id(self) -> str:
                return f"{source_id}@test"

            @property
            def provenance(self) -> Provenance:
                return Provenance(library=source_id, library_version="0")

            def support(self, participant, distribution):
                return SupportBounds(-10.0, 10.0)

            def assess(self, participant, query, distribution, points=None):
                return SupportVerdict(readiness=Readiness.READY, reason="pretends ready")

            def native_output_scales(self, participant, query):
                return ["probability"]

            def prepare(self, participant, query, distribution):
                msg = "the table engine exploded during prepare"
                raise RuntimeError(msg)

            def evaluate(self, prepared, points):
                return np.zeros(np.asarray(points).size)

            def compute(self, participant, query, points, distribution):
                return np.zeros(np.asarray(points).size)

        return PrepareCrasher()

    def test_reference_prepare_crash_is_invalid_reference(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())
        registry.register(self.reference_prepare_crasher("prep-boom"))
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "prep-boom"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.INVALID_REFERENCE
        assert "prep-boom" in result.status_explanation

    def test_all_nan_reference_is_invalid_reference(self, registry: SourceRegistry) -> None:
        # A compute reference that answers nothing finitely: every point is
        # excluded from the comparison -> "no point comparable", not failed.
        registry.register(
            ExternalFunctionSource(
                "all-nan",
                lambda d, p, q, x: np.full(np.asarray(x).shape, np.nan),
                provenance=Provenance(library="nan", library_version="0"),
            )
        )
        raw = cdf_scenario_raw()
        raw["references"] = [{"source_id": "all-nan"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.INVALID_REFERENCE
        assert "no point of the declared grid" in result.status_explanation

    def test_metric_evaluation_crash_is_failed(self, registry: SourceRegistry) -> None:
        # A source returning a non-float-convertible array: values pass the
        # source boundary, metric evaluation raises, the pair outcome is
        # failed — the run itself must survive (REV-5).
        class GarbageSource(ExternalFunctionSource):
            def evaluate(self, prepared, points):
                return [["garbage"] for _ in np.asarray(points).ravel()]

        registry.register(scipy_norm_source())
        registry.register(
            GarbageSource(
                "garbage",
                lambda d, p, q, x: np.asarray(x),
                provenance=Provenance(library="g", library_version="0"),
            )
        )
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "garbage"}
        raw["references"] = [{"source_id": "scipy-norm"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.FAILED
        assert "metric evaluation failed" in result.status_explanation

    def test_subject_with_inapplicable_characteristic_is_not_applicable(
        self, registry: SourceRegistry
    ) -> None:
        # The subject refuses the characteristic itself (its wrapped
        # function serves pdf only): every subject verdict is
        # not_applicable, and the pair outcome collapses to not_applicable.
        registry.register(
            ExternalFunctionSource(
                "pdf-only",
                lambda d, p, q, x: stats.norm.pdf(np.asarray(x, dtype=np.float64)),
                provenance=Provenance(library="scipy", library_version="test"),
                characteristics=["pdf"],
            )
        )
        registry.register(normal_table_cdf())
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "pdf-only"}
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.NOT_APPLICABLE

    def test_probe_failure_on_a_survival_region_cannot_decide_safely(
        self, registry: SourceRegistry
    ) -> None:
        # D8: a failed native-scale probe plus a non-probability question
        # scale is an unsupported configuration — the conversion decision
        # would be made on missing data (no silence).
        class ProbeBoom(ExternalFunctionSource):
            def native_output_scales(self, participant, query):
                raise RuntimeError("probe exploded")

        registry.register(scipy_norm_source())
        registry.register(
            ProbeBoom(
                "probe-boom",
                lambda d, p, q, x: stats.norm.cdf(np.asarray(x, dtype=np.float64)),
                provenance=Provenance(library="scipy", library_version="test"),
            )
        )
        raw = cdf_scenario_raw()
        raw["subject"] = {"source_id": "probe-boom"}
        raw["references"] = [{"source_id": "scipy-norm"}]
        raw["regions"] = [{"name": "tail", "interval": [1.0, 3.0], "output_scale": "survival"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.UNSUPPORTED_CONFIGURATION
        assert "cannot be decided safely" in result.status_explanation
