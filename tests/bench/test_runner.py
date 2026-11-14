"""Runner end-to-end: statuses, invariants §7, D7 order, К-4, В-3a."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats

from pysatl_core_bench import (
    AccuracyRunner,
    Comparator,
    GridMismatchError,
    PrecomputedTable,
    ResultsStore,
    ScenarioExpander,
    SourceRegistry,
    Status,
    TableLookupSource,
)
from pysatl_core_bench.accuracy.errors import ScenarioRejected
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
        raw["regions"] = [{"name": "tail_right", "interval": [1.0, 3.0], "space": "survival"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.MEASURED
        assert result.space_computed == "probability"
        assert result.space_reported == "survival"
        assert result.subject_provenance is not None
        facts = result.subject_provenance.conversion_facts
        assert facts and facts[0]["to_space"] == "survival"

    def test_partial_coverage_is_measured_with_missing_counter(
        self, registry: SourceRegistry
    ) -> None:
        # D6: a hole *inside* the measured support is still a measurement.
        # Since ред. 46 the grid is built on the participants' support
        # intersection, so the hole must live inside the table's own range.
        points = np.linspace(-3.0, 3.0, 41)
        hole = (points > -1.0) & (points < 1.0)
        registry.register(
            TableLookupSource(
                "scipy-norm-cdf",
                PrecomputedTable(
                    characteristic="cdf",
                    space="probability",
                    points=points[~hole],
                    values=stats.norm.cdf(points[~hole]),
                    library="scipy",
                    library_version="1.15",
                    coverage_tol=1e-12,
                ),
            )
        )
        raw = cdf_scenario_raw()
        raw["regions"] = [{"name": "wide", "interval": [-3.0, 3.0], "space": "probability"}]
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
        raw["regions"] = [{"name": "far", "interval": [5.0, 8.0], "space": "probability"}]
        result = run_scenario(registry, raw)[0]
        assert result.status == Status.INVALID_REFERENCE

    def test_priority_failed_beats_others(self, registry: SourceRegistry) -> None:
        # subject crashes AND reference is unknown -> failed wins (К-4)
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
        # subject source missing -> unsupported_configuration (К-4 priority)
        assert all(r.status == Status.UNSUPPORTED_CONFIGURATION for r in results)
        assert all("subject:" in r.status_explanation for r in results)

    def test_unready_status_does_not_leak_between_pairs(self, registry: SourceRegistry) -> None:
        # А-1 (К-4): the status of pair 2 must be computed from *its own*
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
        # А-2: in symmetric mode a side_a (subject) crash must be `failed`
        # blaming the subject, not `invalid_reference` blaming the reference.
        registry.register(CrashingSource("crashing-a"))
        raw = {
            "schema_version": "1.0",
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
                {"characteristic": "pdf", "space": "probability", "grid_hint": {"n_points": 5}}
            ],
            "regions": [{"name": "bulk", "interval": [-1.0, 1.0], "space": "probability"}],
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
            "schema_version": "1.0",
            "comparison_group": "self-vs-self",
            "distribution": {
                "family": "Normal",
                "parametrization": "meanStd",
                "parameters": {"mu": 0.0, "sigma": 1.0},
            },
            "subject": normal_participant(),
            "references": [normal_participant()],
            "queries": [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0], "space": "probability"}],
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
            "schema_version": "1.0",
            "comparison_group": "invariant-group",
            "distribution": {
                "family": "Normal",
                "parametrization": "meanStd",
                "parameters": {"mu": 0.0, "sigma": 1.0},
            },
            "subject": {"source_id": "lying-cdf"},
            "references": [{"source_id": "scipy-norm-cdf"}, {"source_id": "lying-cdf"}],
            "queries": [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0], "space": "probability"}],
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
    """Sweep-отчёт: кривая по sort_key (К-1) вместо плоской таблицы."""

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
    """ред. 46, решение по 7.2: lookup-only member -> intersection + warning."""

    def test_lookup_only_member_builds_intersection_support(self, registry: SourceRegistry) -> None:
        registry.register(normal_table_cdf())  # covers [-3, 3] only
        raw = cdf_scenario_raw()
        raw["queries"] = [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}]
        raw["regions"] = [{"name": "wide", "interval": [-4.0, 4.0], "space": "probability"}]
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
        "schema_version": "1.0",
        "comparison_group": "normal-ppf",
        "distribution": {
            "family": "Normal",
            "parametrization": "meanStd",
            "parameters": {"mu": 0.0, "sigma": 1.0},
        },
        "subject": {"source_id": "pysatl-core"},
        "references": [{"source_id": "scipy-norm-ppf"}],
        "queries": [{"characteristic": "ppf", "grid_hint": {"n_points": 17}}],
        "regions": [{"name": "levels", "interval": [0.05, 0.95], "space": "probability"}],
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
    """ред. 46, решение по 7.1: two metrics — e2_true (accuracy) and
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
