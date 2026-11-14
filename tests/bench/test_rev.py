"""Regression tests of the REV revision (ред. 34).

Each test pins one fix of the REV-series audit: loader rejections that
used to crash or pass silently, the empty-region outcome, store
duplication, and DTO invariants.  Registry codes (М-6, D3, К-4, N6) —
work-area registries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmarks.reference_sources import scipy_norm_source
from pysatl_core_bench import (
    AccuracyQuery,
    AccuracyResult,
    AccuracyRunner,
    DistributionSpec,
    ParticipantSpec,
    ResultsStore,
    ScenarioExpander,
    ScenarioLoader,
    SourceRegistry,
)
from pysatl_core_bench.accuracy.computations.protocol import PreparedSubject, ValueSource
from pysatl_core_bench.accuracy.dto import (
    Agreement,
    MetricRecord,
    Provenance,
    SpecialPoint,
    SupportBounds,
    SupportVerdict,
)
from pysatl_core_bench.accuracy.errors import FrameworkDefect, ScenarioRejected
from pysatl_core_bench.accuracy.reporting.compare import Comparator
from pysatl_core_bench.accuracy.scenarios.detection import SpecialPointDetector
from pysatl_core_bench.accuracy.scenarios.spec import ScenarioSpec
from pysatl_core_bench.accuracy.types import Capability, Readiness, Role, Status

from .conftest import CrashingSource, cdf_scenario_raw, normal_distribution, normal_participant

Float64Array = NDArray[np.float64]


def _loader_load(raw: dict[str, Any]) -> ScenarioSpec:
    return ScenarioLoader().load(raw)


class _BoundedSource(ValueSource):
    """A compute source with support [0, 1]: serves grid-empty scenarios."""

    def __init__(self, source_id: str = "bounded") -> None:
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

    def support(self, participant: ParticipantSpec, distribution: Any) -> SupportBounds:
        return SupportBounds(left=0.0, right=1.0)

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: Any,
        points: Float64Array | None = None,
    ) -> SupportVerdict:
        return SupportVerdict(readiness=Readiness.READY, reason="bounded pretends")

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        return ["probability"]

    def prepare(
        self, participant: ParticipantSpec, query: AccuracyQuery, distribution: Any
    ) -> PreparedSubject:
        return PreparedSubject(participant_id=participant.participant_id, query=query)

    def evaluate(self, prepared: PreparedSubject, points: Any) -> Any:
        return np.zeros(len(np.asarray(points)))

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Float64Array,
        distribution: Any,
    ) -> Float64Array:
        return np.zeros(len(np.asarray(points)))


def _result(
    status: Status = Status.MEASURED,
    metrics: tuple[MetricRecord, ...] = (),
) -> AccuracyResult:
    """Minimal result factory for store/agreement unit tests."""
    return AccuracyResult(
        schema_version="1.0",
        scenario_id="s",
        spec_hash="sha256:x",
        comparison_group_id="g",
        sort_key=None,
        pair=("a", "b"),
        subject_id="a",
        reference="b",
        role_labels={},
        status=status,
        status_explanation="",
        metrics=metrics,
    )


class TestLoaderRejections:
    """REV-3: contradictory grammar is Rejected, not crash or silence."""

    def _raw(self, **overrides: Any) -> dict[str, Any]:
        raw = cdf_scenario_raw()
        raw.update(overrides)
        return raw

    def _pair_list_raw(self, policy: list[list[int]]) -> dict[str, Any]:
        raw = self._raw(
            participants=[
                {**normal_participant(), "role": "subject"},
                {"source_id": "scipy-norm-cdf", "role": "reference"},
            ],
        )
        raw.pop("subject", None)
        raw.pop("references", None)
        raw["pair_policy"] = policy
        return raw

    def test_explicit_pair_list_duplicate_rejected(self) -> None:
        with pytest.raises(ScenarioRejected, match="duplicate"):
            _loader_load(self._pair_list_raw([[0, 1], [1, 0]]))

    def test_explicit_pair_list_normalized_in_canonical_form(self) -> None:
        a = _loader_load(self._pair_list_raw([[0, 1]]))
        b = _loader_load(self._pair_list_raw([[1, 0]]))
        assert a.spec_hash == b.spec_hash

    def test_duplicate_queries_rejected(self) -> None:
        raw = self._raw()
        raw["queries"] = [
            {"characteristic": "cdf", "space": "probability"},
            {"characteristic": "cdf", "space": "probability"},
        ]
        with pytest.raises(ScenarioRejected, match="duplicate query"):
            _loader_load(raw)

    def test_chain_stage_missing_participant_rejected(self) -> None:
        raw = {
            "schema_version": "1.0",
            "comparison_group": "chain",
            "queries": [{"characteristic": "cdf"}],
            "chain_spec": {"stages": [{"id": "a", "reference": normal_participant()}]},
        }
        with pytest.raises(ScenarioRejected, match="missing 'participant'"):
            ScenarioExpander().expand(raw)

    def test_strategy_params_nan_rejected(self) -> None:
        raw = self._raw()
        raw["subject"] = {
            **normal_participant(),
            "strategy": {"name": "ForcedGraphStrategy", "params": {"from": float("nan")}},
        }
        with pytest.raises(ScenarioRejected, match="strategy.params"):
            _loader_load(raw)

    def test_unknown_invariant_target_rejected(self) -> None:
        raw = self._raw(invariant_check_on=["no-such-participant"])
        with pytest.raises(ScenarioRejected, match="invariant_check_on"):
            _loader_load(raw)

    def test_ppf_with_survival_region_rejected(self) -> None:
        raw = {
            "schema_version": "1.0",
            "comparison_group": "ppf-survival",
            "distribution": normal_distribution().identity(),
            "subject": normal_participant(),
            "references": [{"source_id": "scipy-norm-cdf"}],
            "queries": [{"characteristic": "ppf"}],
            "regions": [{"name": "tail", "interval": [0.9, 1.0], "space": "survival"}],
        }
        with pytest.raises(ScenarioRejected, match="ppf"):
            _loader_load(raw)

    def test_non_scalar_sweep_value_rejected(self) -> None:
        raw = self._raw(sweep_spec={"param": "distribution.parameters", "values": [{"a": 1}]})
        with pytest.raises(ScenarioRejected, match="not a scalar"):
            ScenarioExpander().expand(raw)

    def test_peer_role_in_subject_mode_rejected(self) -> None:
        raw = self._raw(
            participants=[
                {**normal_participant(), "role": "subject"},
                {"source_id": "scipy-norm-cdf", "role": "peer"},
            ],
        )
        raw.pop("subject", None)
        raw.pop("references", None)
        with pytest.raises(ScenarioRejected, match="peer"):
            _loader_load(raw)

    def test_duplicate_reference_identity_in_subject_mode_rejected(self) -> None:
        raw = self._raw()
        raw["references"] = [normal_participant(), normal_participant()]
        with pytest.raises(ScenarioRejected, match="duplicate reference identity"):
            _loader_load(raw)

    def test_huge_interval_bound_rejected(self) -> None:
        raw = self._raw()
        raw["regions"] = [{"name": "bulk", "interval": [0, 10**400]}]
        with pytest.raises(ScenarioRejected, match="too large"):
            _loader_load(raw)

    def test_adapter_config_key_collision_rejected(self) -> None:
        raw = self._raw()
        raw["subject"] = {
            "source_id": "pysatl-core",
            "adapter_config": {
                1: "a",
                "1": "b",
            },
        }
        with pytest.raises(ScenarioRejected, match="duplicate key"):
            _loader_load(raw)


class TestEmptyRegionOutcome:
    """REV-2: a region outside the runtime support is an outcome, not a crash."""

    def test_region_outside_support_is_not_applicable(self) -> None:
        registry = SourceRegistry()
        registry.register(_BoundedSource("subj"))
        registry.register(_BoundedSource("ref"))
        raw = {
            "schema_version": "1.0",
            "comparison_group": "void-group",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "subj"},
            "references": [{"source_id": "ref"}],
            "queries": [{"characteristic": "cdf"}],
            "regions": [{"name": "void", "interval": [2.0, 3.0], "space": "probability"}],
        }
        results = AccuracyRunner(registry, ResultsStore()).run(_loader_load(raw))
        assert len(results) == 1
        assert results[0].status == Status.NOT_APPLICABLE
        assert "empty grid" in results[0].status_explanation
        assert results[0].metrics == ()


class TestStoreGuards:
    def test_duplicate_write_is_framework_defect(self) -> None:
        store = ResultsStore()
        store.write_result(_result())
        with pytest.raises(FrameworkDefect, match="duplicate"):
            store.write_result(_result())

    def test_nan_serialized_as_null(self, tmp_path: Path) -> None:
        store = ResultsStore(root=tmp_path)
        uri, _digest = store.put_sidecar(np.asarray([1.0, np.nan]), "probability")
        assert uri.startswith("sidecar:")
        files = list(tmp_path.glob("sidecar-*.json"))
        assert files
        payload = json.loads(files[0].read_text())
        assert payload["values"][1] is None

    def test_get_sidecar_returns_a_copy(self) -> None:
        store = ResultsStore()
        uri, _digest = store.put_sidecar(np.asarray([1.0, 2.0]), "probability")
        arr = store.get_sidecar(uri)
        assert arr is not None
        arr[0] = 99.0
        again = store.get_sidecar(uri)
        assert again is not None
        assert again[0] == 1.0


class TestAgreementSkipped:
    def test_unmatched_metrics_are_reported(self) -> None:
        def record(region: str, metrics: dict[str, float]) -> MetricRecord:
            return MetricRecord(
                characteristic="pdf",
                space="probability",
                region=region,
                grid_hash="sha256:x",
                metrics=metrics,
            )

        a = _result(metrics=(record("bulk", {"abs_error": 0.1, "lre": 1.0}),))
        b = _result(metrics=(record("tail", {"abs_error": 0.2, "lre": 1.0}),))
        agreement: Agreement = Comparator(ResultsStore()).agreement(a, b)
        assert agreement.entries == ()
        assert agreement.skipped and len(agreement.skipped) == 2

    def test_agreement_across_scenario_shells(self) -> None:
        """REV-5: same pair, different spec_hash — comparable (grid arbiter)."""
        record = MetricRecord(
            characteristic="pdf",
            space="probability",
            region="bulk",
            grid_hash="sha256:x",
            metrics={"abs_error": 0.1},
        )
        a = _result(metrics=(record,))
        b = _result(metrics=(record,))
        object.__setattr__(b, "spec_hash", "sha256:different")
        agreement = Comparator(ResultsStore()).agreement(a, b)
        assert agreement.entries and agreement.entries[0].diffs == {"abs_error": 0.0}


class TestDtoInvariants:
    def test_metric_record_has_no_p_value(self) -> None:
        """Инвариант 17 (К-7): MetricRecord — не контейнер других семейств."""
        fields = {f.name for f in MetricRecord.__dataclass_fields__.values()}
        for forbidden in ("p_value", "pit", "pit_record", "pvalues"):
            assert forbidden not in fields
        payload = MetricRecord(
            characteristic="pdf", space="probability", region="bulk", grid_hash="h"
        ).to_json_dict()
        assert "p_value" not in payload


class TestScenarioFilesLoad:
    def test_repo_scenario_files_load(self) -> None:
        """Инвариант 18 в действии: файлы репо-набора валидны для Loader'а."""
        directory = Path(__file__).resolve().parents[2] / "benchmarks" / "scenarios"
        files = sorted(directory.glob("*.json"))
        assert files, "scenario files must exist in the repository"
        expander = ScenarioExpander()
        for path in files:
            raw = json.loads(path.read_text(encoding="utf-8"))
            # Ред. 35: minimal files are named by their stem (run_accuracy rule).
            if "scenario_id" not in raw and "comparison_group" not in raw:
                raw["comparison_group"] = path.stem
            for expanded in expander.expand(raw):
                _loader_load(expanded)


class TestReportCoversStatuses:
    def test_unmeasured_row_reaches_report_text(self) -> None:
        registry = SourceRegistry()
        registry.register(CrashingSource())
        raw = {
            "schema_version": "1.0",
            "comparison_group": "crash-group",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "crashing"},
            "references": [{"source_id": "crashing"}],
            "pair_policy": "symmetric",
            "queries": [{"characteristic": "cdf"}],
            "regions": [{"name": "bulk", "interval": [-1.0, 1.0], "space": "probability"}],
        }
        store = ResultsStore()
        results = AccuracyRunner(registry, store).run(_loader_load(raw))
        assert results and results[0].status == Status.FAILED
        report = store.read_report("crash-group")
        assert report is not None
        assert "failed" in report.text
        assert "explanation:" in report.text
        assert "lre(min)" in report.text

    def test_report_row_summary_shows_worst_region_lre(self) -> None:
        """The per-row summary column carries the worst region (min LRE).

        The convention is fixed by REV-21 (CLI summary: "abs_err max,
        LRE min") — a pair is as good as its worst region; the store report
        used to show max(lre), the best region, contradicting it.
        """
        from pysatl_core_bench.accuracy.dto import ComparisonTable
        from pysatl_core_bench.accuracy.reporting.report import Reporter

        def result_with(lre_a: float, lre_b: float) -> AccuracyResult:
            records = (
                MetricRecord(
                    characteristic="cdf",
                    space="probability",
                    region="bulk",
                    grid_hash="h1",
                    metrics={"lre": lre_a, "abs_error": 0.0},
                ),
                MetricRecord(
                    characteristic="cdf",
                    space="probability",
                    region="tail",
                    grid_hash="h2",
                    metrics={"lre": lre_b, "abs_error": 0.0},
                ),
            )
            return AccuracyResult(
                schema_version="1.0",
                scenario_id="s",
                spec_hash="h",
                comparison_group_id="g",
                sort_key=None,
                pair=("a", "b"),
                subject_id="a",
                reference="b",
                role_labels={},
                status=Status.MEASURED,
                status_explanation="",
                metrics=records,
            )

        table = ComparisonTable(comparison_group_id="g", rows=(result_with(12.0, 4.5),))
        text = Reporter._render(table, {"measured": 1}, None)
        assert "lre(min)" in text
        assert "4.500" in text
        assert "12.000" not in text


class TestWrapperIdentity:
    def test_scipy_wrapper_refuses_foreign_identity(self) -> None:
        source = scipy_norm_source()
        distribution = DistributionSpec(family="Exponential")
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="scipy-norm",
            participant_id="s#x",
        )
        query = AccuracyQuery(**cast("dict[str, Any]", {"characteristic": "pdf"}))
        prepared = source.prepare(participant, query, distribution)
        with pytest.raises(NotImplementedError):
            source.evaluate(prepared, np.zeros(3))
        assert Capability.COMPUTE in source.capabilities


class _OverConfidentCdf(ValueSource):
    """A compute source whose cdf returns 2.0 — breaks the cdf invariant."""

    def __init__(self, source_id: str = "badcdf") -> None:
        self._id = source_id
        self.assess_calls = 0

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

    def support(self, participant: ParticipantSpec, distribution: Any) -> SupportBounds:
        return SupportBounds(left=-3.0, right=3.0)

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: Any,
        points: Any = None,
    ) -> SupportVerdict:
        self.assess_calls += 1
        return SupportVerdict(readiness=Readiness.READY, reason="overconfident")

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        return ["probability"]

    def prepare(
        self, participant: ParticipantSpec, query: AccuracyQuery, distribution: Any
    ) -> PreparedSubject:
        return PreparedSubject(participant_id=participant.participant_id, query=query)

    def evaluate(self, prepared: PreparedSubject, points: Any) -> Any:
        arr = np.asarray(points, dtype=np.float64)
        return np.full(arr.shape, 2.0)

    def compute(
        self, participant: ParticipantSpec, query: AccuracyQuery, points: Any, distribution: Any
    ) -> Any:
        return np.full(np.asarray(points, dtype=np.float64).shape, 2.0)


class TestRev2Series:
    """REV-16..REV-20 и минимальный сценарий (ред. 35)."""

    def test_rev16_group_wide_store_write(self, registry: SourceRegistry) -> None:
        """Инвариант 15 (А-5) для всей группы: строки store согласованы с отчётом.

        Сценарий A ловит инвариант-нарушение участника badcdf#h; сценарий B
        той же группы измеряет пару с тем же участником и без чека.  Store
        обязан получить строки B уже после группового spread — invalid_reference.
        """
        bad = _OverConfidentCdf()
        registry.register(bad)
        scenario_a = {
            "comparison_group": "spread-group",
            "scenario_id": "A",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "badcdf"},
            "references": [normal_participant()],
            "queries": [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0]}],
            "invariant_check_on": ["subject"],
        }
        scenario_b = {
            "comparison_group": "spread-group",
            "scenario_id": "B",
            "distribution": normal_distribution().identity(),
            "subject": normal_participant(),
            "references": [{"source_id": "badcdf"}],
            "queries": [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0]}],
        }
        store = ResultsStore()
        runner = AccuracyRunner(registry, store)
        results = runner.run_group([_loader_load(scenario_a), _loader_load(scenario_b)])
        assert [r.scenario_id for r in results] == ["A", "B"]
        stored = {r.scenario_id: r for r in store.read_group("spread-group")}
        assert set(stored) == {"A", "B"}
        for row in stored.values():
            assert row.status == Status.INVALID_REFERENCE, row.status_explanation
            assert any(v.participant_id.startswith("badcdf") for v in row.invariant_violations)
        assert (
            stored["B"].status_explanation.endswith("invariant check disproved participant(s)")
            or "invariant check disproved" in stored["B"].status_explanation
        )

    def test_rev18_verdicts_assessed_once_per_participant(self, registry: SourceRegistry) -> None:
        """REV-18: k участников symmetric-режима assessed ровно один раз."""
        bad = _OverConfidentCdf()  # готовый ValueSource-стаб с конечными значениями
        # values 2.0 конечны и сравнимы: agreement-режим, инвариант-чек выключен.
        registry.register(bad)
        raw = {
            "comparison_group": "assess-count",
            "pair_policy": "symmetric",
            "distribution": normal_distribution().identity(),
            "participants": [
                {"role": "peer", "source_id": "badcdf", "adapter_config": {"tag": i}}
                for i in range(3)
            ],
            "queries": [{"characteristic": "cdf", "grid_hint": {"n_points": 9}}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0]}],
        }
        results = AccuracyRunner(registry, ResultsStore()).run(_loader_load(raw))
        assert len(results) == 3
        assert all(r.status == Status.MEASURED for r in results)
        assert bad.assess_calls == 3, bad.assess_calls

    def test_rev19_special_points_manifest_is_serialized(self, registry: SourceRegistry) -> None:
        """REV-19: манифест предсказанных особых точек едет в результате."""

        class _FixedDetector(SpecialPointDetector):
            def detect(
                self, distribution: Any, support: SupportBounds | None
            ) -> list[SpecialPoint]:
                return [SpecialPoint(x=0.5, kind="pdf_jump", source="curated_rules")]

        raw = cdf_scenario_raw()
        store = ResultsStore()
        results = AccuracyRunner(registry, store, detector=_FixedDetector()).run(_loader_load(raw))
        assert results[0].special_points[0].x == 0.5
        payload = results[0].to_json_dict()
        assert payload["special_points"] == [
            {"x": 0.5, "kind": "pdf_jump", "source": "curated_rules"}
        ]

    def test_rev20_mapping_fields_are_frozen(self) -> None:
        """REV-20: Mapping-поля frozen-DTO не мутируются после загрузки."""
        from pysatl_core_bench.accuracy.canonical import content_hash, frozen_mapping

        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="s",
            participant_id="s#1",
            adapter_config={"a": {"b": 1.0}},
        )
        # mypy: статическое присваивание в Mapping запрещено типизацией; для
        # проверки рантайм-заморозки приводим к dict (значения не меняются).
        params_typed = cast("dict[str, Any]", participant.adapter_config)
        with pytest.raises(TypeError):
            params_typed["c"] = 1.0
        nested_typed = cast("dict[str, Any]", participant.adapter_config["a"])
        with pytest.raises(TypeError):
            nested_typed["b"] = 2.0
        query = AccuracyQuery(
            **cast("dict[str, Any]", {"characteristic": "pdf", "grid_hint": {"n_points": 9}})
        )
        hint_typed = cast("dict[str, int] | None", query.grid_hint)
        assert hint_typed is not None
        with pytest.raises(TypeError):
            hint_typed["n_points"] = 10
        # Заморозка не меняет каноническую идентичность (хэш).
        assert content_hash({"p": {"a": 1.0}}) == content_hash({"p": frozen_mapping({"a": 1.0})})
        # Provenance с замороженной стратегией сериализуется в строгий JSON.
        provenance = Provenance(library="l", library_version="1", strategy={"name": "x"})
        json.dumps(provenance.to_json_dict())

    def test_minimal_scenario_defaults(self) -> None:
        """Ред. 35: schema_version/comparison_group опциональны, дефолты материализуются."""
        base: dict[str, Any] = {
            "scenario_id": "minimal",
            "distribution": normal_distribution().identity(),
            "subject": normal_participant(),
            "references": [{"source_id": "scipy-norm"}],
            "queries": [{"characteristic": "pdf"}],
        }
        spec = _loader_load(dict(base))
        assert spec.schema_version == "1.0"
        assert spec.comparison_group == "minimal"
        assert spec.regions[0].name == "bulk"  # дефолтный регион
        # только comparison_group: scenario_id схлопывается в группу
        no_id = {k: v for k, v in base.items() if k != "scenario_id"}
        no_id["comparison_group"] = "by-group"
        spec2 = _loader_load(no_id)
        assert spec2.scenario_id == "by-group" == spec2.comparison_group
        # нет ни scenario_id, ни comparison_group -> Rejected
        nameless = {k: v for k, v in base.items() if k != "scenario_id"}
        with pytest.raises(ScenarioRejected):
            _loader_load(nameless)
        # чужой schema_version по-прежнему Rejected; отсутствие — дефолт
        wrong = dict(base, schema_version="9.9")
        with pytest.raises(ScenarioRejected):
            _loader_load(wrong)

    def test_minimal_scenario_file_with_injected_name(self) -> None:
        """S1.0_minimal.json: имя файла становится группой (правило run_accuracy)."""
        path = (
            Path(__file__).resolve().parents[2] / "benchmarks" / "scenarios" / "S1.0_minimal.json"
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" not in raw and "comparison_group" not in raw
        raw["comparison_group"] = path.stem
        spec = _loader_load(raw)
        assert spec.comparison_group == "S1.0_minimal"
        assert spec.scenario_id == "S1.0_minimal"

    def test_deferred_characteristic_is_an_honest_outcome(self, registry: SourceRegistry) -> None:
        """08 §2.3: запрос характеристики вне графа (момент) — не краш, а not_applicable."""
        raw = {
            "comparison_group": "deferred-mean",
            "distribution": normal_distribution().identity(),
            "subject": normal_participant(),
            "references": [normal_participant()],
            "queries": [{"characteristic": "mean"}],
            "regions": [{"name": "bulk", "interval": [-2.0, 2.0]}],
        }
        results = AccuracyRunner(registry, ResultsStore()).run(_loader_load(raw))
        assert len(results) == 1
        # Скалярная характеристика не измеряется на сетке: явный отказ адаптера
        # (М-6), а не краш evaluate и не вводящий в заблуждение failed.
        assert results[0].status == Status.UNSUPPORTED_CONFIGURATION
        assert "scalar characteristic" in results[0].status_explanation

    def test_rev17_on_demand_cache_knobs(self) -> None:
        """REV-17: executor-ручки кэша — cache_info/cache_clear."""
        from pysatl_core_bench import OnDemandComputer

        source = OnDemandComputer(scipy_norm_source(), source_id="on-demand-test")
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="on-demand-test",
            participant_id="od#1",
        )
        query = AccuracyQuery(**cast("dict[str, Any]", {"characteristic": "pdf"}))
        prepared = source.prepare(participant, query, normal_distribution())
        source.evaluate(prepared, np.linspace(-1.0, 1.0, 5))
        assert source.cache_info()["entries"] == 1
        source.cache_clear()
        assert source.cache_info()["entries"] == 0
