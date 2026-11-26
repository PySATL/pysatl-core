"""Value sources, registry and capability rules (C1, D5, E)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy import stats

from pysatl_core.types import CharacteristicName, NumericArray
from pysatl_core_bench import (
    AccuracyQuery,
    AdapterConfigurationError,
    DistributionSpec,
    ExternalFunctionSource,
    OnDemandComputer,
    ParticipantSpec,
    PrecomputedTable,
    Provenance,
    PySATLAdapter,
    Role,
    SourceNotRegistered,
    SourceOperationError,
    SourceRegistry,
    SupportBounds,
    TableLookupSource,
)
from pysatl_core_bench.accuracy.computations.protocol import PreparedSubject
from pysatl_core_bench.accuracy.computations.sources import _MAX_REPORTED_MISSING, TABLE_SCHEMA
from pysatl_core_bench.accuracy.types import Capability, Readiness


def make_distribution(**parameters: object) -> DistributionSpec:
    return DistributionSpec(
        family="Normal",
        parametrization="meanStd",
        parameters={"mu": 0.0, "sigma": 1.0, **parameters},
    )


def make_participant(
    source_id: str = "pysatl-core", role: Role = Role.REFERENCE
) -> ParticipantSpec:
    return ParticipantSpec(
        role=role,
        source_id=source_id,
        participant_id=f"{source_id}#test",
    )


def make_query(characteristic: str = "cdf") -> AccuracyQuery:
    return AccuracyQuery(characteristic=CharacteristicName(characteristic))


def make_table(n: int = 41, tol: float = 1e-12) -> TableLookupSource:
    points = np.linspace(-3.0, 3.0, n)
    return TableLookupSource(
        "tbl",
        PrecomputedTable(
            characteristic="cdf",
            output_scale="probability",
            points=points,
            values=stats.norm.cdf(points),
            coverage_tol=tol,
        ),
    )


class TestRegistry:
    def test_resolve_unknown_raises(self) -> None:
        with pytest.raises(SourceNotRegistered):
            SourceRegistry().resolve("nope")

    def test_register_and_resolve(self) -> None:
        reg = SourceRegistry()
        adapter = PySATLAdapter()
        reg.register(adapter)
        assert reg.resolve("pysatl-core") is adapter

    def test_subject_must_compute(self) -> None:
        table = make_table()
        assert Capability.COMPUTE not in table.capabilities
        assert not SourceRegistry.check_role_capabilities(table.capabilities, as_subject=True)
        assert SourceRegistry.check_role_capabilities(table.capabilities, as_subject=False)


class TestTableLookupSource:
    def test_lookup_ready_and_missing_reported(self) -> None:
        table = make_table()
        query = make_query()
        participant = make_participant("tbl")
        distribution = make_distribution()
        inside = np.array([-3.0, 0.0, 3.0])
        verdict = table.assess(participant, query, distribution, inside)
        assert verdict.readiness == Readiness.READY and not verdict.missing
        outside = np.array([5.0, 6.0])
        verdict = table.assess(participant, query, distribution, outside)
        assert verdict.readiness == Readiness.NOT_COVERED
        assert verdict.missing == (5.0, 6.0)
        partial = np.array([0.0, 99.0])
        verdict = table.assess(participant, query, distribution, partial)
        assert verdict.readiness == Readiness.READY and verdict.missing == (99.0,)

    def test_wrong_characteristic_is_not_applicable(self) -> None:
        verdict = make_table().assess(
            make_participant("tbl"), make_query("pdf"), make_distribution(), None
        )
        assert verdict.readiness == Readiness.NOT_APPLICABLE

    def test_evaluate_returns_nan_for_holes(self) -> None:
        table = make_table()
        prepared = table.prepare(make_participant("tbl"), make_query(), make_distribution())
        values = table.evaluate(prepared, np.array([-3.0, 100.0]))
        assert np.isfinite(values[0]) and np.isnan(values[1])

    def test_compute_is_refused(self) -> None:
        with pytest.raises(SourceOperationError):
            make_table().compute(
                make_participant("tbl"), make_query(), np.zeros(1), make_distribution()
            )

    def test_participant_strategy_is_unsupported_configuration(self) -> None:
        """A lookup source cannot honor strategies: unsupported, not n/a (M-6/REV-5)."""
        participant = ParticipantSpec(
            role=Role.REFERENCE,
            source_id="tbl",
            participant_id="tbl#s",
            strategy={"name": "ForcedGraphStrategy", "params": {"from": "cf"}},
        )
        verdict = make_table().assess(participant, make_query(), make_distribution(), None)
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION
        assert "strategies" in verdict.reason

    def test_table_hash_binds_content(self) -> None:
        assert make_table().provenance.table_hash == make_table().provenance.table_hash
        assert make_table(n=17).provenance.table_hash != make_table(n=19).provenance.table_hash


class TestOnDemandComputer:
    def test_capabilities_and_caching(self) -> None:
        table = make_table()
        wrapped = OnDemandComputer(PySATLAdapter(), source_id="on-demand(core)")
        assert wrapped.capabilities == frozenset({Capability.COMPUTE, Capability.LOOKUP})
        participant = make_participant("pysatl-core")
        query = make_query()
        distribution = make_distribution()
        prepared = wrapped.prepare(participant, query, distribution)
        points = np.linspace(-2.0, 2.0, 21)
        first = wrapped.evaluate(prepared, points)
        second = wrapped.evaluate(prepared, points)
        np.testing.assert_array_equal(first, second)
        verdict = wrapped.assess(participant, query, distribution, points)
        assert verdict.readiness == Readiness.READY
        assert "cache" in verdict.reason
        _ = table

    def test_wrapping_a_table_extends_it(self) -> None:
        # OnDemand over a table still cannot compute new points (honest
        # delegation), but the cache path serves repeated grids.
        source = OnDemandComputer(make_table())
        participant = make_participant("tbl")
        query = make_query()
        points = np.linspace(-3.0, 3.0, 41)
        verdict = source.assess(participant, query, make_distribution(), points)
        assert verdict.readiness == Readiness.READY

    def test_cache_key_is_order_sensitive(self) -> None:
        # A-3 ("no silence"): the same point set in another order must
        # receive values aligned to *that* order, never the first order's
        # array.  The key uses order-sensitive values_hash, not grid_hash.
        calls: list[list[float]] = []

        def func(distribution, participant, query, points):
            calls.append([float(x) for x in np.asarray(points).ravel()])
            return np.asarray(points, dtype=np.float64) * 2.0

        inner = ExternalFunctionSource(
            "inner",
            func,
            provenance=Provenance(library="fake", library_version="0"),
        )
        source = OnDemandComputer(inner, source_id="od")
        participant = make_participant("p")
        query = make_query()
        prepared = source.prepare(participant, query, make_distribution())
        first = source.evaluate(prepared, np.array([1.0, 2.0, 3.0]))
        second = source.evaluate(prepared, np.array([3.0, 2.0, 1.0]))
        np.testing.assert_array_equal(first, [2.0, 4.0, 6.0])
        np.testing.assert_array_equal(second, [6.0, 4.0, 2.0])
        assert len(calls) == 2  # permuted request is a cache miss, not a swap

    def test_cache_key_is_distribution_sensitive(self) -> None:
        # rev. 40: one shared source serves whole sweeps whose scenarios
        # differ only in the scenario-level distribution; the cache key
        # must never serve one distribution's values for another.
        calls: list[float] = []

        def func(distribution, participant, query, points):
            sigma = float(distribution.parameters["sigma"])
            calls.append(sigma)
            return np.asarray(points, dtype=np.float64) * sigma

        inner = ExternalFunctionSource(
            "inner",
            func,
            provenance=Provenance(library="fake", library_version="0"),
        )
        source = OnDemandComputer(inner, source_id="od")
        participant = make_participant("p")
        query = make_query()
        points = np.array([1.0, 2.0])
        first = source.evaluate(
            source.prepare(participant, query, make_distribution(sigma=1.0)), points
        )
        second = source.evaluate(
            source.prepare(participant, query, make_distribution(sigma=2.0)), points
        )
        np.testing.assert_array_equal(first, [1.0, 2.0])
        np.testing.assert_array_equal(second, [2.0, 4.0])
        assert calls == [1.0, 2.0]  # the second distribution is a cache miss


class TestPySATLAdapter:
    def test_support_bounds(self) -> None:
        adapter = PySATLAdapter()
        bounds = adapter.support(make_participant(role=Role.SUBJECT), make_distribution())
        assert bounds.left == float("-inf") and bounds.right == float("inf")

    def test_unknown_family_is_unsupported_not_failed(self) -> None:
        adapter = PySATLAdapter()
        distribution = DistributionSpec(family="Laplace", parameters={})
        verdict = adapter.assess(
            make_participant(role=Role.SUBJECT), make_query(), distribution, None
        )
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION

    def test_unknown_strategy_is_unsupported_no_fallback(self) -> None:
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            strategy="NoSuchStrategy",
        )
        verdict = adapter.assess(participant, make_query(), make_distribution(), None)
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION
        assert "NoSuchStrategy" in verdict.reason

    def test_forced_graph_strategy_from_cf_assesses_ready(self) -> None:
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            strategy={"name": "ForcedGraphStrategy", "params": {"from": "cf"}},
        )
        query = make_query("pdf")
        verdict = adapter.assess(
            participant, query, make_distribution(), np.linspace(-3.0, 3.0, 61)
        )
        assert verdict.readiness == Readiness.READY

    def test_forced_graph_strategy_plan_and_accuracy(self) -> None:
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            strategy={"name": "ForcedGraphStrategy", "params": {"from": "cf"}},
        )
        prepared = adapter.prepare(participant, make_query("pdf"), make_distribution())
        assert prepared.path_dump is not None
        steps = [(s["target"], s["method_name"]) for s in prepared.path_dump["steps"]]
        assert steps == [("cf", "cf"), ("pdf", "cf_to_pdf_1C")]
        xs = np.linspace(-5.0, 5.0, 33)
        values = adapter.evaluate(prepared, xs)
        np.testing.assert_allclose(values, stats.norm.pdf(xs), atol=1e-12)

    def test_forced_graph_strategy_unknown_start_is_unsupported(self) -> None:
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            strategy={"name": "ForcedGraphStrategy", "params": {"from": "quux"}},
        )
        verdict = adapter.assess(participant, make_query("pdf"), make_distribution(), None)
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION
        assert "quux" in verdict.reason

    def test_forced_graph_strategy_route_not_found_is_unsupported(self) -> None:
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            strategy={"name": "ForcedGraphStrategy", "params": {"from": "mean"}},
        )
        verdict = adapter.assess(participant, make_query("pdf"), make_distribution(), None)
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION
        assert "no computation route" in verdict.reason

    def test_default_strategy_is_accepted(self) -> None:
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            strategy={"name": "DefaultComputationStrategy", "params": {}},
        )
        assert (
            adapter.assess(participant, make_query(), make_distribution(), None).readiness
            == Readiness.READY
        )

    def test_strategy_with_params_is_refused(self) -> None:
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            strategy={"name": "DefaultComputationStrategy", "params": {"x": 1}},
        )
        verdict = adapter.assess(participant, make_query(), make_distribution(), None)
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION

    def test_adapter_config_is_refused_no_silent_knobs(self) -> None:
        """rev. 40: the v1 pysatl-core adapter accepts no adapter_config."""
        adapter = PySATLAdapter()
        participant = ParticipantSpec(
            role=Role.SUBJECT,
            source_id="pysatl-core",
            participant_id="x",
            adapter_config={"table_path": "x.json"},
        )
        with pytest.raises(AdapterConfigurationError):
            adapter.support(participant, make_distribution())

    def test_path_dump_serializes_plan(self) -> None:
        adapter = PySATLAdapter()
        participant = make_participant(role=Role.SUBJECT)
        query = make_query("pdf")
        prepared = adapter.prepare(participant, query, make_distribution())
        assert prepared.path_dump is not None
        assert "steps" in prepared.path_dump


class TestExternalFunctionSource:
    """The generic wrapper for one-dimensional functions of other libraries."""

    @staticmethod
    def _source() -> ExternalFunctionSource:
        def _square(
            distribution: DistributionSpec,
            participant: ParticipantSpec,
            query: AccuracyQuery,
            points: NumericArray,
        ) -> ArrayLike:
            return np.asarray(points, dtype=np.float64) ** 2

        return ExternalFunctionSource(
            "ext",
            _square,
            provenance=Provenance(library="extlib", library_version="1.2.3"),
            characteristics=("pdf", "cdf"),
        )

    def test_subject_id_defaults_to_library_identity(self) -> None:
        source = self._source()
        assert source.subject_id == "extlib@1.2.3"
        assert source.source_id == "ext"
        assert Capability.COMPUTE in source.capabilities

    def test_assess_filters_characteristics(self) -> None:
        source = self._source()
        participant = ParticipantSpec(role=Role.REFERENCE, source_id="ext", participant_id="p")
        distribution = make_distribution()
        ready = source.assess(participant, make_query("pdf"), distribution)
        alien = source.assess(participant, make_query("ppf"), distribution)
        assert ready.readiness == Readiness.READY
        assert alien.readiness == Readiness.NOT_APPLICABLE

    def test_assess_refuses_strategies(self) -> None:
        source = self._source()
        participant = ParticipantSpec(
            role=Role.REFERENCE,
            source_id="ext",
            participant_id="p",
            strategy={"name": "ForcedGraphStrategy", "params": {"from": "cf"}},
        )
        verdict = source.assess(participant, make_query("pdf"), make_distribution())
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION

    def test_compute_and_evaluate_match(self) -> None:
        source = self._source()
        participant = ParticipantSpec(role=Role.REFERENCE, source_id="ext", participant_id="p")
        query = make_query("pdf")
        distribution = make_distribution()
        points = np.array([1.0, -2.0, 3.0])
        prepared = source.prepare(participant, query, distribution)
        assert np.array_equal(source.evaluate(prepared, points), points**2)
        assert np.array_equal(source.compute(participant, query, points, distribution), points**2)

    def test_shape_mismatch_is_source_error(self) -> None:
        def _scalar(
            distribution: DistributionSpec,
            participant: ParticipantSpec,
            query: AccuracyQuery,
            points: NumericArray,
        ) -> ArrayLike:
            return 1.0

        source = ExternalFunctionSource(
            "bad",
            _scalar,
            provenance=Provenance(library="bad", library_version="0"),
        )
        participant = ParticipantSpec(role=Role.REFERENCE, source_id="bad", participant_id="p")
        with pytest.raises(SourceOperationError):
            source.compute(
                participant, make_query("pdf"), np.array([1.0, 2.0]), make_distribution()
            )

    def test_support_defaults_to_real_line(self) -> None:
        source = self._source()
        participant = ParticipantSpec(role=Role.REFERENCE, source_id="ext", participant_id="p")
        bounds = source.support(participant, make_distribution())
        assert bounds == SupportBounds(
            left=float("-inf"), right=float("inf"), left_closed=False, right_closed=False
        )


# Repository root: this file lives in tests/bench/computations/.
TABLE_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "reference_tables"


def make_table_data(n: int = 41) -> PrecomputedTable:
    points = np.linspace(-3.0, 3.0, n)
    return PrecomputedTable(
        characteristic="cdf",
        output_scale="probability",
        points=points,
        values=stats.norm.cdf(points),
        library="scipy",
        library_version="1.15",
        family="Normal",
        parametrization="meanStd",
        parameters={"mu": 0.0, "sigma": 1.0},
        dps=None,
        coverage_tol=1e-12,
    )


class TestTableJsonRoundTrip:
    def test_round_trip_preserves_content(self) -> None:
        table = make_table_data()
        restored = PrecomputedTable.from_json_dict(table.to_json_dict())
        assert restored.table_hash == table.table_hash
        assert restored.characteristic == "cdf"
        assert restored.output_scale == "probability"
        assert restored.family == "Normal"
        assert restored.parametrization == "meanStd"
        assert restored.parameters == {"mu": 0.0, "sigma": 1.0}
        assert np.array_equal(restored.points, table.points)
        assert np.array_equal(restored.values, table.values)

    def test_file_round_trip(self, tmp_path: Path) -> None:
        table = make_table_data()
        path = tmp_path / "table.json"
        path.write_text(json.dumps(table.to_json_dict()), encoding="utf-8")
        assert PrecomputedTable.from_json(path).table_hash == table.table_hash

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("schema", "pysatl-core-bench.table/0"),
            ("characteristic", ""),
            ("output_scale", "log"),
            ("dps", "50"),
            ("coverage_tol", "1e-12"),
        ],
        ids=["schema", "characteristic", "output_scale", "dps", "coverage_tol"],
    )
    def test_invalid_fields_are_rejected(self, field: str, bad_value: object) -> None:
        payload = make_table_data().to_json_dict()
        payload[field] = bad_value
        with pytest.raises(ValueError):
            PrecomputedTable.from_json_dict(payload)

    def test_length_mismatch_is_rejected(self) -> None:
        payload = make_table_data().to_json_dict()
        payload["values"] = payload["values"][:-1]
        with pytest.raises(ValueError):
            PrecomputedTable.from_json_dict(payload)

    def test_non_finite_value_is_rejected(self) -> None:
        payload = make_table_data().to_json_dict()
        payload["values"][0] = float("nan")
        with pytest.raises(ValueError):
            PrecomputedTable.from_json_dict(payload)

    def test_distribution_identity_mismatch_is_not_applicable(self) -> None:
        """A table must never serve numbers of another distribution identity."""
        source = TableLookupSource("tbl", make_table_data())
        shifted = make_distribution(mu=2.0)
        verdict = source.assess(make_participant("tbl"), make_query(), shifted, None)
        assert verdict.readiness == Readiness.NOT_APPLICABLE
        assert "identity" in verdict.reason


class TestShippedReferenceTables:
    """The mpmath production tables ship with the repo and must stay valid."""

    def test_tables_exist(self) -> None:
        assert sorted(TABLE_DIR.glob("normal_*.json")), f"missing tables in {TABLE_DIR}"

    def test_tables_cover_scenario_grids_exactly(self) -> None:
        """Grids of the shipped scenarios land on the table lattice exactly."""
        for path in sorted(TABLE_DIR.glob("normal_*.json")):
            table = PrecomputedTable.from_json(path)
            assert table.coverage_tol == 0.0
            source = TableLookupSource(f"mpmath-normal-{table.characteristic}", table)
            grid = np.linspace(-13.0, 13.0, 257)
            verdict = source.assess(
                make_participant(), make_query(table.characteristic), make_distribution(), grid
            )
            assert verdict.readiness == Readiness.READY, f"{path.name}: {verdict.reason}"
            assert verdict.missing == ()

    def test_values_match_scipy_to_float64_rounding(self) -> None:
        for path in sorted(TABLE_DIR.glob("normal_*.json")):
            table = PrecomputedTable.from_json(path)
            reference = (
                stats.norm.pdf(table.points)
                if table.characteristic == "pdf"
                else stats.norm.cdf(table.points)
            )
            assert table.dps is not None and table.dps >= 50
            np.testing.assert_allclose(table.values, reference, atol=5e-16)


class TestTableValidationMatrix:
    """PrecomputedTable JSON validation (TABLE_SCHEMA): a table is data, so
    every schema rule is a pinned refusal, not a warning."""

    @staticmethod
    def payload(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "schema": TABLE_SCHEMA,
            "characteristic": "cdf",
            "output_scale": "probability",
            "points": [0.0, 1.0],
            "values": [0.5, 0.84],
        }
        base.update(overrides)
        return base

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"schema": "other/1"}, "schema"),
            ({"characteristic": ""}, "characteristic"),
            ({"characteristic": 5}, "characteristic"),
            ({"output_scale": "hazard"}, "output_scale"),
            ({"parameters": [1]}, "parameters"),
            ({"dps": "50"}, "dps"),
            ({"coverage_tol": "big"}, "coverage_tol"),
            ({"points": []}, "non-empty"),
            ({"points": [0.0], "values": [0.5, 0.84]}, "equal length"),
            ({"points": ["a", 1.0]}, "numbers only"),
            ({"points": [float("nan"), 1.0]}, "finite"),
            ({"points": "nope"}, "non-empty list"),
        ],
    )
    def test_invalid_payloads_are_rejected(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            PrecomputedTable.from_json_dict(self.payload(**overrides))

    def test_table_sorts_points_ascending(self) -> None:
        table = PrecomputedTable(
            characteristic="cdf",
            output_scale="probability",
            points=np.asarray([1.0, 0.0]),
            values=np.asarray([0.84, 0.5]),
        )
        assert list(table.points) == [0.0, 1.0]
        assert list(table.values) == [0.5, 0.84]
        # the sort is part of the data identity, not a hidden reorder of
        # the file: values travel with their points
        assert table.values[1] == 0.84

    def test_table_hash_is_content_addressed(self) -> None:
        a = PrecomputedTable(
            characteristic="cdf",
            output_scale="probability",
            points=np.asarray([0.0]),
            values=np.asarray([0.5]),
        )
        b = PrecomputedTable(
            characteristic="cdf",
            output_scale="probability",
            points=np.asarray([0.0]),
            values=np.asarray([0.5]),
        )
        assert a.table_hash == b.table_hash


class TestTableLookupSourceVerdicts:
    """Assess verdicts of the lookup service (M-6, D6, rev. 40)."""

    def test_strategy_on_a_table_is_unsupported(self) -> None:
        source = make_table()
        participant = ParticipantSpec(
            role=Role.REFERENCE,
            source_id="t",
            participant_id="t#x",
            adapter_config={},
            strategy={"name": "ForcedGraphStrategy"},
        )
        verdict = source.assess(participant, make_query(), make_distribution())
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION

    def test_wrong_family_is_not_applicable(self) -> None:
        pinned = TableLookupSource(
            "pinned-family",
            PrecomputedTable(
                characteristic="cdf",
                output_scale="probability",
                points=np.linspace(-3.0, 3.0, 41),
                values=stats.norm.cdf(np.linspace(-3.0, 3.0, 41)),
                family="Normal",
            ),
        )
        verdict = pinned.assess(
            make_participant(),
            make_query(),
            DistributionSpec(family="Cauchy", parametrization="meanStd", parameters={}),
        )
        assert verdict.readiness == Readiness.NOT_APPLICABLE

    def test_wrong_parameter_value_is_not_applicable(self) -> None:
        # the identity check only fires for fields the table actually pins
        pinned = TableLookupSource(
            "pinned",
            PrecomputedTable(
                characteristic="cdf",
                output_scale="probability",
                points=np.linspace(-3.0, 3.0, 41),
                values=stats.norm.cdf(np.linspace(-3.0, 3.0, 41)),
                family="Normal",
                parametrization="meanStd",
                parameters={"mu": 0.0, "sigma": 1.0},
            ),
        )
        verdict = pinned.assess(
            make_participant(),
            make_query(),
            DistributionSpec(family="Normal", parametrization="meanStd", parameters={"mu": 5.0}),
        )
        assert verdict.readiness == Readiness.NOT_APPLICABLE

    def test_wrong_characteristic_is_not_applicable(self) -> None:
        source = make_table()
        verdict = source.assess(make_participant(), make_query("pdf"), make_distribution())
        assert verdict.readiness == Readiness.NOT_APPLICABLE

    def test_wrong_output_scale_is_not_applicable(self) -> None:
        source = make_table()
        verdict = source.assess(
            make_participant(),
            AccuracyQuery(characteristic=CharacteristicName.CDF, output_scale="survival"),
            make_distribution(),
        )
        assert verdict.readiness == Readiness.NOT_APPLICABLE

    def test_full_coverage_reports_ready_without_missing(self) -> None:
        source = make_table()  # lattice -3 + 0.15k, coverage_tol = 1e-12
        verdict = source.assess(
            make_participant(), make_query(), make_distribution(), np.asarray([0.0, 0.15])
        )
        assert verdict.readiness == Readiness.READY
        assert verdict.reason == "all requested points are covered"

    def test_no_coverage_is_not_covered_with_reported_points(self) -> None:
        source = make_table()  # table over [-3, 3]
        verdict = source.assess(
            make_participant(), make_query(), make_distribution(), np.asarray([50.0, 60.0])
        )
        assert verdict.readiness == Readiness.NOT_COVERED
        assert verdict.missing and len(verdict.missing) <= _MAX_REPORTED_MISSING

    def test_evaluate_fills_holes_with_nan(self) -> None:
        source = make_table()
        values = source.evaluate(
            source.prepare(make_participant(), make_query(), make_distribution()),
            np.asarray([0.0, 100.0]),
        )
        assert np.isfinite(values[0]) and not np.isfinite(values[1])

    def test_scale_mismatch_names_both_scales(self) -> None:
        source = make_table()
        verdict = source.assess(
            make_participant(),
            AccuracyQuery(characteristic=CharacteristicName.CDF, output_scale="survival"),
            make_distribution(),
        )
        assert verdict.readiness == Readiness.NOT_APPLICABLE
        assert "'probability'" in verdict.reason and "'survival'" in verdict.reason

    def test_compute_is_refused(self) -> None:
        with pytest.raises(SourceOperationError, match="no compute capability"):
            make_table().compute(
                make_participant(), make_query(), np.asarray([0.0]), make_distribution()
            )


class TestExternalFunctionSourceContract:
    """The generic wrapper: normalization, refusal rules, payload checks."""

    def test_declared_characteristics_are_enforced(self) -> None:
        source = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.zeros(np.asarray(x).shape),
            provenance=Provenance(library="ext", library_version="0"),
            characteristics=["pdf"],
        )
        verdict = source.assess(make_participant(), make_query(), make_distribution())
        assert verdict.readiness == Readiness.NOT_APPLICABLE

    def test_shape_mismatch_is_an_operation_error(self) -> None:
        source = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.zeros(3),
            provenance=Provenance(library="ext", library_version="0"),
        )
        prepared = source.prepare(make_participant(), make_query(), make_distribution())
        with pytest.raises(SourceOperationError, match="shape"):
            source.evaluate(prepared, np.asarray([0.0, 1.0]))

    def test_foreign_prepared_payload_is_refused(self) -> None:
        source = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.asarray(x),
            provenance=Provenance(library="ext", library_version="0"),
        )
        prepared = PreparedSubject(
            participant_id="x", query=make_query(), payload="not-a-participant"
        )
        with pytest.raises(SourceOperationError, match="payload"):
            source.evaluate(prepared, np.asarray([0.0]))


class TestOnDemandComputerCache:
    """B-5: reference values are deterministic, so the cache is shared and
    order-sensitive (a permuted request must never be served another
    order's array)."""

    def test_cache_hit_and_info(self) -> None:
        wrapped = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.asarray(x) * 2.0,
            provenance=Provenance(library="ext", library_version="0"),
        )
        source = OnDemandComputer(wrapped)
        prepared = source.prepare(make_participant(), make_query(), make_distribution())
        first = source.evaluate(prepared, np.asarray([1.0, 2.0]))
        assert source.cache_info()["entries"] == 1
        second = source.evaluate(prepared, np.asarray([1.0, 2.0]))
        assert np.array_equal(first, second)
        source.cache_clear()
        assert source.cache_info()["entries"] == 0

    def test_permuted_order_is_a_cache_miss(self) -> None:
        wrapped = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.asarray(x) * 2.0,
            provenance=Provenance(library="ext", library_version="0"),
        )
        source = OnDemandComputer(wrapped)
        prepared = source.prepare(make_participant(), make_query(), make_distribution())
        source.evaluate(prepared, np.asarray([1.0, 2.0]))
        source.evaluate(prepared, np.asarray([2.0, 1.0]))
        assert source.cache_info()["entries"] == 2

    def test_computes_pass_through_to_the_wrapped_source(self) -> None:
        wrapped = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.asarray(x) * 2.0,
            provenance=Provenance(library="ext", library_version="0"),
        )
        source = OnDemandComputer(wrapped)
        values = source.compute(
            make_participant(), make_query(), np.asarray([1.0]), make_distribution()
        )
        assert float(values[0]) == 2.0


class TestSourceSurface:
    """Public-source surface details (properties, error payloads)."""

    def test_table_subject_id_is_the_table_hash(self) -> None:
        source = make_table()
        assert source.subject_id.startswith("table:")

    def test_table_property_exposes_the_data(self) -> None:
        source = make_table()
        assert source.table.characteristic == "cdf"

    def test_plain_assess_without_points_is_ready(self) -> None:
        verdict = make_table().assess(make_participant(), make_query(), make_distribution())
        assert verdict.readiness == Readiness.READY
        assert verdict.reason == "table registered for this characteristic"

    def test_external_evaluate_without_distribution_is_refused(self) -> None:
        source = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.asarray(x),
            provenance=Provenance(library="ext", library_version="0"),
        )
        prepared = PreparedSubject(
            participant_id=make_participant().participant_id,
            query=make_query(),
            payload=make_participant(),
        )
        with pytest.raises(SourceOperationError, match="no distribution"):
            source.evaluate(prepared, np.asarray([0.0]))

    def test_on_demand_delegates_static_surface(self) -> None:
        wrapped = ExternalFunctionSource(
            "ext",
            lambda d, p, q, x: np.asarray(x),
            provenance=Provenance(library="ext", library_version="0"),
        )
        source = OnDemandComputer(wrapped, source_id="cache")
        assert source.source_id == "cache"
        assert source.subject_id == wrapped.subject_id
        assert source.provenance == wrapped.provenance
        assert source.support(make_participant(), make_distribution()) == wrapped.support(
            make_participant(), make_distribution()
        )

    def test_parallel_first_fill_keeps_one_entry(self) -> None:
        # A-9: two threads racing on the same uncached key must both get
        # correct values and the cache must keep exactly one entry (the
        # double-check branch inside the lock).
        import threading

        barrier = threading.Barrier(2, timeout=10)

        class SlowSource(ExternalFunctionSource):
            def evaluate(self, prepared, points):
                barrier.wait()  # both threads compute before either caches
                return super().evaluate(prepared, points)

        wrapped = SlowSource(
            "slow",
            lambda d, p, q, x: np.asarray(x) * 2.0,
            provenance=Provenance(library="ext", library_version="0"),
        )
        source = OnDemandComputer(wrapped)
        prepared = source.prepare(make_participant(), make_query(), make_distribution())
        results: list[Any] = []

        def worker() -> None:
            results.append(source.evaluate(prepared, np.asarray([1.0, 2.0])))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert len(results) == 2
        assert np.array_equal(results[0], results[1])
        assert source.cache_info()["entries"] == 1

    def test_from_json_rejects_a_non_object_file(self, tmp_path: Path) -> None:
        path = tmp_path / "table.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="must contain a JSON object"):
            PrecomputedTable.from_json(path)
