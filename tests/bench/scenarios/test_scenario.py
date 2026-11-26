"""Loader/expander/validator behaviour (D2, K-1/K-5, K-2, B1)."""

from __future__ import annotations

from typing import Any

import pytest

from pysatl_core_bench import (
    ScenarioExpander,
    ScenarioLoader,
    ScenarioRejected,
    ScenarioValidator,
)
from pysatl_core_bench.accuracy.types import Role
from tests.bench.conftest import cdf_scenario_raw, load, normal_distribution, normal_participant


def test_load_materializes_defaults_and_hash() -> None:
    raw = {
        "schema_version": "1.1",
        "comparison_group": "g",
        "distribution": normal_distribution().identity(),
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "cdf"}],
    }
    spec = load(raw)
    assert spec.pair_policy == "subject"
    assert spec.regions[0].name == "bulk"
    assert spec.queries[0].output_scale == "probability"
    assert spec.distribution.family == "Normal"
    assert spec.spec_hash.startswith("sha256:")
    # canonical form is order-independent
    assert spec.spec_hash == load(dict(reversed(list(raw.items())))).spec_hash


def test_distribution_is_required() -> None:
    raw = {
        "schema_version": "1.1",
        "comparison_group": "g",
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "cdf"}],
    }
    with pytest.raises(ScenarioRejected, match="distribution is required"):
        load(raw)


def test_distribution_fields_are_validated() -> None:
    base = {
        "schema_version": "1.1",
        "comparison_group": "g",
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "cdf"}],
    }
    with pytest.raises(ScenarioRejected, match="unknown distribution fields"):
        load(base | {"distribution": {"family": "Normal", "typo": 1}})
    with pytest.raises(ScenarioRejected, match="family must be a non-empty string"):
        load(base | {"distribution": {"family": ""}})


def test_participant_params_field_is_rejected() -> None:
    """rev. 40: the distribution is scenario-level; participant params are gone."""
    raw = cdf_scenario_raw()
    raw["subject"] = {
        "source_id": "pysatl-core",
        "params": {"family": "Normal", "parameters": {"mu": 0.0, "sigma": 1.0}},
    }
    with pytest.raises(ScenarioRejected, match="unknown participant fields"):
        load(raw)


def test_subject_references_synonyms_expand_into_participants() -> None:
    spec = load(cdf_scenario_raw())
    assert spec.participant_by_role(Role.SUBJECT)[0].source_id == "pysatl-core"
    refs = spec.participant_by_role(Role.REFERENCE)
    assert [r.source_id for r in refs] == ["scipy-norm-cdf"]


def test_participants_and_synonyms_are_mutually_exclusive() -> None:
    raw = cdf_scenario_raw() | {
        "participants": [
            {"role": "subject", "source_id": "pysatl-core"},
            {"role": "reference", "source_id": "scipy-norm-cdf"},
        ]
    }
    with pytest.raises(ScenarioRejected, match="not both"):
        load(raw)


def test_participant_identity_includes_adapter_config_and_strategy() -> None:
    a = normal_participant()
    b = normal_participant() | {"strategy": {"name": "DefaultComputationStrategy", "params": {}}}
    spec = load(
        {
            "schema_version": "1.1",
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "participants": [{**a, "role": "subject"}, {**b, "role": "reference"}],
            "queries": [{"characteristic": "cdf"}],
        }
    )
    ids = {p.participant_id for p in spec.participants}
    assert len(ids) == 2


def test_unknown_strategy_shape_rejected() -> None:
    raw = cdf_scenario_raw()
    raw["subject"] = {**normal_participant(), "strategy": {"typo": 1}}
    with pytest.raises(ScenarioRejected, match="unknown strategy fields"):
        load(raw)


def test_sweep_expansion_produces_sort_keys_and_group() -> None:
    raw = cdf_scenario_raw()
    raw["sweep_spec"] = {
        "param": "distribution.parameters.sigma",
        "values": [0.5, 1.0, 2.0],
    }
    scenarios = ScenarioExpander().expand(raw)
    assert len(scenarios) == 3
    assert all(s["comparison_group"] == "normal-cdf-vs-scipy" for s in scenarios)
    assert [s["sort_key"] for s in scenarios] == [0.5, 1.0, 2.0]
    sigmas = [s["distribution"]["parameters"]["sigma"] for s in scenarios]
    assert sigmas == [0.5, 1.0, 2.0]
    specs = [ScenarioLoader().load(s) for s in scenarios]
    assert len({s.spec_hash for s in specs}) == 3
    assert {s.comparison_group for s in specs} == {"normal-cdf-vs-scipy"}


def test_sweep_can_address_strategy_axis() -> None:
    raw = cdf_scenario_raw()
    raw["subject"] = {
        **normal_participant(),
        "strategy": {"name": "DefaultComputationStrategy", "params": {"x": 0}},
    }
    raw["sweep_spec"] = {
        "param": "subject.strategy.params.x",
        "values": [1, 2],
    }
    scenarios = ScenarioExpander().expand(raw)
    assert [s["subject"]["strategy"]["params"]["x"] for s in scenarios] == [1, 2]


def test_sweep_unknown_path_is_rejected() -> None:
    raw = cdf_scenario_raw()
    raw["sweep_spec"] = {"param": "no.such.path", "values": [1]}
    with pytest.raises(ScenarioRejected, match="unknown key"):
        ScenarioExpander().expand(raw)


def test_chain_expansion() -> None:
    raw = cdf_scenario_raw()
    del raw["subject"], raw["references"]
    raw["chain_spec"] = {
        "stages": [
            {
                "id": "core",
                "participant": normal_participant(),
                "reference": {"source_id": "scipy-norm-cdf"},
            },
            {
                "id": "scipy",
                "participant": {"source_id": "scipy-norm-cdf"},
                "reference": normal_participant(),
            },
        ]
    }
    scenarios = ScenarioExpander().expand(raw)
    assert len(scenarios) == 2
    specs = [load(s) for s in scenarios]
    assert [s.scenario_id for s in specs] == [
        "normal-cdf-vs-scipy/core",
        "normal-cdf-vs-scipy/scipy",
    ]
    assert [s.sort_key for s in specs] == [0.0, 1.0]


def test_sweep_and_chain_are_mutually_exclusive() -> None:
    raw = cdf_scenario_raw() | {
        "sweep_spec": {"param": "x", "values": [1]},
        "chain_spec": {"stages": []},
    }
    with pytest.raises(ScenarioRejected, match="mutually exclusive"):
        ScenarioExpander().expand(raw)


def test_sweep_spec_must_be_expanded_before_load() -> None:
    raw = cdf_scenario_raw() | {"sweep_spec": {"param": "x", "values": [1]}}
    with pytest.raises(ScenarioRejected, match="ScenarioExpander"):
        load(raw)


def test_validator_flags_degenerate_pair() -> None:
    raw = {
        "schema_version": "1.1",
        "comparison_group": "self-vs-self",
        "distribution": normal_distribution().identity(),
        "subject": normal_participant(),
        "references": [normal_participant()],
        "queries": [{"characteristic": "cdf"}],
    }
    spec = load(raw)
    notes = ScenarioValidator().validate(spec)
    assert len(notes) == 1
    assert notes[0].code == "degenerate_comparison"


def test_validator_is_quiet_for_distinct_participants() -> None:
    spec = load(cdf_scenario_raw())
    assert ScenarioValidator().validate(spec) == []


def test_ppf_region_outside_level_axis_is_rejected() -> None:
    """A ppf region that intersects no levels is a file defect, not a crash.

    The region [2, 3] on the level axis used to pass the loader and raise
    FrameworkDefect at grid generation, killing the whole batch; the level
    axis (0, 1) is framework knowledge, so the contradiction is rejected at
    load (REV-3: grammar contradictions knowable without a library).
    """
    base = {
        "schema_version": "1.1",
        "comparison_group": "ppf-grammar",
        "distribution": normal_distribution().identity(),
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "ppf"}],
    }
    for bad_interval in ([2.0, 3.0], [-1.0, -0.5], [1.5, None]):
        raw = base | {"regions": [{"name": "levels", "interval": bad_interval}]}
        with pytest.raises(ScenarioRejected, match="level axis"):
            load(raw)


def test_ppf_region_partially_outside_axis_is_legal() -> None:
    """Mixed pdf+ppf scenarios share wide bulk regions; the generator clamps."""
    raw = {
        "schema_version": "1.1",
        "comparison_group": "ppf-mixed",
        "distribution": normal_distribution().identity(),
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "pdf"}, {"characteristic": "ppf"}],
        "regions": [{"name": "bulk", "interval": [-3.0, 3.0], "output_scale": "probability"}],
    }
    spec = load(raw)
    assert spec.regions[0].interval == (-3.0, 3.0)


class TestExplicitPointsGrammar:
    """Loader rules for the query-level ``points`` array (rev. 60).

    ``points`` is the user's contract "measure exactly these coordinates":
    the loader validates the shape (numbers, finite, at least one) and the
    grammar-level consequences (every point in some region; ppf levels
    strictly inside (0, 1)), but never normalizes the content — sorting,
    deduplication and support filtering are runtime policies that must
    stay visible in the report.
    """

    @staticmethod
    def raw_with_points(points: Any, characteristic: str = "pdf") -> dict[str, Any]:
        return {
            "schema_version": "1.1",
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": [{"source_id": "r"}],
            "queries": [
                {
                    "characteristic": characteristic,
                    "grid_hint": {"n_points": 9},
                    "points": points,
                }
            ],
            "regions": [
                {"name": "bulk", "interval": [-3.0, 3.0]},
                {"name": "tail_right", "interval": [3.0, None]},
            ],
        }

    def test_points_are_loaded_as_declared(self) -> None:
        spec = load(self.raw_with_points([1, -0.5, 25.0, -0.5]))
        query = spec.queries[0]
        assert query.points == (1.0, -0.5, 25.0, -0.5)  # order kept, int -> float
        assert query.grid_hint == {"n_points": 9}  # hint and points coexist

    @pytest.mark.parametrize(
        "points",
        [
            [],
            [1, 2, "3"],
            [True],
            [float("nan")],
            [float("inf")],
            {"characteristic": "pdf"},
        ],
        ids=["empty", "non-number", "bool", "nan", "inf", "not-a-list"],
    )
    def test_invalid_points_are_rejected(self, points: Any) -> None:
        with pytest.raises(ScenarioRejected):
            load(self.raw_with_points(points))

    def test_orphan_point_is_rejected_with_its_coordinate(self) -> None:
        # A point outside every region would be silently unmeasurable:
        # metric records are keyed by (query, region), the orphan has no cell.
        with pytest.raises(ScenarioRejected, match=r"fall into no declared region.*-50\.0"):
            load(self.raw_with_points([0.0, -50.0]))

    @staticmethod
    def raw_with_ppf_levels(levels: Any) -> dict[str, Any]:
        raw = TestExplicitPointsGrammar.raw_with_points([0.25], "ppf")
        raw["regions"] = [
            {"name": "bulk", "interval": [0.0, 0.5]},
            {"name": "tail_right", "interval": [0.5, None]},
        ]
        raw["queries"] = [{"characteristic": "ppf", "grid_hint": {"n_points": 9}, "points": levels}]
        return raw

    def test_ppf_point_on_the_level_boundary_is_rejected(self) -> None:
        # 0 and 1 are infinite quantiles; the generator's clip is a
        # generator policy and must not silently rewrite a declared level.
        with pytest.raises(ScenarioRejected, match="strictly.*\\(0, 1\\)"):
            load(self.raw_with_ppf_levels([0.25, 1.0]))

    def test_ppf_point_above_one_is_rejected(self) -> None:
        with pytest.raises(ScenarioRejected, match="\\(0, 1\\)"):
            load(self.raw_with_ppf_levels([0.25, 1.5]))

    def test_canonical_hash_tracks_the_points_array(self) -> None:
        with_duplicates = load(self.raw_with_points([0.0, 0.0, 1.0]))
        deduplicated = load(self.raw_with_points([0.0, 1.0]))
        different_order = load(self.raw_with_points([1.0, 0.0]))
        # duplicates are part of the declared request (the runner will
        # deduplicate them with a note), so the canonical form keeps them
        assert with_duplicates.spec_hash != deduplicated.spec_hash
        assert different_order.spec_hash != deduplicated.spec_hash


class TestSweepChainGrammarErrors:
    """Expander rejections (K-1/K-5): a malformed axis never reaches the loader."""

    @staticmethod
    def base_raw() -> dict[str, Any]:
        return {
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": [{"source_id": "r"}],
            "queries": [{"characteristic": "cdf"}],
        }

    def test_sweep_spec_must_be_an_object(self) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = "n_points"
        with pytest.raises(ScenarioRejected, match="must be an object"):
            ScenarioExpander().expand(raw)

    def test_sweep_unknown_fields_rejected(self) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = {"param": "distribution.parameters.mu", "values": [1.0], "extra": 1}
        with pytest.raises(ScenarioRejected, match="unknown sweep_spec fields"):
            ScenarioExpander().expand(raw)

    @pytest.mark.parametrize(
        "spec",
        [{"values": [1.0]}, {"param": "", "values": [1.0]}, 5],
        ids=["no-param", "empty-param", "not-object"],
    )
    def test_sweep_param_validated(self, spec: Any) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = spec
        with pytest.raises(ScenarioRejected, match="sweep_spec"):
            ScenarioExpander().expand(raw)

    @pytest.mark.parametrize(
        "values", [[None], [{"deep": 1}], [True]], ids=["null-value", "non-scalar", "bool"]
    )
    def test_sweep_values_must_be_scalars(self, values: list[Any]) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = {"param": "distribution.parameters.mu", "values": values}
        with pytest.raises(ScenarioRejected, match="scalar"):
            ScenarioExpander().expand(raw)

    def test_sweep_values_must_be_a_non_empty_list(self) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = {"param": "distribution.parameters.mu", "values": []}
        with pytest.raises(ScenarioRejected, match="non-empty list"):
            ScenarioExpander().expand(raw)

    def test_chain_spec_must_be_an_object_with_stages(self) -> None:
        raw = self.base_raw()
        raw["chain_spec"] = {"stages": "nope"}
        with pytest.raises(ScenarioRejected, match="chain_spec"):
            ScenarioExpander().expand(raw)

    def test_chain_unknown_stage_fields_rejected(self) -> None:
        raw = self.base_raw()
        raw["chain_spec"] = {
            "stages": [
                {"participant": {"source_id": "a"}, "reference": {"source_id": "b"}, "oops": 1}
            ]
        }
        with pytest.raises(ScenarioRejected, match="unknown chain stage fields"):
            ScenarioExpander().expand(raw)

    @pytest.mark.parametrize(
        "stage",
        ["nope", {"participant": None}, {"participant": 5}],
        ids=["not-object", "missing", "not-object-value"],
    )
    def test_chain_stage_validated(self, stage: Any) -> None:
        raw = self.base_raw()
        stage_full = stage if isinstance(stage, dict) and "participant" in stage else stage
        raw["chain_spec"] = {"stages": [stage_full]}
        with pytest.raises(ScenarioRejected):
            ScenarioExpander().expand(raw)

    def test_chain_stage_distribution_must_be_an_object(self) -> None:
        raw = self.base_raw()
        raw["chain_spec"] = {
            "stages": [
                {
                    "participant": {"source_id": "a"},
                    "reference": {"source_id": "b"},
                    "distribution": "Normal",
                }
            ]
        }
        with pytest.raises(ScenarioRejected, match="stage 0.distribution"):
            ScenarioExpander().expand(raw)

    def test_unnamed_sweep_file_rejected(self) -> None:
        raw = self.base_raw()
        del raw["comparison_group"]
        raw["sweep_spec"] = {"param": "distribution.parameters.mu", "values": [1.0]}
        with pytest.raises(ScenarioRejected, match="comparison_group"):
            ScenarioExpander().expand(raw)

    def test_sweep_path_traversing_a_list_requires_an_index(self) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = {"param": "queries.abc.characteristic", "values": ["pdf"]}
        with pytest.raises(ScenarioRejected, match="not an index"):
            ScenarioExpander().expand(raw)

    def test_sweep_path_index_out_of_range_rejected(self) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = {"param": "queries.7.characteristic", "values": ["pdf"]}
        with pytest.raises(ScenarioRejected, match="out of range"):
            ScenarioExpander().expand(raw)

    def test_sweep_path_traversing_a_scalar_rejected(self) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = {"param": "distribution.family.x", "values": ["pdf"]}
        with pytest.raises(ScenarioRejected, match="traverses a scalar"):
            ScenarioExpander().expand(raw)

    def test_sweep_path_unknown_key_rejected(self) -> None:
        raw = self.base_raw()
        raw["sweep_spec"] = {"param": "distribution.no_such_key", "values": ["pdf"]}
        with pytest.raises(ScenarioRejected, match="unknown key"):
            ScenarioExpander().expand(raw)


class TestLoaderValidationMatrix:
    """Loader rejections beyond the happy path (D2): every branch is a
    documented rule of the scenario grammar, so every branch needs a
    pinned refusal."""

    @staticmethod
    def base_raw() -> dict[str, Any]:
        return {
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": [{"source_id": "r"}],
            "queries": [{"characteristic": "cdf"}],
        }

    def test_unknown_scenario_field_rejected(self) -> None:
        raw = self.base_raw()
        raw["unknown_top"] = 1
        with pytest.raises(ScenarioRejected, match="unknown scenario fields"):
            load(raw)

    def test_wrong_schema_version_rejected(self) -> None:
        raw = self.base_raw()
        raw["schema_version"] = "9.9"
        with pytest.raises(ScenarioRejected, match="schema_version"):
            load(raw)

    def test_schema_1_1_files_stay_valid(self) -> None:
        # 1.2 (rev. 61) is a purely additive grammar generation: a 1.1
        # file keeps loading unchanged (no silent migration, no rejection).
        raw = self.base_raw()
        raw["schema_version"] = "1.1"
        spec = load(raw)
        assert spec.schema_version == "1.2"  # results are written by the current generation

    def test_schema_1_2_is_the_default_and_is_accepted(self) -> None:
        raw = self.base_raw()
        raw["schema_version"] = "1.2"
        assert load(raw).schema_version == "1.2"
        assert load({**self.base_raw()}).schema_version == "1.2"  # absent = current

    def test_distribution_must_be_an_object(self) -> None:
        raw = self.base_raw()
        raw["distribution"] = "Normal"
        with pytest.raises(ScenarioRejected, match="distribution is required"):
            load(raw)

    def test_distribution_unknown_fields_rejected(self) -> None:
        raw = self.base_raw()
        raw["distribution"] = {**normal_distribution().identity(), "oops": 1}
        with pytest.raises(ScenarioRejected, match="unknown distribution fields"):
            load(raw)

    def test_distribution_family_required(self) -> None:
        raw = self.base_raw()
        raw["distribution"] = {"parametrization": "meanStd", "parameters": {"mu": 0.0}}
        with pytest.raises(ScenarioRejected, match="family"):
            load(raw)

    def test_distribution_parametrization_must_be_a_string(self) -> None:
        raw = self.base_raw()
        raw["distribution"] = {"family": "Normal", "parametrization": 5}
        with pytest.raises(ScenarioRejected, match="parametrization"):
            load(raw)

    def test_distribution_parameters_must_be_an_object(self) -> None:
        raw = self.base_raw()
        raw["distribution"] = {"family": "Normal", "parameters": [1, 2]}
        with pytest.raises(ScenarioRejected, match="parameters"):
            load(raw)

    def test_participants_list_must_be_non_empty(self) -> None:
        raw = self.base_raw()
        del raw["subject"], raw["references"]
        raw["participants"] = []
        with pytest.raises(ScenarioRejected, match="non-empty"):
            load(raw)

    def test_participant_role_required_in_list_form(self) -> None:
        raw = self.base_raw()
        del raw["subject"], raw["references"]
        raw["participants"] = [{"source_id": "s"}]
        with pytest.raises(ScenarioRejected, match="role"):
            load(raw)

    def test_role_omitted_under_symmetric_policy_defaults_to_peer(self) -> None:
        """Additive 1.2 grammar: under 'symmetric' every participant is a
        peer, so the role carries no information and may be omitted — an
        explicit null behaves like an absent key (A-12)."""
        raw = self.base_raw()
        del raw["subject"], raw["references"]
        raw["pair_policy"] = "symmetric"
        raw["participants"] = [
            {"source_id": "a"},
            {"source_id": "b", "role": None},
        ]
        spec = load(raw)
        assert spec.pair_policy == "symmetric"
        assert [p.role for p in spec.participants] == [Role.PEER, Role.PEER]
        assert spec.participant_by_role(Role.PEER)[0].source_id == "a"

    def test_role_omitted_outside_symmetric_policy_rejected(self) -> None:
        """Only 'symmetric' makes the role redundant; under the default
        policy the role is what distinguishes the measured side."""
        raw = self.base_raw()
        del raw["subject"], raw["references"]
        raw["pair_policy"] = None  # explicit null = the default 'subject'
        raw["participants"] = [{"source_id": "s"}, {"source_id": "r"}]
        with pytest.raises(ScenarioRejected, match="omit it only under"):
            load(raw)

    def test_participant_source_id_required(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{}]
        with pytest.raises(ScenarioRejected, match="source_id"):
            load(raw)

    def test_participant_unknown_fields_rejected(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "extra": 1}]
        with pytest.raises(ScenarioRejected, match="unknown participant fields"):
            load(raw)

    def test_adapter_config_must_be_an_object(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "adapter_config": [1]}]
        with pytest.raises(ScenarioRejected, match="adapter_config"):
            load(raw)

    def test_strategy_must_be_a_string_or_object(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "strategy": 5}]
        with pytest.raises(ScenarioRejected, match="strategy"):
            load(raw)

    def test_strategy_unknown_fields_rejected(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "strategy": {"name": "S", "extra": 1}}]
        with pytest.raises(ScenarioRejected, match="unknown strategy fields"):
            load(raw)

    def test_strategy_params_must_be_an_object(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "strategy": {"name": "S", "params": [1]}}]
        with pytest.raises(ScenarioRejected, match="strategy.params"):
            load(raw)

    def test_pair_policy_unknown_string_rejected(self) -> None:
        raw = self.base_raw()
        raw["pair_policy"] = "round-robin"
        with pytest.raises(ScenarioRejected, match="pair_policy"):
            load(raw)

    def test_pair_policy_entry_must_be_an_index_pair(self) -> None:
        raw = self.base_raw()
        raw["pair_policy"] = [[0, 0]]
        with pytest.raises(ScenarioRejected, match="invalid for given participants"):
            load(raw)

    def test_pair_policy_duplicate_pair_rejected(self) -> None:
        raw = self.base_raw()
        raw["pair_policy"] = [[0, 1], [1, 0]]
        with pytest.raises(ScenarioRejected, match="duplicate pair"):
            load(raw)

    def test_subject_policy_rejects_peer_role(self) -> None:
        raw = self.base_raw()
        del raw["subject"], raw["references"]
        raw["participants"] = [
            {"role": "subject", "source_id": "s"},
            {"role": "peer", "source_id": "p"},
        ]
        with pytest.raises(ScenarioRejected, match="peer-role"):
            load(raw)

    def test_subject_policy_rejects_duplicate_reference_identity(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r"}, {"source_id": "r"}]
        with pytest.raises(ScenarioRejected, match="duplicate reference identity"):
            load(raw)

    def test_unknown_query_field_rejected(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "cdf", "oops": 1}]
        with pytest.raises(ScenarioRejected, match="unknown query fields"):
            load(raw)

    def test_unknown_characteristic_rejected(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "third_moment"}]
        with pytest.raises(ScenarioRejected, match="unknown characteristic"):
            load(raw)

    def test_unknown_query_output_scale_rejected(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "cdf", "output_scale": "hazard"}]
        with pytest.raises(ScenarioRejected, match="unknown query output_scale"):
            load(raw)

    def test_unknown_grid_hint_field_rejected(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "cdf", "grid_hint": {"step": 0.1}}]
        with pytest.raises(ScenarioRejected, match="unknown grid_hint fields"):
            load(raw)

    def test_bad_n_points_rejected(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "cdf", "grid_hint": {"n_points": 3}}]
        with pytest.raises(ScenarioRejected, match="n_points"):
            load(raw)

    def test_tail_extension_hint_is_kept_as_declared(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [
            {
                "characteristic": "cdf",
                "grid_hint": {"n_points": 9, "tail_extension": 12.5, "densify": False},
            }
        ]
        spec = load(raw)
        assert spec.queries[0].grid_hint == {
            "n_points": 9,
            "tail_extension": 12.5,
            "densify": False,
        }

    @pytest.mark.parametrize("bad", [0, -1.0, "10", True, float("inf"), float("nan")])
    def test_bad_tail_extension_hint_rejected(self, bad: Any) -> None:
        raw = self.base_raw()
        raw["queries"] = [
            {"characteristic": "cdf", "grid_hint": {"n_points": 9, "tail_extension": bad}}
        ]
        with pytest.raises(ScenarioRejected, match="tail_extension"):
            load(raw)

    def test_explicit_tail_null_is_accepted_and_stays_null(self) -> None:
        # A null hint field is "not set": it must survive canonicalization
        # as-is (faithful to the file), not be rewritten to the default.
        raw = self.base_raw()
        raw["queries"] = [
            {"characteristic": "cdf", "grid_hint": {"n_points": 9, "tail_extension": None}}
        ]
        spec = load(raw)
        assert spec.queries[0].grid_hint == {"n_points": 9, "tail_extension": None}

    @pytest.mark.parametrize("bad", [0, 1, "false"])
    def test_bad_densify_hint_rejected(self, bad: Any) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "cdf", "grid_hint": {"n_points": 9, "densify": bad}}]
        with pytest.raises(ScenarioRejected, match="densify"):
            load(raw)

    def test_duplicate_query_rejected(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "cdf"}, {"characteristic": "cdf"}]
        with pytest.raises(ScenarioRejected, match="duplicate query"):
            load(raw)

    def test_regions_must_be_a_non_empty_list(self) -> None:
        raw = self.base_raw()
        raw["regions"] = []
        with pytest.raises(ScenarioRejected, match="regions must be a non-empty list"):
            load(raw)

    def test_duplicate_region_name_rejected(self) -> None:
        raw = self.base_raw()
        raw["regions"] = [
            {"name": "bulk", "interval": [-1.0, 1.0]},
            {"name": "bulk", "interval": [1.0, 2.0]},
        ]
        with pytest.raises(ScenarioRejected, match="duplicate region name"):
            load(raw)

    @pytest.mark.parametrize(
        "interval",
        [
            "[-1, 1]",
            [-1.0],
            [-1.0, 0.0, 1.0],
            [1.0, -1.0],
            [float("inf"), 1.0],
            [-1.0, float("nan")],
        ],
        ids=["not-list", "one-bound", "three-bounds", "reversed", "inf", "nan"],
    )
    def test_bad_intervals_rejected(self, interval: Any) -> None:
        raw = self.base_raw()
        raw["regions"] = [{"name": "bulk", "interval": interval}]
        with pytest.raises(ScenarioRejected):
            load(raw)

    def test_interval_bound_must_be_a_number(self) -> None:
        raw = self.base_raw()
        raw["regions"] = [{"name": "bulk", "interval": ["-1", 1.0]}]
        with pytest.raises(ScenarioRejected, match="must be numbers or null"):
            load(raw)

    def test_unknown_invariant_target_rejected(self) -> None:
        raw = self.base_raw()
        raw["invariant_check_on"] = ["ghost"]
        with pytest.raises(ScenarioRejected, match="match no participant"):
            load(raw)

    def test_invariant_check_on_must_be_strings(self) -> None:
        raw = self.base_raw()
        raw["invariant_check_on"] = [1]
        with pytest.raises(ScenarioRejected, match="list of non-empty strings"):
            load(raw)

    def test_sort_key_must_be_a_scalar(self) -> None:
        raw = self.base_raw()
        raw["sort_key"] = [1.0]
        with pytest.raises(ScenarioRejected, match="sort_key"):
            load(raw)

    def test_empty_scenario_id_rejected(self) -> None:
        raw = self.base_raw()
        raw["scenario_id"] = ""
        with pytest.raises(ScenarioRejected, match="scenario_id"):
            load(raw)

    def test_nameless_dict_rejected(self) -> None:
        raw = self.base_raw()
        del raw["comparison_group"]
        with pytest.raises(ScenarioRejected, match="unnamed scenario"):
            load(raw)

    def test_non_json_safe_adapter_config_rejected(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "adapter_config": {"x": object()}}]
        with pytest.raises(ScenarioRejected, match="not JSON-safe"):
            load(raw)

    def test_duplicate_json_key_after_coercion_rejected(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "adapter_config": {1: "a", "1": "b"}}]
        with pytest.raises(ScenarioRejected, match="duplicate key"):
            load(raw)


class TestValidationGaps:
    """Second-pass loader/expander rejections (coverage audit, rev. 60)."""

    @staticmethod
    def base_raw() -> dict[str, Any]:
        return {
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": [{"source_id": "r"}],
            "queries": [{"characteristic": "cdf"}],
        }

    def test_chain_unknown_top_fields_rejected(self) -> None:
        raw = self.base_raw()
        raw["chain_spec"] = {"stages": [], "extra": 1}
        with pytest.raises(ScenarioRejected, match="unknown chain_spec fields"):
            ScenarioExpander().expand(raw)

    def test_expand_requires_a_dict(self) -> None:
        with pytest.raises(ScenarioRejected, match="must be a JSON object"):
            ScenarioExpander().expand(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_non_json_safe_distribution_parameters_rejected(self) -> None:
        raw = self.base_raw()
        raw["distribution"] = {"family": "Normal", "parameters": {"mu": object()}}
        with pytest.raises(ScenarioRejected, match="distribution.parameters is not JSON-safe"):
            load(raw)

    def test_unknown_participant_role_rejected(self) -> None:
        raw = self.base_raw()
        del raw["subject"], raw["references"]
        raw["participants"] = [
            {"role": "manager", "source_id": "s"},
            {"role": "peer", "source_id": "p"},
        ]
        with pytest.raises(ScenarioRejected, match="unknown participant role"):
            load(raw)

    def test_empty_strategy_names_rejected(self) -> None:
        raw = self.base_raw()
        raw["references"] = [{"source_id": "r", "strategy": ""}]
        with pytest.raises(ScenarioRejected, match="non-empty"):
            load(raw)
        raw2 = self.base_raw()
        raw2["references"] = [{"source_id": "r", "strategy": {"name": ""}}]
        with pytest.raises(ScenarioRejected, match="non-empty"):
            load(raw2)

    def test_empty_pair_policy_list_rejected(self) -> None:
        raw = self.base_raw()
        raw["pair_policy"] = []
        with pytest.raises(ScenarioRejected, match="non-empty list"):
            load(raw)

    def test_malformed_pair_policy_entry_rejected(self) -> None:
        raw = self.base_raw()
        raw["pair_policy"] = [[0, "one"]]
        with pytest.raises(ScenarioRejected, match="integer pairs"):
            load(raw)

    def test_subject_policy_needs_exactly_one_subject(self) -> None:
        raw = self.base_raw()
        del raw["subject"], raw["references"]
        raw["participants"] = [
            {"role": "subject", "source_id": "a"},
            {"role": "subject", "source_id": "b"},
        ]
        with pytest.raises(ScenarioRejected, match="exactly one subject"):
            load(raw)

    def test_symmetric_policy_needs_two_participants(self) -> None:
        raw = self.base_raw()
        del raw["references"]  # subject-only scenario
        raw["pair_policy"] = "symmetric"
        with pytest.raises(ScenarioRejected, match="at least two participants"):
            load(raw)

    def test_queries_must_be_a_non_empty_list(self) -> None:
        raw = self.base_raw()
        raw["queries"] = []
        with pytest.raises(ScenarioRejected, match="queries must be a non-empty list"):
            load(raw)

    def test_too_large_explicit_point_rejected(self) -> None:
        raw = self.base_raw()
        raw["queries"] = [{"characteristic": "cdf", "points": [10**400]}]
        with pytest.raises(ScenarioRejected, match="too large for a float"):
            load(raw)

    def test_unknown_region_field_rejected(self) -> None:
        raw = self.base_raw()
        raw["regions"] = [{"name": "bulk", "interval": [-1.0, 1.0], "step": 0.1}]
        with pytest.raises(ScenarioRejected, match="unknown region fields"):
            load(raw)

    def test_empty_region_name_rejected(self) -> None:
        raw = self.base_raw()
        raw["regions"] = [{"name": "", "interval": [-1.0, 1.0]}]
        with pytest.raises(ScenarioRejected, match="region.name"):
            load(raw)

    def test_unknown_region_output_scale_rejected(self) -> None:
        raw = self.base_raw()
        raw["regions"] = [{"name": "bulk", "interval": [-1.0, 1.0], "output_scale": "hazard"}]
        with pytest.raises(ScenarioRejected, match="unknown region output_scale"):
            load(raw)

    def test_empty_comparison_group_rejected(self) -> None:
        raw = self.base_raw()
        raw["comparison_group"] = ""
        with pytest.raises(ScenarioRejected, match="comparison_group must be a non-empty"):
            load(raw)

    def test_group_is_chosen_over_scenario_id_default(self) -> None:
        raw = self.base_raw()
        raw["scenario_id"] = "s"
        raw["comparison_group"] = "g"
        spec = load(raw)
        assert spec.scenario_id == "s" and spec.comparison_group == "g"

    def test_scenario_id_derives_from_group_and_sort_key(self) -> None:
        raw = self.base_raw()
        raw["comparison_group"] = "g"
        raw["sort_key"] = 3.0
        spec = load(raw)
        assert spec.scenario_id == "g#3.0"

    def test_bool_sort_key_rejected(self) -> None:
        raw = self.base_raw()
        raw["sort_key"] = True
        with pytest.raises(ScenarioRejected, match="sort_key"):
            load(raw)

    def test_too_large_integer_parameter_rejected(self) -> None:
        raw = self.base_raw()
        raw["distribution"] = {"family": "Normal", "parameters": {"mu": 10**400}}
        with pytest.raises(ScenarioRejected, match="too large"):
            load(raw)


class TestIdentityDefaults:
    """The row identity defaults (rev. 35) — pinned end to end."""

    def test_group_only_scenario_collapses_id_to_the_group(self) -> None:
        raw = {
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": [{"source_id": "r"}],
            "queries": [{"characteristic": "cdf"}],
        }
        spec = load(raw)
        assert spec.comparison_group == "g"
        assert spec.scenario_id == "g"

    def test_string_sort_key_is_legal(self) -> None:
        raw = {
            "comparison_group": "g",
            "scenario_id": "g/stage-b",
            "sort_key": "b",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": [{"source_id": "r"}],
            "queries": [{"characteristic": "cdf"}],
        }
        spec = load(raw)
        assert spec.sort_key == "b"
        assert spec.scenario_id == "g/stage-b"

    def test_references_must_be_a_non_empty_list(self) -> None:
        raw = {
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": "scipy",
            "queries": [{"characteristic": "cdf"}],
        }
        with pytest.raises(ScenarioRejected, match="references must be a non-empty list"):
            load(raw)

    def test_scenario_needs_at_least_one_participant(self) -> None:
        raw = {
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "queries": [{"characteristic": "cdf"}],
        }
        with pytest.raises(ScenarioRejected, match="at least one participant"):
            load(raw)

    def test_chain_stage_distribution_override_reaches_the_loader(self) -> None:
        raw = {
            "comparison_group": "g",
            "distribution": normal_distribution().identity(),
            "subject": {"source_id": "s"},
            "references": [{"source_id": "r"}],
            "queries": [{"characteristic": "cdf"}],
            "chain_spec": {
                "stages": [
                    {
                        "participant": {"source_id": "core"},
                        "reference": {"source_id": "ref"},
                        "distribution": {
                            "family": "Normal",
                            "parametrization": "meanStd",
                            "parameters": {"mu": 1.0, "sigma": 2.0},
                        },
                    }
                ]
            },
        }
        variants = ScenarioExpander().expand(raw)
        assert len(variants) == 1
        spec = load(variants[0])
        assert spec.distribution.parameters == {"mu": 1.0, "sigma": 2.0}
        assert spec.participant_by_role(Role.SUBJECT)[0].source_id == "core"
