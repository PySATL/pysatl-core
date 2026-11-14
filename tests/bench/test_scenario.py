"""Loader/expander/validator behaviour (D2, К-1/К-5, К-2, B1)."""

from __future__ import annotations

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
        "schema_version": "1.0",
        "comparison_group": "g",
        "distribution": normal_distribution().identity(),
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "cdf"}],
    }
    spec = load(raw)
    assert spec.pair_policy == "subject"
    assert spec.regions[0].name == "bulk"
    assert spec.queries[0].space == "probability"
    assert spec.distribution.family == "Normal"
    assert spec.spec_hash.startswith("sha256:")
    # canonical form is order-independent
    assert spec.spec_hash == load(dict(reversed(list(raw.items())))).spec_hash


def test_distribution_is_required() -> None:
    raw = {
        "schema_version": "1.0",
        "comparison_group": "g",
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "cdf"}],
    }
    with pytest.raises(ScenarioRejected, match="distribution is required"):
        load(raw)


def test_distribution_fields_are_validated() -> None:
    base = {
        "schema_version": "1.0",
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
    """ред. 40: the distribution is scenario-level; participant params are gone."""
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
            "schema_version": "1.0",
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
        "schema_version": "1.0",
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
        "schema_version": "1.0",
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
        "schema_version": "1.0",
        "comparison_group": "ppf-mixed",
        "distribution": normal_distribution().identity(),
        "subject": {"source_id": "s"},
        "references": [{"source_id": "r"}],
        "queries": [{"characteristic": "pdf"}, {"characteristic": "ppf"}],
        "regions": [{"name": "bulk", "interval": [-3.0, 3.0], "space": "probability"}],
    }
    spec = load(raw)
    assert spec.regions[0].interval == (-3.0, 3.0)
