"""PySATLAdapter boundary: strategies, forced plans, verdicts (M-2/M-3/M-6).

The adapter is the only place allowed to know pysatl-core.  These tests
drive it through its public surface — ``prepare``/``assess``/``evaluate``
with strategy specs from the scenario grammar — and pin the contract that
matters to the framework: an unusable strategy is an explicit
``AdapterConfigurationError``/verdict, never a silent fallback, and every
prepared state carries a plan dump for provenance.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysatl_core.types import CharacteristicName
from pysatl_core_bench import (
    AccuracyQuery,
    AdapterConfigurationError,
    DistributionSpec,
    ParticipantSpec,
    PySATLAdapter,
    Readiness,
    Role,
    SupportVerdict,
)
from pysatl_core_bench.accuracy.computations.protocol import PreparedSubject


def distribution() -> DistributionSpec:
    return DistributionSpec(
        family="Normal", parametrization="meanStd", parameters={"mu": 0.0, "sigma": 1.0}
    )


def participant(strategy: dict[str, object] | str | None = None) -> ParticipantSpec:
    return ParticipantSpec(
        role=Role.SUBJECT,
        source_id="pysatl-core",
        participant_id="pysatl-core#t",
        adapter_config={},
        strategy=strategy,
    )


def query(name: str) -> AccuracyQuery:
    return AccuracyQuery(characteristic=CharacteristicName(name))


class TestStrategies:
    def test_default_strategy_dumps_the_plan(self) -> None:
        prepared = PySATLAdapter().prepare(participant(), query("pdf"), distribution())
        assert prepared.path_dump is not None
        assert prepared.distribution == distribution()

    def test_forced_strategy_from_pdf_reuses_the_analytical_loop(self) -> None:
        # from == target: no graph walk is needed, the analytical pdf is used.
        prepared = PySATLAdapter().prepare(
            participant({"name": "ForcedGraphStrategy", "params": {"from": "pdf"}}),
            query("pdf"),
            distribution(),
        )
        assert prepared.path_dump is not None

    def test_forced_strategy_from_cdf_walks_the_graph(self) -> None:
        # cdf -> pdf is a graph route: the forced plan starts at cdf and
        # walks the characteristic graph to pdf (M-2).
        prepared = PySATLAdapter().prepare(
            participant({"name": "ForcedGraphStrategy", "params": {"from": "cdf"}}),
            query("pdf"),
            distribution(),
        )
        assert prepared.path_dump is not None

    def test_forced_strategy_without_route_is_a_configuration_error(self) -> None:
        # lpdf has no route to pdf: the honest answer is an explicit error
        # (the runner turns it into unsupported_configuration), not a
        # silent substitution by another start point (M-6).
        with pytest.raises(AdapterConfigurationError, match="no computation route"):
            PySATLAdapter().prepare(
                participant({"name": "ForcedGraphStrategy", "params": {"from": "lpdf"}}),
                query("pdf"),
                distribution(),
            )

    def test_unknown_strategy_name_is_a_configuration_error(self) -> None:
        with pytest.raises(AdapterConfigurationError):
            PySATLAdapter().prepare(participant({"name": "Nope"}), query("pdf"), distribution())

    def test_strategy_params_must_be_an_object(self) -> None:
        with pytest.raises(AdapterConfigurationError):
            PySATLAdapter().prepare(
                participant({"name": "ForcedGraphStrategy", "params": "pdf"}),
                query("pdf"),
                distribution(),
            )


class TestVerdicts:
    def test_assess_reports_compute_readiness(self) -> None:
        verdict = PySATLAdapter().assess(participant(), query("pdf"), distribution())
        assert verdict.readiness == Readiness.READY
        assert isinstance(verdict, SupportVerdict)

    def test_assess_refuses_an_unknown_strategy(self) -> None:
        verdict = PySATLAdapter().assess(
            participant({"name": "Nope"}), query("pdf"), distribution()
        )
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION

    def test_support_is_the_library_support(self) -> None:
        # The library support of Normal is the whole real line; the exact
        # closedness is the library's answer, the adapter only forwards it.
        support = PySATLAdapter().support(participant(), distribution())
        assert support.left == float("-inf") and support.right == float("inf")


class TestNativeScalesAndEvaluate:
    def test_native_scale_is_probability(self) -> None:
        scales = PySATLAdapter().native_output_scales(participant(), query("pdf"))
        assert scales == ["probability"]

    def test_evaluate_answers_pdf_at_declared_points(self) -> None:
        import numpy as np

        prepared = PySATLAdapter().prepare(participant(), query("pdf"), distribution())
        values = PySATLAdapter().evaluate(prepared, np.asarray([0.0, 1.0]))
        assert values.shape == (2,)
        assert np.all(np.isfinite(values))


class TestAssessVerdictBranches:
    """Assess is the readiness layer: every refusal is a verdict, not a
    crash, and the reason must say what the configuration got wrong."""

    def test_scalar_characteristics_are_not_grid_measurable(self) -> None:
        verdict = PySATLAdapter().assess(participant(), query("mean"), distribution())
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION
        assert "scalar" in verdict.reason

    def test_points_outside_the_support_are_listed(self) -> None:
        # Normal spans the whole real line, so the bounded-support branch
        # needs a distribution whose support actually excludes points:
        # Exponential(rate) lives on [0, inf).
        exponential = DistributionSpec(
            family="Exponential", parametrization="rate", parameters={"lambda_": 1.0}
        )
        verdict = PySATLAdapter().assess(
            participant(), query("pdf"), exponential, np.asarray([-1.0, 0.0, 2.0])
        )
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION
        assert verdict.missing == (-1.0,)

    def test_unknown_family_is_a_verdict_not_a_crash(self) -> None:
        verdict = PySATLAdapter().assess(
            participant(),
            query("pdf"),
            DistributionSpec(family="LeviStatue", parametrization=None, parameters={}),
        )
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION

    def test_unconstructible_distribution_is_a_verdict(self) -> None:
        verdict = PySATLAdapter().assess(
            participant(),
            query("pdf"),
            DistributionSpec(family="Normal", parametrization="meanStd", parameters={"mu": "x"}),
        )
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION

    def test_pestiferous_strategy_in_assess_is_inapplicable(self) -> None:
        verdict = PySATLAdapter().assess(
            participant({"name": "ForcedGraphStrategy", "params": {"from": "lpdf"}}),
            query("pdf"),
            distribution(),
        )
        assert verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION


class TestStrategyParamValidation:
    """ForcedGraphStrategy params are validated, never guessed (M-2/M-6)."""

    def test_unknown_param_fields_rejected(self) -> None:
        with pytest.raises(AdapterConfigurationError, match="unknown ForcedGraphStrategy params"):
            PySATLAdapter().prepare(
                participant(
                    {"name": "ForcedGraphStrategy", "params": {"from": "pdf", "via": "cf"}}
                ),
                query("pdf"),
                distribution(),
            )

    def test_missing_from_rejected(self) -> None:
        with pytest.raises(AdapterConfigurationError, match="requires params"):
            PySATLAdapter().prepare(
                participant({"name": "ForcedGraphStrategy", "params": {}}),
                query("pdf"),
                distribution(),
            )

    def test_unknown_start_characteristic_rejected(self) -> None:
        with pytest.raises(AdapterConfigurationError, match="unknown start characteristic"):
            PySATLAdapter().prepare(
                participant({"name": "ForcedGraphStrategy", "params": {"from": "mode"}}),
                query("pdf"),
                distribution(),
            )

    def test_bad_labels_rejected(self) -> None:
        with pytest.raises(AdapterConfigurationError, match="labels"):
            PySATLAdapter().prepare(
                participant(
                    {"name": "ForcedGraphStrategy", "params": {"from": "pdf", "labels": [1]}}
                ),
                query("pdf"),
                distribution(),
            )

    def test_string_strategy_form(self) -> None:
        prepared = PySATLAdapter().prepare(
            participant("DefaultComputationStrategy"), query("pdf"), distribution()
        )
        assert prepared.path_dump is not None

    def test_malformed_strategy_dict_rejected(self) -> None:
        with pytest.raises(AdapterConfigurationError, match="strategy must be a string"):
            # an intentionally malformed strategy: the adapter must refuse
            # the wrong *type*, not measure a guess
            PySATLAdapter().prepare(participant(5), query("pdf"), distribution())  # type: ignore[arg-type]

    def test_loop_start_needs_no_graph_walk(self) -> None:
        # from == target: the plan is the analytical loop itself.
        prepared = PySATLAdapter().prepare(
            participant({"name": "ForcedGraphStrategy", "params": {"from": "cdf"}}),
            query("cdf"),
            distribution(),
        )
        assert prepared.path_dump is not None

    def test_labels_preference_is_accepted(self) -> None:
        prepared = PySATLAdapter().prepare(
            participant({"name": "ForcedGraphStrategy", "params": {"from": "cdf", "labels": []}}),
            query("pdf"),
            distribution(),
        )
        assert prepared.path_dump is not None


class TestPreparedPayloadGuard:
    def test_foreign_payload_is_refused(self) -> None:
        prepared = PreparedSubject(participant_id="x", query=query("pdf"), payload="alien")
        with pytest.raises(AdapterConfigurationError, match="payload"):
            PySATLAdapter().evaluate(prepared, np.asarray([0.0]))
