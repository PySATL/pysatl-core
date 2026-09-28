"""Singleton lifecycle, formula lookup, and use by estimation methods."""

from __future__ import annotations

import numpy as np
import pytest

from pysatl_core.estimation import MLE, FitProblem
from pysatl_core.estimation.formulas import AnalyticalEstimate, EstimationFormulaRegistry
from pysatl_core.estimation.methods.mle.likelihood import LogLikelihood
from pysatl_core.families.builtins.continuous.normal import configure_normal_family
from pysatl_core.families.configuration import (
    configure_families_register,
    reset_families_register,
)
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import FamilyName, UnivariateContinuous


def test_registry_is_a_singleton_and_registration_is_shared(normal_family):
    EstimationFormulaRegistry.closed_form_for(normal_family, "mle")
    first_instance = EstimationFormulaRegistry._instance
    assert first_instance is not None

    def custom(_sample, _fixed):
        return AnalyticalEstimate({"mu": 9.0, "sigma": 2.0})

    EstimationFormulaRegistry.register_closed_form(normal_family, "custom", custom)
    assert EstimationFormulaRegistry.closed_form_for(normal_family, "custom") is custom
    assert EstimationFormulaRegistry._instance is first_instance


def test_registered_formula_and_start_are_used(normal_family, monkeypatch):
    sample = np.array([1.0, 2.0, 3.0])
    EstimationFormulaRegistry.register_moment_start(
        normal_family, lambda _: {"mu": 9.0, "sigma": 2.0}, replace=True
    )
    EstimationFormulaRegistry.register_closed_form(
        normal_family,
        "mle",
        lambda _sample, _fixed: AnalyticalEstimate({"mu": 9.0, "sigma": 2.0}),
        replace=True,
    )

    assert FitProblem.prepare(normal_family, sample).probe.parameters == {
        "mu": 9.0,
        "sigma": 2.0,
    }
    original = LogLikelihood.of
    calls = 0

    def counted(family, values):
        nonlocal calls
        calls += 1
        return original(family, values)

    monkeypatch.setattr(LogLikelihood, "of", counted)
    result = MLE().estimate(normal_family, sample)
    assert result.route == "closed_form"
    assert result.params.parameters == {"mu": 9.0, "sigma": 2.0}
    assert result.log_likelihood == pytest.approx(original(normal_family, sample).at(result.params))
    assert calls == 1


@pytest.mark.parametrize("log_likelihood", [0.0, 7.5, -np.inf])
def test_precomputed_log_likelihood_is_used(normal_family, monkeypatch, log_likelihood):
    EstimationFormulaRegistry.register_closed_form(
        normal_family,
        "mle",
        lambda _sample, _fixed: AnalyticalEstimate({"mu": 2.0, "sigma": 1.0}, log_likelihood),
        replace=True,
    )

    def unexpected_objective(_family, _sample):
        raise AssertionError("a supplied log-likelihood must be used directly")

    monkeypatch.setattr(LogLikelihood, "of", unexpected_objective)
    result = MLE().estimate(normal_family, np.array([1.0, 2.0, 3.0]))
    assert result.route == "closed_form"
    assert result.params.parameters == {"mu": 2.0, "sigma": 1.0}
    assert result.log_likelihood == log_likelihood
    assert result.success == np.isfinite(log_likelihood)


def test_a_view_finds_its_original_familys_rules(normal_family):
    view = normal_family.view(mu=0.0)
    assert EstimationFormulaRegistry.closed_form_for(
        view, "mle"
    ) is EstimationFormulaRegistry.closed_form_for(normal_family, "mle")
    assert EstimationFormulaRegistry.moment_start_for(
        view
    ) is EstimationFormulaRegistry.moment_start_for(normal_family)


def test_family_identity_prevents_same_name_collisions(normal_family):
    unrelated = ParametricFamily(
        name=normal_family.name,
        distr_type=UnivariateContinuous,
        distr_parametrizations=["other"],
        distr_characteristics={},
    )
    assert EstimationFormulaRegistry.closed_form_for(normal_family, "mle") is not None
    assert EstimationFormulaRegistry.closed_form_for(unrelated, "mle") is None
    assert EstimationFormulaRegistry.moment_start_for(unrelated) is None


def test_duplicate_registration_requires_explicit_replacement(normal_family):
    def first(_sample, _fixed):
        return AnalyticalEstimate({"mu": 1.0, "sigma": 1.0})

    def second(_sample, _fixed):
        return AnalyticalEstimate({"mu": 2.0, "sigma": 1.0})

    EstimationFormulaRegistry.register_closed_form(normal_family, "custom", first)
    with pytest.raises(ValueError, match="already registered"):
        EstimationFormulaRegistry.register_closed_form(normal_family, "custom", second)
    EstimationFormulaRegistry.register_closed_form(normal_family, "custom", second, replace=True)
    assert EstimationFormulaRegistry.closed_form_for(normal_family, "custom") is second

    with pytest.raises(ValueError, match="already registered"):
        EstimationFormulaRegistry.register_moment_start(
            normal_family, lambda _: {"mu": 2.0, "sigma": 1.0}
        )


def test_configuring_one_builtin_family_registers_its_formulas():
    configure_normal_family()
    normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
    assert EstimationFormulaRegistry.closed_form_for(normal, "mle") is not None
    assert EstimationFormulaRegistry.moment_start_for(normal) is not None


def test_reset_of_families_replaces_the_formula_registry(normal_family):
    EstimationFormulaRegistry.closed_form_for(normal_family, "mle")
    old_instance = EstimationFormulaRegistry._instance
    reset_families_register()
    new_normal = configure_families_register().get(FamilyName.NORMAL)
    assert new_normal is not normal_family
    assert EstimationFormulaRegistry._instance is not old_instance
    assert EstimationFormulaRegistry.closed_form_for(new_normal, "mle") is not None
    assert EstimationFormulaRegistry.closed_form_for(normal_family, "mle") is None
