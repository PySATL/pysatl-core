"""
The result object and the support-dependence heuristic.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import dataclasses
import gc
import math
import weakref
from weakref import WeakKeyDictionary

import numpy as np
import pytest

from pysatl_core.estimation import (
    FitDataError,
    MLEResult,
    starting_point,
    support_depends_on_params,
)
from pysatl_core.estimation.problem import support as support_module
from pysatl_core.estimation.problem.problem import FitProblem
from pysatl_core.families.parametric_family import ParametricFamily


class TestMLEResultShape:
    def test_is_frozen(self, normal_family, rng):
        result = normal_family.fit(rng.normal(0.0, 1.0, 50))
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.success = False

    def test_carries_the_family_name(self, gamma_family, rng):
        result = gamma_family.fit(rng.gamma(2.0, 1.0, 200))
        assert result.family_name == "Gamma"

    def test_closed_form_results_report_no_optimizer(self, normal_family, rng):
        result = normal_family.fit(rng.normal(0.0, 1.0, 50))
        assert result.optimizer is None
        assert result.n_iterations is None
        assert result.n_function_evaluations is None

    def test_criteria_use_the_free_parameter_count(self, normal_family):
        result = MLEResult(
            family_name="Normal",
            params=normal_family.base(mu=0.0, sigma=1.0),
            log_likelihood=-100.0,
            n_params=3,
            n_observations=50,
            estimator="mle",
            route="numeric",
            optimizer="L-BFGS-B",
            success=True,
            message="",
        )
        assert result.aic == pytest.approx(2 * 3 + 200.0)
        assert result.bic == pytest.approx(3 * math.log(50) + 200.0)


class TestSupportHeuristic:
    @pytest.mark.parametrize(
        "family_fixture, expected",
        [
            ("normal_family", False),
            ("gamma_family", False),
            ("exponential_family", False),
            ("uniform_family", True),
        ],
    )
    def test_answers_correctly_for_the_built_in_families(
        self, request, rng, family_fixture, expected
    ):
        family = request.getfixturevalue(family_fixture)
        sample = np.abs(rng.normal(2.0, 1.0, 100)) + 0.5
        assert support_depends_on_params(family, starting_point(family, sample)) is expected

    def test_a_uniform_view_still_has_a_moving_support(self, uniform_family, rng):
        view = uniform_family.view(lower_bound=0.0)
        probe = starting_point(view, rng.uniform(0.5, 3.0, 100))
        assert support_depends_on_params(view, probe) is True

    def test_a_normal_view_still_has_a_fixed_support(self, normal_family, rng):
        view = normal_family.view(mu=0.0)
        probe = starting_point(view, rng.normal(0.0, 1.0, 100))
        assert support_depends_on_params(view, probe) is False


class TestSupportDependenceCache:
    def test_reuses_classification_across_fits_but_validates_each_sample(
        self, exponential_family, monkeypatch
    ):
        monkeypatch.setattr(support_module, "_support_info_cache", WeakKeyDictionary())
        original = support_module.classify_support
        calls = 0

        def counted(family, probe):
            nonlocal calls
            calls += 1
            return original(family, probe)

        monkeypatch.setattr(support_module, "classify_support", counted)
        sample = np.array([0.4, 0.8, 1.3, 2.1])
        first = FitProblem.prepare(exponential_family, sample)
        first.reject_data_outside_a_fixed_support()
        assert "probe" in first.__dict__
        second = FitProblem.prepare(exponential_family, sample * 2)
        second.reject_data_outside_a_fixed_support()
        assert "probe" not in second.__dict__
        assert second.support_info.fixed_support is first.support_info.fixed_support

        def unexpected_probe():
            raise AssertionError("a cached support must not rebuild a probe")

        assert not support_module.cached_support_depends_on_params(
            exponential_family, unexpected_probe
        )
        assert exponential_family.fit(sample * 2).success
        with pytest.raises(FitDataError):
            exponential_family.fit(np.array([-1.0, 0.8, 1.3, 2.1]))
        assert calls == 1

    def test_view_has_independent_entry_and_hit_does_not_build_probe(
        self, uniform_family, monkeypatch
    ):
        monkeypatch.setattr(support_module, "_support_info_cache", WeakKeyDictionary())
        original = support_module.classify_support
        calls = 0

        def counted(family, probe):
            nonlocal calls
            calls += 1
            return original(family, probe)

        monkeypatch.setattr(support_module, "classify_support", counted)
        sample = np.array([0.4, 0.8, 1.3, 2.1])
        first = FitProblem.prepare(uniform_family, sample)
        assert first.support_moves
        second = FitProblem.prepare(uniform_family, sample)
        assert second.support_moves
        assert "probe" not in second.__dict__

        view = uniform_family.view(lower_bound=0.0)
        assert FitProblem.prepare(view, sample).support_moves
        assert calls == 2

    def test_view_entry_disappears_when_view_is_collected(self, normal_family, monkeypatch):
        cache: WeakKeyDictionary[ParametricFamily, support_module.SupportInfo] = WeakKeyDictionary()
        monkeypatch.setattr(support_module, "_support_info_cache", cache)
        view = normal_family.view(mu=0.0)
        problem = FitProblem.prepare(view, np.array([-1.0, 0.0, 1.0]))
        assert problem.support_moves is False
        reference = weakref.ref(view)
        assert len(cache) == 1

        del problem, view
        gc.collect()
        assert reference() is None
        assert len(cache) == 0
