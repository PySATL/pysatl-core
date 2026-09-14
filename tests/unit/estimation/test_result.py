"""
The result object and the support-dependence heuristic.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import dataclasses
import math

import numpy as np
import pytest

from pysatl_core.estimation import MLEResult, starting_point, support_depends_on_params


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
        # Assembled by hand so that the arithmetic is checked against numbers
        # chosen here, not against whatever a fit happens to produce.
        result = MLEResult(
            family_name="Normal",
            params=normal_family.base(mu=0.0, sigma=1.0),
            log_likelihood=-100.0,
            n_params=3,
            n_observations=50,
            method="numeric",
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
        # The heuristic needs a point of the parameter space, not the data: the
        # sample only ever served to produce one.  Saying so at the call site
        # is what stopped the fit recomputing the same start three times.
        assert support_depends_on_params(family, starting_point(family, sample)) is expected

    def test_a_uniform_view_still_has_a_moving_support(self, uniform_family, rng):
        view = uniform_family.view(lower_bound=0.0)
        probe = starting_point(view, rng.uniform(0.5, 3.0, 100))
        assert support_depends_on_params(view, probe) is True

    def test_a_normal_view_still_has_a_fixed_support(self, normal_family, rng):
        view = normal_family.view(mu=0.0)
        probe = starting_point(view, rng.normal(0.0, 1.0, 100))
        assert support_depends_on_params(view, probe) is False
