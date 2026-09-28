"""
The objective function, its gradient and the out-of-support penalty.

The gradient checks follow the pattern already used for ``score`` in
``tests/unit/families/builtins/continuous``: the analytical derivative is
compared with a central difference of the objective itself, so a sign error or
a missing chain-rule factor cannot slip through.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
import pytest

from pysatl_core.estimation import (
    EstimationError,
    from_vector,
    to_vector,
)
from pysatl_core.estimation.methods.mle import OUT_OF_SUPPORT_PENALTY, LogLikelihood

EPS = 1e-6


def central_difference(
    fun: Callable[[np.ndarray], float], x0: np.ndarray, eps: float = EPS
) -> np.ndarray:
    """Central difference of *fun* at *x0*, coordinate by coordinate."""
    numeric = np.empty_like(x0)
    for i in range(x0.size):
        step = np.zeros_like(x0)
        step[i] = eps
        numeric[i] = (fun(x0 + step) - fun(x0 - step)) / (2 * eps)
    return numeric


class TestGradient:
    @pytest.mark.parametrize(
        "family_fixture, draw, probe",
        [
            ("normal_family", lambda r: r.normal(1.0, 2.0, 400), np.array([1.3, 1.7])),
            ("gamma_family", lambda r: r.gamma(3.0, 2.0, 400), np.array([2.4, 2.3])),
            ("exponential_family", lambda r: r.exponential(2.0, 400), np.array([0.6])),
        ],
    )
    def test_matches_a_central_difference(self, request, rng, family_fixture, draw, probe):
        family = request.getfixturevalue(family_fixture)
        sample = draw(rng)
        objective = LogLikelihood.of(family, sample)
        gradient = objective.gradient_or_none
        assert gradient is not None

        analytic = gradient(probe)
        numeric = central_difference(objective, probe)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5)

    def test_uniform_gradient_matches_inside_the_support(self, uniform_family, rng):
        sample = rng.uniform(2.0, 5.0, 400)
        objective = LogLikelihood.of(uniform_family, sample)
        gradient = objective.gradient_or_none
        assert gradient is not None
        probe = np.array([sample.min() - 0.5, sample.max() + 0.5])
        np.testing.assert_allclose(gradient(probe), central_difference(objective, probe), rtol=1e-5)

    def test_gradient_is_none_without_score(self, scoreless_family, rng):
        _objective = LogLikelihood.of(scoreless_family, rng.normal(0.0, 1.0, 50))
        gradient = _objective.gradient_or_none
        assert gradient is None

    def test_gradient_is_zero_where_constraints_fail(self, normal_family, rng):
        _objective = LogLikelihood.of(normal_family, rng.normal(0.0, 1.0, 50))
        gradient = _objective.gradient_or_none
        assert gradient is not None
        np.testing.assert_array_equal(gradient(np.array([0.0, -1.0])), np.zeros(2))


class TestObjective:
    def test_is_the_negated_log_likelihood_where_everything_fits(self, normal_family, rng):
        sample = rng.normal(0.0, 1.0, 200)
        objective = LogLikelihood.of(normal_family, sample)
        params = normal_family.base(mu=0.2, sigma=1.1)
        assert objective(to_vector(params)) == pytest.approx(
            -LogLikelihood.of(normal_family, sample).at(params)
        )

    def test_rejects_inadmissible_parameters_with_inf(self, normal_family, rng):
        objective = LogLikelihood.of(normal_family, rng.normal(0.0, 1.0, 50))
        assert objective(np.array([0.0, 0.0])) == np.inf
        assert objective(np.array([0.0, -1.0])) == np.inf

    def test_charges_the_penalty_once_per_unexplained_point(self, uniform_family, rng):
        sample = rng.uniform(0.0, 1.0, 100)
        objective = LogLikelihood.of(uniform_family, sample)
        covering = np.array([-1.0, 2.0])
        n_excluded = int((sample > 0.5).sum())
        shrunk = np.array([-1.0, 0.5])
        difference = objective(shrunk) - objective(covering)
        explained_terms = np.log(3.0) * 100 - np.log(1.5) * (100 - n_excluded)
        assert difference == pytest.approx(
            n_excluded * OUT_OF_SUPPORT_PENALTY - explained_terms, rel=1e-9
        )

    def test_the_penalty_follows_the_documented_rule(self):
        """The constant is defined by a rule; the rule is what a test should pin.

        Asserting the number 70978.27 instead reports a change of multiplier as
        '70978.27 != 141956.54', which says nothing about what moved.
        """
        assert np.isfinite(OUT_OF_SUPPORT_PENALTY)
        assert pytest.approx(float(np.log(np.finfo(np.float64).max) * 100)) == (
            OUT_OF_SUPPORT_PENALTY
        )

    def test_the_penalty_outweighs_any_admissible_configuration(self, uniform_family, rng):
        """Why the multiplier is 100, stated as the property it exists for.

        One unexplained observation has to cost more than any likelihood a
        parameter set can buy back on the rest; otherwise the optimizer would
        trade coverage for density and the fallback policy would be reasoning
        about a surface with no such staircase in it.
        """
        sample = rng.uniform(0.0, 1.0, 100)
        objective = LogLikelihood.of(uniform_family, sample)
        covering = objective(np.array([-1.0, 2.0]))
        excluding_one = objective(np.array([-1.0, float(np.sort(sample)[-2])]))
        assert excluding_one - covering > OUT_OF_SUPPORT_PENALTY / 2


class TestLogLikelihood:
    def test_is_minus_infinity_when_a_point_is_outside_the_support(self, uniform_family, rng):
        sample = rng.uniform(0.0, 1.0, 50)
        params = uniform_family.base(lower_bound=0.25, upper_bound=0.75)
        assert LogLikelihood.of(uniform_family, sample).at(params) == -np.inf

    def test_carries_no_penalty_term(self, uniform_family, rng):
        sample = rng.uniform(0.0, 1.0, 50)
        params = uniform_family.base(lower_bound=-1.0, upper_bound=2.0)
        value = LogLikelihood.of(uniform_family, sample).at(params)
        assert value == pytest.approx(-50 * np.log(3.0))

    def test_a_family_without_lpdf_is_refused(self, lpdfless_family, rng):
        params = lpdfless_family.base(rate=1.0)
        with pytest.raises(EstimationError, match="does not declare the 'lpdf' characteristic"):
            LogLikelihood.of(lpdfless_family, rng.exponential(1.0, 10)).at(params)

    def test_fitting_a_family_without_lpdf_is_refused(self, lpdfless_family, rng):
        with pytest.raises(EstimationError, match="loses precision in the tails"):
            lpdfless_family.fit(rng.exponential(1.0, 10))


class TestVectorConversion:
    def test_round_trips(self, gamma_family):
        params = gamma_family.base(k=2.0, theta=3.0)
        assert from_vector(gamma_family.base, to_vector(params)).parameters == (params.parameters)
