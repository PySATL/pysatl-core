"""
Level 3: the numerical path, the optimizer policy and the fallback.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from scipy.optimize import minimize

from pysatl_core.estimation import (
    DEFAULT_OPTIMIZER,
    FALLBACK_OPTIMIZER,
    fit_family,
    make_objective,
    moments,
    starting_point,
    to_vector,
)


class TestNumericMatchesClosedForm:
    """The numerical path against the exact answer.

    The first two tests below start *at* the answer — for a normal family the
    moment start is already the maximum likelihood estimate — so what they
    really assert is that the optimizer does not wander away from it.  That is
    worth pinning, but it is not convergence;
    ``test_converges_from_a_start_away_from_the_answer`` is the one that
    exercises that, by driving the objective from a displaced point.
    """

    def test_lbfgsb_does_not_wander_off_the_closed_form_estimate(self, normal_family, rng):
        sample = rng.normal(2.0, 1.5, 1000)
        exact = normal_family.fit(sample)
        with pytest.warns(UserWarning, match="closed-form MLE"):
            numeric = normal_family.fit(sample, optimizer="L-BFGS-B")
        assert numeric.method == "numeric"
        assert numeric.optimizer == "L-BFGS-B"
        assert numeric.params.mu == pytest.approx(exact.params.mu, abs=1e-6)
        assert numeric.params.sigma == pytest.approx(exact.params.sigma, abs=1e-6)

    def test_nelder_mead_does_not_wander_off_the_closed_form_estimate(self, normal_family, rng):
        sample = rng.normal(2.0, 1.5, 1000)
        exact = normal_family.fit(sample)
        with pytest.warns(UserWarning, match="closed-form MLE"):
            numeric = normal_family.fit(sample, optimizer="Nelder-Mead")
        assert numeric.params.mu == pytest.approx(exact.params.mu, abs=1e-4)
        assert numeric.params.sigma == pytest.approx(exact.params.sigma, abs=1e-4)

    def test_converges_from_a_start_away_from_the_answer(self, normal_family, rng):
        sample = rng.normal(2.0, 1.5, 1000)
        exact = normal_family.fit(sample)
        objective, gradient = make_objective(normal_family, sample)
        displaced = np.array([-4.0, 6.0])
        found = minimize(
            objective,
            displaced,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[(-np.inf, np.inf), (1e-12, np.inf)],
        )
        assert found.success
        assert found.x[0] == pytest.approx(exact.params.mu, abs=1e-4)
        assert found.x[1] == pytest.approx(exact.params.sigma, abs=1e-4)

    def test_explicit_optimizer_warns_and_switches_branch(self, normal_family, rng):
        sample = rng.normal(0.0, 1.0, 200)
        with pytest.warns(UserWarning, match="has a closed-form MLE"):
            result = normal_family.fit(sample, optimizer="L-BFGS-B")
        assert result.method == "numeric"
        assert "closed-form MLE" in result.message

    def test_options_reach_scipy(self, gamma_family, rng, monkeypatch):
        # 'n_iterations <= 1' alone would also hold if the option were dropped
        # on the floor and the optimizer simply converged in one step. What
        # proves the option arrived is that the capped fit is *worse* than the
        # uncapped one, and reports the failure.
        #
        # The start is deliberately moved far from the answer. With the real
        # moment start, gamma's start is close enough that L-BFGS-B sometimes
        # converges in a single iteration, and 'the uncapped fit took more than
        # one' is then a property of the draw rather than of the option:
        # measured over 300 seeds, that happens on 2 of them (2009, 2282),
        # where the capped and uncapped fits become indistinguishable. Note
        # that enlarging the sample does not help — the number of iterations
        # follows the quality of the start, and a larger sample only makes the
        # moment start *better*.
        monkeypatch.setitem(
            moments._MOMENT_STARTS, gamma_family.name, lambda s: {"k": 20.0, "theta": 20.0}
        )
        sample = rng.gamma(3.0, 2.0, 500)
        capped = gamma_family.fit(sample, options={"maxiter": 1})
        assert capped.n_iterations is not None
        assert capped.n_iterations <= 1
        uncapped = gamma_family.fit(sample)
        assert uncapped.n_iterations is not None
        assert uncapped.n_iterations > 1
        assert capped.log_likelihood < uncapped.log_likelihood
        assert capped.success is False


class TestDefaultNumericPath:
    def test_gamma_has_no_closed_form_and_runs_the_optimizer(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 800)
        result = gamma_family.fit(sample)
        assert result.method == "numeric"
        assert result.optimizer == DEFAULT_OPTIMIZER
        assert result.success
        assert result.n_function_evaluations is not None

    def test_a_family_without_score_is_still_fitted(self, scoreless_family, rng):
        sample = rng.normal(4.0, 2.0, 500)
        objective, gradient = make_objective(scoreless_family, sample)
        assert gradient is None, "the fixture family declares no score"
        result = scoreless_family.fit(sample)
        assert result.method == "numeric"
        assert result.params.mu == pytest.approx(sample.mean(), abs=1e-4)
        assert result.params.sigma == pytest.approx(sample.std(), abs=1e-4)


class TestOptimizerFallback:
    def test_falls_back_to_nelder_mead_and_says_so(self, boxed_family, rng):
        # The support moves with the parameters, so the objective is a
        # staircase: L-BFGS-B takes no step at all. Every built-in family of
        # that shape has a closed form, which is why the fixture family exists.
        sample = rng.uniform(2.0, 5.0, 500)
        with pytest.warns(UserWarning, match="declares no 'param_bounds'"):
            result = boxed_family.fit(sample)

        # What the test is for — the substitution policy, and that it is stated:
        assert result.optimizer == FALLBACK_OPTIMIZER
        assert "fell back to" in result.message
        assert DEFAULT_OPTIMIZER in result.message

        # How *close* Nelder-Mead gets to min(x)/max(x) is not a property: it
        # stops where its simplex collapses, and on some samples that is two
        # orders of magnitude further out than the 1e-3 this once asserted.
        # The property is that the interval it returns explains the whole
        # sample — otherwise the likelihood would be -inf and the estimate
        # meaningless.
        assert result.params.low <= sample.min()
        assert result.params.high >= sample.max()
        assert np.isfinite(result.log_likelihood)

    def test_no_fallback_when_the_caller_named_the_optimizer(self, boxed_family, rng):
        # An explicitly requested method is honoured even when it does badly;
        # substituting it silently is exactly the SciPy behaviour this package
        # avoids. The caller sees the failure in 'success' and 'message'.
        sample = rng.uniform(2.0, 5.0, 500)
        with pytest.warns(UserWarning, match="declares no 'param_bounds'"):
            result = boxed_family.fit(sample, optimizer="L-BFGS-B")
        assert result.optimizer == "L-BFGS-B"
        assert "fell back to" not in result.message


class TestUniformNumericCaveat:
    def test_numeric_uniform_warns_twice_and_records_the_caveat(self, uniform_family, rng):
        sample = rng.uniform(2.0, 5.0, 500)
        with pytest.warns(UserWarning) as caught:
            result = uniform_family.fit(sample, optimizer="L-BFGS-B")
        messages = [str(w.message) for w in caught]
        assert any("has a closed-form MLE" in m for m in messages)
        assert any("unreliable for a uniform family" in m for m in messages)
        assert "unreliable for a uniform family" in result.message
        assert result.method == "numeric"

    def test_the_closed_form_beats_the_numeric_path_here(self, uniform_family, rng):
        sample = rng.uniform(2.0, 5.0, 500)
        exact = uniform_family.fit(sample)
        with pytest.warns(UserWarning):
            numeric = uniform_family.fit(sample, optimizer="L-BFGS-B")
        assert exact.log_likelihood > numeric.log_likelihood


class TestStartingPoint:
    def test_uniform_start_covers_the_sample(self, uniform_family, rng):
        # A start that excludes an observation puts the objective on a wall of
        # pure penalty, where no optimizer has a direction to follow.
        sample = rng.uniform(-3.0, 3.0, 300)
        start = starting_point(uniform_family, sample).parameters
        assert start["lower_bound"] < sample.min()
        assert start["upper_bound"] > sample.max()

    def test_normal_start_is_the_moment_estimate(self, normal_family, rng):
        sample = rng.normal(7.0, 0.5, 300)
        start = starting_point(normal_family, sample).parameters
        assert start["mu"] == pytest.approx(sample.mean())
        assert start["sigma"] == pytest.approx(sample.std())

    def test_start_of_a_view_holds_only_free_parameters(self, normal_family, rng):
        sample = rng.normal(7.0, 0.5, 300)
        start = starting_point(normal_family.view(mu=0.0), sample)
        assert set(start.parameters) == {"sigma"}

    def test_unknown_family_starts_from_ones(self, make_boxed_family, rng):
        from pysatl_core.estimation.moments import _MOMENT_STARTS

        family = make_boxed_family("BoxedWithoutRule")
        assert family.name not in _MOMENT_STARTS
        start = starting_point(family, rng.uniform(0.0, 1.0, 10))
        assert start.parameters == {"low": 1.0, "high": 1.0}

    def test_registering_a_rule_changes_the_start(self, make_boxed_family, rng):
        from pysatl_core.estimation.moments import _MOMENT_STARTS, register_moment_start

        family = make_boxed_family("BoxedWithRule")
        register_moment_start(
            family.name,
            lambda s: {"low": float(s.min()) - 1.0, "high": float(s.max()) + 1.0},
        )
        try:
            sample = rng.uniform(0.0, 1.0, 50)
            start = starting_point(family, sample).parameters
            assert start["low"] == pytest.approx(sample.min() - 1.0)
        finally:
            _MOMENT_STARTS.pop(family.name, None)


class TestVectorRoundTrip:
    def test_order_follows_field_declaration(self, normal_family, rng):
        params = normal_family.base(mu=1.0, sigma=2.0)
        assert list(to_vector(params)) == [1.0, 2.0]

    def test_view_vector_holds_free_parameters_only(self, normal_family):
        view = normal_family.view(mu=0.0)
        assert list(to_vector(view.base(sigma=3.0))) == [3.0]


class TestFitFamilyEntryPoint:
    def test_the_method_and_the_function_agree(self, normal_family, rng):
        sample = rng.normal(1.0, 1.0, 100)
        assert normal_family.fit(sample).params == fit_family(normal_family, sample).params
