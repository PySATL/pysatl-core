"""
Defensive branches: degenerate data, awkward families, custom solvers.

These paths exist because the optimizer probes places a well-behaved caller
never visits, and because a family author is free to write a constraint or a
score that is less total than the built-in ones.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from pysatl_core.estimation import (
    MLE,
    EstimationError,
    starting_point,
    validate_sample,
)
from pysatl_core.estimation.methods.mle import LogLikelihood
from pysatl_core.estimation.optimizers import ObjectiveFunc
from pysatl_core.estimation.parameters.start import project_onto_base
from pysatl_core.estimation.parameters.vectors import satisfies_constraints
from pysatl_core.estimation.problem.support import _perturb
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    constraint,
    parametrization,
)
from pysatl_core.types import UnivariateContinuous
from tests.unit.estimation.conftest import use_moment_start


class TestConstraintPredicates:
    def test_a_predicate_that_raises_counts_as_not_holding(self):
        family = ParametricFamily(
            name="FragileConstraint",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["p"],
            distr_characteristics={},
        )

        @parametrization(family=family, name="p")
        class _P(Parametrization):
            a: float

            @constraint(description="1 / a > 0")
            def check(self) -> bool:
                return 1.0 / self.a > 0.0

        assert satisfies_constraints(_P(a=2.0)) is True
        assert satisfies_constraints(_P(a=0.0)) is False

    def test_a_nan_probe_is_inadmissible(self, normal_family):
        assert satisfies_constraints(normal_family.base(mu=0.0, sigma=np.nan)) is False


class TestGradientWhereScoreIsUndefined:
    def test_a_boundary_point_yields_a_zero_gradient_instead_of_raising(self, gamma_family):
        sample = np.array([0.0, 1.0, 2.0, 3.0])
        objective = LogLikelihood.of(gamma_family, sample)
        gradient = objective.gradient_or_none
        assert gradient is not None
        at_k_one = np.array([1.0, 1.0])
        assert np.isfinite(objective(at_k_one))
        with pytest.raises(ValueError):
            gamma_family.score(gamma_family.base(k=1.0, theta=1.0), sample)
        np.testing.assert_array_equal(gradient(at_k_one), np.zeros(2))


class TestSampleCoercion:
    def test_an_unconvertible_sample_is_reported_in_our_own_words(self, normal_family):
        with pytest.raises(ValueError, match="convertible to a float array"):
            validate_sample(normal_family, "not a sample")

    def test_a_list_is_accepted(self, normal_family):
        """``array_like`` is what the body accepts, and now what it declares."""
        np.testing.assert_array_equal(
            validate_sample(normal_family, [1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
        )


class TestMomentRules:
    def test_a_degenerate_gamma_sample_falls_back_to_ones(self, gamma_family):
        # Zero variance would make k = mean^2 / var infinite.
        rule = gamma_family.moment_start
        assert rule is not None
        assert rule(np.ones(10)) == {"k": 1.0, "theta": 1.0}
        assert rule(np.array([-1.0, -2.0])) == {"k": 1.0, "theta": 1.0}

    def test_a_rule_in_foreign_coordinates_is_not_projected(self, normal_family):
        view = normal_family.view(parametrization_name="meanVar", mu=0.0)
        assert project_onto_base(view, {"mu": 0.0, "sigma": 1.0}) is None

    def test_such_a_view_starts_from_ones(self, normal_family):
        view = normal_family.view(parametrization_name="meanVar", mu=0.0)
        assert starting_point(view, np.array([1.0, 2.0, 3.0])).parameters == {"var": 1.0}

    def test_a_rule_that_raises_does_not_stop_the_support_check(
        self, normal_family, monkeypatch, rng
    ):
        def explode(_sample: Any) -> dict[str, float]:
            raise ArithmeticError("no moments here")

        use_moment_start(monkeypatch, normal_family, explode)
        # The support check needs *some* parameter point to resolve the support
        # at. A broken moment rule must degrade to a default probe rather than
        # crash the fit before the data have even been inspected.
        sample = rng.normal(0.0, 1.0, 50)
        result = normal_family.fit(sample)
        assert result.route == "closed_form"
        assert result.params.mu == pytest.approx(sample.mean())


class TestMomentStartFormulas:
    """The formulas themselves, not just their degenerate fallbacks.

    Swapping ``k`` and ``theta`` in the gamma rule breaks no other test in this
    suite: the optimizer converges from the wrong start too, only slower, and
    nothing measures the iteration count.  The rule is part of what a family
    declares, so it is checked directly, through the family that declares it.
    """

    @staticmethod
    def rule_of(family: ParametricFamily) -> Any:
        rule = family.moment_start
        assert rule is not None, f"family '{family.name}' was expected to declare a rule"
        return rule

    def test_gamma_start_is_the_moment_estimate(self, gamma_family):
        sample = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        mean, var = float(sample.mean()), float(sample.var())
        assert self.rule_of(gamma_family)(sample) == {
            "k": pytest.approx(mean * mean / var),
            "theta": pytest.approx(var / mean),
        }
        assert mean * mean / var != pytest.approx(var / mean)

    def test_exponential_start_is_the_reciprocal_mean(self, exponential_family):
        rule = self.rule_of(exponential_family)
        assert rule(np.array([2.0, 4.0, 6.0])) == {"lambda_": pytest.approx(0.25)}

    def test_exponential_start_survives_a_zero_mean(self, exponential_family):
        # 1 / mean is undefined; the rule must still hand back a usable point.
        assert self.rule_of(exponential_family)(np.zeros(5)) == {"lambda_": 1.0}


class TestDegenerateGradient:
    def test_a_non_finite_gradient_component_is_zeroed(self, gamma_family):
        """``jac`` must return a finite vector: no optimizer survives a nan.

        ``theta = 1e-300`` overflows the derivative with respect to theta, so
        the raw score is not finite. The input is chosen deliberately: at
        gentler values the gradient is finite on its own and the test would
        pass without exercising the guard at all.
        """
        _objective = LogLikelihood.of(gamma_family, np.array([1.0, 2.0]))
        gradient = _objective.gradient_or_none
        assert gradient is not None
        value = gradient(np.array([1.0, 1e-300]))
        assert np.all(np.isfinite(value))

    def test_a_point_inside_the_support_with_infinite_density_is_dropped(self, gamma_family):
        """Inside the support is not the same as usable.

        With ``k < 1`` the gamma log-density is ``+inf`` at zero, which is
        inside the support. Summing it would report a *better* likelihood for a
        sample the parameters cannot actually explain.
        """
        sample = np.array([0.0, 1.0, 2.0])
        params = gamma_family.base(k=0.5, theta=1.0)
        assert LogLikelihood.of(gamma_family, sample).at(params) == -np.inf


class TestPerturbationMovesEveryParameter:
    def test_a_parameter_sitting_at_zero_is_moved(self, uniform_family):
        """``2v + 1``, not ``2v`` — and the ``+ 1`` is what a zero needs.

        With ``2v`` both probe points of a family pinned at zero would resolve
        the same support, ``support_depends_on_params`` would answer ``False``
        for a family whose support does move, and data outside the support
        would start raising ``FitDataError`` instead of being charged a
        penalty. The error contract would change silently.
        """
        params = uniform_family.base(lower_bound=0.0, upper_bound=0.0)
        moved = _perturb(uniform_family, params)
        assert moved.parameters["lower_bound"] != 0.0
        assert moved.parameters["upper_bound"] != 0.0

    def test_the_perturbed_point_keeps_its_class(self, normal_family):
        """``_perturb`` is declared to return the type it was given."""
        view = normal_family.view(mu=0.0)
        moved = _perturb(view, view.base(sigma=2.0))
        assert type(moved) is view.base


class TestCustomOptimizer:
    def test_a_solver_callable_is_accepted_and_named(self, gamma_family, rng):
        calls: list[str] = []

        def fixed_point_solver(
            fun: ObjectiveFunc, x0: NDArray[np.float64], **kwargs: object
        ) -> OptimizeResult:
            """A stand-in solver: records what it was handed, returns the start."""
            calls.append("called")
            assert "jac" in kwargs, "a custom solver should still receive the gradient"
            outcome = OptimizeResult()
            outcome["x"] = np.asarray(x0)
            outcome["fun"] = float(fun(x0))
            outcome["success"] = True
            outcome["nit"] = 1
            outcome["nfev"] = 1
            outcome["message"] = "stopped by the test solver"
            return outcome

        sample = rng.gamma(3.0, 2.0, 200)
        result = gamma_family.fit(sample, estimator=MLE(optimizer=fixed_point_solver))
        assert calls == ["called"]
        assert result.optimizer == "fixed_point_solver"
        assert result.route == "numeric"
        # The solver returned the starting point untouched.
        assert result.params.k == pytest.approx(
            starting_point(gamma_family, sample).parameters["k"]
        )


class TestMinimalCustomSolver:
    """A solver is only obliged to return an ``OptimizeResult`` carrying ``x``.

    ``scipy.optimize.minimize`` always fills ``success``, ``message``, ``nit``
    and ``nfev``, but those are conventions its built-in methods follow, not
    part of the protocol a caller-supplied solver has to honour.
    """

    @staticmethod
    def _bare_solver(
        fun: ObjectiveFunc, x0: NDArray[np.float64], **kwargs: object
    ) -> OptimizeResult:
        outcome = OptimizeResult()
        outcome["x"] = np.asarray(x0)
        return outcome

    def test_a_solver_reporting_only_x_does_not_crash(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 200)
        result = gamma_family.fit(sample, estimator=MLE(optimizer=self._bare_solver))
        assert result.route == "numeric"
        assert result.n_iterations is None
        assert result.n_function_evaluations is None

    def test_an_unstated_verdict_is_not_read_as_success(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 200)
        result = gamma_family.fit(sample, estimator=MLE(optimizer=self._bare_solver))
        assert result.success is False
        assert "no convergence flag" in result.message

    def test_a_solver_returning_no_estimate_is_refused(self, gamma_family, rng):
        def empty_solver(
            fun: ObjectiveFunc, x0: NDArray[np.float64], **kwargs: object
        ) -> OptimizeResult:
            return OptimizeResult()

        with pytest.raises(EstimationError, match="returned no 'x' field"):
            gamma_family.fit(rng.gamma(3.0, 2.0, 200), estimator=MLE(optimizer=empty_solver))


class TestMethodCapabilityGating:
    def test_a_gradient_free_method_is_not_handed_a_gradient(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = gamma_family.fit(sample, estimator=MLE(optimizer="Powell"))
        assert [str(w.message) for w in caught] == []
        assert result.optimizer == "Powell"
        default = gamma_family.fit(sample)
        assert result.log_likelihood <= default.log_likelihood
        assert result.log_likelihood == pytest.approx(default.log_likelihood, rel=1e-2)

    def test_an_unbounded_method_is_not_handed_bounds(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 300)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = gamma_family.fit(sample, estimator=MLE(optimizer="BFGS"))
        assert [str(w.message) for w in caught] == []
        assert result.optimizer == "BFGS"
