"""
Optimizers as objects, and the fallback policy as one of them.

The point of this file is what it does *not* need.  The fallback rule — try
L-BFGS-B, and on failure Nelder-Mead, keeping the better point — used to be
twenty-five lines inside the maximum likelihood fit, and the only way to
exercise it was to find a family and a sample on which L-BFGS-B stalls.  Here
it is driven with two toy optimizers over a parabola: no family, no
parametrization, no likelihood.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.typing import NDArray

from pysatl_core.estimation import (
    MLE,
    CustomSolver,
    Optimizer,
    OptimizerOutcome,
    ParameterBox,
    ScipyMethod,
    WithFallback,
    optimizer_for,
)
from pysatl_core.estimation.methods.mle.likelihood import Evaluation, LogLikelihood
from pysatl_core.estimation.optimizers import GradientFunc, ObjectiveFunc
from pysatl_core.families.parametrizations import Parametrization


def parabola(vec: NDArray[np.float64]) -> float:
    """Minimised at the origin; any point is comparable to any other."""
    return float((np.asarray(vec, dtype=np.float64) ** 2).sum())


@dataclass(frozen=True, slots=True)
class Stub:
    """An optimizer that reports whatever the test tells it to."""

    label: str
    lands_on: NDArray[np.float64]
    success: bool | None = True
    n_iterations: int | None = 3
    seen_gradient: list[GradientFunc | None] = field(default_factory=list, compare=False)

    @property
    def name(self) -> str:
        return self.label

    def minimize(
        self,
        objective: ObjectiveFunc,
        x0: NDArray[np.float64],
        *,
        gradient: GradientFunc | None,
        bounds: list[tuple[float, float]] | None,
    ) -> OptimizerOutcome:
        self.seen_gradient.append(gradient)
        return OptimizerOutcome(
            x=self.lands_on,
            optimizer=self.label,
            success=self.success,
            message="",
            n_iterations=self.n_iterations,
            n_function_evaluations=None,
        )


class TestTheFallbackPolicy:
    def test_a_stub_is_an_optimizer(self):
        assert isinstance(Stub("s", np.zeros(1)), Optimizer)

    def test_the_second_never_runs_when_the_first_converges(self):
        first = Stub("first", np.array([1.0]))
        second = Stub("second", np.array([0.0]))
        outcome = WithFallback(first, second).minimize(
            parabola, np.array([5.0]), gradient=None, bounds=None
        )
        assert outcome.optimizer == "first"
        assert second.seen_gradient == []
        assert outcome.notes == ()

    def test_a_failure_is_retried_and_the_better_point_wins(self):
        first = Stub("first", np.array([4.0]), success=False)
        second = Stub("second", np.array([0.5]))
        outcome = WithFallback(first, second).minimize(
            parabola, np.array([5.0]), gradient=None, bounds=None
        )
        assert outcome.optimizer == "second"
        assert outcome.x == pytest.approx(np.array([0.5]))
        assert "did not converge" in outcome.notes[0]
        assert "fell back to second" in outcome.notes[0]

    def test_a_retry_that_lands_worse_is_not_adopted(self):
        # A second opinion, not a verdict: adopting it unseen returned a worse
        # likelihood on about half of sampled fits.
        first = Stub("first", np.array([0.5]), success=False)
        second = Stub("second", np.array([4.0]))
        outcome = WithFallback(first, second).minimize(
            parabola, np.array([5.0]), gradient=None, bounds=None
        )
        assert outcome.optimizer == "first"
        assert "reached a worse point" in outcome.notes[0]

    def test_taking_no_step_counts_as_faltering(self):
        first = Stub("first", np.array([5.0]), success=True, n_iterations=0)
        second = Stub("second", np.array([0.0]))
        outcome = WithFallback(first, second).minimize(
            parabola, np.array([5.0]), gradient=None, bounds=None
        )
        assert outcome.optimizer == "second"
        assert "converged without taking a step" in outcome.notes[0]

    def test_the_retry_is_derivative_free(self):
        # A quasi-Newton method stalls on a surface that is not smooth; handing
        # the same gradient to the retry would reproduce the stall.
        def gradient(vec: NDArray[np.float64]) -> NDArray[np.float64]:
            return 2.0 * np.asarray(vec, dtype=np.float64)

        first = Stub("first", np.array([4.0]), success=False)
        second = Stub("second", np.array([0.0]))
        WithFallback(first, second).minimize(
            parabola, np.array([5.0]), gradient=gradient, bounds=None
        )
        assert first.seen_gradient == [gradient]
        assert second.seen_gradient == [None]


class TestWhatTheCallerNamed:
    def test_a_method_name_becomes_a_scipy_method(self):
        resolved = optimizer_for("Powell", {"tol": 1e-9})
        assert isinstance(resolved, ScipyMethod)
        assert (resolved.method, resolved.options) == ("Powell", {"tol": 1e-9})

    def test_a_callable_becomes_a_custom_solver(self):
        def solver(fun, x0, **kwargs):  # pragma: no cover - never run here
            raise NotImplementedError

        resolved = optimizer_for(solver, {})
        assert isinstance(resolved, CustomSolver)
        assert resolved.name == "solver"

    def test_a_ready_made_optimizer_is_passed_through(self):
        stub = Stub("stub", np.zeros(1))
        assert optimizer_for(stub, {}) is stub


class TestWhatMLEMeansByIt:
    def test_naming_nothing_means_the_fallback_policy(self):
        search = MLE().search
        assert isinstance(search, WithFallback)
        assert (search.first.name, search.then.name) == ("L-BFGS-B", "Nelder-Mead")
        # The policy names both: which one actually ran is answered by the
        # outcome, since that is the only thing that knows.
        assert search.name == "L-BFGS-B or Nelder-Mead"

    def test_naming_one_replaces_the_whole_policy(self):
        # Including the fallback: the request is honoured, not second-guessed.
        assert MLE(optimizer="Powell").search == ScipyMethod("Powell", {})

    def test_an_optimizer_object_can_be_named(self, normal_family, rng):
        result = normal_family.fit(
            rng.normal(0.0, 1.0, 100), estimator=MLE(optimizer=ScipyMethod("Nelder-Mead"))
        )
        assert result.optimizer == "Nelder-Mead"
        assert result.route == "numeric"

    def test_options_beside_an_optimizer_object_are_refused(self):
        # They would go nowhere: an optimizer carries its own.
        with pytest.raises(ValueError, match="would be ignored"):
            MLE(optimizer=ScipyMethod("Powell"), options={"tol": 1e-9})


class TestTheBox:
    def test_it_reports_whether_the_family_declared_anything(self, normal_family, uniform_family):
        assert ParameterBox.of(normal_family).declared is True
        assert ParameterBox.of(uniform_family).declared is False

    def test_an_undeclared_box_still_has_one_entry_per_parameter(self, uniform_family):
        box = ParameterBox.of(uniform_family)
        assert box.bounds == ((-np.inf, np.inf), (-np.inf, np.inf))

    def test_only_asking_for_the_scipy_form_warns(self, uniform_family):
        box = ParameterBox.of(uniform_family)
        box.clip(uniform_family.base(lower_bound=0.0, upper_bound=1.0))
        with pytest.warns(UserWarning, match="declares no 'param_bounds'"):
            assert box.as_scipy() is None


def count_splits(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every walk of the sample, so a repeated one is visible."""
    seen: list[int] = []
    original = LogLikelihood._split

    def counting(self: LogLikelihood, params: Parametrization) -> Evaluation:
        seen.append(1)
        return original(self, params)

    monkeypatch.setattr(LogLikelihood, "_split", counting)
    return seen


class TestTheSharedEvaluation:
    def test_the_gradient_reuses_the_value_s_split(self, gamma_family, rng, monkeypatch):
        # The measured waste this object was introduced to remove: value and
        # gradient are asked at the same point, one after the other.
        objective = LogLikelihood.of(gamma_family, rng.gamma(2.0, 1.0, 200))
        splits = count_splits(monkeypatch)
        vec = np.array([2.0, 1.0])
        objective(vec)
        objective.gradient(vec)
        assert len(splits) == 1

    def test_a_different_point_is_evaluated_again(self, gamma_family, rng, monkeypatch):
        objective = LogLikelihood.of(gamma_family, rng.gamma(2.0, 1.0, 200))
        splits = count_splits(monkeypatch)
        objective(np.array([2.0, 1.0]))
        objective(np.array([3.0, 1.0]))
        assert len(splits) == 2

    def test_two_samples_do_not_share_a_memo(self, normal_family):
        # The memo is per instance, by 'default_factory'. Were it a plain
        # default the whole class would share one, and the second sample would
        # be answered with the first sample's numbers at the same parameters.
        first = LogLikelihood.of(normal_family, np.array([1.0, 2.0, 3.0]))
        second = LogLikelihood.of(normal_family, np.array([100.0, 200.0, 300.0]))
        vec = np.array([0.0, 1.0])
        assert first(vec) != pytest.approx(second(vec))
        # And neither does the order of evaluation decide the answers.
        third = LogLikelihood.of(normal_family, np.array([100.0, 200.0, 300.0]))
        fourth = LogLikelihood.of(normal_family, np.array([1.0, 2.0, 3.0]))
        assert (fourth(vec), third(vec)) == pytest.approx((first(vec), second(vec)))

    def test_one_object_is_bound_to_one_sample(self, normal_family):
        # 'sample' is a frozen field, so an instance cannot be pointed at other
        # data — which is what makes the parameter vector a complete memo key.
        objective = LogLikelihood.of(normal_family, np.array([1.0, 2.0, 3.0]))
        with pytest.raises(dataclasses.FrozenInstanceError):
            objective.sample = np.array([4.0, 5.0, 6.0])  # type: ignore[misc]

    def test_a_family_without_score_refuses_to_invent_a_gradient(self, scoreless_family, rng):
        # Zeros would tell the optimizer it had reached a stationary point.
        objective = LogLikelihood.of(scoreless_family, rng.normal(0.0, 1.0, 50))
        assert objective.has_gradient is False
        assert objective.gradient_or_none is None
        with pytest.raises(Exception, match="provides no 'score'"):
            objective.gradient(np.array([0.0, 1.0]))
