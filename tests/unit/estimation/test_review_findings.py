"""
Regressions for the defects the second review found, and the gaps it measured.

Three groups, and they are here together because they came from one pass over
the module rather than because they test one thing:

* the two fallback/success blockers, whose fixes live in ``mle.py``.  Both
  assertions fail on the code as it stood: the first on 3 of the 6 shared
  seeds and on about half of a wider 20-seed draw, the second on every
  seed;
* the two mutants that survived a 32-defect mutation run.  Neither is
  reachable through a built-in family, which is why they went unnoticed and
  why both tests have to build a family by hand;
* three branches that were reachable but never taken, because no built-in
  family has the shape that reaches them.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.estimation import make_objective, moments, resolve_bounds
from pysatl_core.families.parametric_family import ParametricFamily
from pysatl_core.families.parametrizations import (
    Parametrization,
    constraint,
    parametrization,
)
from pysatl_core.types import CharacteristicName, NumericArray, UnivariateContinuous


class TestOverflowingLogDensity:
    """A sum of finite log-densities can still overflow; the objective must stay usable.

    Without the ``np.isfinite(total)`` guard in ``make_objective`` this returns
    ``-inf``, and every optimizer reads that as the global minimum: the search
    stops at a point the family cannot justify.  The values are absurd on
    purpose — the guard exists for a family whose ``lpdf`` misbehaves, and no
    built-in family can reach it.
    """

    def test_an_overflowing_sum_is_reported_as_inf_not_minus_inf(self):
        family = ParametricFamily(
            name="HugeDensity",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["h"],
            distr_characteristics={
                CharacteristicName.LPDF: lambda p, x: np.full_like(x, 1e308 * cast(Any, p).scale)
            },
        )

        @parametrization(family=family, name="h")
        class _H(Parametrization):
            scale: float

            @constraint(description="scale > 0")
            def check(self) -> bool:
                return self.scale > 0

        objective, _gradient = make_objective(family, np.array([1.0, 2.0, 3.0]))
        assert objective(np.array([1.0])) == np.inf


class TestClosedFormIsNotUsedInForeignCoordinates:
    """A closed form is written against the base parametrization's coordinates.

    A view fixed in another parametrization may expose a field of the *same
    name* meaning something else.  Handing the formula's output to it would
    report a confidently wrong estimate — here, one that is exactly twice the
    right answer — so the fit has to take the numerical path instead.
    """

    @staticmethod
    def _family() -> ParametricFamily:
        def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
            base = cast(Any, parameters.transform_to_base_parametrization())
            return cast(
                NumericArray,
                -0.5 * np.log(2 * np.pi)
                - np.log(base.sigma)
                - ((x - base.mu) ** 2) / (2 * base.sigma**2),
            )

        def mle(sample: NumericArray, fixed: Mapping[str, float]) -> Parametrization | None:
            mu = float(fixed["mu"]) if "mu" in fixed else float(np.mean(sample))
            return _Base(mu=mu, sigma=float(np.sqrt(np.mean((sample - mu) ** 2))))

        family = ParametricFamily(
            name="CollidingNames",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["base", "halved"],
            distr_characteristics={CharacteristicName.LPDF: lpdf},
            mle=mle,
        )

        @parametrization(family=family, name="base")
        class _Base(Parametrization):
            mu: float
            sigma: float

            @constraint(description="sigma > 0")
            def check(self) -> bool:
                return self.sigma > 0

        @parametrization(family=family, name="halved")
        class _Halved(Parametrization):
            """Same field names as the base one, but 'sigma' is *half* the sd."""

            mu: float
            sigma: float

            @constraint(description="sigma > 0")
            def check(self) -> bool:
                return self.sigma > 0

            def transform_to_base_parametrization(self) -> Parametrization:
                return _Base(mu=self.mu, sigma=2.0 * self.sigma)

        return family

    def test_a_view_in_foreign_coordinates_takes_the_numerical_path(self):
        family = self._family()
        sample = np.random.default_rng(0).normal(0.0, 2.0, 500)
        view = family.view(parametrization_name="halved", mu=0.0)
        with warnings.catch_warnings():
            # This family declares no bounds; that warning is not the subject here.
            warnings.simplefilter("ignore", UserWarning)
            result = view.fit(sample)
        assert result.method == "numeric"
        assert cast(Any, result.params).sigma == pytest.approx(
            float(np.sqrt(np.mean(sample**2))) / 2.0, rel=1e-4
        )


class TestGradientWhereNothingIsExplained:
    """The optimizer does visit parameters that explain no observation at all.

    The objective there is a wall of pure penalty; the gradient still has to be
    a finite vector for the line search to back out of it.
    """

    def test_a_point_explaining_no_observation_yields_a_zero_gradient(self, uniform_family, rng):
        sample = rng.uniform(0.0, 1.0, 50)
        objective, gradient = make_objective(uniform_family, sample)
        assert gradient is not None
        elsewhere = np.array([10.0, 20.0])
        assert np.isfinite(objective(elsewhere))
        np.testing.assert_array_equal(gradient(elsewhere), np.zeros(2))


class TestClosedFormDeclining:
    """A closed-form rule may decline a particular sample by returning ``None``.

    No built-in family does, so the fall-through to the numerical path is
    otherwise never exercised.
    """

    def test_a_formula_returning_none_falls_through_to_the_numerical_path(self):
        def lpdf(parameters: Parametrization, x: NumericArray) -> NumericArray:
            params = cast(Any, parameters)
            return cast(NumericArray, -np.log(params.scale) - x / params.scale)

        def declining_mle(
            sample: NumericArray, fixed: Mapping[str, float]
        ) -> Parametrization | None:
            return None

        family = ParametricFamily(
            name="DecliningFormula",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["s"],
            distr_characteristics={CharacteristicName.LPDF: lpdf},
            mle=declining_mle,
            param_bounds={"scale": (0, None)},
        )

        @parametrization(family=family, name="s")
        class _S(Parametrization):
            scale: float

            @constraint(description="scale > 0")
            def check(self) -> bool:
                return self.scale > 0

        sample = np.random.default_rng(0).exponential(2.0, 200)
        result = family.fit(sample)
        assert result.method == "numeric"
        assert cast(Any, result.params).scale == pytest.approx(float(sample.mean()), rel=1e-3)


class TestUpperBoundOnly:
    """A family bounded from above only: no built-in family declares one."""

    def test_the_upper_edge_is_nudged_inwards_and_the_lower_stays_infinite(self):
        family = ParametricFamily(
            name="UpperBoundOnly",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["p"],
            distr_characteristics={},
            param_bounds={"a": (None, 5.0)},
        )

        @parametrization(family=family, name="p")
        class _P(Parametrization):
            a: float

            @constraint(description="a < 5")
            def check(self) -> bool:
                return self.a < 5.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            bounds = resolve_bounds(family)
        assert bounds is not None
        low, high = bounds[0]
        assert low == -np.inf
        assert high < 5.0, "'a < 5' is open, so the edge must be nudged inwards"
        cast(Any, family.base)(a=high).validate()


class TestAMisspelledMomentRuleIsNotSilent:
    """A rule that returns the wrong names degrades to a start of ones.

    That silent degradation was the one way to get a moment rule wrong that
    nothing diagnosed — and ``register_moment_start`` exists precisely for
    user-written rules, where a misspelled name is the likeliest mistake.
    ``_probe_params`` already announces a rule that *raises*; this is the same
    courtesy for a rule that merely misses.
    """

    def test_a_typo_in_a_parameter_name_is_reported(self, gamma_family, fixed_rng, monkeypatch):
        monkeypatch.setitem(
            moments._MOMENT_STARTS,
            gamma_family.name,
            lambda s: {"kk": 3.0, "theta": 2.0},  # 'kk' is a typo for 'k'
        )
        sample = fixed_rng.gamma(3.0, 2.0, 200)
        with pytest.warns(UserWarning, match="does not cover its free parameters"):
            start = moments.starting_point(gamma_family, sample)
        # The fit still proceeds, from the fallback start.
        assert start.parameters == {"k": 1.0, "theta": 1.0}

    def test_a_rule_in_another_parametrization_is_left_alone(
        self, normal_family, fixed_rng, monkeypatch
    ):
        """The legitimate near-miss, and the reason the check is an intersection.

        A view fixed in a foreign parametrization exposes free parameters that
        the base-parametrization rule knows nothing about.  The rule then
        covers *none* of them — no shared name at all — which is what tells it
        apart from a typo, where some names match and some do not.  Warning
        here would cry wolf on the supported case.
        """
        view = normal_family.view(parametrization_name="meanVar", mu=0.0)
        sample = fixed_rng.normal(0.0, 1.0, 200)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            moments.starting_point(view, sample)
        assert [str(w.message) for w in caught if "does not cover" in str(w.message)] == []


class TestWarningsPointAtTheCaller:
    """A warning attributed to the library is a warning the caller cannot act on.

    ``simplefilter("error")``, filtering by module and ``pytest.warns`` all key
    off the frame the warning is charged to.  At the default ``stacklevel`` the
    two warnings raised below stop inside ``pysatl_core``, so a caller could
    neither locate the offending ``fit`` nor turn just these into errors.
    """

    def test_the_no_bounds_warning_is_charged_to_the_fit_call(self, boxed_family, fixed_rng):
        sample = fixed_rng.uniform(2.0, 5.0, 200)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            boxed_family.fit(sample)
        bounds_warnings = [w for w in caught if "declares no 'param_bounds'" in str(w.message)]
        assert bounds_warnings, "the fixture family declares no bounds"
        assert bounds_warnings[0].filename == __file__

    def test_the_failed_rule_warning_is_charged_to_the_fit_call(
        self, normal_family, fixed_rng, monkeypatch
    ):
        def raises(sample):
            raise ArithmeticError("no moments here")

        monkeypatch.setitem(moments._MOMENT_STARTS, normal_family.name, raises)
        sample = fixed_rng.normal(0.0, 1.0, 200)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normal_family.fit(sample)
        rule_warnings = [w for w in caught if "starting rule for family" in str(w.message)]
        assert rule_warnings, "the patched rule always raises"
        assert rule_warnings[0].filename == __file__


# --------------------------------------------------------------------------
# The two blockers.  Both of these fail on the code as it stood before the
# fixes in 'mle.py' that accompany them.
# --------------------------------------------------------------------------


class TestFallbackKeepsTheBetterEstimate:
    """The retry is a second opinion, not a verdict.

    Both fits start at the same point with the same iteration cap; the only
    difference is that the first is allowed to fall back to Nelder-Mead.
    Adopting that retry unseen loses likelihood on a large share of samples.
    """

    def test_the_fallback_never_reports_a_worse_estimate(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 500)
        with_fallback = gamma_family.fit(sample, options={"maxiter": 2})
        without_fallback = gamma_family.fit(sample, optimizer="L-BFGS-B", options={"maxiter": 2})
        assert with_fallback.log_likelihood >= without_fallback.log_likelihood


class TestZeroLikelihoodIsNotSuccess:
    """An estimate under which the data are impossible is not a converged fit.

    The fixed ``lower_bound`` contradicts the sample, so every observation
    below it is unexplained: the likelihood is zero and ``aic``/``bic`` are
    ``inf``.  Reporting ``success=True`` there tells the caller the opposite of
    what happened.
    """

    def test_a_fit_that_explains_nothing_does_not_claim_success(self, uniform_family, rng):
        sample = rng.uniform(2.0, 5.0, 200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = uniform_family.view(lower_bound=3.0).fit(sample, optimizer="L-BFGS-B")
        assert result.log_likelihood == -np.inf
        assert result.success is False
