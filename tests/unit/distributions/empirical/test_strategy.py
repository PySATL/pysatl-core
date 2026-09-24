"""
Unit tests for EmpiricalComputationStrategy and the tabulated CDF -> PPF edge.

Covers:
  - estimator-identity-aware cache invalidation
  - degenerate behaviour for distributions that expose no estimator
  - inheritance contract with DefaultComputationStrategy
  - routing of PPF through the tabulated graph edge, and the agreement
    between explain_computation_path and query_method it buys
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions.computations.computation import FittedComputationMethod
from pysatl_core.distributions.empirical import (
    EmpiricalComputationStrategy,
    EmpiricalDistribution,
    ScipyGaussianKde,
)
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.types import CharacteristicName

TABULATED_FITTER = "cdf_to_ppf_tabulated_1C"
BISECTION_FITTER = "cdf_to_ppf_1C"


class _FakeDistr:
    """Minimal stand-in exposing only the hook the strategy actually reads."""

    def __init__(self, estimator: object | None) -> None:
        self.estimator = estimator


# Shape of DefaultComputationStrategy._cache keys:
# (distr_id, edge_id, target, frozen_options).  Spelled out here so the tests
# can plant and look up a marker entry without weakening the cache's typing.
type _CacheKey = tuple[int, int, str, frozenset[tuple[str, Hashable]]]


def _put_dummy_cache_entry(strategy: EmpiricalComputationStrategy, tag: int = 0) -> _CacheKey:
    """Plant a recognisable entry in the fitted-method cache, return its key."""
    key: _CacheKey = (tag, tag, CharacteristicName.PPF, frozenset())
    strategy._cache[key] = FittedComputationMethod[Any, Any](
        target=CharacteristicName.PPF,
        sources=(),
        func=lambda *a, **kw: None,
    )
    return key


def _make_distr(sample_size: int = 300, seed: int = 0) -> EmpiricalDistribution:
    rng = np.random.default_rng(seed)
    return EmpiricalDistribution(rng.normal(0.0, 1.0, sample_size))


class TestInheritance:
    def test_is_default_strategy_subclass(self) -> None:
        assert issubclass(EmpiricalComputationStrategy, DefaultComputationStrategy)
        assert isinstance(EmpiricalComputationStrategy(), DefaultComputationStrategy)

    def test_caching_enabled_by_default(self) -> None:
        assert EmpiricalComputationStrategy().is_caching_enabled is True

    def test_caching_can_be_disabled(self) -> None:
        assert EmpiricalComputationStrategy(enable_caching=False).is_caching_enabled is False

    def test_accepts_computation_defaults_like_the_base_strategy(self) -> None:
        strategy = EmpiricalComputationStrategy(computation_defaults={"grid_size": 33})
        assert strategy._computation_defaults == {"grid_size": 33}

    def test_computation_defaults_reach_the_fitter(self) -> None:
        distr = EmpiricalDistribution(
            _make_distr().data,
            computation_strategy=EmpiricalComputationStrategy(
                computation_defaults={"grid_size": 5},
            ),
        )
        coarse = distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.3]))
        fine = _make_distr().calculate_characteristic(CharacteristicName.PPF, np.array([0.3]))
        assert not np.isclose(coarse[0], fine[0])


class TestMaybeInvalidate:
    def test_first_query_records_estimator_without_clearing(self) -> None:
        strategy = EmpiricalComputationStrategy()
        estimator = object()

        strategy._maybe_invalidate(cast(Any, _FakeDistr(estimator)))

        assert strategy._tracked_estimator is estimator
        assert strategy._cache == {}

    def test_same_estimator_keeps_cache(self) -> None:
        strategy = EmpiricalComputationStrategy()
        distr = cast(Any, _FakeDistr(object()))

        strategy._maybe_invalidate(distr)
        key = _put_dummy_cache_entry(strategy)
        strategy._maybe_invalidate(distr)

        assert key in strategy._cache

    def test_swapped_estimator_clears_cache(self) -> None:
        strategy = EmpiricalComputationStrategy()
        second = object()

        strategy._maybe_invalidate(cast(Any, _FakeDistr(object())))
        _put_dummy_cache_entry(strategy)

        strategy._maybe_invalidate(cast(Any, _FakeDistr(second)))

        assert strategy._cache == {}
        assert strategy._tracked_estimator is second

    def test_distribution_without_estimator_property_does_not_crash(self) -> None:
        strategy = EmpiricalComputationStrategy()

        class _Bare:
            pass

        strategy._maybe_invalidate(cast(Any, _Bare()))
        assert strategy._tracked_estimator is None

        key = _put_dummy_cache_entry(strategy)
        strategy._maybe_invalidate(cast(Any, _Bare()))
        assert key in strategy._cache

    def test_explicit_invalidate_resets_everything(self) -> None:
        strategy = EmpiricalComputationStrategy()
        strategy._maybe_invalidate(cast(Any, _FakeDistr(object())))
        _put_dummy_cache_entry(strategy)

        strategy.invalidate()

        assert strategy._cache == {}
        assert strategy._tracked_estimator is None


class TestGraphRouting:
    """The tabulated PPF must be a normal graph edge, not a strategy special case."""

    def test_ppf_plan_uses_the_tabulated_edge(self) -> None:
        plan = _make_distr().explain_computation_path(CharacteristicName.PPF)
        assert plan.steps[-1].method_name == TABULATED_FITTER
        assert plan.steps[-1].edge_kind == "computation"

    def test_ppf_plan_starts_from_the_analytical_cdf(self) -> None:
        """
        Pins the key order of ``_build_analytical_computations``.

        The resolver picks the first source in insertion order rather than the
        shortest path, so listing PDF first would silently route PPF through
        ``pdf -> cdf`` numerical quadrature instead of the exact CDF the
        estimator already provides -- same numbers, ~40x the fit cost.
        """
        plan = _make_distr().explain_computation_path(CharacteristicName.PPF)
        assert plan.source == CharacteristicName.CDF
        assert len(plan.steps) == 1
        assert plan.steps[0].sources == (CharacteristicName.CDF,)

    def test_analytical_cdf_is_not_rebuilt_by_quadrature(self) -> None:
        """The exact estimator CDF must be what the PPF fitter consumes."""
        distr = _make_distr()
        calls = {"n": 0}
        estimator = distr.estimator

        class _Counting:
            def pdf(self, x: Any) -> Any:
                return estimator.pdf(x)

            def cdf(self, x: Any) -> Any:
                calls["n"] += 1
                return estimator.cdf(x)

        # Written through the private slot on purpose: set_method() would also
        # fire the sampler/cache reset, and this test is about what the PPF
        # fitter consumes, not about invalidation.
        distr._estimator = _Counting()
        distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.5]))

        assert calls["n"] > 0

    def test_plan_advertises_the_options_the_fitter_really_has(self) -> None:
        plan = _make_distr().explain_computation_path(CharacteristicName.PPF)
        assert set(plan.required_options()) >= {"tail_margin", "grid_size", "max_iter", "x_tol"}
        # Options belonging to the bisection fitter must no longer be advertised.
        assert "eps" not in plan.required_options()
        assert "x0" not in plan.required_options()

    def test_query_method_executes_the_edge_the_plan_names(self) -> None:
        distr = _make_distr()
        plan = distr.explain_computation_path(CharacteristicName.PPF)
        method = cast(FittedComputationMethod[Any, Any], distr.query_method(CharacteristicName.PPF))
        assert method.target == plan.target
        assert tuple(method.sources) == plan.steps[-1].sources

    def test_distribution_without_tabulation_domain_uses_bisection(self) -> None:
        from pysatl_core.distributions.computations.computation import AnalyticalComputation
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.types import UnivariateContinuous

        class _Plain(Distribution):
            def __init__(self) -> None:
                super().__init__(
                    UnivariateContinuous,
                    {
                        CharacteristicName.PDF: AnalyticalComputation(
                            target=CharacteristicName.PDF,
                            func=lambda x, /, **o: (
                                np.exp(-0.5 * np.asarray(x) ** 2) / np.sqrt(2 * np.pi)
                            ),
                        )
                    },
                )

            def _clone_with_strategies(self, **kwargs: Any) -> Distribution:  # pragma: no cover
                raise NotImplementedError

        plan = _Plain().explain_computation_path(CharacteristicName.PPF)
        assert plan.steps[-1].method_name == BISECTION_FITTER

    def test_none_domain_falls_back_to_bisection(self) -> None:
        distr = _make_distr()
        distr._tabulation_domain = None
        cast(DefaultComputationStrategy, distr.computation_strategy)._path_cache.clear()
        plan = distr.explain_computation_path(CharacteristicName.PPF)
        assert plan.steps[-1].method_name == BISECTION_FITTER


class TestOptionsAreHonoured:
    """Regression guard: options declared by the plan must reach the fitter."""

    # The PPF plan is a single ``cdf -> ppf`` step, so the tabulated fitter's
    # options live on step 0.  See test_ppf_plan_starts_from_the_analytical_cdf.
    PPF_STEP = 0

    def test_grid_size_changes_the_result(self) -> None:
        distr = _make_distr()
        plan = distr.explain_computation_path(CharacteristicName.PPF)
        q = np.array([0.3])

        default = distr.calculate_characteristic(CharacteristicName.PPF, q)
        coarse = distr.calculate_characteristic(
            CharacteristicName.PPF, q, plan.with_options(self.PPF_STEP, grid_size=5)
        )
        assert not np.isclose(default[0], coarse[0])

    def test_tail_margin_changes_the_result(self) -> None:
        distr = _make_distr()
        plan = distr.explain_computation_path(CharacteristicName.PPF)
        q = np.array([0.3])

        default = distr.calculate_characteristic(CharacteristicName.PPF, q)
        padded = distr.calculate_characteristic(
            CharacteristicName.PPF, q, plan.with_options(self.PPF_STEP, tail_margin=3.0)
        )
        assert not np.isclose(default[0], padded[0], atol=1e-12, rtol=0.0)

    def test_invalid_option_value_is_rejected(self) -> None:
        plan = _make_distr().explain_computation_path(CharacteristicName.PPF)
        with pytest.raises(ValueError):
            plan.with_options(self.PPF_STEP, grid_size=1)


class TestTabulatedPpfNumerics:
    def test_array_input_returns_array(self) -> None:
        result = _make_distr().calculate_characteristic(
            CharacteristicName.PPF, np.linspace(0.05, 0.95, 50)
        )
        assert result.shape == (50,)

    def test_scalar_input_returns_array_like_the_builtin_fitter(self) -> None:
        result = _make_distr().calculate_characteristic(CharacteristicName.PPF, 0.5)
        assert np.shape(result) == (1,)

    def test_result_is_monotonic_nondecreasing(self) -> None:
        result = _make_distr().calculate_characteristic(
            CharacteristicName.PPF, np.linspace(0.01, 0.99, 200)
        )
        assert np.all(np.diff(result) >= -1e-10)

    def test_matches_root_finding_within_tolerance(self) -> None:
        rng = np.random.default_rng(7)
        sample = rng.normal(0.0, 1.0, 400)

        fast = EmpiricalDistribution(sample)
        slow = EmpiricalDistribution(sample, computation_strategy=DefaultComputationStrategy())
        quantiles = np.linspace(0.1, 0.9, 9)

        fast_vals = fast.calculate_characteristic(CharacteristicName.PPF, quantiles)
        slow_vals = np.array(
            [
                float(np.squeeze(slow.calculate_characteristic(CharacteristicName.PPF, float(q))))
                for q in quantiles
            ]
        )
        assert np.allclose(fast_vals, slow_vals, atol=5e-2)

    def test_extreme_quantiles_are_solved_not_clipped(self) -> None:
        """Tails outside the tabulated range must round-trip through the CDF."""
        distr = _make_distr(sample_size=400)
        q = np.array([1e-9, 1e-6, 1e-3, 0.5, 1 - 1e-3, 1 - 1e-6])

        x = distr.calculate_characteristic(CharacteristicName.PPF, q)
        back = distr.calculate_characteristic(CharacteristicName.CDF, x)

        assert not np.any(np.isnan(x))
        # Distinct quantiles must map to distinct points -- the old
        # implementation collapsed the whole tail onto the grid endpoint.
        assert np.all(np.diff(x) > 0)
        assert np.allclose(back, q, atol=1e-8)

    def test_out_of_range_quantiles_are_infinite(self) -> None:
        result = _make_distr().calculate_characteristic(
            CharacteristicName.PPF, np.array([0.0, 1.0, -0.5, 2.0])
        )
        assert np.array_equal(result, np.array([-np.inf, np.inf, -np.inf, np.inf]))

    def test_ppf_never_leaves_a_bounded_support(self) -> None:
        """
        The out-of-range solver must not search past a declared bound.

        Exercised through a synthetic distribution rather than
        ``EmpiricalDistribution``, which refuses bounded supports outright:
        the clamping lives in the shared ``cdf_to_ppf_tabulated_1C`` fitter and
        applies to any distribution that declares both a tabulation domain and
        a finite bound.  The CDF below deliberately leaks below zero, the way a
        Gaussian kernel on non-negative data would.
        """
        from scipy.special import ndtr

        from pysatl_core.distributions.computations.computation import AnalyticalComputation
        from pysatl_core.distributions.distribution import Distribution
        from pysatl_core.distributions.support import ContinuousSupport
        from pysatl_core.types import UnivariateContinuous

        class _LeakyBounded(Distribution):
            tabulation_domain = (0.0, 5.0)

            def __init__(self) -> None:
                super().__init__(
                    UnivariateContinuous,
                    {
                        CharacteristicName.CDF: AnalyticalComputation(
                            target=CharacteristicName.CDF,
                            func=lambda x, /, **o: ndtr(np.asarray(x, dtype=float) - 1.0),
                        )
                    },
                    support=ContinuousSupport(left=0.0, right=np.inf),
                )

            def _clone_with_strategies(self, **kwargs: Any) -> Distribution:  # pragma: no cover
                raise NotImplementedError

        distr = _LeakyBounded()
        assert distr.explain_computation_path(CharacteristicName.PPF).steps[-1].method_name == (
            TABULATED_FITTER
        )

        # cdf(0) = ndtr(-1) ~ 0.159, so every quantile below that has no
        # solution inside the support and must land on the boundary, never
        # below it.
        result = distr.calculate_characteristic(
            CharacteristicName.PPF, np.array([1e-9, 1e-4, 0.05, 0.5])
        )
        assert np.all(result >= 0.0)

    def test_ppf_recomputed_after_estimator_swap(self) -> None:
        rng = np.random.default_rng(3)
        distr = EmpiricalDistribution(rng.normal(0.0, 1.0, 300))

        wide = distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.9]))
        distr.set_method(ScipyGaussianKde(bandwidth=0.05))
        narrow = distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.9]))

        assert np.isfinite(wide[0]) and np.isfinite(narrow[0])
        assert not np.isclose(wide[0], narrow[0])


class TestQueryMethodIntegration:
    def test_estimator_swap_clears_fitted_cache_via_query_method(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(0, 1, 200)
        strategy = EmpiricalComputationStrategy()
        distr = EmpiricalDistribution(sample, computation_strategy=strategy)

        strategy.query_method(CharacteristicName.PDF, distr)
        key = _put_dummy_cache_entry(strategy)

        # Deliberately bypasses set_method(): that would clear the cache via the
        # explicit invalidate() hook, so the estimator-identity tracking this
        # test exists for would never be exercised.
        new_estimator = ScipyGaussianKde(bandwidth="silverman").fit(sample)
        distr._estimator = new_estimator

        strategy.query_method(CharacteristicName.PDF, distr)

        assert key not in strategy._cache
        assert strategy._tracked_estimator is distr.estimator

    def test_explain_and_query_stay_in_sync_after_swap(self) -> None:
        distr = _make_distr()

        before = distr.explain_computation_path(CharacteristicName.PPF)
        distr.set_method(ScipyGaussianKde(bandwidth="silverman"))
        after = distr.explain_computation_path(CharacteristicName.PPF)

        assert before.steps[-1].method_name == after.steps[-1].method_name == TABULATED_FITTER
