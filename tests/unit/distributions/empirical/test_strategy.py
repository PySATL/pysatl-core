"""
Unit tests for empirical PPF computation through the characteristic graph.

Covers:
  - routing of PPF through the CDF bisection edge
  - agreement between explain_computation_path and query_method
  - numerical correctness and cache invalidation after estimator changes
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest
from scipy.optimize import brentq

from pysatl_core.distributions.computations.computation import (
    AnalyticalComputation,
    FittedComputationMethod,
)
from pysatl_core.distributions.empirical import (
    EmpiricalDistribution,
    ScipyGaussianKde,
)
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.types import CharacteristicName

BISECTION_FITTER = "cdf_to_ppf_1C"


def _cdf_func(distr: EmpiricalDistribution) -> Any:
    return next(iter(distr.analytical_computations[CharacteristicName.CDF].values())).func


def _replace_cdf(distr: EmpiricalDistribution, func: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    computations = {name: dict(methods) for name, methods in distr.analytical_computations.items()}
    label = next(iter(computations[CharacteristicName.CDF]))
    computations[CharacteristicName.CDF][label] = AnalyticalComputation(
        CharacteristicName.CDF, func
    )
    monkeypatch.setattr(distr, "_analytical_computations", computations)
    # This test helper bypasses set_estimator(), so it must also reset the
    # conversion cache that set_estimator() would normally invalidate.
    cast(DefaultComputationStrategy, distr.computation_strategy).invalidate()


def _make_distr(sample_size: int = 300, seed: int = 0) -> EmpiricalDistribution:
    rng = np.random.default_rng(seed)
    return EmpiricalDistribution(rng.normal(0.0, 1.0, sample_size))


class TestGraphRouting:
    """Empirical PPF uses the graph edge that inverts the analytical CDF."""

    def test_ppf_plan_uses_bisection(self) -> None:
        plan = _make_distr().explain_computation_path(CharacteristicName.PPF)
        assert plan.steps[-1].method_name == BISECTION_FITTER
        assert plan.steps[-1].edge_kind == "computation"

    def test_ppf_plan_starts_from_the_analytical_cdf(self) -> None:
        """
        Pins the key order of the KDE estimator's fitted computations.

        The resolver picks the first source in insertion order rather than the
        shortest path, so listing PDF first would silently route PPF through
        ``pdf -> cdf`` numerical quadrature instead of the exact CDF the
        estimator already provides.
        """
        plan = _make_distr().explain_computation_path(CharacteristicName.PPF)
        assert plan.source == CharacteristicName.CDF
        assert len(plan.steps) == 1
        assert plan.steps[0].sources == (CharacteristicName.CDF,)

    def test_analytical_cdf_is_not_rebuilt_by_quadrature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact fitted CDF must be what the PPF fitter consumes."""
        distr = _make_distr()
        calls = {"n": 0}
        original_cdf = _cdf_func(distr)

        def recording_cdf(x: Any) -> Any:
            calls["n"] += 1
            return original_cdf(x)

        _replace_cdf(distr, recording_cdf, monkeypatch)
        distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.5]))

        assert calls["n"] > 0

    def test_plan_advertises_the_options_the_fitter_really_has(self) -> None:
        plan = _make_distr().explain_computation_path(CharacteristicName.PPF)
        assert set(plan.required_options()) == {"eps", "x0", "max_iter", "x_tol"}

    def test_query_method_executes_the_edge_the_plan_names(self) -> None:
        distr = _make_distr()
        plan = distr.explain_computation_path(CharacteristicName.PPF)
        method = cast(FittedComputationMethod[Any, Any], distr.query_method(CharacteristicName.PPF))
        assert method.target == plan.target
        assert tuple(method.sources) == plan.steps[-1].sources

    def test_pdf_only_distribution_uses_bisection(self) -> None:
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


class TestPpfNumerics:
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

    @pytest.mark.parametrize("multimodal", [False, True], ids=["unimodal", "multimodal"])
    def test_matches_independent_root_finding_within_tolerance(self, multimodal: bool) -> None:
        rng = np.random.default_rng(7)
        if multimodal:
            left_mode = rng.normal(-3.0, 0.2, 200)
            sample = np.concatenate((left_mode, -left_mode))
            estimator = ScipyGaussianKde(bandwidth=0.15)
            # Symmetric, separated modes make the CDF nearly flat around 0.5.
            quantiles = np.array(
                [0.1, 0.49, 0.4999, 0.49999, 0.499999, 0.500001, 0.50001, 0.5001, 0.51, 0.9]
            )
        else:
            sample = rng.normal(0.0, 1.0, 400)
            estimator = ScipyGaussianKde()
            quantiles = np.linspace(0.1, 0.9, 9)

        distr = EmpiricalDistribution(sample, estimator=estimator)
        assert (
            distr.explain_computation_path(CharacteristicName.PPF).steps[-1].method_name
            == BISECTION_FITTER
        )
        margin = 8.0 * float(sample.std())
        lo, hi = float(sample.min()) - margin, float(sample.max()) + margin

        # Invert the analytical CDF with an independent root-finding algorithm.
        cdf = _cdf_func(distr)

        def cdf_minus_q(x: float, q: float) -> float:
            return float(cdf(np.array([x]))[0]) - q

        reference = np.array(
            [brentq(cdf_minus_q, lo, hi, args=(float(q),), xtol=1e-12) for q in quantiles]
        )
        result = distr.calculate_characteristic(CharacteristicName.PPF, quantiles)
        np.testing.assert_allclose(result, reference, atol=1e-3, rtol=0.0)

    def test_out_of_range_quantiles_are_infinite(self) -> None:
        result = _make_distr().calculate_characteristic(
            CharacteristicName.PPF, np.array([0.0, 1.0, -0.5, 2.0])
        )
        assert np.array_equal(result, np.array([-np.inf, np.inf, -np.inf, np.inf]))

    def test_ppf_recomputed_after_estimator_swap(self) -> None:
        rng = np.random.default_rng(3)
        distr = EmpiricalDistribution(rng.normal(0.0, 1.0, 300))

        wide = distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.9]))
        distr.set_estimator(ScipyGaussianKde(bandwidth=0.05))
        narrow = distr.calculate_characteristic(CharacteristicName.PPF, np.array([0.9]))

        assert np.isfinite(wide[0]) and np.isfinite(narrow[0])
        assert not np.isclose(wide[0], narrow[0])


class TestQueryMethodIntegration:
    def test_explain_and_query_stay_in_sync_after_swap(self) -> None:
        distr = _make_distr()

        before = distr.explain_computation_path(CharacteristicName.PPF)
        distr.set_estimator(ScipyGaussianKde(bandwidth="silverman"))
        after = distr.explain_computation_path(CharacteristicName.PPF)

        assert before.steps[-1].method_name == after.steps[-1].method_name == BISECTION_FITTER
