"""Metrics and space conversion (D6, D8/N2, report metric policies)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pysatl_core.types import CharacteristicName
from pysatl_core_bench import AccuracyQuery, Grid, MetricsEvaluator, Region, SpaceConverter


def make_grid(levels: np.ndarray[Any, Any] | None = None) -> Grid:
    points = np.linspace(-2.0, 2.0, 21) if levels is None else levels
    return Grid(
        query=AccuracyQuery(characteristic=CharacteristicName.CDF),
        region=Region(name="bulk", interval=(-2.0, 2.0), space="probability"),
        points=points,
        grid_hash="hash-test",
    )


class TestMetricsEvaluator:
    def test_exact_match_caps_lre(self) -> None:
        y = np.linspace(0.1, 0.9, 9)
        record = MetricsEvaluator().evaluate(
            query=make_grid().query, grid=make_grid(), subject_values=y, reference_values=y
        )
        assert record.metrics["lre"] == 16.0
        assert record.metrics["abs_error"] == 0.0
        assert record.n_compared == 9

    def test_lre_counts_correct_digits(self) -> None:
        y_true = np.full(5, 0.5)
        y = y_true * (1.0 + 1e-6)
        record = MetricsEvaluator().evaluate(
            query=make_grid().query, grid=make_grid(), subject_values=y, reference_values=y_true
        )
        assert record.metrics["lre"] == pytest.approx(6.0, abs=1e-6)

    def test_zero_reference_uses_absolute_policy(self) -> None:
        y_true = np.zeros(4)
        y = np.array([1e-9, 0.0, 0.0, 0.0])
        record = MetricsEvaluator().evaluate(
            query=make_grid().query, grid=make_grid(), subject_values=y, reference_values=y_true
        )
        assert record.metrics["abs_error"] == 1e-9
        assert record.metrics["inf_norm"] == 1e-9
        assert record.metrics["lre"] == pytest.approx(9.0)

    def test_partial_coverage_is_measured_with_holes(self) -> None:
        y = np.linspace(0.1, 0.9, 9)
        missing = np.zeros(9, dtype=bool)
        missing[[2, 7]] = True
        record = MetricsEvaluator().evaluate(
            query=make_grid().query,
            grid=make_grid(),
            subject_values=y,
            reference_values=y,
            missing_reference=missing,
        )
        assert record.n_compared == 7
        assert record.n_missing_reference == 2

    def test_no_compared_points_yields_empty_metrics(self) -> None:
        y = np.linspace(0.1, 0.9, 4)
        missing = np.ones(4, dtype=bool)
        record = MetricsEvaluator().evaluate(
            query=make_grid().query,
            grid=make_grid(),
            subject_values=y,
            reference_values=y,
            missing_reference=missing,
        )
        assert record.n_compared == 0
        assert record.metrics == {}

    def test_e2_self_consistency_is_diagnostic_not_metric(self) -> None:
        levels = np.linspace(0.1, 0.9, 9)
        grid = Grid(
            query=AccuracyQuery(characteristic=CharacteristicName.PPF),
            region=Region(name="bulk", interval=(0.0, 1.0), space="probability"),
            points=levels,
            grid_hash="hash-ppf",
        )
        from scipy import stats

        quantiles = stats.norm.ppf(levels)
        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
            subject_values=quantiles,
            reference_values=quantiles,
            subject_cdf_followup=lambda q: stats.norm.cdf(q),
        )
        # The subject-side follow-up involves no reference (решение ред. 46):
        # it is a diagnostic now, never an accuracy metric.
        assert "e2" not in record.metrics
        assert record.metrics["e1"] == 0.0
        assert record.diagnostics["e2_self_consistency"] == pytest.approx(0.0, abs=1e-12)

    def test_e2_true_uses_reference_cdf_followup(self) -> None:
        levels = np.linspace(0.1, 0.9, 9)
        grid = Grid(
            query=AccuracyQuery(characteristic=CharacteristicName.PPF),
            region=Region(name="bulk", interval=(0.0, 1.0), space="probability"),
            points=levels,
            grid_hash="hash-ppf",
        )
        from scipy import stats

        exact = stats.norm.ppf(levels)
        off = exact + 1e-3  # found quantiles slightly wrong
        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
            subject_values=off,
            reference_values=exact,
            reference_cdf_followup=lambda q: stats.norm.cdf(q),
        )
        expected = float(np.abs(stats.norm.cdf(off) - levels).max())
        assert record.metrics["e2_true"] == pytest.approx(expected, rel=1e-9)
        assert record.diagnostics == {}

    def test_followup_unanswered_points_counted_not_nan(self) -> None:
        levels = np.linspace(0.1, 0.9, 9)
        grid = Grid(
            query=AccuracyQuery(characteristic=CharacteristicName.PPF),
            region=Region(name="bulk", interval=(0.0, 1.0), space="probability"),
            points=levels,
            grid_hash="hash-ppf",
        )
        from scipy import stats

        def flaky(q):
            out = stats.norm.cdf(q)
            out[::2] = np.nan
            return out

        quantiles = stats.norm.ppf(levels)
        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
            subject_values=quantiles,
            reference_values=quantiles,
            subject_cdf_followup=flaky,
            reference_cdf_followup=flaky,
        )
        # A NaN never becomes a metric (a false number is worse than an
        # absent one); unanswered points are counted explicitly.
        assert np.isfinite(record.metrics["e2_true"])
        assert record.diagnostics["e2_true_unanswered"] == 5
        assert np.isfinite(record.diagnostics["e2_self_consistency"])
        assert record.diagnostics["e2_self_consistency_unanswered"] == 5


class TestSpaceConverter:
    def test_identity_spaces(self) -> None:
        values = np.array([0.1, 0.5, 0.9])
        converter = SpaceConverter()
        np.testing.assert_array_equal(
            converter.convert(values, "probability", "probability"), values
        )

    def test_survival_identity(self) -> None:
        f = np.array([0.1, 0.5, 0.9])
        converter = SpaceConverter()
        s = converter.convert(f, "probability", "survival")
        np.testing.assert_allclose(s, 1.0 - f)
        back = converter.convert(s, "survival", "probability")
        np.testing.assert_allclose(back, f)

    def test_unknown_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported space conversion"):
            SpaceConverter().convert(np.zeros(2), "probability", "log")

    def test_conversion_fact(self) -> None:
        fact = SpaceConverter().conversion_fact("probability", "survival")
        assert fact == {"from_space": "probability", "to_space": "survival"}
