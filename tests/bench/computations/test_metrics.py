"""Metrics and output_scale conversion (D6, D8/N2, report metric policies)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pysatl_core.types import CharacteristicName
from pysatl_core_bench import AccuracyQuery, Grid, MetricsEvaluator, OutputScaleConverter, Region
from pysatl_core_bench.accuracy.errors import FrameworkDefect


def make_grid(levels: np.ndarray[Any, Any] | None = None) -> Grid:
    points = np.linspace(-2.0, 2.0, 21) if levels is None else levels
    return Grid(
        query=AccuracyQuery(characteristic=CharacteristicName.CDF),
        region=Region(name="bulk", interval=(-2.0, 2.0), output_scale="probability"),
        points=points,
        grid_hash="hash-test",
    )


class TestMetricsEvaluator:
    def test_exact_match_caps_lre(self) -> None:
        y = np.linspace(0.1, 0.9, 9)
        grid = make_grid(y)
        record = MetricsEvaluator().evaluate(
            query=grid.query, grid=grid, subject_values=y, reference_values=y
        )
        assert record.metrics["lre"] == 16.0
        assert record.metrics["abs_error"] == 0.0
        assert record.n_compared == 9

    def test_lre_counts_correct_digits(self) -> None:
        y_true = np.full(5, 0.5)
        y = y_true * (1.0 + 1e-6)
        grid = make_grid(y_true)
        record = MetricsEvaluator().evaluate(
            query=grid.query, grid=grid, subject_values=y, reference_values=y_true
        )
        assert record.metrics["lre"] == pytest.approx(6.0, abs=1e-6)

    def test_zero_reference_uses_absolute_policy(self) -> None:
        y_true = np.zeros(4)
        y = np.array([1e-9, 0.0, 0.0, 0.0])
        grid = make_grid(y_true)
        record = MetricsEvaluator().evaluate(
            query=grid.query, grid=grid, subject_values=y, reference_values=y_true
        )
        assert record.metrics["abs_error"] == 1e-9
        assert record.metrics["inf_norm"] == 1e-9
        assert record.metrics["lre"] == pytest.approx(9.0)

    def test_partial_coverage_is_measured_with_holes(self) -> None:
        y = np.linspace(0.1, 0.9, 9)
        missing = np.zeros(9, dtype=bool)
        missing[[2, 7]] = True
        grid = make_grid(y)
        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
            subject_values=y,
            reference_values=y,
            missing_reference=missing,
        )
        assert record.n_compared == 7
        assert record.n_missing_reference == 2

    def test_excluded_points_dictionary_carries_coordinates_and_reasons(self) -> None:
        # AP-1: an inside-support point without a comparable value is
        # enumerated with its coordinate and a reason tag named after the
        # counter that aggregates the same fact.
        y = np.linspace(0.1, 0.9, 9)
        subject = y.copy()
        subject[6] = np.nan  # the subject answers nothing there
        missing = np.zeros(9, dtype=bool)
        missing[3] = True  # a reference hole (declared or non-finite)
        grid = make_grid(y)
        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
            subject_values=subject,
            reference_values=y,
            missing_reference=missing,
        )
        by_reason = {(p.x, p.reason) for p in record.excluded_points}
        assert (float(grid.points[3]), "n_missing_reference") in by_reason
        assert (float(grid.points[6]), "n_invalid_subject") in by_reason
        assert len(record.excluded_points) == 2
        assert record.n_missing_reference == 1
        assert record.n_invalid_subject == 1

    def test_singularity_tag_is_reserved_but_wired(self) -> None:
        # The `excluded` mask (special-point exclusion policy) carries the
        # reserved `singularity` reason; no v1 policy emits it, but the
        # dictionary schema must already speak the tag correctly.
        y = np.linspace(0.1, 0.9, 9)
        excluded = np.zeros(9, dtype=bool)
        excluded[4] = True
        grid = make_grid(y)
        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
            subject_values=y,
            reference_values=y,
            excluded=excluded,
        )
        assert record.n_excluded == 1
        assert (float(grid.points[4]), "singularity") in {
            (p.x, p.reason) for p in record.excluded_points
        }
        assert record.n_compared == 8

    def test_all_unanswered_followups_count_without_a_metric(self) -> None:
        # A follow-up that answers non-finitely everywhere produces no
        # e2 number at all (a NaN never becomes a metric) — the unanswered
        # counters carry the fact instead.
        levels = np.linspace(0.1, 0.9, 9)
        grid = Grid(
            query=AccuracyQuery(characteristic=CharacteristicName.PPF),
            region=Region(name="bulk", interval=(0.0, 1.0), output_scale="probability"),
            points=levels,
            grid_hash="hash-ppf",
        )
        from scipy import stats

        quantiles = stats.norm.ppf(levels)

        def dead(q):
            return np.full(np.asarray(q).shape, np.nan)

        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
            subject_values=quantiles,
            reference_values=quantiles,
            subject_cdf_followup=dead,
            reference_cdf_followup=dead,
        )
        assert "e2_true" not in record.metrics
        assert "e2_self_consistency" not in record.diagnostics
        assert record.diagnostics["e2_true_unanswered"] == 9
        assert record.diagnostics["e2_self_consistency_unanswered"] == 9

    def test_misaligned_values_are_a_framework_defect(self) -> None:
        # The values are grid-bound by pipeline contract: a source that
        # answers a different number of points than the grid carries must
        # fail loudly, not silently shift the coordinate attribution.
        y = np.linspace(0.1, 0.9, 4)
        with pytest.raises(FrameworkDefect, match="must align with the grid points"):
            MetricsEvaluator().evaluate(
                query=make_grid().query,
                grid=make_grid(),
                subject_values=y,
                reference_values=y,
            )

    def test_no_compared_points_yields_empty_metrics(self) -> None:
        y = np.linspace(0.1, 0.9, 4)
        missing = np.ones(4, dtype=bool)
        grid = make_grid(y)
        record = MetricsEvaluator().evaluate(
            query=grid.query,
            grid=grid,
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
            region=Region(name="bulk", interval=(0.0, 1.0), output_scale="probability"),
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
        # The subject-side follow-up involves no reference (the rev. 46 decision):
        # it is a diagnostic now, never an accuracy metric.
        assert "e2" not in record.metrics
        assert record.metrics["e1"] == 0.0
        assert record.diagnostics["e2_self_consistency"] == pytest.approx(0.0, abs=1e-12)

    def test_e2_true_uses_reference_cdf_followup(self) -> None:
        levels = np.linspace(0.1, 0.9, 9)
        grid = Grid(
            query=AccuracyQuery(characteristic=CharacteristicName.PPF),
            region=Region(name="bulk", interval=(0.0, 1.0), output_scale="probability"),
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
            region=Region(name="bulk", interval=(0.0, 1.0), output_scale="probability"),
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


class TestOutputScaleConverter:
    def test_identity_spaces(self) -> None:
        values = np.array([0.1, 0.5, 0.9])
        converter = OutputScaleConverter()
        np.testing.assert_array_equal(
            converter.convert(values, "probability", "probability"), values
        )

    def test_survival_identity(self) -> None:
        f = np.array([0.1, 0.5, 0.9])
        converter = OutputScaleConverter()
        s = converter.convert(f, "probability", "survival")
        np.testing.assert_allclose(s, 1.0 - f)
        back = converter.convert(s, "survival", "probability")
        np.testing.assert_allclose(back, f)

    def test_unknown_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported output_scale conversion"):
            OutputScaleConverter().convert(np.zeros(2), "probability", "log")

    def test_conversion_fact(self) -> None:
        fact = OutputScaleConverter().conversion_fact("probability", "survival")
        assert fact == {"from_scale": "probability", "to_scale": "survival"}


class TestFollowupScaleGuard:
    """Follow-ups make sense only on probability levels; a non-probability
    region skips them explicitly instead of producing a meaningless number
    (metrics.py scale guard)."""

    def test_ppf_followups_skip_a_survival_region(self) -> None:
        from pysatl_core_bench import GridGenerator, Region

        grid = GridGenerator().generate(
            [
                AccuracyQuery(
                    characteristic=CharacteristicName.PPF,
                    grid_hint={"n_points": 9},
                )
            ],
            [Region("tail", (0.9, None), "survival")],
            [],
            None,
        )[0]
        levels = np.asarray(grid.points)
        record = MetricsEvaluator().evaluate(
            query=AccuracyQuery(characteristic=CharacteristicName.PPF),
            grid=grid,
            subject_values=np.full(levels.size, 0.2),
            reference_values=np.full(levels.size, 0.8),
            subject_cdf_followup=lambda q: np.full(q.size, 0.5),
            reference_cdf_followup=lambda q: np.full(q.size, 0.5),
        )
        assert "e1" in record.metrics  # e1 works on any scale
        assert "e2_true" not in record.metrics  # follow-up skipped: not probability
        assert "e2_self_consistency" not in record.diagnostics
