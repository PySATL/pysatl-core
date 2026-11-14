"""Error metrics per region.

Metric definitions (fixed for v1, documented policies for the edge cases
required by the report: zero reference, support boundaries, tails):

- ``abs_error`` — ``max |y - y_true|``; the obligatory metric where the
  reference is zero (boundaries, right tail of a CDF, pdf of a uniform).
- ``inf_norm`` — uniform norm of the relative error.  Zero-reference
  policy: for ``y_true == 0`` the absolute error stands in for the
  relative one (no division by zero, no overflow, loss stays visible).
- ``lre`` — ``-log10(inf_norm)``; the number of correct decimal digits.
  A perfect match is capped at 16.0 (no float carries more).
- ``e1`` (ppf scenarios) — relative error of the quantile itself, with the
  denominator floored at 1 to keep it finite near zero quantiles.
- ``e2_true`` (ppf scenarios) — ``max |F_true(q_found) - p|``: plugging the
  found quantile into the *reference's* CDF must return the requested
  level (issue #124 §3.5).  Requires a compute-capable reference; points
  where the reference CDF does not answer (non-finite ``F_true``) are
  excluded from the maximum and counted in
  ``diagnostics['e2_true_unanswered']`` — a NaN never becomes a metric
  (a false number is worse than an absent one).

Diagnostics (:class:`~pysatl_core_bench.accuracy.dto.MetricRecord.diagnostics`)
carry numbers that are *not* accuracy against the reference:

- ``e2_self_consistency`` (ppf scenarios) — the former ``e2``:
  ``max |F_subj(q_found) - p|``, the internal ppf->cdf consistency of the
  subject.  It involves no reference at all, so it lives outside the
  accuracy metrics (решение ред. 46: two metrics instead of one).
- ``e2_self_consistency_unanswered`` / ``e2_true_unanswered`` — counts of
  follow-up points excluded from the respective maxima.

The framework guarantees diagnostic key names never collide with metric
key names (``e2_self_consistency`` prefix vs the accuracy keys).

Partial reference coverage is still a measurement (D6): metrics run over
the covered points and holes are counted in ``n_missing_reference``.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from pysatl_core.types import CharacteristicName
from pysatl_core_bench.accuracy.dto import MetricRecord
from pysatl_core_bench.accuracy.scenarios.grid import Grid
from pysatl_core_bench.accuracy.scenarios.spec import AccuracyQuery
from pysatl_core_bench.accuracy.types import PROBABILITY_SPACE

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__ = ["MetricsEvaluator", "LRE_CAP"]

CdfFollowup = Callable[[Float64Array], Float64Array]
"""Quantile follow-up: CDF of one side evaluated at the found quantiles."""

LRE_CAP = 16.0


class MetricsEvaluator:
    """Computes error metrics of one pair over one region grid."""

    def evaluate(
        self,
        *,
        query: AccuracyQuery,
        grid: Grid,
        subject_values: Float64Array,
        reference_values: Float64Array,
        missing_reference: BoolArray | None = None,
        excluded: BoolArray | None = None,
        subject_cdf_followup: CdfFollowup | None = None,
        reference_cdf_followup: CdfFollowup | None = None,
    ) -> MetricRecord:
        """Build the :class:`MetricRecord` for one (query, region) cell.

        ``missing_reference`` masks points the reference does not cover;
        ``excluded`` masks points excluded by special-point policies.
        ``subject_cdf_followup`` computes the subject CDF and enables the
        ``e2_self_consistency`` diagnostic; ``reference_cdf_followup``
        computes the reference CDF and enables the ``e2_true`` accuracy
        metric (both ppf scenarios only).
        """
        y = np.asarray(subject_values, dtype=np.float64).ravel()
        y_true = np.asarray(reference_values, dtype=np.float64).ravel()
        missing = (
            np.zeros(y.shape, dtype=bool)
            if missing_reference is None
            else np.asarray(missing_reference, dtype=bool).ravel()
        )
        excluded = (
            np.zeros(y.shape, dtype=bool)
            if excluded is None
            else np.asarray(excluded, dtype=bool).ravel()
        )
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(y) | ~np.isfinite(y_true)
        compared = ~(missing | excluded | bad)
        # Reference non-finiteness is already counted by the missing mask;
        # the non-finite *subject* values would otherwise vanish silently
        # (REV-5, no silence): they get their own counter.
        n_invalid_subject = int((~np.isfinite(y) & ~missing & ~excluded).sum())

        record = MetricRecord(
            characteristic=str(query.characteristic),
            space=grid.region.space,
            region=grid.region.name,
            grid_hash=grid.grid_hash,
            n_missing_reference=int(missing.sum()),
            n_excluded=int(excluded.sum()),
            n_compared=int(compared.sum()),
            n_invalid_subject=n_invalid_subject,
        )
        if not compared.any():
            return record

        diff = np.abs(y[compared] - y_true[compared])
        ref = np.abs(y_true[compared])
        denominator = np.where(ref != 0.0, ref, 1.0)
        relative = diff / denominator

        metrics: dict[str, float] = {
            "abs_error": float(diff.max()),
            "inf_norm": float(relative.max()),
        }
        # REV-5: the 16-digit cap applies to any inf_norm below 1e-16,
        # not only to the exact zero.
        metrics["lre"] = (
            LRE_CAP if metrics["inf_norm"] < 10.0**-LRE_CAP else -math.log10(metrics["inf_norm"])
        )
        diagnostics: dict[str, float] = {}
        if query.characteristic == CharacteristicName.PPF:
            metrics["e1"] = float((diff / np.maximum(ref, 1.0)).max())
            self._fill_ppf_followups(
                metrics,
                diagnostics,
                grid,
                y,
                compared,
                subject_cdf_followup,
                reference_cdf_followup,
            )
        if diagnostics:
            record = replace(record, metrics=metrics, diagnostics=diagnostics)
        else:
            record = replace(record, metrics=metrics)
        return record

    def _fill_ppf_followups(
        self,
        metrics: dict[str, float],
        diagnostics: dict[str, float],
        grid: Grid,
        subject_quantiles: Float64Array,
        compared: BoolArray,
        subject_cdf_followup: CdfFollowup | None,
        reference_cdf_followup: CdfFollowup | None,
    ) -> None:
        """Fill ``e2_true`` and the ``e2_self_consistency`` diagnostic.

        Both follow-ups make sense only on probability levels (the levels
        axis of a ppf grid *is* the probability axis); a follow-up that
        answers non-finitely at some compared point is excluded from its
        maximum and counted — silence and NaN are both dishonest (REV-5).
        """
        if grid.region.space != PROBABILITY_SPACE:
            return
        levels = np.asarray(grid.points, dtype=np.float64).ravel()
        if subject_cdf_followup is not None:
            gap, unanswered = self._followup_max_gap(
                subject_cdf_followup, subject_quantiles, levels, compared
            )
            if gap is not None:
                diagnostics["e2_self_consistency"] = gap
            if unanswered:
                diagnostics["e2_self_consistency_unanswered"] = float(unanswered)
        if reference_cdf_followup is not None:
            gap, unanswered = self._followup_max_gap(
                reference_cdf_followup, subject_quantiles, levels, compared
            )
            if gap is not None:
                metrics["e2_true"] = gap
            if unanswered:
                diagnostics["e2_true_unanswered"] = float(unanswered)

    @staticmethod
    def _followup_max_gap(
        followup: CdfFollowup,
        subject_quantiles: Float64Array,
        levels: Float64Array,
        compared: BoolArray,
    ) -> tuple[float | None, int]:
        """``max |F(q_found) - p|`` over compared levels with finite answers.

        Returns the maximum and the count of compared points where the
        follow-up did not answer finitely (excluded from the maximum).
        """
        f_of_q = np.asarray(followup(subject_quantiles), dtype=np.float64).ravel()
        with np.errstate(invalid="ignore"):
            gap = np.abs(f_of_q[compared] - levels[compared])
        finite = np.isfinite(gap)
        value = float(gap[finite].max()) if finite.any() else None
        return value, int((~finite).sum())
