"""Optional invariant post-check (К-6).

The checker is *not* a value source (it returns violations, never ``y``)
and sits outside the ``ValueSource`` hierarchy.  By default
(``invariant_check_on`` empty) no check runs at all.  A violation
disproves the values of one participant; the comparator spreads the
verdict to every pair of the group containing that participant
(``measured`` -> ``invalid_reference``).
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import NDArray

from pysatl_core.types import CharacteristicName
from pysatl_core_bench.accuracy.dto import Violation
from pysatl_core_bench.accuracy.scenarios.spec import AccuracyQuery

Float64Array = NDArray[np.float64]

__all__ = ["InvariantChecker"]

VIOLATION_CODE = "invariant_violation"


class InvariantChecker:
    """Mathematical sanity checks of one participant's values."""

    def check(self, query: AccuracyQuery, values: Float64Array) -> list[Violation]:
        """Check values in the *native* space of the participant (REV-5).

        The runner passes unconverted (native) values — the rule of the
        native space (N2/D8): monotonicity of cdf/ppf holds in the native
        space of v1 sources (probability).  A source answering natively in
        survival space would need direction-aware rules, which v1 does not
        have; characteristics without rules (everything beyond pdf/cdf/ppf)
        get no checks — that limit is a documented v1 boundary, and the
        ``invariant_check_skipped`` note channel reports skips.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        violations: list[Violation] = []
        with np.errstate(invalid="ignore"):
            non_finite = int((~np.isfinite(arr)).sum())
        if non_finite:
            violations.append(
                Violation(
                    participant_id="",
                    code=VIOLATION_CODE,
                    detail=f"{non_finite} values are not finite",
                )
            )
        if query.characteristic == CharacteristicName.CDF:
            violations.extend(self._cdf_checks(arr))
        elif query.characteristic == CharacteristicName.PDF:
            violations.extend(self._pdf_checks(arr))
        elif query.characteristic == CharacteristicName.PPF:
            violations.extend(self._ppf_checks(arr))
        return violations

    @staticmethod
    def _cdf_checks(arr: Float64Array) -> list[Violation]:
        violations: list[Violation] = []
        if arr.size and np.isfinite(arr).any():
            low, high = float(np.nanmin(arr)), float(np.nanmax(arr))
            if low < 0.0 or high > 1.0:
                violations.append(
                    Violation(
                        participant_id="",
                        code=VIOLATION_CODE,
                        detail="cdf values must lie in [0, 1]",
                    )
                )
        if arr.size > 1 and np.any(np.diff(arr) < -np.finfo(np.float64).eps):
            violations.append(
                Violation(
                    participant_id="",
                    code=VIOLATION_CODE,
                    detail="cdf values must be non-decreasing",
                )
            )
        return violations

    @staticmethod
    def _pdf_checks(arr: Float64Array) -> list[Violation]:
        if arr.size and np.isfinite(arr).any() and float(np.nanmin(arr)) < 0.0:
            return [
                Violation(
                    participant_id="",
                    code=VIOLATION_CODE,
                    detail="pdf values must be non-negative",
                )
            ]
        return []

    @staticmethod
    def _ppf_checks(arr: Float64Array) -> list[Violation]:
        if arr.size > 1 and np.any(np.diff(arr) < 0.0):
            return [
                Violation(
                    participant_id="",
                    code=VIOLATION_CODE,
                    detail="ppf values must be non-decreasing in the level",
                )
            ]
        return []
