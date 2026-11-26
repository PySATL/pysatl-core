"""Invariant checker rules (K-6, REV-5).

The checker is the math-independent sanity layer: it disproofs values that
no distribution can produce, whatever the sources claim.  Every rule here
is a documented v1 boundary — cdf/pdf/ppf only, native scale only — and
the tests pin both the rules and the boundary (a characteristic without
rules yields no violations, not a guess).
"""

from __future__ import annotations

import numpy as np
import pytest

from pysatl_core.types import CharacteristicName
from pysatl_core_bench import AccuracyQuery, InvariantChecker
from pysatl_core_bench.accuracy import VIOLATION_CODE


def query(name: str) -> AccuracyQuery:
    return AccuracyQuery(characteristic=CharacteristicName(name))


class TestCdfRules:
    def test_range_violation_is_reported(self) -> None:
        violations = InvariantChecker().check(query("cdf"), np.asarray([0.5, 1.2]))
        assert any("must lie in [0, 1]" in v.detail for v in violations)

    def test_negative_violation_is_reported(self) -> None:
        violations = InvariantChecker().check(query("cdf"), np.asarray([-0.1, 0.5]))
        assert any("must lie in [0, 1]" in v.detail for v in violations)

    def test_decreasing_violation_is_reported(self) -> None:
        violations = InvariantChecker().check(query("cdf"), np.asarray([0.7, 0.3]))
        assert any("non-decreasing" in v.detail for v in violations)

    def test_monotone_values_pass(self) -> None:
        assert InvariantChecker().check(query("cdf"), np.asarray([0.1, 0.5, 0.9])) == []

    def test_nan_is_a_violation(self) -> None:
        violations = InvariantChecker().check(query("cdf"), np.asarray([0.5, np.nan]))
        assert any("not finite" in v.detail for v in violations)


class TestPdfRules:
    def test_negative_density_is_reported(self) -> None:
        violations = InvariantChecker().check(query("pdf"), np.asarray([1.0, -0.2]))
        assert any("non-negative" in v.detail for v in violations)

    def test_non_negative_density_passes(self) -> None:
        assert InvariantChecker().check(query("pdf"), np.asarray([0.2, 0.4])) == []


class TestPpfRules:
    def test_decreasing_quantiles_are_reported(self) -> None:
        violations = InvariantChecker().check(query("ppf"), np.asarray([1.0, 0.5]))
        assert any("non-decreasing" in v.detail for v in violations)

    def test_monotone_quantiles_pass(self) -> None:
        assert InvariantChecker().check(query("ppf"), np.asarray([-1.0, 0.0, 1.0])) == []


class TestRuleBoundaries:
    def test_characteristic_without_rules_has_no_checks(self) -> None:
        # lpdf has no v1 rule: an unconstrained check would be a guess.
        assert InvariantChecker().check(query("lpdf"), np.asarray([-5.0, 3.0])) == []

    def test_violations_carry_the_framework_code(self) -> None:
        violations = InvariantChecker().check(query("pdf"), np.asarray([-1.0]))
        assert violations and all(v.code == VIOLATION_CODE for v in violations)

    @pytest.mark.parametrize("values", [np.asarray([]), np.asarray([1.0])], ids=["empty", "single"])
    def test_degenerate_arrays_are_not_violations(self, values: np.ndarray) -> None:
        # Monotonicity needs at least two points; an empty/single grid is
        # an honest outcome, not a rule break.
        assert InvariantChecker().check(query("cdf"), values) == []
