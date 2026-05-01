"""
Shared built-in transformation methods for binary operations.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from math import inf, isfinite
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import brentq

from pysatl_core.distributions.computations.computation import Method
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.transformations.operations.methods._utils import (
    _central_moments_from_raw,
    _source_requirements_for_roles,
)
from pysatl_core.types import (
    ComputationFunc,
    ContinuousCdfEvaluator,
    GenericCharacteristicName,
    NumericArray,
    ParentRole,
    ResolvedSourceMethods,
    SourceRequirements,
)

if TYPE_CHECKING:
    from pysatl_core.transformations.operations.distributions.binary.base import (
        BinaryDistribution,
    )

_LEFT_ROLE: ParentRole = "left"
_RIGHT_ROLE: ParentRole = "right"


type _RawMomentsEvaluator = Callable[..., tuple[float, float, float, float]]


def _requirements_both(*characteristics: GenericCharacteristicName) -> SourceRequirements:
    """Build source requirements for both binary roles."""
    return _source_requirements_for_roles((_LEFT_ROLE, _RIGHT_ROLE), *characteristics)


def _make_mean_from_raw_moments(
    distribution: BinaryDistribution,
    sources: ResolvedSourceMethods,
    *,
    raw_moments_evaluator: _RawMomentsEvaluator,
) -> ComputationFunc[Any, float]:
    """Build transformed mean."""

    def _mean(**options: Any) -> float:
        m1, _, _, _ = raw_moments_evaluator(distribution, sources, 1, **options)
        return m1

    return _mean


def _make_var_from_raw_moments(
    distribution: BinaryDistribution,
    sources: ResolvedSourceMethods,
    *,
    raw_moments_evaluator: _RawMomentsEvaluator,
) -> ComputationFunc[Any, float]:
    """Build transformed variance."""

    def _var(**options: Any) -> float:
        m1, m2, _, _ = raw_moments_evaluator(distribution, sources, 2, **options)
        return max(m2 - m1**2, 0.0)

    return _var


def _make_skew_from_raw_moments(
    distribution: BinaryDistribution,
    sources: ResolvedSourceMethods,
    *,
    raw_moments_evaluator: _RawMomentsEvaluator,
) -> ComputationFunc[Any, float]:
    """Build transformed skewness."""

    def _skew(**options: Any) -> float:
        m1, m2, m3, _ = raw_moments_evaluator(distribution, sources, 3, **options)
        variance, mu3, _ = _central_moments_from_raw(m1, m2, m3, 0.0)
        if variance <= 0.0:
            return 0.0
        return float(mu3 / variance**1.5)

    return _skew


def _make_kurt_from_raw_moments(
    distribution: BinaryDistribution,
    sources: ResolvedSourceMethods,
    *,
    raw_moments_evaluator: _RawMomentsEvaluator,
) -> ComputationFunc[Any, float]:
    """Build transformed raw or excess kurtosis."""

    def _kurt(*, excess: bool = False, **options: Any) -> float:
        m1, m2, m3, m4 = raw_moments_evaluator(distribution, sources, 4, **options)
        variance, _, mu4 = _central_moments_from_raw(m1, m2, m3, m4)
        raw = 3.0 if variance <= 0.0 else mu4 / variance**2
        return raw - 3.0 if excess else raw

    return _kurt


def _make_continuous_ppf(
    distribution: BinaryDistribution,
    sources: ResolvedSourceMethods,
    *,
    cdf_evaluator: ContinuousCdfEvaluator,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous PPF via numerical inversion of transformed CDF."""
    cdf_method = cast(Method[float, float], cdf_evaluator(distribution, sources))
    support = distribution._precomputed_support  # noqa: SLF001
    support_left = float(support.left) if isinstance(support, ContinuousSupport) else -inf
    support_right = float(support.right) if isinstance(support, ContinuousSupport) else inf

    def _solve_single_quantile(q: float, **options: Any) -> float:
        def _f(x: float) -> float:
            return float(cdf_method(x, **options) - q)

        def _f_brentq(
            x: np.float64,
            *_args: Any,
            **_kwargs: Any,
        ) -> np.float64:
            return np.float64(_f(float(x)))

        left = support_left
        right = support_right
        if not (isfinite(left) and isfinite(right) and left < right):
            left = -1.0
            right = 1.0
            for _ in range(80):
                fl = _f(left)
                fr = _f(right)
                if fl <= 0.0 <= fr:
                    break
                left *= 2.0
                right *= 2.0
            else:
                raise RuntimeError(
                    "Could not bracket transformed PPF root for binary continuous operation."
                )
        return float(brentq(cast(Any, _f_brentq), left, right, xtol=1e-10, maxiter=200))

    def _ppf(data: NumericArray, **options: Any) -> NumericArray:
        probabilities = np.asarray(data, dtype=float)
        flat_probabilities = probabilities.reshape(-1)

        if np.any((flat_probabilities < 0.0) | (flat_probabilities > 1.0)):
            raise ValueError("PPF input must be in [0, 1].")

        results = np.empty_like(flat_probabilities, dtype=float)
        is_zero = flat_probabilities == 0.0
        is_one = flat_probabilities == 1.0
        is_interior = ~(is_zero | is_one)

        results[is_zero] = support_left
        results[is_one] = support_right

        interior_indices = np.nonzero(is_interior)[0]
        for idx in interior_indices:
            results[idx] = _solve_single_quantile(float(flat_probabilities[idx]), **options)

        return cast(NumericArray, results.reshape(probabilities.shape))

    return cast(ComputationFunc[NumericArray, NumericArray], _ppf)


__all__: list[str] = []
