"""
Shared numerical and moment utilities for transformation methods.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from math import sqrt
from typing import Any, cast

import numpy as np
from scipy.integrate import quad

from pysatl_core.distributions.computations.computation import Method
from pysatl_core.types import (
    ComplexArray,
    ComputationFunc,
    GenericCharacteristicName,
    LabelName,
    NumericArray,
    ParentRole,
    ResolvedSourceMethods,
    SourceRequirements,
    TransformationMethodSpec,
    TransformationMethodSpecsMap,
)

_MAX_MOMENT_ORDER = 4


def _source_requirements_for_roles(
    roles: tuple[ParentRole, ...],
    *characteristics: GenericCharacteristicName,
) -> SourceRequirements:
    """Build source requirements for a fixed set of parent roles."""
    required = tuple(characteristics)
    return dict.fromkeys(roles, required)


def _source_requirements_for_distribution_roles(
    *characteristics: GenericCharacteristicName,
) -> Callable[[object], SourceRequirements]:
    """Build a resolver for source requirements across distribution parent roles."""
    required = tuple(characteristics)

    def _resolver(distribution: object) -> SourceRequirements:
        roles = cast(tuple[ParentRole, ...], cast(Any, distribution).parent_roles)
        return dict.fromkeys(roles, required)

    return _resolver


def _merge_transformation_methods(
    *method_sets: TransformationMethodSpecsMap,
) -> dict[GenericCharacteristicName, dict[LabelName, TransformationMethodSpec[Any, Any]]]:
    """Merge transformation method dictionaries preserving per-target labels."""
    merged: dict[
        GenericCharacteristicName, dict[LabelName, TransformationMethodSpec[Any, Any]]
    ] = {}
    for method_set in method_sets:
        for target, variants in method_set.items():
            merged.setdefault(target, {}).update(variants)
    return merged


def _validate_moment_order(max_order: int) -> int:
    """Validate and normalize required moment order."""
    if not 1 <= max_order <= _MAX_MOMENT_ORDER:
        raise ValueError(f"Moment order must be in [1, {_MAX_MOMENT_ORDER}], got {max_order}.")
    return max_order


def _eval_method_scalar(
    method: Method[Any, Any],
    argument: float,
    **options: Any,
) -> float:
    """Evaluate a scalar-valued method at one point and cast to ``float``."""
    return float(np.asarray(method(argument, **options), dtype=float))


def _eval_method_scalar_complex(
    method: Method[Any, Any],
    argument: float,
    **options: Any,
) -> complex:
    """Evaluate a complex-valued method at one point and cast to ``complex``."""
    return complex(np.asarray(method(argument, **options), dtype=complex))


def _eval_nullary_scalar(
    method: Method[Any, Any],
    **options: Any,
) -> float:
    """Evaluate a nullary method and cast to ``float``."""
    return float(np.asarray(method(**options), dtype=float))


def _map_scalar_real(
    data: NumericArray,
    scalar_func: Callable[[float], float],
) -> NumericArray:
    """Apply a scalar real function to scalar or vector input."""
    array = np.asarray(data, dtype=float)
    flat = array.reshape(-1)
    mapped = np.fromiter((scalar_func(float(x)) for x in flat), dtype=float, count=flat.size)
    return cast(NumericArray, mapped.reshape(array.shape))


def _map_scalar_complex(
    data: NumericArray,
    scalar_func: Callable[[float], complex],
) -> ComplexArray:
    """Apply a scalar complex function to scalar or vector input."""
    array = np.asarray(data, dtype=float)
    flat = array.reshape(-1)
    mapped = np.fromiter((scalar_func(float(x)) for x in flat), dtype=complex, count=flat.size)
    return cast(ComplexArray, mapped.reshape(array.shape))


def _quad_real(
    integrand: Callable[[float], float],
    left: float,
    right: float,
) -> float:
    """Integrate a real-valued function with SciPy ``quad``."""
    value, _ = quad(integrand, left, right, limit=300)
    return float(value)


def _integrate_real(
    integrand: Callable[[float], float],
    left: float,
    right: float,
    *,
    split_at_zero: bool = False,
) -> float:
    """Integrate a real function and optionally split around zero."""
    if left >= right:
        return 0.0

    if split_at_zero and left < 0.0 < right:
        eps_pos = float(np.nextafter(0.0, 1.0))
        eps_neg = float(np.nextafter(0.0, -1.0))
        return _quad_real(integrand, left, eps_neg) + _quad_real(integrand, eps_pos, right)

    return _quad_real(integrand, left, right)


def _integrate_complex(
    integrand: Callable[[float], complex],
    left: float,
    right: float,
    *,
    split_at_zero: bool = False,
) -> complex:
    """Integrate a complex function by integrating real and imaginary parts."""
    real_part = _integrate_real(
        lambda x: float(np.real(integrand(x))),
        left,
        right,
        split_at_zero=split_at_zero,
    )
    imag_part = _integrate_real(
        lambda x: float(np.imag(integrand(x))),
        left,
        right,
        split_at_zero=split_at_zero,
    )
    return complex(real_part, imag_part)


def _kurt_raw_from_method(
    method: Method[Any, Any],
    **options: Any,
) -> float:
    """Evaluate raw kurtosis from a method with optional ``excess`` support."""
    try:
        value = method(excess=False, **options)
    except TypeError:
        value = method(**options)
    return float(np.asarray(value, dtype=float))


def _discrete_mass_table(
    distribution: object,
    sources: ResolvedSourceMethods,
    **options: Any,
) -> tuple[NumericArray, NumericArray, NumericArray]:
    """Resolve cached discrete points/masses/CDF table from transformation distribution."""
    table = cast(Any, distribution)._discrete_mass_table(sources, **options)
    return cast(tuple[NumericArray, NumericArray, NumericArray], table)


def _make_discrete_pmf_from_mass_table(
    distribution: object,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed discrete PMF from a finite mass table."""

    def _pmf(data: NumericArray, **options: Any) -> NumericArray:
        points, masses, _ = _discrete_mass_table(distribution, sources, **options)
        array = np.asarray(data, dtype=float)
        flat = array.reshape(-1)

        indices = np.searchsorted(points, flat, side="left")
        values = np.zeros_like(flat, dtype=float)

        in_range = indices < points.size
        if np.any(in_range):
            in_range_indices = indices[in_range]
            close = np.isclose(points[in_range_indices], flat[in_range], atol=1e-12, rtol=0.0)
            if np.any(close):
                values[np.where(in_range)[0][close]] = masses[in_range_indices[close]]

        return cast(NumericArray, values.reshape(array.shape))

    return cast(ComputationFunc[NumericArray, NumericArray], _pmf)


def _make_discrete_cdf_from_mass_table(
    distribution: object,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed discrete CDF from a finite mass table."""

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        points, _, cdf_values = _discrete_mass_table(distribution, sources, **options)
        array = np.asarray(data, dtype=float)
        flat = array.reshape(-1)

        indices = np.searchsorted(points, flat, side="right") - 1
        values = np.zeros_like(flat, dtype=float)

        valid = indices >= 0
        if np.any(valid):
            clipped = np.minimum(indices[valid], cdf_values.size - 1)
            values[valid] = cdf_values[clipped]

        return cast(NumericArray, values.reshape(array.shape))

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


def _make_discrete_ppf_from_mass_table(
    distribution: object,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed discrete PPF from a finite mass table."""

    def _ppf(data: NumericArray, **options: Any) -> NumericArray:
        points, _, cdf_values = _discrete_mass_table(distribution, sources, **options)
        array = np.asarray(data, dtype=float)
        flat = array.reshape(-1)
        if np.any((flat < 0.0) | (flat > 1.0)):
            raise ValueError("PPF input must be in [0, 1].")

        indices = np.searchsorted(cdf_values, flat, side="left")
        clipped = np.clip(indices, 0, points.size - 1)
        values = points[clipped]
        return cast(NumericArray, values.reshape(array.shape))

    return cast(ComputationFunc[NumericArray, NumericArray], _ppf)


def _raw_moments_from_statistics(
    mean: float,
    variance: float,
    skewness: float,
    raw_kurtosis: float,
) -> tuple[float, float, float, float]:
    """Convert mean/variance/skewness/kurtosis to raw moments up to order 4."""
    variance_safe = max(variance, 0.0)
    std = sqrt(variance_safe)
    mu3 = skewness * std**3
    mu4 = raw_kurtosis * variance_safe**2

    m1 = mean
    m2 = variance_safe + mean**2
    m3 = mu3 + 3.0 * mean * variance_safe + mean**3
    m4 = mu4 + 4.0 * mean * mu3 + 6.0 * mean**2 * variance_safe + mean**4
    return m1, m2, m3, m4


def _central_moments_from_raw(
    m1: float,
    m2: float,
    m3: float,
    m4: float,
) -> tuple[float, float, float]:
    """Convert raw moments to central moments ``(var, mu3, mu4)``."""
    variance = max(m2 - m1**2, 0.0)
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1**3
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1**2 * m2 - 3.0 * m1**4
    return variance, mu3, mu4


def _parent_raw_moments(
    sources: ResolvedSourceMethods,
    role: str,
    *,
    mean_name: GenericCharacteristicName,
    var_name: GenericCharacteristicName,
    skew_name: GenericCharacteristicName,
    kurt_name: GenericCharacteristicName,
    max_order: int = _MAX_MOMENT_ORDER,
    **options: Any,
) -> tuple[float, float, float, float]:
    """Get parent raw moments up to ``max_order`` from statistical characteristics."""
    order = _validate_moment_order(max_order)
    mean_method = sources[role][mean_name]
    mean = _eval_nullary_scalar(mean_method, **options)
    if order == 1:
        return mean, 0.0, 0.0, 0.0

    var_method = sources[role][var_name]
    variance = _eval_nullary_scalar(var_method, **options)
    m1 = mean
    m2 = max(variance, 0.0) + mean**2
    if order == 2:
        return m1, m2, 0.0, 0.0

    skew_method = sources[role][skew_name]
    skewness = _eval_nullary_scalar(skew_method, **options)
    _, _, m3, _ = _raw_moments_from_statistics(mean, variance, skewness, 3.0)
    if order == 3:
        return m1, m2, m3, 0.0

    kurt_method = sources[role][kurt_name]
    raw_kurtosis = _kurt_raw_from_method(kurt_method, **options)
    return _raw_moments_from_statistics(mean, variance, skewness, raw_kurtosis)


__all__ = [
    "_central_moments_from_raw",
    "_eval_method_scalar",
    "_eval_method_scalar_complex",
    "_eval_nullary_scalar",
    "_integrate_complex",
    "_integrate_real",
    "_kurt_raw_from_method",
    "_make_discrete_cdf_from_mass_table",
    "_make_discrete_pmf_from_mass_table",
    "_make_discrete_ppf_from_mass_table",
    "_merge_transformation_methods",
    "_map_scalar_complex",
    "_map_scalar_real",
    "_parent_raw_moments",
    "_raw_moments_from_statistics",
    "_source_requirements_for_distribution_roles",
    "_source_requirements_for_roles",
    "_validate_moment_order",
]
