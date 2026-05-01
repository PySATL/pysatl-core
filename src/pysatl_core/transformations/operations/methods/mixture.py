"""
Built-in transformation methods for finite mixture operations.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.computations.computation import Method
from pysatl_core.transformations.operations.methods._utils import (
    _central_moments_from_raw,
    _eval_nullary_scalar,
    _kurt_raw_from_method,
    _make_discrete_cdf_from_mass_table as _make_discrete_cdf,
    _make_discrete_pmf_from_mass_table as _make_discrete_pmf,
    _make_discrete_ppf_from_mass_table as _make_discrete_ppf,
    _merge_transformation_methods,
    _raw_moments_from_statistics,
    _source_requirements_for_distribution_roles,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    CharacteristicName,
    ComplexArray,
    ComputationFunc,
    GenericCharacteristicName,
    Kind,
    NumericArray,
    ResolvedSourceMethods,
    SourceRequirements,
    TransformationMethodSpecsMap,
)

if TYPE_CHECKING:
    from pysatl_core.transformations.operations.distributions.mixture import (
        FiniteMixtureDistribution,
    )


def _requirements_for_all(
    *characteristics: GenericCharacteristicName,
) -> Callable[[object], SourceRequirements]:
    """Build a resolver for requirements over all mixture component roles."""
    return _source_requirements_for_distribution_roles(*characteristics)


def _make_mean(
    distribution: FiniteMixtureDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build mixture mean."""
    roles = distribution.parent_roles
    weights = distribution.weights

    def _mean(**options: Any) -> float:
        total = 0.0
        for weight, role in zip(weights, roles, strict=True):
            mean = _eval_nullary_scalar(sources[role][CharacteristicName.MEAN], **options)
            total += float(weight) * mean
        return total

    return _mean


def _make_var(
    distribution: FiniteMixtureDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build mixture variance."""
    roles = distribution.parent_roles
    weights = distribution.weights

    def _var(**options: Any) -> float:
        m1 = 0.0
        m2 = 0.0
        for weight, role in zip(weights, roles, strict=True):
            methods = sources[role]
            mean = _eval_nullary_scalar(methods[CharacteristicName.MEAN], **options)
            variance = _eval_nullary_scalar(methods[CharacteristicName.VAR], **options)
            weight_float = float(weight)
            m1 += weight_float * mean
            m2 += weight_float * (max(variance, 0.0) + mean**2)
        return max(m2 - m1**2, 0.0)

    return _var


def _make_skew(
    distribution: FiniteMixtureDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build mixture skewness."""
    roles = distribution.parent_roles
    weights = distribution.weights

    def _skew(**options: Any) -> float:
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        for weight, role in zip(weights, roles, strict=True):
            methods = sources[role]
            mean = _eval_nullary_scalar(methods[CharacteristicName.MEAN], **options)
            variance = _eval_nullary_scalar(methods[CharacteristicName.VAR], **options)
            skewness = _eval_nullary_scalar(methods[CharacteristicName.SKEW], **options)
            _, component_m2, component_m3, _ = _raw_moments_from_statistics(
                mean,
                variance,
                skewness,
                3.0,
            )
            weight_float = float(weight)
            m1 += weight_float * mean
            m2 += weight_float * component_m2
            m3 += weight_float * component_m3

        variance, mu3, _ = _central_moments_from_raw(m1, m2, m3, 0.0)
        if variance <= 0.0:
            return 0.0
        return float(mu3 / variance**1.5)

    return _skew


def _make_kurt(
    distribution: FiniteMixtureDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build mixture raw or excess kurtosis."""
    roles = distribution.parent_roles
    weights = distribution.weights

    def _kurt(*, excess: bool = False, **options: Any) -> float:
        m1 = 0.0
        m2 = 0.0
        m3 = 0.0
        m4 = 0.0
        for weight, role in zip(weights, roles, strict=True):
            methods = sources[role]
            mean = _eval_nullary_scalar(methods[CharacteristicName.MEAN], **options)
            variance = _eval_nullary_scalar(methods[CharacteristicName.VAR], **options)
            skewness = _eval_nullary_scalar(methods[CharacteristicName.SKEW], **options)
            raw_kurtosis = _kurt_raw_from_method(methods[CharacteristicName.KURT], **options)
            component_m1, component_m2, component_m3, component_m4 = _raw_moments_from_statistics(
                mean,
                variance,
                skewness,
                raw_kurtosis,
            )
            weight_float = float(weight)
            m1 += weight_float * component_m1
            m2 += weight_float * component_m2
            m3 += weight_float * component_m3
            m4 += weight_float * component_m4

        variance, _, mu4 = _central_moments_from_raw(m1, m2, m3, m4)
        raw = 3.0 if variance <= 0.0 else mu4 / variance**2
        return raw - 3.0 if excess else raw

    return _kurt


def _make_cf(
    distribution: FiniteMixtureDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, ComplexArray]:
    """Build mixture characteristic function."""
    roles = distribution.parent_roles
    weights = distribution.weights
    methods = [
        cast(Method[NumericArray, ComplexArray], sources[role][CharacteristicName.CF])
        for role in roles
    ]

    def _cf(data: NumericArray, **options: Any) -> ComplexArray:
        array = np.asarray(data, dtype=float)
        result = np.zeros(array.shape, dtype=complex)
        for weight, method in zip(weights, methods, strict=True):
            result += float(weight) * np.asarray(method(array, **options), dtype=complex)
        return cast(ComplexArray, result)

    return cast(ComputationFunc[NumericArray, ComplexArray], _cf)


def _make_continuous_pdf(
    distribution: FiniteMixtureDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build mixture PDF for continuous components."""
    roles = distribution.parent_roles
    weights = distribution.weights
    methods = [
        cast(Method[NumericArray, NumericArray], sources[role][CharacteristicName.PDF])
        for role in roles
    ]

    def _pdf(data: NumericArray, **options: Any) -> NumericArray:
        array = np.asarray(data, dtype=float)
        result = np.zeros(array.shape, dtype=float)
        for weight, method in zip(weights, methods, strict=True):
            result += float(weight) * np.asarray(method(array, **options), dtype=float)
        return cast(NumericArray, result)

    return cast(ComputationFunc[NumericArray, NumericArray], _pdf)


def _make_continuous_cdf(
    distribution: FiniteMixtureDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build mixture CDF for continuous components."""
    roles = distribution.parent_roles
    weights = distribution.weights
    methods = [
        cast(Method[NumericArray, NumericArray], sources[role][CharacteristicName.CDF])
        for role in roles
    ]

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        array = np.asarray(data, dtype=float)
        result = np.zeros(array.shape, dtype=float)
        for weight, method in zip(weights, methods, strict=True):
            result += float(weight) * np.asarray(method(array, **options), dtype=float)
        return cast(NumericArray, np.clip(result, 0.0, 1.0))

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


DEFAULT_FINITE_MIXTURE_COMMON_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(CharacteristicName.CF),
            _make_cf,
        )
    },
    CharacteristicName.MEAN: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(CharacteristicName.MEAN),
            _make_mean,
        )
    },
    CharacteristicName.VAR: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(
                CharacteristicName.MEAN,
                CharacteristicName.VAR,
            ),
            _make_var,
        )
    },
    CharacteristicName.SKEW: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(
                CharacteristicName.MEAN,
                CharacteristicName.VAR,
                CharacteristicName.SKEW,
            ),
            _make_skew,
        )
    },
    CharacteristicName.KURT: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(
                CharacteristicName.MEAN,
                CharacteristicName.VAR,
                CharacteristicName.SKEW,
                CharacteristicName.KURT,
            ),
            _make_kurt,
        )
    },
}

DEFAULT_FINITE_MIXTURE_CONTINUOUS_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(CharacteristicName.CDF),
            _make_continuous_cdf,
        )
    },
    CharacteristicName.PDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(CharacteristicName.PDF),
            _make_continuous_pdf,
        )
    },
}

DEFAULT_FINITE_MIXTURE_DISCRETE_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.PMF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(CharacteristicName.PMF),
            _make_discrete_pmf,
        )
    },
    CharacteristicName.CDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(CharacteristicName.PMF),
            _make_discrete_cdf,
        )
    },
    CharacteristicName.PPF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_for_all(CharacteristicName.PMF),
            _make_discrete_ppf,
        )
    },
}

DEFAULT_FINITE_MIXTURE_TRANSFORMATION_METHODS: Mapping[Kind, TransformationMethodSpecsMap] = {
    Kind.CONTINUOUS: _merge_transformation_methods(
        DEFAULT_FINITE_MIXTURE_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_FINITE_MIXTURE_CONTINUOUS_TRANSFORMATION_METHODS,
    ),
    Kind.DISCRETE: _merge_transformation_methods(
        DEFAULT_FINITE_MIXTURE_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_FINITE_MIXTURE_DISCRETE_TRANSFORMATION_METHODS,
    ),
}


def default_finite_mixture_transformation_methods(
    *,
    kind: Kind | None,
) -> TransformationMethodSpecsMap:
    """Select built-in finite mixture transformation methods by kind."""
    if kind == Kind.CONTINUOUS:
        return DEFAULT_FINITE_MIXTURE_TRANSFORMATION_METHODS[Kind.CONTINUOUS]
    if kind == Kind.DISCRETE:
        return DEFAULT_FINITE_MIXTURE_TRANSFORMATION_METHODS[Kind.DISCRETE]
    raise TypeError("Unsupported distribution kind for finite mixture.")


__all__ = [
    "DEFAULT_FINITE_MIXTURE_COMMON_TRANSFORMATION_METHODS",
    "DEFAULT_FINITE_MIXTURE_CONTINUOUS_TRANSFORMATION_METHODS",
    "DEFAULT_FINITE_MIXTURE_DISCRETE_TRANSFORMATION_METHODS",
    "DEFAULT_FINITE_MIXTURE_TRANSFORMATION_METHODS",
    "default_finite_mixture_transformation_methods",
]
