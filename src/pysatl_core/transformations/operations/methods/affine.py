"""
Built-in transformation methods for affine operations.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.computation import Method
from pysatl_core.transformations.operations.methods._utils import (
    _merge_transformation_methods,
    _source_requirements_for_roles,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    CharacteristicName,
    ComplexArray,
    ComputationFunc,
    GenericCharacteristicName,
    Kind,
    NumericArray,
    ParentRole,
    ResolvedSourceMethods,
    SourceRequirements,
    TransformationMethodSpecsMap,
)

if TYPE_CHECKING:
    from pysatl_core.transformations.operations.distributions.affine import AffineDistribution

_BASE_ROLE: ParentRole = "base"


def _base_requirements(*characteristics: GenericCharacteristicName) -> SourceRequirements:
    """Build source requirements for the affine base role."""
    return _source_requirements_for_roles((_BASE_ROLE,), *characteristics)


def _make_continuous_cdf(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed CDF for a continuous base distribution."""
    base_cdf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.CDF])

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        values = base_cdf((data - distribution.shift) / distribution.scale, **options)
        return 1.0 - values if distribution.scale < 0.0 else values

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


def _make_continuous_pdf(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed PDF for a continuous base distribution."""
    base_pdf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PDF])

    def _pdf(data: NumericArray, **options: Any) -> NumericArray:
        return cast(
            NumericArray,
            base_pdf((data - distribution.shift) / distribution.scale, **options)
            / abs(distribution.scale),
        )

    return cast(ComputationFunc[NumericArray, NumericArray], _pdf)


def _make_continuous_ppf(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed PPF for a continuous base distribution."""
    base_ppf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PPF])

    def _ppf(data: NumericArray, **options: Any) -> NumericArray:
        probabilities = data if distribution.scale > 0.0 else 1.0 - data
        return distribution.scale * base_ppf(probabilities, **options) + distribution.shift

    return cast(ComputationFunc[NumericArray, NumericArray], _ppf)


def _make_discrete_cdf(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed CDF for a discrete base distribution with positive scale."""
    base_cdf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.CDF])

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        return base_cdf((data - distribution.shift) / distribution.scale, **options)

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


def _make_discrete_cdf_negative_scale(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed CDF for a discrete base distribution with negative scale."""
    base_cdf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.CDF])
    base_pmf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PMF])

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        x = (data - distribution.shift) / distribution.scale
        return np.asarray(1.0 - base_cdf(x, **options) + base_pmf(x, **options))

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


def _make_discrete_pmf(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed PMF for a discrete base distribution."""
    base_pmf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PMF])

    def _pmf(data: NumericArray, **options: Any) -> NumericArray:
        return base_pmf((data - distribution.shift) / distribution.scale, **options)

    return cast(ComputationFunc[NumericArray, NumericArray], _pmf)


def _make_discrete_ppf(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed PPF for a discrete base distribution."""
    base_ppf = cast(Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.PPF])

    def _ppf(data: NumericArray, **options: Any) -> NumericArray:
        x = data if distribution.scale > 0.0 else np.nextafter(1.0 - data, 1.0)
        return distribution.scale * base_ppf(x, **options) + distribution.shift

    return cast(ComputationFunc[NumericArray, NumericArray], _ppf)


def _make_cf(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, ComplexArray]:
    """Build characteristic function for affine transformation."""
    base_cf = cast(Method[NumericArray, ComplexArray], sources[_BASE_ROLE][CharacteristicName.CF])

    def _cf(data: NumericArray, **options: Any) -> ComplexArray:
        return cast(
            ComplexArray,
            np.exp(1j * distribution.shift * data) * base_cf(distribution.scale * data, **options),
        )

    return cast(ComputationFunc[NumericArray, ComplexArray], _cf)


def _make_mean(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed mean."""
    base_mean = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.MEAN])

    def _mean(**options: Any) -> float:
        return distribution.scale * base_mean(**options) + distribution.shift

    return _mean


def _make_var(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed variance."""
    base_var = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.VAR])

    def _var(**options: Any) -> float:
        return distribution.scale**2 * base_var(**options)

    return _var


def _make_skew(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed skewness."""
    base_skew = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.SKEW])

    def _skew(**options: Any) -> float:
        sign = -1.0 if distribution.scale < 0.0 else 1.0
        return sign * base_skew(**options)

    return _skew


def _make_kurt(
    distribution: AffineDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed kurtosis."""
    base_kurt = cast(Method[Any, float], sources[_BASE_ROLE][CharacteristicName.KURT])

    def _kurt(**options: Any) -> float:
        return base_kurt(**options)

    return _kurt


DEFAULT_AFFINE_COMMON_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.CF),
            _make_cf,
        )
    },
    CharacteristicName.MEAN: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.MEAN),
            _make_mean,
        )
    },
    CharacteristicName.VAR: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.VAR),
            _make_var,
        )
    },
    CharacteristicName.SKEW: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.SKEW),
            _make_skew,
        )
    },
    CharacteristicName.KURT: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.KURT),
            _make_kurt,
        )
    },
}

DEFAULT_AFFINE_CONTINUOUS_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.CDF),
            _make_continuous_cdf,
        )
    },
    CharacteristicName.PDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.PDF),
            _make_continuous_pdf,
        )
    },
    CharacteristicName.PPF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.PPF),
            _make_continuous_ppf,
        )
    },
}

DEFAULT_AFFINE_DISCRETE_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.PMF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.PMF),
            _make_discrete_pmf,
        )
    },
    CharacteristicName.PPF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.PPF),
            _make_discrete_ppf,
        )
    },
}

DEFAULT_AFFINE_DISCRETE_POSITIVE_SCALE_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.CDF),
            _make_discrete_cdf,
        )
    },
}

DEFAULT_AFFINE_DISCRETE_NEGATIVE_SCALE_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _base_requirements(CharacteristicName.CDF, CharacteristicName.PMF),
            _make_discrete_cdf_negative_scale,
        )
    },
}


def default_affine_transformation_methods(
    *,
    kind: Kind | None,
    scale: float,
) -> TransformationMethodSpecsMap:
    """Select built-in affine transformation methods for the given distribution kind."""
    if kind == Kind.CONTINUOUS:
        return _merge_transformation_methods(
            DEFAULT_AFFINE_COMMON_TRANSFORMATION_METHODS,
            DEFAULT_AFFINE_CONTINUOUS_TRANSFORMATION_METHODS,
        )

    if kind == Kind.DISCRETE:
        discrete_cdf_methods = (
            DEFAULT_AFFINE_DISCRETE_POSITIVE_SCALE_TRANSFORMATION_METHODS
            if scale > 0.0
            else DEFAULT_AFFINE_DISCRETE_NEGATIVE_SCALE_TRANSFORMATION_METHODS
        )
        return _merge_transformation_methods(
            DEFAULT_AFFINE_COMMON_TRANSFORMATION_METHODS,
            DEFAULT_AFFINE_DISCRETE_TRANSFORMATION_METHODS,
            discrete_cdf_methods,
        )

    raise TypeError("Unsupported distribution kind for affine transformation.")


__all__ = [
    "DEFAULT_AFFINE_COMMON_TRANSFORMATION_METHODS",
    "DEFAULT_AFFINE_CONTINUOUS_TRANSFORMATION_METHODS",
    "DEFAULT_AFFINE_DISCRETE_NEGATIVE_SCALE_TRANSFORMATION_METHODS",
    "DEFAULT_AFFINE_DISCRETE_POSITIVE_SCALE_TRANSFORMATION_METHODS",
    "DEFAULT_AFFINE_DISCRETE_TRANSFORMATION_METHODS",
    "default_affine_transformation_methods",
]
