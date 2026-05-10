"""
Built-in transformation methods for multiplicative binary operations.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.transformations.operations.methods._utils import (
    _eval_method_scalar,
    _eval_method_scalar_complex,
    _integrate_complex,
    _integrate_real,
    _make_discrete_cdf_from_mass_table as _make_discrete_cdf,
    _make_discrete_pmf_from_mass_table as _make_discrete_pmf,
    _make_discrete_ppf_from_mass_table as _make_discrete_ppf,
    _map_scalar_complex,
    _map_scalar_real,
    _merge_transformation_methods,
    _parent_raw_moments,
)
from pysatl_core.transformations.operations.methods.binary.base import (
    _make_continuous_ppf,
    _make_kurt_from_raw_moments,
    _make_mean_from_raw_moments,
    _make_skew_from_raw_moments,
    _make_var_from_raw_moments,
    _requirements_both,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    CharacteristicName,
    ComplexArray,
    ComputationFunc,
    Kind,
    NumericArray,
    ParentRole,
    ResolvedSourceMethods,
    TransformationMethodSpecsMap,
)

if TYPE_CHECKING:
    from pysatl_core.transformations.operations.distributions.binary.multiplication import (
        MultiplicationBinaryDistribution,
    )

_LEFT_ROLE: ParentRole = "left"
_RIGHT_ROLE: ParentRole = "right"
_MAX_MOMENT_ORDER = 4


def _compute_multiplication_raw_moments(
    _distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
    max_order: int = _MAX_MOMENT_ORDER,
    **options: Any,
) -> tuple[float, float, float, float]:
    """Compute raw moments up to ``max_order`` for ``X * Y``."""
    left_raw = _parent_raw_moments(
        sources,
        _LEFT_ROLE,
        mean_name=CharacteristicName.MEAN,
        var_name=CharacteristicName.VAR,
        skew_name=CharacteristicName.SKEW,
        kurt_name=CharacteristicName.KURT,
        max_order=max_order,
        **options,
    )
    right_raw = _parent_raw_moments(
        sources,
        _RIGHT_ROLE,
        mean_name=CharacteristicName.MEAN,
        var_name=CharacteristicName.VAR,
        skew_name=CharacteristicName.SKEW,
        kurt_name=CharacteristicName.KURT,
        max_order=max_order,
        **options,
    )

    output = [0.0, 0.0, 0.0, 0.0]
    for idx in range(max_order):
        output[idx] = left_raw[idx] * right_raw[idx]
    return output[0], output[1], output[2], output[3]


def _make_mean(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed mean."""
    return _make_mean_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_multiplication_raw_moments,
    )


def _make_var(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed variance."""
    return _make_var_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_multiplication_raw_moments,
    )


def _make_skew(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed skewness."""
    return _make_skew_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_multiplication_raw_moments,
    )


def _make_kurt(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed raw or excess kurtosis."""
    return _make_kurt_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_multiplication_raw_moments,
    )


def _make_cf(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, ComplexArray]:
    """Build transformed characteristic function."""
    left_cf = sources[_LEFT_ROLE][CharacteristicName.CF]
    kind = getattr(distribution.left_distribution.distribution_type, "kind", None)

    if kind == Kind.CONTINUOUS:
        right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
        right_left, right_right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)

        def _cf_scalar_continuous(t: float, **options: Any) -> complex:
            def _integrand(y: float) -> complex:
                return _eval_method_scalar_complex(left_cf, t * y, **options) * _eval_method_scalar(
                    right_pdf, y, **options
                )

            return _integrate_complex(_integrand, right_left, right_right)

        def _cf_continuous(
            data: NumericArray,
            **options: Any,
        ) -> ComplexArray:
            return _map_scalar_complex(data, lambda t: _cf_scalar_continuous(t, **options))

        return cast(ComputationFunc[NumericArray, ComplexArray], _cf_continuous)

    right_pmf = sources[_RIGHT_ROLE][CharacteristicName.PMF]
    right_points = distribution._discrete_points_for_role(_RIGHT_ROLE)

    def _cf_scalar_discrete(t: float, **options: Any) -> complex:
        total = 0.0j
        for y in right_points:
            y_float = float(y)
            py = _eval_method_scalar(right_pmf, y_float, **options)
            total += _eval_method_scalar_complex(left_cf, t * y_float, **options) * py
        return total

    def _cf_discrete(data: NumericArray, **options: Any) -> ComplexArray:
        return _map_scalar_complex(data, lambda t: _cf_scalar_discrete(t, **options))

    return cast(ComputationFunc[NumericArray, ComplexArray], _cf_discrete)


def _make_continuous_pdf(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous PDF."""
    left_pdf = sources[_LEFT_ROLE][CharacteristicName.PDF]
    right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
    right_left, right_right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)

    def _pdf_scalar(z: float, **options: Any) -> float:
        def _integrand(y: float) -> float:
            return (
                _eval_method_scalar(left_pdf, z / y, **options)
                * _eval_method_scalar(right_pdf, y, **options)
                / abs(y)
            )

        return _integrate_real(_integrand, right_left, right_right, split_at_zero=True)

    def _pdf(data: NumericArray, **options: Any) -> NumericArray:
        return _map_scalar_real(data, lambda z: _pdf_scalar(z, **options))

    return cast(ComputationFunc[NumericArray, NumericArray], _pdf)


def _make_continuous_cdf(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous CDF."""
    left_cdf = sources[_LEFT_ROLE][CharacteristicName.CDF]
    right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
    right_left, right_right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)

    def _cdf_scalar(z: float, **options: Any) -> float:
        negative_left = right_left
        negative_right = min(right_right, 0.0)
        positive_left = max(right_left, 0.0)
        positive_right = right_right

        if negative_right == 0.0:
            negative_right = float(np.nextafter(0.0, -1.0))
        if positive_left == 0.0:
            positive_left = float(np.nextafter(0.0, 1.0))

        negative = 0.0
        if negative_left < negative_right:

            def _neg_integrand(y: float) -> float:
                return (
                    1.0 - _eval_method_scalar(left_cdf, z / y, **options)
                ) * _eval_method_scalar(right_pdf, y, **options)

            negative = _integrate_real(_neg_integrand, negative_left, negative_right)

        positive = 0.0
        if positive_left < positive_right:

            def _pos_integrand(y: float) -> float:
                return _eval_method_scalar(left_cdf, z / y, **options) * _eval_method_scalar(
                    right_pdf, y, **options
                )

            positive = _integrate_real(_pos_integrand, positive_left, positive_right)

        return float(np.clip(negative + positive, 0.0, 1.0))

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        return _map_scalar_real(data, lambda z: _cdf_scalar(z, **options))

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


def _make_continuous_ppf_for_multiplication(
    distribution: MultiplicationBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous PPF for multiplication operation."""
    return _make_continuous_ppf(
        distribution,
        sources,
        cdf_evaluator=_make_continuous_cdf,
    )


DEFAULT_BINARY_MULTIPLICATION_COMMON_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.MEAN: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.MEAN),
            _make_mean,
        )
    },
    CharacteristicName.VAR: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.MEAN, CharacteristicName.VAR),
            _make_var,
        )
    },
    CharacteristicName.SKEW: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(
                CharacteristicName.MEAN,
                CharacteristicName.VAR,
                CharacteristicName.SKEW,
            ),
            _make_skew,
        )
    },
    CharacteristicName.KURT: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(
                CharacteristicName.MEAN,
                CharacteristicName.VAR,
                CharacteristicName.SKEW,
                CharacteristicName.KURT,
            ),
            _make_kurt,
        )
    },
}

DEFAULT_BINARY_MULTIPLICATION_CONTINUOUS_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.CF,),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_cf,
        )
    },
    CharacteristicName.CDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.CDF,),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_continuous_cdf,
        )
    },
    CharacteristicName.PDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.PDF),
            _make_continuous_pdf,
        )
    },
    CharacteristicName.PPF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.CDF,),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_continuous_ppf_for_multiplication,
        )
    },
}

DEFAULT_BINARY_MULTIPLICATION_DISCRETE_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.CF,),
                _RIGHT_ROLE: (CharacteristicName.PMF,),
            },
            _make_cf,
        )
    },
    CharacteristicName.PMF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.PMF),
            _make_discrete_pmf,
        )
    },
    CharacteristicName.CDF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.PMF),
            _make_discrete_cdf,
        )
    },
    CharacteristicName.PPF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.PMF),
            _make_discrete_ppf,
        )
    },
}

DEFAULT_BINARY_MULTIPLICATION_TRANSFORMATION_METHODS: dict[Kind, TransformationMethodSpecsMap] = {
    Kind.CONTINUOUS: _merge_transformation_methods(
        DEFAULT_BINARY_MULTIPLICATION_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_BINARY_MULTIPLICATION_CONTINUOUS_TRANSFORMATION_METHODS,
    ),
    Kind.DISCRETE: _merge_transformation_methods(
        DEFAULT_BINARY_MULTIPLICATION_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_BINARY_MULTIPLICATION_DISCRETE_TRANSFORMATION_METHODS,
    ),
}


def default_multiplication_binary_transformation_methods(
    *,
    kind: Kind | None,
) -> TransformationMethodSpecsMap:
    """Select built-in multiplication binary transformation methods by kind."""
    if kind == Kind.CONTINUOUS:
        return DEFAULT_BINARY_MULTIPLICATION_TRANSFORMATION_METHODS[Kind.CONTINUOUS]
    if kind == Kind.DISCRETE:
        return DEFAULT_BINARY_MULTIPLICATION_TRANSFORMATION_METHODS[Kind.DISCRETE]
    raise TypeError("Unsupported distribution kind for multiplication binary transformation.")


__all__ = [
    "DEFAULT_BINARY_MULTIPLICATION_COMMON_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_MULTIPLICATION_CONTINUOUS_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_MULTIPLICATION_DISCRETE_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_MULTIPLICATION_TRANSFORMATION_METHODS",
    "default_multiplication_binary_transformation_methods",
]
