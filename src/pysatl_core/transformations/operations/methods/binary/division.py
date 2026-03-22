"""
Built-in transformation methods for division binary operations.
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
    from pysatl_core.transformations.operations.distributions.binary.division import (
        DivisionBinaryDistribution,
    )

_LEFT_ROLE: ParentRole = "left"
_RIGHT_ROLE: ParentRole = "right"
_MAX_MOMENT_ORDER = 4


def _right_inverse_moments(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
    max_order: int = _MAX_MOMENT_ORDER,
    **options: Any,
) -> tuple[float, float, float, float]:
    """Compute right-parent inverse moments ``E[Y^{-k}]`` up to ``max_order``."""
    kind = getattr(distribution.distribution_type, "kind", None)

    if kind == Kind.CONTINUOUS:
        right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
        left, right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)
        if left <= 0.0 <= right:
            raise RuntimeError(
                "Division transformation requires denominator support that does not cross zero."
            )

        def _moment(current_order: int) -> float:
            def _integrand(y: float) -> float:
                return y ** (-current_order) * _eval_method_scalar(right_pdf, y, **options)

            return _integrate_real(_integrand, left, right)

        output = [0.0, 0.0, 0.0, 0.0]
        for current_order in range(1, max_order + 1):
            output[current_order - 1] = _moment(current_order)
        return output[0], output[1], output[2], output[3]

    right_pmf = sources[_RIGHT_ROLE][CharacteristicName.PMF]
    points = distribution._discrete_points_for_role(_RIGHT_ROLE)
    if np.any(np.isclose(points, 0.0, atol=1e-14, rtol=0.0)):
        zero_mass = _eval_method_scalar(right_pmf, 0.0, **options)
        if zero_mass > 0.0:
            raise RuntimeError(
                "Division transformation is undefined when denominator has positive mass at zero."
            )

    inverse_moments = [0.0, 0.0, 0.0, 0.0]
    for y in points:
        y_float = float(y)
        if y_float == 0.0:
            continue
        py = _eval_method_scalar(right_pmf, y_float, **options)
        for current_order in range(1, max_order + 1):
            inverse_moments[current_order - 1] += py / y_float**current_order

    return cast(tuple[float, float, float, float], tuple(inverse_moments))


def _compute_division_raw_moments(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
    max_order: int = _MAX_MOMENT_ORDER,
    **options: Any,
) -> tuple[float, float, float, float]:
    """Compute raw moments up to ``max_order`` for ``X / Y``."""
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
    right_inverse = _right_inverse_moments(
        distribution,
        sources,
        max_order=max_order,
        **options,
    )
    output = [0.0, 0.0, 0.0, 0.0]
    for idx in range(max_order):
        output[idx] = left_raw[idx] * right_inverse[idx]
    return output[0], output[1], output[2], output[3]


def _make_mean(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed mean."""
    return _make_mean_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_division_raw_moments,
    )


def _make_var(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed variance."""
    return _make_var_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_division_raw_moments,
    )


def _make_skew(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed skewness."""
    return _make_skew_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_division_raw_moments,
    )


def _make_kurt(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed raw or excess kurtosis."""
    return _make_kurt_from_raw_moments(
        distribution,
        sources,
        raw_moments_evaluator=_compute_division_raw_moments,
    )


def _make_cf(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, ComplexArray]:
    """Build transformed characteristic function."""
    left_cf = sources[_LEFT_ROLE][CharacteristicName.CF]
    kind = getattr(distribution.left_distribution.distribution_type, "kind", None)

    if kind == Kind.CONTINUOUS:
        right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
        right_left, right_right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)
        if right_left <= 0.0 <= right_right:
            raise RuntimeError(
                "Characteristic function for division requires denominator support "
                "that does not cross zero."
            )

        def _cf_scalar_continuous(t: float, **options: Any) -> complex:
            def _integrand(y: float) -> complex:
                return _eval_method_scalar_complex(left_cf, t / y, **options) * _eval_method_scalar(
                    right_pdf, y, **options
                )

            return _integrate_complex(_integrand, right_left, right_right)

        def _cf_continuous(
            data: NumericArray,
            **options: Any,
        ) -> ComplexArray:
            return cast(
                ComplexArray,
                _map_scalar_complex(data, lambda t: _cf_scalar_continuous(t, **options)),
            )

        return cast(ComputationFunc[NumericArray, ComplexArray], _cf_continuous)

    right_pmf = sources[_RIGHT_ROLE][CharacteristicName.PMF]
    right_points = distribution._discrete_points_for_role(_RIGHT_ROLE)

    def _cf_scalar_discrete(t: float, **options: Any) -> complex:
        total = 0.0j
        for y in right_points:
            y_float = float(y)
            if y_float == 0.0:
                continue
            py = _eval_method_scalar(right_pmf, y_float, **options)
            total += _eval_method_scalar_complex(left_cf, t / y_float, **options) * py
        return total

    def _cf_discrete(data: NumericArray, **options: Any) -> ComplexArray:
        return cast(
            ComplexArray, _map_scalar_complex(data, lambda t: _cf_scalar_discrete(t, **options))
        )

    return cast(ComputationFunc[NumericArray, ComplexArray], _cf_discrete)


def _make_continuous_pdf(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous PDF."""
    left_pdf = sources[_LEFT_ROLE][CharacteristicName.PDF]
    right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
    right_left, right_right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)

    if right_left <= 0.0 <= right_right:
        raise RuntimeError(
            "Continuous division PDF requires denominator support that does not cross zero."
        )

    def _pdf_scalar(z: float, **options: Any) -> float:
        def _integrand(y: float) -> float:
            return (
                abs(y)
                * _eval_method_scalar(left_pdf, z * y, **options)
                * _eval_method_scalar(right_pdf, y, **options)
            )

        return _integrate_real(_integrand, right_left, right_right)

    def _pdf(data: NumericArray, **options: Any) -> NumericArray:
        return _map_scalar_real(data, lambda z: _pdf_scalar(z, **options))

    return cast(ComputationFunc[NumericArray, NumericArray], _pdf)


def _make_continuous_cdf(
    distribution: DivisionBinaryDistribution,
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

        negative = 0.0
        if negative_left < negative_right:

            def _neg_integrand(y: float) -> float:
                return (
                    1.0 - _eval_method_scalar(left_cdf, z * y, **options)
                ) * _eval_method_scalar(right_pdf, y, **options)

            negative = _integrate_real(_neg_integrand, negative_left, negative_right)

        positive = 0.0
        if positive_left < positive_right:

            def _pos_integrand(y: float) -> float:
                return _eval_method_scalar(left_cdf, z * y, **options) * _eval_method_scalar(
                    right_pdf, y, **options
                )

            positive = _integrate_real(_pos_integrand, positive_left, positive_right)

        return float(np.clip(negative + positive, 0.0, 1.0))

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        return _map_scalar_real(data, lambda z: _cdf_scalar(z, **options))

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


def _make_continuous_ppf_for_division(
    distribution: DivisionBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous PPF for division operation."""
    return _make_continuous_ppf(
        distribution,
        sources,
        cdf_evaluator=_make_continuous_cdf,
    )


DEFAULT_BINARY_DIVISION_CONTINUOUS_COMMON_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.CF,),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_cf,
        )
    },
    CharacteristicName.MEAN: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.MEAN,),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_mean,
        )
    },
    CharacteristicName.VAR: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.MEAN, CharacteristicName.VAR),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_var,
        )
    },
    CharacteristicName.SKEW: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (
                    CharacteristicName.MEAN,
                    CharacteristicName.VAR,
                    CharacteristicName.SKEW,
                ),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_skew,
        )
    },
    CharacteristicName.KURT: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (
                    CharacteristicName.MEAN,
                    CharacteristicName.VAR,
                    CharacteristicName.SKEW,
                    CharacteristicName.KURT,
                ),
                _RIGHT_ROLE: (CharacteristicName.PDF,),
            },
            _make_kurt,
        )
    },
}

DEFAULT_BINARY_DIVISION_CONTINUOUS_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
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
            _make_continuous_ppf_for_division,
        )
    },
}

DEFAULT_BINARY_DIVISION_DISCRETE_COMMON_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.CF,),
                _RIGHT_ROLE: (CharacteristicName.PMF,),
            },
            _make_cf,
        )
    },
    CharacteristicName.MEAN: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.MEAN,),
                _RIGHT_ROLE: (CharacteristicName.PMF,),
            },
            _make_mean,
        )
    },
    CharacteristicName.VAR: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (CharacteristicName.MEAN, CharacteristicName.VAR),
                _RIGHT_ROLE: (CharacteristicName.PMF,),
            },
            _make_var,
        )
    },
    CharacteristicName.SKEW: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (
                    CharacteristicName.MEAN,
                    CharacteristicName.VAR,
                    CharacteristicName.SKEW,
                ),
                _RIGHT_ROLE: (CharacteristicName.PMF,),
            },
            _make_skew,
        )
    },
    CharacteristicName.KURT: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            {
                _LEFT_ROLE: (
                    CharacteristicName.MEAN,
                    CharacteristicName.VAR,
                    CharacteristicName.SKEW,
                    CharacteristicName.KURT,
                ),
                _RIGHT_ROLE: (CharacteristicName.PMF,),
            },
            _make_kurt,
        )
    },
}

DEFAULT_BINARY_DIVISION_DISCRETE_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
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

DEFAULT_BINARY_DIVISION_TRANSFORMATION_METHODS: dict[Kind, TransformationMethodSpecsMap] = {
    Kind.CONTINUOUS: _merge_transformation_methods(
        DEFAULT_BINARY_DIVISION_CONTINUOUS_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_BINARY_DIVISION_CONTINUOUS_TRANSFORMATION_METHODS,
    ),
    Kind.DISCRETE: _merge_transformation_methods(
        DEFAULT_BINARY_DIVISION_DISCRETE_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_BINARY_DIVISION_DISCRETE_TRANSFORMATION_METHODS,
    ),
}


def default_division_binary_transformation_methods(
    *,
    kind: Kind | None,
) -> TransformationMethodSpecsMap:
    """Select built-in division binary transformation methods by kind."""
    if kind == Kind.CONTINUOUS:
        return DEFAULT_BINARY_DIVISION_TRANSFORMATION_METHODS[Kind.CONTINUOUS]
    if kind == Kind.DISCRETE:
        return DEFAULT_BINARY_DIVISION_TRANSFORMATION_METHODS[Kind.DISCRETE]
    raise TypeError("Unsupported distribution kind for division binary transformation.")


__all__ = [
    "DEFAULT_BINARY_DIVISION_CONTINUOUS_COMMON_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_DIVISION_CONTINUOUS_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_DIVISION_DISCRETE_COMMON_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_DIVISION_DISCRETE_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_DIVISION_TRANSFORMATION_METHODS",
    "default_division_binary_transformation_methods",
]
