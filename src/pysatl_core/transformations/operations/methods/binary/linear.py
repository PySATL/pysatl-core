"""
Built-in transformation methods for linear binary operations.
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
    _eval_nullary_scalar,
    _integrate_real,
    _kurt_raw_from_method,
    _make_discrete_cdf_from_mass_table as _make_discrete_cdf,
    _make_discrete_pmf_from_mass_table as _make_discrete_pmf,
    _make_discrete_ppf_from_mass_table as _make_discrete_ppf,
    _map_scalar_complex,
    _map_scalar_real,
    _merge_transformation_methods,
)
from pysatl_core.transformations.operations.methods.binary.base import (
    _make_continuous_ppf,
    _requirements_both,
)
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    BinaryOperationName,
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
    from pysatl_core.transformations.operations.distributions.binary.linear import (
        LinearBinaryDistribution,
    )

_LEFT_ROLE: ParentRole = "left"
_RIGHT_ROLE: ParentRole = "right"


def _make_mean(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed mean."""
    left_mean = sources[_LEFT_ROLE][CharacteristicName.MEAN]
    right_mean = sources[_RIGHT_ROLE][CharacteristicName.MEAN]
    sign = 1.0 if distribution.operation == BinaryOperationName.ADD else -1.0

    def _mean(**options: Any) -> float:
        mean_left = _eval_nullary_scalar(left_mean, **options)
        mean_right = _eval_nullary_scalar(right_mean, **options)
        return mean_left + sign * mean_right

    return _mean


def _make_var(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed variance."""
    _ = distribution
    left_var = sources[_LEFT_ROLE][CharacteristicName.VAR]
    right_var = sources[_RIGHT_ROLE][CharacteristicName.VAR]

    def _var(**options: Any) -> float:
        variance_left = max(_eval_nullary_scalar(left_var, **options), 0.0)
        variance_right = max(_eval_nullary_scalar(right_var, **options), 0.0)
        return variance_left + variance_right

    return _var


def _make_skew(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed skewness."""
    left_var = sources[_LEFT_ROLE][CharacteristicName.VAR]
    right_var = sources[_RIGHT_ROLE][CharacteristicName.VAR]
    left_skew = sources[_LEFT_ROLE][CharacteristicName.SKEW]
    right_skew = sources[_RIGHT_ROLE][CharacteristicName.SKEW]
    sign = 1.0 if distribution.operation == BinaryOperationName.ADD else -1.0

    def _skew(**options: Any) -> float:
        variance_left = max(_eval_nullary_scalar(left_var, **options), 0.0)
        variance_right = max(_eval_nullary_scalar(right_var, **options), 0.0)
        variance = variance_left + variance_right
        if variance <= 0.0:
            return 0.0

        skew_left = _eval_nullary_scalar(left_skew, **options)
        skew_right = _eval_nullary_scalar(right_skew, **options)
        mu3_left = skew_left * variance_left**1.5
        mu3_right = skew_right * variance_right**1.5
        mu3 = mu3_left + sign * mu3_right
        return float(mu3 / variance**1.5)

    return _skew


def _make_kurt(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[Any, float]:
    """Build transformed raw or excess kurtosis."""
    _ = distribution
    left_var = sources[_LEFT_ROLE][CharacteristicName.VAR]
    right_var = sources[_RIGHT_ROLE][CharacteristicName.VAR]
    left_kurt = sources[_LEFT_ROLE][CharacteristicName.KURT]
    right_kurt = sources[_RIGHT_ROLE][CharacteristicName.KURT]

    def _kurt(*, excess: bool = False, **options: Any) -> float:
        variance_left = max(_eval_nullary_scalar(left_var, **options), 0.0)
        variance_right = max(_eval_nullary_scalar(right_var, **options), 0.0)
        variance = variance_left + variance_right
        if variance <= 0.0:
            return 0.0 if excess else 3.0

        kurt_left_raw = _kurt_raw_from_method(left_kurt, **options)
        kurt_right_raw = _kurt_raw_from_method(right_kurt, **options)
        mu4_left = kurt_left_raw * variance_left**2
        mu4_right = kurt_right_raw * variance_right**2
        mu4 = mu4_left + mu4_right + 6.0 * variance_left * variance_right
        raw = float(mu4 / variance**2)
        return raw - 3.0 if excess else raw

    return _kurt


def _make_cf(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, ComplexArray]:
    """Build transformed characteristic function."""
    left_cf = sources[_LEFT_ROLE][CharacteristicName.CF]
    right_cf = sources[_RIGHT_ROLE][CharacteristicName.CF]

    def _cf_scalar(t: float, **options: Any) -> complex:
        left_value = _eval_method_scalar_complex(left_cf, t, **options)
        right_arg = t if distribution.operation == BinaryOperationName.ADD else -t
        right_value = _eval_method_scalar_complex(right_cf, right_arg, **options)
        return left_value * right_value

    def _cf(data: NumericArray, **options: Any) -> ComplexArray:
        return _map_scalar_complex(data, lambda t: _cf_scalar(t, **options))

    return cast(ComputationFunc[NumericArray, ComplexArray], _cf)


def _make_continuous_pdf(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous PDF."""
    left_pdf = sources[_LEFT_ROLE][CharacteristicName.PDF]
    right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
    right_left, right_right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)

    def _pdf_scalar(z: float, **options: Any) -> float:
        if distribution.operation == BinaryOperationName.ADD:

            def _integrand_add(y: float) -> float:
                return _eval_method_scalar(left_pdf, z - y, **options) * _eval_method_scalar(
                    right_pdf, y, **options
                )

            return _integrate_real(_integrand_add, right_left, right_right)

        def _integrand_sub(y: float) -> float:
            return _eval_method_scalar(left_pdf, z + y, **options) * _eval_method_scalar(
                right_pdf, y, **options
            )

        return _integrate_real(_integrand_sub, right_left, right_right)

    def _pdf(data: NumericArray, **options: Any) -> NumericArray:
        return _map_scalar_real(data, lambda z: _pdf_scalar(z, **options))

    return cast(ComputationFunc[NumericArray, NumericArray], _pdf)


def _make_continuous_cdf(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous CDF."""
    left_cdf = sources[_LEFT_ROLE][CharacteristicName.CDF]
    right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
    right_left, right_right = distribution._continuous_bounds_for_role(_RIGHT_ROLE)

    def _cdf_scalar(z: float, **options: Any) -> float:
        if distribution.operation == BinaryOperationName.ADD:

            def _integrand_add(y: float) -> float:
                return _eval_method_scalar(left_cdf, z - y, **options) * _eval_method_scalar(
                    right_pdf, y, **options
                )

            value = _integrate_real(_integrand_add, right_left, right_right)
            return float(np.clip(value, 0.0, 1.0))

        def _integrand_sub(y: float) -> float:
            return _eval_method_scalar(left_cdf, z + y, **options) * _eval_method_scalar(
                right_pdf, y, **options
            )

        value = _integrate_real(_integrand_sub, right_left, right_right)
        return float(np.clip(value, 0.0, 1.0))

    def _cdf(data: NumericArray, **options: Any) -> NumericArray:
        return _map_scalar_real(data, lambda z: _cdf_scalar(z, **options))

    return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


def _make_continuous_ppf_for_linear(
    distribution: LinearBinaryDistribution,
    sources: ResolvedSourceMethods,
) -> ComputationFunc[NumericArray, NumericArray]:
    """Build transformed continuous PPF for linear binary operations."""
    return _make_continuous_ppf(
        distribution,
        sources,
        cdf_evaluator=_make_continuous_cdf,
    )


DEFAULT_BINARY_LINEAR_COMMON_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
    CharacteristicName.CF: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.CF),
            _make_cf,
        )
    },
    CharacteristicName.MEAN: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.MEAN),
            _make_mean,
        )
    },
    CharacteristicName.VAR: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(CharacteristicName.VAR),
            _make_var,
        )
    },
    CharacteristicName.SKEW: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(
                CharacteristicName.VAR,
                CharacteristicName.SKEW,
            ),
            _make_skew,
        )
    },
    CharacteristicName.KURT: {
        DEFAULT_ANALYTICAL_COMPUTATION_LABEL: (
            _requirements_both(
                CharacteristicName.VAR,
                CharacteristicName.KURT,
            ),
            _make_kurt,
        )
    },
}

DEFAULT_BINARY_LINEAR_CONTINUOUS_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
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
            _make_continuous_ppf_for_linear,
        )
    },
}

DEFAULT_BINARY_LINEAR_DISCRETE_TRANSFORMATION_METHODS: TransformationMethodSpecsMap = {
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

DEFAULT_BINARY_LINEAR_TRANSFORMATION_METHODS: dict[Kind, TransformationMethodSpecsMap] = {
    Kind.CONTINUOUS: _merge_transformation_methods(
        DEFAULT_BINARY_LINEAR_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_BINARY_LINEAR_CONTINUOUS_TRANSFORMATION_METHODS,
    ),
    Kind.DISCRETE: _merge_transformation_methods(
        DEFAULT_BINARY_LINEAR_COMMON_TRANSFORMATION_METHODS,
        DEFAULT_BINARY_LINEAR_DISCRETE_TRANSFORMATION_METHODS,
    ),
}


def default_linear_binary_transformation_methods(
    *,
    kind: Kind | None,
) -> TransformationMethodSpecsMap:
    """Select built-in linear binary transformation methods by kind."""
    if kind == Kind.CONTINUOUS:
        return DEFAULT_BINARY_LINEAR_TRANSFORMATION_METHODS[Kind.CONTINUOUS]
    if kind == Kind.DISCRETE:
        return DEFAULT_BINARY_LINEAR_TRANSFORMATION_METHODS[Kind.DISCRETE]
    raise TypeError("Unsupported distribution kind for linear binary transformation.")


__all__ = [
    "DEFAULT_BINARY_LINEAR_COMMON_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_LINEAR_CONTINUOUS_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_LINEAR_DISCRETE_TRANSFORMATION_METHODS",
    "DEFAULT_BINARY_LINEAR_TRANSFORMATION_METHODS",
    "default_linear_binary_transformation_methods",
]
