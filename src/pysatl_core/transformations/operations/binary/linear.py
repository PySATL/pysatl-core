"""
Linear binary transformations: addition and subtraction.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from math import comb
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.distribution import _KEEP
from pysatl_core.distributions.support import ContinuousSupport, Support
from pysatl_core.transformations.transformation_method import (
    ResolvedSourceMethods,
    SourceRequirements,
    TransformationEvaluator,
)
from pysatl_core.types import (
    BinaryOperationName,
    CharacteristicName,
    ComplexArray,
    ComputationFunc,
    GenericCharacteristicName,
    Kind,
    NumericArray,
)

from .base import (
    _LEFT_ROLE,
    _RIGHT_ROLE,
    BinaryDistribution,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.strategies import (
        ComputationStrategy,
        SamplingStrategy,
    )

_SUPPORTED_LINEAR_OPERATIONS: frozenset[BinaryOperationName] = frozenset(
    {BinaryOperationName.ADD, BinaryOperationName.SUB}
)


class LinearBinaryDistribution(BinaryDistribution):
    """
    Binary distribution for linear operations ``X + Y`` and ``X - Y``.
    """

    def __init__(
        self,
        left_distribution: Distribution,
        right_distribution: Distribution,
        *,
        operation: BinaryOperationName,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        if operation not in _SUPPORTED_LINEAR_OPERATIONS:
            raise ValueError(
                f"Unsupported linear operation '{operation}'. "
                f"Supported operations: {', '.join(sorted(_SUPPORTED_LINEAR_OPERATIONS))}."
            )
        super().__init__(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            operation=operation,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
        )

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> LinearBinaryDistribution:
        """Return a copy of the linear binary distribution with updated strategies."""
        return LinearBinaryDistribution(
            left_distribution=self.left_distribution,
            right_distribution=self.right_distribution,
            operation=self.operation,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

    def _characteristic_specs(
        self, *, kind: Kind | None
    ) -> tuple[
        tuple[GenericCharacteristicName, SourceRequirements, TransformationEvaluator[Any, Any]],
        ...,
    ]:
        """Return characteristic registration specs for linear transformations."""
        specs: list[
            tuple[GenericCharacteristicName, SourceRequirements, TransformationEvaluator[Any, Any]]
        ] = [
            (
                CharacteristicName.CF,
                self._requirements_both(CharacteristicName.CF),
                self._make_cf,
            ),
            (
                CharacteristicName.MEAN,
                self._requirements_both(*self._statistics_for_moment_order(1)),
                self._make_mean,
            ),
            (
                CharacteristicName.VAR,
                self._requirements_both(*self._statistics_for_moment_order(2)),
                self._make_var,
            ),
            (
                CharacteristicName.SKEW,
                self._requirements_both(*self._statistics_for_moment_order(3)),
                self._make_skew,
            ),
            (
                CharacteristicName.KURT,
                self._requirements_both(*self._statistics_for_moment_order(4)),
                self._make_kurt,
            ),
        ]

        if kind == Kind.CONTINUOUS:
            specs.extend(
                [
                    (
                        CharacteristicName.CDF,
                        {
                            _LEFT_ROLE: (CharacteristicName.CDF,),
                            _RIGHT_ROLE: (CharacteristicName.PDF,),
                        },
                        self._make_continuous_cdf,
                    ),
                    (
                        CharacteristicName.PDF,
                        self._requirements_both(CharacteristicName.PDF),
                        self._make_continuous_pdf,
                    ),
                    (
                        CharacteristicName.PPF,
                        {
                            _LEFT_ROLE: (CharacteristicName.CDF,),
                            _RIGHT_ROLE: (CharacteristicName.PDF,),
                        },
                        self._make_continuous_ppf,
                    ),
                ]
            )
        else:
            discrete_requirements = self._requirements_both(CharacteristicName.PMF)
            specs.extend(
                [
                    (CharacteristicName.PMF, discrete_requirements, self._make_discrete_pmf),
                    (CharacteristicName.CDF, discrete_requirements, self._make_discrete_cdf),
                    (CharacteristicName.PPF, discrete_requirements, self._make_discrete_ppf),
                ]
            )

        return tuple(specs)

    def _operation_value(
        self,
        left: float | NumericArray,
        right: float | NumericArray,
    ) -> float | NumericArray:
        """Apply linear operation in scalar or array semantics."""
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if self.operation == BinaryOperationName.ADD:
            result = left_array + right_array
        else:
            result = left_array - right_array

        if result.ndim == 0:
            return float(result)
        return cast(NumericArray, result)

    def _result_raw_moments(
        self,
        sources: ResolvedSourceMethods,
        *,
        max_order: int = 4,
        **options: Any,
    ) -> tuple[float, float, float, float]:
        """Compute raw moments up to ``max_order`` for ``X ± Y``."""
        order = self._validate_moment_order(max_order)
        left_raw = self._parent_raw_moments(
            sources,
            _LEFT_ROLE,
            max_order=order,
            **options,
        )
        right_raw = self._parent_raw_moments(
            sources,
            _RIGHT_ROLE,
            max_order=order,
            **options,
        )
        sign = 1.0 if self.operation == BinaryOperationName.ADD else -1.0

        left_poly = (1.0, *left_raw)
        right_poly = (
            1.0,
            sign * right_raw[0],
            right_raw[1],
            sign * right_raw[2],
            right_raw[3],
        )

        output = [0.0, 0.0, 0.0, 0.0]
        for current_order in range(1, order + 1):
            moment = 0.0
            for i in range(current_order + 1):
                moment += comb(current_order, i) * left_poly[i] * right_poly[current_order - i]
            output[current_order - 1] = moment
        return output[0], output[1], output[2], output[3]

    def _make_cf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, ComplexArray]:
        """Build transformed characteristic function."""
        left_cf = sources[_LEFT_ROLE][CharacteristicName.CF]
        right_cf = sources[_RIGHT_ROLE][CharacteristicName.CF]

        def _cf_scalar(t: float, **options: Any) -> complex:
            left_value = self._eval_method_scalar_complex(left_cf, t, **options)
            right_arg = t if self.operation == BinaryOperationName.ADD else -t
            right_value = self._eval_method_scalar_complex(right_cf, right_arg, **options)
            return left_value * right_value

        def _cf(data: NumericArray, **options: Any) -> ComplexArray:
            return self._map_scalar_complex(
                data,
                lambda t: _cf_scalar(t, **options),
            )

        return cast(ComputationFunc[NumericArray, ComplexArray], _cf)

    def _make_continuous_pdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build transformed continuous PDF."""
        left_pdf = sources[_LEFT_ROLE][CharacteristicName.PDF]
        right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
        right_left, right_right = self._continuous_bounds_for_role(_RIGHT_ROLE)

        def _pdf_scalar(z: float, **options: Any) -> float:
            if self.operation == BinaryOperationName.ADD:

                def _integrand_add(y: float) -> float:
                    return self._eval_method_scalar(
                        left_pdf, z - y, **options
                    ) * self._eval_method_scalar(right_pdf, y, **options)

                return self._integrate_real(_integrand_add, right_left, right_right)

            def _integrand_sub(y: float) -> float:
                return self._eval_method_scalar(
                    left_pdf, z + y, **options
                ) * self._eval_method_scalar(right_pdf, y, **options)

            return self._integrate_real(_integrand_sub, right_left, right_right)

        def _pdf(data: NumericArray, **options: Any) -> NumericArray:
            return self._map_scalar_real(
                data,
                lambda z: _pdf_scalar(z, **options),
            )

        return cast(ComputationFunc[NumericArray, NumericArray], _pdf)

    def _make_continuous_cdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build transformed continuous CDF."""
        left_cdf = sources[_LEFT_ROLE][CharacteristicName.CDF]
        right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
        right_left, right_right = self._continuous_bounds_for_role(_RIGHT_ROLE)

        def _cdf_scalar(z: float, **options: Any) -> float:
            if self.operation == BinaryOperationName.ADD:

                def _integrand_add(y: float) -> float:
                    return self._eval_method_scalar(
                        left_cdf, z - y, **options
                    ) * self._eval_method_scalar(right_pdf, y, **options)

                value = self._integrate_real(_integrand_add, right_left, right_right)
                return float(np.clip(value, 0.0, 1.0))

            def _integrand_sub(y: float) -> float:
                return self._eval_method_scalar(
                    left_cdf, z + y, **options
                ) * self._eval_method_scalar(right_pdf, y, **options)

            value = self._integrate_real(_integrand_sub, right_left, right_right)
            return float(np.clip(value, 0.0, 1.0))

        def _cdf(data: NumericArray, **options: Any) -> NumericArray:
            return self._map_scalar_real(
                data,
                lambda z: _cdf_scalar(z, **options),
            )

        return cast(ComputationFunc[NumericArray, NumericArray], _cdf)

    def _transform_support(
        self,
        left_support: Support | None,
        right_support: Support | None,
    ) -> Support | None:
        """Transform support metadata for linear operations."""
        if isinstance(left_support, ContinuousSupport) and isinstance(
            right_support, ContinuousSupport
        ):
            if self.operation == BinaryOperationName.ADD:
                return ContinuousSupport(
                    left=float(left_support.left + right_support.left),
                    right=float(left_support.right + right_support.right),
                    left_closed=left_support.left_closed and right_support.left_closed,
                    right_closed=left_support.right_closed and right_support.right_closed,
                )
            return ContinuousSupport(
                left=float(left_support.left - right_support.right),
                right=float(left_support.right - right_support.left),
                left_closed=left_support.left_closed and right_support.right_closed,
                right_closed=left_support.right_closed and right_support.left_closed,
            )
        return super()._transform_support(left_support, right_support)


__all__ = [
    "LinearBinaryDistribution",
]
