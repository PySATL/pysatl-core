"""
Multiplicative binary transformation ``X * Y``.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.distribution import _KEEP
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


class MultiplicationBinaryDistribution(BinaryDistribution):
    """
    Binary distribution for multiplicative transformation ``X * Y``.
    """

    def __init__(
        self,
        left_distribution: Distribution,
        right_distribution: Distribution,
        *,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        super().__init__(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            operation=BinaryOperationName.MUL,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
        )

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> MultiplicationBinaryDistribution:
        """Return a copy of the multiplication binary distribution with updated strategies."""
        return MultiplicationBinaryDistribution(
            left_distribution=self.left_distribution,
            right_distribution=self.right_distribution,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

    def _characteristic_specs(
        self, *, kind: Kind | None
    ) -> tuple[
        tuple[GenericCharacteristicName, SourceRequirements, TransformationEvaluator[Any, Any]],
        ...,
    ]:
        """Return characteristic registration specs for multiplication."""
        density = self._density_characteristic(kind)
        specs: list[
            tuple[GenericCharacteristicName, SourceRequirements, TransformationEvaluator[Any, Any]]
        ] = [
            (
                CharacteristicName.CF,
                {
                    _LEFT_ROLE: (CharacteristicName.CF,),
                    _RIGHT_ROLE: (density,),
                },
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
        """Apply multiplication in scalar or array semantics."""
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        result = left_array * right_array
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
        """Compute raw moments up to ``max_order`` for ``X * Y``."""
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

        output = [0.0, 0.0, 0.0, 0.0]
        for idx in range(order):
            output[idx] = left_raw[idx] * right_raw[idx]
        return output[0], output[1], output[2], output[3]

    def _make_cf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, ComplexArray]:
        """Build transformed characteristic function."""
        left_cf = sources[_LEFT_ROLE][CharacteristicName.CF]
        kind = getattr(self.left_distribution.distribution_type, "kind", None)

        if kind == Kind.CONTINUOUS:
            right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
            right_left, right_right = self._continuous_bounds_for_role(_RIGHT_ROLE)

            def _cf_scalar_continuous(t: float, **options: Any) -> complex:
                def _integrand(y: float) -> complex:
                    return self._eval_method_scalar_complex(
                        left_cf, t * y, **options
                    ) * self._eval_method_scalar(right_pdf, y, **options)

                return self._integrate_complex(_integrand, right_left, right_right)

            def _cf_continuous(
                data: NumericArray,
                **options: Any,
            ) -> ComplexArray:
                return self._map_scalar_complex(
                    data,
                    lambda t: _cf_scalar_continuous(t, **options),
                )

            return cast(ComputationFunc[NumericArray, ComplexArray], _cf_continuous)

        right_pmf = sources[_RIGHT_ROLE][CharacteristicName.PMF]
        right_points = self._discrete_points_for_role(_RIGHT_ROLE)

        def _cf_scalar_discrete(t: float, **options: Any) -> complex:
            total = 0.0j
            for y in right_points:
                y_float = float(y)
                py = self._eval_method_scalar(right_pmf, y_float, **options)
                total += self._eval_method_scalar_complex(left_cf, t * y_float, **options) * py
            return total

        def _cf_discrete(data: NumericArray, **options: Any) -> ComplexArray:
            return self._map_scalar_complex(
                data,
                lambda t: _cf_scalar_discrete(t, **options),
            )

        return cast(ComputationFunc[NumericArray, ComplexArray], _cf_discrete)

    def _make_continuous_pdf(
        self,
        sources: ResolvedSourceMethods,
    ) -> ComputationFunc[NumericArray, NumericArray]:
        """Build transformed continuous PDF."""
        left_pdf = sources[_LEFT_ROLE][CharacteristicName.PDF]
        right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
        right_left, right_right = self._continuous_bounds_for_role(_RIGHT_ROLE)

        def _pdf_scalar(z: float, **options: Any) -> float:
            def _integrand(y: float) -> float:
                return (
                    self._eval_method_scalar(left_pdf, z / y, **options)
                    * self._eval_method_scalar(right_pdf, y, **options)
                    / abs(y)
                )

            return self._integrate_real(
                _integrand,
                right_left,
                right_right,
                split_at_zero=True,
            )

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
                        1.0 - self._eval_method_scalar(left_cdf, z / y, **options)
                    ) * self._eval_method_scalar(right_pdf, y, **options)

                negative = self._integrate_real(_neg_integrand, negative_left, negative_right)

            positive = 0.0
            if positive_left < positive_right:

                def _pos_integrand(y: float) -> float:
                    return self._eval_method_scalar(
                        left_cdf, z / y, **options
                    ) * self._eval_method_scalar(right_pdf, y, **options)

                positive = self._integrate_real(_pos_integrand, positive_left, positive_right)

            return float(np.clip(negative + positive, 0.0, 1.0))

        def _cdf(data: NumericArray, **options: Any) -> NumericArray:
            return self._map_scalar_real(
                data,
                lambda z: _cdf_scalar(z, **options),
            )

        return cast(ComputationFunc[NumericArray, NumericArray], _cdf)


__all__ = [
    "MultiplicationBinaryDistribution",
]
