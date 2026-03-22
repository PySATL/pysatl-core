"""
Division binary transformation ``X / Y``.
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


class DivisionBinaryDistribution(BinaryDistribution):
    """
    Binary distribution for ratio transformation ``X / Y``.
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
            operation=BinaryOperationName.DIV,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
        )

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> DivisionBinaryDistribution:
        """Return a copy of the division binary distribution with updated strategies."""
        return DivisionBinaryDistribution(
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
        """Return characteristic registration specs for division."""
        density = self._density_characteristic(kind)
        mean_requirements = {
            _LEFT_ROLE: self._statistics_for_moment_order(1),
            _RIGHT_ROLE: (density,),
        }
        var_requirements = {
            _LEFT_ROLE: self._statistics_for_moment_order(2),
            _RIGHT_ROLE: (density,),
        }
        skew_requirements = {
            _LEFT_ROLE: self._statistics_for_moment_order(3),
            _RIGHT_ROLE: (density,),
        }
        kurt_requirements = {
            _LEFT_ROLE: self._statistics_for_moment_order(4),
            _RIGHT_ROLE: (density,),
        }

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
            (CharacteristicName.MEAN, mean_requirements, self._make_mean),
            (CharacteristicName.VAR, var_requirements, self._make_var),
            (CharacteristicName.SKEW, skew_requirements, self._make_skew),
            (CharacteristicName.KURT, kurt_requirements, self._make_kurt),
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
        """Apply division in scalar or array semantics."""
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if np.any(np.isclose(right_array, 0.0, atol=0.0, rtol=0.0)):
            raise ZeroDivisionError("Division by zero support point in binary transformation.")

        result = left_array / right_array
        if result.ndim == 0:
            return float(result)
        return cast(NumericArray, result)

    def _right_inverse_moments(
        self,
        sources: ResolvedSourceMethods,
        *,
        max_order: int = 4,
        **options: Any,
    ) -> tuple[float, float, float, float]:
        """Compute right-parent inverse moments ``E[Y^{-k}]`` up to ``max_order``."""
        order = self._validate_moment_order(max_order)
        kind = getattr(self.distribution_type, "kind", None)

        if kind == Kind.CONTINUOUS:
            right_pdf = sources[_RIGHT_ROLE][CharacteristicName.PDF]
            left, right = self._continuous_bounds_for_role(_RIGHT_ROLE)
            if left <= 0.0 <= right:
                raise RuntimeError(
                    "Division transformation requires denominator support that does not cross zero."
                )

            def _moment(order: int) -> float:
                def _integrand(y: float) -> float:
                    return y ** (-order) * self._eval_method_scalar(right_pdf, y, **options)

                return self._integrate_real(_integrand, left, right)

            output = [0.0, 0.0, 0.0, 0.0]
            for current_order in range(1, order + 1):
                output[current_order - 1] = _moment(current_order)
            return output[0], output[1], output[2], output[3]

        right_pmf = sources[_RIGHT_ROLE][CharacteristicName.PMF]
        points = self._discrete_points_for_role(_RIGHT_ROLE)
        if np.any(np.isclose(points, 0.0, atol=1e-14, rtol=0.0)):
            zero_mass = self._eval_method_scalar(right_pmf, 0.0, **options)
            if zero_mass > 0.0:
                raise RuntimeError(
                    "Division transformation is undefined when denominator has "
                    "positive mass at zero."
                )

        inverse_moments = [0.0, 0.0, 0.0, 0.0]
        for y in points:
            y_float = float(y)
            if y_float == 0.0:
                continue
            py = self._eval_method_scalar(right_pmf, y_float, **options)
            for current_order in range(1, order + 1):
                inverse_moments[current_order - 1] += py / y_float**current_order

        return cast(tuple[float, float, float, float], tuple(inverse_moments))

    def _result_raw_moments(
        self,
        sources: ResolvedSourceMethods,
        *,
        max_order: int = 4,
        **options: Any,
    ) -> tuple[float, float, float, float]:
        """Compute raw moments up to ``max_order`` for ``X / Y``."""
        order = self._validate_moment_order(max_order)
        left_raw = self._parent_raw_moments(
            sources,
            _LEFT_ROLE,
            max_order=order,
            **options,
        )
        right_inverse = self._right_inverse_moments(
            sources,
            max_order=order,
            **options,
        )
        output = [0.0, 0.0, 0.0, 0.0]
        for idx in range(order):
            output[idx] = left_raw[idx] * right_inverse[idx]
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
            if right_left <= 0.0 <= right_right:
                raise RuntimeError(
                    "Characteristic function for division requires denominator support "
                    "that does not cross zero."
                )

            def _cf_scalar_continuous(t: float, **options: Any) -> complex:
                def _integrand(y: float) -> complex:
                    return self._eval_method_scalar_complex(
                        left_cf, t / y, **options
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
                if y_float == 0.0:
                    continue
                py = self._eval_method_scalar(right_pmf, y_float, **options)
                total += self._eval_method_scalar_complex(left_cf, t / y_float, **options) * py
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

        if right_left <= 0.0 <= right_right:
            raise RuntimeError(
                "Continuous division PDF requires denominator support that does not cross zero."
            )

        def _pdf_scalar(z: float, **options: Any) -> float:
            def _integrand(y: float) -> float:
                return (
                    abs(y)
                    * self._eval_method_scalar(left_pdf, z * y, **options)
                    * self._eval_method_scalar(right_pdf, y, **options)
                )

            return self._integrate_real(_integrand, right_left, right_right)

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

            negative = 0.0
            if negative_left < negative_right:

                def _neg_integrand(y: float) -> float:
                    return (
                        1.0 - self._eval_method_scalar(left_cdf, z * y, **options)
                    ) * self._eval_method_scalar(right_pdf, y, **options)

                negative = self._integrate_real(_neg_integrand, negative_left, negative_right)

            positive = 0.0
            if positive_left < positive_right:

                def _pos_integrand(y: float) -> float:
                    return self._eval_method_scalar(
                        left_cdf, z * y, **options
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
    "DivisionBinaryDistribution",
]
