"""
Division binary transformation ``X / Y``.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, cast

import numpy as np

from pysatl_core.distributions.distribution import _KEEP
from pysatl_core.transformations.operations.distributions.binary.base import (
    BinaryDistribution,
)
from pysatl_core.transformations.operations.methods.binary.division import (
    default_division_binary_transformation_methods,
)
from pysatl_core.types import (
    BinaryOperationName,
    NumericArray,
    TransformationMethodSpecsMap,
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
        methods: TransformationMethodSpecsMap | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        self._transformation_methods = self._resolve_transformation_methods(
            methods=methods,
            default_methods=default_division_binary_transformation_methods(
                kind=getattr(left_distribution.distribution_type, "kind", None)
            ),
        )
        super().__init__(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            operation=BinaryOperationName.DIV,
            methods=self._transformation_methods,
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
            methods=self.transformation_methods,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

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


__all__ = [
    "DivisionBinaryDistribution",
]
