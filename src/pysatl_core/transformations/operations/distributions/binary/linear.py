"""
Linear binary transformations: addition and subtraction.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, cast

import numpy as np

from pysatl_core.distributions.distribution import _KEEP
from pysatl_core.distributions.support import ContinuousSupport, Support
from pysatl_core.transformations.operations.distributions.binary.base import (
    BinaryDistribution,
)
from pysatl_core.transformations.operations.methods.binary.linear import (
    default_linear_binary_transformation_methods,
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
        methods: TransformationMethodSpecsMap | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        if operation not in _SUPPORTED_LINEAR_OPERATIONS:
            raise ValueError(
                f"Unsupported linear operation '{operation}'. "
                f"Supported operations: {', '.join(sorted(_SUPPORTED_LINEAR_OPERATIONS))}."
            )
        self._transformation_methods = self._resolve_transformation_methods(
            methods=methods,
            default_methods=default_linear_binary_transformation_methods(
                kind=getattr(left_distribution.distribution_type, "kind", None)
            ),
        )
        super().__init__(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            operation=operation,
            methods=self._transformation_methods,
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
            methods=self.transformation_methods,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

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
