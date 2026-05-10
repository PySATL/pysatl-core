"""
Base abstractions for binary transformations over distributions.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import inf, isfinite
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    Support,
)
from pysatl_core.transformations.distribution import DerivedDistribution
from pysatl_core.transformations.lightweight_distribution import LightweightDistribution
from pysatl_core.transformations.operations.methods._utils import _eval_method_scalar
from pysatl_core.transformations.transformation_method import TransformationMethod
from pysatl_core.types import (
    BinaryOperationName,
    CharacteristicName,
    DistributionType,
    GenericCharacteristicName,
    Kind,
    LabelName,
    NumericArray,
    ParentRole,
    ResolvedSourceMethods,
    TransformationMethodSpecsMap,
    TransformationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.strategies import (
        ComputationStrategy,
        SamplingStrategy,
    )

_LEFT_ROLE: ParentRole = "left"
_RIGHT_ROLE: ParentRole = "right"


class BinaryDistribution(DerivedDistribution, ABC):
    """
    Base class for binary transformations over two parent distributions.
    """

    def __init__(
        self,
        left_distribution: Distribution,
        right_distribution: Distribution,
        *,
        operation: BinaryOperationName,
        methods: TransformationMethodSpecsMap,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        self._operation = operation

        left_snapshot = LightweightDistribution.from_distribution(left_distribution)
        right_snapshot = LightweightDistribution.from_distribution(right_distribution)

        self._left_distribution = left_snapshot
        self._right_distribution = right_snapshot
        self._cached_discrete_mass_table: tuple[NumericArray, NumericArray, NumericArray] | None = (
            None
        )

        distribution_type = self._validate_distribution_types(
            left_snapshot.distribution_type,
            right_snapshot.distribution_type,
        )
        bases: dict[ParentRole, LightweightDistribution] = {
            _LEFT_ROLE: left_snapshot,
            _RIGHT_ROLE: right_snapshot,
        }
        self._transformation_methods = methods
        transformed_support = self._transform_support(left_snapshot.support, right_snapshot.support)
        self._precomputed_support = transformed_support
        analytical_computations, loop_analytical_flags = self._build_analytical_computations(
            distribution_type=distribution_type,
            bases=bases,
            methods=self._transformation_methods,
        )

        super().__init__(
            distribution_type=distribution_type,
            bases=bases,
            analytical_computations=analytical_computations,
            transformation_name=TransformationName.BINARY,
            support=transformed_support,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
            loop_analytical_flags=loop_analytical_flags,
        )

    @property
    def left_distribution(self) -> LightweightDistribution:
        """Get the lightweight snapshot of the left parent distribution."""
        return self._left_distribution

    @property
    def right_distribution(self) -> LightweightDistribution:
        """Get the lightweight snapshot of the right parent distribution."""
        return self._right_distribution

    @property
    def operation(self) -> BinaryOperationName:
        """Get the binary operation name."""
        return self._operation

    @property
    def parent_roles(self) -> tuple[ParentRole, ...]:
        """Return parent role sequence for binary transformation."""
        return (_LEFT_ROLE, _RIGHT_ROLE)

    @property
    def transformation_methods(self) -> TransformationMethodSpecsMap:
        """Get transformation method specifications used to build this distribution."""
        return self._transformation_methods

    def sample(self, n: int, **options: Any) -> NumericArray:
        """
        Generate transformed samples from two parent-distribution samples.
        """
        left_samples = np.asarray(
            self.sampling_strategy.sample(n, distr=self.left_distribution, **options),
            dtype=float,
        )
        right_samples = np.asarray(
            self.sampling_strategy.sample(n, distr=self.right_distribution, **options),
            dtype=float,
        )
        transformed_samples = self._operation_value_array(left_samples, right_samples)
        return cast(NumericArray, np.asarray(transformed_samples, dtype=float))

    @abstractmethod
    def _operation_value(
        self,
        left: float | NumericArray,
        right: float | NumericArray,
    ) -> float | NumericArray:
        """Apply the concrete binary operation in scalar or array semantics."""

    def _operation_value_array(self, left: NumericArray, right: NumericArray) -> NumericArray:
        """Apply binary operation to broadcast-compatible arrays."""
        return cast(NumericArray, np.asarray(self._operation_value(left, right), dtype=float))

    def _build_analytical_computations(
        self,
        *,
        distribution_type: DistributionType,
        bases: Mapping[ParentRole, LightweightDistribution],
        methods: TransformationMethodSpecsMap,
    ) -> tuple[
        Mapping[GenericCharacteristicName, Mapping[LabelName, TransformationMethod[Any, Any]]],
        Mapping[GenericCharacteristicName, Mapping[LabelName, bool]],
    ]:
        """
        Build analytical computations for the concrete binary transformation.
        """
        kind = cast(Kind | None, getattr(distribution_type, "kind", None))
        if kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for binary transformation.")

        declared_registry_characteristics = characteristic_registry().declared_characteristics

        def _source_validator(role: ParentRole, characteristic: GenericCharacteristicName) -> bool:
            if characteristic in declared_registry_characteristics:
                return True
            return characteristic in bases[role].analytical_computations

        computations, loop_analytical_flags = self._build_transformation_analytical_computations(
            transformation_name=TransformationName.BINARY,
            bases=bases,
            methods=methods,
            source_validator=_source_validator,
        )

        if computations:
            return computations, loop_analytical_flags

        raise RuntimeError(
            "Binary transformation produced no analytical computations. "
            "At least one source characteristic must be present."
        )

    @staticmethod
    def _validate_distribution_types(
        left_type: DistributionType,
        right_type: DistributionType,
    ) -> DistributionType:
        """Validate compatibility of parent distribution types."""
        left_dimension = getattr(left_type, "dimension", None)
        right_dimension = getattr(right_type, "dimension", None)
        left_kind = getattr(left_type, "kind", None)
        right_kind = getattr(right_type, "kind", None)

        if left_dimension != 1 or right_dimension != 1:
            raise TypeError(
                "BinaryDistribution currently supports only one-dimensional distributions."
            )
        if left_kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for binary transformation.")
        if left_kind != right_kind:
            raise TypeError(
                "BinaryDistribution currently requires both parents to have the same kind."
            )
        return left_type

    def _continuous_bounds_for_role(self, role: ParentRole) -> tuple[float, float]:
        """Get integration bounds from continuous support or fallback to real line."""
        support = (
            self.left_distribution.support
            if role == _LEFT_ROLE
            else self.right_distribution.support
        )
        if isinstance(support, ContinuousSupport):
            return float(support.left), float(support.right)
        return -inf, inf

    def _discrete_points_for_role(self, role: ParentRole) -> NumericArray:
        """Get explicit discrete support points for one parent role."""
        support = (
            self.left_distribution.support
            if role == _LEFT_ROLE
            else self.right_distribution.support
        )
        if not isinstance(support, ExplicitTableDiscreteSupport):
            raise RuntimeError(
                "Binary discrete computations require ExplicitTableDiscreteSupport "
                "for both parents."
            )
        return cast(NumericArray, np.asarray(support.points, dtype=float))

    def _discrete_mass_table(
        self,
        sources: ResolvedSourceMethods,
        **options: Any,
    ) -> tuple[NumericArray, NumericArray, NumericArray]:
        """Build transformed finite PMF table for discrete parent supports."""
        if not options and self._cached_discrete_mass_table is not None:
            return self._cached_discrete_mass_table

        left_pmf = sources[_LEFT_ROLE][CharacteristicName.PMF]
        right_pmf = sources[_RIGHT_ROLE][CharacteristicName.PMF]
        left_points = self._discrete_points_for_role(_LEFT_ROLE)
        right_points = self._discrete_points_for_role(_RIGHT_ROLE)
        left_masses = np.asarray(
            [_eval_method_scalar(left_pmf, float(x), **options) for x in left_points],
            dtype=float,
        )
        right_masses = np.asarray(
            [_eval_method_scalar(right_pmf, float(y), **options) for y in right_points],
            dtype=float,
        )

        positive_left = np.nonzero(left_masses > 0.0)[0]
        positive_right = np.nonzero(right_masses > 0.0)[0]
        left_values = left_points[positive_left]
        left_weights = left_masses[positive_left]
        right_values = right_points[positive_right]
        right_weights = right_masses[positive_right]

        if self.operation == BinaryOperationName.DIV:
            nonzero_mask = ~np.isclose(right_values, 0.0, atol=0.0, rtol=0.0)
            right_values = right_values[nonzero_mask]
            right_weights = right_weights[nonzero_mask]

        if left_values.size == 0 or right_values.size == 0:
            raise RuntimeError("Binary discrete transformation produced an empty PMF table.")

        transformed = self._operation_value_array(
            left_values[:, None], right_values[None, :]
        ).reshape(-1)
        pair_weights = (left_weights[:, None] * right_weights[None, :]).reshape(-1)

        rounded_points = np.round(transformed, 12)
        unique_points, inverse = np.unique(rounded_points, return_inverse=True)
        accumulated_masses = np.zeros_like(unique_points, dtype=float)
        np.add.at(accumulated_masses, inverse, pair_weights)

        positive_mass_mask = accumulated_masses > 0.0
        points = unique_points[positive_mass_mask]
        masses = accumulated_masses[positive_mass_mask]
        total = float(np.sum(masses))
        if total <= 0.0:
            raise RuntimeError("Binary discrete transformation produced non-positive total mass.")
        masses = cast(NumericArray, masses / total)
        cdf_values = cast(NumericArray, np.cumsum(masses, dtype=float))
        cdf_values[-1] = 1.0
        output = (points, masses, cdf_values)
        if not options:
            self._cached_discrete_mass_table = output
        return output

    def _transform_support(
        self,
        left_support: Support | None,
        right_support: Support | None,
    ) -> Support | None:
        """Transform support metadata when both parent supports are explicit enough."""
        if isinstance(left_support, ContinuousSupport) and isinstance(
            right_support, ContinuousSupport
        ):
            finite_bounds = all(
                isfinite(value)
                for value in (
                    float(left_support.left),
                    float(left_support.right),
                    float(right_support.left),
                    float(right_support.right),
                )
            )
            if not finite_bounds:
                return None

            left_bounds = (float(left_support.left), float(left_support.right))
            right_bounds = (float(right_support.left), float(right_support.right))
            if (
                self.operation == BinaryOperationName.DIV
                and right_bounds[0] <= 0.0 <= right_bounds[1]
            ):
                return None

            left_values = np.asarray(
                [left_bounds[0], left_bounds[0], left_bounds[1], left_bounds[1]],
                dtype=float,
            )
            right_values = np.asarray(
                [right_bounds[0], right_bounds[1], right_bounds[0], right_bounds[1]],
                dtype=float,
            )
            values = self._operation_value_array(left_values, right_values)
            closed = (
                left_support.left_closed
                and left_support.right_closed
                and right_support.left_closed
                and right_support.right_closed
            )
            return ContinuousSupport(
                left=float(np.min(values)),
                right=float(np.max(values)),
                left_closed=closed,
                right_closed=closed,
            )

        if isinstance(left_support, ExplicitTableDiscreteSupport) and isinstance(
            right_support, ExplicitTableDiscreteSupport
        ):
            left_points = np.asarray(left_support.points, dtype=float)
            right_points = np.asarray(right_support.points, dtype=float)
            if self.operation == BinaryOperationName.DIV:
                right_points = right_points[~np.isclose(right_points, 0.0, atol=0.0, rtol=0.0)]

            if left_points.size == 0 or right_points.size == 0:
                return None

            transformed = self._operation_value_array(
                left_points[:, None],
                right_points[None, :],
            ).reshape(-1)
            return ExplicitTableDiscreteSupport(points=transformed.tolist(), assume_sorted=False)

        return None


_SUPPORTED_BINARY_OPERATIONS: frozenset[BinaryOperationName] = frozenset(BinaryOperationName)


def binary(
    left_distribution: Distribution,
    right_distribution: Distribution,
    *,
    operation: BinaryOperationName,
    methods: TransformationMethodSpecsMap | None = None,
) -> BinaryDistribution:
    """
    Apply a binary operation to two distributions.
    """
    from pysatl_core.transformations.operations.distributions.binary.division import (
        DivisionBinaryDistribution,
    )
    from pysatl_core.transformations.operations.distributions.binary.linear import (
        LinearBinaryDistribution,
    )
    from pysatl_core.transformations.operations.distributions.binary.multiplication import (
        MultiplicationBinaryDistribution,
    )

    if operation in {BinaryOperationName.ADD, BinaryOperationName.SUB}:
        return LinearBinaryDistribution(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            operation=operation,
            methods=methods,
        )
    if operation == BinaryOperationName.MUL:
        return MultiplicationBinaryDistribution(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            methods=methods,
        )
    if operation == BinaryOperationName.DIV:
        return DivisionBinaryDistribution(
            left_distribution=left_distribution,
            right_distribution=right_distribution,
            methods=methods,
        )
    raise ValueError(
        f"Unsupported binary operation '{operation}'. "
        f"Supported operations: {', '.join(sorted(_SUPPORTED_BINARY_OPERATIONS))}."
    )


__all__ = [
    "BinaryDistribution",
    "BinaryOperationName",
    "binary",
]
