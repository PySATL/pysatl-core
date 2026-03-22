"""
Affine transformation for probability distributions.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    Support,
)
from pysatl_core.transformations.distribution import DerivedDistribution
from pysatl_core.transformations.lightweight_distribution import LightweightDistribution
from pysatl_core.transformations.operations.methods.affine import (
    default_affine_transformation_methods,
)
from pysatl_core.transformations.transformation_method import TransformationMethod
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
    Kind,
    LabelName,
    NumericArray,
    ParentRole,
    TransformationMethodSpecsMap,
    TransformationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.strategies import (
        ComputationStrategy,
        SamplingStrategy,
    )

_BASE_ROLE: ParentRole = "base"


class AffineDistribution(DerivedDistribution):
    """
    Distribution obtained from the affine transformation ``Y = aX + b``.

    Parameters
    ----------
    base_distribution : Distribution
        Source distribution being transformed.
    scale : float
        Multiplicative coefficient ``a``.
    shift : float, default=0.0
        Additive coefficient ``b``.
    methods : TransformationMethodSpecsMap | None, default=None
        Transformation methods for building derived characteristics.
        When ``None``, built-in methods are used.
    sampling_strategy : SamplingStrategy | None, optional
        Sampling strategy exposed by the transformed distribution.
    computation_strategy : ComputationStrategy | None, optional
        Computation strategy exposed by the transformed distribution.
    """

    def __init__(
        self,
        base_distribution: Distribution,
        *,
        scale: float,
        shift: float = 0.0,
        methods: TransformationMethodSpecsMap | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        if scale == 0.0:
            raise ValueError("scale must be non-zero for an affine transformation.")

        base_snapshot = LightweightDistribution.from_distribution(base_distribution)
        self._base_distribution: LightweightDistribution = base_snapshot
        self._scale = scale
        self._shift = shift
        distribution_type = self._validate_distribution_type(base_snapshot.distribution_type)
        bases: dict[ParentRole, LightweightDistribution] = {_BASE_ROLE: base_snapshot}
        self._transformation_methods = self._resolve_transformation_methods(
            methods=methods,
            default_methods=default_affine_transformation_methods(
                kind=getattr(distribution_type, "kind", None),
                scale=scale,
            ),
        )
        analytical_computations, loop_analytical_flags = self._build_analytical_computations(
            distribution_type=distribution_type,
            bases=bases,
            methods=self._transformation_methods,
        )

        super().__init__(
            distribution_type=distribution_type,
            bases=bases,
            analytical_computations=analytical_computations,
            transformation_name=TransformationName.AFFINE,
            support=self._transform_support(base_snapshot.support),
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
            loop_analytical_flags=loop_analytical_flags,
        )

    @property
    def base_distribution(self) -> LightweightDistribution:
        """Get the lightweight snapshot of the source distribution."""
        return self._base_distribution

    @property
    def parent_roles(self) -> tuple[ParentRole, ...]:
        """Return parent role sequence for affine transformation."""
        return (_BASE_ROLE,)

    @property
    def scale(self) -> float:
        """Get the multiplicative coefficient ``a``."""
        return self._scale

    @property
    def shift(self) -> float:
        """Get the additive coefficient ``b``."""
        return self._shift

    def sample(self, n: int, **options: Any) -> NumericArray:
        """
        Generate affine samples from transformed base-distribution samples.
        """
        base_samples = self.sampling_strategy.sample(n, distr=self.base_distribution, **options)
        transformed_samples = np.asarray(base_samples, dtype=float) * self.scale + self.shift
        return cast(NumericArray, transformed_samples)

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> AffineDistribution:
        """Return a copy of the affine distribution with updated strategies."""
        return AffineDistribution(
            base_distribution=self.base_distribution,
            scale=self.scale,
            shift=self.shift,
            methods=self._transformation_methods,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

    def _validate_distribution_type(self, distribution_type: DistributionType) -> DistributionType:
        """
        Validate that the affine transformation can be applied.
        """
        dimension = getattr(distribution_type, "dimension", None)
        kind = getattr(distribution_type, "kind", None)

        if dimension != 1:
            raise TypeError(
                "AffineDistribution currently supports only one-dimensional distributions."
            )

        if kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for affine transformation.")

        return distribution_type

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
        """Build analytical computations for the affine transformation."""
        kind = getattr(distribution_type, "kind", None)
        if kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for affine transformation.")

        computations, loop_analytical_flags = self._build_transformation_analytical_computations(
            transformation_name=TransformationName.AFFINE,
            bases=bases,
            methods=methods,
        )
        if computations:
            return computations, loop_analytical_flags

        raise RuntimeError(
            "Affine transformation produced no analytical computations. "
            "At least one source characteristic must be present."
        )

    def _transform_support(self, support: Support | None) -> Support | None:
        """Transform the parent support when its structure is known."""
        if support is None:
            return None

        if isinstance(support, ContinuousSupport):
            return self._transform_continuous_support(support)

        if isinstance(support, ExplicitTableDiscreteSupport):
            transformed_points = np.asarray(support.points, dtype=float) * self.scale + self.shift
            return ExplicitTableDiscreteSupport(points=transformed_points, assume_sorted=False)

        return None

    def _transform_continuous_support(self, support: ContinuousSupport) -> ContinuousSupport:
        """Transform a continuous interval support under the affine map."""
        left = float(self.scale * support.left + self.shift)
        right = float(self.scale * support.right + self.shift)

        if self.scale > 0.0:
            return ContinuousSupport(
                left=left,
                right=right,
                left_closed=support.left_closed,
                right_closed=support.right_closed,
            )

        return ContinuousSupport(
            left=right,
            right=left,
            left_closed=support.right_closed,
            right_closed=support.left_closed,
        )


def affine(
    distribution: Distribution,
    *,
    scale: float,
    shift: float = 0.0,
    methods: TransformationMethodSpecsMap | None = None,
) -> AffineDistribution:
    """
    Apply the affine transformation ``Y = aX + b`` to a distribution.
    """
    return AffineDistribution(distribution, scale=scale, shift=shift, methods=methods)


__all__ = [
    "AffineDistribution",
    "affine",
]
