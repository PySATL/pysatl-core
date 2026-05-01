"""
Finite weighted mixture transformation for probability distributions.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.computations.computation import Method
from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    Support,
)
from pysatl_core.transformations.distribution import DerivedDistribution
from pysatl_core.transformations.lightweight_distribution import LightweightDistribution
from pysatl_core.transformations.operations.methods.mixture import (
    default_finite_mixture_transformation_methods,
)
from pysatl_core.transformations.transformation_method import TransformationMethod
from pysatl_core.types import (
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

_COMPONENT_ROLE_PREFIX = "component_"


def _component_role(index: int) -> ParentRole:
    """Build deterministic parent role for a mixture component."""
    return cast(ParentRole, f"{_COMPONENT_ROLE_PREFIX}{index}")


class FiniteMixtureDistribution(DerivedDistribution):
    """
    Distribution obtained as a finite weighted mixture of components.

    Parameters
    ----------
    weighted_components : Sequence[tuple[float, Distribution]]
        Ordered pairs ``(weight, distribution)``.
        Weights must be finite, non-negative and sum to one.
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
        weighted_components: Sequence[tuple[float, Distribution]],
        *,
        methods: TransformationMethodSpecsMap | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        if not weighted_components:
            raise ValueError("Finite mixture requires at least one component distribution.")

        components = tuple(component for _, component in weighted_components)
        weights = [float(weight) for weight, _ in weighted_components]

        component_snapshots = tuple(
            LightweightDistribution.from_distribution(component) for component in components
        )
        validated_weights = self._validate_weights(weights)
        roles = tuple(_component_role(index) for index in range(len(component_snapshots)))

        self._components = component_snapshots
        self._weights = validated_weights
        self._roles = roles
        self._role_to_index = {role: index for index, role in enumerate(roles)}
        self._cached_discrete_mass_table: tuple[NumericArray, NumericArray, NumericArray] | None = (
            None
        )

        distribution_type = self._validate_distribution_types(
            [component.distribution_type for component in component_snapshots]
        )
        self._discrete_support = self._build_discrete_support()
        self._continuous_support = self._build_continuous_support()

        self._discrete_points = (
            cast(NumericArray, np.asarray(self._discrete_support.points, dtype=float))
            if self._discrete_support is not None
            else None
        )

        bases: dict[ParentRole, LightweightDistribution] = dict(
            zip(roles, component_snapshots, strict=True)
        )
        self._transformation_methods = self._resolve_transformation_methods(
            methods=methods,
            default_methods=default_finite_mixture_transformation_methods(
                kind=getattr(distribution_type, "kind", None)
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
            transformation_name=TransformationName.FINITE_MIXTURE,
            support=self._transform_support(distribution_type),
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
            loop_analytical_flags=loop_analytical_flags,
        )

    @property
    def components(self) -> tuple[LightweightDistribution, ...]:
        """Get the lightweight component snapshots."""
        return self._components

    @property
    def parent_roles(self) -> tuple[ParentRole, ...]:
        """Return ordered component roles used by this finite mixture."""
        return self._roles

    @property
    def weights(self) -> NumericArray:
        """Get validated component weights."""
        return cast(NumericArray, np.array(self._weights, copy=True))

    def sample(self, n: int, **options: Any) -> NumericArray:
        """
        Generate mixture samples by sampling selected components.
        """
        rng = np.random.default_rng()
        component_indices = np.asarray(rng.choice(len(self._components), size=n, p=self._weights))
        samples = np.empty(component_indices.shape, dtype=float)

        for index, component in enumerate(self.components):
            selected_positions = np.nonzero(component_indices == index)[0]
            selected_count = int(selected_positions.size)
            if selected_count == 0:
                continue

            component_samples = np.asarray(
                self.sampling_strategy.sample(selected_count, distr=component, **options),
                dtype=float,
            ).reshape(-1)
            if component_samples.size != selected_count:
                raise RuntimeError(
                    "Component sampler returned incompatible sample shape for finite mixture."
                )
            samples[selected_positions] = component_samples

        return cast(NumericArray, samples)

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> FiniteMixtureDistribution:
        """Return a copy of the mixture distribution with updated strategies."""
        return FiniteMixtureDistribution(
            weighted_components=[
                (float(weight), component)
                for weight, component in zip(self._weights, self.components, strict=True)
            ],
            methods=self._transformation_methods,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )

    @staticmethod
    def _validate_weights(weights: Sequence[float]) -> NumericArray:
        """Validate mixture weights."""
        validated = np.asarray(weights, dtype=float)
        if validated.ndim != 1:
            raise ValueError("Mixture weights must be a one-dimensional sequence.")
        if np.any(~np.isfinite(validated)):
            raise ValueError("Mixture weights must be finite numbers.")
        if np.any(validated < 0.0):
            raise ValueError("Mixture weights must be non-negative.")

        total = float(np.sum(validated))
        if not np.isclose(total, 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError(f"Sum of mixture weights must be equal to 1.0, got {total}.")
        return cast(NumericArray, np.array(validated, copy=True))

    @staticmethod
    def _validate_distribution_types(
        distribution_types: Sequence[DistributionType],
    ) -> DistributionType:
        """Validate component distribution type compatibility."""
        first = distribution_types[0]
        first_dimension = getattr(first, "dimension", None)
        first_kind = getattr(first, "kind", None)

        if first_dimension != 1:
            raise TypeError("Finite mixture currently supports only one-dimensional distributions.")
        if first_kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for finite mixture.")

        for distribution_type in distribution_types[1:]:
            dimension = getattr(distribution_type, "dimension", None)
            kind = getattr(distribution_type, "kind", None)
            if dimension != first_dimension:
                raise TypeError("Finite mixture requires components with equal dimension.")
            if kind != first_kind:
                raise TypeError("Finite mixture requires components of the same distribution kind.")

        return first

    def _build_discrete_support(self) -> ExplicitTableDiscreteSupport | None:
        """Build explicit discrete support union, if available."""
        points_blocks: list[NumericArray] = []
        for component in self.components:
            support = component.support
            if not isinstance(support, ExplicitTableDiscreteSupport):
                return None
            points_blocks.append(cast(NumericArray, np.asarray(support.points, dtype=float)))

        if not points_blocks:
            return None
        merged = np.unique(np.concatenate(points_blocks)).tolist()
        return ExplicitTableDiscreteSupport(points=merged, assume_sorted=True)

    def _build_continuous_support(self) -> ContinuousSupport | None:
        """Build continuous support envelope, if available."""
        supports: list[ContinuousSupport] = []
        for component in self.components:
            support = component.support
            if not isinstance(support, ContinuousSupport):
                return None
            supports.append(support)

        left = min(float(support.left) for support in supports)
        right = max(float(support.right) for support in supports)
        left_closed = any(
            float(support.left) == left and support.left_closed for support in supports
        )
        right_closed = any(
            float(support.right) == right and support.right_closed for support in supports
        )
        return ContinuousSupport(
            left=left,
            right=right,
            left_closed=left_closed,
            right_closed=right_closed,
        )

    def _transform_support(self, distribution_type: DistributionType) -> Support | None:
        """Transform support metadata for the mixture."""
        kind = getattr(distribution_type, "kind", None)

        if kind == Kind.CONTINUOUS:
            return self._continuous_support

        if kind == Kind.DISCRETE:
            return self._discrete_support

        return None

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
        """Build analytical computations for finite mixture transformation."""
        kind = getattr(distribution_type, "kind", None)
        if kind not in {Kind.CONTINUOUS, Kind.DISCRETE}:
            raise TypeError("Unsupported distribution kind for finite mixture.")

        declared_registry_characteristics = characteristic_registry().declared_characteristics

        def _source_validator(role: ParentRole, characteristic: GenericCharacteristicName) -> bool:
            if characteristic in declared_registry_characteristics:
                return True
            return characteristic in bases[role].analytical_computations

        computations, loop_analytical_flags = self._build_transformation_analytical_computations(
            transformation_name=TransformationName.FINITE_MIXTURE,
            bases=bases,
            methods=methods,
            source_validator=_source_validator,
        )
        if computations:
            return computations, loop_analytical_flags

        raise RuntimeError(
            "Finite mixture produced no analytical computations. "
            "At least one source characteristic must be present."
        )

    def _discrete_mass_table(
        self,
        sources: ResolvedSourceMethods,
        **options: Any,
    ) -> tuple[NumericArray, NumericArray, NumericArray]:
        """Build normalized finite PMF table for discrete mixture."""
        if not options and self._cached_discrete_mass_table is not None:
            return self._cached_discrete_mass_table

        if self._discrete_points is None:
            raise RuntimeError(
                "Discrete finite mixture requires ExplicitTableDiscreteSupport "
                "for every component."
            )

        methods = [
            cast(Method[NumericArray, NumericArray], sources[role][CharacteristicName.PMF])
            for role in self._roles
        ]
        points = self._discrete_points
        masses = np.zeros(points.shape, dtype=float)
        for weight, method in zip(self._weights, methods, strict=True):
            masses += float(weight) * np.asarray(method(points, **options), dtype=float)

        masses = np.clip(masses, 0.0, None)
        total = float(np.sum(masses))
        if total <= 0.0:
            raise RuntimeError("Discrete finite mixture produced non-positive total mass.")

        masses = cast(NumericArray, masses / total)
        cdf_values = cast(NumericArray, np.cumsum(masses, dtype=float))
        cdf_values[-1] = 1.0
        output = (points, masses, cdf_values)
        if not options:
            self._cached_discrete_mass_table = output
        return output


def finite_mixture(
    weighted_components: Sequence[tuple[float, Distribution]],
    *,
    methods: TransformationMethodSpecsMap | None = None,
) -> FiniteMixtureDistribution:
    """
    Build a finite weighted mixture distribution.
    """
    return FiniteMixtureDistribution(
        weighted_components=weighted_components,
        methods=methods,
    )


def discrete_mixture(
    weighted_components: Sequence[tuple[float, Distribution]],
    *,
    methods: TransformationMethodSpecsMap | None = None,
) -> FiniteMixtureDistribution:
    """
    Build a finite mixture with a discrete set of component weights.

    This is an alias of :func:`finite_mixture`.
    """
    return finite_mixture(
        weighted_components=weighted_components,
        methods=methods,
    )


__all__ = [
    "FiniteMixtureDistribution",
    "discrete_mixture",
    "finite_mixture",
]
