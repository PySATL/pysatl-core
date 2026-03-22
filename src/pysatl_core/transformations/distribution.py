"""
Base classes for transformed distributions.

This module introduces the first architectural layer for derived
probability distributions produced by transformations. The goal is to
keep them fully compatible with the existing :class:`Distribution`
protocol and computation graph while still preserving transformation
metadata.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    SamplingStrategy,
)
from pysatl_core.transformations.lightweight_distribution import LightweightDistribution
from pysatl_core.transformations.operators_mixin import TransformationOperatorsMixin
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
    LabelName,
    ParentRole,
    TransformationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support
    from pysatl_core.transformations.approximations.approximation import (
        CharacteristicApproximationMethod,
    )


class DerivedDistribution(TransformationOperatorsMixin, Distribution):
    """
    Base class for distributions obtained from one or more parents.

    Parameters
    ----------
    distribution_type : DistributionType
        Type descriptor of the derived distribution.
    bases : Mapping[ParentRole, Distribution]
        Parent distributions participating in the transformation.
        Internally, they are stored as lightweight snapshots to avoid
        retaining full parent distribution objects.
    analytical_computations : Mapping[
        GenericCharacteristicName,
        (
            AnalyticalComputation[Any, Any]
            | Mapping[LabelName, AnalyticalComputation[Any, Any]]
        ),
    ]
        Derived characteristic methods exposed by the transformation.
        Presence here means that at least one ancestor in the derivation
        chain is analytical.
    transformation_name : TransformationName
        Logical name of the transformation.
    support : Support | None, optional
        Support of the transformed distribution.
    sampling_strategy : SamplingStrategy | None, optional
        Strategy used to generate random samples.
    computation_strategy : ComputationStrategy | None, optional
        Strategy used to resolve characteristics.
    loop_analytical_flags : Mapping[
        GenericCharacteristicName,
        Mapping[LabelName, bool],
    ] | None, optional
        Optional graph flags for loop analytical status.
        A loop has ``is_analytical=True`` only when all required ancestors
        of that characteristic are analytical.
    """

    def __init__(
        self,
        *,
        distribution_type: DistributionType,
        bases: Mapping[ParentRole, Distribution],
        analytical_computations: Mapping[
            GenericCharacteristicName,
            (AnalyticalComputation[Any, Any] | Mapping[LabelName, AnalyticalComputation[Any, Any]]),
        ],
        transformation_name: TransformationName,
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
        loop_analytical_flags: (
            Mapping[GenericCharacteristicName, Mapping[LabelName, bool]] | None
        ) = None,
    ) -> None:
        super().__init__(
            distribution_type=distribution_type,
            analytical_computations=analytical_computations,
            support=support,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
        )
        self._bases = {
            role: LightweightDistribution.from_distribution(base) for role, base in bases.items()
        }
        self._transformation_name = transformation_name
        self._loop_analytical_flags = {
            characteristic_name: dict(flags_by_label)
            for characteristic_name, flags_by_label in (loop_analytical_flags or {}).items()
        }

    @property
    def bases(self) -> Mapping[ParentRole, Distribution]:
        """Get parent distributions grouped by their logical roles."""
        return self._bases

    @property
    def transformation_name(self) -> TransformationName:
        """Get the logical name of the transformation."""
        return self._transformation_name

    def loop_is_analytical(
        self,
        characteristic_name: GenericCharacteristicName,
        label_name: LabelName,
    ) -> bool:
        """
        Return transformation-aware analytical flag for a loop method.

        The method returns ``True`` only when all required predecessors are
        analytical. A method can still be present in
        ``analytical_computations`` when this returns ``False``.
        """
        return self._loop_analytical_flags.get(characteristic_name, {}).get(label_name, True)

    def approximate(
        self,
        methods: Mapping[GenericCharacteristicName, CharacteristicApproximationMethod],
        **options: Any,
    ) -> ApproximatedDistribution:
        """
        Approximate selected characteristics of the current derivation.

        Parameters
        ----------
        methods : Mapping[GenericCharacteristicName, CharacteristicApproximationMethod]
            Mapping from characteristic names to characteristic-level
            approximation methods.
        **options : Any
            Extra options forwarded to each approximation method.

        Returns
        -------
        ApproximatedDistribution
            Distribution with materialized approximations for selected
            characteristics.
        """
        if not methods:
            raise ValueError("At least one characteristic approximation method must be provided.")

        analytical_computations: dict[
            GenericCharacteristicName, AnalyticalComputation[Any, Any]
        ] = {}
        for characteristic_name, method in methods.items():
            computation = method.approximate(
                self,
                **options,
            )
            if computation.target != characteristic_name:
                raise ValueError(
                    "Approximation method returned computation for a mismatched "
                    f"target: expected '{characteristic_name}', got '{computation.target}'."
                )
            analytical_computations[characteristic_name] = computation

        return ApproximatedDistribution(
            distribution_type=self.distribution_type,
            analytical_computations=analytical_computations,
            support=self.support,
            sampling_strategy=self.sampling_strategy,
            computation_strategy=self.computation_strategy,
        )

    @abstractmethod
    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> DerivedDistribution:
        """
        Return a copy of the derived distribution with updated strategies.

        Concrete subclasses must preserve their own transformation
        parameters while applying strategy overrides.
        """
        _ = sampling_strategy, computation_strategy
        raise NotImplementedError


class ApproximatedDistribution(DerivedDistribution):
    """
    Derived distribution whose analytical computations were materialized by an
    external approximator.

    Parameters
    ----------
    distribution_type : DistributionType
        Type descriptor of the approximated distribution.
    analytical_computations : Mapping[
        GenericCharacteristicName,
        (
            AnalyticalComputation[Any, Any]
            | Mapping[LabelName, AnalyticalComputation[Any, Any]]
        ),
    ]
        Materialized methods produced by the approximator.
        They are exposed in ``analytical_computations`` for strategy
        resolution, but are never treated as fully analytical.
    support : Support | None, optional
        Support of the approximated distribution.
    sampling_strategy : SamplingStrategy | None, optional
        Sampling strategy to expose.
    computation_strategy : ComputationStrategy | None, optional
        Characteristic resolution strategy.
    """

    def __init__(
        self,
        *,
        distribution_type: DistributionType,
        analytical_computations: Mapping[
            GenericCharacteristicName,
            (AnalyticalComputation[Any, Any] | Mapping[LabelName, AnalyticalComputation[Any, Any]]),
        ],
        support: Support | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        computation_strategy: ComputationStrategy | None = None,
    ) -> None:
        super().__init__(
            distribution_type=distribution_type,
            bases={},
            analytical_computations=analytical_computations,
            transformation_name=TransformationName.APPROXIMATION,
            support=support,
            sampling_strategy=sampling_strategy,
            computation_strategy=computation_strategy,
            loop_analytical_flags={},
        )
        self._loop_analytical_flags = self._build_non_analytical_loop_flags(
            self.analytical_computations
        )

    @staticmethod
    def _build_non_analytical_loop_flags(
        analytical_computations: Mapping[
            GenericCharacteristicName,
            Mapping[LabelName, AnalyticalComputation[Any, Any]],
        ],
    ) -> dict[GenericCharacteristicName, dict[LabelName, bool]]:
        """Build loop flags where every approximation loop is non-analytical."""
        return {
            characteristic_name: dict.fromkeys(labeled_methods, False)
            for characteristic_name, labeled_methods in analytical_computations.items()
        }

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> ApproximatedDistribution:
        """Return a copy of the approximated distribution with updated strategies."""
        return ApproximatedDistribution(
            distribution_type=self.distribution_type,
            analytical_computations=self.analytical_computations,
            support=self.support,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
        )


__all__ = [
    "ApproximatedDistribution",
    "DerivedDistribution",
    "LightweightDistribution",
]
