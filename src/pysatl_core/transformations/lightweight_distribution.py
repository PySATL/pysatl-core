"""
Lightweight distribution snapshot for transformation pipelines.

The snapshot keeps only metadata required by transformations and
strategies, while avoiding strong references to full parent
distribution objects.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import _KEEP, Distribution
from pysatl_core.distributions.strategies import (
    ComputationStrategy,
    SamplingStrategy,
)
from pysatl_core.types import (
    DistributionType,
    GenericCharacteristicName,
    LabelName,
    ParentRole,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support


class LightweightDistribution(Distribution):
    """
    Lightweight ``Distribution`` implementation for transformation internals.

    Parameters
    ----------
    distribution_type : DistributionType
        Type descriptor of the distribution.
    analytical_computations : Mapping[
        GenericCharacteristicName,
        (
            AnalyticalComputation[Any, Any]
            | Mapping[LabelName, AnalyticalComputation[Any, Any]]
        ),
    ]
        Labeled characteristic methods exposed by the snapshot.
    support : Support | None, optional
        Support metadata copied from the source distribution.
    sampling_strategy : SamplingStrategy | None, optional
        Sampling strategy attached to the snapshot.
    computation_strategy : ComputationStrategy | None, optional
        Computation strategy attached to the snapshot.
    bases : Mapping[ParentRole, LightweightDistribution] | None, optional
        Lightweight base snapshots for chained transformations.
    loop_analytical_flags : Mapping[
        GenericCharacteristicName,
        Mapping[LabelName, bool],
    ] | None, optional
        Optional analytical flags for loop variants.
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
        bases: Mapping[ParentRole, LightweightDistribution] | None = None,
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
        self._bases = dict(bases or {})
        self._loop_analytical_flags = {
            characteristic_name: dict(flags_by_label)
            for characteristic_name, flags_by_label in (loop_analytical_flags or {}).items()
        }

    @classmethod
    def from_distribution(cls, distribution: Distribution) -> LightweightDistribution:
        """
        Build a lightweight snapshot from an arbitrary distribution.

        The method copies only strategy-relevant fields and recursively
        snapshots known bases when the source distribution exposes them.

        Parameters
        ----------
        distribution : Distribution
            Source distribution.

        Returns
        -------
        LightweightDistribution
            Lightweight snapshot compatible with ``Distribution``.
        """
        if isinstance(distribution, cls):
            return distribution
        return cls._from_distribution(distribution, memo={})

    @classmethod
    def _from_distribution(
        cls,
        distribution: Distribution,
        memo: dict[int, LightweightDistribution],
    ) -> LightweightDistribution:
        """
        Internal recursive snapshot builder with memoization.
        """
        if isinstance(distribution, cls):
            return distribution

        distribution_id = id(distribution)
        cached = memo.get(distribution_id)
        if cached is not None:
            return cached

        snapshot = cls(
            distribution_type=distribution.distribution_type,
            analytical_computations=distribution.analytical_computations,
            support=distribution.support,
            sampling_strategy=distribution.sampling_strategy,
            computation_strategy=distribution.computation_strategy,
            bases={},
            loop_analytical_flags=cls._collect_loop_flags(distribution),
        )
        memo[distribution_id] = snapshot
        snapshot._bases = cls._collect_bases(distribution, memo)
        return snapshot

    @staticmethod
    def _collect_loop_flags(
        distribution: Distribution,
    ) -> dict[GenericCharacteristicName, dict[LabelName, bool]]:
        """Collect loop analytical flags from the source distribution."""
        return {
            characteristic_name: {
                label_name: distribution.loop_is_analytical(characteristic_name, label_name)
                for label_name in labeled_methods
            }
            for characteristic_name, labeled_methods in distribution.analytical_computations.items()
        }

    @classmethod
    def _collect_bases(
        cls,
        distribution: Distribution,
        memo: dict[int, LightweightDistribution],
    ) -> dict[ParentRole, LightweightDistribution]:
        """
        Recursively collect known base distributions for transformation chains.
        """
        maybe_bases = getattr(distribution, "bases", None)
        if not isinstance(maybe_bases, Mapping):
            return {}

        bases: dict[ParentRole, LightweightDistribution] = {}
        for role, base in maybe_bases.items():
            if isinstance(base, Distribution):
                bases[role] = cls._from_distribution(base, memo)
        return bases

    @property
    def bases(self) -> Mapping[ParentRole, Distribution]:
        """Get lightweight base snapshots grouped by role."""
        return self._bases

    def loop_is_analytical(
        self,
        characteristic_name: GenericCharacteristicName,
        label_name: LabelName,
    ) -> bool:
        """Return preserved loop analytical flag for the snapshot."""
        return self._loop_analytical_flags.get(characteristic_name, {}).get(label_name, True)

    def _clone_with_strategies(
        self,
        *,
        sampling_strategy: SamplingStrategy | None | object = _KEEP,
        computation_strategy: ComputationStrategy | None | object = _KEEP,
    ) -> LightweightDistribution:
        """Return a copy of the snapshot with updated strategies."""
        return LightweightDistribution(
            distribution_type=self.distribution_type,
            analytical_computations=self.analytical_computations,
            support=self.support,
            sampling_strategy=self._new_sampling_strategy(sampling_strategy),
            computation_strategy=self._new_computation_strategy(computation_strategy),
            bases=self._bases,
            loop_analytical_flags=self._loop_analytical_flags,
        )


__all__ = [
    "LightweightDistribution",
]
