"""
Transformation computation primitives.

This module defines entry computations produced by distribution
transformations. A transformation method remains compatible with AnalyticalComputation
so that transformed distributions continue to participate in the
existing characteristic graph and computation strategy.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.types import (
    GenericCharacteristicName,
    ParentRole,
    ResolvedSourceMethods,
    SourceRequirements,
    TransformationEvaluator,
    TransformationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


class TransformationMethod[In, Out](AnalyticalComputation[In, Out]):
    """
    Entry computation originating from a transformation.

    Parameters
    ----------
    target : GenericCharacteristicName
        Name of the target characteristic produced by the transformation.
    transformation : TransformationName
        Logical name of the transformation that created this computation.
    bases : Mapping[ParentRole, Distribution]
        Parent distributions grouped by role.
    source_requirements : SourceRequirements
        Required parent characteristics used to build the computation.
    evaluator : TransformationEvaluator[In, Out]
        Factory producing a transformed computation callable.
    owner : object
        Transformation object passed to ``evaluator``.
    """

    __slots__ = (
        "transformation",
        "source_requirements",
        "_is_analytical",
    )

    transformation: TransformationName
    source_requirements: SourceRequirements
    _is_analytical: bool

    @staticmethod
    def _source_status(
        base: Distribution,
        characteristic: GenericCharacteristicName,
        selected_method: object,
    ) -> tuple[bool, bool]:
        """
        Resolve presence and analytical status for a parent characteristic.

        Parameters
        ----------
        base : Distribution
            Parent distribution.
        characteristic : GenericCharacteristicName
            Required characteristic name.

        Returns
        -------
        tuple[bool, bool]
            ``(is_present, is_analytical)`` for the method selected by the
            parent strategy. A conversion method is not an entry loop.
        """
        methods = base.entry_computations.get(characteristic)
        if methods is None:
            return False, False

        matching_labels = [label for label, method in methods.items() if method is selected_method]
        if not matching_labels:
            return True, False

        statuses = {base.loop_is_analytical(characteristic, label) for label in matching_labels}
        if len(statuses) == 1:
            return True, statuses.pop()

        # The same method can be exposed under labels with different flags.
        # In that case, only the parent strategy's pinned plan identifies the
        # label it selected.
        plan = base.explain_computation_path(characteristic)
        return True, len(plan.steps) == 1 and plan.steps[0].edge_kind == "analytical_loop"

    def __init__(
        self,
        *,
        target: GenericCharacteristicName,
        transformation: TransformationName,
        bases: Mapping[ParentRole, Distribution],
        source_requirements: SourceRequirements,
        evaluator: TransformationEvaluator[In, Out],
        owner: object,
    ) -> None:
        """
        Build a transformation method with source-semantics metadata.

        Parameters
        ----------
        target : GenericCharacteristicName
            Target characteristic produced by the method.
        transformation : TransformationName
            Logical transformation name.
        bases : Mapping[ParentRole, Distribution]
            Parent distributions grouped by role.
        source_requirements : SourceRequirements
            Required parent characteristics.
        evaluator : TransformationEvaluator[In, Out]
            Factory producing the bound transformed computation from resolved
            parent methods.
        owner : object
            Transformation object passed to ``evaluator``.

        Raises
        ------
        ValueError
            If none of required source characteristics is present in parent
            entry computations.
        """
        normalized_requirements: SourceRequirements = {
            role: tuple(characteristics) for role, characteristics in source_requirements.items()
        }
        has_any_present_source = False
        is_analytical = True

        for role, characteristics in normalized_requirements.items():
            base = bases[role]
            for characteristic in characteristics:
                has_any_present_source = (
                    has_any_present_source or characteristic in base.entry_computations
                )

        if not has_any_present_source:
            raise ValueError(
                "Transformation method requires at least one present source characteristic."
            )

        resolved: ResolvedSourceMethods = {
            role: {
                characteristic: bases[role].query_method(characteristic)
                for characteristic in characteristics
            }
            for role, characteristics in normalized_requirements.items()
        }
        for role, source_methods in resolved.items():
            base = bases[role]
            for characteristic, selected_method in source_methods.items():
                is_present, source_is_analytical = self._source_status(
                    base, characteristic, selected_method
                )
                is_analytical = is_analytical and is_present and source_is_analytical
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "func", evaluator(owner, resolved))
        object.__setattr__(self, "transformation", transformation)
        object.__setattr__(self, "source_requirements", normalized_requirements)
        object.__setattr__(self, "_is_analytical", is_analytical)

    @property
    def is_analytical(self) -> bool:
        """Return whether all required source loops are analytical."""
        return self._is_analytical


__all__ = [
    "TransformationMethod",
]
