"""
Transformation computation primitives.

This module defines analytical computations produced by distribution
transformations. A transformation method remains compatible with AnalyticalComputation
so that transformed distributions continue to participate in the
existing characteristic graph and computation strategy.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.computation import AnalyticalComputation, Method
from pysatl_core.types import (
    ComputationFunc,
    GenericCharacteristicName,
    ParentRole,
    TransformationName,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution

type SourceRequirements = dict[ParentRole, tuple[GenericCharacteristicName, ...]]
"""Required parent characteristics grouped by logical parent role."""

type ResolvedSourceMethods = dict[
    ParentRole,
    dict[GenericCharacteristicName, Method[Any, Any]],
]
"""Resolved parent methods grouped by logical parent role and characteristic."""

type TransformationEvaluator[In, Out] = Callable[[ResolvedSourceMethods], ComputationFunc[In, Out]]
"""Factory producing a bound computation function from resolved parent methods."""


@dataclass(frozen=True, slots=True)
class TransformationMethod[In, Out](AnalyticalComputation[In, Out]):
    """
    Analytical computation originating from a transformation.

    Parameters
    ----------
    target : GenericCharacteristicName
        Name of the target characteristic produced by the transformation.
    func : ComputationFunc[In, Out]
        Bound callable implementing the transformed characteristic.
    transformation : TransformationName
        Logical name of the transformation that created this computation.
    source_requirements : SourceRequirements
        Required parent characteristics used to build the computation.
    """

    transformation: TransformationName
    source_requirements: SourceRequirements = field(default_factory=dict)

    @classmethod
    def from_parents(
        cls,
        *,
        target: GenericCharacteristicName,
        transformation: TransformationName,
        bases: Mapping[ParentRole, Distribution],
        source_requirements: SourceRequirements,
        evaluator: TransformationEvaluator[In, Out],
    ) -> TransformationMethod[In, Out]:
        """
        Build a transformation method from parent distributions.

        Parent requirements are resolved through ``query_method()``. This
        allows a transformation to depend on either directly analytical
        parent characteristics or characteristics obtained from the parent
        computation graph.

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
        evaluator : TransformationEvaluator[InT, OutT]
            Factory producing the bound transformed computation from the
            resolved parent methods.

        Returns
        -------
        TransformationMethod[InT, OutT]
            Bound transformation method.
        """
        resolved: ResolvedSourceMethods = {
            role: {
                characteristic: bases[role].query_method(characteristic)
                for characteristic in characteristics
            }
            for role, characteristics in source_requirements.items()
        }

        return cls(
            target=target,
            func=evaluator(resolved),
            transformation=transformation,
            source_requirements=source_requirements,
        )


__all__ = [
    "ResolvedSourceMethods",
    "SourceRequirements",
    "TransformationEvaluator",
    "TransformationMethod",
]
