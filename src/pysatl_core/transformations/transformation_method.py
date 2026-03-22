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

    @staticmethod
    def _source_status(
        base: Distribution,
        characteristic: GenericCharacteristicName,
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
            ``(is_present, is_analytical)`` for the first loop variant of
            the characteristic.
        """
        methods = base.analytical_computations.get(characteristic)
        if methods is None:
            return False, False

        first_label = next(iter(methods))
        return True, base.loop_is_analytical(characteristic, first_label)

    @classmethod
    def try_from_parents(
        cls,
        *,
        target: GenericCharacteristicName,
        transformation: TransformationName,
        bases: Mapping[ParentRole, Distribution],
        source_requirements: SourceRequirements,
        evaluator: TransformationEvaluator[In, Out],
    ) -> tuple[TransformationMethod[In, Out] | None, bool, bool]:
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

        Returns
        -------
        tuple[TransformationMethod[In, Out] | None, bool, bool]
            ``(method, is_analytical, has_any_present_source)``:

            - ``method`` is ``None`` when no required source characteristic is
              present in ``analytical_computations`` of parents.
            - ``is_analytical`` is ``True`` only when all required sources are
              present and marked analytical.
            - ``has_any_present_source`` indicates whether at least one required
              source is present in parent ``analytical_computations``.
        """
        has_any_present_source = False
        is_analytical = True

        for role, characteristics in source_requirements.items():
            base = bases[role]
            for characteristic in characteristics:
                is_present, source_is_analytical = cls._source_status(base, characteristic)
                has_any_present_source = has_any_present_source or is_present
                is_analytical = is_analytical and is_present and source_is_analytical

        if not has_any_present_source:
            return None, False, False

        resolved: ResolvedSourceMethods = {
            role: {
                characteristic: bases[role].query_method(characteristic)
                for characteristic in characteristics
            }
            for role, characteristics in source_requirements.items()
        }
        return (
            cls(
                target=target,
                func=evaluator(resolved),
                transformation=transformation,
                source_requirements=source_requirements,
            ),
            is_analytical,
            True,
        )


__all__ = [
    "ResolvedSourceMethods",
    "SourceRequirements",
    "TransformationEvaluator",
    "TransformationMethod",
]
