"""
Approximation interfaces for derived distributions.

This module intentionally keeps the approximation layer minimal.
At this stage the public abstraction is a single protocol describing
objects that can materialize an :class:`ApproximatedDistribution`
from a :class:`DerivedDistribution`.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pysatl_core.transformations.distribution import (
        ApproximatedDistribution,
        DerivedDistribution,
    )


class DistributionApproximator(Protocol):
    """
    Protocol for objects approximating a derived distribution.

    Implementations may use interpolation, tabulation, polynomial
    approximation, or any other technique, as long as they return a new
    derived distribution with materialized analytical computations.
    """

    def approximate(
        self,
        distribution: DerivedDistribution,
        **options: Any,
    ) -> ApproximatedDistribution:
        """
        Build an approximated distribution.

        Parameters
        ----------
        distribution : DerivedDistribution
            Distribution to approximate.
        **options : Any
            Extra approximation options.

        Returns
        -------
        ApproximatedDistribution
            Materialized approximation of the input distribution.
        """
        ...


__all__ = [
    "DistributionApproximator",
]
