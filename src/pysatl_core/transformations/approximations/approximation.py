"""
Protocols for characteristic-level approximation methods.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pysatl_core.distributions.computation import AnalyticalComputation
    from pysatl_core.transformations.distribution import DerivedDistribution


class CharacteristicApproximationMethod(Protocol):
    """
    Protocol for a single characteristic approximation method.

    Implementations are responsible only for one characteristic and can
    use any numeric approximation strategy (interpolation, splines,
    tabulation, etc.).
    """

    def approximate(
        self,
        distribution: DerivedDistribution,
        **options: Any,
    ) -> AnalyticalComputation[Any, Any]:
        """
        Build an analytical computation for a target characteristic.

        Parameters
        ----------
        distribution : DerivedDistribution
            Distribution to approximate.
        **options : Any
            Extra approximation options.

        Returns
        -------
        AnalyticalComputation[Any, Any]
            Approximate analytical computation for the target
            characteristic.
        """
        ...


__all__ = [
    "CharacteristicApproximationMethod",
]
