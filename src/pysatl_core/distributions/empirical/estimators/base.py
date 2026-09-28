"""Contract for fitting an empirical distribution estimate."""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.types import DistributionType, GenericCharacteristicName

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support


class EmpiricalDistributionEstimator(Protocol):
    """Procedure that fits observations to produce a distribution estimate.

    ``fit`` returns the estimate's direct characteristics; ``EmpiricalDistribution``
    uses them to represent the resulting distribution.
    """

    @property
    def distribution_type(self) -> DistributionType:
        """Type used to select applicable characteristic graph edges."""
        ...

    def resolve_support(
        self, sample: NDArray[np.float64], support: Support | None
    ) -> Support | None:
        """Validate an explicit support or derive one from the observations."""
        ...

    def fit(
        self, sample: NDArray[np.float64]
    ) -> Mapping[GenericCharacteristicName, AnalyticalComputation[Any, Any]]:
        """Return the characteristics available directly after fitting."""
        ...
