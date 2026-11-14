"""Space conversion between value spaces of characteristics.

The converter performs only clean identities (``S = 1 - F`` and back), is
stateless, has no access to value sources, and is the *last resort*: the
runner converts only when the region's question space differs from the
native space of a side.  Conversion cannot repair a precision loss — it can
only avoid adding one and make the fact visible; every conversion is
recorded as a fact in the result.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import NDArray

from pysatl_core_bench.accuracy.types import PROBABILITY_SPACE, SURVIVAL_SPACE

Float64Array = NDArray[np.float64]

__all__ = ["SpaceConverter"]


class SpaceConverter:
    """Stateless value-space converter (probability <-> survival in v1)."""

    def convert(self, values: Float64Array, from_space: str, to_space: str) -> Float64Array:
        """Convert ``values`` from one value space to another."""
        arr = np.asarray(values, dtype=np.float64)
        if from_space == to_space:
            return arr.copy()
        pair = {from_space, to_space}
        if pair == {PROBABILITY_SPACE, SURVIVAL_SPACE}:
            return np.float64(1.0) - arr
        msg = f"unsupported space conversion: {from_space!r} -> {to_space!r}"
        raise ValueError(msg)

    def conversion_fact(self, from_space: str, to_space: str) -> dict[str, str]:
        """The fact record stored in provenance when a conversion happens."""
        return {"from_space": from_space, "to_space": to_space}
