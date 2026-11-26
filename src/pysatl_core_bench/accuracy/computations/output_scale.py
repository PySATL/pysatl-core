"""Output-scale conversion between value scales of characteristics.

The module exists because a question may be asked on a scale the answering
side does not natively speak (a region wants survival ``S``, the source
answers CDF ``F``).  Left to the sources, that would push a conversion
decision into every adapter; left to the caller, it would silently compare
values living on different axes.  The converter is the single, runner-side
place where the decision is made and made visible.

The converter performs only clean identities (``S = 1 - F`` and back), is
stateless, has no access to value sources, and is the *last resort*: the
runner converts only when the region's question scale differs from the
native scale of a side.  Conversion cannot repair a precision loss — it can
only avoid adding one and make the fact visible; every conversion is
recorded as a fact in the result.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import NDArray

from pysatl_core_bench.accuracy.types import PROBABILITY_SCALE, SURVIVAL_SCALE

Float64Array = NDArray[np.float64]

__all__ = ["OutputScaleConverter"]


class OutputScaleConverter:
    """Converts values between output scales (probability <-> survival in v1).

    The v1 identity set is deliberately minimal: every added identity is a
    new way to silently reinterpret an answer, so the converter grows only
    when the accuracy model can justify the identity and its loss profile.
    An unknown scale pair raises instead of guessing.
    """

    def convert(self, values: Float64Array, from_scale: str, to_scale: str) -> Float64Array:
        """Convert ``values`` from one output scale to another.

        Same-scale requests return a copy (never the input array), so the
        caller cannot mutate a cached value buffer through the result.
        """
        arr = np.asarray(values, dtype=np.float64)
        if from_scale == to_scale:
            return arr.copy()
        pair = {from_scale, to_scale}
        if pair == {PROBABILITY_SCALE, SURVIVAL_SCALE}:
            return np.float64(1.0) - arr
        msg = f"unsupported output_scale conversion: {from_scale!r} -> {to_scale!r}"
        raise ValueError(msg)

    def conversion_fact(self, from_scale: str, to_scale: str) -> dict[str, str]:
        """The fact record stored in provenance when a conversion happens.

        The fact is what makes the conversion auditable after the run: the
        reader of a result can see that the compared values are not the
        source's native answers but a derived scale.
        """
        return {"from_scale": from_scale, "to_scale": to_scale}
