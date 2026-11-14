"""Special point prediction.

Special points are *predicted by the framework*, never declared in the
scenario JSON and never taken from the output of the measured
implementation (N1).  V1 rules: support boundaries from the public API
plus curated per-family knowledge; the numerical fallback against a
trusted reference is deliberately deferred.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from collections.abc import Mapping
from typing import Any

from pysatl_core_bench.accuracy.dto import SpecialPoint, SupportBounds
from pysatl_core_bench.accuracy.scenarios.spec import DistributionSpec

__all__ = ["SpecialPointDetector"]

KIND_SUPPORT_BOUNDARY = "support_boundary"
KIND_PDF_JUMP = "pdf_jump"
KIND_SINGULARITY = "singularity"

SOURCE_SUPPORT = "support"
SOURCE_CURATED = "curated_rules"


class SpecialPointDetector:
    """Predicts special points for grid densification and point policies."""

    def detect(
        self, distribution: DistributionSpec, support: SupportBounds | None
    ) -> list[SpecialPoint]:
        """Predict special points of the scenario distribution (ред. 40).

        The signature takes the distribution spec (not a participant, not a
        library distribution object) because the runner never holds a
        library object: sources own their computational objects and only
        expose the public API.  The distribution is scenario-level state,
        shared by every participant of the scenario.
        """
        points: list[SpecialPoint] = []
        if support is not None:
            for bound in (support.left, support.right):
                if math.isfinite(bound):
                    points.append(
                        SpecialPoint(
                            x=float(bound),
                            kind=KIND_SUPPORT_BOUNDARY,
                            source=SOURCE_SUPPORT,
                        )
                    )
        points.extend(self._curated(distribution.family, distribution.parameters, support))
        seen: set[tuple[float, str]] = set()
        unique: list[SpecialPoint] = []
        for point in points:
            key = (point.x, point.kind)
            if key not in seen:
                seen.add(key)
                unique.append(point)
        return unique

    def _curated(
        self, family: str, parameters: Mapping[str, Any], support: SupportBounds | None
    ) -> list[SpecialPoint]:
        """Curated rules for known builtin families (v1)."""
        parameters = parameters or {}
        left = support.left if support else -math.inf
        # REV-5: exact comparison — the boundary value comes from the public
        # support API of Core (an exact constant), and isclose with its
        # default tolerance would declare "support at 0" for a support at
        # 1e-10.  The gamma shape lookup ignores ``parametrization`` — a
        # documented v1 heuristic (see 08_smoke_scenarios §2.3).
        if family == "Exponential" and left == 0.0:
            return [SpecialPoint(0.0, KIND_PDF_JUMP, SOURCE_CURATED)]
        if family == "Gamma" and left == 0.0:
            shape = self._shape(parameters)
            kind = KIND_SINGULARITY if shape is not None and shape < 1.0 else KIND_PDF_JUMP
            return [SpecialPoint(0.0, kind, SOURCE_CURATED)]
        if family == "ContinuousUniform":
            # Both endpoints are already support boundaries; pdf jumps there
            # coincide with them, no extra points are needed.
            return []
        return []

    @staticmethod
    def _shape(parameters: Mapping[str, Any]) -> float | None:
        """Extract the gamma shape parameter if the parametrization has one."""
        for key in ("shape", "alpha", "k"):
            value = parameters.get(key)
            if isinstance(value, (int, float)) and float(value) > 0.0:
                return float(value)
        return None
