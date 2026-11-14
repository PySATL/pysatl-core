"""Grid generation.

Grids are built *per region* and belong to the region, not to a pair: one
grid per region, one prepare per participant, one metric per pair.  The
generator densifies points near framework-predicted special points and
never places points outside the distribution support.

Unbounded region sides (``None`` endpoints) are resolved against the
support; a side that stays unbounded (infinite support) is measured over a
fixed deterministic extension (``TAIL_EXTENSION``) beyond the finite
anchor — a documented v1 heuristic.  The only overridable knob is the
point count (``grid_hint.n_points``); the tail extension is not
scenario-configurable in v1.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pysatl_core.types import CharacteristicName
from pysatl_core_bench.accuracy.canonical import grid_hash
from pysatl_core_bench.accuracy.dto import SpecialPoint, SupportBounds
from pysatl_core_bench.accuracy.errors import FrameworkDefect  # noqa: F401 (ppf guard)
from pysatl_core_bench.accuracy.scenarios.spec import AccuracyQuery, Region

__all__ = ["Grid", "GridGenerator", "TAIL_EXTENSION", "DENSIFICATION_OFFSETS"]

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Grid:
    """Runtime grid built for one (query, region); belongs to the region."""

    query: AccuracyQuery
    region: Region
    points: Float64Array
    grid_hash: str


TAIL_EXTENSION = 10.0
"""Extension used when a region side is unbounded and the support is infinite."""

DENSIFICATION_OFFSETS: tuple[float, ...] = tuple(10.0**-k for k in range(1, 8))
"""Offsets (scaled by the point magnitude) added around special points."""

DEFAULT_N_POINTS = 129


class GridGenerator:
    """Builds deterministic grids for every (query, region) cell."""

    def generate(
        self,
        inputs: list[AccuracyQuery],
        regions: list[Region],
        special_points: list[SpecialPoint] | None = None,
        support: SupportBounds | None = None,
    ) -> list[Grid]:
        """Build one grid per (query, region); denser near special points."""
        grids: list[Grid] = []
        for query in inputs:
            for region in regions:
                points = self._points_for(query, region, special_points or [], support)
                grids.append(
                    Grid(
                        query=query,
                        region=region,
                        points=points,
                        grid_hash=grid_hash(points),
                    )
                )
        return grids

    # -- point construction -------------------------------------------------

    def _points_for(
        self,
        query: AccuracyQuery,
        region: Region,
        special_points: list[SpecialPoint],
        support: SupportBounds | None,
    ) -> Float64Array:
        if query.characteristic == CharacteristicName.PPF:
            # Quantile levels live on the probability axis (0, 1); the
            # distribution support bounds the *argument* axis and is
            # meaningless here (REV-2: an empty mask for a support entirely
            # right of 1 used to crash the whole run).
            return self._ppf_points(query, region)
        base = self._domain_points(query, region, support)
        densified = self._densify(base, region, special_points)
        return self._keep_in_support(densified, support)

    def _domain_points(
        self, query: AccuracyQuery, region: Region, support: SupportBounds | None
    ) -> Float64Array:
        resolved = self._resolve_interval(region, support)
        if resolved is None:
            # Region contradicts the runtime support (REV-2): the runner
            # reports it, it is not a framework crash.
            return np.empty(0, dtype=np.float64)
        lo, hi = resolved
        n = self._hint_n(query)
        return np.linspace(lo, hi, num=n, dtype=np.float64)

    def _ppf_points(self, query: AccuracyQuery, region: Region) -> Float64Array:
        """Quantile levels: the region interval lives in probability space."""
        lo = 0.0 if region.interval[0] is None else float(region.interval[0])
        hi = 1.0 if region.interval[1] is None else float(region.interval[1])
        lo, hi = max(lo, 0.0), min(hi, 1.0)
        if not lo < hi:
            msg = f"ppf region {region.name!r} is empty on (0, 1)"
            raise FrameworkDefect(msg)
        n = self._hint_n(query)
        levels = np.linspace(lo, hi, num=n, dtype=np.float64)
        # Quantiles at exactly 0 or 1 are infinite; keep levels strictly inside.
        levels = np.clip(levels, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
        return levels

    def _resolve_interval(
        self, region: Region, support: SupportBounds | None
    ) -> tuple[float, float] | None:
        """Resolve the region interval; ``None`` = empty against the support."""
        lo = -math.inf if region.interval[0] is None else float(region.interval[0])
        hi = math.inf if region.interval[1] is None else float(region.interval[1])
        if support is not None:
            lo = max(lo, self._inner_bound(support.left, support.left_closed, +1))
            hi = min(hi, self._inner_bound(support.right, support.right_closed, -1))
        if math.isinf(lo):
            lo = -TAIL_EXTENSION if math.isinf(hi) else hi - TAIL_EXTENSION
        if math.isinf(hi):
            hi = lo + TAIL_EXTENSION
        if not lo < hi:
            return None
        return lo, hi

    @staticmethod
    def _inner_bound(bound: float, closed: bool, direction: int) -> float:
        """Open-boundary adjustment; infinite bounds pass through untouched."""
        if math.isinf(bound) or closed:
            return bound
        return float(np.nextafter(bound, math.inf * direction))

    def _densify(
        self, points: Float64Array, region: Region, special_points: list[SpecialPoint]
    ) -> Float64Array:
        if points.size == 0:
            # REV-2: an already-empty base (region outside the support) has
            # no bounds to densify around.
            return points
        lo, hi = points.min(), points.max()
        extra: list[float] = []
        for special in special_points:
            if not lo < special.x < hi:
                continue
            scale = max(1.0, abs(special.x))
            for offset in DENSIFICATION_OFFSETS:
                step = offset * scale
                for candidate in (special.x - step, special.x + step):
                    if lo <= candidate <= hi:
                        extra.append(float(candidate))
            if lo <= special.x <= hi:
                extra.append(float(special.x))
        if not extra:
            return points
        merged = np.unique(np.concatenate([points, np.asarray(extra, dtype=np.float64)]))
        return np.asarray(merged, dtype=np.float64)

    @staticmethod
    def _keep_in_support(points: Float64Array, support: SupportBounds | None) -> Float64Array:
        if support is None:
            return points
        mask = (points >= support.left) & (points <= support.right)
        return points[mask]

    @staticmethod
    def _hint_n(query: AccuracyQuery) -> int:
        hint = query.grid_hint
        if not hint:
            return DEFAULT_N_POINTS
        n = hint.get("n_points")
        if not isinstance(n, int) or isinstance(n, bool) or n < 4:
            msg = f"grid_hint.n_points must be an integer >= 4, got {n!r}"
            raise FrameworkDefect(msg)
        return n
