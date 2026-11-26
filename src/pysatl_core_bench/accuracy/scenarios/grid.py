"""Grid generation.

Grids are built *per region* and belong to the region, not to a pair: one
grid per region, one prepare per participant, one metric per pair.  The
generator exists so that "where do we measure" is decided deterministically
from the scenario alone — a run is reproducible and two runs of the same
spec are comparable because they hold the same grids.

The measurement set of a query is the user's decision, not the library's:
the query either names its points explicitly (``points`` — the user's own
coordinates, measured exactly as declared) or asks for a generated grid
over the region intervals (``grid_hint.n_points`` or the default count).
Both forms may coexist in one query; when they do, the generated part and
the user's points are concatenated (generated first, user order preserved
— no densification, no reordering of the user coordinates; duplicates
inside the array are measured once, and the runner explains both
transformations with validator notes).

What the framework *offers on top of the user's request* is special-point
handling: it predicts the likely special points of the distribution
(``detection.py``, best effort — no completeness guarantee) and, unless
``grid_hint.densify: false``, densifies the *generated* part around them
(the extremes of the error usually live there).  The generator never
places points outside the distribution support.

Unbounded region sides (``None`` endpoints) are resolved against the
support; a side that stays unbounded (infinite support) is measured over
a deterministic extension beyond the finite anchor.  The length is
resolved by the runner from the distribution parameters (curated
per-family arithmetic, ``scenarios.tail``), and a query may override it
with ``grid_hint.tail_extension`` — the user asked for a concrete
tail geometry, the framework measures exactly that.  Unknown families
fall back to the fixed ``TAIL_EXTENSION`` constant, and every fallback
is reported (no silence).

Densification around framework-predicted special points is a default,
not a dogma: ``grid_hint.densify: false`` turns it off and the generated
part is exactly the uniform linspace (a user who wants a clean uniform
grid gets exactly ``n_points`` points, no special-point offsets).
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
from pysatl_core_bench.accuracy.errors import FrameworkDefect
from pysatl_core_bench.accuracy.scenarios.spec import AccuracyQuery, Region
from pysatl_core_bench.accuracy.scenarios.tail import (  # noqa: F401 (re-export)
    TAIL_EXTENSION,
)

__all__ = ["Grid", "GridGenerator", "TAIL_EXTENSION", "DENSIFICATION_OFFSETS"]

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Grid:
    """Runtime grid built for one (query, region); belongs to the region.

    The grid is the join key between the scenario file and the sidecar
    values: ``grid_hash`` is content-addressed over the sorted point set,
    so two grids agree iff they measure the same points.

    ``n_removed_out_of_support`` carries the AP-1 count of generated
    points the support filter dropped: the coordinates are reconstructible
    from the deterministic generator (and the drop is expected to be
    empty — the resolved interval lives inside the support), so only the
    *fact* is reported, never a coordinate listing.
    """

    query: AccuracyQuery
    region: Region
    points: Float64Array
    grid_hash: str
    n_removed_out_of_support: int = 0


# ``TAIL_EXTENSION`` moved to ``scenarios.tail`` (the tail policy lives
# there since the parameter-derived extension); re-imported here so the
# documented import path keeps working.

DENSIFICATION_OFFSETS: tuple[float, ...] = tuple(10.0**-k for k in range(1, 8))
"""Offsets (scaled by the point magnitude) added around special points.

Seven decades of relative offsets, so the neighbourhood of a special point
is sampled from 1e-1 down to 1e-7 of its magnitude — enough to expose a
defect at any realistic error floor without knowing that floor in advance.
"""

DEFAULT_N_POINTS = 129
"""Point count of the generated grid when the scenario does not hint one.

Odd on purpose: a linspace with an odd count has a midpoint, so bulk
regions get a point exactly in the middle of the interval.
"""


class GridGenerator:
    """Builds deterministic grids for every (query, region) cell.

    Deterministic means: the same (query, region, special points, support)
    always yields the same points, hence the same ``grid_hash`` — the
    property the comparator's agreement and the sidecar joinability rest
    on.  The class is stateless on purpose; runs never share grids.
    """

    def generate(
        self,
        inputs: list[AccuracyQuery],
        regions: list[Region],
        special_points: list[SpecialPoint] | None = None,
        support: SupportBounds | None = None,
        tail_extension: float = TAIL_EXTENSION,
    ) -> list[Grid]:
        """Build one grid per (query, region); denser near special points.

        ``tail_extension`` is the runner-resolved default length for the
        unbounded sides (the per-family arithmetic lives in
        ``scenarios.tail``); a query overrides it per cell with
        ``grid_hint.tail_extension``.  Both are pure geometry inputs — the
        generator computes nothing and calls nobody.
        """
        grids: list[Grid] = []
        for query in inputs:
            tail = self._hint_tail(query, tail_extension)
            for region in regions:
                points, removed = self._points_for(
                    query, region, special_points or [], support, tail
                )
                grids.append(
                    Grid(
                        query=query,
                        region=region,
                        points=points,
                        grid_hash=grid_hash(points),
                        n_removed_out_of_support=removed,
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
        tail_extension: float,
    ) -> tuple[Float64Array, int]:
        """Points and removal count of one (query, region) cell.

        The generated part is skipped when the query declares ``points``
        without a ``grid_hint`` — the user asked for exactly their points,
        and a default grid would dilute the answer with points nobody
        asked for.  With both present the two parts are concatenated
        (generated first, explicit second; user order preserved): the
        grid and the user's points coexist in one run by design.  The
        removal count is the AP-1 fact the runner reports (no silence).
        """
        generated, removed = self._generated_points(
            query, region, special_points, support, tail_extension
        )
        explicit = self._explicit_points(query, region, support)
        if generated.size == 0:
            return explicit, 0
        if explicit.size == 0:
            return generated, removed
        return np.concatenate([generated, explicit]), removed

    def _generated_points(
        self,
        query: AccuracyQuery,
        region: Region,
        special_points: list[SpecialPoint],
        support: SupportBounds | None,
        tail_extension: float,
    ) -> tuple[Float64Array, int]:
        """The hint-driven part: linspace, densification, support clip.

        Empty when the query relies on explicit points alone (no
        ``grid_hint``) or when the region resolves to nothing against the
        runtime support (REV-2: the runner reports that as an honest empty
        outcome, not a crash).  ``grid_hint.densify: false`` skips the
        special-point densification entirely — the generated part is then
        exactly the uniform linspace.
        """
        if query.grid_hint is None and query.points is not None:
            return np.empty(0, dtype=np.float64), 0
        if query.characteristic == CharacteristicName.PPF:
            # Quantile levels live on the probability axis (0, 1); the
            # distribution support bounds the *argument* axis and is
            # meaningless here (REV-2: an empty mask for a support entirely
            # right of 1 used to crash the whole run).
            return self._ppf_points(query, region), 0
        base = self._domain_points(query, region, support, tail_extension)
        if not self._hint_densify(query):
            densified = base
        else:
            densified = self._densify(base, region, special_points)
        return self._keep_in_support(densified, support)

    def _domain_points(
        self,
        query: AccuracyQuery,
        region: Region,
        support: SupportBounds | None,
        tail_extension: float,
    ) -> Float64Array:
        """Evenly spaced points over the region interval resolved to the support."""
        resolved = self._resolve_interval(region, support, tail_extension)
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

    def _explicit_points(
        self, query: AccuracyQuery, region: Region, support: SupportBounds | None
    ) -> Float64Array:
        """The user-declared ``points`` of one query, filtered for the cell.

        The array enters the grid as declared: order preserved, no sorting,
        no densification, duplicates inside the array measured once (first
        occurrence wins — the runner reports the drop).  The only filters
        applied are cell membership (the raw region interval) and — for
        non-ppf queries — the runtime support, because the generator's
        contract is to never place points outside the support.  A ppf
        level is never support-filtered: the support bounds the argument
        axis, not the probability axis (REV-2).  Points dropped by the
        support filter are reported by the runner as validator notes
        (standalone form of the AP-1 excluded-points report).
        """
        if query.points is None:
            return np.empty(0, dtype=np.float64)
        deduped = self._deduplicate(query.points)
        lo = -math.inf if region.interval[0] is None else float(region.interval[0])
        hi = math.inf if region.interval[1] is None else float(region.interval[1])
        in_region = [x for x in deduped if lo <= x <= hi]
        if query.characteristic != CharacteristicName.PPF and support is not None:
            in_region = [x for x in in_region if support.left <= x <= support.right]
        if not in_region:
            return np.empty(0, dtype=np.float64)
        return np.asarray(in_region, dtype=np.float64)

    @staticmethod
    def _deduplicate(points: tuple[float, ...]) -> list[float]:
        """First-occurrence deduplication, order preserved.

        A repeated coordinate carries no extra information (same point,
        same values, max-based metrics unchanged) but would inflate the
        point counters; keeping the first occurrence keeps the array
        order — the join key between the scenario file and the sidecar —
        stable.
        """
        seen: set[float] = set()
        ordered: list[float] = []
        for x in points:
            if x not in seen:
                seen.add(x)
                ordered.append(x)
        return ordered

    def _resolve_interval(
        self, region: Region, support: SupportBounds | None, tail_extension: float
    ) -> tuple[float, float] | None:
        """Resolve the region interval; ``None`` = empty against the support.

        ``tail_extension`` is the effective length for an unbounded side on
        an infinite support: the runner-resolved per-family value, or the
        query's ``grid_hint.tail_extension`` override, or the documented
        fallback constant.
        """
        lo = -math.inf if region.interval[0] is None else float(region.interval[0])
        hi = math.inf if region.interval[1] is None else float(region.interval[1])
        if support is not None:
            lo = max(lo, self._inner_bound(support.left, support.left_closed, +1))
            hi = min(hi, self._inner_bound(support.right, support.right_closed, -1))
        if math.isinf(lo):
            lo = -tail_extension if math.isinf(hi) else hi - tail_extension
        if math.isinf(hi):
            hi = lo + tail_extension
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
        """Add relative-offset points around the special points inside the grid.

        Only the *generated* part is densified: offsets around a
        user-declared point would answer a question nobody asked (who
        wants a dense grid asks the grid; who wants their points asks
        ``points``).  The whole mechanism is opt-out per query
        (``grid_hint.densify: false``): the predicted points are the
        framework's best effort (detection carries no completeness
        guarantee), and a user who wants the untouched uniform geometry
        gets exactly that.
        """
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

    def _keep_in_support(
        self, points: Float64Array, support: SupportBounds | None
    ) -> tuple[Float64Array, int]:
        """Drop generated points outside the support; return the drop count.

        The resolved interval already lives inside the support, so a
        generated point can only land outside through float rounding at
        an open boundary — the filter is the guard for exactly that dust.
        The coordinates are reconstructible from the deterministic
        generator, so the AP-1 report carries the *count* (the runner
        turns it into a validator note), not a coordinate listing — in
        contrast to explicit user points, whose drops change the requested
        measurement set and are reported individually.
        """
        if support is None:
            return points, 0
        mask = (points >= support.left) & (points <= support.right)
        return points[mask], int((~mask).sum())

    @staticmethod
    def _hint_n(query: AccuracyQuery) -> int:
        """Point count from the hint, or the documented default."""
        hint = query.grid_hint
        if not hint:
            return DEFAULT_N_POINTS
        n = hint.get("n_points")
        if not isinstance(n, int) or isinstance(n, bool) or n < 4:
            msg = f"grid_hint.n_points must be an integer >= 4, got {n!r}"
            raise FrameworkDefect(msg)
        return n

    def _hint_tail(self, query: AccuracyQuery, tail_extension: float) -> float:
        """Effective tail length: the query override or the runner-resolved value.

        The loader validates ``grid_hint.tail_extension``; the check here
        is the same defense-in-depth as ``_hint_n`` — a spec built around
        the loader cannot carry an invalid hint, and one built by hand
        fails loudly instead of silently distorting the grid geometry.
        """
        hint = query.grid_hint
        if not hint or hint.get("tail_extension") is None:
            return tail_extension
        value = hint["tail_extension"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"grid_hint.tail_extension must be a positive finite number, got {value!r}"
            raise FrameworkDefect(msg)
        tail = float(value)
        if not math.isfinite(tail) or tail <= 0.0:
            msg = f"grid_hint.tail_extension must be a positive finite number, got {value!r}"
            raise FrameworkDefect(msg)
        return tail

    @staticmethod
    def _hint_densify(query: AccuracyQuery) -> bool:
        """Densification opt-out flag: ``densify: false`` keeps the uniform grid."""
        hint = query.grid_hint
        if not hint or hint.get("densify") is None:
            return True
        value = hint["densify"]
        if not isinstance(value, bool):
            msg = f"grid_hint.densify must be a boolean, got {value!r}"
            raise FrameworkDefect(msg)
        return value
