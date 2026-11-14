"""Reference-side value sources: precomputed tables and on-demand computers.

Naming (ред. 40): :class:`PrecomputedTable` is the *data* — an immutable
reference table with provenance; :class:`TableLookupSource` is the
*service* — the ``ValueSource`` that looks values up in such a table and
reports honest coverage (``missing``).  A table is an optimization, never
a requirement, and a compute-capable source fills the gaps.
``OnDemandComputer`` wraps any other source, adds a content-addressed value
cache (В-5: reference values are deterministic, so the cache is shared
across the scenarios of a comparison group) and exposes both capabilities.
The cache key carries the *order* of the requested points (``values_hash``,
not the order-independent ``grid_hash``) and the distribution fingerprint:
values are aligned to the requested order, and the same source instance
serves whole sweeps whose scenarios differ only in the scenario-level
distribution, so a permuted request can never be served another order's
array and one distribution's values can never be served for another.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import json
import math
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pysatl_core.types import NumericArray
from pysatl_core_bench.accuracy.canonical import grid_hash, values_hash
from pysatl_core_bench.accuracy.computations.protocol import PreparedSubject, ValueSource
from pysatl_core_bench.accuracy.dto import Provenance, SupportBounds, SupportVerdict
from pysatl_core_bench.accuracy.errors import SourceOperationError
from pysatl_core_bench.accuracy.scenarios.spec import (
    AccuracyQuery,
    DistributionSpec,
    ParticipantSpec,
)
from pysatl_core_bench.accuracy.types import KNOWN_SPACES, Capability, Readiness

Float64Array = NDArray[np.float64]

TABLE_SCHEMA = "pysatl-core-bench.table/1"
"""Schema tag of the JSON reference-table format loaded by ``PrecomputedTable``."""

__all__ = [
    "ExternalFunction",
    "ExternalFunctionSource",
    "PrecomputedTable",
    "TABLE_SCHEMA",
    "TableLookupSource",
    "OnDemandComputer",
]


@dataclass(frozen=True)
class PrecomputedTable:
    """Immutable reference table with provenance (the *data*, ред. 40)."""

    characteristic: str
    space: str
    points: Float64Array
    values: Float64Array
    library: str = "precomputed"
    library_version: str = "0"
    family: str | None = None
    parametrization: str | None = None
    parameters: Mapping[str, Any] | None = None
    dps: int | None = None
    coverage_tol: float = 0.0

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64).ravel()
        values = np.asarray(self.values, dtype=np.float64).ravel()
        if points.size == 0 or points.size != values.size:
            msg = "table points/values must be non-empty arrays of equal length"
            raise ValueError(msg)
        order = np.argsort(points)
        object.__setattr__(self, "points", points[order])
        object.__setattr__(self, "values", values[order])

    @property
    def table_hash(self) -> str:
        """Fingerprint of the table content (points and values)."""
        return f"{grid_hash(self.points)}+{values_hash(self.values)}"

    # -- JSON round trip ------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Full-fidelity JSON payload of the table.

        ``json.dumps`` serializes float64 through its shortest exact
        round-trip repr, so points and values survive a save/load cycle
        bit-exactly; the ``table_hash`` is therefore stable across the
        round trip as well.
        """
        return {
            "schema": TABLE_SCHEMA,
            "characteristic": self.characteristic,
            "space": self.space,
            "library": self.library,
            "library_version": self.library_version,
            "family": self.family,
            "parametrization": self.parametrization,
            "parameters": dict(self.parameters) if self.parameters is not None else None,
            "dps": self.dps,
            "coverage_tol": self.coverage_tol,
            "points": [float(x) for x in self.points],
            "values": [float(x) for x in self.values],
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> PrecomputedTable:
        """Validate a JSON payload and rebuild the table from it."""
        if payload.get("schema") != TABLE_SCHEMA:
            msg = f"table schema must be {TABLE_SCHEMA!r}, got {payload.get('schema')!r}"
            raise ValueError(msg)
        characteristic = payload.get("characteristic")
        if not isinstance(characteristic, str) or not characteristic:
            msg = "table characteristic must be a non-empty string"
            raise ValueError(msg)
        space = payload.get("space")
        if space not in KNOWN_SPACES:
            msg = f"table space must be one of {KNOWN_SPACES}, got {space!r}"
            raise ValueError(msg)
        parameters = payload.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            msg = "table parameters must be an object or null"
            raise ValueError(msg)
        dps = payload.get("dps")
        if dps is not None and (not isinstance(dps, int) or isinstance(dps, bool)):
            msg = "table dps must be an integer or null"
            raise ValueError(msg)
        coverage_tol = payload.get("coverage_tol", 0.0)
        if not isinstance(coverage_tol, (int, float)) or isinstance(coverage_tol, bool):
            msg = "table coverage_tol must be a number"
            raise ValueError(msg)
        points = cls._finite_series(payload.get("points"), "points")
        values = cls._finite_series(payload.get("values"), "values")
        return cls(
            characteristic=characteristic,
            space=space,
            points=np.asarray(points, dtype=np.float64),
            values=np.asarray(values, dtype=np.float64),
            library=str(payload.get("library", "precomputed")),
            library_version=str(payload.get("library_version", "0")),
            family=payload.get("family"),
            parametrization=payload.get("parametrization"),
            parameters=dict(parameters) if parameters is not None else None,
            dps=dps,
            coverage_tol=float(coverage_tol),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PrecomputedTable:
        """Load a table from a JSON file written by :meth:`to_json_dict`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            msg = f"table file {path} must contain a JSON object"
            raise ValueError(msg)
        return cls.from_json_dict(payload)

    @staticmethod
    def _finite_series(value: Any, what: str) -> list[float]:
        if not isinstance(value, list) or not value:
            msg = f"table {what} must be a non-empty list of numbers"
            raise ValueError(msg)
        numbers: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                msg = f"table {what} must contain numbers only, got {item!r}"
                raise ValueError(msg)
            number = float(item)
            if not math.isfinite(number):
                msg = f"table {what} must be finite, got {number!r}"
                raise ValueError(msg)
            numbers.append(number)
        return numbers


class TableLookupSource(ValueSource):
    """A lookup-only value source backed by a precomputed table.

    The service half of the ред. 40 rename: it *owns* one
    :class:`PrecomputedTable` (the data) and answers protocol calls by
    looking values up in it.  The table identity (family, parametrization,
    parameters) is checked against the scenario-level distribution, not
    against participant fields — one table serves every participant of a
    scenario honestly or refuses the whole scenario.
    """

    def __init__(self, source_id: str, table: PrecomputedTable) -> None:
        self._source_id = source_id
        self._table = table

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.LOOKUP})

    @property
    def subject_id(self) -> str:
        """Tables are not computational subjects; identity is the table hash."""
        return f"table:{self._table.table_hash[:32]}"

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            library=self._table.library,
            library_version=self._table.library_version,
            table_hash=self._table.table_hash,
        )

    @property
    def table(self) -> PrecomputedTable:
        return self._table

    def support(
        self, participant: ParticipantSpec, distribution: DistributionSpec
    ) -> SupportBounds:
        points = self._table.points
        return SupportBounds(
            left=float(points[0]), right=float(points[-1]), left_closed=True, right_closed=True
        )

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Float64Array | None = None,
    ) -> SupportVerdict:
        if participant.strategy is not None:
            # М-6/REV-5: a strategy on a lookup source is an *inapplicable
            # configuration* (the operation exists, the configuration does
            # not) — unsupported_configuration, not not_applicable.
            return SupportVerdict(
                readiness=Readiness.UNSUPPORTED_CONFIGURATION,
                reason=(
                    "a precomputed table cannot honor computation strategies "
                    f"(participant declared {participant.strategy!r})"
                ),
            )
        mismatch = self._identity_mismatch(distribution)
        if mismatch is not None:
            return SupportVerdict(
                readiness=Readiness.NOT_APPLICABLE,
                reason=(
                    "the table was generated for a different distribution "
                    f"identity: participant declares {mismatch}"
                ),
            )
        if query.characteristic != self._table.characteristic:
            return SupportVerdict(
                readiness=Readiness.NOT_APPLICABLE,
                reason=(
                    f"table holds {self._table.characteristic!r}, "
                    f"scenario asks {str(query.characteristic)!r}"
                ),
            )
        if query.space not in (self._table.space,):
            return SupportVerdict(
                readiness=Readiness.NOT_APPLICABLE,
                reason=(
                    f"table holds values in {self._table.space!r} space, "
                    f"query asks in {query.space!r}"
                ),
            )
        if points is None:
            return SupportVerdict(
                readiness=Readiness.READY, reason="table registered for this characteristic"
            )
        mask = self._missing_mask(points)
        uncovered = points[mask]
        if mask.all():
            return SupportVerdict(
                readiness=Readiness.NOT_COVERED,
                reason="the table covers none of the requested points",
                missing=tuple(float(x) for x in uncovered[:_MAX_REPORTED_MISSING]),
            )
        reason = (
            "all requested points are covered"
            if uncovered.size == 0
            else f"{uncovered.size} of {points.size} requested points are not covered"
        )
        return SupportVerdict(
            readiness=Readiness.READY,
            reason=reason,
            missing=tuple(float(x) for x in uncovered[:_MAX_REPORTED_MISSING]),
        )

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        return [self._table.space]

    def prepare(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
    ) -> PreparedSubject:
        return PreparedSubject(
            participant_id=participant.participant_id,
            query=query,
            distribution=distribution,
            payload=None,
        )

    def evaluate(self, prepared: PreparedSubject, points: Float64Array) -> Float64Array:
        arr = np.asarray(points, dtype=np.float64).ravel()
        mask = self._missing_mask(arr)
        values = self._lookup(arr)
        if mask.any():
            # The runner evaluates only covered points; this guard keeps the
            # uncovered holes explicit instead of silently returning junk.
            values = values.copy()
            values[mask] = np.nan
        return values

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Float64Array,
        distribution: DistributionSpec,
    ) -> Float64Array:
        msg = "a precomputed table has no compute capability"
        raise SourceOperationError(msg)

    # -- internals -----------------------------------------------------------

    def _identity_mismatch(self, distribution: DistributionSpec) -> str | None:
        """First identity field in which the scenario distribution and the table disagree.

        A table is generated for one distribution identity (family,
        parametrization, parameters); a scenario measuring a different
        one must never be served numbers of another distribution.  Fields
        the distribution leaves unset (``None``) are not checked.
        """
        if self._table.family is not None and str(distribution.family) != self._table.family:
            return f"family {distribution.family!r}"
        if self._table.parametrization is not None and (
            distribution.parametrization is not None
            and str(distribution.parametrization) != self._table.parametrization
        ):
            return f"parametrization {distribution.parametrization!r}"
        if self._table.parameters is not None:
            for key, value in self._table.parameters.items():
                if key in distribution.parameters and (
                    float(distribution.parameters[key]) != float(value)
                ):
                    return f"parameters[{key}]={distribution.parameters[key]!r}"
        return None

    def _missing_mask(self, points: Float64Array) -> NDArray[np.bool_]:
        tol = self._table.coverage_tol
        idx = np.searchsorted(self._table.points, points)
        idx_lo = np.clip(idx - 1, 0, self._table.points.size - 1)
        idx_hi = np.clip(idx, 0, self._table.points.size - 1)
        dist_lo = np.abs(points - self._table.points[idx_lo])
        dist_hi = np.abs(points - self._table.points[idx_hi])
        dist = np.minimum(dist_lo, dist_hi)
        return dist > tol

    def _lookup(self, points: Float64Array) -> Float64Array:
        idx = np.searchsorted(self._table.points, points)
        idx_lo = np.clip(idx - 1, 0, self._table.points.size - 1)
        idx_hi = np.clip(idx, 0, self._table.points.size - 1)
        dist_lo = np.abs(points - self._table.points[idx_lo])
        dist_hi = np.abs(points - self._table.points[idx_hi])
        nearest = np.where(dist_lo <= dist_hi, idx_lo, idx_hi)
        return np.asarray(self._table.values[nearest], dtype=np.float64)


_MAX_REPORTED_MISSING = 16


type ExternalFunction = Callable[
    [DistributionSpec, ParticipantSpec, AccuracyQuery, NumericArray], ArrayLike
]
"""A one-dimensional function of another library, wrapped for comparison.

The points argument is a pysatl-core ``NumericArray``; ``distribution``
(red. 40) carries the measured distribution (e.g. ``mu``/``sigma``),
``participant`` carries the adapter knobs (``adapter_config``) and
``query`` the requested characteristic, so one callable can serve
parametric libraries and several characteristics alike.
"""


class ExternalFunctionSource(ValueSource):
    """Value source delegating to a one-dimensional function of another library.

    The wrapped callable receives the query points as ``NumericArray`` and
    may return anything array-like; the source normalizes both sides to
    float64 arrays and verifies the shape.  This is the generic wrapper for
    external libraries (scipy, mpmath, ...) — no bespoke class per library:
    a fixed-parameter function is lifted with a closure
    (``lambda _p, _q, x: f(x)``).  Capabilities default to ``{compute}``;
    wrap the source in ``OnDemandComputer`` to add the shared lookup cache
    (В-5).  Computation strategies are refused with ``not_applicable``:
    plain library functions have no strategy knobs (М-6, no silent
    fallback).
    """

    def __init__(
        self,
        source_id: str,
        func: ExternalFunction,
        *,
        provenance: Provenance,
        characteristics: Collection[str] | None = None,
        capabilities: frozenset[Capability] = frozenset({Capability.COMPUTE}),
        native_spaces: Sequence[str] = ("probability",),
        support: SupportBounds = SupportBounds(-inf, inf, False, False),
        subject_id: str | None = None,
    ) -> None:
        self._source_id = source_id
        self._func = func
        self._provenance = provenance
        self._characteristics = characteristics
        self._capabilities = capabilities
        self._native_spaces = tuple(native_spaces)
        self._support = support
        self._subject_id = subject_id or f"{provenance.library}@{provenance.library_version}"

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._capabilities

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def provenance(self) -> Provenance:
        return self._provenance

    def support(
        self, participant: ParticipantSpec, distribution: DistributionSpec
    ) -> SupportBounds:
        return self._support

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Float64Array | None = None,
    ) -> SupportVerdict:
        if participant.strategy is not None:
            # М-6/REV-5: same rule as TableLookupSource — unsupported
            # configuration, not not_applicable.
            return SupportVerdict(
                readiness=Readiness.UNSUPPORTED_CONFIGURATION,
                reason=(
                    "an external library function cannot honor computation "
                    f"strategies (participant declared {participant.strategy!r})"
                ),
            )
        if self._characteristics is not None and (
            str(query.characteristic) not in self._characteristics
        ):
            return SupportVerdict(
                readiness=Readiness.NOT_APPLICABLE,
                reason=(
                    f"the wrapped function serves {sorted(self._characteristics)}, "
                    f"not {str(query.characteristic)!r}"
                ),
            )
        return SupportVerdict(readiness=Readiness.READY, reason="external function")

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        return list(self._native_spaces)

    def prepare(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
    ) -> PreparedSubject:
        # The participant is stashed so that evaluate() can pull the adapter
        # knobs exactly like compute() does; the distribution travels in the
        # prepared state (ред. 40).
        return PreparedSubject(
            participant_id=participant.participant_id,
            query=query,
            distribution=distribution,
            payload=participant,
        )

    def evaluate(self, prepared: PreparedSubject, points: Float64Array) -> Float64Array:
        participant = self._participant_of(prepared)
        if prepared.distribution is None:
            msg = "prepared state carries no distribution"
            raise SourceOperationError(msg)
        return self._call(prepared.distribution, participant, prepared.query, points)

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Float64Array,
        distribution: DistributionSpec,
    ) -> Float64Array:
        return self._call(distribution, participant, query, points)

    # -- internals -----------------------------------------------------------

    def _call(
        self,
        distribution: DistributionSpec,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Float64Array,
    ) -> Float64Array:
        arr = np.asarray(points, dtype=np.float64)
        values = np.asarray(self._func(distribution, participant, query, arr), dtype=np.float64)
        if values.shape != arr.shape:
            msg = f"external function returned shape {values.shape} for points of shape {arr.shape}"
            raise SourceOperationError(msg)
        return values

    @staticmethod
    def _participant_of(prepared: PreparedSubject) -> ParticipantSpec:
        payload = prepared.payload
        if not isinstance(payload, ParticipantSpec):
            msg = "prepared payload is not the stashed ParticipantSpec"
            raise SourceOperationError(msg)
        return payload


class OnDemandComputer(ValueSource):
    """Wraps any source; serves cached values as lookup, computes the rest.

    The cache is unbounded in v1 (REV-17: an eviction policy belongs to the
    future time/memory model, К-7) and thread-safe (А-9: the wrapped source
    may be shared by parallel runs); ``cache_info()`` reports its size and
    ``cache_clear()`` resets it between scenario batches.
    """

    def __init__(self, wrapped: ValueSource, source_id: str | None = None) -> None:
        self._wrapped = wrapped
        self._source_id = source_id or f"on-demand({wrapped.source_id})"
        self._cache: dict[tuple[str, str, str, str], Float64Array] = {}
        self._lock = threading.Lock()

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.COMPUTE, Capability.LOOKUP})

    @property
    def subject_id(self) -> str:
        return self._wrapped.subject_id

    @property
    def provenance(self) -> Provenance:
        return self._wrapped.provenance

    def support(
        self, participant: ParticipantSpec, distribution: DistributionSpec
    ) -> SupportBounds:
        return self._wrapped.support(participant, distribution)

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Float64Array | None = None,
    ) -> SupportVerdict:
        if (
            points is not None
            and self._cached(participant, query, distribution, points) is not None
        ):
            return SupportVerdict(readiness=Readiness.READY, reason="served from cache")
        return self._wrapped.assess(participant, query, distribution, points)

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        return self._wrapped.native_spaces(participant, query)

    def prepare(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
    ) -> PreparedSubject:
        return self._wrapped.prepare(participant, query, distribution)

    def evaluate(self, prepared: PreparedSubject, points: Float64Array) -> Float64Array:
        arr = np.asarray(points, dtype=np.float64).ravel()
        key = self._cache_key(prepared, arr)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached.copy()
        values = self._wrapped.evaluate(prepared, arr)
        with self._lock:
            # Double-check: a parallel evaluator may have filled the key
            # while this thread was computing the same request (А-9).
            cached = self._cache.get(key)
            if cached is None:
                self._cache[key] = values.copy()
                return values
        return cached.copy()

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Float64Array,
        distribution: DistributionSpec,
    ) -> Float64Array:
        return self._wrapped.compute(participant, query, points, distribution)

    def cache_info(self) -> dict[str, int]:
        """Current cache size (REV-17 executor knob; entry = one request order)."""
        with self._lock:
            return {"entries": len(self._cache)}

    def cache_clear(self) -> None:
        """Drop every cached value array (e.g. between scenario batches)."""
        with self._lock:
            self._cache.clear()

    def _cache_key(
        self, prepared: PreparedSubject, points: Float64Array
    ) -> tuple[str, str, str, str]:
        # values_hash is order-sensitive by design (D1): cached values are
        # aligned to the requested point order, so the key must not collapse
        # permutations of the same point set (no silent data swap).  The
        # distribution fingerprint is in the key (ред. 40): one shared
        # source serves whole sweeps whose scenarios differ only in the
        # scenario-level distribution.
        fingerprint = prepared.distribution.fingerprint() if prepared.distribution else ""
        return (
            prepared.participant_id,
            fingerprint,
            str(prepared.query.characteristic),
            values_hash(points),
        )

    def _cached(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Float64Array,
    ) -> Float64Array | None:
        key = (
            participant.participant_id,
            distribution.fingerprint(),
            str(query.characteristic),
            values_hash(points),
        )
        return self._cache.get(key)
