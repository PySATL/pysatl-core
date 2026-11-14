"""Results store: the persistence sink between pipeline stages.

Writers are the runner (pair results) and the reporter (rendered reports);
the comparator only reads (В-4a).  V1 keeps everything in memory and, when
constructed with a ``root`` directory, mirrors results, sidecars and
reports to JSON files.  Instances are independent, so parallel runs in one
process stay isolated; a shared instance is safe too (internal lock).
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import json
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysatl_core_bench.accuracy.canonical import values_hash
from pysatl_core_bench.accuracy.dto import AccuracyResult, Report
from pysatl_core_bench.accuracy.errors import FrameworkDefect

Float64Array = NDArray[np.float64]

__all__ = ["ResultsStore"]


class ResultsStore:
    """In-memory (optionally file-backed) store of results and sidecars."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._results: dict[str, list[AccuracyResult]] = {}
        self._reports: dict[str, Report] = {}
        self._sidecars: dict[str, Float64Array] = {}
        self._root = Path(root) if root is not None else None
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)

    # -- results ---------------------------------------------------------------

    def write_result(self, result: AccuracyResult) -> None:
        """Runner-side write of one pair result.

        One result per (scenario, pair) is the invariant (D3/В-7; REV-5
        refines the key: sweep/chain variants of one group legitimately
        repeat the same pair under different ``scenario_id``).  A duplicate
        write raises instead of appending silently.
        """
        with self._lock:
            group = self._results.setdefault(result.comparison_group_id, [])
            if any(
                existing.pair == result.pair and existing.scenario_id == result.scenario_id
                for existing in group
            ):
                msg = (
                    f"duplicate result for pair {result.pair} of scenario "
                    f"{result.scenario_id!r} in group {result.comparison_group_id!r}"
                )
                raise FrameworkDefect(msg)
            group.append(result)
            if self._root is not None:
                self._dump_result(result)

    def read_group(self, group_id: str) -> list[AccuracyResult]:
        """Comparator-side read of all results of one comparison group."""
        with self._lock:
            return list(self._results.get(group_id, ()))

    # -- reports -----------------------------------------------------------------

    def write_report(self, report: Report) -> None:
        """Reporter-side write of the rendered report."""
        with self._lock:
            self._reports[report.comparison_group_id] = report
            if self._root is not None:
                self._dump_json(
                    self._root / f"report-{report.comparison_group_id}.json",
                    report.to_json_dict(),
                )

    def read_report(self, group_id: str) -> Report | None:
        with self._lock:
            return self._reports.get(group_id)

    # -- sidecars ------------------------------------------------------------------

    def put_sidecar(self, values: Float64Array, space: str) -> tuple[str, str]:
        """Store pointwise values; returns ``(uri, hash)``.

        Sidecars keep pointwise curves out of the result rows; the URI is
        stable and content-addressed, so repeated measurements deduplicate.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        digest = values_hash(arr)
        uri = f"sidecar:{digest}"
        with self._lock:
            self._sidecars.setdefault(digest, arr)
            if self._root is not None:
                self._dump_json(
                    self._root / f"sidecar-{digest[:24]}.json",
                    {"space": space, "values": arr.tolist()},
                )
        return uri, digest

    def get_sidecar(self, uri: str) -> Float64Array | None:
        if not uri.startswith("sidecar:"):
            return None
        with self._lock:
            arr = self._sidecars.get(uri.removeprefix("sidecar:"))
            # REV-5: callers must not be able to mutate the stock through
            # the returned reference.
            return None if arr is None else arr.copy()

    # -- file persistence -------------------------------------------------------------

    def _dump_result(self, result: AccuracyResult) -> None:
        directory = self._root / result.comparison_group_id if self._root else None
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        safe_id = result.scenario_id.replace("/", "_")
        self._dump_json(
            directory / f"result-{safe_id}-{result.pair[0]}--{result.pair[1]}.json",
            result.to_json_dict(),
        )

    @staticmethod
    def _dump_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(
                _nan_to_null(payload),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )


def _nan_to_null(value: Any) -> Any:
    """Map non-finite floats to ``null`` for strict-JSON files (REV-5).

    Pointwise sidecars carry NaN holes of partial table coverage (D6); the
    holes are counted in the result row, the file keeps them explicit nulls
    instead of the invalid-JSON ``NaN`` token.
    """
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Mapping):
        # REV-20 (persistence fix): frozen DTOs keep their dict fields as
        # ``mappingproxy`` (``frozen_mapping``); the shallow ``dict(...)`` in
        # ``to_json_dict`` does not unfold *nested* mappings, so any
        # mapping value — not just ``dict`` — must be materialized here,
        # otherwise ``json.dumps`` raises on the persisted artifact.
        return {k: _nan_to_null(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_nan_to_null(v) for v in value]
    return value
