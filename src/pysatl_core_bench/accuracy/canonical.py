"""Canonical forms and content hashes.

Every hash in the framework is computed over a *canonical* representation:
key order is normalized, all numbers are represented uniformly, and floats
are serialized through their exact hexadecimal form.  Without this, two
equivalent configurations could produce different hashes and a cached state
could miss on an equivalent input.

Adopted detail of the joinability decision (D1): ``grid_hash`` is computed
over the *sorted* point set, each point rendered with ``float.hex()``, so
two grids agree iff they contain the same points.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import hashlib
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]

__all__ = ["canonical_json", "frozen_mapping", "grid_hash", "values_hash", "content_hash"]


def frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze a JSON-like mapping (REV-20).

    A ``frozen`` dataclass with a ``dict`` field is only *shallowly*
    immutable: nothing in the framework mutates these mappings, but a
    consumer holding a ``ParticipantSpec`` could mutate ``adapter_config``
    after loading and silently change the participant identity behind a
    cached ``participant_id``.  Every ``Mapping``-typed field of the
    canonical specs and DTOs therefore passes through this helper at
    construction time; mutation attempts fail with ``TypeError``.  Equality,
    ``content_hash`` (see :func:`_canonicalize`) and ``dict(...)``
    conversion are unaffected.

    Lives in the kernel (like ``canonical`` hashing): both ``scenarios.spec``
    and ``dto`` freeze their mapping fields, and neither may import the
    other's package.
    """
    return MappingProxyType(
        {key: frozen_mapping(v) if isinstance(v, Mapping) else v for key, v in value.items()}
    )


def _canonicalize(value: Any) -> Any:
    """Return a JSON-safe structure with normalized numbers and ordered keys."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (str,)):
        return value
    if isinstance(value, (int,)):
        try:
            return float(value)
        except OverflowError as exc:
            msg = f"integer {value} is too large for a float"
            raise ValueError(msg) from exc
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"non-finite value {value!r} cannot be canonicalized"
            raise ValueError(msg)
        return value + 0.0  # REV-3: -0.0 and 0.0 hash identically
    if isinstance(value, np.integer):
        return float(value)
    if isinstance(value, np.floating):
        return _canonicalize(float(value))
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if isinstance(value, dict):
        return {
            str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, Mapping):
        # REV-20: frozen DTO mapping fields (MappingProxyType) hash exactly
        # like their plain-dict content — freezing must not change identity.
        return {
            str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize(v) for v in value), key=repr)
    msg = f"value of type {type(value).__name__} cannot be canonicalized"
    raise ValueError(msg)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` into its canonical JSON string form."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Stable content hash prefixed with the algorithm name."""
    digest = hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
    return f"sha256:{digest}"


def _float_hex(x: float) -> str:
    if math.isnan(x):
        return "nan"  # fixed literal: missing values hash deterministically
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return float(x).hex()


def grid_hash(points: Float64Array | list[float]) -> str:
    """Hash of a *set* of grid points (order-independent, float-exact)."""
    arr = np.asarray(points, dtype=np.float64).ravel()
    ordered = sorted(_float_hex(float(x)) for x in arr)
    payload = ",".join(ordered).encode("ascii")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def values_hash(values: Float64Array | list[float]) -> str:
    """Hash of a sequence of values; order matters (it mirrors the grid)."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    payload = ",".join(_float_hex(float(x)) for x in arr).encode("ascii")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
