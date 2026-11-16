"""Profiles and contracts for native backend values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pysatl_core.arrays.dtypes import DType


class ValueKind(StrEnum):
    """Kinds of native values handled by an array backend."""

    SCALAR = "scalar"
    ARRAY = "array"


@dataclass(frozen=True, slots=True)
class ValueProfile:
    """Backend-independent metadata extracted from a native value."""

    kind: ValueKind
    dtype: DType
    device: str
    shape: tuple[int, ...]
    ndim: int
    size: int


@dataclass(frozen=True, slots=True)
class ValueContract:
    """Input constraints checked at a public computation boundary."""

    allowed_dtypes: frozenset[DType] | None = None
    min_ndim: int = 0
    max_ndim: int | None = None


__all__ = [
    "ValueContract",
    "ValueKind",
    "ValueProfile",
]
