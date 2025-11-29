from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from math import floor
from typing import TYPE_CHECKING, Protocol, cast, overload, runtime_checkable

import numpy as np

from pysatl_core.types import BoolArray, Interval1D, Number, NumericArray

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@runtime_checkable
class Support(Protocol):
    @overload
    def contains(self, x: Number) -> bool: ...
    @overload
    def contains(self, x: NumericArray) -> BoolArray: ...


class ContinuousSupport(Interval1D, Support): ...


@runtime_checkable
class DiscreteSupport(Support, Protocol):
    def iter_points(self) -> Iterator[Number]: ...

    def iter_leq(self, x: Number) -> Iterator[Number]: ...

    def prev(self, x: Number) -> Number | None: ...


class ExplicitTableDiscreteSupport(DiscreteSupport):
    __slots__ = ("_points",)

    def __init__(self, points: Iterable[Number], assume_sorted: bool = False) -> None:
        arr = np.array(points)

        if arr.size == 0:
            raise ValueError("Points must be non-empty")

        if not assume_sorted:
            arr.sort()

        unique_mask = np.empty(arr.size, dtype=bool)
        unique_mask[0] = True
        unique_mask[1:] = arr[1:] != arr[:-1]

        self._points = arr[unique_mask]

    @overload
    def contains(self, x: Number) -> bool: ...
    @overload
    def contains(self, x: NumericArray) -> BoolArray: ...

    def contains(self, x: Number | NumericArray) -> bool | BoolArray:
        arr = np.asarray(x)
        idx = np.searchsorted(self._points, arr, side="left")

        size = self._points.size
        in_bounds = (idx >= 0) & (idx < size)

        idx_clipped = np.minimum(idx, size - 1)
        eq = self._points[idx_clipped] == arr

        result = in_bounds & eq

        if np.ndim(arr) == 0:
            return bool(result)
        return cast(BoolArray, result)

    def __contains__(self, x: object) -> bool:
        return bool(self.contains(cast(Number, x)))

    def iter_points(self) -> Iterator[Number]:
        return iter(self._points)

    def iter_leq(self, x: Number) -> Iterator[Number]:
        return iter(self._points[: np.searchsorted(self._points, x, side="right")])

    def prev(self, x: Number) -> Number | None:
        idx = np.searchsorted(self._points, x, side="left")
        if idx == 0:
            return None
        return cast(Number, self._points[idx - 1])

    def first(self) -> Number:
        return cast(Number, self._points[0])

    def next(self, current: Number) -> Number | None:
        idx = np.searchsorted(self._points, current, side="right")
        if idx == self._points.size:
            return None
        return cast(Number, self._points[idx])

    @property
    def points(self) -> NumericArray:
        return cast(NumericArray, self._points.copy())

    __iter__ = iter_points


@dataclass(slots=True)
class IntegerLatticeDiscreteSupport(DiscreteSupport):
    residue: int
    modulus: int
    min_k: int | None = None
    max_k: int | None = None

    def __post_init__(self) -> None:
        if self.modulus <= 0:
            raise ValueError("modulus must be a positive integer.")

    @overload
    def contains(self, x: Number) -> bool: ...
    @overload
    def contains(self, x: NumericArray) -> BoolArray: ...

    def contains(self, x: Number | NumericArray) -> bool | BoolArray:
        xf = np.asarray(x, dtype=float)
        v = np.floor(xf).astype(int)
        is_integer = xf == v

        mask = is_integer
        if self.min_k is not None:
            mask &= v >= self.min_k
        if self.max_k is not None:
            mask &= v <= self.max_k

        step_ok = ((v - self.residue) % self.modulus) == 0
        mask &= step_ok

        result = mask.astype(bool)

        if np.ndim(xf) == 0:
            return bool(result)
        return cast(BoolArray, result)

    def __contains__(self, x: object) -> bool:
        return bool(self.contains(cast(Number, x)))

    def iter_points(self) -> Iterator[int]:
        first = self.first()
        last = self.last()

        if first is not None and last is not None and first > last:
            return iter(())

        if first is not None:

            def _gen_lr() -> Iterator[int]:
                current = first
                while self.max_k is None or current <= self.max_k:
                    yield current
                    current += self.modulus

            return _gen_lr()

        if last is not None:

            def _gen_rl() -> Iterator[int]:
                current = last
                while self.min_k is None or current >= self.min_k:
                    yield current
                    current -= self.modulus

            return _gen_rl()

        raise RuntimeError(
            "Cannot iterate points for an unbounded IntegerLatticeDiscreteSupport "
            "(both min_k and max_k are None). Provide at least one bound to enable enumeration."
        )

    def iter_leq(self, x: Number) -> Iterator[int]:
        if self.min_k is None:
            raise RuntimeError(
                "iter_leq is not supported for left-unbounded IntegerLatticeDiscreteSupport. "
                "Provide min_k to enable iter_leq."
            )
        first = self.first()
        if first is None:
            return iter(())

        threshold = int(floor(float(x)))
        if threshold < first:
            return iter(())
        last = threshold
        if self.max_k is not None and last > self.max_k:
            last = self.max_k
        offset = (last - self.residue) % self.modulus
        last = last - offset
        if last < first:
            return iter(())

        def _gen() -> Iterator[int]:
            current = first
            while current <= last:
                yield current
                current += self.modulus

        return _gen()

    def prev(self, x: Number) -> int | None:
        if self.min_k is not None and float(x) <= self.min_k:
            return None
        target = int(floor(float(x))) - 1
        if self.max_k is not None and target > self.max_k:
            target = self.max_k
        if self.min_k is not None and target < self.min_k:
            return None
        candidate = self.residue + ((target - self.residue) // self.modulus) * self.modulus
        if self.min_k is not None and candidate < self.min_k:
            return None
        if self.max_k is not None and candidate > self.max_k:
            return None
        return candidate

    def first(self) -> int | None:
        if self.min_k is None:
            return None
        first = self.min_k
        offset = (first - self.residue) % self.modulus
        if offset != 0:
            first = first + (self.modulus - offset)
        if self.max_k is not None and first > self.max_k:
            return None
        return first

    def last(self) -> int | None:
        if self.max_k is None:
            return None
        last = self.max_k
        offset = (last - self.residue) % self.modulus
        last = last - offset
        if self.min_k is not None and last < self.min_k:
            return None
        return last

    def next(self, current: int) -> int | None:
        nxt = current + self.modulus
        if self.max_k is not None and nxt > self.max_k:
            return None
        if self.min_k is not None and nxt < self.min_k:
            return None
        return nxt

    @property
    def is_left_bounded(self) -> bool:
        return self.min_k is not None

    @property
    def is_right_bounded(self) -> bool:
        return self.max_k is not None

    __iter__ = iter_points


__all__ = [
    # Base support protocol
    "Support",
    "ContinuousSupport",
    # Discrete support protocol and implementations
    "DiscreteSupport",
    "ExplicitTableDiscreteSupport",
    "IntegerLatticeDiscreteSupport",
]
