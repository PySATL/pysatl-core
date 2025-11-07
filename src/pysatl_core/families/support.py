"""
Support primitives for distribution families.

This module defines concrete implementations of the base `DiscreteSupport`
protocol that provide ordered access to discrete supports (sets of points)
required for robust conversions between distribution characteristics
(e.g., PMF ↔ CDF) in discrete settings.

Maybe it'll be Support for other kinds of distributions soon

Classes
-------
ExplicitTableDiscreteSupport
    Finite, explicitly provided, ordered set of support points.

IntegerLatticeDiscreteSupport
    Integer lattice support of the form { start + n * step | n ∈ Z },
    optionally bounded to a finite interval.

Notes
-----
These implementations are intentionally minimal and agnostic to any particular
family; they provide only the ordered traversal primitives needed by fitters.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import bisect
from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable


class Support(Protocol):
    """Marker protocol for any support object."""


@runtime_checkable
class DiscreteSupport(Support, Protocol):
    """
    Protocol for ordered access to discrete supports.

    Methods
    -------
    iter_points() -> Iterable[int | float]
        Iterate over all support points in non-decreasing order.

    iter_leq(x: int | float) -> Iterable[int | float]
        Iterate over support points that are less than or equal to `x`.

    prev(x: int | float) -> int | float | None
        Return the greatest support point strictly less than `x`, or `None`
        if such a point does not exist.
    """

    def iter_points(self) -> Iterable[int | float]: ...
    def iter_leq(self, x: int | float) -> Iterable[int | float]: ...
    def prev(self, x: int | float) -> int | float | None: ...


class ExplicitTableDiscreteSupport(DiscreteSupport):
    """
    Finite discrete support defined by an explicit list of points.

    Parameters
    ----------
    points : Iterable[int | float]
        Collection of support points. Duplicates are allowed but not recommended.
    assume_sorted : bool, default False
        If `True`, the input `points` are assumed to be already sorted
        in non-decreasing order. If `False`, the constructor sorts them.

    Attributes
    ----------
    _points : list[int | float]
        Sorted list of support points.

    Notes
    -----
    - `iter_points` returns an iterator over the entire support.
    - `iter_leq(x)` returns an iterator over the prefix `<= x`.
    - `prev(x)` returns the largest support point strictly less than `x`,
      or `None` if no such point exists.
    """

    __slots__ = ("_points",)

    def __init__(self, points: Iterable[int | float]) -> None:
        points_list = list(points)
        points_list.sort()
        self._points: list[int | float] = points_list

    def iter_points(self) -> Iterable[int | float]:
        """
        Iterate over the entire set of support points.

        Returns
        -------
        Iterable[int | float]
            Iterator over all points in non-decreasing order.
        """
        return iter(self._points)

    def iter_leq(self, x: int | float) -> Iterable[int | float]:
        """
        Iterate over support points less than or equal to `x`.

        Parameters
        ----------
        x : int | float
            Threshold value.

        Returns
        -------
        Iterable[int | float]
            Iterator over the prefix of points `<= x`.
        """
        last_index = bisect.bisect_right(self._points, x) - 1
        if last_index < 0:
            return iter(())
        return iter(self._points[: last_index + 1])

    def prev(self, x: int | float) -> int | float | None:
        """
        Return the greatest support point strictly less than `x`.

        Parameters
        ----------
        x : int | float
            Reference value.

        Returns
        -------
        int | float | None
            The greatest point `< x`, or `None` if none exists.
        """
        insertion_index = bisect.bisect_left(self._points, x)
        if insertion_index <= 0:
            return None
        return self._points[insertion_index - 1]


class IntegerLatticeDiscreteSupport(DiscreteSupport):
    """
    Discrete support on an integer lattice with optional finite bounds.

    The support is defined as:
        S = { start + n * step | n ∈ Z }
    and can be restricted to an interval by `min_k` and/or `max_k`.

    Parameters
    ----------
    start : int
        Lattice origin.
    step : int, default 1
        Positive lattice step.
    min_k : int | None, optional
        Inclusive lower bound on support values. If not aligned to the lattice,
        the first yielded value will be the smallest lattice point `>= min_k`.
    max_k : int | None, optional
        Inclusive upper bound on support values.

    Attributes
    ----------
    start : int
        Lattice origin.
    step : int
        Positive lattice step.
    min_k : int | None
        Inclusive lower bound on support values (may be unaligned).
    max_k : int | None
        Inclusive upper bound on support values.

    Notes
    -----
    - `iter_points` yields all lattice points within bounds in ascending order.
    - `iter_leq(x)` yields all bounded lattice points `<= x`.
    - `prev(x)` returns the largest lattice point strictly less than `x`,
      or `None` if it falls below `min_k`.
    """

    __slots__ = ("start", "step", "min_k", "max_k")

    def __init__(
        self,
        start: int,
        step: int = 1,
        *,
        min_k: int | None = None,
        max_k: int | None = None,
    ) -> None:
        if step <= 0:
            raise ValueError("step must be positive")
        self.start: int = int(start)
        self.step: int = int(step)
        self.min_k: int | None = None if min_k is None else int(min_k)
        self.max_k: int | None = None if max_k is None else int(max_k)

    def iter_points(self) -> Iterable[int]:
        """
        Iterate over all lattice points subject to bounds.

        Returns
        -------
        Iterable[int]
            Iterator over bounded lattice points in ascending order.
        """
        first_value = self.min_k if self.min_k is not None else self.start
        if (first_value - self.start) % self.step != 0:
            alignment_offset = (self.step - ((first_value - self.start) % self.step)) % self.step
            first_value = first_value + alignment_offset

        current_value = first_value
        while self.max_k is None or current_value <= self.max_k:
            yield current_value
            current_value += self.step

    def iter_leq(self, x: int | float) -> Iterable[int]:
        """
        Iterate over lattice points less than or equal to `x`.

        Parameters
        ----------
        x : int | float
            Threshold value.

        Returns
        -------
        Iterable[int]
            Iterator over lattice points `<= x` within bounds.
        """
        smallest_value = self.min_k if self.min_k is not None else self.start
        if x < smallest_value:
            return iter(())

        def generate() -> Iterator[int]:
            for support_value in self.iter_points():
                if support_value <= int(x):
                    yield support_value
                else:
                    break

        return generate()

    def prev(self, x: int | float) -> int | None:
        """
        Return the greatest lattice point strictly less than `x`.

        Parameters
        ----------
        x : int | float
            Reference value.

        Returns
        -------
        int | None
            The lattice point `< x` with the largest value, or `None` if it
            would be below `min_k`.
        """
        x_int = int(x)
        target_value = x_int - 1
        candidate_value = self.start + ((target_value - self.start) // self.step) * self.step
        if self.min_k is not None and candidate_value < self.min_k:
            return None
        return candidate_value
