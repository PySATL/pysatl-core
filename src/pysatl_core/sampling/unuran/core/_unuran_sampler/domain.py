"""
Domain inference helpers for UNU.RAN distributions.

Provides utilities to derive discrete domains and continuous bounds
from distribution metadata needed during UNU.RAN setup.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numbers
from typing import TYPE_CHECKING

import numpy as np

from pysatl_core.distributions.support import (
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support


class UnuranDomain:
    """
    Derives domain boundaries from a distribution's support for UNU.RAN setup.

    Wraps the support object and exposes two helpers:
    one for discrete domains (integer bounds) and one for continuous domains
    (float bounds).
    """

    def __init__(self, support: Support | None) -> None:
        """
        Parameters
        ----------
        support : Support or None
            The distribution's support to inspect.
        """
        self._support = support

    def determine_discrete_domain(self) -> tuple[int, int] | None:
        """
        Determine domain boundaries from distribution support if available.

        Returns
        -------
        tuple[int, int] or None
            Domain as (left, right) if both boundaries are known, or None if support
            is unavailable or unbounded on either side.
        """
        support = self._support
        if support is None:
            return None

        if isinstance(support, ExplicitTableDiscreteSupport):
            points = support.points
            if points.size == 0:
                return None
            return 0, int(points.size) - 1

        if isinstance(support, IntegerLatticeDiscreteSupport):
            first = support.first()
            last = support.last()

            if first is not None and last is not None:
                return first, last

        return None

    def explicit_table_points(self) -> np.ndarray | None:
        """
        Return the support points array if support is ExplicitTableDiscreteSupport.

        Returns
        -------
        np.ndarray or None
            Points array when support is an explicit table with at least one point,
            otherwise None.
        """
        if isinstance(self._support, ExplicitTableDiscreteSupport):
            pts = self._support.points
            return pts if pts.size > 0 else None
        return None

    def determine_continuous_domain(self) -> tuple[float, float] | None:
        """
        Determine continuous-domain bounds if the support exposes them.

        Returns
        -------
        tuple[float, float] or None
            Returns (left, right) bounds when both edges are numeric, otherwise
            None when support is missing or incomplete.
        """
        support = self._support
        if support is None:
            return None

        left = getattr(support, "left", None)
        right = getattr(support, "right", None)

        if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
            return float(left), float(right)

        return None
