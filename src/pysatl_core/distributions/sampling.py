"""
Sampling Interfaces
===================

This module defines the sampling protocol and a concrete array-backed
implementation for un/structured samples.

Notes
-----
- :class:`ArraySample` strictly requires a **2D** array of shape ``(n, d)``.
  For univariate distributions, ``d == 1``.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class Sample(Protocol):
    """Protocol for sample-like containers."""

    def __len__(self) -> int: ...
    @property
    def array(self) -> npt.NDArray[np.floating[Any]]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...


class ArraySample(Sample):
    """
    Array-backed sample.

    Parameters
    ----------
    data : numpy.ndarray
        2D floating array of shape ``(n, d)``.

    Attributes
    ----------
    data : numpy.ndarray
        Backing array.
    dimension : int
        Dimensionality ``d``.

    Raises
    ------
    ValueError
        If ``data`` is not 2D.
    """

    dimension: int
    data: npt.NDArray[np.floating[Any]]

    def __init__(self, data: npt.NDArray[np.floating[Any]]) -> None:
        if data.ndim != 2:
            raise ValueError("ArraySample expects 2D array of shape (n, d).")
        self.data = data
        self.dimension = int(data.shape[1])

    def __len__(self) -> int:
        """Number of rows ``n``."""
        return int(self.data.shape[0])

    @property
    def dim(self) -> int:
        """Alias for :attr:`dimension`."""
        return self.dimension

    def __iter__(self) -> Iterator[npt.NDArray[np.floating[Any]]]:
        """Iterate over rows."""
        yield from self.data

    @property
    def array(self) -> npt.NDArray[np.floating[Any]]:
        """Return the backing array."""
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        """Return ``(n, d)``."""
        n, d = self.data.shape
        return int(n), int(d)
