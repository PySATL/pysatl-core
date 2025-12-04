"""
Sampling Interfaces
===================

This module defines protocols and implementations for sample containers
used in distribution sampling.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    import numpy.typing as npt


class Sample(Protocol):
    """
    Protocol for sample containers.

    Attributes
    ----------
    array : numpy.ndarray
        Array representation of the samples.
    shape : tuple[int, ...]
        Shape of the sample array.
    """

    def __len__(self) -> int: ...
    @property
    def array(self) -> npt.NDArray[np.floating[Any]]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...


class ArraySample:
    """
    Array-backed sample container.

    This implementation stores samples as a 2D floating-point array
    of shape (n_samples, n_dimensions).

    Parameters
    ----------
    data : numpy.ndarray
        2D floating-point array of shape (n, d).

    Attributes
    ----------
    data : numpy.ndarray
        Backing array containing the samples.
    dimension : int
        Dimensionality of the samples (d).

    Raises
    ------
    ValueError
        If data is not 2D.
    """

    dimension: int
    data: npt.NDArray[np.floating[Any]]

    def __init__(self, data: npt.NDArray[np.floating[Any]]) -> None:
        if data.ndim != 2:
            raise ValueError("ArraySample expects 2D array of shape (n, d).")
        self.data = data
        self.dimension = int(data.shape[1])

    def __len__(self) -> int:
        """Return the number of samples (n)."""
        return int(self.data.shape[0])

    @property
    def dim(self) -> int:
        """Alias for dimension attribute."""
        return self.dimension

    def __iter__(self) -> Iterator[npt.NDArray[np.floating[Any]]]:
        """Iterate over samples (rows of the array)."""
        yield from self.data

    @property
    def array(self) -> npt.NDArray[np.floating[Any]]:
        """Return the backing array."""
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the sample array (n, d)."""
        n, d = self.data.shape
        return int(n), int(d)
