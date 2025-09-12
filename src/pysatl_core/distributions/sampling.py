from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class Sample(Protocol):
    def __len__(self) -> int: ...
    @property
    def array(self) -> npt.NDArray[np.floating[Any]]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...


class ArraySample(Sample):
    dimension: int
    data: npt.NDArray[np.floating[Any]]

    def __init__(self, data: npt.NDArray[np.floating[Any]]) -> None:
        if data.ndim != 2:
            raise ValueError("ArraySample expects 2D array of shape (n, d).")
        self.data = data
        self.dimension = int(data.shape[1])

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def dim(self) -> int:
        return self.dimension

    def __iter__(self) -> Iterator[npt.NDArray[np.floating[Any]]]:
        yield from self.data

    @property
    def array(self) -> npt.NDArray[np.floating[Any]]:
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        n, d = self.data.shape
        return int(n), int(d)
