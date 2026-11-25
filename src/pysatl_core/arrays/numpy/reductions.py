"""NumPy reduction operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from pysatl_core.arrays.backend import array_operation
from pysatl_core.arrays.numpy._backend import numpy_array_backend

any = array_operation(numpy_array_backend)(np.any)
all = array_operation(numpy_array_backend)(np.all)
sum = array_operation(numpy_array_backend)(np.sum)
min = array_operation(numpy_array_backend)(np.min)
max = array_operation(numpy_array_backend)(np.max)
argmax = array_operation(numpy_array_backend)(np.argmax)
cumsum = array_operation(numpy_array_backend)(np.cumsum)
diff = array_operation(numpy_array_backend)(np.diff)


@array_operation(numpy_array_backend)
def cummax(value: np.ndarray[Any, Any], axis: int = 0) -> np.ndarray[Any, Any]:
    """Return cumulative maxima."""
    return np.maximum.accumulate(value, axis=axis)


__all__: list[str] = []
