"""NumPy shape and indexing operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from pysatl_core.arrays.backend import array_operation
from pysatl_core.arrays.numpy._backend import numpy_array_backend

atleast_1d = array_operation(numpy_array_backend)(np.atleast_1d)
reshape = array_operation(numpy_array_backend)(np.reshape)
expand_dims = array_operation(numpy_array_backend)(np.expand_dims)
stack = array_operation(numpy_array_backend)(np.stack)
concatenate = array_operation(numpy_array_backend)(np.concatenate)
take = array_operation(numpy_array_backend)(np.take)
nonzero = array_operation(numpy_array_backend)(np.nonzero)
unique = array_operation(numpy_array_backend)(np.unique)
argsort = array_operation(numpy_array_backend)(np.argsort)
searchsorted = array_operation(numpy_array_backend)(np.searchsorted)


@array_operation(numpy_array_backend)
def flatten(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Flatten an array."""
    return value.flatten()


@array_operation(numpy_array_backend)
def getitem(value: Any, item: Any) -> Any:
    """Read an item from an array."""
    return value[item]


@array_operation(numpy_array_backend)
def setitem(value: np.ndarray[Any, Any], item: Any, update: Any) -> np.ndarray[Any, Any]:
    """Return a copy of an array with one functional update."""
    result = value.copy()
    result[item] = update
    return result


@array_operation(numpy_array_backend)
def scatter_add(value: np.ndarray[Any, Any], indices: Any, updates: Any) -> np.ndarray[Any, Any]:
    """Return a copy with updates accumulated at indices."""
    result = value.copy()
    np.add.at(result, indices, updates)
    return result


__all__: list[str] = []
