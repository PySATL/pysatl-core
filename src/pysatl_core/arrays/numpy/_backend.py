"""NumPy numerical backend composition."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, cast

import numpy as np

from pysatl_core.arrays.backend import ArrayBackend, array_operation
from pysatl_core.arrays.dtypes import DType
from pysatl_core.arrays.runtime import Backend

numpy_array_backend = ArrayBackend("numpy")


@array_operation(numpy_array_backend, "normalize_native_dtype")
def _normalize_numpy_dtype(dtype: object) -> Hashable:
    return cast(Hashable, np.dtype(cast(Any, dtype)))


@array_operation(numpy_array_backend, "dtype_mappings")
def _dtype_mappings() -> dict[DType, Hashable]:
    return {dtype: _normalize_numpy_dtype(dtype.value) for dtype in DType}


@array_operation(numpy_array_backend, "result_type")
def _result_type(*dtypes: np.dtype[Any]) -> np.dtype[Any]:
    return np.result_type(*dtypes)


numpy_backend = Backend(identity="numpy", arrays=numpy_array_backend)

__all__ = [
    "numpy_array_backend",
    "numpy_backend",
]
