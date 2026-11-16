"""NumPy numerical backend composition."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, cast

import numpy as np

from pysatl_core.arrays.backend import ArrayBackend, array_operation
from pysatl_core.arrays.dtypes import DType
from pysatl_core.arrays.runtime import Backend


def _normalize_numpy_dtype(dtype: object) -> Hashable:
    return cast(Hashable, np.dtype(cast(Any, dtype)))


numpy_array_backend = ArrayBackend("numpy", dtype_normalizer=_normalize_numpy_dtype)

for _dtype in DType:
    numpy_array_backend.register_dtype(_dtype, np.dtype(_dtype.value))


@array_operation(numpy_array_backend, "result_type")
def _result_type(*dtypes: object) -> np.dtype[Any]:
    return np.result_type(*dtypes)


numpy_backend = Backend(identity="numpy", arrays=numpy_array_backend)

del _dtype

__all__ = [
    "numpy_array_backend",
    "numpy_backend",
]
