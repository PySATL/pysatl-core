"""NumPy numerical backend."""

from __future__ import annotations

from pysatl_core.arrays.numpy import (
    creation as _creation,
    elementwise as _elementwise,
    manipulation as _manipulation,
    reductions as _reductions,
    validation as _validation,
)
from pysatl_core.arrays.numpy._backend import numpy_array_backend, numpy_backend

__all__ = [
    "numpy_array_backend",
    "numpy_backend",
]

del _creation, _elementwise, _manipulation, _reductions, _validation
