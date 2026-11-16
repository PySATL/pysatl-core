"""NumPy element-wise operations."""

from __future__ import annotations

import numpy as np

from pysatl_core.arrays.backend import array_operation
from pysatl_core.arrays.numpy._backend import numpy_array_backend

add = array_operation(numpy_array_backend)(np.add)
subtract = array_operation(numpy_array_backend)(np.subtract)
multiply = array_operation(numpy_array_backend)(np.multiply)
divide = array_operation(numpy_array_backend)(np.divide)
floor_divide = array_operation(numpy_array_backend)(np.floor_divide)
remainder = array_operation(numpy_array_backend)(np.remainder)
power = array_operation(numpy_array_backend)(np.power)
negative = array_operation(numpy_array_backend)(np.negative)
abs = array_operation(numpy_array_backend, "abs")(np.abs)
exp = array_operation(numpy_array_backend)(np.exp)
log = array_operation(numpy_array_backend)(np.log)
log1p = array_operation(numpy_array_backend)(np.log1p)
sqrt = array_operation(numpy_array_backend)(np.sqrt)
sinc = array_operation(numpy_array_backend)(np.sinc)
floor = array_operation(numpy_array_backend)(np.floor)
round = array_operation(numpy_array_backend)(np.round)
nextafter = array_operation(numpy_array_backend)(np.nextafter)
maximum = array_operation(numpy_array_backend)(np.maximum)
minimum = array_operation(numpy_array_backend)(np.minimum)
clip = array_operation(numpy_array_backend)(np.clip)
where = array_operation(numpy_array_backend)(np.where)
equal = array_operation(numpy_array_backend)(np.equal)
not_equal = array_operation(numpy_array_backend)(np.not_equal)
less = array_operation(numpy_array_backend)(np.less)
less_equal = array_operation(numpy_array_backend)(np.less_equal)
greater = array_operation(numpy_array_backend)(np.greater)
greater_equal = array_operation(numpy_array_backend)(np.greater_equal)
logical_and = array_operation(numpy_array_backend)(np.logical_and)
logical_or = array_operation(numpy_array_backend)(np.logical_or)
logical_not = array_operation(numpy_array_backend)(np.logical_not)
isfinite = array_operation(numpy_array_backend)(np.isfinite)
isnan = array_operation(numpy_array_backend)(np.isnan)
isinf = array_operation(numpy_array_backend)(np.isinf)
isclose = array_operation(numpy_array_backend)(np.isclose)
nan_to_num = array_operation(numpy_array_backend)(np.nan_to_num)
real = array_operation(numpy_array_backend)(np.real)
imag = array_operation(numpy_array_backend)(np.imag)
conjugate = array_operation(numpy_array_backend)(np.conjugate)


__all__: list[str] = []
