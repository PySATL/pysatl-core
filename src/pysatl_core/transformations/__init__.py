"""
Transformations framework for derived probability distributions.

This package provides the base primitives for constructing distributions
obtained from other distributions, together with approximation interfaces
and concrete transformation implementations.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .approximations import *
from .approximations import __all__ as _approximations_all
from .distribution import *
from .distribution import __all__ as _distribution_all
from .lightweight_distribution import *
from .lightweight_distribution import __all__ as _lightweight_all
from .operations import *
from .operations import __all__ as _operations_all
from .operators_mixin import *
from .operators_mixin import __all__ as _operators_mixin_all
from .transformation_method import *
from .transformation_method import __all__ as _methods_all

__all__ = [
    *_approximations_all,
    *_distribution_all,
    *_lightweight_all,
    *_operations_all,
    *_operators_mixin_all,
    *_methods_all,
]

del _approximations_all

del _distribution_all

del _lightweight_all

del _operations_all

del _operators_mixin_all

del _methods_all
