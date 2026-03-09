"""
Approximation utilities for transformed distributions.

This subpackage contains approximation interfaces and concrete
approximators that can materialize analytical characteristics for
complex transformation trees.
"""

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .approximation import *
from .approximation import __all__ as _approximation_all
from .chebyshev import *
from .chebyshev import __all__ as _chebyshev_all

__all__ = [
    *_approximation_all,
    *_chebyshev_all,
]

del _approximation_all

del _chebyshev_all
