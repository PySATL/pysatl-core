"""
Built-in method registries for binary operations.
"""

from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .division import *
from .division import __all__ as _division_all
from .linear import *
from .linear import __all__ as _linear_all
from .multiplication import *
from .multiplication import __all__ as _multiplication_all

__all__ = [
    *_division_all,
    *_linear_all,
    *_multiplication_all,
]

del _division_all
del _linear_all
del _multiplication_all
