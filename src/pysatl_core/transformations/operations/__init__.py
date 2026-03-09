"""
Concrete distribution transformation operations.

This subpackage contains concrete transformed-distribution implementations.
At the moment it provides the first affine transformation primitive.
"""

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .affine import *
from .affine import __all__ as _affine_all

__all__ = [
    *_affine_all,
]

del _affine_all
