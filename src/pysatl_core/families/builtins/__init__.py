"""
Built-in distribution families for PySATL.

This package contains implementations of standard statistical distribution families
that are available by default in PySATL.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from .continuous import *
from .continuous import __all__ as _continuous_all

__all__ = [
    *_continuous_all,
]

del _continuous_all
