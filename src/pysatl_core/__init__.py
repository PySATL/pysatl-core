"""
PySATL Core
===========

Minimal core for probabilistic distributions: types, strategies, fitters,
and graph-based characteristic resolution.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from importlib.metadata import version

from .distributions import *
from .distributions import __all__ as _distr_all
from .families import *
from .families import __all__ as _family_all

__version__ = version("pysatl-core")
__all__ = [
    "__version__",
    *_distr_all,
    *_family_all,
]
