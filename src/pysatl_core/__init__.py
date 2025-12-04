"""
PySATL Core
===========

Core framework for probabilistic distributions providing type definitions,
distribution abstractions, characteristic computation graphs, and parametric
family management.
"""

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from importlib.metadata import version

from .distributions import *
from .distributions import __all__ as _distr_all
from .families import *
from .families import __all__ as _family_all
from .types import *
from .types import __all__ as _types_all

__version__ = version("pysatl-core")
__all__ = [
    "__version__",
    *_distr_all,
    *_family_all,
    *_types_all,
]

del _distr_all
del _family_all
del _types_all
