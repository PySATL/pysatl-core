"""
Computations subpackage.

Provides option and descriptor abstractions, fitter/evaluator implementations,
helper utilities, and a registry for matching fitter descriptors
to distribution characteristics.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .continuous import *
from .continuous import __all__ as _continuous_all
from .descriptors import *
from .descriptors import __all__ as _descriptors_all
from .discrete import *
from .discrete import __all__ as _discrete_all
from .options import *
from .options import __all__ as _options_all
from .registry import *
from .registry import __all__ as _registry_all

__all__ = [
    *_options_all,
    *_descriptors_all,
    *_continuous_all,
    *_discrete_all,
    *_registry_all,
]

del _options_all, _descriptors_all, _continuous_all, _discrete_all, _registry_all
