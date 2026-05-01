"""
Computations subpackage.

Provides descriptor abstractions, fitter/evaluator implementations,
helper utilities, and a registry for matching fitter descriptors
to distribution characteristics.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .base import *
from .base import __all__ as _base_all
from .continuous import *
from .continuous import __all__ as _continuous_all
from .discrete import *
from .discrete import __all__ as _discrete_all
from .registry import *
from .registry import __all__ as _registry_all

ALL_FITTER_DESCRIPTORS: list[FitterDescriptor] = [  # noqa: F405
    FITTER_PDF_TO_CDF_1C,  # noqa: F405
    FITTER_CDF_TO_PDF_1C,  # noqa: F405
    FITTER_CDF_TO_PPF_1C,  # noqa: F405
    FITTER_PPF_TO_CDF_1C,  # noqa: F405
    FITTER_PMF_TO_CDF_1D,  # noqa: F405
    FITTER_CDF_TO_PMF_1D,  # noqa: F405
    FITTER_CDF_TO_PPF_1D,  # noqa: F405
    FITTER_PPF_TO_CDF_1D,  # noqa: F405
]

__all__ = [
    *_base_all,
    *_continuous_all,
    *_discrete_all,
    *_registry_all,
    "ALL_FITTER_DESCRIPTORS",
]

del _base_all, _continuous_all, _discrete_all, _registry_all
