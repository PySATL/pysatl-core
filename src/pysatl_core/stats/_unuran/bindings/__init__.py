from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from importlib import import_module
from types import ModuleType

from ._core import DefaultUnuranSampler, DefaultUnuranSamplingStrategy

_unuran_cffi_module: ModuleType | None = None
try:
    _unuran_cffi_module = import_module("pysatl_core.stats._unuran.bindings._unuran_cffi")
except ModuleNotFoundError:  # pragma: no cover - optional binary module
    try:
        _unuran_cffi_module = import_module("_unuran_cffi")
    except ModuleNotFoundError:  # pragma: no cover - optional binary module
        _unuran_cffi_module = None

_unuran_cffi: ModuleType | None = _unuran_cffi_module

__all__ = [
    "DefaultUnuranSampler",
    "DefaultUnuranSamplingStrategy",
    "_unuran_cffi",
]
