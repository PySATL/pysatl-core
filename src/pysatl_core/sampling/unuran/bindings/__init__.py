"""
PySATL Core — UNU.RAN CFFI Bindings
=====================================

Lazy loader for the compiled CFFI extension module ``_unuran_cffi``.
Tries the fully-qualified package path first, then falls back to a
bare top-level import so the extension can be found regardless of
how it was installed.  Exposes ``_unuran_cffi`` (``None`` when the
binary extension is not available).
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from importlib import import_module
from types import ModuleType

_unuran_cffi_module = None

for name in (
    "pysatl_core.sampling.unuran.bindings._unuran_cffi",
    "_unuran_cffi",
):
    try:
        _unuran_cffi_module = import_module(name)
        break
    except ModuleNotFoundError:  # pragma: no cover - optional binary module
        pass

_unuran_cffi: ModuleType | None = _unuran_cffi_module

__all__ = [
    "_unuran_cffi",
]
