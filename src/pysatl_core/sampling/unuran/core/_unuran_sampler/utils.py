"""
Utility helpers for UNU.RAN error reporting.

Provides a single helper that queries the UNU.RAN errno and formats a
human-readable error message including the library's own error string.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any


def get_unuran_error_message(lib: Any, ffi: Any, base_msg: str) -> str:
    """
    Format a UNU.RAN error message with errno and the library error string.

    Parameters
    ----------
    lib : Any
        CFFI library handle exposing ``unur_get_errno`` and ``unur_get_strerror``.
    ffi : Any
        CFFI FFI instance used to decode the C string.
    base_msg : str
        Base error message to prepend.

    Returns
    -------
    str
        Formatted error message of the form ``"<base_msg> (errno: N): <strerror>"``,
        where the strerror suffix is omitted when errno is zero.
    """
    errno = lib.unur_get_errno()
    error_str = lib.unur_get_strerror(errno) if errno != 0 else None
    error_msg = f"{base_msg} (errno: {errno})"
    if error_str:
        error_msg += f": {ffi.string(error_str).decode('utf-8')}"
    return error_msg
