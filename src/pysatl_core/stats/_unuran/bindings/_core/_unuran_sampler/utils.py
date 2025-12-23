from __future__ import annotations

from typing import Any


def get_unuran_error_message(lib: Any, ffi: Any, base_msg: str) -> str:
    """
    Format UNURAN error message with errno and error string.

    Parameters
    ----------
    base_msg : str
        Base error message to format.

    Returns
    -------
    str
        Formatted error message including errno and UNURAN error string
        (if available).

    Notes
    -----
    This helper method retrieves the current UNURAN error state and formats
    a comprehensive error message for debugging purposes.
    """
    errno = lib.unur_get_errno()
    error_str = lib.unur_get_strerror(errno) if errno != 0 else None
    error_msg = f"{base_msg} (errno: {errno})"
    if error_str:
        error_msg += f": {ffi.string(error_str).decode('utf-8')}"
    return error_msg


__all__ = ["get_unuran_error_message"]
