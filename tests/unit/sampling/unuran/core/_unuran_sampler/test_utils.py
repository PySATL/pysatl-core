"""
Unit tests for pysatl_core.sampling.unuran.core._unuran_sampler.utils

Tests the get_unuran_error_message helper that formats UNU.RAN error messages
with errno and the library's own error string.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from unittest.mock import MagicMock

from pysatl_core.sampling.unuran.core._unuran_sampler.utils import get_unuran_error_message


class TestGetUnuranErrorMessage:
    """Tests for get_unuran_error_message function."""

    def test_zero_errno_returns_base_message_with_errno_only(self) -> None:
        """When errno is 0, only the base message and errno are included — no strerror call."""
        mock_lib = MagicMock()
        mock_lib.unur_get_errno.return_value = 0
        mock_ffi = MagicMock()

        result = get_unuran_error_message(mock_lib, mock_ffi, "initialization failed")

        assert result == "initialization failed (errno: 0)"
        mock_lib.unur_get_strerror.assert_not_called()
        mock_ffi.string.assert_not_called()

    def test_nonzero_errno_appends_decoded_error_string(self) -> None:
        """When errno is non-zero, the decoded strerror is appended to the message."""
        mock_lib = MagicMock()
        mock_lib.unur_get_errno.return_value = 5
        mock_lib.unur_get_strerror.return_value = MagicMock()
        mock_ffi = MagicMock()
        mock_ffi.string.return_value = b"distribution not set"

        result = get_unuran_error_message(mock_lib, mock_ffi, "setup error")

        assert "setup error (errno: 5)" in result
        assert "distribution not set" in result

    def test_nonzero_errno_calls_get_strerror_with_errno(self) -> None:
        """get_unuran_error_message passes the errno value to unur_get_strerror."""
        mock_lib = MagicMock()
        mock_lib.unur_get_errno.return_value = 99
        mock_lib.unur_get_strerror.return_value = MagicMock()
        mock_ffi = MagicMock()
        mock_ffi.string.return_value = b"unknown error"

        get_unuran_error_message(mock_lib, mock_ffi, "msg")

        mock_lib.unur_get_strerror.assert_called_once_with(99)

    def test_result_format_with_nonzero_errno(self) -> None:
        """The formatted result follows the pattern '<base> (errno: N): <strerror>'."""
        mock_lib = MagicMock()
        mock_lib.unur_get_errno.return_value = 3
        mock_lib.unur_get_strerror.return_value = MagicMock()
        mock_ffi = MagicMock()
        mock_ffi.string.return_value = b"generator error"

        result = get_unuran_error_message(mock_lib, mock_ffi, "base msg")

        assert result == "base msg (errno: 3): generator error"

    def test_empty_base_message_is_allowed(self) -> None:
        """An empty base_msg produces a valid error string."""
        mock_lib = MagicMock()
        mock_lib.unur_get_errno.return_value = 0
        mock_ffi = MagicMock()

        result = get_unuran_error_message(mock_lib, mock_ffi, "")

        assert result == " (errno: 0)"
