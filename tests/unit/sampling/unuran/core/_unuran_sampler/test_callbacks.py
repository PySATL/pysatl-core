"""
Unit tests for pysatl_core.sampling.unuran.core._unuran_sampler.callbacks

Tests the UnuranCallback factory/registry class that creates CFFI callbacks
for PDF, dPDF, CDF, PPF, and PMF and registers them with a UNU.RAN distribution.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any
from unittest.mock import MagicMock

import pytest

from pysatl_core.sampling.unuran.core._unuran_sampler.callbacks import UnuranCallback
from pysatl_core.types import CharacteristicName, Kind


class _MockHandle:
    """Opaque handle that is only equal to itself (mimics a CFFI pointer)."""


def _make_callback_factory(
    kind: Kind = Kind.CONTINUOUS,
    characteristics: dict[CharacteristicName, Any] | None = None,
) -> tuple[UnuranCallback, MagicMock, MagicMock, _MockHandle]:
    """
    Build a UnuranCallback with mock CFFI objects.

    Returns (factory, mock_lib, mock_ffi, unuran_distr_handle).
    """
    mock_lib = MagicMock()
    mock_ffi = MagicMock()
    mock_ffi.callback.side_effect = lambda sig, func: func  # return callback as-is

    unuran_distr = _MockHandle()
    chars = characteristics or {}

    factory = UnuranCallback(
        unuran_distr=unuran_distr,
        kind=kind,
        lib=mock_lib,
        ffi=mock_ffi,
        characteristics=chars,
    )
    return factory, mock_lib, mock_ffi, unuran_distr


class TestUnuranCallbackCreation:
    """Tests for __init__ and callbacks property."""

    def test_callbacks_list_is_initially_empty(self) -> None:
        """Freshly created UnuranCallback has an empty callbacks list."""
        factory, _, _, _ = _make_callback_factory()

        assert factory.callbacks == []


class TestCreateCallback:
    """Tests for the _create_callback internal method."""

    def test_returns_none_for_missing_characteristic(self) -> None:
        """_create_callback returns None when the characteristic is not available."""
        factory, _, _, _ = _make_callback_factory(characteristics={})

        result = factory._create_callback(
            CharacteristicName.PDF, "double(double, const struct unur_distr*)"
        )

        assert result is None

    def test_returns_callback_for_available_characteristic(self) -> None:
        """_create_callback creates and returns a CFFI callback when the characteristic exists."""
        mock_pdf = MagicMock(return_value=0.5)
        factory, _, mock_ffi, _ = _make_callback_factory(
            characteristics={CharacteristicName.PDF: mock_pdf}
        )
        mock_ffi.callback.return_value = MagicMock(name="cffi_cb")

        result = factory._create_callback(
            CharacteristicName.PDF, "double(double, const struct unur_distr*)"
        )

        assert result is not None
        mock_ffi.callback.assert_called_once()

    def test_callback_wraps_characteristic_function(self) -> None:
        """The created callback evaluates the characteristic and returns a float."""
        called_with = []

        def mock_pdf(x: float) -> float:
            called_with.append(x)
            return 0.3

        # Use a real callable callback to verify the wrapping
        factory, _, mock_ffi, _ = _make_callback_factory(
            characteristics={CharacteristicName.PDF: mock_pdf}
        )

        # Capture the inner function passed to ffi.callback
        captured = {}

        def capture_callback(sig, inner_func):
            captured["func"] = inner_func
            return inner_func

        mock_ffi.callback.side_effect = capture_callback

        factory._create_callback(CharacteristicName.PDF, "double(double, const struct unur_distr*)")

        assert "func" in captured
        result = captured["func"](2.5, None)
        assert result == pytest.approx(0.3)
        assert called_with == [2.5]


class TestSetupCallback:
    """Tests for the setup_callback method."""

    def test_none_callback_is_silently_skipped(self) -> None:
        """Passing None as the callback skips registration without error."""
        factory, mock_lib, _, _ = _make_callback_factory()
        setter = MagicMock(return_value=0)

        factory.setup_callback(None, setter, "Failed to set callback (error code: {})")

        setter.assert_not_called()
        assert factory.callbacks == []

    def test_successful_callback_is_appended_to_callbacks_list(self) -> None:
        """A successfully registered callback is added to the live callbacks list."""
        factory, mock_lib, _, unuran_distr = _make_callback_factory()
        mock_cb = MagicMock(name="cb")
        setter = MagicMock(return_value=0)

        factory.setup_callback(mock_cb, setter, "Failed (error code: {})")

        assert mock_cb in factory.callbacks

    def test_successful_registration_calls_setter_with_correct_args(self) -> None:
        """setup_callback calls the setter with the distribution handle and callback."""
        factory, _, _, unuran_distr = _make_callback_factory()
        mock_cb = MagicMock(name="cb")
        setter = MagicMock(return_value=0)

        factory.setup_callback(mock_cb, setter, "err {}")

        setter.assert_called_once_with(unuran_distr, mock_cb)

    def test_nonzero_return_code_raises_runtime_error(self) -> None:
        """A non-zero return code from the setter triggers a RuntimeError."""
        factory, _, _, _ = _make_callback_factory()
        mock_cb = MagicMock(name="cb")
        setter = MagicMock(return_value=-1)

        with pytest.raises(RuntimeError, match="-1"):
            factory.setup_callback(mock_cb, setter, "Failed with code {}")

    def test_error_message_contains_error_code_placeholder(self) -> None:
        """The error message in the raised RuntimeError contains the actual code."""
        factory, _, _, _ = _make_callback_factory()
        mock_cb = MagicMock(name="cb")
        setter = MagicMock(return_value=42)

        with pytest.raises(RuntimeError, match="42"):
            factory.setup_callback(mock_cb, setter, "code is {}")


class TestSetupContinuousCallbacks:
    """Tests for setup_continuous_callbacks method."""

    def test_all_four_setters_called_when_all_chars_available(self) -> None:
        """All four setters are called and four callbacks stored when all chars are present."""
        chars = {
            CharacteristicName.PDF: MagicMock(return_value=0.5),
            CharacteristicName.DPDF: MagicMock(return_value=0.1),
            CharacteristicName.CDF: MagicMock(return_value=0.5),
            CharacteristicName.PPF: MagicMock(return_value=0.5),
        }
        factory, mock_lib, _, _ = _make_callback_factory(characteristics=chars)
        mock_lib.unur_distr_cont_set_pdf.return_value = 0
        mock_lib.unur_distr_cont_set_dpdf.return_value = 0
        mock_lib.unur_distr_cont_set_cdf.return_value = 0
        mock_lib.unur_distr_cont_set_invcdf.return_value = 0

        factory.setup_continuous_callbacks()

        mock_lib.unur_distr_cont_set_pdf.assert_called_once()
        mock_lib.unur_distr_cont_set_dpdf.assert_called_once()
        mock_lib.unur_distr_cont_set_cdf.assert_called_once()
        mock_lib.unur_distr_cont_set_invcdf.assert_called_once()
        assert len(factory.callbacks) == 4

    def test_only_pdf_setter_called_when_only_pdf_available(self) -> None:
        """When only PDF is available, only unur_distr_cont_set_pdf is called."""
        chars = {CharacteristicName.PDF: MagicMock(return_value=0.5)}
        factory, mock_lib, _, _ = _make_callback_factory(characteristics=chars)
        mock_lib.unur_distr_cont_set_pdf.return_value = 0

        factory.setup_continuous_callbacks()

        mock_lib.unur_distr_cont_set_pdf.assert_called_once()
        mock_lib.unur_distr_cont_set_dpdf.assert_not_called()
        mock_lib.unur_distr_cont_set_cdf.assert_not_called()
        mock_lib.unur_distr_cont_set_invcdf.assert_not_called()

    def test_setter_failure_raises_runtime_error(self) -> None:
        """If a setter returns a non-zero code, RuntimeError is raised."""
        chars = {CharacteristicName.PDF: MagicMock(return_value=0.5)}
        factory, mock_lib, _, _ = _make_callback_factory(characteristics=chars)
        mock_lib.unur_distr_cont_set_pdf.return_value = -3

        with pytest.raises(RuntimeError):
            factory.setup_continuous_callbacks()


class TestSetupDiscreteCallbacks:
    """Tests for setup_discrete_callbacks method."""

    def test_pmf_and_cdf_setters_called_when_both_available(self) -> None:
        """Both discrete setters are called and two callbacks stored when PMF and CDF present."""
        chars = {
            CharacteristicName.PMF: MagicMock(return_value=0.3),
            CharacteristicName.CDF: MagicMock(return_value=0.7),
        }
        factory, mock_lib, _, _ = _make_callback_factory(kind=Kind.DISCRETE, characteristics=chars)
        mock_lib.unur_distr_discr_set_pmf.return_value = 0
        mock_lib.unur_distr_discr_set_cdf.return_value = 0

        factory.setup_discrete_callbacks()

        mock_lib.unur_distr_discr_set_pmf.assert_called_once()
        mock_lib.unur_distr_discr_set_cdf.assert_called_once()
        assert len(factory.callbacks) == 2

    def test_only_pmf_setter_called_when_only_pmf_available(self) -> None:
        """When only PMF is available, only unur_distr_discr_set_pmf is called."""
        chars = {CharacteristicName.PMF: MagicMock(return_value=0.5)}
        factory, mock_lib, _, _ = _make_callback_factory(kind=Kind.DISCRETE, characteristics=chars)
        mock_lib.unur_distr_discr_set_pmf.return_value = 0

        factory.setup_discrete_callbacks()

        mock_lib.unur_distr_discr_set_pmf.assert_called_once()
        mock_lib.unur_distr_discr_set_cdf.assert_not_called()

    def test_pmf_setter_failure_raises_runtime_error(self) -> None:
        """If the PMF setter returns a non-zero code, RuntimeError is raised."""
        chars = {CharacteristicName.PMF: MagicMock(return_value=0.5)}
        factory, mock_lib, _, _ = _make_callback_factory(kind=Kind.DISCRETE, characteristics=chars)
        mock_lib.unur_distr_discr_set_pmf.return_value = -2

        with pytest.raises(RuntimeError):
            factory.setup_discrete_callbacks()
