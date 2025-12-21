from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

from pysatl_core.stats._unuran.bindings._core.unuran_sampler import DefaultUnuranSampler
from pysatl_core.types import CharacteristicName


class TestCallbacks:
    """Tests for PDF/PMF/CDF/DPDF callbacks and error helpers."""

    def test_create_pdf_callback_returns_wrapper(self, sampler_stub: DefaultUnuranSampler) -> None:
        """Ensures PDF callback wraps the distribution's PDF for continuous case."""
        sampler_stub._is_continuous = True
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.PDF: lambda x: x * 2}
        )
        callback = sampler_stub._create_pdf_callback()
        assert callable(callback)
        assert callback(1.5, None) == 3.0

    def test_create_pdf_callback_none_when_not_continuous(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Discrete distributions should not expose PDF callback."""
        sampler_stub._is_continuous = False
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.PDF: lambda x: x}
        )
        assert sampler_stub._create_pdf_callback() is None

    def test_create_pmf_callback_prefers_pmf(self, sampler_stub: DefaultUnuranSampler) -> None:
        """Discrete samplers prefer PMF characteristic when available."""
        sampler_stub._is_continuous = False
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={
                CharacteristicName.PMF: lambda x: x + 1,
                CharacteristicName.PDF: lambda x: x + 2,
            }
        )
        callback = sampler_stub._create_pmf_callback()
        assert callable(callback)
        assert callback(3, None) == 4.0

    def test_create_pmf_callback_falls_back_to_pdf(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """PMF should fall back to PDF when PMF is missing."""
        sampler_stub._is_continuous = False
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.PDF: lambda x: x + 5}
        )
        callback = sampler_stub._create_pmf_callback()
        assert callable(callback)
        assert callback(2, None) == 7.0

    def test_create_cdf_callback_continuous(self, sampler_stub: DefaultUnuranSampler) -> None:
        """Continuous CDF callback should pass floating-point arguments through."""
        sampler_stub._is_continuous = True
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.CDF: lambda x: x**2}
        )
        callback = sampler_stub._create_cdf_callback()
        assert callable(callback)
        assert callback(3.0, None) == 9.0

    def test_create_cdf_callback_discrete(self, sampler_stub: DefaultUnuranSampler) -> None:
        """Discrete CDF callback should accept integer inputs."""
        sampler_stub._is_continuous = False
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.CDF: lambda x: x + 0.5}
        )
        callback = sampler_stub._create_cdf_callback()
        assert callable(callback)
        assert callback(4, None) == 4.5

    def test_create_dpdf_callback_is_none(self, sampler_stub: DefaultUnuranSampler) -> None:
        """Currently no dPDF callback should be created."""
        assert sampler_stub._create_dpdf_callback() is None

    def test_get_unuran_error_message_formats_errno(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Error message should include errno and text from UNURAN."""
        sampler_stub._lib.unur_get_errno = lambda: 5
        sampler_stub._lib.unur_get_strerror = lambda err: b"boom"
        sampler_stub._ffi = SimpleNamespace(string=lambda b: b, NULL=None)
        msg = sampler_stub._get_unuran_error_message("Failed")
        assert "Failed" in msg and "errno: 5" in msg and "boom" in msg
