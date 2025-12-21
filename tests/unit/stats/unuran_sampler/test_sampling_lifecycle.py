from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from pysatl_core.stats._unuran.api import UnuranMethod
from pysatl_core.stats._unuran.bindings._core.unuran_sampler import DefaultUnuranSampler


class TestSamplingAndLifecycle:
    """Tests for sampling behavior and lifecycle helpers."""

    def test_sample_continuous_uses_unur_sample_cont(self) -> None:
        """Sampling continuous distributions should call unur_sample_cont each time."""
        sampler = object.__new__(DefaultUnuranSampler)
        values = iter([0.1, 0.2, 0.3])

        def fake_unur_sample_cont(gen: object) -> float:
            return next(values)

        sampler._is_continuous = True
        sampler._lib = SimpleNamespace(unur_sample_cont=fake_unur_sample_cont)
        sampler._unuran_gen = object()
        sampler._unuran_distr = object()
        sampler._ffi = SimpleNamespace(NULL=None)

        result = sampler.sample(3)
        assert np.allclose(result, np.array([0.1, 0.2, 0.3]))

    def test_sample_discrete_uses_unur_sample_discr(self) -> None:
        """Discrete sampling should call unur_sample_discr and cast to floats."""
        sampler = object.__new__(DefaultUnuranSampler)
        calls: list[int] = []

        def fake_unur_sample_discr(gen: object) -> int:
            value = len(calls)
            calls.append(value)
            return value

        sampler._is_continuous = False
        sampler._lib = SimpleNamespace(unur_sample_discr=fake_unur_sample_discr)
        sampler._unuran_gen = object()
        sampler._unuran_distr = object()
        sampler._ffi = SimpleNamespace(NULL=None)

        result = sampler.sample(4)
        assert np.array_equal(result, np.array([0.0, 1.0, 2.0, 3.0]))
        assert len(calls) == 4

    def test_sample_raises_when_not_initialized(self) -> None:
        """Calling sample without initialized generator should raise."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._is_continuous = True
        sampler._lib = SimpleNamespace()
        sampler._ffi = SimpleNamespace(NULL=None)
        sampler._unuran_gen = None
        sampler._unuran_distr = None

        with pytest.raises(RuntimeError, match="Sampler is not initialized"):
            sampler.sample(1)

    def test_cleanup_idempotent_and_frees_resources(self) -> None:
        """_cleanup should free generator/distr once and tolerate repeated calls."""
        sampler = object.__new__(DefaultUnuranSampler)
        freed: list[str] = []

        class DummyLib:
            def unur_free(self, gen: object) -> None:
                freed.append("gen")

            def unur_par_free(self, par: object) -> None:
                freed.append("par")

            def unur_distr_free(self, distr: object) -> None:
                freed.append("distr")

        sampler._lib = DummyLib()
        sampler._ffi = SimpleNamespace(NULL=None)
        sampler._unuran_gen = "GEN"
        sampler._unuran_par = "PAR"
        sampler._unuran_distr = "DISTR"
        sampler._callbacks = []
        sampler._cleaned_up = False

        sampler._cleanup()
        sampler._cleanup()

        assert freed == ["gen", "distr"]

    def test_reset_reinitializes_when_not_initialized(self) -> None:
        """reset() should trigger initialization only when sampler is not ready."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._unuran_gen = None
        sampler._unuran_distr = None

        with patch.object(DefaultUnuranSampler, "_initialize_unuran", autospec=True) as mock_init:
            sampler.reset()

        mock_init.assert_called_once_with(sampler, None)

    def test_is_initialized_property(self) -> None:
        """is_initialized should reflect generator/distr being non-null."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._ffi = SimpleNamespace(NULL=None)
        sampler._unuran_gen = object()
        sampler._unuran_distr = object()
        assert sampler.is_initialized is True
        sampler._unuran_gen = None
        assert sampler.is_initialized is False

    def test_method_property(self) -> None:
        """method property should expose the configured UnuranMethod."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._method = UnuranMethod.PINV
        assert sampler.method == UnuranMethod.PINV

    def test_del_calls_cleanup(self) -> None:
        """__del__ should invoke cleanup safely."""
        sampler = object.__new__(DefaultUnuranSampler)

        with patch.object(DefaultUnuranSampler, "_cleanup", autospec=True) as mock_cleanup:
            sampler.__del__()

        mock_cleanup.assert_called_once_with(sampler)
