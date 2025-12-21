from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import pytest

from pysatl_core.stats._unuran.bindings._core.unuran_sampler import DefaultUnuranSampler
from pysatl_core.types import CharacteristicName, EuclideanDistributionType, Kind


@pytest.fixture
def sampler_stub() -> DefaultUnuranSampler:
    """Return an uninitialized sampler instance with default ffi/lib placeholders."""
    sampler = object.__new__(DefaultUnuranSampler)
    sampler._ffi = SimpleNamespace(callback=lambda sig, fn: fn, NULL=None)
    sampler._lib = SimpleNamespace()
    sampler.distr = SimpleNamespace(analytical_computations={})
    sampler._callbacks = []
    sampler._cleaned_up = False
    return sampler


@pytest.fixture
def dummy_distribution() -> SimpleNamespace:
    """Factory for a minimal continuous univariate distribution."""
    return SimpleNamespace(
        distribution_type=EuclideanDistributionType(kind=Kind.CONTINUOUS, dimension=1),
        analytical_computations={CharacteristicName.PDF: lambda x: x},
        support=None,
    )


@pytest.fixture
def dummy_cffi(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide a fake _unuran_cffi module with ffi/lib placeholders."""

    class DummyFFI(SimpleNamespace):
        NULL = object()

        def callback(self, signature: str, func: object) -> object:
            return func

    dummy = SimpleNamespace(ffi=DummyFFI(), lib=SimpleNamespace())
    monkeypatch.setattr("pysatl_core.stats._unuran.bindings._unuran_cffi", dummy)
    return dummy
