from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import pytest

from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig
from pysatl_core.stats._unuran.bindings._core.unuran_sampler import DefaultUnuranSampler
from pysatl_core.types import CharacteristicName, EuclideanDistributionType, Kind


class TestInitialization:
    """Tests covering sampler initialization and configuration."""

    def test_init_raises_when_bindings_absent(
        self, monkeypatch: pytest.MonkeyPatch, dummy_distribution: SimpleNamespace
    ) -> None:
        """Sampler should raise if CFFI bindings are missing."""
        monkeypatch.setattr("pysatl_core.stats._unuran.bindings._unuran_cffi", None)

        with pytest.raises(RuntimeError, match="UNURAN CFFI bindings are not available"):
            DefaultUnuranSampler(dummy_distribution)

    def test_init_rejects_non_euclidean_distribution(self, dummy_cffi: SimpleNamespace) -> None:
        """Non-Euclidean distribution types must be rejected."""
        distr = SimpleNamespace(
            distribution_type=SimpleNamespace(kind=Kind.CONTINUOUS, dimension=1),
            analytical_computations={},
            support=None,
        )

        with pytest.raises(RuntimeError, match="Unsupported distribution type"):
            DefaultUnuranSampler(distr)

    def test_init_rejects_multi_dimensional_distribution(self, dummy_cffi: SimpleNamespace) -> None:
        """Sampler should only allow univariate (dimension=1) distributions."""
        distr = SimpleNamespace(
            distribution_type=EuclideanDistributionType(kind=Kind.CONTINUOUS, dimension=2),
            analytical_computations={},
            support=None,
        )

        with pytest.raises(RuntimeError, match="Unsupported distribution dimension"):
            DefaultUnuranSampler(distr)

    def test_auto_method_selection_uses_helper(
        self,
        dummy_cffi: SimpleNamespace,
        dummy_distribution: SimpleNamespace,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AUTO method configuration should delegate to helper with available characteristics."""
        chosen_method = UnuranMethod.NINV
        captured_args: list[tuple[set[str], Kind]] = []

        def fake_select(chars: set[str], kind: Kind, config: UnuranMethodConfig) -> UnuranMethod:
            captured_args.append((chars, kind))
            return chosen_method

        monkeypatch.setattr(
            "pysatl_core.stats._unuran.bindings._core.unuran_sampler._get_available_characteristics",
            lambda distr: {CharacteristicName.PDF},
        )
        monkeypatch.setattr(
            "pysatl_core.stats._unuran.bindings._core.unuran_sampler._select_best_method",
            fake_select,
        )
        monkeypatch.setattr(DefaultUnuranSampler, "_initialize_unuran", lambda self, seed: None)

        sampler = DefaultUnuranSampler(
            dummy_distribution, UnuranMethodConfig(method=UnuranMethod.AUTO)
        )

        assert sampler.method == chosen_method
        assert captured_args == [({CharacteristicName.PDF}, Kind.CONTINUOUS)]
