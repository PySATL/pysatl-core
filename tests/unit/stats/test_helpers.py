from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace

import pytest

from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig
from pysatl_core.stats._unuran.bindings._core.helpers import (
    _get_available_characteristics,
    _select_best_method,
)
from pysatl_core.types import CharacteristicName, Kind


class TestHelpers:
    """Tests covering helper functions in helpers.py."""

    def test_get_available_characteristics_returns_empty_without_analytics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensures registry is not consulted when distribution lacks analytical characteristics."""
        distr = SimpleNamespace(analytical_computations={})

        monkeypatch.setattr(
            "pysatl_core.distributions.registry.characteristic_registry",
            lambda: (_ for _ in ()).throw(AssertionError("registry should not be called")),
        )

        assert _get_available_characteristics(distr) == set()

    def test_get_available_characteristics_collects_reachable_nodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validates BFS collects base characteristic and all reachable nodes from registry view."""
        adjacency = {
            CharacteristicName.PDF: {CharacteristicName.CDF},
            CharacteristicName.CDF: {CharacteristicName.PPF},
        }

        class DummyView:
            def __init__(self) -> None:
                succ_targets = {node for targets in adjacency.values() for node in targets}
                self.all_characteristics = set(adjacency.keys()) | succ_targets

            def successors_nodes(self, node: CharacteristicName) -> set[CharacteristicName]:
                return adjacency.get(node, set())

        class DummyRegistry:
            def view(self, distr: object) -> DummyView:
                return DummyView()

        monkeypatch.setattr(
            "pysatl_core.distributions.registry.characteristic_registry",
            lambda: DummyRegistry(),
        )

        distr = SimpleNamespace(analytical_computations={CharacteristicName.PDF: object()})

        result = _get_available_characteristics(distr)
        assert result == {
            CharacteristicName.PDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
        }

    def test_get_available_characteristics_skips_nodes_missing_in_view(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that nodes absent from the registry view remain as-is without BFS expansion."""

        class DummyView:
            all_characteristics = {CharacteristicName.CDF}

            def successors_nodes(self, node: CharacteristicName) -> set[CharacteristicName]:
                return {CharacteristicName.PPF} if node == CharacteristicName.CDF else set()

        class DummyRegistry:
            def view(self, distr: object) -> DummyView:
                return DummyView()

        monkeypatch.setattr(
            "pysatl_core.distributions.registry.characteristic_registry",
            lambda: DummyRegistry(),
        )

        distr = SimpleNamespace(analytical_computations={CharacteristicName.PMF: object()})

        result = _get_available_characteristics(distr)
        assert result == {CharacteristicName.PMF}

    def test_select_best_method_prefers_ppf_when_enabled(self) -> None:
        """Ensures AUTO selects PINV when PPF is available and use_ppf is True."""
        config = UnuranMethodConfig(use_ppf=True, use_pdf=False, use_cdf=False)
        result = _select_best_method({CharacteristicName.PPF}, Kind.CONTINUOUS, config)
        assert result == UnuranMethod.PINV

    def test_select_best_method_prefers_pdf_fallback(self) -> None:
        """Validates PDF availability leads to PINV even when use_ppf=False."""
        config = UnuranMethodConfig(use_ppf=False, use_pdf=True, use_cdf=False)
        result = _select_best_method({CharacteristicName.PDF}, Kind.CONTINUOUS, config)
        assert result == UnuranMethod.PINV

    def test_select_best_method_prefers_cdf_when_requested(self) -> None:
        """Checks CDF-only continuous distributions fall back to NINV if use_cdf=True."""
        config = UnuranMethodConfig(use_ppf=False, use_pdf=False, use_cdf=True)
        result = _select_best_method({CharacteristicName.CDF}, Kind.CONTINUOUS, config)
        assert result == UnuranMethod.NINV

    def test_select_best_method_requires_characteristics_for_continuous(self) -> None:
        """Raises RuntimeError if no usable characteristics exist for continuous distributions."""
        config = UnuranMethodConfig(use_ppf=True, use_pdf=True, use_cdf=True)
        with pytest.raises(RuntimeError):
            _select_best_method(set(), Kind.CONTINUOUS, config)

    def test_select_best_method_prefers_dgt_for_discrete(self) -> None:
        """Verifies discrete distributions with PMF select DGT method."""
        config = UnuranMethodConfig()
        result = _select_best_method({CharacteristicName.PMF}, Kind.DISCRETE, config)
        assert result == UnuranMethod.DGT

    def test_select_best_method_allows_pdf_for_discrete(self) -> None:
        """Confirms discrete distributions can use DGT even if only PDF is present."""
        config = UnuranMethodConfig()
        result = _select_best_method({CharacteristicName.PDF}, Kind.DISCRETE, config)
        assert result == UnuranMethod.DGT

    def test_select_best_method_requires_pmf_or_pdf_for_discrete(self) -> None:
        """Raises RuntimeError when discrete distributions lack PMF/PDF characteristics."""
        config = UnuranMethodConfig()
        with pytest.raises(RuntimeError):
            _select_best_method({CharacteristicName.CDF}, Kind.DISCRETE, config)
