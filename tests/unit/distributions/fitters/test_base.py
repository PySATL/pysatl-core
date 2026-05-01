"""
Tests for fitter abstractions: ComputationOption, CharacteristicOption, and FitterDescriptor.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import pytest

from pysatl_core.distributions.computations.base import (
    CharacteristicOption,
    ComputationOption,
    FitterDescriptor,
)
from pysatl_core.types import CharacteristicName


class TestFitterDescriptor:
    """Tests for the FitterDescriptor dataclass."""

    @staticmethod
    def _dummy_fitter(distribution: Any, /, **kwargs: Any) -> Any:
        return None

    def _make_descriptor(self, **overrides: Any) -> FitterDescriptor:
        defaults: dict[str, Any] = {
            "name": "test_fitter",
            "target": CharacteristicName.CDF,
            "sources": [CharacteristicName.PDF],
            "fitter": self._dummy_fitter,
            "characteristic_options": (CharacteristicOption(name="eps", type=float, default=1e-6),),
            "computation_options": (
                ComputationOption(name="limit", type=int, default=200, validate=lambda v: v > 0),
                ComputationOption(name="h", type=float, default=1e-5),
            ),
            "constraint_tags": frozenset({"continuous", "univariate"}),
            "description": "Test fitter.",
        }
        defaults.update(overrides)
        return FitterDescriptor(**defaults)

    def test_resolve_options_returns_defaults(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {}
        opts = desc.resolve_options(kwargs)
        assert opts == {"eps": 1e-6, "limit": 200, "h": 1e-5}

    def test_resolve_options_uses_caller_values(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"eps": 1e-3, "limit": 500, "h": 0.01}
        opts = desc.resolve_options(kwargs)
        assert opts == {"eps": pytest.approx(1e-3), "limit": 500, "h": pytest.approx(0.01)}

    def test_resolve_options_leaves_unrecognised_keys(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"limit": 500, "extra_key": "hello"}
        desc.resolve_options(kwargs)
        assert "extra_key" in kwargs
        assert "limit" not in kwargs

    def test_resolve_characteristic_options_does_not_consume_computation_keys(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"eps": 1e-3, "limit": 500}
        desc.resolve_characteristic_options(kwargs)
        assert "limit" in kwargs
        assert "eps" not in kwargs

    def test_resolve_computation_options_does_not_consume_characteristic_keys(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"eps": 1e-3, "limit": 500}
        desc.resolve_computation_options(kwargs)
        assert "eps" in kwargs
        assert "limit" not in kwargs

    def test_option_names_returns_all(self) -> None:
        desc = self._make_descriptor()
        assert desc.option_names() == ("eps", "limit", "h")

    def test_characteristic_option_names(self) -> None:
        desc = self._make_descriptor()
        assert desc.characteristic_option_names() == ("eps",)

    def test_computation_option_names(self) -> None:
        desc = self._make_descriptor()
        assert desc.computation_option_names() == ("limit", "h")

    def test_option_defaults_returns_all(self) -> None:
        desc = self._make_descriptor()
        assert desc.option_defaults() == {"eps": 1e-6, "limit": 200, "h": 1e-5}

    def test_empty_options(self) -> None:
        desc = self._make_descriptor(characteristic_options=(), computation_options=())
        assert desc.option_names() == ()
        assert desc.option_defaults() == {}
        assert desc.resolve_options({}) == {}

    def test_frozen_dataclass(self) -> None:
        desc = self._make_descriptor()
        with pytest.raises(AttributeError):
            desc.name = "other"  # type: ignore[misc]

    def test_to_computation_method_returns_fitter_method(self) -> None:
        """FitterDescriptor.to_computation_method() returns a FitterMethod (always cacheable)."""
        desc = self._make_descriptor()
        cm = desc.to_computation_method()
        assert cm.fitter is not None
        assert cm.cacheable is True
        assert cm.target == CharacteristicName.CDF
        assert list(cm.sources) == [CharacteristicName.PDF]

    def test_all_builtin_descriptors_produce_cacheable_methods(self) -> None:
        """All 8 built-in fitters produce cacheable FitterMethod instances."""
        from pysatl_core.distributions.computations import ALL_FITTER_DESCRIPTORS

        for desc in ALL_FITTER_DESCRIPTORS:
            cm = desc.to_computation_method()
            assert (
                cm.cacheable is True
            ), f"Descriptor '{desc.name}' should produce a cacheable FitterMethod"
