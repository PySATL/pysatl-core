"""
Tests for fitter abstractions: FitterOption and FitterDescriptor.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import pytest

from pysatl_core.distributions.fitters.base import FitterDescriptor, FitterOption
from pysatl_core.types import CharacteristicName


class TestFitterOption:
    """Tests for the FitterOption dataclass."""

    def test_resolve_returns_default_when_key_absent(self) -> None:
        opt = FitterOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {}
        assert opt.resolve(kwargs) == 200

    def test_resolve_returns_caller_value_when_present(self) -> None:
        opt = FitterOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {"limit": 500}
        assert opt.resolve(kwargs) == 500

    def test_resolve_pops_key_from_kwargs(self) -> None:
        opt = FitterOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {"limit": 500, "other": 42}
        opt.resolve(kwargs)
        assert "limit" not in kwargs
        assert "other" in kwargs

    def test_resolve_casts_to_declared_type(self) -> None:
        opt = FitterOption(name="h", type=float, default=1e-5)
        kwargs: dict[str, Any] = {"h": 1}
        result = opt.resolve(kwargs)
        assert isinstance(result, float)
        assert result == 1.0

    def test_resolve_raises_type_error_on_bad_cast(self) -> None:
        opt = FitterOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {"limit": "not_a_number"}
        with pytest.raises(TypeError, match="cannot convert"):
            opt.resolve(kwargs)

    def test_resolve_raises_value_error_on_failed_validation(self) -> None:
        opt = FitterOption(name="limit", type=int, default=200, validate=lambda v: v > 0)
        kwargs: dict[str, Any] = {"limit": -1}
        with pytest.raises(ValueError, match="failed validation"):
            opt.resolve(kwargs)

    def test_resolve_passes_validation_when_valid(self) -> None:
        opt = FitterOption(name="limit", type=int, default=200, validate=lambda v: v > 0)
        kwargs: dict[str, Any] = {"limit": 100}
        assert opt.resolve(kwargs) == 100

    def test_resolve_no_validation_when_none(self) -> None:
        opt = FitterOption(name="x", type=float, default=0.0, validate=None)
        kwargs: dict[str, Any] = {"x": -999.0}
        assert opt.resolve(kwargs) == -999.0

    def test_frozen_dataclass(self) -> None:
        opt = FitterOption(name="x", type=float, default=0.0)
        with pytest.raises(AttributeError):
            opt.name = "y"  # type: ignore[misc]


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
            "options": (
                FitterOption(name="limit", type=int, default=200, validate=lambda v: v > 0),
                FitterOption(name="h", type=float, default=1e-5),
            ),
            "tags": frozenset({"continuous", "univariate"}),
            "priority": 0,
            "description": "Test fitter.",
        }
        defaults.update(overrides)
        return FitterDescriptor(**defaults)

    def test_resolve_options_returns_defaults(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {}
        opts = desc.resolve_options(kwargs)
        assert opts == {"limit": 200, "h": 1e-5}

    def test_resolve_options_uses_caller_values(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"limit": 500, "h": 0.01}
        opts = desc.resolve_options(kwargs)
        assert opts == {"limit": 500, "h": 0.01}

    def test_resolve_options_leaves_unrecognised_keys(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"limit": 500, "extra_key": "hello"}
        desc.resolve_options(kwargs)
        assert "extra_key" in kwargs
        assert "limit" not in kwargs

    def test_option_names(self) -> None:
        desc = self._make_descriptor()
        assert desc.option_names() == ("limit", "h")

    def test_option_defaults(self) -> None:
        desc = self._make_descriptor()
        assert desc.option_defaults() == {"limit": 200, "h": 1e-5}

    def test_empty_options(self) -> None:
        desc = self._make_descriptor(options=())
        assert desc.option_names() == ()
        assert desc.option_defaults() == {}
        assert desc.resolve_options({}) == {}

    def test_frozen_dataclass(self) -> None:
        desc = self._make_descriptor()
        with pytest.raises(AttributeError):
            desc.name = "other"  # type: ignore[misc]

    def test_cacheable_defaults_to_true(self) -> None:
        desc = self._make_descriptor()
        assert desc.cacheable is True

    def test_cacheable_can_be_set_to_false(self) -> None:
        desc = self._make_descriptor(cacheable=False)
        assert desc.cacheable is False

    def test_to_computation_method_cacheable_uses_fitter_slot(self) -> None:
        """When cacheable=True, to_computation_method() populates the fitter slot."""
        desc = self._make_descriptor(cacheable=True)
        cm = desc.to_computation_method()
        assert cm.fitter is not None
        assert cm.evaluator is None
        assert cm.cacheable is True
        assert cm.target == CharacteristicName.CDF
        assert list(cm.sources) == [CharacteristicName.PDF]

    def test_to_computation_method_non_cacheable_uses_evaluator_slot(self) -> None:
        """When cacheable=False, to_computation_method() populates the evaluator slot."""
        desc = self._make_descriptor(cacheable=False)
        cm = desc.to_computation_method()
        assert cm.fitter is None
        assert cm.evaluator is not None
        assert cm.cacheable is False
        assert cm.target == CharacteristicName.CDF
        assert list(cm.sources) == [CharacteristicName.PDF]

    def test_all_builtin_descriptors_are_cacheable(self) -> None:
        """All 8 built-in fitters use fit_ prefix and should be cacheable."""
        from pysatl_core.distributions.fitters import ALL_FITTER_DESCRIPTORS

        for desc in ALL_FITTER_DESCRIPTORS:
            assert (
                desc.cacheable is True
            ), f"Descriptor '{desc.name}' should be cacheable (fit_ prefix)"
