"""
Tests for CharacteristicOption descriptor.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import pytest

from pysatl_core.distributions.computations.base import CharacteristicOption


class TestCharacteristicOption:
    """Tests for the CharacteristicOption dataclass."""

    def test_resolve_returns_default_when_key_absent(self) -> None:
        opt = CharacteristicOption(name="eps", type=float, default=1e-6)
        kwargs: dict[str, Any] = {}
        assert opt.resolve(kwargs) == 1e-6

    def test_resolve_returns_caller_value_when_present(self) -> None:
        opt = CharacteristicOption(name="eps", type=float, default=1e-6)
        kwargs: dict[str, Any] = {"eps": 1e-3}
        assert opt.resolve(kwargs) == pytest.approx(1e-3)

    def test_resolve_pops_key_from_kwargs(self) -> None:
        opt = CharacteristicOption(name="eps", type=float, default=1e-6)
        kwargs: dict[str, Any] = {"eps": 1e-3, "other": 42}
        opt.resolve(kwargs)
        assert "eps" not in kwargs
        assert "other" in kwargs

    def test_resolve_casts_to_declared_type(self) -> None:
        opt = CharacteristicOption(name="x0", type=float, default=0.0)
        kwargs: dict[str, Any] = {"x0": 1}
        result = opt.resolve(kwargs)
        assert isinstance(result, float)
        assert result == 1.0

    def test_resolve_raises_type_error_on_bad_cast(self) -> None:
        opt = CharacteristicOption(name="eps", type=float, default=1e-6)
        kwargs: dict[str, Any] = {"eps": "not_a_number"}
        with pytest.raises(TypeError, match="cannot convert"):
            opt.resolve(kwargs)

    def test_resolve_raises_value_error_on_failed_validation(self) -> None:
        opt = CharacteristicOption(
            name="eps", type=float, default=1e-6, validate=lambda v: 0 < v < 0.5
        )
        kwargs: dict[str, Any] = {"eps": -1.0}
        with pytest.raises(ValueError, match="failed validation"):
            opt.resolve(kwargs)

    def test_resolve_passes_validation_when_valid(self) -> None:
        opt = CharacteristicOption(
            name="eps", type=float, default=1e-6, validate=lambda v: 0 < v < 0.5
        )
        kwargs: dict[str, Any] = {"eps": 0.1}
        assert opt.resolve(kwargs) == pytest.approx(0.1)

    def test_frozen_dataclass(self) -> None:
        opt = CharacteristicOption(name="eps", type=float, default=1e-6)
        with pytest.raises(AttributeError):
            opt.name = "other"  # type: ignore[misc]
