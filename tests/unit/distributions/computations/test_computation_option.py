"""
Tests for ComputationOption descriptor.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import pytest

from pysatl_core.distributions.computations.base import ComputationOption


class TestComputationOption:
    """Tests for the ComputationOption dataclass."""

    def test_resolve_returns_default_when_key_absent(self) -> None:
        opt = ComputationOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {}
        assert opt.resolve(kwargs) == 200

    def test_resolve_returns_caller_value_when_present(self) -> None:
        opt = ComputationOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {"limit": 500}
        assert opt.resolve(kwargs) == 500

    def test_resolve_pops_key_from_kwargs(self) -> None:
        opt = ComputationOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {"limit": 500, "other": 42}
        opt.resolve(kwargs)
        assert "limit" not in kwargs
        assert "other" in kwargs

    def test_resolve_casts_to_declared_type(self) -> None:
        opt = ComputationOption(name="h", type=float, default=1e-5)
        kwargs: dict[str, Any] = {"h": 1}
        result = opt.resolve(kwargs)
        assert isinstance(result, float)
        assert result == 1.0

    def test_resolve_raises_type_error_on_bad_cast(self) -> None:
        opt = ComputationOption(name="limit", type=int, default=200)
        kwargs: dict[str, Any] = {"limit": "not_a_number"}
        with pytest.raises(TypeError, match="cannot convert"):
            opt.resolve(kwargs)

    def test_resolve_raises_value_error_on_failed_validation(self) -> None:
        opt = ComputationOption(name="limit", type=int, default=200, validate=lambda v: v > 0)
        kwargs: dict[str, Any] = {"limit": -1}
        with pytest.raises(ValueError, match="failed validation"):
            opt.resolve(kwargs)

    def test_resolve_passes_validation_when_valid(self) -> None:
        opt = ComputationOption(name="limit", type=int, default=200, validate=lambda v: v > 0)
        kwargs: dict[str, Any] = {"limit": 100}
        assert opt.resolve(kwargs) == 100

    def test_resolve_no_validation_when_none(self) -> None:
        opt = ComputationOption(name="x", type=float, default=0.0, validate=None)
        kwargs: dict[str, Any] = {"x": -999.0}
        assert opt.resolve(kwargs) == -999.0

    def test_frozen_dataclass(self) -> None:
        opt = ComputationOption(name="x", type=float, default=0.0)
        with pytest.raises(AttributeError):
            opt.name = "y"  # type: ignore[misc]
