"""
Tests for EvaluatorDescriptor and EvaluatorMethod.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import numpy as np
import pytest

from pysatl_core.distributions.computations.descriptors import EvaluatorDescriptor
from pysatl_core.distributions.computations.options import (
    CharacteristicOption,
    ComputationOption,
)
from pysatl_core.types import CharacteristicName, NumericArray


class TestEvaluatorDescriptor:
    """Tests for the EvaluatorDescriptor dataclass (non-cacheable evaluators)."""

    @staticmethod
    def _dummy_evaluator(distribution: Any, x: NumericArray, /, **kwargs: Any) -> NumericArray:
        return np.zeros_like(np.asarray(x, dtype=float))

    def _make_descriptor(self, **overrides: Any) -> EvaluatorDescriptor:
        defaults: dict[str, Any] = {
            "name": "test_evaluator",
            "target": CharacteristicName.PDF,
            "sources": [CharacteristicName.CDF],
            "evaluator": self._dummy_evaluator,
            "characteristic_options": (
                CharacteristicOption(
                    name="tol", type=float, default=1e-8, validate=lambda v: v > 0
                ),
            ),
            "computation_options": (
                ComputationOption(name="max_iter", type=int, default=50, validate=lambda v: v > 0),
            ),
            "constraint_tags": frozenset({"continuous", "univariate"}),
            "description": "Test evaluator.",
        }
        defaults.update(overrides)
        return EvaluatorDescriptor(**defaults)

    def test_options_property_combines_both_kinds(self) -> None:
        desc = self._make_descriptor()
        names = tuple(o.name for o in desc.options)
        # characteristic options come first
        assert names == ("tol", "max_iter")

    def test_resolve_characteristic_options_returns_defaults(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {}
        opts = desc.resolve_characteristic_options(kwargs)
        assert opts == {"tol": 1e-8}

    def test_resolve_characteristic_options_does_not_consume_computation_keys(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"tol": 1e-6, "max_iter": 100}
        desc.resolve_characteristic_options(kwargs)
        assert "max_iter" in kwargs
        assert "tol" not in kwargs

    def test_resolve_computation_options_returns_defaults(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {}
        opts = desc.resolve_computation_options(kwargs)
        assert opts == {"max_iter": 50}

    def test_resolve_computation_options_does_not_consume_characteristic_keys(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"tol": 1e-6, "max_iter": 100}
        desc.resolve_computation_options(kwargs)
        assert "tol" in kwargs
        assert "max_iter" not in kwargs

    def test_resolve_options_returns_all_defaults(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {}
        opts = desc.resolve_options(kwargs)
        assert opts == {"tol": 1e-8, "max_iter": 50}

    def test_resolve_options_uses_caller_values(self) -> None:
        desc = self._make_descriptor()
        kwargs: dict[str, Any] = {"tol": 1e-6, "max_iter": 100}
        opts = desc.resolve_options(kwargs)
        assert opts["tol"] == pytest.approx(1e-6)
        assert opts["max_iter"] == 100

    def test_option_names_returns_all(self) -> None:
        desc = self._make_descriptor()
        assert desc.option_names() == ("tol", "max_iter")

    def test_characteristic_option_names(self) -> None:
        desc = self._make_descriptor()
        assert desc.characteristic_option_names() == ("tol",)

    def test_computation_option_names(self) -> None:
        desc = self._make_descriptor()
        assert desc.computation_option_names() == ("max_iter",)

    def test_option_defaults_returns_all(self) -> None:
        desc = self._make_descriptor()
        assert desc.option_defaults() == {"tol": 1e-8, "max_iter": 50}

    def test_frozen_dataclass(self) -> None:
        desc = self._make_descriptor()
        with pytest.raises(AttributeError):
            desc.name = "other"  # type: ignore[misc]

    def test_to_computation_method_returns_evaluator_method(self) -> None:
        from pysatl_core.distributions.computations.computation import EvaluatorMethod

        desc = self._make_descriptor()
        cm = desc.to_computation_method()
        assert isinstance(cm, EvaluatorMethod)
        assert cm.evaluator is not None
        assert cm.target == CharacteristicName.PDF
        assert list(cm.sources) == [CharacteristicName.CDF]

    def test_empty_options(self) -> None:
        desc = self._make_descriptor(characteristic_options=(), computation_options=())
        assert desc.option_names() == ()
        assert desc.option_defaults() == {}
        assert desc.resolve_options({}) == {}
        assert desc.resolve_characteristic_options({}) == {}
        assert desc.resolve_computation_options({}) == {}


class TestEvaluatorMethodPrepare:
    """Tests for EvaluatorMethod.prepare() option forwarding."""

    def test_prepare_time_options_are_forwarded_to_evaluator(self) -> None:
        """
        Options passed to ``prepare()`` must be forwarded to the evaluator
        when the returned wrapper is called, so that edge-level options
        (resolved by the strategy before calling ``prepare``) are not silently
        dropped.
        """
        from pysatl_core.distributions.computations.computation import EvaluatorMethod

        received: dict[str, Any] = {}

        def evaluator(_distribution: Any, x: NumericArray, /, **kwargs: Any) -> NumericArray:
            received.update(kwargs)
            return np.asarray(x, dtype=float)

        method = EvaluatorMethod(target="cdf", sources=["pdf"], evaluator=evaluator)
        fitted = method.prepare(None, tol=1e-6, max_iter=100)  # type: ignore[arg-type]
        fitted(np.array([0.5]))

        assert received.get("tol") == pytest.approx(1e-6)
        assert received.get("max_iter") == 100

    def test_call_time_options_override_prepare_time_options(self) -> None:
        """
        Options supplied at call time must take precedence over those
        supplied at prepare time, so callers can still override per-call.
        """
        from pysatl_core.distributions.computations.computation import EvaluatorMethod

        received: dict[str, Any] = {}

        def evaluator(_distribution: Any, x: NumericArray, /, **kwargs: Any) -> NumericArray:
            received.update(kwargs)
            return np.asarray(x, dtype=float)

        method = EvaluatorMethod(target="cdf", sources=["pdf"], evaluator=evaluator)
        fitted = method.prepare(None, tol=1e-6)  # type: ignore[arg-type]
        fitted(np.array([0.5]), tol=1e-3)  # call-time override

        assert received.get("tol") == pytest.approx(1e-3)

    def test_prepare_without_options_works_as_before(self) -> None:
        """
        Calling ``prepare()`` with no options must still produce a working
        wrapper that forwards only call-time kwargs (backward-compatible).
        """
        from pysatl_core.distributions.computations.computation import EvaluatorMethod

        received: dict[str, Any] = {}

        def evaluator(_distribution: Any, x: NumericArray, /, **kwargs: Any) -> NumericArray:
            received.update(kwargs)
            return np.asarray(x, dtype=float)

        method = EvaluatorMethod(target="cdf", sources=["pdf"], evaluator=evaluator)
        fitted = method.prepare(None)  # type: ignore[arg-type]
        fitted(np.array([0.5]), tol=1e-9)

        assert received.get("tol") == pytest.approx(1e-9)
