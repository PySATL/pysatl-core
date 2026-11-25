from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_core.arrays import ArrayBackend, ArrayContext, Backend
from pysatl_core.distributions import requires_array_operations, strategies as strategies_module
from pysatl_core.types import CharacteristicName
from tests.unit.families.test_basic import TestBaseFamily


class TestAnalyticalPlan(TestBaseFamily):
    def test_non_registry_method_selects_supported_variant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @requires_array_operations("unsupported_operation")
        def unsupported(_parameters: object) -> float:
            raise AssertionError("The unsupported variant must not be used.")

        def supported(_parameters: object) -> float:
            return 2.0

        family = self.make_default_family(
            distr_characteristics={
                CharacteristicName.MEAN: {"base": {"first": unsupported, "second": supported}}
            }
        )
        distribution = family.distribution("base", value=0.0)
        context = ArrayContext(Backend("custom", ArrayBackend("custom")))
        monkeypatch.setattr(strategies_module, "resolve_array_context", lambda: context)

        assert distribution.query_method(CharacteristicName.MEAN)() == 2.0

    def test_family_analytical_plan_picks_provider_correctly(self) -> None:
        fam = self.make_default_family()

        plan = fam._analytical_plan
        assert set(plan.keys()) == {"base", "alt"}

        # For 'alt': CDF provided by 'alt'; PDF/PPF fallback to base (PPF has only base)
        assert plan["alt"][CharacteristicName.CDF] == "alt"
        assert plan["alt"][CharacteristicName.PDF] == "base"
        assert plan["alt"][CharacteristicName.PPF] == "base"

        # For 'base': all come from base
        assert plan["base"][CharacteristicName.PDF] == "base"
        assert plan["base"][CharacteristicName.CDF] == "base"
        assert plan["base"][CharacteristicName.PPF] == "base"

    def test_family_propagates_array_operation_requirements(self) -> None:
        @requires_array_operations("exp", "multiply")
        def pdf(_parameters: object, value: object) -> object:
            return value

        family = self.make_default_family(
            distr_characteristics={
                CharacteristicName.PDF: {"base": {"default": pdf}},
            }
        )
        distribution = family.distribution("base", value=0.0)

        assert distribution.analytical_computations[CharacteristicName.PDF][
            "default"
        ].required_operations == frozenset(
            {
                "exp",
                "multiply",
            }
        )

    @pytest.mark.parametrize("parametrization", ["base", "alt"])
    def test_view_preserves_requirements_and_selects_supported_variant(
        self, monkeypatch: pytest.MonkeyPatch, parametrization: str
    ) -> None:
        @requires_array_operations("unsupported_operation")
        def unsupported() -> float:
            raise AssertionError("The unsupported provider must not be selected.")

        def supported() -> float:
            return 2.0

        family = self.make_default_family(
            distr_characteristics={
                CharacteristicName.MEAN: {"base": {"first": unsupported, "second": supported}}
            }
        )
        distribution = family.view(parametrization_name=parametrization).distribution(value=0.0)
        context = ArrayContext(Backend("custom", ArrayBackend("custom")))
        monkeypatch.setattr(strategies_module, "resolve_array_context", lambda: context)

        assert distribution.analytical_computations[CharacteristicName.MEAN][
            "first"
        ].required_operations == frozenset({"unsupported_operation"})
        assert distribution.query_method(CharacteristicName.MEAN)() == 2.0

    def test_non_registry_method_requires_declared_operations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        @requires_array_operations("exp")
        def mean(_parameters: object) -> float:
            return 0.0

        family = self.make_default_family(
            distr_characteristics={
                CharacteristicName.MEAN: {"base": {"default": mean}},
            }
        )
        distribution = family.distribution("base", value=0.0)
        monkeypatch.setattr(
            strategies_module,
            "resolve_array_context",
            lambda: ArrayContext(Backend("custom", ArrayBackend("custom"))),
        )

        with pytest.raises(RuntimeError, match=r"missing array operations \['exp'\]"):
            distribution.query_method(CharacteristicName.MEAN)
