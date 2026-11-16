from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Hashable

import pytest

from pysatl_core.arrays import ArrayBackend, ArrayContext, Backend
from pysatl_core.distributions import requires_array_operations, strategies as strategies_module
from pysatl_core.types import CharacteristicName
from tests.unit.families.test_basic import TestBaseFamily


class TestAnalyticalPlan(TestBaseFamily):
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
        ].required_operations == frozenset({"exp", "multiply"})

    def test_non_registry_method_requires_declared_operations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def normalize_dtype(value: object) -> Hashable:
            if not isinstance(value, Hashable):
                raise TypeError
            return value

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
            lambda: ArrayContext(
                Backend("custom", ArrayBackend("custom", dtype_normalizer=normalize_dtype))
            ),
        )

        with pytest.raises(RuntimeError, match=r"missing array operations \['exp'\]"):
            distribution.query_method(CharacteristicName.MEAN)
