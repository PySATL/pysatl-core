from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

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
