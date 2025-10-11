from __future__ import annotations

from tests.unit.families.test_basic import TestBaseFamily


class TestAnalyticalPlan(TestBaseFamily):
    def test_family_analytical_plan_picks_provider_correctly(self) -> None:
        fam = self.make_default_family()

        plan = fam._analytical_plan
        assert set(plan.keys()) == {"base", "alt"}

        # For 'alt': CDF provided by 'alt'; PDF/PPF fallback to base (PPF has only base)
        assert plan["alt"][self.CDF] == "alt"
        assert plan["alt"][self.PDF] == "base"
        assert plan["alt"][self.PPF] == "base"

        # For 'base': all come from base
        assert plan["base"][self.PDF] == "base"
        assert plan["base"][self.CDF] == "base"
        assert plan["base"][self.PPF] == "base"
