from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_core.families import (
    ParametricFamily,
    Parametrization,
    ParametrizationConstraint,
    constraint,
)
from pysatl_core.types import UnivariateContinuous
from tests.unit.families.test_basic import TestBaseFamily
from tests.utils.mocks import MockSamplingStrategy


class TestParametrizationAPI(TestBaseFamily):
    # ---------- Constraint basics ----------

    def test_constraint_is_a_simple_holder(self) -> None:
        """Constraint holds a description and a predicate function."""

        def is_positive(obj: object) -> bool:
            return getattr(obj, "value", 0) > 0

        c = ParametrizationConstraint(description="Value must be positive", check=is_positive)
        assert c.description == "Value must be positive"
        assert c.check is is_positive

    def test_constraint_decorator_marks_function(self) -> None:
        """Decorator should tag a function so Parametrization.validate() can discover it."""

        @constraint("Value must be positive")
        def check_positive(self: Any) -> bool:  # noqa: ANN001 (test signature)
            return getattr(self, "value", 0) > 0

        # Different code versions may use single or double underscore attributes.
        is_flag = getattr(check_positive, "_is_constraint", None) or getattr(
            check_positive, "__is_constraint", None
        )
        desc = getattr(check_positive, "_constraint_description", None) or getattr(
            check_positive, "__constraint_description", None
        )
        assert bool(is_flag) is True
        assert desc == "Value must be positive"

    # ---------- Parametrization via decorators ----------

    def test_free_function_parametrization_decorator(self) -> None:
        """@parametrization(family=..., name=...) should dataclass-ify and register the class."""
        family = ParametricFamily(
            name="FreeDecoratorFamily",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["base"],
            distr_characteristics={},
            sampling_strategy=MockSamplingStrategy(),
        )

        @family.parametrization(name="kind")
        class Kind(Parametrization):
            value: float

        obj = Kind(value=1.25)  # type: ignore[call-arg]
        assert obj.name == "kind"
        assert obj.parameters == {"value": 1.25}
        assert getattr(Kind, "__family__", None) is family
        assert getattr(Kind, "__param_name__", None) == "kind"
        assert hasattr(Kind, "__dataclass_fields__")

    # ---------- Family-level conversion to base ----------

    def test_get_base_parameters_uses_family_logic(self) -> None:
        """Defines Base/Alt so that Alt transforms to Base with the same value."""
        # Use shared test factory so we stay consistent with the rest of the suite.
        family = self.make_default_family()

        BaseCls = family.parametrizations["base"]
        AltCls = family.parametrizations["alt"]

        base_params = BaseCls(value=5.0)  # type: ignore[call-arg]
        assert family.to_base(base_params) is base_params

        alt_params = AltCls(value=3.0)  # type: ignore[call-arg]
        base_from_alt = family.to_base(alt_params)
        assert isinstance(base_from_alt, BaseCls)
        # Our default factory maps Alt(value=v) → Base(value=v)
        assert base_from_alt.value == 3.0  # type: ignore[attr-defined]
