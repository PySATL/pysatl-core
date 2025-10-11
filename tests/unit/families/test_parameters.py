from __future__ import annotations

import pytest

from pysatl_core.families import (
    ParametricFamily,
    Parametrization,
    ParametrizationConstraint,
    constraint,
    parametrization,
)
from pysatl_core.types import UnivariateContinuous
from tests.unit.families.test_basic import TestBaseFamily


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
        def check_positive(self) -> bool:  # noqa: ANN001 (test signature)
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
            sampling_strategy=lambda n, d, **_: __import__("numpy").random.random((n, 1)),  # type: ignore[assignment]
            computation_strategy=lambda: None,  # type: ignore[assignment]
        )

        @parametrization(family=family, name="base")
        class Base(Parametrization):
            value: float

            @constraint("Value must be positive")
            def check_positive(self) -> bool:
                return self.value > 0

        instance = Base(value=5.0)  # type: ignore[call-arg]
        assert instance.name == "base"
        assert instance.parameters == {"value": 5.0}
        assert getattr(Base, "__family__", None) is family
        assert getattr(Base, "__param_name__", None) == "base"
        assert hasattr(Base, "__dataclass_fields__")

        # Validation succeeds
        instance.validate()

        # Validation fails
        invalid = Base(value=-1.0)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="Constraint.*does not hold"):
            invalid.validate()

    def test_method_style_parametrization_decorator(self) -> None:
        """family.parametrization(name=...) should behave the same as the free decorator."""
        family = ParametricFamily(
            name="MethodDecoratorFamily",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["kind"],
            distr_characteristics={},
            sampling_strategy=lambda n, d, **_: __import__("numpy").random.random((n, 1)),  # type: ignore[assignment]
            computation_strategy=lambda: None,  # type: ignore[assignment]
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
        assert family.get_base_parameters(base_params) is base_params

        alt_params = AltCls(value=3.0)  # type: ignore[call-arg]
        base_from_alt = family.get_base_parameters(alt_params)
        assert isinstance(base_from_alt, BaseCls)
        # Our default factory maps Alt(value=v) → Base(value=v)
        assert base_from_alt.value == 3.0
