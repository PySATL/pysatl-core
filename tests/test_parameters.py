import pytest

from pysatl_core.families import (
    ParametricFamily,
    Parametrization,
    ParametrizationConstraint,
    ParametrizationSpec,
    constraint,
    parametrization,
)


class TestParametrizationConstraint:
    """Test the ParametrizationConstraint class."""

    def test_constraint_creation(self):
        """Test creating a constraint with description and check function."""

        def check_func(obj):
            return obj.value > 0

        constraint = ParametrizationConstraint(
            description="Value must be positive", check=check_func
        )

        assert constraint.description == "Value must be positive"
        assert constraint.check is check_func


class TestParametrization:
    """Test the base Parametrization class."""

    def test_abstract_methods(self):
        """Test that Parametrization is abstract and requires name and parameters properties."""
        with pytest.raises(TypeError):
            Parametrization()  # Can't instantiate abstract class

        # Create a concrete implementation
        class ConcreteParametrization(Parametrization):
            @property
            def name(self):
                return "concrete"

            @property
            def parameters(self):
                return {"param": 1.0}

        # Should be able to instantiate
        param = ConcreteParametrization()
        assert param.name == "concrete"
        assert param.parameters == {"param": 1.0}


class TestParametrizationSpec:
    """Test the ParametrizationSpec class."""

    def test_add_and_get_parametrization(self):
        """Test adding and retrieving parametrizations."""
        spec = ParametrizationSpec()

        # Create a mock parametrization class
        class MockParametrization(Parametrization):
            @property
            def name(self):
                return "mock"

            @property
            def parameters(self):
                return {}

        # Add parametrization
        spec.add_parametrization("mock", MockParametrization, is_base=True)

        # Check it was added
        assert "mock" in spec.parametrizations
        assert spec.parametrizations["mock"] is MockParametrization
        assert spec.base_parametrization_name == "mock"
        assert spec.base is MockParametrization

    def test_get_base_parameters(self):
        """Test converting parameters to base parametrization."""
        spec = ParametrizationSpec()

        # Create mock parametrizations
        class BaseParametrization(Parametrization):
            def __init__(self, value):
                self.value = value

            @property
            def name(self):
                return "base"

            @property
            def parameters(self):
                return {"value": self.value}

        class OtherParametrization(Parametrization):
            def __init__(self, other_value):
                self.other_value = other_value

            @property
            def name(self):
                return "other"

            @property
            def parameters(self):
                return {"other_value": self.other_value}

            def transform_to_base_parametrization(self):
                return BaseParametrization(self.other_value * 2)

        # Add parametrizations
        spec.add_parametrization("base", BaseParametrization, is_base=True)
        spec.add_parametrization("other", OtherParametrization)

        # Test with base parametrization
        base_params = BaseParametrization(5.0)
        result = spec.get_base_parameters(base_params)
        assert result is base_params

        # Test with other parametrization
        other_params = OtherParametrization(3.0)
        result = spec.get_base_parameters(other_params)
        assert isinstance(result, BaseParametrization)
        assert result.value == 6.0  # 3.0 * 2


class TestDecorators:
    """Test the constraint and parametrization decorators."""

    def test_constraint_decorator(self):
        """Test the constraint decorator."""

        @constraint("Value must be positive")
        def check_positive(self):
            return self.value > 0

        # Check that the decorator added the required attributes
        assert hasattr(check_positive, "_is_constraint")
        assert hasattr(check_positive, "_constraint_description")
        assert check_positive._is_constraint is True
        assert check_positive._constraint_description == "Value must be positive"

    def test_parametrization_decorator(self):
        """Test the parametrization decorator."""
        # Create a mock family
        family = ParametricFamily(
            name="TestFamily",
            distr_type="Continuous",
            distr_parametrizations=["test"],
            distr_characteristics={},
            sampling_strategy=None,
            computation_strategy=None,
        )

        # Apply the decorator
        @parametrization(family=family, name="test")
        class TestParametrization:
            value: float

            @constraint("Value must be positive")
            def check_positive(self):
                return self.value > 0

        # Check that the class was modified
        assert hasattr(TestParametrization, "name")
        assert hasattr(TestParametrization, "parameters")
        assert hasattr(TestParametrization, "_constraints")
        assert hasattr(TestParametrization, "validate")

        # Check that it was registered with the family
        assert "test" in family.parametrizations.parametrizations
        assert family.parametrizations.parametrizations["test"] is TestParametrization
        assert family.parametrizations.base_parametrization_name == "test"

        # Test instantiation and validation
        instance = TestParametrization(value=5.0)
        assert instance.name == "test"
        assert instance.parameters == {"value": 5.0}

        # This should not raise an exception
        instance.validate()

        # Test with invalid parameters
        invalid_instance = TestParametrization(value=-1.0)
        with pytest.raises(ValueError, match="Constraint.*does not hold"):
            invalid_instance.validate()
