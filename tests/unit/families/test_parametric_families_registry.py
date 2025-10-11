import pytest

from pysatl_core.families import ParametricFamilyRegister


class TestParametricFamiliesRegister:
    """Test the ParametricFamiliesRegister singleton."""

    def test_singleton_pattern(self):
        """Test that only one instance exists."""
        register1 = ParametricFamilyRegister()
        register2 = ParametricFamilyRegister()
        assert register1 is register2

    def test_register_and_get_family(self):
        """Test registering and retrieving a family."""
        register = ParametricFamilyRegister()

        # Create a mock family
        mock_family = type("MockFamily", (), {"name": "TestFamily"})()

        # Register and retrieve
        register.register(mock_family)
        retrieved = register.get("TestFamily")

        assert retrieved is mock_family

    def test_get_nonexistent_family(self):
        """Test error when getting a non-existent family."""
        register = ParametricFamilyRegister()

        with pytest.raises(ValueError, match="No family Nonexistent found in register"):
            register.get("Nonexistent")
