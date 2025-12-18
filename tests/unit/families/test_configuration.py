"""
Tests for Distribution Families Configuration

This module tests the configuration and registration of distribution families
in the global ParametricFamilyRegister.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_core.families.configuration import (
    configure_families_register,
    reset_families_register,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import FamilyName


class TestConfiguration:
    """Test suite for configuration functionality."""

    def setup_method(self):
        """Setup before each test method."""
        self.registry = configure_families_register()

    def test_configure_families_register_returns_registry(self):
        """Test that configure_families_register returns a ParametricFamilyRegister."""
        assert isinstance(self.registry, ParametricFamilyRegister)

    def test_configure_families_register_is_singleton(self):
        """Test that configure_families_register returns the same instance."""
        registry2 = configure_families_register()
        assert self.registry is registry2

    def test_families_registered(self):
        """Test that all expected families are registered."""
        expected_families = {
            FamilyName.NORMAL,
            FamilyName.CONTINUOUS_UNIFORM,
        }

        registered_families = set(self.registry._registered_families.keys())
        assert expected_families.issubset(registered_families)

    def test_reset_families_register(self):
        """Test that reset_families_register clears the cache."""
        registry1 = configure_families_register()
        reset_families_register()
        registry2 = configure_families_register()

        # They should be different instances after reset
        assert registry1 is not registry2

    def test_registry_singleton_pattern(self):
        """Test that ParametricFamilyRegister itself follows singleton pattern."""
        registry1 = ParametricFamilyRegister()
        registry2 = ParametricFamilyRegister()
        assert registry1 is registry2

    def test_registry_get_family_method(self):
        """Test the get method of ParametricFamilyRegister."""
        # Test getting existing family
        normal_family = self.registry.get(FamilyName.NORMAL)
        assert normal_family is not None
        assert normal_family.name == FamilyName.NORMAL

        # Test getting non-existent family
        with pytest.raises(ValueError):
            self.registry.get("NonExistentFamily")

    def test_registry_list_registered_families(self):
        """Test the list_registered_families method of ParametricFamilyRegister."""
        families_list = ParametricFamilyRegister.list_registered_families()

        assert isinstance(families_list, list)
        assert FamilyName.NORMAL in families_list
        assert FamilyName.CONTINUOUS_UNIFORM in families_list
        assert "NonExistentFamily" not in families_list
