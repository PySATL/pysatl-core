"""
Tests for Uniform Distribution Family

This module tests the functionality of the uniform distribution family,
including parameterizations, characteristics, and sampling.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from scipy.stats import uniform

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.types import (
    CharacteristicName,
    ContinuousSupportShape1D,
    FamilyName,
    UnivariateContinuous,
)

from .base import BaseDistributionTest


class TestUniformFamily(BaseDistributionTest):
    """Test suite for Uniform distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.uniform_family = registry.get(FamilyName.CONTINUOUS_UNIFORM)
        self.uniform_dist_example = self.uniform_family(lower_bound=2.0, upper_bound=5.0)

    def test_family_properties(self):
        """Test basic properties of uniform family."""
        assert self.uniform_family.name == FamilyName.CONTINUOUS_UNIFORM

        # Check parameterizations
        expected_parametrizations = {"standard", "meanWidth", "minRange"}
        assert set(self.uniform_family.parametrization_names) == expected_parametrizations
        assert self.uniform_family.base_parametrization_name == "standard"

    def test_standard_parametrization_creation(self):
        """Test creation of distribution with standard parametrization."""
        dist = self.uniform_family(lower_bound=2.0, upper_bound=5.0)

        assert dist.family_name == FamilyName.CONTINUOUS_UNIFORM
        assert dist.distribution_type == UnivariateContinuous
        assert dist.parameters == {"lower_bound": 2.0, "upper_bound": 5.0}
        assert dist.parametrization_name == "standard"

    def test_mean_width_parametrization_creation(self):
        """Test creation of distribution with mean-width parametrization."""
        dist = self.uniform_family(mean=3.5, width=3.0, parametrization_name="meanWidth")

        assert dist.parameters == {"mean": 3.5, "width": 3.0}
        assert dist.parametrization_name == "meanWidth"

    def test_min_range_parametrization_creation(self):
        """Test creation of distribution with min-range parametrization."""
        dist = self.uniform_family(minimum=2.0, range_val=3.0, parametrization_name="minRange")

        assert dist.parameters == {"minimum": 2.0, "range_val": 3.0}
        assert dist.parametrization_name == "minRange"

    def test_parametrization_constraints(self):
        """Test parameter constraints validation."""
        # lower_bound must be less than upper_bound
        with pytest.raises(ValueError, match="lower_bound < upper_bound"):
            self.uniform_family(lower_bound=5.0, upper_bound=2.0)

        # width must be positive
        with pytest.raises(ValueError, match="width > 0"):
            self.uniform_family(mean=3.5, width=0.0, parametrization_name="meanWidth")

        # range_val must be positive
        with pytest.raises(ValueError, match="range_val > 0"):
            self.uniform_family(minimum=2.0, range_val=0.0, parametrization_name="minRange")

    def test_characteristic_function_at_zero(self):
        """Test characteristic function at zero returns 1."""
        char_func = self.uniform_dist_example.query_method(CharacteristicName.CF)

        cf_value_zero = char_func(0.0)
        assert abs(cf_value_zero.real - 1.0) < self.CALCULATION_PRECISION
        assert abs(cf_value_zero.imag) < self.CALCULATION_PRECISION

    def test_moments(self):
        """Test moment calculations."""
        # Mean
        mean_func = self.uniform_dist_example.query_method(CharacteristicName.MEAN)
        assert abs(mean_func(None) - 3.5) < self.CALCULATION_PRECISION

        # Variance
        var_func = self.uniform_dist_example.query_method(CharacteristicName.VAR)
        assert abs(var_func(None) - 0.75) < self.CALCULATION_PRECISION

        # Skewness
        skew_func = self.uniform_dist_example.query_method(CharacteristicName.SKEW)
        assert abs(skew_func(None) - 0.0) < self.CALCULATION_PRECISION

    def test_kurtosis_calculation(self):
        """Test kurtosis calculation with excess parameter."""
        kurt_func = self.uniform_dist_example.query_method(CharacteristicName.KURT)

        raw_kurt = kurt_func(None)
        assert abs(raw_kurt - 1.8) < self.CALCULATION_PRECISION

        excess_kurt = kurt_func(None, excess=True)
        assert abs(excess_kurt + 1.2) < self.CALCULATION_PRECISION

        raw_kurt_explicit = kurt_func(None, excess=False)
        assert abs(raw_kurt_explicit - 1.8) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "parametrization_name, params, expected_lower, expected_upper",
        [
            ("standard", {"lower_bound": 2.0, "upper_bound": 5.0}, 2.0, 5.0),
            ("meanWidth", {"mean": 3.5, "width": 3.0}, 2.0, 5.0),
            ("minRange", {"minimum": 2.0, "range_val": 3.0}, 2.0, 5.0),
        ],
    )
    def test_parametrization_conversions(
        self, parametrization_name, params, expected_lower, expected_upper
    ):
        """Test conversions between different parameterizations."""
        base_params = self.uniform_family.to_base(
            self.uniform_family.get_parametrization(parametrization_name)(**params)
        )

        assert (
            abs(base_params.parameters["lower_bound"] - expected_lower) < self.CALCULATION_PRECISION
        )
        assert (
            abs(base_params.parameters["upper_bound"] - expected_upper) < self.CALCULATION_PRECISION
        )

    def test_analytical_computations_availability(self):
        """Test that analytical computations are available for exponential distribution."""
        comp = self.uniform_family(lower_bound=0.0, upper_bound=1.0).analytical_computations

        expected_chars = {
            CharacteristicName.PDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
            CharacteristicName.CF,
            CharacteristicName.MEAN,
            CharacteristicName.VAR,
            CharacteristicName.SKEW,
            CharacteristicName.KURT,
        }
        assert set(comp.keys()) == expected_chars

    @pytest.mark.parametrize(
        "char_name, test_data, scipy_func, scipy_kwargs",
        [
            (
                CharacteristicName.PDF,
                [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
                uniform.pdf,
                {"loc": 2.0, "scale": 3.0},
            ),
            (
                CharacteristicName.CDF,
                [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
                uniform.cdf,
                {"loc": 2.0, "scale": 3.0},
            ),
            (
                CharacteristicName.PPF,
                [0.0, 0.25, 0.5, 0.75, 1.0],
                uniform.ppf,
                {"loc": 2.0, "scale": 3.0},
            ),
        ],
    )
    def test_array_input_for_characteristics(self, char_name, test_data, scipy_func, scipy_kwargs):
        """Test that characteristics support array inputs."""
        dist = self.uniform_dist_example
        char_func = dist.query_method(char_name)

        input_array = np.array(test_data)
        result_array = char_func(input_array)

        assert result_array.shape == input_array.shape

        expected_array = scipy_func(input_array, **scipy_kwargs)

        self.assert_arrays_almost_equal(result_array, expected_array)

    def test_characteristic_function_array_input(self):
        """Test characteristic function calculation with array input."""
        char_func = self.uniform_dist_example.query_method(CharacteristicName.CF)
        t_array = np.array([-2.0, -1.0, 1.0, 2.0, 0.0])

        cf_array = char_func(t_array)
        assert cf_array.shape == t_array.shape

        a, b = 2.0, 5.0
        width = b - a
        center = (a + b) / 2

        x = width * t_array / (2 * np.pi)
        expected = np.sinc(x) * np.exp(1j * center * t_array)

        self.assert_arrays_almost_equal(cf_array.real, expected.real)
        self.assert_arrays_almost_equal(cf_array.imag, expected.imag)

    def test_uniform_support(self):
        """Test that uniform distribution has correct support [lower_bound, upper_bound]."""
        dist = self.uniform_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)

        assert dist.support.left == 2.0
        assert dist.support.right == 5.0
        assert dist.support.left_closed
        assert dist.support.right_closed

        # Test containment
        assert dist.support.contains(2.0) is True
        assert dist.support.contains(5.0) is True
        assert dist.support.contains(3.5) is True
        assert dist.support.contains(1.9) is False
        assert dist.support.contains(5.1) is False

        # Test array
        test_points = np.array([1.9, 2.0, 3.5, 5.0, 5.1])
        expected = np.array([False, True, True, True, False])
        results = dist.support.contains(test_points)
        np.testing.assert_array_equal(results, expected)

        assert dist.support.shape == ContinuousSupportShape1D.BOUNDED_INTERVAL


class TestUniformFamilyEdgeCases(BaseDistributionTest):
    """Test edge cases and error conditions for uniform distribution."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.uniform_family = registry.get(FamilyName.CONTINUOUS_UNIFORM)

    def test_invalid_parameterization(self):
        """Test error for invalid parameterization name."""
        with pytest.raises(KeyError):
            self.uniform_family.distribution(
                parametrization_name="invalid_name", lower_bound=0.0, upper_bound=1.0
            )

    def test_missing_parameters(self):
        """Test error for missing required parameters."""
        with pytest.raises(TypeError):
            self.uniform_family.distribution(lower_bound=0.0)  # Missing upper_bound

        with pytest.raises(TypeError):
            self.uniform_family.distribution(upper_bound=1.0)  # Missing lower_bound

    def test_invalid_probability_ppf(self):
        """Test PPF with invalid probability values."""
        dist = self.uniform_family(lower_bound=0.0, upper_bound=1.0)
        ppf = dist.query_method(CharacteristicName.PPF)

        # Test boundaries
        assert ppf(0.0) == 0.0
        assert ppf(1.0) == 1.0

        # Test invalid probabilities
        with pytest.raises(ValueError):
            ppf(-0.1)
        with pytest.raises(ValueError):
            ppf(1.1)

    def test_single_value_uniform(self):
        """Test uniform distribution with single value (lower_bound == upper_bound)."""
        with pytest.raises(ValueError, match="lower_bound < upper_bound"):
            self.uniform_family(lower_bound=2.0, upper_bound=2.0)

    def test_characteristic_function_edge_cases(self):
        """Test characteristic function at edge cases."""
        dist = self.uniform_family(lower_bound=0.0, upper_bound=1.0)
        char_func = dist.query_method(CharacteristicName.CF)

        # Test with very small t
        cf_value_small = char_func(self.CALCULATION_PRECISION)
        assert abs(cf_value_small.real - 1.0) < self.CALCULATION_PRECISION

        # Test with large t
        cf_value_large = char_func(1000.0)
        assert isinstance(cf_value_large, complex)
        assert abs(cf_value_large) <= 1.0

    def test_negative_width(self):
        """Test that negative width is rejected."""
        with pytest.raises(ValueError, match="width > 0"):
            self.uniform_family(mean=0.0, width=-1.0, parametrization_name="meanWidth")

    def test_negative_range(self):
        """Test that negative range is rejected."""
        with pytest.raises(ValueError, match="range_val > 0"):
            self.uniform_family(minimum=0.0, range_val=-1.0, parametrization_name="minRange")
