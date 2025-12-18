"""
Tests for Exponential Distribution Family

This module tests the functionality of the exponential distribution family,
including parameterizations, characteristics, and sampling.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from scipy.stats import expon

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.types import (
    CharacteristicName,
    ContinuousSupportShape1D,
    FamilyName,
    UnivariateContinuous,
)

from .base import BaseDistributionTest


class TestExponentialFamily(BaseDistributionTest):
    """Test suite for Exponential distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.exponential_family = registry.get(FamilyName.EXPONENTIAL)
        self.exponential_dist_example = self.exponential_family(lambda_=0.5)

    def test_family_properties(self):
        """Test basic properties of exponential family."""
        assert self.exponential_family.name == FamilyName.EXPONENTIAL

        # Check parameterizations
        expected_parametrizations = {"rate", "scale"}
        assert set(self.exponential_family.parametrization_names) == expected_parametrizations
        assert self.exponential_family.base_parametrization_name == "rate"

    def test_rate_parametrization_creation(self):
        """Test creation of distribution with rate parametrization."""
        dist = self.exponential_family(lambda_=0.5)

        assert dist.family_name == FamilyName.EXPONENTIAL
        assert dist.distribution_type == UnivariateContinuous
        assert dist.parameters == {"lambda_": 0.5}
        assert dist.parametrization_name == "rate"

    def test_scale_parametrization_creation(self):
        """Test creation of distribution with scale parametrization."""
        dist = self.exponential_family(beta=2.0, parametrization_name="scale")

        assert dist.parameters == {"beta": 2.0}
        assert dist.parametrization_name == "scale"

    def test_parametrization_constraints(self):
        """Test parameter constraints validation."""
        # lambda_ must be positive
        with pytest.raises(ValueError, match="lambda_ > 0"):
            self.exponential_family(lambda_=-1.0)

        # beta must be positive
        with pytest.raises(ValueError, match="beta > 0"):
            self.exponential_family(beta=0.0, parametrization_name="scale")

    def test_moments(self):
        """Test moment calculations."""
        # Mean
        mean_func = self.exponential_dist_example.query_method(CharacteristicName.MEAN)
        assert abs(mean_func(None) - 2.0) < self.CALCULATION_PRECISION

        # Variance
        var_func = self.exponential_dist_example.query_method(CharacteristicName.VAR)
        assert abs(var_func(None) - 4.0) < self.CALCULATION_PRECISION

        # Skewness
        skew_func = self.exponential_dist_example.query_method(CharacteristicName.SKEW)
        assert abs(skew_func(None) - 2.0) < self.CALCULATION_PRECISION

    def test_kurtosis_calculation(self):
        """Test kurtosis calculation with excess parameter."""
        kurt_func = self.exponential_dist_example.query_method(CharacteristicName.KURT)

        raw_kurt = kurt_func(None)
        assert abs(raw_kurt - 9.0) < self.CALCULATION_PRECISION

        excess_kurt = kurt_func(None, excess=True)
        assert abs(excess_kurt - 6.0) < self.CALCULATION_PRECISION

        raw_kurt_explicit = kurt_func(None, excess=False)
        assert abs(raw_kurt_explicit - 9.0) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "parametrization_name, params, expected_lambda",
        [
            ("rate", {"lambda_": 0.5}, 0.5),
            ("scale", {"beta": 2.0}, 0.5),  # lambda = 1/beta = 0.5
        ],
    )
    def test_parametrization_conversions(self, parametrization_name, params, expected_lambda):
        """Test conversions between different parameterizations."""
        base_params = self.exponential_family.to_base(
            self.exponential_family.get_parametrization(parametrization_name)(**params)
        )

        assert abs(base_params.parameters["lambda_"] - expected_lambda) < self.CALCULATION_PRECISION

    def test_analytical_computations_availability(self):
        """Test that analytical computations are available for exponential distribution."""
        comp = self.exponential_family(lambda_=1.0).analytical_computations

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
            (CharacteristicName.PDF, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], expon.pdf, {"scale": 2.0}),
            (CharacteristicName.CDF, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], expon.cdf, {"scale": 2.0}),
            (
                CharacteristicName.PPF,
                [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999],
                expon.ppf,
                {"scale": 2.0},
            ),
        ],
    )
    def test_array_input_for_characteristics(self, char_name, test_data, scipy_func, scipy_kwargs):
        """Test that characteristics support array inputs."""
        dist = self.exponential_dist_example
        char_func = dist.query_method(char_name)

        input_array = np.array(test_data)
        result_array = char_func(input_array)

        assert result_array.shape == input_array.shape

        expected_array = scipy_func(input_array, **scipy_kwargs)

        self.assert_arrays_almost_equal(result_array, expected_array)

    def test_characteristic_function_array_input(self):
        """Test characteristic function calculation with array input."""
        char_func = self.exponential_dist_example.query_method(CharacteristicName.CF)
        t_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        cf_array = char_func(t_array)
        assert cf_array.shape == t_array.shape

        lambda_ = 0.5
        denominator = lambda_**2 + t_array**2
        expected_real = lambda_**2 / denominator
        expected_imag = lambda_ * t_array / denominator

        expected_real = np.where(np.abs(t_array) < self.CALCULATION_PRECISION, 1.0, expected_real)
        expected_imag = np.where(np.abs(t_array) < self.CALCULATION_PRECISION, 0.0, expected_imag)

        expected = expected_real + 1j * expected_imag

        self.assert_arrays_almost_equal(cf_array.real, expected.real)
        self.assert_arrays_almost_equal(cf_array.imag, expected.imag)

    def test_exponential_support(self):
        """Test that exponential distribution has correct support [0, ∞)."""
        dist = self.exponential_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)

        assert dist.support.left == 0.0
        assert dist.support.right == float("inf")
        assert dist.support.left_closed
        assert not dist.support.right_closed

        # Test containment
        assert dist.support.contains(0.0) is True
        assert dist.support.contains(1.0) is True
        assert dist.support.contains(-0.1) is False
        assert dist.support.contains(float("inf")) is False

        # Test array
        test_points = np.array([-0.1, 0.0, 1.0, 10.0])
        expected = np.array([False, True, True, True])
        results = dist.support.contains(test_points)
        np.testing.assert_array_equal(results, expected)

        assert dist.support.shape == ContinuousSupportShape1D.RAY_RIGHT


class TestExponentialFamilyEdgeCases(BaseDistributionTest):
    """Test edge cases and error conditions for exponential distribution."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.exponential_family = registry.get(FamilyName.EXPONENTIAL)

    def test_invalid_parameterization(self):
        """Test error for invalid parameterization name."""
        with pytest.raises(KeyError):
            self.exponential_family.distribution(parametrization_name="invalid_name", lambda_=1.0)

    def test_missing_parameters(self):
        """Test error for missing required parameters."""
        with pytest.raises(TypeError):
            self.exponential_family.distribution()  # Missing lambda_

    def test_invalid_probability_ppf(self):
        """Test PPF with invalid probability values."""
        dist = self.exponential_family(lambda_=1.0)
        ppf = dist.query_method(CharacteristicName.PPF)

        # Test boundaries
        assert ppf(0.0) == 0.0
        assert ppf(1.0) == float("inf")

        # Test invalid probabilities
        with pytest.raises(ValueError):
            ppf(-0.1)
        with pytest.raises(ValueError):
            ppf(1.1)

    def test_characteristic_function_at_zero(self):
        """Test characteristic function at zero returns 1."""
        dist = self.exponential_family(lambda_=1.0)
        char_func = dist.query_method(CharacteristicName.CF)

        cf_value_zero = char_func(0.0)
        assert abs(cf_value_zero.real - 1.0) < self.CALCULATION_PRECISION
        assert abs(cf_value_zero.imag) < self.CALCULATION_PRECISION

    def test_characteristic_function_large_t(self):
        """Test characteristic function with large t."""
        dist = self.exponential_family(lambda_=1.0)
        char_func = dist.query_method(CharacteristicName.CF)

        cf_value_large = char_func(1000.0)
        assert np.iscomplexobj(cf_value_large)
        assert abs(cf_value_large) <= 1.0
