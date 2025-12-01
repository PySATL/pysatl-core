"""
Tests for Normal Distribution Family Configuration

This module tests the functionality of the normal distribution family
defined in config.py, including parameterizations, characteristics,
and sampling.
"""

import math
from typing import cast

import numpy as np
import pytest
from scipy.stats import norm

from pysatl_core.distributions.characteristics import GenericCharacteristic
from pysatl_core.families import ParametricFamilyRegister, configure_family_register
from pysatl_core.families.config import (
    ExpParametrization,
    MeanPrecParametrization,
    MeanVarParametrization,
)

# Import PySATL components
from pysatl_core.families.registry import _reset_families_register_for_tests
from pysatl_core.types import UnivariateContinuous

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

class TestNormalFamily:
    """Test suite for Normal distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        _reset_families_register_for_tests()
        configure_family_register()
        self.normal_family = ParametricFamilyRegister.get("Normal Family")

    def test_family_registration(self):
        """Test that normal family is properly registered."""
        family = ParametricFamilyRegister.get("Normal Family")
        assert family.name == "Normal Family"

        # Check parameterizations
        expected_parametrizations = {"meanVar", "meanPrec", "exponential"}
        assert set(family.parametrization_names) == expected_parametrizations
        assert family.base_parametrization_name == "meanVar"

    def test_mean_var_parametrization_creation(self):
        """Test creation of distribution with mean-variance parametrization."""
        dist = self.normal_family(mu=2.0, sigma=1.5)

        assert dist.distr_name == "Normal Family"
        assert dist.distribution_type == UnivariateContinuous

        # Приводим к конкретному типу параметризации
        params = cast(MeanVarParametrization, dist.parameters)
        assert params.mu == 2.0
        assert params.sigma == 1.5
        assert params.name == "meanVar"

    def test_mean_prec_parametrization_creation(self):
        """Test creation of distribution with mean-precision parametrization."""
        dist = self.normal_family(mu=2.0, tau=0.25, parametrization_name="meanPrec")

        # Приводим к конкретному типу параметризации
        params = cast(MeanPrecParametrization, dist.parameters)
        assert params.mu == 2.0
        assert params.tau == 0.25
        assert params.name == "meanPrec"

    def test_exponential_parametrization_creation(self):
        """Test creation of distribution with exponential parametrization."""
        # For N(2, 1.5): a = -1/(2*1.5²) = -0.222..., b = 2/1.5² = 0.888...
        dist = self.normal_family(a=-0.222, b=0.888, parametrization_name="exponential")

        # Приводим к конкретному типу параметризации
        params = cast(ExpParametrization, dist.parameters)
        assert params.a == -0.222
        assert params.b == 0.888
        assert params.name == "exponential"

    def test_parametrization_constraints(self):
        """Test parameter constraints validation."""
        # Sigma must be positive
        with pytest.raises(ValueError, match="sigma > 0"):
            self.normal_family(mu=0, sigma=-1.0)

        # Tau must be positive
        with pytest.raises(ValueError, match="tau > 0"):
            self.normal_family(mu=0, tau=-1.0, parametrization_name="meanPrec")

        # a must be negative
        with pytest.raises(ValueError, match="a < 0"):
            self.normal_family(a=1.0, b=0.0, parametrization_name="exponential")

    def test_pdf_calculation(self):
        """Test PDF calculation against scipy.stats.norm."""
        dist = self.normal_family(mu=2.0, sigma=1.5)
        pdf = dist.computation_strategy.query_method("pdf", dist)

        # Test points
        test_points = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

        for x in test_points:
            # Our implementation
            our_pdf = pdf(x)
            # Scipy reference
            scipy_pdf = norm.pdf(x, loc=2.0, scale=1.5)

            assert abs(our_pdf - scipy_pdf) < 1e-10

    def test_cdf_calculation(self):
        """Test CDF calculation against scipy.stats.norm."""
        dist = self.normal_family(mu=2.0, sigma=1.5)
        cdf = dist.computation_strategy.query_method("cdf", dist)

        test_points = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

        for x in test_points:
            our_cdf = cdf(x)
            scipy_cdf = norm.cdf(x, loc=2.0, scale=1.5)

            assert abs(our_cdf - scipy_cdf) < 1e-10

    def test_ppf_calculation(self):
        """Test PPF calculation against scipy.stats.norm."""
        dist = self.normal_family(mu=2.0, sigma=1.5)
        ppf = dist.computation_strategy.query_method("ppf", dist)

        test_probabilities = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

        for p in test_probabilities:
            our_ppf = ppf(p)
            scipy_ppf = norm.ppf(p, loc=2.0, scale=1.5)

            assert abs(our_ppf - scipy_ppf) < 1e-10

    def test_characteristic_function(self):
        """Test characteristic function calculation."""
        dist = self.normal_family(mu=2.0, sigma=1.5)
        char_func = dist.computation_strategy.query_method("char_func", dist)

        test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]

        for t in test_points:
            cf_value = char_func(t)

            # Characteristic function of N(μ, σ²) is exp(iμt - ½σ²t²)
            expected_real = math.exp(-0.5 * (1.5 * t) ** 2) * math.cos(2.0 * t)
            expected_imag = math.exp(-0.5 * (1.5 * t) ** 2) * math.sin(2.0 * t)

            assert abs(cf_value.real - expected_real) < 1e-10
            assert abs(cf_value.imag - expected_imag) < 1e-10

    def test_moments(self):
        """Test moment calculations."""
        dist = self.normal_family(mu=2.0, sigma=1.5)

        mean_func = dist.computation_strategy.query_method("mean", dist)
        var_func = dist.computation_strategy.query_method("var", dist)
        skew_func = dist.computation_strategy.query_method("skewness", dist)
        kurt_func = dist.computation_strategy.query_method("excess_kurtosis", dist)

        our_mean = mean_func(None)
        our_var = var_func(None)
        our_skew = skew_func(None)
        our_kurt = kurt_func(None)

        assert abs(our_mean - 2.0) < 1e-10
        assert abs(our_var - 2.25) < 1e-10
        assert abs(our_skew - 0.0) < 1e-10
        assert abs(our_kurt - 0.0) < 1e-10

    def test_parametrization_conversions(self):
        """Test conversions between different parameterizations."""
        # Create with mean-variance
        dist_mv = self.normal_family(mu=2.0, sigma=1.5)

        # Convert to base should return same for meanVar
        base_params = self.normal_family.to_base(dist_mv.parameters)
        base_params = cast(MeanVarParametrization, base_params)
        assert base_params.mu == 2.0
        assert base_params.sigma == 1.5

        # Test meanPrec conversion
        dist_mp = self.normal_family(mu=2.0, tau=0.25, parametrization_name="meanPrec")
        base_from_mp = self.normal_family.to_base(dist_mp.parameters)
        base_from_mp = cast(MeanVarParametrization, base_from_mp)
        assert abs(base_from_mp.mu - 2.0) < 1e-10
        assert abs(base_from_mp.sigma - 2.0) < 1e-10  # sigma = 1/sqrt(tau) = 1/sqrt(0.25) = 2

        # Test exponential conversion
        dist_exp = self.normal_family(a=-0.222, b=0.888, parametrization_name="exponential")
        base_from_exp = self.normal_family.to_base(dist_exp.parameters)
        base_from_exp = cast(MeanVarParametrization, base_from_exp)
        # Should be approximately N(2, 1.5)
        assert abs(base_from_exp.mu - 2.0) < 0.1
        assert abs(base_from_exp.sigma - 1.5) < 0.1

    def test_analytical_computations_caching(self):
        """Test that analytical computations are properly cached."""
        dist = self.normal_family(mu=0.0, sigma=1.0)

        # Access analytical computations multiple times
        comp1 = dist.analytical_computations
        comp2 = dist.analytical_computations

        # Should be the same object (cached)
        assert comp1 is comp2

        # Should contain expected characteristics
        expected_chars = {
            "pdf",
            "cdf",
            "ppf",
            "char_func",
            "mean",
            "var",
            "skewness",
            "raw_kurtosis",
            "excess_kurtosis",
        }
        assert set(comp1.keys()) == expected_chars

    def test_array_input_support(self):
        """Test that characteristics support array inputs."""
        dist = self.normal_family(mu=0.0, sigma=1.0)

        # Test with numpy array input
        x_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        pdf = dist.computation_strategy.query_method("pdf", dist)
        cdf = dist.computation_strategy.query_method("cdf", dist)

        pdf_array = pdf(x_array)
        cdf_array = cdf(x_array)

        # Results should be arrays of same shape
        assert pdf_array.shape == x_array.shape
        assert cdf_array.shape == x_array.shape

        # Compare with scipy
        scipy_pdf = norm.pdf(x_array)
        scipy_cdf = norm.cdf(x_array)

        np.testing.assert_array_almost_equal(pdf_array, scipy_pdf, decimal=10)
        np.testing.assert_array_almost_equal(cdf_array, scipy_cdf, decimal=10)


class TestNormalFamilyEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup before each test method."""
        _reset_families_register_for_tests()
        configure_family_register()
        self.normal_family = ParametricFamilyRegister.get("Normal Family")

    def test_invalid_parameterization(self):
        """Test error for invalid parameterization name."""
        with pytest.raises(KeyError):
            self.normal_family.distribution(parametrization_name="invalid_name", mu=0, sigma=1)

    def test_missing_parameters(self):
        """Test error for missing required parameters."""
        with pytest.raises(TypeError):
            self.normal_family.distribution(mu=0)  # Missing sigma

    def test_invalid_probability_ppf(self):
        """Test PPF with invalid probability values."""
        dist = self.normal_family(mu=0.0, sigma=1.0)
        ppf_char = GenericCharacteristic[float, float]("ppf")

        # Test boundaries
        assert ppf_char(dist, 0.0) == float("-inf")
        assert ppf_char(dist, 1.0) == float("inf")

        # Test invalid probabilities
        with pytest.raises(ValueError):
            ppf_char(dist, -0.1)
        with pytest.raises(ValueError):
            ppf_char(dist, 1.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
