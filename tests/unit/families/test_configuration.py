"""
Tests for Normal Distribution Family Configuration

This module tests the functionality of the normal distribution family
defined in configuration.py, including parameterizations, characteristics,
and sampling.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from typing import cast

import numpy as np
import pytest
from scipy.stats import norm, uniform

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import (
    NormalExpParametrization,
    NormalMeanPrecParametrization,
    NormalMeanStdParametrization,
    UniformMeanWidthParametrization,
    UniformMinRangeParametrization,
    UniformStandardParametrization,
    configure_families_register,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import ContinuousSupportShape1D, UnivariateContinuous


class BaseDistributionTest:
    """Based class for all distribution families' tests"""

    CALCULATION_PRECISION = 1e-10


class TestNormalFamily(BaseDistributionTest):
    """Test suite for Normal distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.normal_family = registry.get("Normal")
        self.normal_dist_example = self.normal_family(mu=2.0, sigma=1.5)

    def test_family_registration(self):
        """Test that normal family is properly registered."""
        family = ParametricFamilyRegister.get("Normal")
        assert family.name == "Normal"

        # Check parameterizations
        expected_parametrizations = {"meanStd", "meanPrec", "exponential"}
        assert set(family.parametrization_names) == expected_parametrizations
        assert family.base_parametrization_name == "meanStd"

    def test_mean_var_parametrization_creation(self):
        """Test creation of distribution with standard parametrization."""
        dist = self.normal_family(mu=2.0, sigma=1.5)

        assert dist.family_name == "Normal"
        assert dist.distribution_type == UnivariateContinuous

        params = cast(NormalMeanStdParametrization, dist.parameters)
        assert params.mu == 2.0
        assert params.sigma == 1.5
        assert params.name == "meanStd"

    def test_mean_prec_parametrization_creation(self):
        """Test creation of distribution with mean-precision parametrization."""
        dist = self.normal_family(mu=2.0, tau=0.25, parametrization_name="meanPrec")

        params = cast(NormalMeanPrecParametrization, dist.parameters)
        assert params.mu == 2.0
        assert params.tau == 0.25
        assert params.name == "meanPrec"

    def test_exponential_parametrization_creation(self):
        """Test creation of distribution with exponential parametrization."""
        # For N(2, 1.5): a = -1/(2*1.5²) = -0.222..., b = 2/1.5² = 0.888...
        dist = self.normal_family(a=-0.222, b=0.888, parametrization_name="exponential")

        params = cast(NormalExpParametrization, dist.parameters)
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
        pdf = self.normal_dist_example.computation_strategy.query_method(
            "pdf", self.normal_dist_example
        )
        test_points = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

        for x in test_points:
            # Our implementation
            our_pdf = pdf(x)
            # Scipy reference
            scipy_pdf = norm.pdf(x, loc=2.0, scale=1.5)

            assert abs(our_pdf - scipy_pdf) < self.CALCULATION_PRECISION

    def test_cdf_calculation(self):
        """Test CDF calculation against scipy.stats.norm."""
        cdf = self.normal_dist_example.computation_strategy.query_method(
            "cdf", self.normal_dist_example
        )
        test_points = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

        for x in test_points:
            our_cdf = cdf(x)
            scipy_cdf = norm.cdf(x, loc=2.0, scale=1.5)

            assert abs(our_cdf - scipy_cdf) < self.CALCULATION_PRECISION

    def test_ppf_calculation(self):
        """Test PPF calculation against scipy.stats.norm."""
        ppf = self.normal_dist_example.computation_strategy.query_method(
            "ppf", self.normal_dist_example
        )
        test_probabilities = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

        for p in test_probabilities:
            our_ppf = ppf(p)
            scipy_ppf = norm.ppf(p, loc=2.0, scale=1.5)

            assert abs(our_ppf - scipy_ppf) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "char_func_arg",
        [
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
        ],
    )
    def test_characteristic_function(self, char_func_arg):
        """Test characteristic function calculation at specific points."""
        char_func = self.normal_dist_example.computation_strategy.query_method(
            "char_func", self.normal_dist_example
        )
        cf_value = char_func(char_func_arg)

        expected_real = math.exp(-0.5 * (1.5 * char_func_arg) ** 2) * math.cos(2.0 * char_func_arg)
        expected_imag = math.exp(-0.5 * (1.5 * char_func_arg) ** 2) * math.sin(2.0 * char_func_arg)

        assert abs(cf_value.real - expected_real) < self.CALCULATION_PRECISION
        assert abs(cf_value.imag - expected_imag) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "char_func_getter, expected",
        [
            (lambda distr: distr.computation_strategy.query_method("mean", distr)(None), 2.0),
            (lambda distr: distr.computation_strategy.query_method("var", distr)(None), 2.25),
            (lambda distr: distr.computation_strategy.query_method("skewness", distr)(None), 0.0),
        ],
    )
    def test_moments(self, char_func_getter, expected):
        """Test moment calculations using parameterized tests."""
        actual = char_func_getter(self.normal_dist_example)
        assert abs(actual - expected) < self.CALCULATION_PRECISION

    def test_kurtosis_calculation(self):
        """Test kurtosis calculation with excess parameter."""
        kurt_func = self.normal_dist_example.computation_strategy.query_method(
            "kurtosis", self.normal_dist_example
        )

        raw_kurt = kurt_func(None)
        assert abs(raw_kurt - 3.0) < self.CALCULATION_PRECISION

        excess_kurt = kurt_func(None, excess=True)
        assert abs(excess_kurt - 0.0) < self.CALCULATION_PRECISION

        raw_kurt_explicit = kurt_func(None, excess=False)
        assert abs(raw_kurt_explicit - 3.0) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "parametrization_name, params, expected_mu, expected_sigma",
        [
            ("meanStd", {"mu": 2.0, "sigma": 1.5}, 2.0, 1.5),
            ("meanPrec", {"mu": 2.0, "tau": 0.25}, 2.0, math.sqrt(1 / 0.25)),
            ("exponential", {"a": -1 / (2 * 1.5**2), "b": 2 / (1.5**2)}, 2.0, 1.5),
        ],
    )
    def test_parametrization_conversions(
        self, parametrization_name, params, expected_mu, expected_sigma
    ):
        """Test conversions between different parameterizations."""
        base_params = cast(
            NormalMeanStdParametrization,
            self.normal_family.to_base(
                self.normal_family.get_parametrization(parametrization_name)(**params)
            ),
        )

        assert abs(base_params.mu - expected_mu) < self.CALCULATION_PRECISION
        assert abs(base_params.sigma - expected_sigma) < self.CALCULATION_PRECISION

    def test_analytical_computations_caching(self):
        """Test that analytical computations are properly cached."""
        comp = self.normal_family(mu=0.0, sigma=1.0).analytical_computations

        expected_chars = {
            "pdf",
            "cdf",
            "ppf",
            "char_func",
            "mean",
            "var",
            "skewness",
            "kurtosis",
        }
        assert set(comp.keys()) == expected_chars

    def test_array_input_support(self):
        """Test that PDF supports array inputs."""
        dist = self.normal_family(mu=0.0, sigma=1.0)
        x_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        pdf = dist.computation_strategy.query_method("pdf", dist)
        pdf_array = pdf(x_array)

        assert pdf_array.shape == x_array.shape
        scipy_pdf = norm.pdf(x_array, loc=0.0, scale=1.0)

        np.testing.assert_array_almost_equal(
            pdf_array, scipy_pdf, decimal=int(-math.log10(self.CALCULATION_PRECISION))
        )

    def test_normal_support(self):
        """Test that normal distribution has correct support (entire real line)."""
        dist = self.normal_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)

        assert dist.support.left == float("-inf")
        assert dist.support.right == float("inf")
        assert not dist.support.left_closed  # -∞ is always open
        assert not dist.support.right_closed  # +∞ is always open

        assert dist.support.contains(0) is True
        assert dist.support.contains(float("inf")) is False  # ∞ is not in the support
        assert dist.support.contains(float("-inf")) is False  # -∞ is not in the support

        test_points = np.array([-500, 0, 5])
        results = dist.support.contains(test_points)
        assert np.all(results)

        assert dist.support.shape == ContinuousSupportShape1D.REAL_LINE


class TestNormalFamilyEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.normal_family = registry.get("Normal")
        self.normal_dist_example = self.normal_family(mu=2.0, sigma=1.5)

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
        ppf = self.normal_dist_example.computation_strategy.query_method(
            "ppf", self.normal_dist_example
        )

        # Test boundaries
        assert ppf(0.0) == float("-inf")
        assert ppf(1.0) == float("inf")

        # Test invalid probabilities
        with pytest.raises(ValueError):
            ppf(-0.1)
        with pytest.raises(ValueError):
            ppf(1.1)


class TestUniformFamily(BaseDistributionTest):
    """Test suite for Uniform distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.uniform_family = registry.get("ContinuousUniform")
        self.uniform_dist_example = self.uniform_family(lower_bound=2.0, upper_bound=5.0)

    def test_family_registration(self):
        """Test that uniform family is properly registered."""
        family = ParametricFamilyRegister.get("ContinuousUniform")
        assert family.name == "ContinuousUniform"

        # Check parameterizations
        expected_parametrizations = {"standard", "meanWidth", "minRange"}
        assert set(family.parametrization_names) == expected_parametrizations
        assert family.base_parametrization_name == "standard"

    def test_standard_parametrization_creation(self):
        """Test creation of distribution with standard parametrization."""
        dist = self.uniform_family(lower_bound=2.0, upper_bound=5.0)

        assert dist.family_name == "ContinuousUniform"
        assert dist.distribution_type == UnivariateContinuous

        params = cast(UniformStandardParametrization, dist.parameters)
        assert params.lower_bound == 2.0
        assert params.upper_bound == 5.0
        assert params.name == "standard"

    def test_mean_width_parametrization_creation(self):
        """Test creation of distribution with mean-width parametrization."""
        dist = self.uniform_family(mean=3.5, width=3.0, parametrization_name="meanWidth")

        params = cast(UniformMeanWidthParametrization, dist.parameters)
        assert params.mean == 3.5
        assert params.width == 3.0
        assert params.name == "meanWidth"

    def test_min_range_parametrization_creation(self):
        """Test creation of distribution with min-range parametrization."""
        dist = self.uniform_family(minimum=2.0, range_val=3.0, parametrization_name="minRange")

        params = cast(UniformMinRangeParametrization, dist.parameters)
        assert params.minimum == 2.0
        assert params.range_val == 3.0
        assert params.name == "minRange"

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

    def test_pdf_calculation(self):
        """Test PDF calculation against scipy.stats.uniform."""
        pdf = self.uniform_dist_example.computation_strategy.query_method(
            "pdf", self.uniform_dist_example
        )
        test_points = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

        for x in test_points:
            # Our implementation
            our_pdf = pdf(x)
            # Scipy reference
            scipy_pdf = uniform.pdf(x, loc=2.0, scale=3.0)

            assert abs(our_pdf - scipy_pdf) < self.CALCULATION_PRECISION

    def test_cdf_calculation(self):
        """Test CDF calculation against scipy.stats.uniform."""
        cdf = self.uniform_dist_example.computation_strategy.query_method(
            "cdf", self.uniform_dist_example
        )
        test_points = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

        for x in test_points:
            our_cdf = cdf(x)
            scipy_cdf = uniform.cdf(x, loc=2.0, scale=3.0)

            assert abs(our_cdf - scipy_cdf) < self.CALCULATION_PRECISION

    def test_ppf_calculation(self):
        """Test PPF calculation against scipy.stats.uniform."""
        ppf = self.uniform_dist_example.computation_strategy.query_method(
            "ppf", self.uniform_dist_example
        )
        test_probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]

        for p in test_probabilities:
            our_ppf = ppf(p)
            scipy_ppf = uniform.ppf(p, loc=2.0, scale=3.0)

            assert abs(our_ppf - scipy_ppf) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "char_func_arg",
        [
            -2.0,
            -1.0,
            1.0,
            2.0,
        ],
    )
    def test_characteristic_function(self, char_func_arg):
        """Test characteristic function calculation at specific points."""
        char_func = self.uniform_dist_example.computation_strategy.query_method(
            "char_func", self.uniform_dist_example
        )
        cf_value = char_func(char_func_arg)

        # Analytical formula for characteristic function of uniform distribution
        a, b = 2.0, 5.0
        width = b - a

        # φ(t) = (e^{itb} - e^{ita}) / (it(b-a))
        # Re(φ(t)) = (sin(tb) - sin(ta)) / (t(b-a))
        # Im(φ(t)) = -(cos(tb) - cos(ta)) / (t(b-a))
        if abs(char_func_arg) < self.CALCULATION_PRECISION:
            expected_real = 1.0
            expected_imag = 0.0
        else:
            expected_real = (math.sin(b * char_func_arg) - math.sin(a * char_func_arg)) / (
                char_func_arg * width
            )
            expected_imag = -(math.cos(b * char_func_arg) - math.cos(a * char_func_arg)) / (
                char_func_arg * width
            )

        assert abs(cf_value.real - expected_real) < self.CALCULATION_PRECISION
        assert abs(cf_value.imag - expected_imag) < self.CALCULATION_PRECISION

    def test_characteristic_function_at_zero(self):
        """Test characteristic function at zero returns 1."""
        char_func = self.uniform_dist_example.computation_strategy.query_method(
            "char_func", self.uniform_dist_example
        )

        cf_value_zero = char_func(0.0)
        assert abs(cf_value_zero.real - 1.0) < self.CALCULATION_PRECISION
        assert abs(cf_value_zero.imag) < self.CALCULATION_PRECISION

        cf_value_small = char_func(self.CALCULATION_PRECISION)
        assert abs(cf_value_small.real - 1.0) < self.CALCULATION_PRECISION

        cf_value_large = char_func(1000.0)
        assert isinstance(cf_value_large, complex)
        assert abs(cf_value_large) <= 1

    def test_moments(self):
        """Test moment calculations."""
        dist = self.uniform_dist_example

        # Mean
        mean_func = dist.computation_strategy.query_method("mean", dist)
        assert abs(mean_func(None) - 3.5) < self.CALCULATION_PRECISION

        # Variance
        var_func = dist.computation_strategy.query_method("var", dist)
        assert abs(var_func(None) - 0.75) < self.CALCULATION_PRECISION

        # Skewness
        skew_func = dist.computation_strategy.query_method("skewness", dist)
        assert abs(skew_func(None) - 0.0) < self.CALCULATION_PRECISION

    def test_kurtosis_calculation(self):
        """Test kurtosis calculation with excess parameter."""
        kurt_func = self.uniform_dist_example.computation_strategy.query_method(
            "kurtosis", self.uniform_dist_example
        )

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
        base_params = cast(
            UniformStandardParametrization,
            self.uniform_family.to_base(
                self.uniform_family.get_parametrization(parametrization_name)(**params)
            ),
        )

        assert abs(base_params.lower_bound - expected_lower) < self.CALCULATION_PRECISION
        assert abs(base_params.upper_bound - expected_upper) < self.CALCULATION_PRECISION

    def test_analytical_computations_caching(self):
        """Test that analytical computations are properly cached."""
        comp = self.uniform_family(lower_bound=0.0, upper_bound=1.0).analytical_computations

        expected_chars = {
            "pdf",
            "cdf",
            "ppf",
            "char_func",
            "mean",
            "var",
            "skewness",
            "kurtosis",
        }
        assert set(comp.keys()) == expected_chars

    def test_array_input_support_pdf(self):
        """Test that PDF supports array inputs."""
        dist = self.uniform_family(lower_bound=0.0, upper_bound=1.0)
        x_array = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])

        pdf = dist.computation_strategy.query_method("pdf", dist)
        pdf_array = pdf(x_array)

        assert pdf_array.shape == x_array.shape
        scipy_pdf = uniform.pdf(x_array, loc=0.0, scale=1.0)

        np.testing.assert_array_almost_equal(
            pdf_array, scipy_pdf, decimal=int(-math.log10(self.CALCULATION_PRECISION))
        )

    def test_array_input_support_cdf(self):
        """Test that CDF supports array inputs."""
        dist = self.uniform_family(lower_bound=0.0, upper_bound=1.0)
        x_array = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5])

        cdf = dist.computation_strategy.query_method("cdf", dist)
        cdf_array = cdf(x_array)

        assert cdf_array.shape == x_array.shape
        scipy_cdf = uniform.cdf(x_array, loc=0.0, scale=1.0)

        np.testing.assert_array_almost_equal(
            cdf_array, scipy_cdf, decimal=int(-math.log10(self.CALCULATION_PRECISION))
        )

    def test_array_input_support_ppf(self):
        """Test that PPF supports array inputs."""
        dist = self.uniform_family(lower_bound=0.0, upper_bound=1.0)
        p_array = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        ppf = dist.computation_strategy.query_method("ppf", dist)
        ppf_array = ppf(p_array)

        assert ppf_array.shape == p_array.shape
        scipy_ppf = uniform.ppf(p_array, loc=0.0, scale=1.0)

        np.testing.assert_array_almost_equal(
            ppf_array, scipy_ppf, decimal=int(-math.log10(self.CALCULATION_PRECISION))
        )

    def test_uniform_support(self):
        """Test that uniform distribution has correct support [lower_bound, upper_bound]."""
        dist = self.uniform_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)

        assert dist.support.left == 2.0
        assert dist.support.right == 5.0
        assert dist.support.left_closed  # [a, b] - inclusive
        assert dist.support.right_closed  # [a, b] - inclusive

        # Test containment
        assert dist.support.contains(2.0) is True  # boundary included
        assert dist.support.contains(5.0) is True  # boundary included
        assert dist.support.contains(3.5) is True  # inside
        assert dist.support.contains(1.9) is False  # outside left
        assert dist.support.contains(5.1) is False  # outside right

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
        self.uniform_family = registry.get("ContinuousUniform")
        self.uniform_dist = self.uniform_family(lower_bound=0.0, upper_bound=1.0)

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
        ppf = self.uniform_dist.computation_strategy.query_method("ppf", self.uniform_dist)

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
        # This should fail validation
        with pytest.raises(ValueError, match="lower_bound < upper_bound"):
            self.uniform_family(lower_bound=2.0, upper_bound=2.0)

    def test_characteristic_function_edge_cases(self):
        """Test characteristic function at edge cases."""
        char_func = self.uniform_dist.computation_strategy.query_method(
            "char_func", self.uniform_dist
        )

        # Test with very small t
        cf_value_small = char_func(self.CALCULATION_PRECISION)
        # Should be close to 1, but may have numerical issues
        assert abs(cf_value_small.real - 1.0) < self.CALCULATION_PRECISION

        # Test with large t
        cf_value_large = char_func(1000.0)
        # Characteristic function should still be a complex number
        assert isinstance(cf_value_large, complex)
        assert abs(cf_value_large) <= 1.0  # |φ(t)| ≤ 1 for all t

    def test_negative_width(self):
        """Test that negative width is rejected."""
        with pytest.raises(ValueError, match="width > 0"):
            self.uniform_family(mean=0.0, width=-1.0, parametrization_name="meanWidth")

    def test_negative_range(self):
        """Test that negative range is rejected."""
        with pytest.raises(ValueError, match="range_val > 0"):
            self.uniform_family(minimum=0.0, range_val=-1.0, parametrization_name="minRange")
