"""
Tests for Normal Distribution Family Configuration

This module tests the functionality of the normal distribution family
defined in config.py, including parameterizations, characteristics,
and sampling.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from typing import cast

import numpy as np
import pytest
from scipy.stats import norm

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import (
    NormalExpParametrization,
    NormalMeanPrecParametrization,
    NormalMeanStdParametrization,
    configure_families_register,
)
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import ContinuousSupportShape1D, UnivariateContinuous


class TestNormalFamily:
    """Test suite for Normal distribution family."""

    # Precision for floating point comparisons
    CALCULATION_PRECISION = 1e-10

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.normal_family = registry.get("Normal Family")
        self.normal_dist_example = self.normal_family(mu=2.0, sigma=1.5)

    def test_family_registration(self):
        """Test that normal family is properly registered."""
        family = ParametricFamilyRegister.get("Normal Family")
        assert family.name == "Normal Family"

        # Check parameterizations
        expected_parametrizations = {"meanStd", "meanPrec", "exponential"}
        assert set(family.parametrization_names) == expected_parametrizations
        assert family.base_parametrization_name == "meanStd"

    def test_mean_var_parametrization_creation(self):
        """Test creation of distribution with standard parametrization."""
        dist = self.normal_family(mu=2.0, sigma=1.5)

        assert dist.family_name == "Normal Family"
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
        self.normal_family = registry.get("Normal Family")
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
        self.normal_family(mu=0.0, sigma=1.0)
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
