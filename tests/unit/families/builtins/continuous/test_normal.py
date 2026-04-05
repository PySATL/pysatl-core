"""
Tests for Normal Distribution Family

This module tests the functionality of the normal distribution family,
including parameterizations, characteristics, and sampling.
"""

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math

import numpy as np
import pytest
from scipy.stats import norm

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.types import (
    CharacteristicName,
    ContinuousSupportShape1D,
    FamilyName,
    UnivariateContinuous,
)

from .base import BaseDistributionTest


class TestNormalFamily(BaseDistributionTest):
    """Test suite for Normal distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.normal_family = registry.get(FamilyName.NORMAL)
        self.normal_dist_example = self.normal_family(mu=2.0, sigma=1.5)

    def test_family_properties(self):
        """Test basic properties of normal family."""
        assert self.normal_family.name == FamilyName.NORMAL

        # Check parameterizations
        expected_parametrizations = {"meanStd", "meanVar", "meanPrec", "exponential"}
        assert set(self.normal_family.parametrization_names) == expected_parametrizations
        assert self.normal_family.base_parametrization_name == "meanStd"

    def test_mean_std_parametrization_creation(self):
        """Test creation of distribution with standard parametrization."""
        dist = self.normal_family(mu=2.0, sigma=1.5)

        assert dist.family_name == FamilyName.NORMAL
        assert dist.distribution_type == UnivariateContinuous
        assert dist.parameters == {"mu": 2.0, "sigma": 1.5}
        assert dist.parametrization_name == "meanStd"

    def test_mean_var_parametrization_creation(self):
        """Test creation of distribution with mean-variance parametrization."""
        dist = self.normal_family(mu=2.0, var=2.25, parametrization_name="meanVar")

        assert dist.parameters == {"mu": 2.0, "var": 2.25}
        assert dist.parametrization_name == "meanVar"

    def test_mean_prec_parametrization_creation(self):
        """Test creation of distribution with mean-precision parametrization."""
        dist = self.normal_family(mu=2.0, tau=0.25, parametrization_name="meanPrec")

        assert dist.parameters == {"mu": 2.0, "tau": 0.25}
        assert dist.parametrization_name == "meanPrec"

    def test_exponential_parametrization_creation(self):
        """Test creation of distribution with exponential parametrization."""
        # For N(2, 1.5): a = -1/(2*1.5²) = -0.222..., b = 2/1.5² = 0.888...
        dist = self.normal_family(a=-0.222, b=0.888, parametrization_name="exponential")

        assert dist.parameters == {"a": -0.222, "b": 0.888}
        assert dist.parametrization_name == "exponential"

    def test_parametrization_constraints(self):
        """Test parameter constraints validation."""
        # Sigma must be positive
        with pytest.raises(ValueError, match="sigma > 0"):
            self.normal_family(mu=0, sigma=-1.0)

        # Var must be positive
        with pytest.raises(ValueError, match="var > 0"):
            self.normal_family(mu=0, var=-1.0, parametrization_name="meanVar")

        # Tau must be positive
        with pytest.raises(ValueError, match="tau > 0"):
            self.normal_family(mu=0, tau=-1.0, parametrization_name="meanPrec")

        # a must be negative
        with pytest.raises(ValueError, match="a < 0"):
            self.normal_family(a=1.0, b=0.0, parametrization_name="exponential")

    @pytest.mark.parametrize(
        "char_func_getter, expected",
        [
            (lambda distr: distr.query_method(CharacteristicName.MEAN)(), 2.0),
            (lambda distr: distr.query_method(CharacteristicName.VAR)(), 2.25),
            (lambda distr: distr.query_method(CharacteristicName.SKEW)(), 0.0),
        ],
    )
    def test_moments(self, char_func_getter, expected):
        """Test moment calculations using parameterized tests."""
        actual = char_func_getter(self.normal_dist_example)
        assert abs(actual - expected) < self.CALCULATION_PRECISION

    def test_kurtosis_calculation(self):
        """Test kurtosis calculation with excess parameter."""
        kurt_func = self.normal_dist_example.query_method(CharacteristicName.KURT)

        raw_kurt = kurt_func()
        assert abs(raw_kurt - 3.0) < self.CALCULATION_PRECISION

        excess_kurt = kurt_func(excess=True)
        assert abs(excess_kurt - 0.0) < self.CALCULATION_PRECISION

        raw_kurt_explicit = kurt_func(excess=False)
        assert abs(raw_kurt_explicit - 3.0) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "parametrization_name, params, expected_mu, expected_sigma",
        [
            ("meanStd", {"mu": 2.0, "sigma": 1.5}, 2.0, 1.5),
            ("meanVar", {"mu": 2.0, "var": 2.25}, 2.0, 1.5),
            ("meanPrec", {"mu": 2.0, "tau": 0.25}, 2.0, math.sqrt(1 / 0.25)),
            ("exponential", {"a": -1 / (2 * 1.5**2), "b": 2 / (1.5**2)}, 2.0, 1.5),
        ],
    )
    def test_parametrization_conversions(
        self, parametrization_name, params, expected_mu, expected_sigma
    ):
        """Test conversions between different parameterizations."""
        base_params = self.normal_family.to_base(
            self.normal_family.get_parametrization(parametrization_name)(**params)
        )

        assert abs(base_params.parameters["mu"] - expected_mu) < self.CALCULATION_PRECISION
        assert abs(base_params.parameters["sigma"] - expected_sigma) < self.CALCULATION_PRECISION

    def test_analytical_computations_availability(self):
        """Test that analytical computations are available for exponential distribution."""
        comp = self.normal_family(mu=0.0, sigma=1.0).analytical_computations

        expected_chars = {
            CharacteristicName.PDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
            CharacteristicName.CF,
            CharacteristicName.LPDF,
            CharacteristicName.SCORE,
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
                [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                norm.pdf,
                {"loc": 2.0, "scale": 1.5},
            ),
            (
                CharacteristicName.CDF,
                [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                norm.cdf,
                {"loc": 2.0, "scale": 1.5},
            ),
            (
                CharacteristicName.PPF,
                [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999],
                norm.ppf,
                {"loc": 2.0, "scale": 1.5},
            ),
            (
                CharacteristicName.LPDF,
                [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
                norm.logpdf,
                {"loc": 2.0, "scale": 1.5},
            ),
        ],
    )
    def test_array_input_for_characteristics(self, char_name, test_data, scipy_func, scipy_kwargs):
        """Test that characteristics support array inputs."""
        dist = self.normal_dist_example
        char_func = dist.query_method(char_name)

        input_array = np.array(test_data)
        result_array = char_func(input_array)

        assert result_array.shape == input_array.shape

        expected_array = scipy_func(input_array, **scipy_kwargs)

        self.assert_arrays_almost_equal(result_array, expected_array)

    def test_characteristic_function_array_input(self):
        """Test characteristic function calculation with array input."""
        char_func = self.normal_dist_example.query_method(CharacteristicName.CF)
        t_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        cf_array = char_func(t_array)
        assert cf_array.shape == t_array.shape

        mu, sigma = 2.0, 1.5
        expected = np.exp(1j * mu * t_array - 0.5 * (sigma**2) * (t_array**2))

        self.assert_arrays_almost_equal(cf_array.real, expected.real)
        self.assert_arrays_almost_equal(cf_array.imag, expected.imag)

    def test_normal_support(self):
        """Test that normal distribution has correct support (entire real line)."""
        dist = self.normal_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)

        assert dist.support.left == float("-inf")
        assert dist.support.right == float("inf")
        assert not dist.support.left_closed
        assert not dist.support.right_closed

        assert dist.support.contains(0) is True
        assert dist.support.contains(float("inf")) is False
        assert dist.support.contains(float("-inf")) is False

        test_points = np.array([-500, 0, 5])
        results = dist.support.contains(test_points)
        assert np.all(results)

        assert dist.support.shape == ContinuousSupportShape1D.REAL_LINE

    def test_score_shape_and_type(self):
        """Test that SCORE returns correct shape and type for all parametrizations."""
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

        for param_name, params in [
            ("meanStd", {"mu": 2.0, "sigma": 1.5}),
            ("meanVar", {"mu": 2.0, "var": 2.25}),
            ("meanPrec", {"mu": 2.0, "tau": 0.25}),
            ("exponential", {"a": -0.222, "b": 0.888}),
        ]:
            dist = self.normal_family(parametrization_name=param_name, **params)  # type: ignore[arg-type]
            score_func = dist.query_method(CharacteristicName.SCORE)
            grad = score_func(x)

            # Shape: (len(x), d_params) where d_params = 2 for all normal parametrizations
            assert grad.shape == (len(x), 2)
            assert grad.dtype == float
            assert not np.any(np.isnan(grad))  # all gradients exist for normal

    def test_score_base_parametrization(self):
        """Test SCORE for meanStd parametrization against analytical formula."""
        mu, sigma = 2.0, 1.5
        dist = self.normal_family(mu=mu, sigma=sigma)
        score_func = dist.query_method(CharacteristicName.SCORE)

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        grad = score_func(x)

        # Analytical gradients: d/dmu = (x-mu)/sigma^2, d/dsigma = -1/sigma + (x-mu)^2/sigma^3
        expected_mu = (x - mu) / sigma**2
        expected_sigma = -1.0 / sigma + (x - mu) ** 2 / sigma**3
        expected = np.stack([expected_mu, expected_sigma], axis=-1)

        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    def test_score_meanVar_parametrization(self):
        """Test SCORE for meanVar parametrization via chain rule."""
        mu, var = 2.0, 2.25
        dist = self.normal_family(parametrization_name="meanVar", mu=mu, var=var)
        score_func = dist.query_method(CharacteristicName.SCORE)

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        grad = score_func(x)

        # Compute base gradient then transform: var = sigma^2 -> dvar = dsigma / (2*sigma)
        sigma = np.sqrt(var)
        base_mu = (x - mu) / sigma**2
        base_sigma = -1.0 / sigma + (x - mu) ** 2 / sigma**3
        expected_mu = base_mu
        expected_var = base_sigma / (2 * sigma)
        expected = np.stack([expected_mu, expected_var], axis=-1)

        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    def test_score_meanPrec_parametrization(self):
        """Test SCORE for meanPrec parametrization via chain rule."""
        mu, tau = 2.0, 0.25  # sigma = 2.0, var = 4.0
        dist = self.normal_family(parametrization_name="meanPrec", mu=mu, tau=tau)
        score_func = dist.query_method(CharacteristicName.SCORE)

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        grad = score_func(x)

        sigma = 1.0 / np.sqrt(tau)  # = 2.0
        base_mu = (x - mu) / sigma**2
        base_sigma = -1.0 / sigma + (x - mu) ** 2 / sigma**3
        expected_mu = base_mu
        expected_tau = -base_sigma * sigma**3 / 2.0
        expected = np.stack([expected_mu, expected_tau], axis=-1)

        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    def test_score_exponential_parametrization(self):
        """Test SCORE for exponential parametrization via chain rule."""
        # Parameters for N(2, 1.5): a = -1/(2*1.5^2) = -0.222..., b = 2/1.5^2 = 0.888...
        a = -1.0 / (2 * 1.5**2)
        b = 2.0 / 1.5**2
        dist = self.normal_family(parametrization_name="exponential", a=a, b=b)
        score_func = dist.query_method(CharacteristicName.SCORE)

        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        grad = score_func(x)

        # Base gradient
        mu = 2.0
        sigma = 1.5
        base_mu = (x - mu) / sigma**2
        base_sigma = -1.0 / sigma + (x - mu) ** 2 / sigma**3

        # Jacobian: mu = -b/(2a), sigma = sqrt(-1/(2a))
        dmu_da = b / (2 * a * a)
        dmu_db = -1.0 / (2 * a)
        dsigma_da = 1.0 / (2 * np.sqrt(-2 * a**3))
        dsigma_db = 0.0

        expected_a = base_mu * dmu_da + base_sigma * dsigma_da
        expected_b = base_mu * dmu_db + base_sigma * dsigma_db
        expected = np.stack([expected_a, expected_b], axis=-1)

        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    def test_score_numerical_derivative(self):
        """Compare analytical SCORE with numerical gradient for one parametrization."""
        mu, sigma = 2.0, 1.5
        dist = self.normal_family(mu=mu, sigma=sigma)
        score_func = dist.query_method(CharacteristicName.SCORE)

        x = 1.0  # single point

        def logpdf_mu(mu_val: float) -> float:
            return norm.logpdf(x, loc=mu_val, scale=sigma)

        def logpdf_sigma(sigma_val: float) -> float:
            return norm.logpdf(x, loc=mu, scale=sigma_val)

        eps = 1e-6
        grad_mu_num = (logpdf_mu(mu + eps) - logpdf_mu(mu - eps)) / (2 * eps)
        grad_sigma_num = (logpdf_sigma(sigma + eps) - logpdf_sigma(sigma - eps)) / (2 * eps)

        grad_analytical = score_func(np.array([x]))[0]  # shape (2,)

        np.testing.assert_allclose(grad_analytical[0], grad_mu_num, rtol=1e-5)
        np.testing.assert_allclose(grad_analytical[1], grad_sigma_num, rtol=1e-5)


class TestNormalFamilyEdgeCases(BaseDistributionTest):
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.normal_family = registry.get(FamilyName.NORMAL)

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
        dist = self.normal_family(mu=2.0, sigma=1.5)
        ppf = dist.query_method(CharacteristicName.PPF)

        # Test boundaries
        assert ppf(0.0) == float("-inf")
        assert ppf(1.0) == float("inf")

        # Test invalid probabilities
        with pytest.raises(ValueError):
            ppf(-0.1)
        with pytest.raises(ValueError):
            ppf(1.1)
