"""
Tests for Gamma Distribution Family

This module tests the functionality of the gamma distribution family,
including parameterizations, characteristics, and support.
"""

__author__ = "Elizaveta Bykova"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import gamma as gamma_dist

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.types import (
    CharacteristicName,
    ContinuousSupportShape1D,
    FamilyName,
    UnivariateContinuous,
)

from .base import BaseDistributionTest


class TestGammaFamily(BaseDistributionTest):
    """Test suite for Gamma distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.gamma_family = registry.get(FamilyName.GAMMA)
        self.gamma_dist_example = self.gamma_family(k=2.0, theta=3.0)

    def test_family_properties(self):
        """Test basic properties of gamma family."""
        assert self.gamma_family.name == FamilyName.GAMMA

        # Check parameterizations
        expected_parametrizations = {"shapeScale", "shapeRate", "meanVar"}
        assert set(self.gamma_family.parametrization_names) == expected_parametrizations
        assert self.gamma_family.base_parametrization_name == "shapeScale"

    def test_shape_scale_parametrization_creation(self):
        """Test creation of distribution with shape-scale parametrization."""
        dist = self.gamma_family(k=2.0, theta=3.0)

        assert dist.family_name == FamilyName.GAMMA
        assert dist.distribution_type == UnivariateContinuous
        assert dist.parameters == {"k": 2.0, "theta": 3.0}
        assert dist.parametrization_name == "shapeScale"

    def test_shape_rate_parametrization_creation(self):
        """Test creation of distribution with shape-rate parametrization."""
        dist = self.gamma_family(k=2.0, beta=0.5, parametrization_name="shapeRate")

        assert dist.parameters == {"k": 2.0, "beta": 0.5}
        assert dist.parametrization_name == "shapeRate"

    def test_mean_var_parametrization_creation(self):
        """Test creation of distribution with mean-variance parametrization."""
        dist = self.gamma_family(m=6.0, v=18.0, parametrization_name="meanVar")

        assert dist.parameters == {"m": 6.0, "v": 18.0}
        assert dist.parametrization_name == "meanVar"

    def test_parametrization_constraints(self):
        """Test parameter constraints validation."""
        with pytest.raises(ValueError, match="k > 0"):
            self.gamma_family(k=-1.0, theta=1.0)

        with pytest.raises(ValueError, match="theta > 0"):
            self.gamma_family(k=1.0, theta=-1.0)

        with pytest.raises(ValueError, match="k > 0"):
            self.gamma_family(k=-1.0, beta=1.0, parametrization_name="shapeRate")

        with pytest.raises(ValueError, match="beta > 0"):
            self.gamma_family(k=1.0, beta=-1.0, parametrization_name="shapeRate")

        with pytest.raises(ValueError, match="m > 0"):
            self.gamma_family(m=-1.0, v=1.0, parametrization_name="meanVar")

        with pytest.raises(ValueError, match="v > 0"):
            self.gamma_family(m=1.0, v=-1.0, parametrization_name="meanVar")

    def test_moments(self):
        """Test moment calculations for k=2, theta=3."""
        mean_func = self.gamma_dist_example.query_method(CharacteristicName.MEAN)
        assert abs(mean_func() - 6.0) < self.CALCULATION_PRECISION

        var_func = self.gamma_dist_example.query_method(CharacteristicName.VAR)
        assert abs(var_func() - 18.0) < self.CALCULATION_PRECISION

    @pytest.mark.parametrize(
        "parametrization_name, params, expected_k, expected_theta",
        [
            ("shapeScale", {"k": 2.0, "theta": 3.0}, 2.0, 3.0),
            ("shapeRate", {"k": 2.0, "beta": 1 / 3}, 2.0, 3.0),
            ("meanVar", {"m": 6.0, "v": 18.0}, 2.0, 3.0),
        ],
    )
    def test_parametrization_conversions(
        self, parametrization_name, params, expected_k, expected_theta
    ):
        """Test conversions between different parameterizations."""
        base_params = self.gamma_family.to_base(
            self.gamma_family.get_parametrization(parametrization_name)(**params)
        )

        assert abs(base_params.parameters["k"] - expected_k) < self.CALCULATION_PRECISION
        assert abs(base_params.parameters["theta"] - expected_theta) < self.CALCULATION_PRECISION

    def test_analytical_computations_availability(self):
        """Test that analytical computations are available for gamma distribution."""
        comp = self.gamma_family(k=1.0, theta=1.0).analytical_computations

        expected_chars = {
            CharacteristicName.PDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
            CharacteristicName.CF,
            CharacteristicName.LPDF,
            CharacteristicName.MEAN,
            CharacteristicName.VAR,
        }
        assert set(comp.keys()) == expected_chars

    @pytest.mark.parametrize(
        "char_name, test_data, scipy_func, scipy_kwargs",
        [
            (
                CharacteristicName.PDF,
                [0.5, 1.0, 2.0, 3.0, 5.0],
                gamma_dist.pdf,
                {"a": 2.0, "scale": 3.0},
            ),
            (
                CharacteristicName.CDF,
                [0.5, 1.0, 2.0, 3.0, 5.0],
                gamma_dist.cdf,
                {"a": 2.0, "scale": 3.0},
            ),
            (
                CharacteristicName.PPF,
                [0.1, 0.25, 0.5, 0.75, 0.9],
                gamma_dist.ppf,
                {"a": 2.0, "scale": 3.0},
            ),
            (
                CharacteristicName.LPDF,
                [0.5, 1.0, 2.0, 3.0, 5.0],
                gamma_dist.logpdf,
                {"a": 2.0, "scale": 3.0},
            ),
        ],
    )
    def test_array_input_for_characteristics(self, char_name, test_data, scipy_func, scipy_kwargs):
        """Test that characteristics support array inputs."""
        dist = self.gamma_dist_example
        char_func = dist.query_method(char_name)

        input_array = np.array(test_data)
        result_array = char_func(input_array)

        assert result_array.shape == input_array.shape

        expected_array = scipy_func(input_array, **scipy_kwargs)

        self.assert_arrays_almost_equal(result_array, expected_array)

    def test_characteristic_function_array_input(self):
        """Test characteristic function calculation with array input."""
        char_func = self.gamma_dist_example.query_method(CharacteristicName.CF)
        t_array = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        cf_array = char_func(t_array)
        assert cf_array.shape == t_array.shape

        k, theta = 2.0, 3.0
        expected = (1 - 1j * theta * t_array) ** (-k)

        self.assert_arrays_almost_equal(cf_array.real, expected.real)
        self.assert_arrays_almost_equal(cf_array.imag, expected.imag)

    def test_gamma_support(self):
        """Test that gamma distribution has correct support [0, inf)."""
        dist = self.gamma_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)

        assert dist.support.left == 0.0
        assert dist.support.right == float("inf")
        assert dist.support.left_closed
        assert not dist.support.right_closed

        assert dist.support.contains(1.0) is True
        assert dist.support.contains(-0.1) is False
        assert dist.support.contains(float("inf")) is False

        test_points = np.array([-0.1, 0.5, 1.0, 10.0])
        results = dist.support.contains(test_points)
        np.testing.assert_array_equal(results, [False, True, True, True])

        assert dist.support.shape == ContinuousSupportShape1D.RAY_RIGHT

    def test_cdf_at_zero(self):
        """Test that CDF at x=0 returns 0."""
        dist = self.gamma_family(k=2.0, theta=1.0)
        cdf_func = dist.query_method(CharacteristicName.CDF)
        assert cdf_func(np.array([0.0]))[0] == 0.0

    @pytest.mark.parametrize(
        "k, theta, expected_pdf_0, expected_lpdf_0",
        [
            (2.0, 3.0, 0.0, float("-inf")),  # k > 1
            (1.0, 4.0, 0.25, -np.log(4.0)),  # k == 1 (1/theta)
            (0.5, 2.0, float("inf"), float("inf")),  # 0 < k < 1
        ],
    )
    def test_pdf_and_lpdf_at_zero_boundary(self, k, theta, expected_pdf_0, expected_lpdf_0):
        """Test PDF and LPDF values at the x=0 boundary across all shape regimes."""
        dist = self.gamma_family(k=k, theta=theta)
        pdf_func = dist.query_method(CharacteristicName.PDF)
        lpdf_func = dist.query_method(CharacteristicName.LPDF)

        pdf_val = pdf_func(np.array([0.0]))[0]
        lpdf_val = lpdf_func(np.array([0.0]))[0]

        if np.isinf(expected_pdf_0):
            assert np.isinf(pdf_val) and pdf_val > 0
        else:
            assert abs(pdf_val - expected_pdf_0) < self.CALCULATION_PRECISION

        if np.isinf(expected_lpdf_0):
            if expected_lpdf_0 > 0:
                assert np.isinf(lpdf_val) and lpdf_val > 0
            else:
                assert np.isinf(lpdf_val) and lpdf_val < 0
        else:
            assert abs(lpdf_val - expected_lpdf_0) < self.CALCULATION_PRECISION


class TestGammaFamilyEdgeCases(BaseDistributionTest):
    """Test edge cases and error conditions for gamma distribution."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.gamma_family = registry.get(FamilyName.GAMMA)

    def test_invalid_parameterization(self):
        """Test error for invalid parameterization name."""
        with pytest.raises(KeyError):
            self.gamma_family.distribution(parametrization_name="invalid_name", k=1.0, theta=1.0)

    def test_missing_parameters(self):
        """Test error for missing required parameters."""
        with pytest.raises(TypeError):
            self.gamma_family.distribution(k=1.0)  # Missing theta

    def test_invalid_probability_ppf(self):
        """Test PPF with invalid probability values."""
        dist = self.gamma_family(k=1.0, theta=1.0)
        ppf = dist.query_method(CharacteristicName.PPF)

        assert ppf(0.0) == 0.0
        assert ppf(1.0) == float("inf")

        with pytest.raises(ValueError):
            ppf(-0.1)
        with pytest.raises(ValueError):
            ppf(1.1)

    def test_characteristic_function_at_zero(self):
        """Test characteristic function at zero returns 1."""
        dist = self.gamma_family(k=2.0, theta=3.0)
        char_func = dist.query_method(CharacteristicName.CF)

        cf_value_zero = char_func(0.0)
        assert abs(cf_value_zero.real - 1.0) < self.CALCULATION_PRECISION
        assert abs(cf_value_zero.imag) < self.CALCULATION_PRECISION

    def test_characteristic_function_large_t(self):
        """Test characteristic function with large t."""
        dist = self.gamma_family(k=2.0, theta=3.0)
        char_func = dist.query_method(CharacteristicName.CF)

        cf_value_large = char_func(1000.0)
        assert np.iscomplexobj(cf_value_large)
        assert abs(cf_value_large) <= 1.0

    def test_score_shape_scale(self):
        """Test SCORE for shapeScale (base) parametrization against analytical formula."""
        k, theta = 3.0, 2.0
        dist = self.gamma_family(k=k, theta=theta)
        x = np.array([1.0, 2.0, 5.0, 10.0])
        grad = dist.family.score(dist.parametrization, x)

        from scipy.special import digamma as _digamma

        expected_k = np.log(x) - np.log(theta) - _digamma(k)
        expected_theta = (x - k * theta) / (theta**2)
        expected = np.stack([expected_k, expected_theta], axis=-1)

        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    def test_score_shape_rate(self):
        """Test SCORE for shapeRate parametrization via chain rule."""
        k, beta = 3.0, 0.5
        dist = self.gamma_family(parametrization_name="shapeRate", k=k, beta=beta)
        x = np.array([1.0, 2.0, 5.0, 10.0])
        grad = dist.family.score(dist.parametrization, x)

        from scipy.special import digamma as _digamma

        theta = 1.0 / beta
        base_k = np.log(x) - np.log(theta) - _digamma(k)
        base_theta = (x - k * theta) / (theta**2)

        expected_k = base_k
        expected_beta = base_theta * (-1.0 / beta**2)
        expected = np.stack([expected_k, expected_beta], axis=-1)

        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    def test_score_mean_var(self):
        """Test SCORE for meanVar parametrization via chain rule."""
        m, v = 6.0, 2.0
        dist = self.gamma_family(parametrization_name="meanVar", m=m, v=v)
        x = np.array([1.0, 2.0, 5.0, 10.0])
        grad = dist.family.score(dist.parametrization, x)

        from scipy.special import digamma as _digamma

        k = m**2 / v
        theta = v / m

        base_k = np.log(x) - np.log(theta) - _digamma(k)
        base_theta = (x - k * theta) / (theta**2)

        dk_dm = 2 * m / v
        dk_dv = -(m**2) / (v**2)
        dtheta_dm = -v / (m**2)
        dtheta_dv = 1.0 / m

        expected_m = base_k * dk_dm + base_theta * dtheta_dm
        expected_v = base_k * dk_dv + base_theta * dtheta_dv
        expected = np.stack([expected_m, expected_v], axis=-1)

        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    def test_score_numerical_derivative(self):
        """Compare analytical SCORE with numerical gradient for base parametrization."""
        k, theta = 3.0, 2.0
        dist = self.gamma_family(k=k, theta=theta)
        x_val = 4.0

        def logpdf_k(k_val: float) -> float:
            return (
                (k_val - 1) * np.log(x_val) - x_val / theta - k_val * np.log(theta) - gammaln(k_val)
            )

        def logpdf_theta(theta_val: float) -> float:
            return (k - 1) * np.log(x_val) - x_val / theta_val - k * np.log(theta_val) - gammaln(k)

        eps = 1e-7
        grad_k_num = (logpdf_k(k + eps) - logpdf_k(k - eps)) / (2 * eps)
        grad_theta_num = (logpdf_theta(theta + eps) - logpdf_theta(theta - eps)) / (2 * eps)

        grad = dist.family.score(dist.parametrization, np.array([x_val]))
        np.testing.assert_allclose(grad[0, 0], grad_k_num, rtol=1e-5)
        np.testing.assert_allclose(grad[0, 1], grad_theta_num, rtol=1e-5)

    def test_score_x_outside_support_raises(self):
        """Test SCORE raises ValueError for x <= 0."""
        dist = self.gamma_family(k=2.0, theta=1.0)
        with pytest.raises(ValueError, match="base_score is undefined"):
            dist.family.score(dist.parametrization, np.array([-1.0, 1.0]))
        with pytest.raises(ValueError, match="base_score is undefined"):
            dist.family.score(dist.parametrization, np.array([1.0, np.inf]))
        with pytest.raises(ValueError, match="base_score is undefined"):
            dist.family.score(dist.parametrization, np.array([np.nan, 1.0]))
        with pytest.raises(ValueError, match="base_score is undefined"):
            dist.family.score(dist.parametrization, np.array([0.0, 1.0]))

    @pytest.mark.parametrize(
        "parametrization_name, params",
        [
            ("shapeScale", {"k": 2.0, "theta": 1.5}),
            ("shapeRate", {"k": 2.0, "beta": 2.0 / 3.0}),
            ("meanVar", {"m": 3.0, "v": 1.5}),
        ],
    )
    def test_score_shape(self, parametrization_name, params):
        """Test SCORE output shape for all parametrizations."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = self.gamma_family(parametrization_name=parametrization_name, **params)
        grad = dist.family.score(dist.parametrization, x)
        assert grad.shape == (len(x), 2)
        assert grad.dtype == float
