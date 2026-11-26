"""
Tests for Pareto Distribution Family

This module tests the Pareto (type I) distribution family: the parametrization,
the analytical characteristics against ``scipy.stats.pareto``, the support and the score.
"""

__author__ = "Duan Nguyen"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import warnings
from typing import Any

import numpy as np
import pytest
from scipy.special import exp1
from scipy.stats import pareto

from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.transformations.operations.distributions.affine import affine
from pysatl_core.types import (
    CharacteristicName,
    ContinuousSupportShape1D,
    FamilyName,
    UnivariateContinuous,
)

from .base import BaseDistributionTest

# (x_m, alpha) pairs covering the regimes of the moments: alpha <= 1, 1 < alpha <= 2, alpha > 2.
PARAMETER_SETS = [(1.0, 0.5), (2.0, 1.0), (1.5, 1.5), (3.0, 2.0), (2.0, 3.5), (0.5, 10.0)]

X_POINTS = [-1.0, 0.0, 0.4, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0, 1e3, 1e8]


def _scipy_dist(x_m: float, alpha: float) -> Any:
    return pareto(b=alpha, scale=x_m)


class TestParetoFamily(BaseDistributionTest):
    """Test suite for the Pareto distribution family."""

    def setup_method(self):
        """Setup before each test method."""
        registry = configure_families_register()
        self.pareto_family = registry.get(FamilyName.PARETO)
        self.pareto_dist_example = self.pareto_family(x_m=2.0, alpha=3.0)

    def test_family_name(self):
        """The family is registered under the name ``Pareto``."""
        assert str(FamilyName.PARETO) == "Pareto"
        assert self.pareto_family.name == FamilyName.PARETO

    def test_family_properties(self):
        """Test basic properties of the Pareto family."""
        assert set(self.pareto_family.parametrization_names) == {"scaleShape"}
        assert self.pareto_family.base_parametrization_name == "scaleShape"

    def test_scale_shape_parametrization_creation(self):
        """Test creation of a distribution with the scaleShape parametrization."""
        dist = self.pareto_family(x_m=2.0, alpha=3.0)

        assert dist.family_name == FamilyName.PARETO
        assert dist.distribution_type == UnivariateContinuous
        assert dist.parameters == {"x_m": 2.0, "alpha": 3.0}
        assert dist.parametrization_name == "scaleShape"

    def test_creation_with_explicit_parametrization_name(self):
        """The base parametrization can be named explicitly."""
        dist = self.pareto_family(x_m=2.0, alpha=3.0, parametrization_name="scaleShape")
        assert dist.parameters == {"x_m": 2.0, "alpha": 3.0}

    def test_parametrization_constraints(self):
        """Test parameter constraints validation."""
        with pytest.raises(ValueError, match="x_m > 0"):
            self.pareto_family(x_m=0.0, alpha=1.0)

        with pytest.raises(ValueError, match="x_m > 0"):
            self.pareto_family(x_m=-1.0, alpha=1.0)

        with pytest.raises(ValueError, match="alpha > 0"):
            self.pareto_family(x_m=1.0, alpha=0.0)

        with pytest.raises(ValueError, match="alpha > 0"):
            self.pareto_family(x_m=1.0, alpha=-2.0)

    def test_analytical_computations_availability(self):
        """Test that the required analytical characteristics are available."""
        comp = self.pareto_dist_example.analytical_computations

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

    # ------------------------------------------------------------------
    # pdf, cdf, ppf, lpdf against scipy
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("x_m, alpha", PARAMETER_SETS)
    @pytest.mark.parametrize(
        "char_name, scipy_method",
        [
            (CharacteristicName.PDF, "pdf"),
            (CharacteristicName.CDF, "cdf"),
            (CharacteristicName.LPDF, "logpdf"),
        ],
    )
    def test_characteristics_against_scipy(self, x_m, alpha, char_name, scipy_method):
        """pdf, cdf and lpdf agree with scipy on points below, at and above the support start."""
        dist = self.pareto_family(x_m=x_m, alpha=alpha)
        x = np.array(X_POINTS)

        result = dist.query_method(char_name)(x)
        expected = getattr(_scipy_dist(x_m, alpha), scipy_method)(x)

        assert result.shape == x.shape
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=0.0)

    @pytest.mark.parametrize("x_m, alpha", PARAMETER_SETS)
    def test_ppf_against_scipy(self, x_m, alpha):
        """ppf agrees with scipy, including the endpoints."""
        dist = self.pareto_family(x_m=x_m, alpha=alpha)
        p = np.array([0.0, 1e-12, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999999, 1.0])

        result = dist.query_method(CharacteristicName.PPF)(p)
        expected = _scipy_dist(x_m, alpha).ppf(p)

        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=0.0)

    def test_ppf_endpoints(self):
        """F^-1(0) = x_m and F^-1(1) = +inf."""
        ppf = self.pareto_dist_example.query_method(CharacteristicName.PPF)

        assert ppf(np.array([0.0]))[0] == 2.0
        assert np.isposinf(ppf(np.array([1.0]))[0])

    @pytest.mark.parametrize("bad_p", [-0.1, 1.1])
    def test_ppf_raises_outside_unit_interval(self, bad_p):
        """ppf raises ValueError for probabilities outside [0, 1]."""
        ppf = self.pareto_dist_example.query_method(CharacteristicName.PPF)
        with pytest.raises(ValueError, match=r"Probability must be in \[0, 1\]"):
            ppf(np.array([0.5, bad_p]))

    @pytest.mark.parametrize("x_m, alpha", PARAMETER_SETS)
    def test_ppf_inverts_cdf(self, x_m, alpha):
        """ppf is the inverse of cdf on the support."""
        dist = self.pareto_family(x_m=x_m, alpha=alpha)
        p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])

        x = dist.query_method(CharacteristicName.PPF)(p)
        back = dist.query_method(CharacteristicName.CDF)(x)

        np.testing.assert_allclose(back, p, rtol=1e-10)

    def test_branches_below_the_support_start(self):
        """For x < x_m: pdf = 0, cdf = 0 and lpdf = -inf, with no warnings."""
        dist = self.pareto_dist_example
        x = np.array([-5.0, 0.0, 1.0, 1.999999])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pdf = dist.query_method(CharacteristicName.PDF)(x)
            cdf = dist.query_method(CharacteristicName.CDF)(x)
            lpdf = dist.query_method(CharacteristicName.LPDF)(x)

        np.testing.assert_array_equal(pdf, np.zeros_like(x))
        np.testing.assert_array_equal(cdf, np.zeros_like(x))
        assert np.all(np.isneginf(lpdf))

    def test_value_at_the_support_start(self):
        """At x = x_m: f = alpha / x_m, F = 0."""
        x_m, alpha = 2.0, 3.0
        dist = self.pareto_family(x_m=x_m, alpha=alpha)
        x = np.array([x_m])

        np.testing.assert_allclose(dist.query_method(CharacteristicName.PDF)(x), [alpha / x_m])
        assert dist.query_method(CharacteristicName.CDF)(x)[0] == 0.0

    def test_tail_values_do_not_overflow(self):
        """Far in the tail the density is 0, the cdf is 1 and no warning is raised."""
        dist = self.pareto_dist_example
        x = np.array([1e150, np.inf])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pdf = dist.query_method(CharacteristicName.PDF)(x)
            cdf = dist.query_method(CharacteristicName.CDF)(x)

        np.testing.assert_array_equal(pdf, [0.0, 0.0])
        np.testing.assert_allclose(cdf, [1.0, 1.0])

    def test_lpdf_matches_log_of_pdf(self):
        """lpdf = ln(pdf) on the support."""
        dist = self.pareto_dist_example
        x = np.array([2.0, 2.5, 4.0, 100.0])

        lpdf = dist.query_method(CharacteristicName.LPDF)(x)
        pdf = dist.query_method(CharacteristicName.PDF)(x)

        np.testing.assert_allclose(lpdf, np.log(pdf), rtol=1e-12)

    def test_scalar_input(self):
        """Characteristics accept scalars."""
        dist = self.pareto_dist_example
        pdf = dist.query_method(CharacteristicName.PDF)

        np.testing.assert_allclose(pdf(3.0), pareto(b=3.0, scale=2.0).pdf(3.0), rtol=1e-12)

    def test_array_input_preserves_shape(self):
        """Characteristics keep the shape of a 2-D input."""
        dist = self.pareto_dist_example
        x = np.array([[2.0, 3.0], [4.0, 5.0]])

        for name in (CharacteristicName.PDF, CharacteristicName.CDF, CharacteristicName.LPDF):
            assert dist.query_method(name)(x).shape == x.shape

    # ------------------------------------------------------------------
    # mean and var
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("x_m, alpha", PARAMETER_SETS)
    def test_moments_against_scipy(self, x_m, alpha):
        """mean and var agree with scipy in every regime, with +inf where they do not exist."""
        dist = self.pareto_family(x_m=x_m, alpha=alpha)
        reference = _scipy_dist(x_m, alpha)

        mean = dist.query_method(CharacteristicName.MEAN)()
        var = dist.query_method(CharacteristicName.VAR)()

        np.testing.assert_allclose(mean, reference.mean(), rtol=1e-12)
        np.testing.assert_allclose(var, reference.var(), rtol=1e-12)

    @pytest.mark.parametrize(
        "alpha, mean_finite, var_finite",
        [
            (0.5, False, False),
            (1.0, False, False),  # alpha = 1: the mean does not exist
            (1.5, True, False),
            (2.0, True, False),  # alpha = 2: the variance does not exist
            (2.5, True, True),
            (50.0, True, True),
        ],
    )
    def test_moment_existence_regimes(self, alpha, mean_finite, var_finite):
        """A moment that does not exist is +inf: it neither raises nor goes negative."""
        dist = self.pareto_family(x_m=2.0, alpha=alpha)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mean = dist.query_method(CharacteristicName.MEAN)()
            var = dist.query_method(CharacteristicName.VAR)()

        if mean_finite:
            assert np.isfinite(mean) and mean > 0
        else:
            assert np.isposinf(mean)

        if var_finite:
            assert np.isfinite(var) and var > 0
        else:
            assert np.isposinf(var)

    def test_moment_formulas(self):
        """Check the closed forms for alpha > 2."""
        x_m, alpha = 2.0, 4.0
        dist = self.pareto_family(x_m=x_m, alpha=alpha)

        mean = dist.query_method(CharacteristicName.MEAN)()
        var = dist.query_method(CharacteristicName.VAR)()

        np.testing.assert_allclose(mean, alpha * x_m / (alpha - 1), rtol=1e-14)
        np.testing.assert_allclose(
            var, x_m**2 * alpha / ((alpha - 1) ** 2 * (alpha - 2)), rtol=1e-14
        )

    @pytest.mark.parametrize(
        "alpha, scale, expected_mean, expected_var",
        [
            (0.8, 2.0, np.inf, np.inf),
            (0.8, -2.0, -np.inf, np.inf),
            (1.5, 2.0, 2.0 * 6.0 + 1.0, np.inf),
            (4.0, -2.0, -2.0 * (4.0 * 2.0 / 3.0) + 1.0, 4.0 * 2.0**2 * 4.0 / (9.0 * 2.0)),
        ],
    )
    def test_infinite_moments_propagate_through_affine(
        self, alpha, scale, expected_mean, expected_var
    ):
        """An affine transform of a Pareto keeps a missing moment infinite, with the right sign."""
        transformed = affine(self.pareto_family(x_m=2.0, alpha=alpha), scale=scale, shift=1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mean = transformed.query_method(CharacteristicName.MEAN)()
            var = transformed.query_method(CharacteristicName.VAR)()

        np.testing.assert_allclose(mean, expected_mean, rtol=1e-12)
        np.testing.assert_allclose(var, expected_var, rtol=1e-12)

    # ------------------------------------------------------------------
    # characteristic function
    # ------------------------------------------------------------------

    def test_characteristic_function_at_zero(self):
        """phi(0) = 1."""
        cf = self.pareto_dist_example.query_method(CharacteristicName.CF)

        np.testing.assert_allclose(cf(np.array([0.0])), [1.0 + 0.0j], atol=1e-12)

    @pytest.mark.parametrize("x_m, alpha", PARAMETER_SETS)
    def test_characteristic_function_is_a_characteristic_function(self, x_m, alpha):
        """|phi(t)| <= 1 and phi(-t) = conj(phi(t))."""
        cf = self.pareto_family(x_m=x_m, alpha=alpha).query_method(CharacteristicName.CF)
        t = np.array([0.1, 0.7, 1.0, 3.0, 12.5])

        positive = cf(t)
        negative = cf(-t)

        assert np.all(np.abs(positive) <= 1.0 + 1e-9)
        np.testing.assert_allclose(negative, np.conj(positive), rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("x_m", [0.5, 1.0, 2.0])
    def test_characteristic_function_alpha_one_closed_form(self, x_m):
        """For alpha = 1: phi(t) = exp(-z) - z * E1(z) with z = -i t x_m (independent reference)."""
        cf = self.pareto_family(x_m=x_m, alpha=1.0).query_method(CharacteristicName.CF)
        t = np.array([0.25, 1.0, 2.0, 7.0])
        z = -1j * t * x_m

        expected = np.exp(-z) - z * exp1(z)

        np.testing.assert_allclose(cf(t), expected, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("x_m", [0.5, 1.0, 2.0])
    def test_characteristic_function_alpha_two_closed_form(self, x_m):
        """For alpha = 2: phi(t) = exp(-z) (1 - z) + z^2 E1(z) with z = -i t x_m."""
        cf = self.pareto_family(x_m=x_m, alpha=2.0).query_method(CharacteristicName.CF)
        t = np.array([0.25, 1.0, 2.0, 7.0])
        z = -1j * t * x_m

        expected = np.exp(-z) * (1 - z) + z**2 * exp1(z)

        np.testing.assert_allclose(cf(t), expected, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_characteristic_function_over_a_wide_frequency_range(self, alpha):
        """The closed forms hold from tiny to huge frequencies, without integration warnings."""
        cf = self.pareto_family(x_m=1.0, alpha=alpha).query_method(CharacteristicName.CF)
        y = np.array([1e-7, 1e-5, 1e-3, 0.1, 1.0, 10.0, 1e3, 1e5])
        z = -1j * y

        if alpha == 1.0:
            expected = np.exp(-z) - z * exp1(z)
        else:
            expected = np.exp(-z) * (1 - z) + z**2 * exp1(z)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = cf(y)

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("alpha", [3.0, 8.0, 20.0])
    def test_characteristic_function_small_frequency_expansion(self, alpha):
        """phi(t) = 1 + i t E[X] - t^2 E[X^2] / 2 + o(t^2) for alpha > 2."""
        x_m = 1.5
        cf = self.pareto_family(x_m=x_m, alpha=alpha).query_method(CharacteristicName.CF)
        # The next term is O(t^3 ln t) for alpha = 3 (E[X^3] does not exist), so keep t small.
        t = np.array([1e-6, 1e-5, 1e-4])

        first = alpha * x_m / (alpha - 1)
        second = alpha * x_m**2 / (alpha - 2)
        expected = 1 + 1j * t * first - t**2 * second / 2

        np.testing.assert_allclose(cf(t), expected, rtol=0.0, atol=1e-8)

    def test_characteristic_function_depends_on_t_times_x_m_only(self):
        """phi(t; x_m) = phi(t * x_m; 1), since X / x_m does not depend on x_m."""
        t = np.array([0.01, 1.0, 30.0])
        scaled = self.pareto_family(x_m=4.0, alpha=1.5).query_method(CharacteristicName.CF)
        unit = self.pareto_family(x_m=1.0, alpha=1.5).query_method(CharacteristicName.CF)

        np.testing.assert_allclose(scaled(t), unit(t * 4.0), rtol=0.0, atol=1e-9)

    def test_characteristic_function_array_input(self):
        """The characteristic function accepts a 2-D array and returns a complex array."""
        cf = self.pareto_dist_example.query_method(CharacteristicName.CF)
        t = np.array([[0.0, 0.5], [1.0, -1.0]])

        result = cf(t)

        assert result.shape == t.shape
        assert np.iscomplexobj(result)

    # ------------------------------------------------------------------
    # support
    # ------------------------------------------------------------------

    def test_pareto_support(self):
        """The support is [x_m, +inf) and depends on the parameters."""
        dist = self.pareto_dist_example

        assert dist.support is not None
        assert isinstance(dist.support, ContinuousSupport)
        assert dist.support.left == 2.0
        assert dist.support.right == float("inf")
        assert dist.support.left_closed
        assert dist.support.shape == ContinuousSupportShape1D.RAY_RIGHT

        assert dist.support.contains(2.0) is True
        assert dist.support.contains(1e9) is True
        assert dist.support.contains(1.999) is False

        results = dist.support.contains(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(results, np.array([False, True, True]))

    def test_support_follows_the_scale_parameter(self):
        """Two distributions with different x_m have different supports."""
        for x_m in (0.5, 7.0):
            support = self.pareto_family(x_m=x_m, alpha=1.0).support
            assert isinstance(support, ContinuousSupport)
            assert support.left == x_m

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def test_score_against_analytical_formula(self):
        """d ln f / d x_m = alpha / x_m, d ln f / d alpha = 1/alpha + ln x_m - ln x."""
        x_m, alpha = 2.0, 3.0
        dist = self.pareto_family(x_m=x_m, alpha=alpha)
        x = np.array([2.0, 2.5, 4.0, 50.0])

        grad = dist.family.score(dist.parametrization, x)

        expected = np.stack(
            [np.full_like(x, alpha / x_m), 1.0 / alpha + np.log(x_m) - np.log(x)], axis=-1
        )
        np.testing.assert_allclose(grad, expected, rtol=self.CALCULATION_PRECISION)

    @pytest.mark.parametrize("x_m, alpha", PARAMETER_SETS)
    def test_score_numerical_derivative(self, x_m, alpha):
        """Compare the analytical score with a central finite-difference gradient of lpdf."""
        x = np.array([1.001, 1.5, 3.0, 20.0]) * x_m
        dist = self.pareto_family(x_m=x_m, alpha=alpha)

        def lpdf(xm: float, a: float) -> np.ndarray:
            return np.log(a) + a * np.log(xm) - (a + 1.0) * np.log(x)

        eps = 1e-6
        # x_m stays below x, so the perturbed point is still inside the support.
        grad_xm = (lpdf(x_m + eps, alpha) - lpdf(x_m - eps, alpha)) / (2 * eps)
        grad_alpha = (lpdf(x_m, alpha + eps) - lpdf(x_m, alpha - eps)) / (2 * eps)

        grad = dist.family.score(dist.parametrization, x)

        np.testing.assert_allclose(grad[:, 0], grad_xm, rtol=1e-5)
        np.testing.assert_allclose(grad[:, 1], grad_alpha, rtol=1e-5, atol=1e-8)

    def test_score_is_consistent_with_the_family_lpdf(self):
        """The score equals the finite-difference gradient of the family's own lpdf."""
        x_m, alpha = 2.0, 3.0
        x = np.array([2.5, 4.0, 9.0])
        eps = 1e-6

        def family_lpdf(xm: float, a: float) -> np.ndarray:
            dist = self.pareto_family(x_m=xm, alpha=a)
            return dist.query_method(CharacteristicName.LPDF)(x)

        grad_xm = (family_lpdf(x_m + eps, alpha) - family_lpdf(x_m - eps, alpha)) / (2 * eps)
        grad_alpha = (family_lpdf(x_m, alpha + eps) - family_lpdf(x_m, alpha - eps)) / (2 * eps)

        dist = self.pareto_family(x_m=x_m, alpha=alpha)
        grad = dist.family.score(dist.parametrization, x)

        np.testing.assert_allclose(grad[:, 0], grad_xm, rtol=1e-5)
        np.testing.assert_allclose(grad[:, 1], grad_alpha, rtol=1e-5)

    def test_score_with_integer_x(self):
        """The score is float even for integer x."""
        dist = self.pareto_family(x_m=2.0, alpha=3.0)
        x_int = np.array([2, 3, 4], dtype=int)

        grad = dist.family.score(dist.parametrization, x_int)

        assert grad.dtype == np.float64
        np.testing.assert_allclose(grad[:, 0], np.full(3, 1.5), rtol=self.CALCULATION_PRECISION)

    def test_score_shape(self):
        """The score has shape (..., 2)."""
        dist = self.pareto_family(x_m=2.0, alpha=3.0)
        x = np.array([2.0, 3.5, 5.0])

        grad = dist.family.score(dist.parametrization, x)

        assert grad.shape == (len(x), 2)
        assert grad.dtype == float

    def test_score_at_the_support_start(self):
        """x = x_m lies in the (closed) support, so the score is defined there."""
        dist = self.pareto_family(x_m=2.0, alpha=3.0)

        grad = dist.family.score(dist.parametrization, np.array([2.0]))

        np.testing.assert_allclose(grad, [[1.5, 1.0 / 3.0]], rtol=self.CALCULATION_PRECISION)

    @pytest.mark.parametrize("bad_x", [np.inf, -np.inf, np.nan])
    def test_score_raises_for_nonfinite_x(self, bad_x):
        """The score raises ValueError for non-finite x."""
        dist = self.pareto_family(x_m=2.0, alpha=3.0)
        with pytest.raises(ValueError, match="non-finite"):
            dist.family.score(dist.parametrization, np.array([3.0, bad_x]))

    @pytest.mark.parametrize("bad_x", [1.999999, 0.0, -1.0])
    def test_score_raises_for_x_outside_support(self, bad_x):
        """The score raises ValueError for x < x_m."""
        dist = self.pareto_family(x_m=2.0, alpha=3.0)
        with pytest.raises(ValueError, match="outside"):
            dist.family.score(dist.parametrization, np.array([3.0, bad_x]))


class TestParetoFamilyEdgeCases(BaseDistributionTest):
    """Edge cases of the Pareto family."""

    def setup_method(self):
        """Setup before each test method."""
        self.pareto_family = configure_families_register().get(FamilyName.PARETO)

    def test_invalid_parametrization_name(self):
        """An unknown parametrization name is rejected."""
        with pytest.raises(KeyError):
            self.pareto_family.distribution(parametrization_name="invalid_name", x_m=1.0, alpha=1.0)

    def test_missing_parameters(self):
        """Missing parameters are rejected."""
        with pytest.raises(TypeError):
            self.pareto_family.distribution(x_m=1.0)

        with pytest.raises(TypeError):
            self.pareto_family.distribution(alpha=1.0)

    def test_very_small_and_very_large_parameters(self):
        """Extreme but valid parameters keep the characteristics finite where they should be."""
        dist = self.pareto_family(x_m=1e-8, alpha=1e-3)

        cdf = dist.query_method(CharacteristicName.CDF)(np.array([1e-8, 1.0]))

        assert cdf[0] == 0.0
        assert 0.0 < cdf[1] < 1.0

    def test_registration_is_idempotent(self):
        """Configuring the register twice does not fail or duplicate the family."""
        from pysatl_core.families.builtins import configure_pareto_family

        configure_pareto_family()
        configure_pareto_family()

        assert configure_families_register().get(FamilyName.PARETO) is self.pareto_family
