"""
Error handling: which situations raise, and which are reported in the result.

The dividing line is recoverability.  A sample that cannot be fitted at all, or
data no parameter value could have produced, raises.  An optimizer that fails
to converge does not — that is a result carrying ``success=False``, so a caller
can tell "did not converge" from "wrong data" without reading exception text.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_core.estimation import (
    FitDataError,
    InsufficientDataError,
    MLEError,
    validate_sample,
)


class TestSampleValidation:
    def test_a_two_dimensional_sample_is_rejected(self, normal_family, rng):
        with pytest.raises(ValueError, match="one-dimensional"):
            normal_family.fit(rng.normal(0.0, 1.0, (10, 10)))

    def test_a_scalar_is_rejected(self, normal_family):
        with pytest.raises(ValueError, match="one-dimensional"):
            normal_family.fit(np.float64(1.0))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_observation_is_rejected(self, normal_family, bad):
        with pytest.raises(ValueError, match="must be finite"):
            normal_family.fit(np.array([1.0, 2.0, bad, 4.0]))

    def test_the_message_counts_the_offenders(self, normal_family):
        with pytest.raises(ValueError, match="2 NaN and 1 infinite"):
            normal_family.fit(np.array([1.0, np.nan, np.nan, np.inf]))

    def test_too_few_observations(self, normal_family):
        with pytest.raises(InsufficientDataError, match="2 free parameter"):
            normal_family.fit(np.array([1.0]))

    def test_a_view_needs_fewer_observations(self, normal_family):
        result = normal_family.view(mu=0.0).fit(np.array([2.0]))
        assert result.params.sigma == pytest.approx(2.0)

    def test_insufficient_data_is_also_a_value_error(self, normal_family):
        assert issubclass(InsufficientDataError, ValueError)
        assert issubclass(InsufficientDataError, MLEError)

    def test_validate_sample_returns_a_float_array(self, normal_family):
        out = validate_sample(normal_family, np.array([1, 2, 3]))
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])


class TestFixedSupport:
    def test_a_negative_value_cannot_be_fitted_by_gamma(self, gamma_family):
        with pytest.raises(FitDataError, match="does not depend on its parameters"):
            gamma_family.fit(np.array([-1.0, 1.0, 2.0, 3.0]))

    def test_a_negative_value_cannot_be_fitted_by_the_exponential(self, exponential_family):
        with pytest.raises(FitDataError, match="outside the support"):
            exponential_family.fit(np.array([-0.5, 1.0, 2.0]))

    def test_the_message_names_an_offending_point(self, gamma_family):
        with pytest.raises(FitDataError, match=r"-1\.0"):
            gamma_family.fit(np.array([-1.0, 1.0, 2.0]))

    def test_a_moving_support_charges_a_penalty_instead(self, uniform_family, rng):
        sample = rng.uniform(-100.0, 100.0, 50)
        assert uniform_family.fit(sample).success

    def test_fit_data_error_is_also_a_value_error(self):
        assert issubclass(FitDataError, ValueError)
        assert issubclass(FitDataError, MLEError)


class TestFixedParametersContradictingData:
    def test_a_fixed_lower_bound_above_the_minimum(self, uniform_family, rng):
        sample = rng.uniform(2.0, 5.0, 200)
        with pytest.raises(FitDataError, match="lower_bound is fixed"):
            uniform_family.view(lower_bound=3.0).fit(sample)

    def test_a_fixed_upper_bound_below_the_maximum(self, uniform_family, rng):
        sample = rng.uniform(2.0, 5.0, 200)
        with pytest.raises(FitDataError, match="upper_bound is fixed"):
            uniform_family.view(upper_bound=4.0).fit(sample)

    def test_an_all_zero_exponential_sample_has_no_finite_estimate(self, exponential_family):
        with pytest.raises(FitDataError, match="no finite estimate exists"):
            exponential_family.fit(np.zeros(10))


class TestParametrizationConversion:
    def test_a_non_base_parametrization_is_refused_with_an_explanation(self, normal_family, rng):
        with pytest.raises(NotImplementedError, match="inverse transform"):
            normal_family.fit(rng.normal(0.0, 1.0, 50), parametrization="meanVar")

    def test_the_base_parametrization_is_accepted_explicitly(self, normal_family, rng):
        sample = rng.normal(0.0, 1.0, 50)
        result = normal_family.fit(sample, parametrization="meanStd")
        assert result.params.name == "meanStd"


class TestNonConvergenceIsNotAnException:
    def test_a_failed_optimizer_yields_a_result(self, gamma_family, rng):
        sample = rng.gamma(3.0, 2.0, 400)
        result = gamma_family.fit(sample, optimizer="L-BFGS-B", options={"maxiter": 1})
        assert not result.success
        assert result.message
        assert result.method == "numeric"


class TestStartingPointFailure:
    def test_an_unusable_start_is_reported_clearly(self, make_boxed_family, rng):
        family = make_boxed_family("BoxedUnusableStart")
        with (
            pytest.warns(UserWarning, match="declares no 'param_bounds'"),
            pytest.raises(MLEError, match="not finite at the starting point"),
        ):
            family.fit(rng.uniform(2.0, 5.0, 100))
