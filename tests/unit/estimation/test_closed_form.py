"""
Level 1: closed-form estimates against the analytical formulas.

These are the tightest checks in the suite — the estimator is compared with the
textbook expression, not with another implementation — so they run at machine
precision.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_core.estimation import MLEResult


class TestNormalClosedForm:
    def test_mu_is_the_sample_mean(self, normal_family, rng):
        sample = rng.normal(3.0, 2.0, 500)
        result = normal_family.fit(sample)
        assert result.params.mu == pytest.approx(sample.mean())

    def test_sigma_divides_by_n_not_n_minus_one(self, normal_family, rng):
        sample = rng.normal(3.0, 2.0, 500)
        result = normal_family.fit(sample)
        expected = np.sqrt(np.mean((sample - sample.mean()) ** 2))
        assert result.params.sigma == pytest.approx(expected)
        # The unbiased estimator is a different number; the test would pass
        # vacuously if the two were indistinguishable at this sample size.
        assert result.params.sigma != pytest.approx(sample.std(ddof=1), rel=1e-12)

    def test_reports_the_closed_form_branch(self, normal_family, rng):
        result = normal_family.fit(rng.normal(0.0, 1.0, 100))
        assert isinstance(result, MLEResult)
        assert result.method == "closed_form"
        assert result.optimizer is None
        assert result.success
        assert result.n_params == 2
        assert result.n_observations == 100

    def test_fixed_mu_leaves_sigma_about_that_point(self, normal_family, rng):
        sample = rng.normal(3.0, 2.0, 400)
        result = normal_family.view(mu=0.0).fit(sample)
        assert result.params.sigma == pytest.approx(np.sqrt(np.mean(sample**2)))
        assert result.n_params == 1
        assert result.method == "closed_form"

    def test_fixed_sigma_leaves_mu_at_the_sample_mean(self, normal_family, rng):
        sample = rng.normal(3.0, 2.0, 400)
        result = normal_family.view(sigma=1.0).fit(sample)
        assert result.params.mu == pytest.approx(sample.mean())
        assert result.n_params == 1


class TestUniformClosedForm:
    def test_estimates_are_the_sample_extremes(self, uniform_family, rng):
        sample = rng.uniform(-2.0, 7.0, 500)
        result = uniform_family.fit(sample)
        assert result.params.lower_bound == sample.min()
        assert result.params.upper_bound == sample.max()
        assert result.method == "closed_form"

    def test_fixed_lower_bound_leaves_the_sample_maximum(self, uniform_family, rng):
        sample = rng.uniform(0.5, 4.0, 300)
        result = uniform_family.view(lower_bound=0.0).fit(sample)
        assert result.params.upper_bound == sample.max()
        assert result.n_params == 1

    def test_fixed_upper_bound_leaves_the_sample_minimum(self, uniform_family, rng):
        sample = rng.uniform(0.5, 4.0, 300)
        result = uniform_family.view(upper_bound=9.0).fit(sample)
        assert result.params.lower_bound == sample.min()


class TestExponentialClosedForm:
    def test_rate_is_the_reciprocal_mean(self, exponential_family, rng):
        sample = rng.exponential(1 / 3.0, 400)
        result = exponential_family.fit(sample)
        assert result.params.lambda_ == pytest.approx(1.0 / sample.mean())
        assert result.method == "closed_form"
        assert result.n_params == 1


class TestLogLikelihoodAndCriteria:
    def test_log_likelihood_matches_the_summed_log_density(self, normal_family, rng):
        sample = rng.normal(1.0, 2.0, 200)
        result = normal_family.fit(sample)
        mu, sigma = result.params.mu, result.params.sigma
        expected = np.sum(
            -0.5 * np.log(2 * np.pi) - np.log(sigma) - ((sample - mu) ** 2) / (2 * sigma**2)
        )
        assert result.log_likelihood == pytest.approx(expected)

    def test_information_criteria(self, normal_family, rng):
        sample = rng.normal(1.0, 2.0, 200)
        result = normal_family.fit(sample)
        assert result.aic == pytest.approx(2 * 2 - 2 * result.log_likelihood)
        assert result.bic == pytest.approx(2 * np.log(200) - 2 * result.log_likelihood)

    def test_a_view_estimates_one_parameter_fewer(self, normal_family, rng):
        """The deterministic half: AIC is exactly ``2k - 2l`` for both models."""
        sample = rng.normal(0.0, 2.0, 200)
        full = normal_family.fit(sample)
        pinned = normal_family.view(mu=0.0).fit(sample)
        assert pinned.n_params == full.n_params - 1
        assert pinned.aic - full.aic == pytest.approx(
            2 * (pinned.n_params - full.n_params)
            - 2 * (pinned.log_likelihood - full.log_likelihood)
        )

    def test_aic_prefers_the_smaller_model_at_equal_likelihood(self, normal_family, rng):
        """Model selection on data where the conclusion is guaranteed by construction.

        Pinning mu at the *true* value and expecting AIC to prefer the smaller
        model is not a property: ``pinned.aic < full.aic`` is equivalent to
        ``2(l_full - l_pinned) < 2``, and that statistic is chi-squared with one
        degree of freedom, so it fails about 16% of the time — at *every* sample
        size, since the distribution does not depend on n.  Pinning mu at the
        estimate instead costs no likelihood at all and saves a parameter, so
        AIC must prefer it on every sample.
        """
        sample = rng.normal(0.0, 2.0, 200)
        full = normal_family.fit(sample)
        pinned = normal_family.view(mu=float(sample.mean())).fit(sample)
        assert pinned.log_likelihood == pytest.approx(full.log_likelihood)
        assert pinned.n_params == full.n_params - 1
        assert pinned.aic < full.aic


class TestResultDistribution:
    def test_builds_a_usable_distribution(self, normal_family, rng):
        sample = rng.normal(5.0, 1.0, 300)
        result = normal_family.fit(sample)
        distribution = result.distribution
        assert distribution.parametrization.parameters == result.params.parameters
        density = distribution.calculate_characteristic("pdf", np.array([5.0]))
        assert density[0] > 0.0

    def test_builds_a_distribution_from_a_view(self, normal_family, rng):
        sample = rng.normal(0.0, 3.0, 300)
        result = normal_family.view(mu=0.0).fit(sample)
        distribution = result.distribution
        assert distribution.parametrization.parameters == {"sigma": result.params.sigma}
