"""
Level 2: estimates against ``scipy.stats``.

SciPy is a second implementation, not an oracle, so the tolerances here are
looser than in the closed-form tests — most of all for the gamma family, where
both sides are running a numerical search and the comparison is numeric against
numeric.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from scipy import stats


class TestAgainstScipy:
    def test_normal(self, normal_family, rng):
        sample = rng.normal(-1.5, 2.5, 1000)
        loc, scale = stats.norm.fit(sample)
        result = normal_family.fit(sample)
        assert result.params.mu == pytest.approx(loc, rel=1e-10)
        assert result.params.sigma == pytest.approx(scale, rel=1e-10)

    def test_exponential(self, exponential_family, rng):
        sample = rng.exponential(2.0, 1000)
        _loc, scale = stats.expon.fit(sample, floc=0)
        result = exponential_family.fit(sample)
        assert result.params.lambda_ == pytest.approx(1.0 / scale, rel=1e-10)

    def test_uniform(self, uniform_family, rng):
        sample = rng.uniform(3.0, 8.0, 1000)
        loc, scale = stats.uniform.fit(sample)
        result = uniform_family.fit(sample)
        assert result.params.lower_bound == pytest.approx(loc, rel=1e-12)
        assert result.params.upper_bound == pytest.approx(loc + scale, rel=1e-12)

    def test_gamma(self, gamma_family, rng):
        sample = rng.gamma(2.5, 1.7, 2000)
        shape, _loc, scale = stats.gamma.fit(sample, floc=0)
        result = gamma_family.fit(sample)
        assert result.route == "numeric"
        assert result.success
        assert result.params.k == pytest.approx(shape, rel=1e-4)
        assert result.params.theta == pytest.approx(scale, rel=1e-4)

    def test_gamma_log_likelihood_is_not_worse_than_scipy(self, gamma_family, rng):
        sample = rng.gamma(2.5, 1.7, 2000)
        shape, _loc, scale = stats.gamma.fit(sample, floc=0)
        scipy_value = float(np.sum(stats.gamma.logpdf(sample, shape, loc=0, scale=scale)))
        result = gamma_family.fit(sample)
        assert result.log_likelihood >= scipy_value - 1e-6
