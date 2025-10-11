from __future__ import annotations

import numpy as np
import pytest

from tests.unit.distributions.test_basic import DistributionTestBase


class TestSampling(DistributionTestBase):
    def test_sample_uniform_ppf_only_shape_bounds_and_mean(self) -> None:
        distr = self.make_uniform_ppf_distribution()

        n = 1000
        sample = distr.sample(n)

        assert sample.shape == (n, 1)
        arr = sample.array
        assert np.isfinite(arr).all()
        assert ((arr >= 0.0) & (arr <= 1.0)).all()

        mean = float(arr.mean())
        assert mean == pytest.approx(0.5, abs=0.1)
