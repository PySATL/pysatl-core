from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from tests.unit.distributions.test_basic import DistributionTestBase


class TestSampling(DistributionTestBase):
    def test_sample_uniform_ppf_only_shape_bounds_and_mean(self) -> None:
        distr = self.make_uniform_ppf_distribution()

        n = 1000
        samples = distr.sample(n)

        assert isinstance(samples, np.ndarray)

        flat = np.asarray(samples, dtype=np.float64).reshape(-1)
        assert flat.shape == (n,)

        assert np.isfinite(flat).all()
        assert ((flat >= 0.0) & (flat <= 1.0)).all()

        mean = float(flat.mean())
        assert mean == pytest.approx(0.5, abs=0.1)
