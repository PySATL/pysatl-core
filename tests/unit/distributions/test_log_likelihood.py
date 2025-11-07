from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_core.distributions.sampling import ArraySample
from tests.unit.distributions.test_basic import DistributionTestBase


class TestLogLikelihood(DistributionTestBase):
    def test_uniform_all_in_support_is_zero(self) -> None:
        distr = self.make_uniform_pdf_distribution()
        arr = np.array([[0.1], [0.9], [0.3]], dtype=np.float64)  # shape (n, 1)
        sample = ArraySample(arr)
        # log L = sum log(1) = 0
        assert distr.log_likelihood(sample) == pytest.approx(0.0, abs=1e-12)

    def test_uniform_out_of_support_is_minus_inf(self) -> None:
        distr = self.make_uniform_pdf_distribution()
        arr = np.array([[0.1], [1.5], [0.3]], dtype=np.float64)  # contains a point outside support
        sample = ArraySample(arr)
        assert np.isneginf(distr.log_likelihood(sample))
