from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_core.families import ParametricFamilyRegister
from tests.unit.families.test_basic import TestBaseFamily


class TestFamilyRegistrationAndSampling(TestBaseFamily):
    def test_family_registration_and_distribution_sampling(self) -> None:
        fam = self.make_default_family(
            distr_characteristics={
                self.PDF: {"base": lambda p, x: 1.0 if 0.0 <= x <= 1.0 else 0.0},
                self.CDF: {
                    "base": lambda p, x: x if 0.0 <= x <= 1.0 else (0.0 if x < 0.0 else 1.0)
                },
                self.PPF: {"base": lambda p, q: q},
            },
        )

        ParametricFamilyRegister.register(fam)

        distr = fam.distribution("base", value=0.0)

        n = 128
        sample = distr.sample(n)
        assert sample.shape == (n, 1)
        arr = sample.array
        assert (arr >= 0.0).all() and (arr <= 1.0).all()

        computations = distr.analytical_computations
        assert set(computations) == {self.PDF, self.CDF, self.PPF}
        assert computations[self.CDF](0.25) == pytest.approx(0.25)
        assert computations[self.PPF](0.75) == pytest.approx(0.75)
