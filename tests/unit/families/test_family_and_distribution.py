from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest

from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import CharacteristicName
from tests.unit.families.test_basic import TestBaseFamily


class TestFamilyRegistrationAndSampling(TestBaseFamily):
    def test_family_registration_and_distribution_sampling(self) -> None:
        fam = self.make_default_family(
            distr_characteristics={
                CharacteristicName.PDF: {"base": lambda p, x: 1.0 if 0.0 <= x <= 1.0 else 0.0},
                CharacteristicName.CDF: {
                    "base": lambda p, x: x if 0.0 <= x <= 1.0 else (0.0 if x < 0.0 else 1.0)
                },
                CharacteristicName.PPF: {"base": lambda p, q: q},
            },
        )

        ParametricFamilyRegister.register(fam)

        distr = fam.distribution("base", value=0.0)

        n = 128
        samples = distr.sample(n)

        assert isinstance(samples, np.ndarray)

        flat = np.asarray(samples, dtype=np.float64).reshape(-1)
        assert flat.shape == (n,)

        assert ((flat >= 0.0) & (flat <= 1.0)).all()

        computations = distr.analytical_computations
        assert set(computations) == {
            CharacteristicName.PDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
        }
        assert computations[CharacteristicName.CDF](0.25) == pytest.approx(0.25)
        assert computations[CharacteristicName.PPF](0.75) == pytest.approx(0.75)
