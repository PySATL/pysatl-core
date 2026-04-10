from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import CharacteristicName, GenericCharacteristicName
from tests.unit.families.test_basic import TestBaseFamily


class TestAnalyticalComputationCache(TestBaseFamily):
    def _fallback_characteristics(
        self,
    ) -> dict[GenericCharacteristicName, dict[str, dict[str, object]]]:
        return {
            CharacteristicName.PDF: {"base": {"default": lambda params, x: params.value}},
            CharacteristicName.CDF: {"base": {"default": lambda params, x: params.value}},
        }

    def test_analytical_computations_are_built_at_distribution_creation(self) -> None:
        family = self.make_default_family(distr_characteristics=self._fallback_characteristics())
        ParametricFamilyRegister.register(family)

        # alt(value=2.0) → Base(value=2.0) via transform_to_base_parametrization
        distribution = family.distribution("alt", value=2.0)

        computations1 = distribution.analytical_computations
        computations1_again = distribution.analytical_computations
        assert computations1 is computations1_again

        # Replacing with a *new* object of the same parametrization does not
        # rebuild computations because they are now materialized at creation time.
        distribution._parametrization = family.parametrizations["alt"](value=5.0)  # type: ignore[call-arg]
        computations2 = distribution.analytical_computations
        assert computations2 is computations1

        # Switching to the base parametrization also does not rebuild computations.
        distribution._parametrization = family.parametrizations["base"](value=7.0)  # type: ignore[call-arg]
        computations3 = distribution.analytical_computations
        assert computations3 is computations2

        # Both mappings must contain PDF and CDF and be callable
        for mapping in (computations2, computations3):
            assert CharacteristicName.PDF in mapping and CharacteristicName.CDF in mapping
            assert callable(mapping[CharacteristicName.PDF]["default"])
            assert callable(mapping[CharacteristicName.CDF]["default"])

        # Computations remain bound to initial alt(value=2.0) -> base(value=2.0)
        assert computations2[CharacteristicName.PDF]["default"](1.23) == pytest.approx(2.0)
        assert computations2[CharacteristicName.CDF]["default"](0.5) == pytest.approx(2.0)

        assert computations3[CharacteristicName.PDF]["default"](42.0) == pytest.approx(2.0)
        assert computations3[CharacteristicName.CDF]["default"](0.0) == pytest.approx(2.0)

    def test_fallback_to_base_for_missing_form(self) -> None:
        family = self.make_default_family(distr_characteristics=self._fallback_characteristics())
        ParametricFamilyRegister.register(family)

        distribution = family.distribution("alt", value=2.0)
        computations = distribution.analytical_computations

        # Both characteristics are present via fallback
        assert CharacteristicName.PDF in computations and CharacteristicName.CDF in computations

        # For alt(value=2.0) → base(value=2.0)
        assert computations[CharacteristicName.PDF]["default"](1.23) == pytest.approx(2.0)
        assert computations[CharacteristicName.CDF]["default"](0.5) == pytest.approx(2.0)

    def test_distribution_creation_requires_analytical_computations(self) -> None:
        family = self.make_default_family(distr_characteristics={})
        ParametricFamilyRegister.register(family)

        with pytest.raises(
            ValueError, match="Distribution requires at least one analytical computation."
        ):
            family.distribution("alt", value=2.0)

    def test_family_creation_rejects_empty_labeled_providers(self) -> None:
        with pytest.raises(
            ValueError,
            match=("Characteristic 'pdf' has no labeled providers for parametrization 'base'."),
        ):
            self.make_default_family(
                distr_characteristics={
                    CharacteristicName.PDF: {"base": {}},
                }
            )
