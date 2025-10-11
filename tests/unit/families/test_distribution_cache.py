from __future__ import annotations

import pytest

from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import GenericCharacteristicName
from tests.unit.families.test_basic import TestBaseFamily


class TestAnalyticalComputationCache(TestBaseFamily):
    def _fallback_characteristics(self) -> dict[GenericCharacteristicName, dict[str, object]]:
        """Provide characteristics only for the base parametrization.

        Both characteristics return the same base parameter (``value``) to make
        assertions straightforward. The ``alt`` parametrization must therefore
        fallback to base for both PDF and CDF.
        """
        return {
            self.PDF: {"base": lambda params, x: params.value},
            self.CDF: {"base": lambda params, x: params.value},
        }

    def test_cache_auto_invalidation(self) -> None:
        """Mapping should be cached and auto-invalidated on parameter object/name changes."""
        family = self.make_default_family(distr_characteristics=self._fallback_characteristics())
        ParametricFamilyRegister.register(family)

        # alt(value=2.0) → Base(value=2.0) via transform_to_base_parametrization
        distribution = family.distribution("alt", value=2.0)

        computations1 = distribution.analytical_computations
        computations1_again = distribution.analytical_computations
        assert computations1 is computations1_again  # cache hit

        # Replacing with a *new* object of the same parametrization should rebuild the cache
        distribution.parameters = family.parametrizations["alt"](value=5.0)  # type: ignore[call-arg]
        computations2 = distribution.analytical_computations
        assert computations2 is not computations1

        # Switching to the base parametrization should also rebuild the cache
        distribution.parameters = family.parametrizations["base"](value=7.0)  # type: ignore[call-arg]
        computations3 = distribution.analytical_computations
        assert computations3 is not computations2

        # Both mappings must contain PDF and CDF and be callable
        for mapping in (computations2, computations3):
            assert self.PDF in mapping and self.CDF in mapping
            assert callable(mapping[self.PDF]) and callable(mapping[self.CDF])

        # For alt(value=5.0) → fallback to base(value=5.0)
        assert computations2[self.PDF](1.23) == pytest.approx(5.0)
        assert computations2[self.CDF](0.5) == pytest.approx(5.0)

        # For base(value=7.0)
        assert computations3[self.PDF](42.0) == pytest.approx(7.0)
        assert computations3[self.CDF](0.0) == pytest.approx(7.0)

    def test_fallback_to_base_for_missing_form(self) -> None:
        """Missing forms in current parametrization must be supplied from base."""
        family = self.make_default_family(distr_characteristics=self._fallback_characteristics())
        ParametricFamilyRegister.register(family)

        distribution = family.distribution("alt", value=2.0)
        computations = distribution.analytical_computations

        # Both characteristics are present via fallback
        assert self.PDF in computations and self.CDF in computations

        # For alt(value=2.0) → base(value=2.0)
        assert computations[self.PDF](1.23) == pytest.approx(2.0)
        assert computations[self.CDF](0.5) == pytest.approx(2.0)
