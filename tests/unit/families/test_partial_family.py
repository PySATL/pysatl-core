from __future__ import annotations

__author__ = "Myznikov Fedor"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_core.families import ParametricFamilyRegister
from pysatl_core.types import CharacteristicName
from tests.unit.families.test_basic import TestBaseFamily


class TestPartialParametricFamily(TestBaseFamily):
    """Test the PartialParametricFamily view and its methods."""

    def test_view_creates_partial_family(self) -> None:
        """Test that ParametricFamily.view returns a PartialParametricFamily."""
        fam = self.make_default_family()
        partial = fam.view(lower_bound=0)
        assert partial._family is fam
        assert partial._fixed_params == {"lower_bound": 0}
        assert partial._fixed_in_param == fam.base_parametrization_name

    def test_view_with_explicit_parametrization(self) -> None:
        """Test view with a specific parametrization name."""
        fam = self.make_default_family()
        partial = fam.view(parametrization_name="alt", value=5.0)
        assert partial._fixed_in_param == "alt"
        assert partial._fixed_params == {"value": 5.0}

    def test_distribution_uses_fixed_parameters(self) -> None:
        """Test that distribution correctly applies fixed parameters."""
        fam = self.make_default_family(
            distr_characteristics={
                CharacteristicName.PDF: {"base": {"default": lambda p, x: p.value}},
            }
        )
        ParametricFamilyRegister.register(fam)

        # Fix 'value' in base parametrization
        partial = fam.view(value=2.0)
        dist = partial.distribution()
        assert dist.parametrization.value == 2.0  # type: ignore[attr-defined]

    def test_distribution_raises_on_conflicting_values(self) -> None:
        """Test that providing a different value for a fixed parameter raises error."""
        fam = self.make_default_family()
        partial = fam.view(value=2.0)

        with pytest.raises(ValueError, match="Parameter 'value' is fixed to 2.0, but got 5.0"):
            partial.distribution(value=5.0)

    def test_distribution_raises_on_different_parametrization(self) -> None:
        """Test that using a different parametrization than the fixed one raises error."""
        fam = self.make_default_family()
        partial = fam.view(parametrization_name="base", value=2.0)

        with pytest.raises(
            ValueError,
            match="Fixed parameters are defined for parametrization 'base'. "
            "Cannot create distribution in 'alt'.",
        ):
            partial.distribution(parametrization_name="alt")

    def test_with_fixed_params_creates_new_view(self) -> None:
        """Test that with_fixed_params returns a new PartialParametricFamily."""
        fam = self.make_default_family()
        partial1 = fam.view(value=1.0)
        partial2 = partial1.with_fixed_params({"other": 2.0})

        assert partial2 is not partial1
        assert partial2._fixed_params == {"value": 1.0, "other": 2.0}
        assert partial2._fixed_in_param == partial1._fixed_in_param

    def test_with_fixed_params_overrides_existing(self) -> None:
        """Test that with_fixed_params can override previously fixed parameters."""
        fam = self.make_default_family()
        partial1 = fam.view(value=1.0)
        partial2 = partial1.with_fixed_params({"value": 3.0})

        assert partial2._fixed_params == {"value": 3.0}
        dist = partial2.distribution()
        assert dist.parametrization.value == 3.0  # type: ignore[attr-defined]

    def test_delegation_of_attributes(self) -> None:
        """Test that PartialParametricFamily delegates unknown attributes to family."""
        fam = self.make_default_family()
        partial = fam.view(value=0.0)

        # Properties from family should be accessible
        assert partial.name == fam.name
        assert partial.base is fam.base
        assert partial.parametrizations is fam.parametrizations

        # Methods like get_parametrization should work
        assert partial.get_parametrization("base") is fam.get_parametrization("base")

    def test_callable_interface(self) -> None:
        """Test that __call__ works as alias for distribution."""
        fam = self.make_default_family(
            distr_characteristics={
                CharacteristicName.PDF: {"base": {"default": lambda p, x: p.value}},
            }
        )
        ParametricFamilyRegister.register(fam)

        partial = fam.view(value=2.0)
        dist = partial()  # call without arguments
        assert dist.parametrization.value == 2.0  # type: ignore[attr-defined]

        # Calling with a different value for a fixed parameter should raise error
        with pytest.raises(ValueError, match="Parameter 'value' is fixed to 2.0, but got 3.0"):
            partial(value=3.0)

    def test_distribution_passes_sampling_and_computation_strategies(self) -> None:
        """Test that distribution forwards sampling_strategy and computation_strategy."""
        from pysatl_core.distributions.strategies import DefaultComputationStrategy
        from pysatl_core.sampling.default import DefaultSamplingUnivariateStrategy

        fam = self.make_default_family(
            distr_characteristics={
                CharacteristicName.PDF: {"base": {"default": lambda p, x: p.value}},
            }
        )
        partial = fam.view(value=1.0)

        sampling = DefaultSamplingUnivariateStrategy()
        computation = DefaultComputationStrategy()

        dist = partial.distribution(sampling_strategy=sampling, computation_strategy=computation)
        assert dist.sampling_strategy is sampling
        assert dist.computation_strategy is computation

    def test_unknown_parameter_raises_type_error(self) -> None:
        """Test that providing an unknown parameter name raises TypeError
        (because parametrization class __init__ rejects it)."""
        fam = self.make_default_family()
        partial = fam.view()  # no fixed params

        with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
            partial.distribution(unknown=42)
