from __future__ import annotations

__author__ = "Myznikov Fedor"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.families.parametric_family import ParametricFamily, PartialParametricFamily
from pysatl_core.families.parametrizations import Parametrization
from pysatl_core.sampling.default import DefaultSamplingUnivariateStrategy
from pysatl_core.types import (
    CharacteristicName,
    DistributionType,
    UnivariateContinuous,
    UnivariateDiscrete,
)
from tests.unit.families.test_basic import TestBaseFamily


@dataclass
class TwoParam(Parametrization):
    a: float
    b: float

    def transform_to_base_parametrization(self) -> Parametrization:
        return self


TwoParam.__param_name__ = "base"


@dataclass
class AltParam(Parametrization):
    mean: float
    var: float

    def transform_to_base_parametrization(self) -> TwoParam:
        return TwoParam(a=self.mean, b=self.var)


AltParam.__param_name__ = "alt"


@dataclass
class TypeSwitchingParams(Parametrization):
    a: float
    b: float

    def transform_to_base_parametrization(self) -> Parametrization:
        return self


TypeSwitchingParams.__param_name__ = "base"


class TestPartialParametricFamily(TestBaseFamily):
    """Tests for PartialParametricFamily view with partially fixed parameters."""

    def _make_two_param_family(self) -> ParametricFamily:
        fam = ParametricFamily(
            name="TwoParamFamily",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["base"],
            distr_characteristics={
                CharacteristicName.PDF: {"base": {"default": lambda p, x: p.a + p.b}},
                CharacteristicName.CDF: {"base": {"default": lambda p, x: p.a}},
            },
        )
        fam.register_parametrization("base", TwoParam)
        return fam

    def test_view_returns_partial_family(self) -> None:
        fam = self.make_default_family()
        partial = fam.view()
        assert isinstance(partial, PartialParametricFamily)
        assert partial._base_family is fam
        assert partial._fixed_params == {}
        assert partial._fixed_in_param == fam.base_parametrization_name

    def test_view_with_explicit_parametrization(self) -> None:
        fam = self.make_default_family()
        partial = fam.view(parametrization_name="alt")
        assert partial._fixed_in_param == "alt"
        assert partial._fixed_params == {}

    def test_view_with_partial_fixation(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        assert partial._fixed_params == {"a": 2.0}
        assert partial._fixed_in_param == "base"

    def test_full_fixation_raises(self) -> None:
        fam = self._make_two_param_family()
        with pytest.raises(
            ValueError, match="All parameters of parametrization 'base' are already fixed"
        ):
            fam.view(a=1.0, b=2.0)

    def test_unknown_parameter_raises(self) -> None:
        fam = self._make_two_param_family()
        with pytest.raises(ValueError, match="Unknown parameters for parametrization 'base':"):
            fam.view(a=1.0, unknown=42)

    # Properties (parent_family, fixed/free parameter info)
    def test_parent_family(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        assert partial.parent_family is fam

    def test_fixed_parameters(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        assert partial.fixed_parameters == {"a": 2.0}
        # MappingProxyType does not allow modification
        with pytest.raises(TypeError):
            partial.fixed_parameters["a"] = 3.0  # type: ignore[index]

    def test_fixed_parameter_names(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        assert partial.fixed_parameter_names == frozenset({"a"})

    def test_free_parameter_names(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        assert partial.free_parameter_names == ("b",)

    # distribution()
    def test_distribution_uses_fixed_parameters(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        dist = partial.distribution(b=3.0)
        params = cast(Any, dist.parametrization)
        assert params.b == 3.0
        assert partial.fixed_parameters == {"a": 2.0}

    def test_distribution_raises_on_conflicting_values(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        with pytest.raises(ValueError, match="Parameter 'a' is fixed to 2.0, but got 5.0"):
            partial.distribution(a=5.0, b=1.0)

    def test_distribution_rejects_fixed_param_even_if_correct_value(self) -> None:
        """Passing a fixed parameter (even with the correct value) is not allowed
        because the lightweight parametrization class does not accept it."""
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        with pytest.raises(TypeError):
            partial.distribution(a=2.0, b=3.0)

    def test_distribution_raises_on_different_parametrization(self) -> None:
        fam = self.make_default_family()
        partial = fam.view(parametrization_name="base")
        with pytest.raises(
            ValueError,
            match="Only parametrization 'base' is available in this view.",
        ):
            partial.distribution(parametrization_name="alt")

    def test_distribution_passes_sampling_and_computation_strategies(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=1.0)
        sampling = DefaultSamplingUnivariateStrategy()
        computation = DefaultComputationStrategy()
        dist = partial.distribution(
            b=2.0,
            sampling_strategy=sampling,
            computation_strategy=computation,
        )
        assert dist.sampling_strategy is sampling
        assert dist.computation_strategy is computation

    # Chaining view
    def test_view_on_view_creates_new_view_until_complete(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=1.0)
        assert partial._fixed_params == {"a": 1.0}
        with pytest.raises(
            ValueError, match="All parameters of parametrization 'base' are already fixed"
        ):
            partial.view(b=2.0)
        dist = partial.distribution(b=2.0)
        params = cast(Any, dist.parametrization)
        assert params.b == 2.0
        assert partial.fixed_parameters == {"a": 1.0}

    # Parametrization access (base, parametrizations, get_parametrization)
    def test_only_fixed_parametrization_visible(self) -> None:
        fam = self.make_default_family()
        partial = fam.view(parametrization_name="alt")
        assert partial.parametrization_names == ["alt"]
        assert partial.base_parametrization_name == "alt"
        assert list(partial.parametrizations.keys()) == ["alt"]
        # The registered class is the lightweight one, not the original
        assert partial.get_parametrization("alt") is partial.base
        with pytest.raises(KeyError, match="Parametrization 'base' is not available"):
            partial.get_parametrization("base")

    def test_get_parametrization_without_name(self) -> None:
        fam = self.make_default_family()
        partial = fam.view(parametrization_name="alt")
        assert partial.get_parametrization() is partial.base

    def test_base_is_free_param_class(self) -> None:
        """base returns a class that contains only the free fields."""
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        free_cls = partial.base
        fields = getattr(free_cls, "__dataclass_fields__", {})
        assert "b" in fields
        assert "a" not in fields

    # Inherited attributes
    def test_inherited_attributes_and_methods(self) -> None:
        fam = self.make_default_family()
        partial = fam.view()
        assert partial.name == fam.name
        assert partial.base is not None
        assert list(partial.parametrizations.keys()) == ["base"]
        assert partial.get_parametrization("base") is partial.base
        with pytest.raises(KeyError):
            partial.get_parametrization("alt")

    # __call__
    def test_callable_interface(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        dist = partial(b=3.0)
        params = cast(Any, dist.parametrization)
        assert params.b == 3.0
        assert partial.fixed_parameters == {"a": 2.0}
        with pytest.raises(ValueError, match="Parameter 'a' is fixed to 2.0, but got 4.0"):
            partial(a=4.0, b=1.0)

    # Analytical characteristics
    def test_analytical_characteristics_preserved(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=2.0)
        dist = partial.distribution(b=3.0)
        pdf_val = dist.calculate_characteristic(CharacteristicName.PDF, 0.0)
        assert pdf_val == 5.0

    def test_analytical_characteristics_with_non_base_fixation(self) -> None:
        fam = ParametricFamily(
            name="Test",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["base", "alt"],
            distr_characteristics={
                CharacteristicName.PDF: {
                    "base": {"default": lambda p, x: p.a + p.b},
                },
                CharacteristicName.CDF: {
                    "alt": {"default": lambda p, x: p.mean},
                },
            },
            support_by_parametrization=None,
            base_score=None,
        )
        fam.register_parametrization("base", TwoParam)
        fam.register_parametrization("alt", AltParam)

        partial = fam.view(parametrization_name="alt", mean=2.0)
        dist = partial.distribution(var=3.0)

        pdf_val = dist.calculate_characteristic(CharacteristicName.PDF, 0.0)
        assert pdf_val == 5.0

        cdf_val = dist.calculate_characteristic(CharacteristicName.CDF, 0.0)
        assert cdf_val == 2.0

    def test_characteristic_with_nullary_provider_is_wrapped_correctly(self) -> None:
        """A characteristic that does not accept a parametrization argument
        (e.g., skew) must work through the wrapper that uses _bind_parametrization."""
        fam = ParametricFamily(
            name="SkewFamily",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["base", "alt"],
            distr_characteristics={
                CharacteristicName.SKEW: {
                    "base": {"default": lambda *, excess=False: 0.0 if not excess else -3.0},
                },
            },
            support_by_parametrization=None,
            base_score=None,
        )
        fam.register_parametrization("base", TwoParam)
        fam.register_parametrization("alt", AltParam)

        partial = fam.view(parametrization_name="alt", mean=2.0)
        dist = partial.distribution(var=3.0)

        skew_func = dist.query_method(CharacteristicName.SKEW)
        val = skew_func()
        val_excess = skew_func(excess=True)
        assert val == 0.0
        assert val_excess == -3.0

    # distr_type wrapping
    def test_distr_type_wrapped_with_dynamic_type(self) -> None:
        def dynamic_type(params: Parametrization) -> DistributionType:
            p = cast(TypeSwitchingParams, params)
            return UnivariateContinuous if p.a > 0 else UnivariateDiscrete

        fam = ParametricFamily(
            name="DynamicType",
            distr_type=dynamic_type,
            distr_parametrizations=["base"],
            distr_characteristics={
                CharacteristicName.PDF: {"base": {"default": lambda p, x: p.a + p.b}},
            },
        )
        fam.register_parametrization("base", TypeSwitchingParams)

        partial = fam.view(a=2.0)
        dist = partial.distribution(b=3.0)
        assert dist.distribution_type == UnivariateContinuous

        partial_neg = fam.view(a=-1.0)
        dist_neg = partial_neg.distribution(b=3.0)
        assert dist_neg.distribution_type == UnivariateDiscrete

    # score / gradient_transform
    def test_score_returns_gradient_wrt_free_parameters_only(self) -> None:
        """score() must return an array whose last dimension equals the number of
        free parameters."""

        # Use a simple family with base_score computed via base parametrization.
        def base_score(params: Parametrization, x: np.ndarray) -> np.ndarray:
            cast(TwoParam, params)
            # Dummy gradient: shape (..., 2)
            return np.broadcast_to(np.array([1.0, 2.0]), (*x.shape, 2))

        fam = ParametricFamily(
            name="ScoreTest",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["base", "alt"],
            distr_characteristics={
                CharacteristicName.PDF: {"base": {"default": lambda p, x: np.zeros_like(x)}}
            },
            base_score=base_score,
        )
        fam.register_parametrization("base", TwoParam)
        fam.register_parametrization("alt", AltParam)

        partial = fam.view(parametrization_name="alt", mean=2.0)
        free_params = partial.distribution(var=3.0).parametrization
        grad = partial.score(free_params, np.array([0.5]))
        assert grad.shape == (1, 1)  # only 'var' is free
        np.testing.assert_array_almost_equal(grad, [[2.0]])

    def test_view_unknown_parametrization_raises_in_base(self) -> None:
        fam = self._make_two_param_family()
        with pytest.raises(ValueError, match="Unknown parametrization 'nonexistent'"):
            fam.view(parametrization_name="nonexistent")

    def test_unexpected_free_parameter_raises_type_error(self) -> None:
        fam = self._make_two_param_family()
        partial = fam.view(a=1.0)
        with pytest.raises(TypeError, match="got unexpected keyword argument"):
            partial.distribution(unknown=42)
