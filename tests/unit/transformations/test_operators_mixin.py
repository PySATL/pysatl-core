from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import pytest

from pysatl_core.distributions.distribution import Distribution
from pysatl_core.families.distribution import ParametricFamilyDistribution
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.transformations import TransformationOperatorsMixin
from pysatl_core.transformations.distribution import DerivedDistribution
from pysatl_core.transformations.operations import affine
from pysatl_core.types import CharacteristicName
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.unit.families.test_basic import TestBaseFamily


class TestTransformationOperatorsMixin(DistributionTestBase, TestBaseFamily):
    def test_distribution_does_not_inherit_transformation_operator_mixin(self) -> None:
        assert not issubclass(Distribution, TransformationOperatorsMixin)

    def test_derived_distribution_inherits_transformation_operator_mixin(self) -> None:
        assert issubclass(DerivedDistribution, TransformationOperatorsMixin)

    def test_parametric_family_distribution_inherits_transformation_operator_mixin(self) -> None:
        assert issubclass(ParametricFamilyDistribution, TransformationOperatorsMixin)

    def test_base_distribution_without_mixin_rejects_affine_operators(self) -> None:
        base = self.make_logistic_cdf_distribution()
        with pytest.raises(TypeError):
            _ = cast(Any, base) + 2.0

    def test_parametric_family_distribution_supports_affine_operators(self) -> None:
        family = self.make_default_family()
        ParametricFamilyRegister.register(family)
        distribution = family.distribution("base", value=0.0)

        shifted = distribution + 2.0
        assert isinstance(shifted, DerivedDistribution)

        shifted_cdf = shifted.query_method(CharacteristicName.CDF)
        assert shifted_cdf(2.0) == pytest.approx(0.0)

    def test_derived_distribution_supports_affine_operators(self) -> None:
        base = self.make_logistic_cdf_distribution()
        transformed = affine(base, scale=2.0, shift=1.0)

        scaled = transformed * 3.0
        assert isinstance(scaled, DerivedDistribution)

        scaled_cdf = scaled.query_method(CharacteristicName.CDF)
        assert scaled_cdf(3.0) == pytest.approx(0.5)

    def test_unsupported_operand_raises_type_error(self) -> None:
        base = self.make_logistic_cdf_distribution()
        transformed = affine(base, scale=1.0, shift=0.0)
        with pytest.raises(TypeError):
            _ = transformed + "bad"

    def test_division_by_zero_raises_error(self) -> None:
        base = self.make_logistic_cdf_distribution()
        transformed = affine(base, scale=1.0, shift=0.0)
        with pytest.raises(ZeroDivisionError, match="Cannot divide a distribution by zero."):
            _ = transformed / 0.0
