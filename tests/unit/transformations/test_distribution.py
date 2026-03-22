from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.transformations.distribution import ApproximatedDistribution, DerivedDistribution
from pysatl_core.types import (
    DEFAULT_ANALYTICAL_COMPUTATION_LABEL,
    CharacteristicName,
    ComputationFunc,
    Kind,
    TransformationName,
)
from tests.unit.distributions.test_basic import DistributionTestBase


class TestDerivedDistribution(DistributionTestBase):
    def test_derived_distribution_is_abstract(self) -> None:
        base = self.make_logistic_cdf_distribution()
        derived_distribution_cls: Any = DerivedDistribution

        with pytest.raises(TypeError, match="abstract"):
            _ = derived_distribution_cls(
                distribution_type=base.distribution_type,
                bases={},
                analytical_computations=base.analytical_computations,
                transformation_name=TransformationName.AFFINE,
            )


class TestApproximatedDistribution(DistributionTestBase):
    def test_loops_are_never_analytical(self) -> None:
        def _cdf(data: float, **_options: Any) -> float:
            return data

        source = self.make_logistic_cdf_distribution()
        approx = ApproximatedDistribution(
            distribution_type=source.distribution_type,
            analytical_computations={
                CharacteristicName.CDF: AnalyticalComputation[float, float](
                    target=CharacteristicName.CDF,
                    func=cast(ComputationFunc[float, float], _cdf),
                )
            },
            support=source.support,
        )

        assert getattr(approx.distribution_type, "kind", None) == Kind.CONTINUOUS
        assert not approx.loop_is_analytical(
            CharacteristicName.CDF, DEFAULT_ANALYTICAL_COMPUTATION_LABEL
        )

        view = characteristic_registry().view(approx)
        cdf_loop = view.variants(CharacteristicName.CDF, CharacteristicName.CDF)[
            DEFAULT_ANALYTICAL_COMPUTATION_LABEL
        ]
        assert cdf_loop.edge_kind() == "transformation_loop"
        assert not cdf_loop.is_analytical
