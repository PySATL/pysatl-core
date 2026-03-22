from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import gc
import weakref

import pytest

from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.transformations.lightweight_distribution import LightweightDistribution
from pysatl_core.transformations.operations import affine
from pysatl_core.types import DEFAULT_ANALYTICAL_COMPUTATION_LABEL, CharacteristicName
from tests.unit.distributions.test_basic import DistributionTestBase

_BASE_ROLE = "base"


class TestLightweightBaseStorage(DistributionTestBase):
    def test_affine_stores_lightweight_base_distribution(self) -> None:
        base = self.make_logistic_cdf_distribution()
        transformed = affine(base, scale=2.0, shift=1.0)

        base_snapshot = transformed.bases[_BASE_ROLE]

        assert isinstance(base_snapshot, LightweightDistribution)
        assert isinstance(base_snapshot, Distribution)
        assert id(base_snapshot) != id(base)
        assert base_snapshot.computation_strategy is base.computation_strategy
        assert base_snapshot.sampling_strategy is base.sampling_strategy

        cdf = base_snapshot.query_method(CharacteristicName.CDF)
        assert cdf(0.0) == pytest.approx(0.5)

    def test_original_base_is_collectible_after_transformation(self) -> None:
        base = self.make_logistic_cdf_distribution()
        base_reference = weakref.ref(base)

        _ = affine(base, scale=1.5, shift=-0.5)

        del base
        gc.collect()

        assert base_reference() is None

    def test_lightweight_preserves_chain_of_bases(self) -> None:
        base = self.make_discrete_point_pmf_distribution()
        first = affine(base, scale=-1.0, shift=0.0)
        second = affine(first, scale=2.0, shift=1.0)

        first_snapshot = second.bases[_BASE_ROLE]
        assert isinstance(first_snapshot, LightweightDistribution)

        nested_base = first_snapshot.bases[_BASE_ROLE]
        assert isinstance(nested_base, LightweightDistribution)
        assert id(nested_base) != id(base)

    def test_loop_analytical_flags_survive_lightweight_snapshot(self) -> None:
        base = self.make_discrete_point_pmf_distribution()
        first = affine(base, scale=-1.0, shift=0.0)
        second = affine(first, scale=2.0, shift=1.0)

        view = characteristic_registry().view(second)
        cdf_loop = view.variants(CharacteristicName.CDF, CharacteristicName.CDF)[
            DEFAULT_ANALYTICAL_COMPUTATION_LABEL
        ]

        assert cdf_loop.edge_kind() == "transformation_loop"
        assert not cdf_loop.is_analytical
