from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.transformations.operations import affine
from pysatl_core.types import DEFAULT_ANALYTICAL_COMPUTATION_LABEL, CharacteristicName
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import MockSamplingStrategy


class TestAffineDistribution(DistributionTestBase):
    def test_negative_scale_marks_cdf_as_transformation_loop(self) -> None:
        base = self.make_discrete_point_pmf_distribution()
        transformed = affine(base, scale=-1.0, shift=0.0)

        view = characteristic_registry().view(transformed)
        cdf_loop = view.variants(CharacteristicName.CDF, CharacteristicName.CDF)[
            DEFAULT_ANALYTICAL_COMPUTATION_LABEL
        ]
        pmf_loop = view.variants(CharacteristicName.PMF, CharacteristicName.PMF)[
            DEFAULT_ANALYTICAL_COMPUTATION_LABEL
        ]

        assert cdf_loop.edge_kind() == "transformation_loop"
        assert not cdf_loop.is_analytical
        assert pmf_loop.edge_kind() == "analytical_loop"
        assert pmf_loop.is_analytical
        assert view.analytical_variants(CharacteristicName.CDF) == {}

        cdf = transformed.query_method(CharacteristicName.CDF)
        assert cdf(0.0) == pytest.approx(1.0)
        assert cdf(-2.0) == pytest.approx(0.3)

    def test_includes_only_present_base_characteristics(self) -> None:
        base = self.make_logistic_cdf_distribution()
        transformed = affine(base, scale=2.0, shift=1.0)

        assert set(transformed.analytical_computations) == {CharacteristicName.CDF}
        view = characteristic_registry().view(transformed)
        cdf_loop = view.variants(CharacteristicName.CDF, CharacteristicName.CDF)[
            DEFAULT_ANALYTICAL_COMPUTATION_LABEL
        ]
        assert cdf_loop.is_analytical

    def test_with_sampling_strategy_preserves_concrete_type(self) -> None:
        base = self.make_logistic_cdf_distribution()
        transformed = affine(base, scale=2.0, shift=1.0)
        sampling_strategy = MockSamplingStrategy()

        clone = transformed.with_sampling_strategy(sampling_strategy)

        assert type(clone) is type(transformed)
        assert clone.sampling_strategy is sampling_strategy
        assert clone.scale == pytest.approx(transformed.scale)
        assert clone.shift == pytest.approx(transformed.shift)

    def test_with_computation_strategy_preserves_concrete_type(self) -> None:
        base = self.make_logistic_cdf_distribution()
        transformed = affine(base, scale=2.0, shift=1.0)
        computation_strategy = DefaultComputationStrategy(enable_caching=True)

        clone = transformed.with_computation_strategy(computation_strategy)

        assert type(clone) is type(transformed)
        assert clone.computation_strategy is computation_strategy
        assert clone.scale == pytest.approx(transformed.scale)
        assert clone.shift == pytest.approx(transformed.shift)

    def test_sample_uses_base_distribution_and_affine_transform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = self.make_uniform_ppf_distribution()
        transformed = affine(base, scale=2.0, shift=-1.0)
        captured: dict[str, object] = {}

        def _fake_sample(n: int, distr: object, **options: object) -> object:
            captured["n"] = n
            captured["distr"] = distr
            captured["options"] = options
            return [0.0, 0.5, 1.0]

        monkeypatch.setattr(transformed.sampling_strategy, "sample", _fake_sample)
        samples = transformed.sample(3, token="test")

        assert captured["n"] == 3
        assert captured["distr"] is transformed.base_distribution
        assert captured["options"] == {"token": "test"}
        assert samples == pytest.approx([-1.0, 0.0, 1.0])
