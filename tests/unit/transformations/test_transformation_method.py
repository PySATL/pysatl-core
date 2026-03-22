from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import pytest

from pysatl_core.transformations.transformation_method import (
    ResolvedSourceMethods,
    TransformationMethod,
)
from pysatl_core.types import (
    CharacteristicName,
    ComputationFunc,
    Method,
    TransformationName,
)
from tests.unit.distributions.test_basic import DistributionTestBase

_BASE_ROLE = "base"


class TestTransformationMethod(DistributionTestBase):
    def test_try_from_parents_marks_non_analytical_for_mixed_sources(self) -> None:
        base = self.make_logistic_cdf_distribution()

        def _evaluator(sources: ResolvedSourceMethods) -> ComputationFunc[float, float]:
            base_cdf = cast(Method[float, float], sources[_BASE_ROLE][CharacteristicName.CDF])
            _ = sources[_BASE_ROLE][CharacteristicName.PDF]

            def _cdf(data: float, **options: Any) -> float:
                return float(base_cdf(data, **options))

            return cast(ComputationFunc[float, float], _cdf)

        method: TransformationMethod[float, float] | None
        method, is_analytical, has_any_present_source = TransformationMethod.try_from_parents(
            target=CharacteristicName.CDF,
            transformation=TransformationName.AFFINE,
            bases={_BASE_ROLE: base},
            source_requirements={
                _BASE_ROLE: (
                    CharacteristicName.CDF,
                    CharacteristicName.PDF,
                )
            },
            evaluator=_evaluator,
        )

        assert method is not None
        assert has_any_present_source
        assert not is_analytical
        assert method(0.0) == pytest.approx(0.5)

    def test_try_from_parents_skips_when_no_sources_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = self.make_logistic_cdf_distribution()

        def _unexpected_query(*_args: Any, **_kwargs: Any) -> Method[Any, Any]:
            raise AssertionError("query_method should not be called when no sources are present.")

        def _unused_evaluator(_sources: ResolvedSourceMethods) -> ComputationFunc[float, float]:
            def _zero(data: float, **_options: Any) -> float:
                _ = data
                return 0.0

            return cast(ComputationFunc[float, float], _zero)

        monkeypatch.setattr(base, "query_method", _unexpected_query)

        method: TransformationMethod[float, float] | None
        method, is_analytical, has_any_present_source = TransformationMethod.try_from_parents(
            target=CharacteristicName.PMF,
            transformation=TransformationName.AFFINE,
            bases={_BASE_ROLE: base},
            source_requirements={_BASE_ROLE: (CharacteristicName.PMF,)},
            evaluator=_unused_evaluator,
        )

        assert method is None
        assert not has_any_present_source
        assert not is_analytical
