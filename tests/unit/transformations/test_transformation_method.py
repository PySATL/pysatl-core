from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions.computations.computation import AnalyticalComputation
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.distributions.registry.graph import RegistryView
from pysatl_core.distributions.registry.graph_primitives import EdgeMeta
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.transformations.lightweight_distribution import LightweightDistribution
from pysatl_core.transformations.transformation_method import TransformationMethod
from pysatl_core.types import (
    CharacteristicName,
    ComputationFunc,
    GenericCharacteristicName,
    Method,
    NumericArray,
    ResolvedSourceMethods,
    TransformationName,
)
from tests.unit.distributions.test_basic import DistributionTestBase

_BASE_ROLE = "base"


class LastLoopStrategy(DefaultComputationStrategy):
    @staticmethod
    def _pick_loop_edge(state: GenericCharacteristicName, view: RegistryView) -> EdgeMeta | None:
        loops = tuple(view.variants(state, state).values())
        return loops[-1] if loops else None


class TestTransformationMethod(DistributionTestBase):
    @pytest.mark.parametrize(
        ("strategy", "expected_analytical", "expected_value"),
        [
            (DefaultComputationStrategy(), False, 1.0),
            (LastLoopStrategy(), True, 2.0),
        ],
    )
    def test_analytical_status_follows_selected_parent_label(
        self,
        strategy: DefaultComputationStrategy,
        expected_analytical: bool,
        expected_value: float,
    ) -> None:
        source = self.make_logistic_cdf_distribution()
        base = LightweightDistribution(
            distribution_type=source.distribution_type,
            entry_computations={
                CharacteristicName.CDF: {
                    "approximate": AnalyticalComputation(
                        target=CharacteristicName.CDF,
                        func=cast(ComputationFunc[float, float], lambda _x, **_options: 1.0),
                    ),
                    "exact": AnalyticalComputation(
                        target=CharacteristicName.CDF,
                        func=cast(ComputationFunc[float, float], lambda _x, **_options: 2.0),
                    ),
                }
            },
            computation_strategy=strategy,
            loop_analytical_flags={CharacteristicName.CDF: {"approximate": False, "exact": True}},
        )
        loops = (
            characteristic_registry()
            .view(base)
            .variants(CharacteristicName.CDF, CharacteristicName.CDF)
        )
        assert not loops["approximate"].is_analytical
        assert loops["exact"].is_analytical

        def evaluator(
            _owner: object, sources: ResolvedSourceMethods
        ) -> ComputationFunc[float, float]:
            return cast(ComputationFunc[float, float], sources[_BASE_ROLE][CharacteristicName.CDF])

        method = TransformationMethod[float, float](
            target=CharacteristicName.CDF,
            transformation=TransformationName.AFFINE,
            bases={_BASE_ROLE: base},
            source_requirements={_BASE_ROLE: (CharacteristicName.CDF,)},
            evaluator=evaluator,
            owner=object(),
        )

        assert method.is_analytical is expected_analytical
        assert method(0.0) == pytest.approx(expected_value)

    def test_analytical_status_uses_selected_label_when_labels_share_a_method(self) -> None:
        source = self.make_logistic_cdf_distribution()
        shared_method = AnalyticalComputation[float, float](
            target=CharacteristicName.CDF,
            func=cast(ComputationFunc[float, float], lambda _x, **_options: 2.0),
        )
        base = LightweightDistribution(
            distribution_type=source.distribution_type,
            entry_computations={
                CharacteristicName.CDF: {
                    "approximate": shared_method,
                    "exact": shared_method,
                }
            },
            computation_strategy=LastLoopStrategy(),
            loop_analytical_flags={CharacteristicName.CDF: {"approximate": False, "exact": True}},
        )

        def evaluator(
            _owner: object, sources: ResolvedSourceMethods
        ) -> ComputationFunc[float, float]:
            return cast(ComputationFunc[float, float], sources[_BASE_ROLE][CharacteristicName.CDF])

        method = TransformationMethod[float, float](
            target=CharacteristicName.CDF,
            transformation=TransformationName.AFFINE,
            bases={_BASE_ROLE: base},
            source_requirements={_BASE_ROLE: (CharacteristicName.CDF,)},
            evaluator=evaluator,
            owner=object(),
        )

        assert method.is_analytical
        assert method(0.0) == pytest.approx(2.0)

    def test_constructor_marks_non_analytical_for_mixed_sources(self) -> None:
        base = self.make_logistic_cdf_distribution()

        def _evaluator(
            _owner: object,
            sources: ResolvedSourceMethods,
        ) -> ComputationFunc[NumericArray, NumericArray]:
            base_cdf = cast(
                Method[NumericArray, NumericArray], sources[_BASE_ROLE][CharacteristicName.CDF]
            )
            _ = sources[_BASE_ROLE][CharacteristicName.PDF]

            def _cdf(data: NumericArray, **options: Any) -> NumericArray:
                return cast(NumericArray, np.asarray(base_cdf(data, **options), dtype=float))

            return cast(ComputationFunc[NumericArray, NumericArray], _cdf)

        method = TransformationMethod[NumericArray, NumericArray](
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
            owner=object(),
        )

        assert not method.is_analytical
        result = method(np.array([0.0]))
        assert float(np.asarray(result, dtype=float).flat[0]) == pytest.approx(0.5)

    def test_constructor_raises_when_no_sources_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = self.make_logistic_cdf_distribution()

        def _unexpected_query(*_args: Any, **_kwargs: Any) -> Method[Any, Any]:
            raise AssertionError("query_method should not be called when no sources are present.")

        def _unused_evaluator(
            _owner: object,
            _sources: ResolvedSourceMethods,
        ) -> ComputationFunc[float, float]:
            def _zero(data: float, **_options: Any) -> float:
                _ = data
                return 0.0

            return cast(ComputationFunc[float, float], _zero)

        monkeypatch.setattr(base, "query_method", _unexpected_query)

        with pytest.raises(
            ValueError,
            match="Transformation method requires at least one present source characteristic.",
        ):
            TransformationMethod[float, float](
                target=CharacteristicName.PMF,
                transformation=TransformationName.AFFINE,
                bases={_BASE_ROLE: base},
                source_requirements={_BASE_ROLE: (CharacteristicName.PMF,)},
                evaluator=_unused_evaluator,
                owner=object(),
            )
