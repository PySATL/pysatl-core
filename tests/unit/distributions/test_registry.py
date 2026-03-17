from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from mypy_extensions import KwArg

from pysatl_core.distributions import strategies as strategies_module
from pysatl_core.distributions.computation import (
    AnalyticalComputation,
    ComputationMethod,
    FittedComputationMethod,
)
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.registry import (
    DEFAULT_COMPUTATION_KEY,
    CharacteristicRegistry,
    GraphInvariantError,
    GraphPrimitiveConstraint,
    NumericConstraint,
    characteristic_registry,
    reset_characteristic_registry,
)
from pysatl_core.distributions.strategies import DefaultComputationStrategy
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.types import CharacteristicName, Kind
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


class TestCharacteristicRegistry(DistributionTestBase):
    def setup_method(self) -> None:
        self.distr_example = self.make_logistic_cdf_distribution()

    def test_configuration_continuous_presence_and_connectivity(self) -> None:
        reg = characteristic_registry()
        distr = self.make_logistic_cdf_distribution()

        view = reg.view(distr)

        # Presence / definitiveness (pdf may be absent by config; don't require it)
        assert CharacteristicName.PMF not in view.all_characteristics
        assert {CharacteristicName.CDF, CharacteristicName.PPF}.issubset(view.all_characteristics)
        assert {CharacteristicName.CDF, CharacteristicName.PPF}.issubset(
            view.definitive_characteristics
        )

        # Paths must exist between definitive characteristics we rely on
        assert view.find_path(CharacteristicName.CDF, CharacteristicName.PPF) is not None
        assert view.find_path(CharacteristicName.PPF, CharacteristicName.CDF) is not None

        # Strategy resolves and roundtrips: CDF(PPF(q)) ~ q
        strategy = DefaultComputationStrategy(enable_caching=False)
        ppf = strategy.query_method(CharacteristicName.PPF, distr)
        cdf = strategy.query_method(CharacteristicName.CDF, distr)
        qs = np.linspace(1e-6, 1.0 - 1e-6, 7)
        errs = [abs(float(cdf(float(ppf(float(q))))) - q) for q in qs]
        assert max(errs) < 5e-3

    def test_view_adds_analytical_self_loops_with_labels(self) -> None:
        def cdf_primary(x: float, **_kwargs: Any) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        def cdf_secondary(x: float, **_kwargs: Any) -> float:
            return 1.0 / (1.0 + np.exp(-x))

        cdf_primary_func = cast(Callable[[float, KwArg(Any)], float], cdf_primary)
        cdf_secondary_func = cast(Callable[[float, KwArg(Any)], float], cdf_secondary)

        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.CDF: {
                    "primary": AnalyticalComputation[float, float](
                        target=CharacteristicName.CDF, func=cdf_primary_func
                    ),
                    "secondary": AnalyticalComputation[float, float](
                        target=CharacteristicName.CDF, func=cdf_secondary_func
                    ),
                }
            },
            support=ContinuousSupport(),
        )

        view = characteristic_registry().view(distr)
        loops = view.variants(CharacteristicName.CDF, CharacteristicName.CDF)

        assert set(loops.keys()) == {"primary", "secondary"}
        assert all(edge.is_analytical for edge in loops.values())
        assert view.analytical_variants(CharacteristicName.CDF) == loops

    def test_strategy_prefers_first_analytical_loop(self) -> None:
        def cdf_first(_x: float, **_kwargs: Any) -> float:
            return 0.25

        def cdf_second(_x: float, **_kwargs: Any) -> float:
            return 0.75

        cdf_first_func = cast(Callable[[float, KwArg(Any)], float], cdf_first)
        cdf_second_func = cast(Callable[[float, KwArg(Any)], float], cdf_second)

        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.CDF: {
                    "first": AnalyticalComputation[float, float](
                        target=CharacteristicName.CDF, func=cdf_first_func
                    ),
                    "second": AnalyticalComputation[float, float](
                        target=CharacteristicName.CDF, func=cdf_second_func
                    ),
                }
            },
            support=ContinuousSupport(),
        )

        strategy = DefaultComputationStrategy(enable_caching=False)
        cdf = strategy.query_method(CharacteristicName.CDF, distr)

        assert cdf(0.0) == pytest.approx(0.25)

    def test_configuration_discrete_requires_support_then_ok(self) -> None:
        reg = characteristic_registry()
        distr = self.make_discrete_point_pmf_distribution(is_with_support=False)

        # Without support, discrete edges (pmf<->cdf, cdf<->ppf) are disabled -> invariants fail
        with pytest.raises(GraphInvariantError):
            reg.view(distr)

        # Add support and try again
        distr = self.make_discrete_point_pmf_distribution()
        view = reg.view(distr)

        assert CharacteristicName.PDF not in view.all_characteristics
        assert {CharacteristicName.PMF, CharacteristicName.CDF, CharacteristicName.PPF}.issubset(
            view.all_characteristics
        )
        assert {CharacteristicName.PMF, CharacteristicName.CDF, CharacteristicName.PPF}.issubset(
            view.definitive_characteristics
        )

        assert view.find_path(CharacteristicName.PMF, CharacteristicName.CDF) is not None
        assert view.find_path(CharacteristicName.CDF, CharacteristicName.PMF) is not None
        assert view.find_path(CharacteristicName.CDF, CharacteristicName.PPF) is not None
        assert view.find_path(CharacteristicName.PPF, CharacteristicName.CDF) is not None

        strategy = DefaultComputationStrategy(enable_caching=False)
        cdf = strategy.query_method(CharacteristicName.CDF, distr)
        assert cdf(0.0) == pytest.approx(0.2, abs=1e-10)
        assert cdf(1.0) == pytest.approx(0.7, abs=1e-10)
        assert cdf(2.0) == pytest.approx(1.0, abs=1e-10)

        ppf = strategy.query_method(CharacteristicName.PPF, distr)
        assert ppf(0.10) == pytest.approx(0.0, abs=1e-12)
        assert ppf(0.70) == pytest.approx(1.0, abs=1e-12)
        assert ppf(0.95) == pytest.approx(2.0, abs=1e-12)

    def test_edge_dims_constraint_filters_edges(self) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("A", is_definitive=True)
        reg.add_characteristic("B", is_definitive=True)

        cons_dim2 = GraphPrimitiveConstraint(
            distribution_type_feature_constraints={
                "dimension": NumericConstraint(allowed=frozenset({2}))
            }
        )
        reg.add_computation(
            self.make_fictitious_computation_method(target="B", sources=["A"]),
            constraint=cons_dim2,
        )
        reg.add_computation(
            self.make_fictitious_computation_method(target="A", sources=["B"]),
            constraint=cons_dim2,
        )

        with pytest.raises(GraphInvariantError):
            reg.view(self.distr_example)

    def test_add_computation_validation_and_duplicate_rules(self) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("a", is_definitive=True)
        reg.add_characteristic("b", is_definitive=True)
        reg.add_characteristic("c", is_definitive=False)

        # empty-sources validation
        with pytest.raises(ValueError):
            reg.add_computation(self.make_fictitious_computation_method(target="b", sources=[]))

        # many-to-one computation is supported
        reg.add_computation(self.make_fictitious_computation_method(target="c", sources=["a", "b"]))

        # undeclared nodes
        with pytest.raises(ValueError):
            reg.add_computation(self.make_fictitious_computation_method(target="y", sources=["x"]))

        # duplicate presence/definitiveness rules warn (idempotent)
        with pytest.warns(UserWarning):
            reg.add_characteristic("a", is_definitive=True)
        with pytest.warns(UserWarning):
            reg.add_characteristic("b", is_definitive=True)

    def test_reset_cached_singleton(self) -> None:
        r1 = characteristic_registry()
        r2 = characteristic_registry()
        assert r1 is r2
        reset_characteristic_registry()
        r3 = characteristic_registry()
        assert r3 is not r1

    def test_invariant_violations(self) -> None:
        reg = CharacteristicRegistry()

        reg.add_characteristic("A", is_definitive=True)
        reg.add_characteristic("B", is_definitive=True)

        reg.add_computation(self.make_fictitious_computation_method(target="B", sources=["A"]))
        # Definitive are not strongly connected
        with pytest.raises(GraphInvariantError):
            reg.view(self.distr_example)

        # Everything is good
        reg.add_computation(self.make_fictitious_computation_method(target="A", sources=["B"]))
        reg.view(self.distr_example)

        # Idefinitive X is not reachable
        reg.add_characteristic("X", is_definitive=False)
        with pytest.raises(GraphInvariantError):
            reg.view(self.distr_example)

        # Everything is good
        reg.add_computation(self.make_fictitious_computation_method(target="X", sources=["A"]))
        reg.view(self.distr_example)

        # Indefinitive reaches definitive
        reg.add_computation(self.make_fictitious_computation_method(target="A", sources=["X"]))
        with pytest.raises(GraphInvariantError):
            reg.view(self.distr_example)

    def test_label_variants_and_picking(self) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("src", is_definitive=True)
        reg.add_characteristic("dst", is_definitive=True)

        default_method = self.make_fictitious_computation_method(target="dst", sources=["src"])
        alternative_method = self.make_fictitious_computation_method(target="dst", sources=["src"])
        # reverse edge to keep the definitive subgraph strongly connected
        reverse_method = self.make_fictitious_computation_method(target="src", sources=["dst"])

        reg.add_computation(default_method)
        reg.add_computation(alternative_method, label="fast")
        reg.add_computation(reverse_method)

        view = reg.view(self.make_logistic_cdf_distribution())

        variants = view.variants("src", "dst")
        assert set(variants.keys()) == {DEFAULT_COMPUTATION_KEY, "fast"}
        assert all(not edge.is_analytical for edge in variants.values())

        path = view.find_path("src", "dst", prefer_label="fast")
        assert path == [alternative_method]

        path = view.find_path("src", "dst")
        assert path == [default_method]

    def test_hyperedge_many_to_one_projection_and_single_fitter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("A", is_definitive=True)
        reg.add_characteristic("B", is_definitive=True)
        reg.add_characteristic("C", is_definitive=False)

        reg.add_computation(self.make_fictitious_computation_method(target="B", sources=["A"]))
        reg.add_computation(self.make_fictitious_computation_method(target="A", sources=["B"]))

        fitter_calls: dict[str, int] = {"count": 0}

        def fit_ab_to_c(
            distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            fitter_calls["count"] += 1
            a_method = distribution.query_method("A")
            b_method = distribution.query_method("B")

            def c_func(**_options: Any) -> float:
                return float(a_method() + b_method())

            return FittedComputationMethod(
                target="C",
                sources=("A", "B"),
                func=cast(Callable[[KwArg(Any)], float], c_func),
            )

        hyper_method = ComputationMethod[Any, Any](
            target="C",
            sources=("A", "B"),
            fitter=cast(
                Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                fit_ab_to_c,
            ),
        )
        reg.add_computation(hyper_method, label="ab_to_c")

        a_func = cast(Callable[[KwArg(Any)], float], lambda **_kwargs: 2.0)
        b_func = cast(Callable[[KwArg(Any)], float], lambda **_kwargs: 3.0)
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "A": {"default": AnalyticalComputation[Any, Any](target="A", func=a_func)},
                "B": {"default": AnalyticalComputation[Any, Any](target="B", func=b_func)},
            },
            support=ContinuousSupport(),
        )

        view = reg.view(distr)
        assert view.variants("A", "C")["ab_to_c"].method is hyper_method
        assert view.variants("B", "C")["ab_to_c"].method is hyper_method
        assert view.find_path("A", "C") == [hyper_method]
        assert view.find_path("B", "C") == [hyper_method]

        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)
        c_method_first = strategy.query_method("C", distr)
        c_method_second = strategy.query_method("C", distr)

        assert c_method_first() == pytest.approx(5.0)
        assert c_method_second() == pytest.approx(5.0)
        assert fitter_calls["count"] == 1

    def test_hyperedge_requires_all_sources_present_in_view(self) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("A", is_definitive=True)
        reg.add_characteristic(
            "B",
            is_definitive=False,
            presence_constraint=GraphPrimitiveConstraint(
                distribution_type_feature_constraints={
                    "dimension": NumericConstraint(allowed=frozenset({2}))
                }
            ),
        )
        reg.add_characteristic("C", is_definitive=False)

        reg.add_computation(self.make_fictitious_computation_method(target="C", sources=["A", "B"]))

        # Source B is absent for 1D distributions, so hyperedge A,B -> C must be filtered out.
        # Then C becomes unreachable from definitive nodes and invariant validation must fail.
        with pytest.raises(GraphInvariantError):
            reg.view(self.make_logistic_cdf_distribution())

    def test_strategy_resolves_diamond_graph_from_single_analytical_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("pdf", is_definitive=True)
        reg.add_characteristic("mean", is_definitive=False)
        reg.add_characteristic("second_moment", is_definitive=False)
        reg.add_characteristic("mean_sq", is_definitive=False)
        reg.add_characteristic("var", is_definitive=False)

        def fit_pdf_to_mean(
            _distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            return FittedComputationMethod(
                target="mean",
                sources=("pdf",),
                func=cast(Callable[[KwArg(Any)], float], lambda **_opts: 2.0),
            )

        def fit_pdf_to_second_moment(
            _distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            return FittedComputationMethod(
                target="second_moment",
                sources=("pdf",),
                func=cast(Callable[[KwArg(Any)], float], lambda **_opts: 5.0),
            )

        def fit_mean_to_mean_sq(
            distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            mean_method = distribution.query_method("mean")
            return FittedComputationMethod(
                target="mean_sq",
                sources=("mean",),
                func=cast(
                    Callable[[KwArg(Any)], float],
                    lambda **_opts: float(mean_method() ** 2),
                ),
            )

        def fit_second_moment_and_mean_sq_to_var(
            distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            second_moment_method = distribution.query_method("second_moment")
            mean_sq_method = distribution.query_method("mean_sq")
            return FittedComputationMethod(
                target="var",
                sources=("second_moment", "mean_sq"),
                func=cast(
                    Callable[[KwArg(Any)], float],
                    lambda **_opts: float(second_moment_method() - mean_sq_method()),
                ),
            )

        reg.add_computation(
            ComputationMethod[Any, Any](
                target="mean",
                sources=("pdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    fit_pdf_to_mean,
                ),
            )
        )
        reg.add_computation(
            ComputationMethod[Any, Any](
                target="second_moment",
                sources=("pdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    fit_pdf_to_second_moment,
                ),
            )
        )
        reg.add_computation(
            ComputationMethod[Any, Any](
                target="mean_sq",
                sources=("mean",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    fit_mean_to_mean_sq,
                ),
            )
        )
        reg.add_computation(
            ComputationMethod[Any, Any](
                target="var",
                sources=("second_moment", "mean_sq"),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    fit_second_moment_and_mean_sq_to_var,
                ),
            )
        )

        pdf_func = cast(Callable[[float, KwArg(Any)], float], lambda _x, **_opts: 1.0)
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "pdf": {"default": AnalyticalComputation[Any, Any](target="pdf", func=pdf_func)}
            },
            support=ContinuousSupport(),
        )

        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)
        var_method = strategy.query_method("var", distr)

        assert var_method() == pytest.approx(1.0)
