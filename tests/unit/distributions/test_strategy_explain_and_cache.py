from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from typing import Any, cast

import pytest
from mypy_extensions import KwArg

from pysatl_core.distributions import strategies as strategies_module
from pysatl_core.distributions.computations.base import (
    ComputationOption,
    OptionsDescriptor,
)
from pysatl_core.distributions.computations.computation import (
    AnalyticalComputation,
    EvaluatorMethod,
    FittedComputationMethod,
    FitterMethod,
)
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.registry import CharacteristicRegistry
from pysatl_core.distributions.strategies import (
    DefaultComputationStrategy,
    ExecutionPlan,
    ExecutionStep,
    _freeze_options,
)
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.types import EvaluatorFunc, Kind
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_pdf_to_cdf_registry(
    fit_calls: dict[str, int],
    options_descriptor: OptionsDescriptor,
) -> tuple[CharacteristicRegistry, FitterMethod, FitterMethod]:
    """
    Build a tiny registry with two definitive nodes ``pdf`` and ``cdf``
    connected in both directions, where the ``pdf -> cdf`` edge carries
    a real :class:`OptionsDescriptor` so we can probe option-aware caching.
    """
    reg = CharacteristicRegistry()
    reg.add_characteristic("pdf", is_definitive=True)
    reg.add_characteristic("cdf", is_definitive=True)

    def fit_pdf_to_cdf(
        _distribution: Distribution, **kwargs: Any
    ) -> FittedComputationMethod[Any, Any]:
        fit_calls["count"] += 1
        tol = kwargs.get("tol", 0.0)

        def cdf(_x: float, **_opts: Any) -> float:
            return 0.5 + tol  # encode the tolerance into the result

        return FittedComputationMethod(
            target="cdf",
            sources=("pdf",),
            func=cast(Callable[[float, KwArg(Any)], float], cdf),
        )

    def fit_cdf_to_pdf(
        _distribution: Distribution, **_kwargs: Any
    ) -> FittedComputationMethod[Any, Any]:
        return FittedComputationMethod(
            target="pdf",
            sources=("cdf",),
            func=cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 1.0),
        )

    pdf_to_cdf = FitterMethod(
        target="cdf",
        sources=("pdf",),
        fitter=cast(
            Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
            fit_pdf_to_cdf,
        ),
    )
    cdf_to_pdf = FitterMethod(
        target="pdf",
        sources=("cdf",),
        fitter=cast(
            Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
            fit_cdf_to_pdf,
        ),
    )

    reg.add_computation(pdf_to_cdf, options_descriptor=options_descriptor)
    reg.add_computation(cdf_to_pdf)

    return reg, pdf_to_cdf, cdf_to_pdf


def _make_pdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    pdf_func = cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 1.0)
    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations={
            "pdf": {"default": AnalyticalComputation[float, float](target="pdf", func=pdf_func)}
        },
        support=ContinuousSupport(),
    )


# --------------------------------------------------------------------------- #
# Option-aware caching
# --------------------------------------------------------------------------- #


class TestOptionsAwareCaching:
    def test_different_options_produce_independent_cache_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fit_calls = {"count": 0}
        options_descriptor = OptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, _pdf_to_cdf, _cdf_to_pdf = _build_pdf_to_cdf_registry(fit_calls, options_descriptor)
        distr = _make_pdf_distribution()

        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        cdf_a = strategy.query_method("cdf", distr, tol=0.1)
        cdf_b = strategy.query_method("cdf", distr, tol=0.2)
        cdf_a_again = strategy.query_method("cdf", distr, tol=0.1)

        # Different options -> different fitted result; same options -> shared.
        assert cdf_a(0.0) == pytest.approx(0.6)
        assert cdf_b(0.0) == pytest.approx(0.7)
        assert cdf_a_again is cdf_a
        # Two distinct option sets means exactly two fit calls.
        assert fit_calls["count"] == 2

    def test_caching_disabled_refits_every_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fit_calls = {"count": 0}
        options_descriptor = OptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry(fit_calls, options_descriptor)
        distr = _make_pdf_distribution()

        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        strategy.query_method("cdf", distr, tol=0.1)
        strategy.query_method("cdf", distr, tol=0.1)
        assert fit_calls["count"] == 2

    def test_undeclared_options_are_ignored_for_cache_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Options that are *not* declared by the edge's :class:`OptionsDescriptor`
        must not influence the cache key (they are not consumed by the fitter
        either, so they would be a no-op).
        """
        fit_calls = {"count": 0}
        options_descriptor = OptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry(fit_calls, options_descriptor)
        distr = _make_pdf_distribution()

        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        first = strategy.query_method("cdf", distr, tol=0.1, irrelevant="a")
        second = strategy.query_method("cdf", distr, tol=0.1, irrelevant="b")
        assert first is second
        assert fit_calls["count"] == 1


# --------------------------------------------------------------------------- #
# Path caching + explain
# --------------------------------------------------------------------------- #


class TestExplainAndPathCache:
    def test_explain_returns_loop_plan_for_directly_provided_characteristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg, *_ = _build_pdf_to_cdf_registry({"count": 0}, OptionsDescriptor())
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain("pdf", distr)

        assert isinstance(plan, ExecutionPlan)
        assert plan.target == "pdf"
        assert plan.source == "pdf"
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert isinstance(step, ExecutionStep)
        assert step.edge_kind in {"analytical_loop", "transformation_loop"}
        assert step.target == "pdf"
        assert step.sources == ("pdf",)

    def test_explain_returns_conversion_plan_with_options_descriptor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        options_descriptor = OptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry({"count": 0}, options_descriptor)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain("cdf", distr)

        assert plan.source == "pdf"
        assert plan.target == "cdf"
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.edge_kind == "computation"
        assert step.target == "cdf"
        assert step.sources == ("pdf",)
        assert step.method_name == "pdf_to_cdf"
        assert plan.required_options() == ("tol",)

    def test_path_cache_pins_query_method_to_the_explained_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        ``explain`` is supposed to *pin* the chosen path so a subsequent
        ``query_method`` follows the very same edges, even when picking
        a path could be non-deterministic.  We simulate that by replacing
        :meth:`RegistryView.find_path` after explain has cached its result;
        if the strategy still produced the correct cdf, it means the second
        call did not re-run BFS.
        """
        fit_calls = {"count": 0}
        options_descriptor = OptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry(fit_calls, options_descriptor)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        plan = strategy.explain("cdf", distr)
        assert plan.source == "pdf"

        # Sabotage path-finding: any subsequent BFS would now break or
        # return None.  The strategy must rely on the cached plan instead.
        from pysatl_core.distributions.registry.graph import RegistryView

        def _broken_find_path(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("find_path should not be called after explain pinned the plan")

        monkeypatch.setattr(RegistryView, "find_path", _broken_find_path)

        cdf = strategy.query_method("cdf", distr, tol=0.4)
        assert cdf(0.0) == pytest.approx(0.9)
        assert fit_calls["count"] == 1

    def test_distribution_explain_shortcut(self, monkeypatch: pytest.MonkeyPatch) -> None:
        options_descriptor = OptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry({"count": 0}, options_descriptor)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        # Ensure the distribution uses a strategy that sees the patched registry.
        distr = distr.with_computation_strategy(DefaultComputationStrategy(enable_caching=True))

        plan = distr.explain("cdf")

        assert isinstance(plan, ExecutionPlan)
        assert plan.target == "cdf"
        assert plan.required_options() == ("tol",)


# --------------------------------------------------------------------------- #
# _freeze_options helper
# --------------------------------------------------------------------------- #


class TestFreezeOptions:
    def test_freeze_handles_basic_hashables(self) -> None:
        key = _freeze_options({"a": 1, "b": "x", "c": (1, 2)})
        assert key == frozenset({("a", 1), ("b", "x"), ("c", (1, 2))})

    def test_freeze_handles_nested_unhashables(self) -> None:
        key1 = _freeze_options({"a": [1, 2, 3], "b": {"k": 1}})
        key2 = _freeze_options({"a": [1, 2, 3], "b": {"k": 1}})
        assert key1 == key2

    def test_freeze_distinguishes_distinct_values(self) -> None:
        a = _freeze_options({"a": [1, 2]})
        b = _freeze_options({"a": [1, 3]})
        assert a != b


# --------------------------------------------------------------------------- #
# Multi-edge conversion paths
# --------------------------------------------------------------------------- #


def _build_two_hop_registry(
    fit_calls: dict[str, int],
) -> CharacteristicRegistry:
    """
    Registry shape::

        pdf <-> cdf <-> ppf

    Both forward edges (``pdf -> cdf`` and ``cdf -> ppf``) carry their
    own :class:`OptionsDescriptor` so the strategy must route options
    per edge.  Reverse edges keep the definitive subgraph strongly
    connected (graph invariant).
    """
    reg = CharacteristicRegistry()
    reg.add_characteristic("pdf", is_definitive=True)
    reg.add_characteristic("cdf", is_definitive=True)
    reg.add_characteristic("ppf", is_definitive=True)

    def fit_pdf_to_cdf(
        _distribution: Distribution, **kwargs: Any
    ) -> FittedComputationMethod[Any, Any]:
        fit_calls["pdf_to_cdf"] = fit_calls.get("pdf_to_cdf", 0) + 1
        bias = kwargs.get("cdf_bias", 0.0)

        def cdf(_x: float, **_o: Any) -> float:
            return 0.5 + bias

        return FittedComputationMethod(
            target="cdf",
            sources=("pdf",),
            func=cast(Callable[[float, KwArg(Any)], float], cdf),
        )

    def fit_cdf_to_ppf(
        _distribution: Distribution, **kwargs: Any
    ) -> FittedComputationMethod[Any, Any]:
        fit_calls["cdf_to_ppf"] = fit_calls.get("cdf_to_ppf", 0) + 1
        eps = kwargs.get("eps", 0.0)

        def ppf(_q: float, **_o: Any) -> float:
            return eps

        return FittedComputationMethod(
            target="ppf",
            sources=("cdf",),
            func=cast(Callable[[float, KwArg(Any)], float], ppf),
        )

    def fit_trivial(target: str, sources: tuple[str, ...]) -> FitterMethod:
        def _fit(_distribution: Distribution, **_kwargs: Any) -> FittedComputationMethod[Any, Any]:
            return FittedComputationMethod(
                target=target,
                sources=sources,
                func=cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 0.0),
            )

        return FitterMethod(
            target=target,
            sources=sources,
            fitter=cast(
                Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                _fit,
            ),
        )

    pdf_to_cdf = FitterMethod(
        target="cdf",
        sources=("pdf",),
        fitter=cast(
            Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
            fit_pdf_to_cdf,
        ),
    )
    cdf_to_ppf = FitterMethod(
        target="ppf",
        sources=("cdf",),
        fitter=cast(
            Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
            fit_cdf_to_ppf,
        ),
    )

    reg.add_computation(
        pdf_to_cdf,
        options_descriptor=OptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="cdf_bias", type=float, default=0.0),),
        ),
    )
    reg.add_computation(
        cdf_to_ppf,
        options_descriptor=OptionsDescriptor(
            name="cdf_to_ppf",
            computation_options=(ComputationOption(name="eps", type=float, default=0.0),),
        ),
    )

    # reverse edges to keep the definitive subgraph strongly connected
    reg.add_computation(fit_trivial("pdf", ("cdf",)))
    reg.add_computation(fit_trivial("cdf", ("ppf",)))

    return reg


class TestMultiEdgePaths:
    def test_options_route_per_edge_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fit_calls: dict[str, int] = {}
        reg = _build_two_hop_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        # Same cdf_bias, different eps -> only the second edge re-fits.
        ppf_a = strategy.query_method("ppf", distr, cdf_bias=0.1, eps=1e-3)
        ppf_b = strategy.query_method("ppf", distr, cdf_bias=0.1, eps=2e-3)

        assert ppf_a(0.0) == pytest.approx(1e-3)
        assert ppf_b(0.0) == pytest.approx(2e-3)
        assert fit_calls["pdf_to_cdf"] == 1, "first edge should be cached (cdf_bias unchanged)"
        assert fit_calls["cdf_to_ppf"] == 2, "second edge should be re-fitted (eps changed)"

    def test_explain_lists_all_options_along_multi_hop_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = _build_two_hop_registry({})
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain("ppf", distr)

        assert plan.source == "pdf"
        assert plan.target == "ppf"
        assert tuple(s.method_name for s in plan.steps) == ("pdf_to_cdf", "cdf_to_ppf")
        assert tuple(s.target for s in plan.steps) == ("cdf", "ppf")
        assert set(plan.required_options()) == {"cdf_bias", "eps"}


# --------------------------------------------------------------------------- #
# Non-registry analytical characteristics + error branches
# --------------------------------------------------------------------------- #


class TestNonRegistryAndErrors:
    def test_explain_for_non_registry_analytical_characteristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``state`` provided directly by the distribution but not declared
        in the registry must produce a single analytical-loop step."""
        reg = CharacteristicRegistry()
        reg.add_characteristic("pdf", is_definitive=True)

        custom_func = cast(Callable[[KwArg(Any)], float], lambda **_o: 42.0)
        pdf_func = cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 1.0)
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "pdf": {
                    "default": AnalyticalComputation[float, float](target="pdf", func=pdf_func)
                },
                "custom_metric": {
                    "default": AnalyticalComputation[Any, Any](
                        target="custom_metric", func=custom_func
                    )
                },
            },
            support=ContinuousSupport(),
        )
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain("custom_metric", distr)
        assert plan.source == "custom_metric"
        assert plan.target == "custom_metric"
        assert len(plan.steps) == 1
        assert plan.steps[0].edge_kind == "analytical_loop"

        method = strategy.query_method("custom_metric", distr)
        assert method() == pytest.approx(42.0)

    def test_query_method_raises_when_state_unknown_everywhere(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("pdf", is_definitive=True)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy()

        with pytest.raises(RuntimeError, match="not declared in the registry"):
            strategy.query_method("does_not_exist", distr)

    def test_query_method_raises_when_no_path_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two definitive nodes that are isolated from each other must
        cause the strategy to fail with a clear message."""
        reg = CharacteristicRegistry()
        reg.add_characteristic("pdf", is_definitive=True)
        reg.add_characteristic("cdf", is_definitive=True)
        # Trivial mutual reverse edges so the invariant validation passes,
        # but no edge exists from any present source loop towards a node
        # the distribution does not provide.
        reg.add_computation(
            FitterMethod(
                target="cdf",
                sources=("pdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    lambda *_a, **_k: FittedComputationMethod(
                        target="cdf",
                        sources=("pdf",),
                        func=cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 0.0),
                    ),
                ),
            )
        )
        reg.add_computation(
            FitterMethod(
                target="pdf",
                sources=("cdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    lambda *_a, **_k: FittedComputationMethod(
                        target="pdf",
                        sources=("cdf",),
                        func=cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 0.0),
                    ),
                ),
            )
        )

        # Distribution provides only an unrelated, *non-registry* analytical
        # characteristic.  ``cdf`` is in the registry but has no source loop
        # to start from.
        custom_func = cast(Callable[[KwArg(Any)], float], lambda **_o: 1.0)
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "unrelated": {
                    "default": AnalyticalComputation[Any, Any](target="unrelated", func=custom_func)
                },
            },
            support=ContinuousSupport(),
        )
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy()

        with pytest.raises(RuntimeError, match="No conversion path"):
            strategy.query_method("cdf", distr)


# --------------------------------------------------------------------------- #
# Hyperedge plan
# --------------------------------------------------------------------------- #


class TestHyperedgePlan:
    def test_explain_step_carries_all_hyperedge_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("A", is_definitive=True)
        reg.add_characteristic("B", is_definitive=True)
        reg.add_characteristic("C", is_definitive=False)
        reg.add_computation(
            FitterMethod(
                target="B",
                sources=("A",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    lambda *_a, **_k: FittedComputationMethod(
                        target="B",
                        sources=("A",),
                        func=cast(Callable[[KwArg(Any)], float], lambda **_o: 0.0),
                    ),
                ),
            )
        )
        reg.add_computation(
            FitterMethod(
                target="A",
                sources=("B",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    lambda *_a, **_k: FittedComputationMethod(
                        target="A",
                        sources=("B",),
                        func=cast(Callable[[KwArg(Any)], float], lambda **_o: 0.0),
                    ),
                ),
            )
        )
        reg.add_computation(
            FitterMethod(
                target="C",
                sources=("A", "B"),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    lambda *_a, **_k: FittedComputationMethod(
                        target="C",
                        sources=("A", "B"),
                        func=cast(Callable[[KwArg(Any)], float], lambda **_o: 5.0),
                    ),
                ),
            ),
            label="ab_to_c",
        )

        a_func = cast(Callable[[KwArg(Any)], float], lambda **_o: 1.0)
        b_func = cast(Callable[[KwArg(Any)], float], lambda **_o: 2.0)
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                "A": {"default": AnalyticalComputation[Any, Any](target="A", func=a_func)},
                "B": {"default": AnalyticalComputation[Any, Any](target="B", func=b_func)},
            },
            support=ContinuousSupport(),
        )
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain("C", distr)
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.target == "C"
        # Both sources must be reflected in the step, regardless of which
        # one BFS picked as the entry point.
        assert set(step.sources) == {"A", "B"}


# --------------------------------------------------------------------------- #
# Evaluator edges should bypass the cache
# --------------------------------------------------------------------------- #


class TestEvaluatorBypassCache:
    def test_evaluator_edge_is_not_cached_even_with_caching_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("pdf", is_definitive=True)
        reg.add_characteristic("cdf", is_definitive=True)

        eval_calls = {"count": 0}

        def evaluate_cdf(_distribution: Distribution, *_args: Any, **_kwargs: Any) -> float:
            eval_calls["count"] += 1
            return 0.5

        reg.add_computation(
            EvaluatorMethod(
                target="cdf",
                sources=("pdf",),
                evaluator=cast("EvaluatorFunc", evaluate_cdf),
            )
        )
        reg.add_computation(
            FitterMethod(
                target="pdf",
                sources=("cdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    lambda *_a, **_k: FittedComputationMethod(
                        target="pdf",
                        sources=("cdf",),
                        func=cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 1.0),
                    ),
                ),
            )
        )
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        cdf_a = strategy.query_method("cdf", distr)
        cdf_b = strategy.query_method("cdf", distr)

        # The fitted wrapper is rebuilt on every call -> different instances.
        assert cdf_a is not cdf_b
        # And the cache stays empty (evaluator methods report cacheable=False).
        assert strategy._cache == {}
        # Sanity: the actual evaluator is called when the wrapper is used.
        assert cdf_a(0.0) == pytest.approx(0.5)
        assert cdf_b(0.0) == pytest.approx(0.5)
        assert eval_calls["count"] == 2


# --------------------------------------------------------------------------- #
# Cycle detection
# --------------------------------------------------------------------------- #


class TestCycleDetection:
    def test_cycle_inside_fitter_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fitter that recursively asks the strategy for *its own*
        target must blow up with a clear error instead of silently
        recursing forever."""
        reg = CharacteristicRegistry()
        reg.add_characteristic("pdf", is_definitive=True)
        reg.add_characteristic("cdf", is_definitive=True)

        def fit_pdf_to_cdf(
            distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            # Re-asking for ``cdf`` while resolving ``cdf`` -> cycle.
            distribution.query_method("cdf")
            return FittedComputationMethod(
                target="cdf",
                sources=("pdf",),
                func=cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 0.0),
            )

        reg.add_computation(
            FitterMethod(
                target="cdf",
                sources=("pdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    fit_pdf_to_cdf,
                ),
            )
        )
        reg.add_computation(
            FitterMethod(
                target="pdf",
                sources=("cdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    lambda *_a, **_k: FittedComputationMethod(
                        target="pdf",
                        sources=("cdf",),
                        func=cast(Callable[[float, KwArg(Any)], float], lambda _x, **_o: 0.0),
                    ),
                ),
            )
        )

        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        # Need to plug the strategy into the distribution so that
        # ``distribution.query_method`` from inside the fitter actually
        # reuses the same strategy instance with its ``_resolving`` guard.
        distr = distr.with_computation_strategy(DefaultComputationStrategy(enable_caching=False))

        with pytest.raises(RuntimeError, match="Cycle detected"):
            distr.query_method("cdf")
