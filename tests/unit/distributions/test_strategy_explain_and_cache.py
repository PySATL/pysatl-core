from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from typing import Any, cast

import pytest
from mypy_extensions import KwArg

from pysatl_core.distributions import strategies as strategies_module
from pysatl_core.distributions.computations.computation import (
    AnalyticalComputation,
    EvaluatorMethod,
    FittedComputationMethod,
    FitterMethod,
)
from pysatl_core.distributions.computations.options import (
    CharacteristicOption,
    ComputationOption,
    EdgeOptionsDescriptor,
    ResolvedEdgeOptions,
)
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.registry import CharacteristicRegistry
from pysatl_core.distributions.strategies import (
    ComputationPlan,
    ComputationStep,
    DefaultComputationStrategy,
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
    options_descriptor: EdgeOptionsDescriptor,
) -> tuple[CharacteristicRegistry, FitterMethod, FitterMethod]:
    """
    Build a tiny registry with two definitive nodes ``pdf`` and ``cdf``
    connected in both directions, where the ``pdf -> cdf`` edge carries
    a real :class:`EdgeOptionsDescriptor` so we can probe option-aware caching.
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
        options_descriptor = EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, _pdf_to_cdf, _cdf_to_pdf = _build_pdf_to_cdf_registry(fit_calls, options_descriptor)
        distr = _make_pdf_distribution()

        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        plan = strategy.explain_computation_path("cdf", distr)
        opts_a = plan.with_options(0, tol=0.1)
        opts_b = plan.with_options(0, tol=0.2)

        cdf_a = strategy.query_method("cdf", distr, opts_a)
        cdf_b = strategy.query_method("cdf", distr, opts_b)
        cdf_a_again = strategy.query_method("cdf", distr, opts_a)

        # Different options -> different fitted result; same options -> shared.
        assert cdf_a(0.0) == pytest.approx(0.6)
        assert cdf_b(0.0) == pytest.approx(0.7)
        assert cdf_a_again is cdf_a
        # Two distinct option sets means exactly two fit calls.
        assert fit_calls["count"] == 2

    def test_caching_disabled_refits_every_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fit_calls = {"count": 0}
        options_descriptor = EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry(fit_calls, options_descriptor)
        distr = _make_pdf_distribution()

        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain_computation_path("cdf", distr)
        opts = plan.with_options(0, tol=0.1)
        strategy.query_method("cdf", distr, opts)
        strategy.query_method("cdf", distr, opts)
        assert fit_calls["count"] == 2

    def test_with_values_validates_eagerly(self) -> None:
        """
        :meth:`EdgeOptionsDescriptor.with_values` validates option values
        eagerly, so type errors surface before ``query_method`` is called.
        """
        descriptor = EdgeOptionsDescriptor(
            name="test",
            computation_options=(
                ComputationOption(
                    name="tol",
                    type=float,
                    default=0.0,
                    validate=lambda v: v >= 0,
                ),
            ),
        )
        # Valid value works.
        resolved = descriptor.with_values(tol=0.1)
        assert isinstance(resolved, ResolvedEdgeOptions)
        assert resolved.values == {"tol": 0.1}

        # Invalid value raises immediately.
        with pytest.raises(ValueError, match="failed validation"):
            descriptor.with_values(tol=-1.0)


# --------------------------------------------------------------------------- #
# Path caching + explain_computation_path
# --------------------------------------------------------------------------- #


class TestExplainAndPathCache:
    def test_explain_returns_loop_plan_for_directly_provided_characteristic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg, *_ = _build_pdf_to_cdf_registry({"count": 0}, EdgeOptionsDescriptor())
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain_computation_path("pdf", distr)

        assert isinstance(plan, ComputationPlan)
        assert plan.target == "pdf"
        assert plan.source == "pdf"
        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert isinstance(step, ComputationStep)
        assert step.edge_kind in {"analytical_loop", "transformation_loop"}
        assert step.target == "pdf"
        assert step.sources == ("pdf",)

    def test_explain_returns_conversion_plan_with_options_descriptor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        options_descriptor = EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry({"count": 0}, options_descriptor)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain_computation_path("cdf", distr)

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
        ``explain_computation_path`` is supposed to *pin* the chosen path so a subsequent
        ``query_method`` follows the very same edges, even when picking
        a path could be non-deterministic.  We simulate that by replacing
        :meth:`RegistryView.find_path` after explain_computation_path has cached its result;
        if the strategy still produced the correct cdf, it means the second
        call did not re-run BFS.
        """
        fit_calls = {"count": 0}
        options_descriptor = EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry(fit_calls, options_descriptor)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        plan = strategy.explain_computation_path("cdf", distr)
        assert plan.source == "pdf"

        # Sabotage path-finding: any subsequent BFS would now break or
        # return None.  The strategy must rely on the cached plan instead.
        from pysatl_core.distributions.registry.graph import RegistryView

        def _broken_find_path(*_args: Any, **_kwargs: Any) -> None:
            raise AssertionError(
                "find_path should not be called after explain_computation_path pinned the plan"
            )

        monkeypatch.setattr(RegistryView, "find_path", _broken_find_path)

        opts = plan.with_options(0, tol=0.4)
        cdf = strategy.query_method("cdf", distr, opts)
        assert cdf(0.0) == pytest.approx(0.9)
        assert fit_calls["count"] == 1

    def test_distribution_explain_shortcut(self, monkeypatch: pytest.MonkeyPatch) -> None:
        options_descriptor = EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="tol", type=float, default=0.0),),
        )
        reg, *_ = _build_pdf_to_cdf_registry({"count": 0}, options_descriptor)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        # Ensure the distribution uses a strategy that sees the patched registry.
        distr = distr.with_computation_strategy(DefaultComputationStrategy(enable_caching=True))

        plan = distr.explain_computation_path("cdf")

        assert isinstance(plan, ComputationPlan)
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
    own :class:`EdgeOptionsDescriptor` so the strategy must route options
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
        options_descriptor=EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="cdf_bias", type=float, default=0.0),),
        ),
    )
    reg.add_computation(
        cdf_to_ppf,
        options_descriptor=EdgeOptionsDescriptor(
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
        plan = strategy.explain_computation_path("ppf", distr)
        opts_a = plan.with_options(0, cdf_bias=0.1) | plan.with_options(1, eps=1e-3)
        opts_b = plan.with_options(0, cdf_bias=0.1) | plan.with_options(1, eps=2e-3)
        ppf_a = strategy.query_method("ppf", distr, opts_a)
        ppf_b = strategy.query_method("ppf", distr, opts_b)

        assert ppf_a(0.0) == pytest.approx(1e-3)
        assert ppf_b(0.0) == pytest.approx(2e-3)
        assert fit_calls["pdf_to_cdf"] == 1, "first edge should be cached (cdf_bias unchanged)"
        assert fit_calls["cdf_to_ppf"] == 2, "second edge should be re-fitted (eps changed)"

    def test_second_edge_consumes_fitted_output_of_first_edge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        In a two-hop path ``pdf -> cdf -> ppf``, the fitter for the second
        edge (``cdf -> ppf``) calls ``strategy.query_method("cdf", distribution)``
        internally.  That call must return the *already-fitted* CDF produced
        by the first edge (with its ``cdf_bias`` applied), not a fresh CDF
        resolved from the original distribution.
        """
        fit_calls: dict[str, int] = {}

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

        # Capture strategy in a mutable cell so fit_cdf_to_ppf can reference
        # it after the strategy is created below.
        strategy_ref: list[DefaultComputationStrategy] = []

        def fit_cdf_to_ppf(
            distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            """
            This fitter deliberately queries the strategy for 'cdf' so
            that the test can verify the strategy threads the fitted CDF
            (with cdf_bias) through to this fitter rather than re-resolving
            from the original distribution.
            """
            fit_calls["cdf_to_ppf"] = fit_calls.get("cdf_to_ppf", 0) + 1
            # Ask the strategy directly for the CDF method — it should
            # supply the already-fitted version from the first edge.
            cdf_method = strategy_ref[0].query_method("cdf", distribution)
            # Capture the CDF value at 0.0 as the ppf constant so the test
            # can assert it reflects the cdf_bias from the first edge.
            captured_cdf_value = cdf_method(0.0)

            def ppf(_q: float, **_o: Any) -> float:
                return captured_cdf_value

            return FittedComputationMethod(
                target="ppf",
                sources=("cdf",),
                func=cast(Callable[[float, KwArg(Any)], float], ppf),
            )

        def _trivial_fitter(target: str, sources: tuple[str, ...]) -> FitterMethod:
            def _fit(_d: Distribution, **_kw: Any) -> FittedComputationMethod[Any, Any]:
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
            options_descriptor=EdgeOptionsDescriptor(
                name="pdf_to_cdf",
                computation_options=(ComputationOption(name="cdf_bias", type=float, default=0.0),),
            ),
        )
        reg.add_computation(cdf_to_ppf)
        reg.add_computation(_trivial_fitter("pdf", ("cdf",)))
        reg.add_computation(_trivial_fitter("cdf", ("ppf",)))

        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)
        strategy_ref.append(strategy)

        plan = strategy.explain_computation_path("ppf", distr)
        # Apply cdf_bias=0.25 on the first edge (pdf -> cdf).
        opts = plan.with_options(0, cdf_bias=0.25)
        ppf = strategy.query_method("ppf", distr, opts)

        # The second fitter queried strategy.query_method("cdf", distribution) and
        # captured its value at 0.0.  With cdf_bias=0.25 the fitted CDF
        # returns 0.75, so ppf(anything) must equal 0.75.
        assert ppf(0.0) == pytest.approx(0.75)

    def test_explain_lists_all_options_along_multi_hop_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg = _build_two_hop_registry({})
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain_computation_path("ppf", distr)

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

        plan = strategy.explain_computation_path("custom_metric", distr)
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

        plan = strategy.explain_computation_path("C", distr)
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


# --------------------------------------------------------------------------- #
# Fix 1: characteristic_options broadcast
# --------------------------------------------------------------------------- #


def _build_char_options_registry(
    fit_calls: dict[str, int],
) -> CharacteristicRegistry:
    """
    Registry with ``pdf -> cdf -> ppf`` where both forward edges carry
    *characteristic* options (``cdf_bias`` and ``eps``).  Used to verify
    that ``characteristic_options`` are broadcast to every step that
    declares them.
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
        def _fit(_d: Distribution, **_kw: Any) -> FittedComputationMethod[Any, Any]:
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

    reg.add_computation(
        FitterMethod(
            target="cdf",
            sources=("pdf",),
            fitter=cast(
                Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                fit_pdf_to_cdf,
            ),
        ),
        options_descriptor=EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            characteristic_options=(
                CharacteristicOption(name="cdf_bias", type=float, default=0.0),
            ),
        ),
    )
    reg.add_computation(
        FitterMethod(
            target="ppf",
            sources=("cdf",),
            fitter=cast(
                Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                fit_cdf_to_ppf,
            ),
        ),
        options_descriptor=EdgeOptionsDescriptor(
            name="cdf_to_ppf",
            characteristic_options=(CharacteristicOption(name="eps", type=float, default=0.0),),
        ),
    )
    reg.add_computation(fit_trivial("pdf", ("cdf",)))
    reg.add_computation(fit_trivial("cdf", ("ppf",)))
    return reg


class TestCharacteristicOptionsBroadcast:
    """Tests for Fix 1: characteristic_options are broadcast to all steps."""

    def test_characteristic_options_broadcast_to_all_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        ``characteristic_options`` passed to ``query_method`` must be applied
        to every step that declares a matching ``CharacteristicOption``,
        without the caller needing to know step indices.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_char_options_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        # Pass characteristic options as a shared dict — no step indices needed.
        ppf = strategy.query_method(
            "ppf",
            distr,
            characteristic_options={"cdf_bias": 0.3, "eps": 1e-4},
        )

        # cdf_bias=0.3 was applied to step 0 (pdf->cdf).
        # eps=1e-4 was applied to step 1 (cdf->ppf).
        assert ppf(0.0) == pytest.approx(1e-4)
        assert fit_calls["pdf_to_cdf"] == 1
        assert fit_calls["cdf_to_ppf"] == 1

    def test_characteristic_options_broadcast_produces_independent_cache_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Different ``characteristic_options`` values must produce independent
        cache entries because they affect the meaning of the result.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_char_options_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=True)

        ppf_a = strategy.query_method("ppf", distr, characteristic_options={"eps": 1e-3})
        ppf_b = strategy.query_method("ppf", distr, characteristic_options={"eps": 2e-3})
        ppf_a_again = strategy.query_method("ppf", distr, characteristic_options={"eps": 1e-3})

        assert ppf_a(0.0) == pytest.approx(1e-3)
        assert ppf_b(0.0) == pytest.approx(2e-3)
        # Same eps -> same cached fitted method.
        assert ppf_a_again is ppf_a
        # Two distinct eps values -> two fit calls for cdf_to_ppf.
        assert fit_calls["cdf_to_ppf"] == 2

    def test_per_step_override_takes_precedence_over_broadcast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A per-step value in ``options`` must override the shared
        ``characteristic_options`` for that specific step.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_char_options_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        plan = strategy.explain_computation_path("ppf", distr)
        # Per-step override for step 1 (cdf->ppf): eps=5e-3.
        # Broadcast has eps=1e-3 — the per-step value must win.
        per_step = plan.with_options(1, eps=5e-3)
        ppf = strategy.query_method(
            "ppf",
            distr,
            per_step,
            characteristic_options={"eps": 1e-3},
        )

        assert ppf(0.0) == pytest.approx(5e-3)

    def test_characteristic_options_fall_back_to_declared_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When ``characteristic_options`` does not supply a value for an option,
        the declared ``CharacteristicOption.default`` is used.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_char_options_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        # No characteristic_options supplied -> defaults (eps=0.0) are used.
        ppf = strategy.query_method("ppf", distr)
        assert ppf(0.0) == pytest.approx(0.0)

    def test_distribution_query_method_forwards_characteristic_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        ``Distribution.query_method`` must forward ``characteristic_options``
        to the underlying strategy.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_char_options_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        distr = distr.with_computation_strategy(DefaultComputationStrategy(enable_caching=False))

        ppf = distr.query_method("ppf", characteristic_options={"eps": 7e-4})
        assert ppf(0.0) == pytest.approx(7e-4)

    def test_characteristic_options_propagate_through_recursive_query_method(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When a fitter on step N calls ``distribution.query_method(intermediate)``
        internally, the characteristic options from the outer call must be
        automatically inherited by that recursive call — the fitter does not
        need to forward them explicitly.

        Setup: ``pdf -> cdf -> ppf`` where the ``cdf -> ppf`` fitter
        explicitly calls ``strategy.query_method("cdf", distribution)``
        to obtain the CDF.  The CDF fitter encodes ``cdf_bias`` into its
        result.  We verify that the ``cdf_bias`` supplied via
        ``characteristic_options`` reaches the CDF fitter even though the
        PPF fitter does not forward it.
        """
        fit_calls: dict[str, int] = {}

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

        # Capture strategy reference so the fitter can call query_method.
        strategy_ref: list[DefaultComputationStrategy] = []

        def fit_cdf_to_ppf(
            distribution: Distribution, **_kwargs: Any
        ) -> FittedComputationMethod[Any, Any]:
            """
            This fitter calls strategy.query_method("cdf") internally.
            It does NOT forward characteristic_options — the strategy must
            propagate them automatically via the _char_options_stack.
            """
            fit_calls["cdf_to_ppf"] = fit_calls.get("cdf_to_ppf", 0) + 1
            # Deliberately call without characteristic_options — they must
            # be inherited from the outer query_method call automatically.
            cdf_method = strategy_ref[0].query_method("cdf", distribution)
            captured = cdf_method(0.0)

            def ppf(_q: float, **_o: Any) -> float:
                return captured

            return FittedComputationMethod(
                target="ppf",
                sources=("cdf",),
                func=cast(Callable[[float, KwArg(Any)], float], ppf),
            )

        def _trivial(target: str, sources: tuple[str, ...]) -> FitterMethod:
            def _fit(_d: Distribution, **_kw: Any) -> FittedComputationMethod[Any, Any]:
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

        reg.add_computation(
            FitterMethod(
                target="cdf",
                sources=("pdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    fit_pdf_to_cdf,
                ),
            ),
            options_descriptor=EdgeOptionsDescriptor(
                name="pdf_to_cdf",
                characteristic_options=(
                    CharacteristicOption(name="cdf_bias", type=float, default=0.0),
                ),
            ),
        )
        reg.add_computation(
            FitterMethod(
                target="ppf",
                sources=("cdf",),
                fitter=cast(
                    Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                    fit_cdf_to_ppf,
                ),
            ),
        )
        reg.add_computation(_trivial("pdf", ("cdf",)))
        reg.add_computation(_trivial("cdf", ("ppf",)))

        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)
        strategy_ref.append(strategy)

        # Pass cdf_bias=0.3 via characteristic_options.
        # The PPF fitter calls query_method("cdf") without forwarding it —
        # the stack must propagate it automatically.
        ppf = strategy.query_method("ppf", distr, characteristic_options={"cdf_bias": 0.3})

        # The CDF fitter received cdf_bias=0.3 -> cdf(0) = 0.8.
        # The PPF fitter captured that value -> ppf(anything) = 0.8.
        assert ppf(0.0) == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Fix 2: computation_defaults layer
# --------------------------------------------------------------------------- #


def _build_computation_defaults_registry(
    fit_calls: dict[str, int],
) -> CharacteristicRegistry:
    """
    Registry with ``pdf -> cdf`` where the edge carries a *computation* option
    ``limit`` (default 10).  Used to verify the computation_defaults layer.
    """
    reg = CharacteristicRegistry()
    reg.add_characteristic("pdf", is_definitive=True)
    reg.add_characteristic("cdf", is_definitive=True)

    def fit_pdf_to_cdf(
        _distribution: Distribution, **kwargs: Any
    ) -> FittedComputationMethod[Any, Any]:
        fit_calls["pdf_to_cdf"] = fit_calls.get("pdf_to_cdf", 0) + 1
        limit = kwargs.get("limit", 10)

        def cdf(_x: float, **_o: Any) -> float:
            return float(limit)

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

    reg.add_computation(
        FitterMethod(
            target="cdf",
            sources=("pdf",),
            fitter=cast(
                Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                fit_pdf_to_cdf,
            ),
        ),
        options_descriptor=EdgeOptionsDescriptor(
            name="pdf_to_cdf",
            computation_options=(ComputationOption(name="limit", type=int, default=10),),
        ),
    )
    reg.add_computation(
        FitterMethod(
            target="pdf",
            sources=("cdf",),
            fitter=cast(
                Callable[[Distribution, KwArg(Any)], FittedComputationMethod[Any, Any]],
                fit_cdf_to_pdf,
            ),
        )
    )
    return reg


class TestComputationDefaultsLayer:
    """Tests for Fix 2: computation_defaults layer."""

    def test_strategy_level_computation_defaults_override_hardcoded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        ``computation_defaults`` set at strategy construction time must
        override the hardcoded ``ComputationOption.default``.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_computation_defaults_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        # Strategy-level default: limit=99 (overrides hardcoded 10).
        strategy = DefaultComputationStrategy(
            enable_caching=False,
            computation_defaults={"limit": 99},
        )

        cdf = strategy.query_method("cdf", distr)
        # The fitter encoded ``limit`` as the return value.
        assert cdf(0.0) == pytest.approx(99.0)

    def test_per_call_computation_defaults_override_strategy_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Per-call ``computation_defaults`` must override the strategy-level
        ``computation_defaults``.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_computation_defaults_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(
            enable_caching=False,
            computation_defaults={"limit": 99},
        )

        # Per-call override: limit=42 wins over strategy-level 99.
        cdf = strategy.query_method("cdf", distr, computation_defaults={"limit": 42})
        assert cdf(0.0) == pytest.approx(42.0)

    def test_per_step_override_takes_precedence_over_computation_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A per-step value in ``options`` must override both strategy-level and
        per-call ``computation_defaults``.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_computation_defaults_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(
            enable_caching=False,
            computation_defaults={"limit": 99},
        )

        plan = strategy.explain_computation_path("cdf", distr)
        per_step = plan.with_options(0, limit=7)
        cdf = strategy.query_method(
            "cdf",
            distr,
            per_step,
            computation_defaults={"limit": 42},
        )
        # Per-step value 7 wins over per-call 42 and strategy-level 99.
        assert cdf(0.0) == pytest.approx(7.0)

    def test_hardcoded_default_used_when_no_defaults_supplied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When neither strategy-level nor per-call defaults are supplied,
        the hardcoded ``ComputationOption.default`` is used.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_computation_defaults_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        strategy = DefaultComputationStrategy(enable_caching=False)

        cdf = strategy.query_method("cdf", distr)
        # Hardcoded default is 10.
        assert cdf(0.0) == pytest.approx(10.0)

    def test_distribution_calculate_characteristic_forwards_computation_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        ``Distribution.calculate_characteristic`` must forward
        ``computation_defaults`` to the underlying strategy.
        """
        fit_calls: dict[str, int] = {}
        reg = _build_computation_defaults_registry(fit_calls)
        distr = _make_pdf_distribution()
        monkeypatch.setattr(strategies_module, "characteristic_registry", lambda: reg)
        distr = distr.with_computation_strategy(DefaultComputationStrategy(enable_caching=False))

        result = distr.calculate_characteristic("cdf", 0.0, computation_defaults={"limit": 55})
        assert result == pytest.approx(55.0)
