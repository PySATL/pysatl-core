from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions import DefaultComputationStrategy
from pysatl_core.distributions.computation import ComputationMethod, FittedComputationMethod
from pysatl_core.distributions.registry import (
    DEFAULT_COMPUTATION_KEY,
    CharacteristicRegistry,
    EdgeConstraint,
    GraphInvariantError,
    NumericConstraint,
    characteristic_registry,
    reset_characteristic_registry,
)
from tests.unit.distributions.test_basic import DistributionTestBase


# Helper: make a fitted method that returns a constant value regardless of input.
def _fitted_const(val: Any) -> FittedComputationMethod[Any, Any]:
    def _impl(*_args: Any, **_kwargs: Any) -> Any:
        return val

    return cast(FittedComputationMethod[Any, Any], _impl)


class TestCharacteristicRegistry(DistributionTestBase):
    # --------------------- configuration sanity ---------------------

    def test_configuration_continuous_presence_and_connectivity(self) -> None:
        """
        For a continuous 1D distribution we at least expect CDF/PPF to be present
        and mutually reachable; verify CDF(PPF(q)) ≈ q.
        """
        reg = characteristic_registry()
        distr = self.make_logistic_cdf_distribution()

        view = reg.view(distr)

        # Presence / definitiveness (pdf may be absent by config; don't require it)
        assert self.PMF not in view.all_characteristics
        assert {self.CDF, self.PPF}.issubset(view.all_characteristics)
        assert {self.CDF, self.PPF}.issubset(view.definitive_characteristics)

        # Paths must exist between definitive characteristics we rely on
        assert view.find_path(self.CDF, self.PPF) is not None
        assert view.find_path(self.PPF, self.CDF) is not None

        # Strategy resolves and roundtrips: CDF(PPF(q)) ~ q
        strategy = DefaultComputationStrategy[float, float](enable_caching=False)
        ppf = strategy.query_method(self.PPF, distr)
        cdf = strategy.query_method(self.CDF, distr)
        qs = np.linspace(1e-6, 1.0 - 1e-6, 7)
        errs = [abs(float(cdf(float(ppf(float(q))))) - q) for q in qs]
        assert max(errs) < 5e-3

    def test_configuration_discrete_requires_support_then_ok(self) -> None:
        """
        Discrete edges require support; without it invariants fail, with it they pass.
        Also validate pmf->cdf values and cdf->ppf thresholds.
        """
        reg = characteristic_registry()
        distr = self.make_discrete_point_pmf_distribution()

        # Without support, discrete edges (pmf<->cdf, cdf<->ppf) are disabled -> invariants fail
        with pytest.raises(GraphInvariantError):
            reg.view(distr)

        # Add support and try again
        distr._support = self.make_discrete_support()  # type: ignore[attr-defined]
        view = reg.view(distr)

        # Presence / definitiveness
        assert self.PDF not in view.all_characteristics
        assert {self.PMF, self.CDF, self.PPF}.issubset(view.all_characteristics)
        assert {self.PMF, self.CDF, self.PPF}.issubset(view.definitive_characteristics)

        # Conversions exist
        assert view.find_path(self.PMF, self.CDF) is not None
        assert view.find_path(self.CDF, self.PMF) is not None
        assert view.find_path(self.CDF, self.PPF) is not None
        assert view.find_path(self.PPF, self.CDF) is not None

        strategy = DefaultComputationStrategy[float, float](enable_caching=False)
        cdf = strategy.query_method(self.CDF, distr)
        assert cdf(0.0) == pytest.approx(0.2, abs=1e-10)
        assert cdf(1.0) == pytest.approx(0.7, abs=1e-10)
        assert cdf(2.0) == pytest.approx(1.0, abs=1e-10)

        ppf = strategy.query_method(self.PPF, distr)
        assert ppf(0.10) == pytest.approx(0.0, abs=1e-12)
        assert ppf(0.70) == pytest.approx(1.0, abs=1e-12)
        assert ppf(0.95) == pytest.approx(2.0, abs=1e-12)

    # --------------------- constraints & errors ---------------------

    def test_edge_dims_constraint_filters_edges(self) -> None:
        """
        Edge-level dims constraint: if an edge requires dim==2 but profile is dim==1,
        the definitive subgraph (with >=2 nodes) becomes not strongly connected -> invariant fails.
        """
        reg = CharacteristicRegistry()
        reg.add_characteristic("A", is_definitive=True)
        reg.add_characteristic("B", is_definitive=True)

        cons_dim2 = EdgeConstraint(dims=NumericConstraint(allowed=frozenset({2})))
        m_ab: ComputationMethod[Any, Any] = ComputationMethod(
            target="B", sources=["A"], fitter=lambda *_a, **_k: _fitted_const(None)
        )
        m_ba: ComputationMethod[Any, Any] = ComputationMethod(
            target="A", sources=["B"], fitter=lambda *_a, **_k: _fitted_const(None)
        )
        reg.add_computation(m_ab, constraint=cons_dim2)
        reg.add_computation(m_ba, constraint=cons_dim2)

        # For a 1D profile the edges are filtered out → not strongly connected
        with pytest.raises(GraphInvariantError):
            reg.view(self.make_logistic_cdf_distribution())

    def test_add_computation_validation_and_duplicate_rules(self) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("a", is_definitive=True)
        reg.add_characteristic("b", is_definitive=True)

        # non-unary validation
        m_bad: ComputationMethod[Any, Any] = ComputationMethod(
            target="b", sources=["a", "b"], fitter=lambda *_a, **_k: _fitted_const(None)
        )
        with pytest.raises(ValueError):
            reg.add_computation(m_bad)

        # undeclared nodes
        reg2 = CharacteristicRegistry()
        m: ComputationMethod[Any, Any] = ComputationMethod(
            target="y", sources=["x"], fitter=lambda *_a, **_k: _fitted_const(None)
        )
        with pytest.raises(ValueError):
            reg2.add_computation(m)

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

    # --------------------- invariants: explicit breaks ---------------------

    @pytest.mark.parametrize(
        "case",
        ["defs_not_strongly_connected", "indef_not_reachable", "indef_to_def_cycle"],
    )
    def test_invariant_violations(self, case: str) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("A", is_definitive=True)
        reg.add_characteristic("B", is_definitive=True)
        reg.add_characteristic("X", is_definitive=False)

        m_ab: ComputationMethod[Any, Any] = ComputationMethod(
            target="B", sources=["A"], fitter=lambda *_a, **_k: _fitted_const(None)
        )
        m_ba: ComputationMethod[Any, Any] = ComputationMethod(
            target="A", sources=["B"], fitter=lambda *_a, **_k: _fitted_const(None)
        )
        m_xa: ComputationMethod[Any, Any] = ComputationMethod(
            target="A", sources=["X"], fitter=lambda *_a, **_k: _fitted_const(None)
        )

        if case == "defs_not_strongly_connected":
            reg.add_computation(m_ab)  # missing B->A
        elif case == "indef_not_reachable":
            reg.add_computation(m_ab)
            reg.add_computation(m_ba)
            # X has no incoming edge from defs/indefs
        elif case == "indef_to_def_cycle":
            reg.add_computation(m_ab)
            reg.add_computation(m_ba)
            reg.add_computation(m_xa)  # X -> A is forbidden by invariant (3)
        else:
            raise AssertionError("unexpected case")

        with pytest.raises(GraphInvariantError):
            reg.view(self.make_logistic_cdf_distribution())

    # --------------------- labels ---------------------

    def test_label_variants_and_picking(self) -> None:
        reg = CharacteristicRegistry()
        reg.add_characteristic("src", is_definitive=True)
        reg.add_characteristic("dst", is_definitive=True)

        m_default: ComputationMethod[Any, Any] = ComputationMethod(
            target="dst", sources=["src"], fitter=lambda *_a, **_k: _fitted_const("default")
        )
        m_fast: ComputationMethod[Any, Any] = ComputationMethod(
            target="dst", sources=["src"], fitter=lambda *_a, **_k: _fitted_const("fast")
        )
        # reverse edge to keep the definitive subgraph strongly connected
        m_back: ComputationMethod[Any, Any] = ComputationMethod(
            target="src", sources=["dst"], fitter=lambda *_a, **_k: _fitted_const("back")
        )

        reg.add_computation(m_default)
        reg.add_computation(m_fast, label="fast")
        reg.add_computation(m_back)

        view = reg.view(self.make_logistic_cdf_distribution())

        variants = view.variants("src", "dst")
        assert set(variants.keys()) == {DEFAULT_COMPUTATION_KEY, "fast"}

        path = view.find_path("src", "dst", prefer_label="fast")
        assert path == [m_fast]

        path = view.find_path("src", "dst")
        assert path == [m_default]
