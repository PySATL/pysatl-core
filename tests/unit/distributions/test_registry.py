from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest

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
from pysatl_core.types import CharacteristicName
from tests.unit.distributions.test_basic import DistributionTestBase


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
        strategy = DefaultComputationStrategy[float, float](enable_caching=False)
        ppf = strategy.query_method(CharacteristicName.PPF, distr)
        cdf = strategy.query_method(CharacteristicName.CDF, distr)
        qs = np.linspace(1e-6, 1.0 - 1e-6, 7)
        errs = [abs(float(cdf(float(ppf(float(q))))) - q) for q in qs]
        assert max(errs) < 5e-3

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

        strategy = DefaultComputationStrategy[float, float](enable_caching=False)
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

        # non-unary validation
        with pytest.raises(ValueError):
            reg.add_computation(
                self.make_fictitious_computation_method(target="b", sources=["a", "b"])
            )

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

        path = view.find_path("src", "dst", prefer_label="fast")
        assert path == [alternative_method]

        path = view.find_path("src", "dst")
        assert path == [default_method]
