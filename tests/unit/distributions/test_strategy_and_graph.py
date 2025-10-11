from __future__ import annotations

import math

import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import StandaloneEuclideanUnivariateDistribution
from pysatl_core.distributions.registry import distribution_type_register
from pysatl_core.types import EuclideanDistributionType, Kind
from tests.unit.distributions.test_basic import DistributionTestBase


class TestDefinitiveGraph(DistributionTestBase):
    def test_full_connectivity(self) -> None:
        dt = EuclideanDistributionType(kind=Kind.CONTINUOUS, dimension=1)
        registry = distribution_type_register().get(dt)

        definitive = registry.definitive_nodes()
        assert {self.PDF, self.CDF, self.PPF}.issubset(definitive)

        for src in (self.PDF, self.CDF, self.PPF):
            for dst in (self.PDF, self.CDF, self.PPF):
                path = registry.find_path(src, dst)
                assert path is not None, f"no path {src!r} -> {dst!r}"


class TestComputationStrategy(DistributionTestBase):
    def test_uniform_ppf_only(self) -> None:
        distr = self.make_uniform_ppf_distribution()

        cdf = distr.computation_strategy.query_method(self.CDF, distr)
        pdf = distr.computation_strategy.query_method(self.PDF, distr)
        ppf = distr.computation_strategy.query_method(self.PPF, distr)

        assert ppf(0.3) == pytest.approx(0.3, rel=1e-12, abs=1e-12)

        for x, expected in [(-0.5, 0.0), (0.2, 0.2), (0.9, 0.9), (1.5, 1.0)]:
            assert cdf(x) == pytest.approx(expected, rel=5e-3, abs=5e-4)

        for x, expected in [(0.25, 1.0), (0.75, 1.0), (-0.1, 0.0), (1.1, 0.0)]:
            assert pdf(x) == pytest.approx(expected, rel=5e-3, abs=5e-3)

    def test_uniform_pdf_only(self) -> None:
        distr = self.make_uniform_pdf_distribution()

        cdf = distr.computation_strategy.query_method(self.CDF, distr)
        ppf = distr.computation_strategy.query_method(self.PPF, distr)

        assert cdf(0.3) == pytest.approx(0.3, rel=5e-3, abs=5e-4)
        assert cdf(0.8) == pytest.approx(0.8, rel=5e-3, abs=5e-4)
        assert cdf(-2.0) == pytest.approx(0.0, abs=1e-6)
        assert cdf(3.0) == pytest.approx(1.0, abs=1e-6)

        for q in (0.1, 0.5, 0.9):
            assert ppf(q) == pytest.approx(q, rel=5e-3, abs=5e-4)

    @pytest.mark.parametrize("mu, sigma", [(1.5, 0.7)])
    def test_normal_with_pdf_only(self, mu: float, sigma: float) -> None:
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations=[
                AnalyticalComputation[float, float](
                    target=self.PDF, func=self.make_normal_pdf_function(mu, sigma)
                ),
            ],
        )

        pdf = distr.computation_strategy.query_method(self.PDF, distr)
        cdf = distr.computation_strategy.query_method(self.CDF, distr)
        ppf = distr.computation_strategy.query_method(self.PPF, distr)

        expected_pdf_mu = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
        assert pdf(mu) == pytest.approx(expected_pdf_mu, rel=5e-3, abs=5e-4)

        def cdf_closed(x: float) -> float:
            return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))

        assert cdf(mu) == pytest.approx(0.5, abs=2e-3)

        x1 = mu + sigma
        assert cdf(x1) == pytest.approx(cdf_closed(x1), rel=5e-3, abs=5e-4)

        assert ppf(0.5) == pytest.approx(mu, rel=5e-3, abs=5e-3)

        q1 = cdf_closed(mu + sigma)
        assert ppf(q1) == pytest.approx(mu + sigma, rel=7e-3, abs=7e-3)

    def test_cdf_to_ppf_respects_most_left_option_on_plateau(self) -> None:
        distr = self.make_plateau_cdf_distribution()

        ppf_rightmost = distr.computation_strategy.query_method(self.PPF, distr)
        ppf_leftmost = distr.computation_strategy.query_method(self.PPF, distr, most_left=True)

        q = 0.5
        x_right = float(ppf_rightmost(q))
        x_left = float(ppf_leftmost(q))

        assert x_left == pytest.approx(0.0, abs=1e-9)
        assert x_right == pytest.approx(1.0, abs=1e-9)
