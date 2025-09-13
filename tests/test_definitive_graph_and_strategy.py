from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import StandaloneEuclideanUnivariateDistribution
from pysatl_core.distributions.registry import distribution_type_register
from pysatl_core.types import EuclideanDistributionType, Kind

PDF = "pdf"
CDF = "cdf"
PPF = "ppf"


def test_definitive_graph_full_connectivity() -> None:
    dt = EuclideanDistributionType(kind=Kind.CONTINUOUS, dimension=1)
    reg = distribution_type_register().get(dt)

    defs = reg.definitive_nodes()
    assert {PDF, CDF, PPF}.issubset(defs)

    for src in (PDF, CDF, PPF):
        for dst in (PDF, CDF, PPF):
            path = reg.find_path(src, dst)
            assert path is not None, f"no path {src!r} -> {dst!r}"


def test_strategy_uniform_ppf_only() -> None:
    distr = StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=[
            AnalyticalComputation[float, float](target=PPF, func=lambda q: q),
        ],
    )

    cdf = distr.computation_strategy.query_method(CDF, distr)
    pdf = distr.computation_strategy.query_method(PDF, distr)
    ppf = distr.computation_strategy.query_method(PPF, distr)

    assert ppf(0.3) == pytest.approx(0.3, rel=1e-12, abs=1e-12)

    for x, expected in [(-0.5, 0.0), (0.2, 0.2), (0.9, 0.9), (1.5, 1.0)]:
        assert cdf(x) == pytest.approx(expected, rel=5e-3, abs=5e-4)

    for x, expected in [(0.25, 1.0), (0.75, 1.0), (-0.1, 0.0), (1.1, 0.0)]:
        assert pdf(x) == pytest.approx(expected, rel=5e-3, abs=5e-3)


def test_strategy_uniform_pdf_only() -> None:
    def pdf_func(x: float) -> float:
        return 1.0 if (0.0 <= x <= 1.0) else 0.0

    distr = StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=[
            AnalyticalComputation[float, float](target=PDF, func=pdf_func),
        ],
    )

    cdf = distr.computation_strategy.query_method(CDF, distr)
    ppf = distr.computation_strategy.query_method(PPF, distr)

    assert cdf(0.3) == pytest.approx(0.3, rel=5e-3, abs=5e-4)
    assert cdf(0.8) == pytest.approx(0.8, rel=5e-3, abs=5e-4)
    assert cdf(-2.0) == pytest.approx(0.0, abs=1e-6)
    assert cdf(3.0) == pytest.approx(1.0, abs=1e-6)

    for q in (0.1, 0.5, 0.9):
        assert ppf(q) == pytest.approx(q, rel=5e-3, abs=5e-4)


def _normal_pdf(mu: float, sigma: float) -> Callable[[float], float]:
    inv = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    def pdf(x: float) -> float:
        z = (x - mu) / sigma
        return inv * math.exp(-0.5 * z * z)

    return pdf


@pytest.mark.parametrize("mu, sigma", [(1.5, 0.7)])
def test_normal_with_pdf_only(mu: float, sigma: float) -> None:
    distr = StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=[
            AnalyticalComputation[float, float](target=PDF, func=_normal_pdf(mu, sigma)),
        ],
    )

    pdf = distr.computation_strategy.query_method(PDF, distr)
    cdf = distr.computation_strategy.query_method(CDF, distr)
    ppf = distr.computation_strategy.query_method(PPF, distr)

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


def test_cdf_to_ppf_respects_most_left_option_on_plateau() -> None:
    def plateau_cdf(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x < 1.0:
            return 0.5
        return 1.0

    distr = StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=[
            AnalyticalComputation[float, float](target=CDF, func=plateau_cdf),
        ],
    )

    ppf_rightmost = distr.computation_strategy.query_method("ppf", distr)
    ppf_leftmost = distr.computation_strategy.query_method("ppf", distr, most_left=True)

    q = 0.5
    x_right = float(ppf_rightmost(q))
    x_left = float(ppf_leftmost(q))

    assert x_left == pytest.approx(0.0, abs=1e-9)
    assert x_right == pytest.approx(1.0, abs=1e-9)
