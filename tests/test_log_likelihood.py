import numpy as np
import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import StandaloneEuclideanUnivariateDistribution
from pysatl_core.distributions.sampling import ArraySample
from pysatl_core.types import Kind

PDF = "pdf"


def _pdf_uniform_scalar(x: float) -> float:
    return 1.0 if (0.0 <= x <= 1.0) else 0.0


def _make_uniform_pdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=[
            AnalyticalComputation[float, float](target=PDF, func=_pdf_uniform_scalar),
        ],
    )


def test_log_likelihood_uniform_all_in_support_is_zero() -> None:
    distr = _make_uniform_pdf_distribution()
    arr = np.array([[0.1], [0.9], [0.3]], dtype=np.float64)  # shape (n, 1)
    sample = ArraySample(arr)
    # log L = sum log(1) = 0
    assert distr.log_likelihood(sample) == pytest.approx(0.0, abs=1e-12)


def test_log_likelihood_uniform_out_of_support_is_minus_inf() -> None:
    distr = _make_uniform_pdf_distribution()
    arr = np.array([[0.1], [1.5], [0.3]], dtype=np.float64)  # есть точка вне носителя
    sample = ArraySample(arr)
    assert np.isneginf(distr.log_likelihood(sample))
