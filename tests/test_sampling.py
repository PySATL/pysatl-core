import numpy as np
import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.distribution import StandaloneEuclideanUnivariateDistribution
from pysatl_core.types import Kind

PPF = "ppf"


def test_sample_uniform_ppf_only_shape_bounds_and_mean() -> None:
    distr = StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=[
            AnalyticalComputation[float, float](target=PPF, func=lambda q: q),
        ],
    )

    n = 1000
    sample = distr.sample(n)

    assert sample.shape == (n, 1)
    arr = sample.array
    assert np.isfinite(arr).all()
    assert ((arr >= 0.0) & (arr <= 1.0)).all()

    mean = float(arr.mean())
    assert mean == pytest.approx(0.5, abs=0.1)
