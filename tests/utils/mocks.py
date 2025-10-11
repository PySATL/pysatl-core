from typing import Any

import numpy as np

from pysatl_core.distributions import (
    ArraySample,
    Distribution,
    Sample,
    SamplingStrategy,
)


class MockSamplingStrategy(SamplingStrategy):
    def sample(self, n: int, distr: Distribution, **options: Any) -> Sample:
        return ArraySample(np.random.random((n, 1)))
