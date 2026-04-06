"""
Simple Inverse-Transform Sampling Strategy
==========================================

This module provides a basic univariate sampler based on inverse transform
sampling (also known as the quantile/PPF method). It is used as a fallback
when advanced sampling methods (e.g. UNU.RAN) are not available.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pysatl_core.distributions.strategies import SamplingStrategy
from pysatl_core.types import CharacteristicName, NumericArray

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


class DefaultSamplingUnivariateStrategy(SamplingStrategy):
    """
    Default univariate sampler based on inverse transform sampling.

    This strategy generates samples by applying the PPF (inverse CDF)
    to uniformly distributed random variables.

    Notes
    -----
    - Requires the distribution to provide a PPF computation method.
    - Assumes that the PPF follows NumPy semantics (vectorized evaluation).
    - Graph-derived PPFs (scalar-only) are currently not supported.
    - Returns a NumPy array containing the generated samples.
    """

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray:
        """
        Generate samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        distr : Distribution
            Distribution to sample from.
        **options : Any
            Additional options forwarded to the PPF computation.

        Returns
        -------
        NumericArray
            NumPy array containing ``n`` generated samples.
            The exact array shape depends on the distribution and sampling strategy.
        """
        ppf = distr.query_method(CharacteristicName.PPF, **options)
        rng = np.random.default_rng()
        U = rng.random(n)
        # TODO: Now it will be based on the fact that the characteristic
        #  has NumPy semantics (It is much more faster), that is,
        #  it will not work with the graph computed characteristics currently.
        samples = ppf(U)
        return cast(NumericArray, samples)
