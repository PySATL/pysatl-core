"""
Bootstrap inference for statistical functionals.

Supports classical resampling (sampling with replacement from the empirical
distribution) and smooth resampling (sampling from a KDE-fitted continuous
approximation).
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from pysatl_core.distributions.empirical.distribution import EmpiricalDistribution, EmpiricalMethod, ScipyGaussianKde


class StatisticalFunctional(Protocol):
    """A function that maps a sample to a scalar summary."""

    def __call__(self, sample: NDArray[np.float64]) -> float: ...


class ResamplingMethod(Protocol):
    """Strategy for generating a single bootstrap resample from data."""

    def resample(self, data: NDArray[np.float64], size: int) -> NDArray[np.float64]: ...


class ClassicalResampling:
    """Sample with replacement from the original data (discrete empirical distribution)."""

    def __init__(self, rng: Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

    def resample(self, data: NDArray[np.float64], size: int) -> NDArray[np.float64]:
        return self._rng.choice(data, size=size, replace=True)


class SmoothResampling:
    """Sample from a KDE-fitted continuous approximation of the empirical distribution.

    The fitted distribution is cached after the first resample call and reused
    across bootstrap iterations for the same data array.
    """

    def __init__(self, method: EmpiricalMethod = ScipyGaussianKde()) -> None:
        self._method = method
        self._distr: EmpiricalDistribution | None = None

    def resample(self, data: NDArray[np.float64], size: int) -> NDArray[np.float64]:
        if self._distr is None or self._distr.data is not data:
            self._distr = EmpiricalDistribution(data, method=self._method)
        return self._distr.sample(size)


@dataclass
class BootstrapResult:
    """Outcome of a bootstrap run for a single statistical functional.

    Parameters
    ----------
    observed:
        Value of the functional on the original data.
    replicates:
        Array of shape ``(B,)`` holding the functional value on each
        bootstrap resample.
    """

    observed: float
    replicates: NDArray[np.float64]

    def standard_error(self) -> float:
        """Standard deviation of the bootstrap replicates."""
        return float(self.replicates.std())

    def bias(self) -> float:
        """Estimated bias: mean of replicates minus the observed value."""
        return float(self.replicates.mean() - self.observed)

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Percentile bootstrap confidence interval.

        Parameters
        ----------
        level:
            Coverage level in the open interval (0, 1). Default is 0.95.

        Returns
        -------
        (lower, upper) bounds of the interval.

        TODO: implement BCa (bias-corrected and accelerated) and Normal
        approximation methods for better accuracy in skewed or small-sample
        settings.
        """
        alpha = 1.0 - level
        lo, hi = np.percentile(self.replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)


class Bootstrap:
    """Bootstrap procedure for estimating the sampling distribution of a functional.

    Parameters
    ----------
    data:
        One-dimensional array of observed values.
    B:
        Number of bootstrap resamples. Default is 1000.
    method:
        Strategy for generating each resample. Defaults to
        :class:`ClassicalResampling` (sampling with replacement).
    rng:
        NumPy random generator. A fresh generator is created when ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.normal(0, 1, 200)
    >>> result = Bootstrap(data, B=500, rng=rng).run(np.mean)
    >>> lo, hi = result.confidence_interval()
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        B: int = 1000,
        method: ResamplingMethod | None = None,
        rng: Generator | None = None,
    ) -> None:
        self._data = np.asarray(data, dtype=float)
        self._B = B
        self._rng = rng if rng is not None else np.random.default_rng()
        self._method = method if method is not None else ClassicalResampling(rng=self._rng)

    def run(self, functional: StatisticalFunctional) -> BootstrapResult:
        """Run the bootstrap and return a :class:`BootstrapResult`.

        Parameters
        ----------
        functional:
            A callable that accepts a sample array and returns a scalar.
        """
        observed = float(functional(self._data))
        n = len(self._data)
        replicates = np.array(
            [
                float(functional(self._method.resample(self._data, n)))
                for _ in range(self._B)
            ]
        )
        return BootstrapResult(observed=observed, replicates=replicates)


__all__ = [
    "Bootstrap",
    "BootstrapResult",
    "ClassicalResampling",
    "ResamplingMethod",
    "SmoothResampling",
    "StatisticalFunctional",
]
