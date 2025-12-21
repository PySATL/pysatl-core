"""
UNU.RAN API Specification
========================

This module defines the complete API specification for UNU.RAN integration.
The interfaces described here will be implemented through C bindings to the
UNU.RAN library.

The API is designed to integrate seamlessly with the existing distribution
sampling infrastructure in pysatl_core.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.sampling import Sample


class UnuranMethod(StrEnum):
    """
    UNU.RAN sampling methods.

    Methods are categorized by their approach:
    - Inversion methods: use inverse CDF
    - Rejection methods: accept/reject based on envelope
    - Transformation methods: transform from simpler distributions
    - Numerical inversion: approximate inverse CDF
    - Specialized methods: optimized for specific distribution types
    """

    # Automatic method selection
    AUTO = "auto"

    # Inversion methods
    PINV = "pinv"  # Polynomial interpolation inversion
    TDR = "tdr"  # Transformed density rejection

    # Rejection methods
    ARS = "ars"  # Adaptive rejection sampling
    AROU = "arou"  # Automatic ratio-of-uniforms
    HINV = "hinv"  # Hermite interpolation inversion
    NINV = "ninv"  # Numerical inversion

    # Transformation methods
    SROU = "srou"  # Simple ratio-of-uniforms
    SSROU = "ssrou"  # Striped simple ratio-of-uniforms

    # Discrete methods
    DARI = "dari"  # Discrete automatic rejection inversion
    DAU = "dau"  # Discrete automatic universal
    DGT = "dgt"  # Discrete guide table method
    DSROU = "dsrou"  # Discrete simple ratio-of-uniforms

    # Specialized continuous methods
    CEMP = "cemp"  # Continuous empirical distribution
    EMPK = "empk"  # Empirical distribution with kernel smoothing


@dataclass(frozen=True, slots=True)
class UnuranMethodConfig:
    """
    Configuration for a UNU.RAN sampling method.

    Parameters
    ----------
    method : UnuranMethod
        The sampling method to use. If ``UnuranMethod.AUTO``, UNU.RAN will
        automatically select the best method based on available distribution
        characteristics.
    method_params : dict[str, Any], optional
        Method-specific parameters. The exact parameters depend on the chosen
        method. Common parameters include:
        - ``accuracy``: target accuracy for numerical methods
        - ``max_iterations``: maximum iterations for iterative methods
        - ``grid_size``: grid size for interpolation methods
        - ``smooth``: smoothing parameter for kernel methods
    use_ppf : bool, default False
        If ``True``, prefer using PPF (inverse CDF) when available. This is
        typically the fastest method for univariate distributions.
    use_pdf : bool, default True
        If ``True``, allow using PDF for rejection-based methods.
    use_cdf : bool, default False
        If ``True``, allow using CDF for inversion-based methods.
    use_registry_characteristics : bool, default True
        If ``True``, allow using distribution characteristics from the registry
        in addition to those directly available in the Distribution object.
        If ``False``, use only characteristics that are directly available
        in the Distribution object, without querying the registry.
    seed : int | None, default None
        Random seed for the sampler. If ``None``, uses system entropy.

    Notes
    -----
    - Method-specific parameters are validated when the sampler is created
    - Some methods require specific characteristics (e.g., rejection methods
      typically need PDF)
    - The ``use_*`` flags control which distribution characteristics can be
      used, but do not guarantee their use
    - When ``use_registry_characteristics`` is ``True``, characteristics may
      be retrieved from the distribution registry if not directly available
      in the Distribution object
    """

    method: UnuranMethod = UnuranMethod.AUTO
    method_params: dict[str, Any] | None = None
    use_ppf: bool = False
    use_pdf: bool = True
    use_cdf: bool = False
    use_registry_characteristics: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.method_params is None:
            object.__setattr__(self, "method_params", {})


class UnuranSampler(Protocol):
    """
    Protocol for UNU.RAN sampler instances.

    A sampler is created for a specific distribution and can generate
    random variates efficiently. The sampler encapsulates the UNU.RAN
    generator object and provides a Python interface for sampling.

    Notes
    -----
    - Samplers are stateful and maintain internal random number generator state
    - Multiple calls to ``sample()`` produce independent variates
    - The sampler is bound to the distribution characteristics used at creation
    """

    def __init__(
        self,
        distr: Distribution,
        config: UnuranMethodConfig | None = None,
        **override_options: Any,
    ) -> None:
        """Initialize the sampler for ``distr`` with optional configuration overrides."""
        ...

    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """
        Generate ``n`` random variates from the distribution.

        Parameters
        ----------
        n : int
            Number of variates to generate.

        Returns
        -------
        numpy.ndarray
            1D array of shape ``(n,)`` containing the random variates.

        Raises
        ------
        RuntimeError
            If sampling fails (e.g., method configuration is invalid,
            distribution characteristics are insufficient).
        """
        ...

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the sampler's random number generator.

        Parameters
        ----------
        seed : int | None, optional
            New random seed. If ``None``, uses system entropy.
        """
        ...

    @property
    def method(self) -> UnuranMethod:
        """The sampling method used by this sampler."""
        ...

    @property
    def is_initialized(self) -> bool:
        """Whether the sampler has been successfully initialized."""
        ...


class UnuranSamplingStrategy(Protocol):
    """
    Protocol for UNU.RAN-based sampling strategies.

    This protocol extends the standard :class:`SamplingStrategy` protocol
    with UNU.RAN-specific functionality. It integrates with the distribution
    system by implementing the standard ``sample()`` method.

    Notes
    -----
    - The strategy may create and cache samplers per distribution
    - The strategy handles the conversion from UNU.RAN output to
      :class:`~pysatl_core.distributions.sampling.Sample` format
    """

    def sample(self, n: int, distr: Distribution, **options: Any) -> Sample:
        """
        Generate a sample from the distribution using UNU.RAN.

        Parameters
        ----------
        n : int
            Number of observations to draw.
        distr : Distribution
            The distribution to sample from.
        **options : Any
            Additional options that may override the default configuration:
            - ``method``: override the sampling method
            - ``seed``: override the random seed
            - Other method-specific parameters

        Returns
        -------
        Sample
            A 2D sample of shape ``(n, 1)`` for univariate distributions.

        Raises
        ------
        RuntimeError
            If the distribution type is not supported, or if UNU.RAN
            cannot create a sampler with the available characteristics.
        ValueError
            If the configuration is invalid.
        """
        ...

    @property
    def default_config(self) -> UnuranMethodConfig:
        """Default method configuration."""
        ...
