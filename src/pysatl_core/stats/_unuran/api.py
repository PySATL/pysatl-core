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

from pysatl_core.types import Kind


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
    seed : int | None, default None
        Random seed for the sampler. If ``None``, uses system entropy.

    Notes
    -----
    - Method-specific parameters are validated when the sampler is created
    - Some methods require specific characteristics (e.g., rejection methods
      typically need PDF)
    - The ``use_*`` flags control which distribution characteristics can be
      used, but do not guarantee their use
    """

    method: UnuranMethod = UnuranMethod.AUTO
    method_params: dict[str, Any] | None = None
    use_ppf: bool = False
    use_pdf: bool = True
    use_cdf: bool = False
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

    @classmethod
    def create(
        cls,
        distr: Distribution,
        config: UnuranMethodConfig | None = None,
        **override_options: Any,
    ) -> UnuranSampler:
        """
        Create a UNU.RAN sampler for the given distribution.

        This method delegates to C bindings to UNU.RAN. It analyzes the
        distribution's available characteristics and creates an appropriate sampler.

        Parameters
        ----------
        distr : Distribution
            The distribution to create a sampler for.
        config : UnuranMethodConfig | None, optional
            Method configuration. If ``None``, uses default configuration
            (auto method selection, PDF allowed, no seed).
        **override_options : Any
            Options that override the config:
            - ``method``: override the method
            - ``seed``: override the seed
            - Other method-specific parameters

        Returns
        -------
        UnuranSampler
            A configured sampler instance.

        Raises
        ------
        RuntimeError
            If the distribution type is not supported (currently only univariate
            continuous and discrete distributions are supported), or if UNU.RAN
            cannot create a sampler with the available characteristics.
        ValueError
            If the configuration or override options are invalid.

        Notes
        -----
        - The method queries the distribution's analytical computations and
          computation strategy to determine available characteristics (PDF, CDF, PPF)
        - For continuous distributions, PDF is typically required for rejection methods
        - For discrete distributions, PMF is required
        - If PPF is available and ``use_ppf=True``, inversion methods are preferred
        - The actual method selection depends on UNU.RAN's internal heuristics when
          ``method=AUTO``

        Examples
        --------
        Create a sampler with automatic method selection:

            >>> config = UnuranMethodConfig(method=UnuranMethod.AUTO)
            >>> sampler = UnuranSampler.create(distr, config)
            >>> variates = sampler.sample(1000)

        Create a sampler with a specific method:

            >>> config = UnuranMethodConfig(
            ...     method=UnuranMethod.TDR,
            ...     method_params={"accuracy": 1e-6}
            ... )
            >>> sampler = UnuranSampler.create(distr, config)
        """
        from pysatl_core.stats._unuran.bindings import create_sampler_impl

        return create_sampler_impl(distr, config, **override_options)


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


# Factory functions (to be implemented via bindings)


def create_unuran_strategy(
    config: UnuranMethodConfig | None = None,
) -> UnuranSamplingStrategy:
    """
    Create a UNU.RAN-based sampling strategy.

    This function creates a strategy instance that can be used as a
    distribution's sampling strategy. The strategy will create and manage
    UNU.RAN samplers as needed.

    Parameters
    ----------
    config : UnuranMethodConfig | None, optional
        Default method configuration for all samplers created by this strategy.
        If ``None``, uses default configuration.

    Returns
    -------
    UnuranSamplingStrategy
        A sampling strategy instance.

    Examples
    --------
    Create a strategy and use it with a distribution:

        >>> strategy = create_unuran_strategy(
        ...     UnuranMethodConfig(method=UnuranMethod.AUTO)
        ... )
        >>> distr.sampling_strategy = strategy
        >>> sample = distr.sample(1000)
    """
    from pysatl_core.stats._unuran.bindings import create_strategy_impl

    return create_strategy_impl(config)


# Helper functions for characteristic detection


def _get_available_characteristics(distr: Distribution) -> set[str]:
    """
    Get the set of available characteristic names for a distribution.

    This helper function queries the distribution's analytical computations
    and uses the characteristic graph to determine all reachable characteristics.
    
    The function works by:
    1. Getting all analytical (base) characteristics from the distribution
    2. For each analytical characteristic, finding all characteristics reachable
       through the graph using BFS
    3. Combining all reachable characteristics into a single set

    Parameters
    ----------
    distr : Distribution
        The distribution to query.

    Returns
    -------
    set[str]
        Set of available characteristic names (e.g., {"pdf", "cdf", "ppf"}).
        Includes both analytical characteristics and those reachable through
        the characteristic graph.

    Notes
    -----
    - If the distribution has no analytical computations, returns an empty set
    - The graph is obtained from the distribution type registry
    - Characteristics are considered "available" if they can be computed either
      analytically or through graph-based conversions
    """
    from pysatl_core.distributions.registry import distribution_type_register

    analytical_chars = set(distr.analytical_computations.keys())
    
    if not analytical_chars:
        return set()
    
    reg = distribution_type_register().get(distr.distribution_type)
    
    available = set(analytical_chars)
    
    for src_char in analytical_chars:
        reachable = reg.reachable_from(src_char, allowed=None)
        available.update(reachable)
    
    return available


def _select_best_method(
    available_chars: set[str],
    kind: Kind,
    config: UnuranMethodConfig,
) -> UnuranMethod:
    """
    Select the best UNU.RAN method based on available characteristics.

    This function implements heuristics for method selection when
    ``method=AUTO``.

    Parameters
    ----------
    available_chars : set[str]
        Set of available characteristic names.
    kind : Kind
        Distribution kind (continuous or discrete).
    config : UnuranMethodConfig
        Method configuration.

    Returns
    -------
    UnuranMethod
        The selected method.

    Notes
    -----
    - If PPF is available and ``use_ppf=True``, prefer PINV or HINV
    - If PDF is available, prefer rejection methods (TDR, ARS)
    - If CDF is available, prefer numerical inversion (NINV)
    - For discrete distributions, prefer discrete-specific methods
    """
    # This will be implemented with method selection heuristics
    raise NotImplementedError("Helper function for method selection")

