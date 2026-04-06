"""
UNU.RAN Default Sampler
=======================

This module provides the default UNU.RAN sampler implementation that uses
the UNU.RAN library for efficient random variate generation from probability
distributions. The sampler automatically selects appropriate sampling methods
based on available distribution characteristics.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import contextlib
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from pysatl_core.distributions.support import ExplicitTableDiscreteSupport
from pysatl_core.sampling.default import DefaultSamplingUnivariateStrategy
from pysatl_core.sampling.unuran.core._unuran_sampler import (
    UnuranSamplerInitializer,
    ensure_default_urng,
)
from pysatl_core.sampling.unuran.core.method_requirements import METHOD_CHARACTERISTIC_REQUIREMENTS
from pysatl_core.sampling.unuran.method_config import UnuranMethod, UnuranMethodConfig
from pysatl_core.types import (
    CharacteristicName,
    EuclideanDistributionType,
    Kind,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.types import NumericArray


class DefaultUnuranSampler:
    """
    Default UNU.RAN sampler implementation.

    This sampler provides a default implementation for generating random variates
    from probability distributions using the UNU.RAN library. It automatically
    selects appropriate sampling methods based on available distribution
    characteristics (PDF, CDF, PPF, PMF).

    The sampler supports both continuous and discrete univariate distributions
    and uses various UNU.RAN methods (PINV, HINV, NINV, DGT, etc.) depending on
    the available characteristics and configuration.

    Attributes
    ----------
    distr : Distribution
        The probability distribution to sample from.
    config : UnuranMethodConfig
        Configuration for method selection and parameters.
    method : UnuranMethod
        The sampling method currently used by this sampler.

    Notes
    -----
    - Only univariate Euclidean distributions are supported
    - The sampler automatically selects the best method if ``method=AUTO``
    - Callbacks are kept alive to prevent garbage collection issues with CFFI
    - Resources are automatically cleaned up when the object is deleted
    """

    def __init__(self, distr: Distribution, config: UnuranMethodConfig | None = None):
        """
        Initialize the UNU.RAN sampler.

        Parameters
        ----------
        distr : Distribution
            The probability distribution to sample from. Must be a univariate
            Euclidean distribution (dimension=1).
        config : UnuranMethodConfig, optional
            Configuration for method selection and parameters. If None, uses
            default configuration.

        Raises
        ------
        RuntimeError
            If the distribution type is not supported (not Euclidean or
            dimension != 1), or if initialization fails.

        Notes
        -----
        The initialization process:
        1. Validates distribution type and dimension
        2. Determines available characteristics
        3. Selects appropriate sampling method (if AUTO)
        4. Creates UNURAN distribution object
        5. Sets up callbacks for available characteristics
        6. Initializes the UNURAN generator
        """
        self._support = distr.support
        self._config = config or UnuranMethodConfig()
        method = self._config.method

        from pysatl_core.sampling.unuran.bindings import _unuran_cffi

        if _unuran_cffi is None:
            raise RuntimeError("UNURAN CFFI bindings are not available. ")

        ensure_default_urng()

        self._ffi: Any = _unuran_cffi.ffi
        self._lib: Any = _unuran_cffi.lib

        distr_type = distr.distribution_type
        if not isinstance(distr_type, EuclideanDistributionType):
            raise RuntimeError(
                f"Unsupported distribution type: {distr_type}. "
                "Only univariate Euclidean distributions are supported."
            )

        if distr_type.dimension != 1:
            raise RuntimeError(
                f"Unsupported distribution dimension: {distr_type.dimension}. "
                "Only univariate (dimension=1) distributions are supported."
            )

        self._kind = distr_type.kind

        self.available_chars: set[CharacteristicName] = self._resolve_available_chars(
            distr, self._config.use_registry_characteristics
        )

        if method == UnuranMethod.AUTO:
            method = self._select_best_method(self.available_chars, self._kind, self._config)

        self._method: UnuranMethod = method
        self._check_method_suitability()
        self._cleaned_up: bool = False
        self._fallback_sampling_method: Callable[[int], NumericArray] | None = None
        self._unuran_distr: Any | None = None
        self._unuran_par: Any | None = None
        self._unuran_gen: Any | None = None
        self._callbacks: list[Any] = []

        support = distr.support
        if isinstance(support, ExplicitTableDiscreteSupport) and support.points.size > 0:
            self._index_remap_points: np.ndarray | None = support.points
        else:
            self._index_remap_points = None

        self._initializer = UnuranSamplerInitializer(distr, method, self._lib, self._ffi)
        try:
            result = self._initializer.initialize_unuran_components(self.available_chars)
        except RuntimeError:
            if not self._config.use_fallback_sampler:
                raise
            sampler = DefaultSamplingUnivariateStrategy()
            self._fallback_sampling_method = partial(sampler.sample, distr=distr)
            return

        self._unuran_distr, self._unuran_par, self._unuran_gen, self._callbacks = result

    def _check_method_suitability(self) -> None:
        """Validate that the chosen UNURAN method is supported and has required inputs."""
        if self._method not in METHOD_CHARACTERISTIC_REQUIREMENTS:
            raise RuntimeError(f"Unsupported sampling method: {self._method}")

        requirements = METHOD_CHARACTERISTIC_REQUIREMENTS[self._method]

        if not self.available_chars.issuperset(requirements.required):
            raise RuntimeError(
                f"Method {self._method} requires the following characteristics: "
                f"{', '.join(sorted(requirements.required))}. "
                f"Available characteristics: {', '.join(sorted(self.available_chars))}"
            )

        # NOTE: Now support is always finite, in future it may be updated
        if requirements.requires_support and self._support is None:
            raise RuntimeError(
                "Method "
                f"{self._method} requires a finite support, but the distribution has an "
                "infinite support."
            )

    @staticmethod
    def _resolve_available_chars(
        distr: Distribution, use_registry_characteristics: bool
    ) -> set[CharacteristicName]:
        """Return the set of UNU.RAN-relevant characteristics available for the distribution.

        Parameters
        ----------
        distr : Distribution
            The distribution to probe.
        use_registry_characteristics : bool
            If ``True``, include characteristics derivable via the registry graph
            (delegates to ``query_method``). If ``False``, include only those
            present in ``analytical_computations``.

        Returns
        -------
        set[CharacteristicName]
            Subset of UNU.RAN-relevant characteristics that can be resolved.
        """
        _unuran_chars = [
            CharacteristicName.PDF,
            CharacteristicName.DPDF,
            CharacteristicName.CDF,
            CharacteristicName.PPF,
            CharacteristicName.PMF,
        ]
        chars: set[CharacteristicName] = set()
        if use_registry_characteristics:
            for char in _unuran_chars:
                try:
                    distr.query_method(char)
                    chars.add(char)
                except RuntimeError:
                    pass
        else:
            for char in _unuran_chars:
                if char in distr.analytical_computations:
                    chars.add(char)
        return chars

    def _cleanup(self) -> None:
        """Release UNURAN resources held by the initializer."""
        self._initializer.cleanup()
        self._cleaned_up = True

    def __del__(self) -> None:
        """
        Cleanup on object deletion.

        Automatically calls ``_cleanup()`` when the object is garbage collected.
        All exceptions during finalization are silently ignored.
        """
        with contextlib.suppress(Exception):
            self._cleanup()

    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """
        Generate random variates from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate. Must be non-negative.

        Returns
        -------
        npt.NDArray[np.float64]
            1D array of shape ``(n,)`` containing the generated samples.

        Raises
        ------
        RuntimeError
            If the sampler is not initialized.
        ValueError
            If ``n < 0``.

        Notes
        -----
        Uses UNURAN's sampling functions:
        - ``unur_sample_cont()`` for continuous distributions
        - ``unur_sample_discr()`` for discrete distributions

        Samples are generated sequentially in a loop. For large ``n``, this
        may be slower than vectorized operations, but it's necessary due to
        UNURAN's C API design.
        """
        if n < 0:
            raise ValueError(f"Number of samples must be non-negative, got {n}")

        if self._fallback_sampling_method is not None:
            return cast(npt.NDArray[np.float64], self._fallback_sampling_method(n))

        if self._unuran_gen is None or self._lib is None:
            raise RuntimeError("UNURAN sampler is not initialized")

        if n == 0:
            return np.empty(0, dtype=np.float64)

        match self._kind:
            case Kind.CONTINUOUS:
                sample_func_name = "unur_sample_cont"
            case Kind.DISCRETE:
                sample_func_name = "unur_sample_discr"
            case _:
                raise RuntimeError(f"Unsupported distribution kind: {self._kind}")

        sample_func = getattr(self._lib, sample_func_name, None)
        if sample_func is None:
            raise RuntimeError(
                f"UNURAN sampler is not initialized (missing {sample_func_name} binding)"
            )

        samples = np.empty(n, dtype=np.float64)

        match self._kind:
            case Kind.CONTINUOUS:
                samples = np.fromiter(
                    (sample_func(self._unuran_gen) for _ in range(n)),
                    dtype=np.float64,
                    count=n,
                )
            case Kind.DISCRETE:
                samples = np.fromiter(
                    (float(sample_func(self._unuran_gen)) for _ in range(n)),
                    dtype=np.float64,
                    count=n,
                )
                if self._index_remap_points is not None:
                    samples = self._index_remap_points[samples.astype(np.intp)].astype(np.float64)
            case _:
                raise RuntimeError(f"Unsupported distribution kind: {self._kind}")

        return samples

    @staticmethod
    def _select_best_method(
        available_chars: set[CharacteristicName],
        kind: Kind,
        config: UnuranMethodConfig,
    ) -> UnuranMethod:
        """
        Select the best UNU.RAN method based on available characteristics.

        This function implements heuristics for method selection when
        ``method=AUTO``.

        Parameters
        ----------
        available_chars : set[CharacteristicName]
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
        match kind:
            case Kind.CONTINUOUS:
                if CharacteristicName.PPF in available_chars and config.use_ppf:
                    return UnuranMethod.PINV
                elif CharacteristicName.PDF in available_chars and config.use_pdf:
                    # PINV works with PDF (and optionally CDF/mode)
                    # AROU and TDR require dPDF which we may not have
                    return UnuranMethod.PINV
                elif CharacteristicName.CDF in available_chars and config.use_cdf:
                    return UnuranMethod.NINV
                elif CharacteristicName.PDF in available_chars:
                    return UnuranMethod.PINV  # PINV works with PDF
                else:
                    raise RuntimeError(
                        "No suitable method found. Need at least PDF, CDF, or PPF "
                        "for continuous distributions."
                    )
            case Kind.DISCRETE:
                if CharacteristicName.PMF in available_chars:
                    return UnuranMethod.DGT
                else:
                    raise RuntimeError(
                        "No suitable method found. Need at least PMF for discrete distributions."
                    )
            case _:
                raise RuntimeError(f"Unsupported distribution kind: {kind}")

    @property
    def method(self) -> UnuranMethod:
        """The sampling method used by this sampler."""
        return self._method
