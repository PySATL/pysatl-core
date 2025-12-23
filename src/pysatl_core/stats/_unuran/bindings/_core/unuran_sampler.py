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
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig
from pysatl_core.stats._unuran.bindings._core._unuran_sampler import (
    calculate_pmf_sum,
    cleanup_unuran_resources,
    create_and_init_generator,
    create_cdf_callback,
    create_dpdf_callback,
    create_parameter_object,
    create_pdf_callback,
    create_pmf_callback,
    create_ppf_callback,
    create_unuran_distribution,
    determine_domain_from_pmf,
    determine_domain_from_support,
    get_unuran_error_message,
    initialize_unuran_components,
    setup_continuous_callbacks,
    setup_dgt_method,
    setup_discrete_callbacks,
)
from pysatl_core.stats._unuran.bindings._core.helpers import (
    _get_available_characteristics,
    _select_best_method,
)
from pysatl_core.types import EuclideanDistributionType, GenericCharacteristicName, Kind

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


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
    is_initialized : bool
        Whether the sampler has been successfully initialized.

    Notes
    -----
    - Only univariate Euclidean distributions are supported
    - The sampler automatically selects the best method if ``method=AUTO``
    - Callbacks are kept alive to prevent garbage collection issues with CFFI
    - Resources are automatically cleaned up when the object is deleted
    """

    def __init__(
        self, distr: Distribution, config: UnuranMethodConfig | None = None, **override_options: Any
    ):
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
        **override_options : Any
            Additional options that override config values. Supported keys:
            - ``method``: Override the sampling method
            - ``seed``: Override the random seed (currently not used)

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
        self.distr = distr
        self.config = config or UnuranMethodConfig()
        self.override_options = override_options

        method_option = override_options.get("method", self.config.method)
        seed = override_options.get("seed", self.config.seed)

        from pysatl_core.stats._unuran.bindings import _unuran_cffi

        if _unuran_cffi is None:
            raise RuntimeError(
                "UNURAN CFFI bindings are not available. "
                "Please build them via `python "
                "src/pysatl_core/stats/_unuran/bindings/_cffi_build.py` "
                "or install pysatl-core with the compiled extension."
            )

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
        self._is_continuous: bool = self._kind == Kind.CONTINUOUS

        if isinstance(method_option, UnuranMethod):
            method = method_option
        else:
            try:
                method = UnuranMethod(str(method_option))
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported UNU.RAN method: {method_option}") from exc

        if self.config.use_registry_characteristics:
            available_chars: set[GenericCharacteristicName] = _get_available_characteristics(distr)
        else:
            available_chars = {str(name) for name in distr.analytical_computations}

        if method == UnuranMethod.AUTO:
            method = self._select_method(available_chars)

        self._method: UnuranMethod = method

        self._unuran_distr: Any | None = None
        self._unuran_par: Any | None = None
        self._unuran_gen: Any | None = None
        self._callbacks: list[Any] = []  # Keep callbacks alive
        self._cleaned_up: bool = False  # Flag to prevent double cleanup

        self._initialize_unuran(seed)

    def _select_method(self, available_chars: set[GenericCharacteristicName]) -> UnuranMethod:
        """
        Select the best UNU.RAN method based on available characteristics.

        Parameters
        ----------
        available_chars : set[GenericCharacteristicName]
            Set of available characteristic names (e.g., PDF, CDF, PPF, PMF).

        Returns
        -------
        UnuranMethod
            The selected UNU.RAN sampling method.

        Notes
        -----
        This method delegates to ``_select_best_method`` helper function which
        implements heuristics for method selection based on available
        characteristics and distribution kind.
        """
        return _select_best_method(available_chars, self._kind, self.config)

    def _create_pdf_callback(self) -> Any:
        return create_pdf_callback(self)

    def _create_pmf_callback(self) -> Any:
        return create_pmf_callback(self)

    def _create_cdf_callback(self) -> Any:
        return create_cdf_callback(self)

    def _create_ppf_callback(self) -> Any:
        return create_ppf_callback(self)

    def _create_dpdf_callback(self) -> Any:
        return create_dpdf_callback()

    def _get_unuran_error_message(self, base_msg: str) -> str:
        return get_unuran_error_message(self._lib, self._ffi, base_msg)

    def _create_unuran_distribution(self) -> None:
        create_unuran_distribution(self)

    def _setup_continuous_callbacks(self) -> None:
        setup_continuous_callbacks(self)

    def _setup_discrete_callbacks(self) -> None:
        setup_discrete_callbacks(self)

    def _determine_domain_from_support(self) -> tuple[int, int | None] | None:
        return determine_domain_from_support(self)

    def _determine_domain_from_pmf(self, domain_left: int | None = None) -> tuple[int, int]:
        return determine_domain_from_pmf(self, domain_left)

    def _calculate_pmf_sum(self, domain_left: int, domain_right: int) -> float:
        return calculate_pmf_sum(self, domain_left, domain_right)

    def _setup_dgt_method(self) -> None:
        setup_dgt_method(self)

    def _create_and_init_generator(self) -> None:
        create_and_init_generator(self)

    def _initialize_unuran(self, seed: int | None) -> None:
        initialize_unuran_components(self, seed)

    def _create_parameter_object(self) -> Any:
        return create_parameter_object(self)

    def _cleanup(self) -> None:
        cleanup_unuran_resources(self)

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
        if not self.is_initialized:
            raise RuntimeError("Sampler is not initialized")

        if n < 0:
            raise ValueError(f"Number of samples must be non-negative, got {n}")

        samples = np.empty(n, dtype=np.float64)

        if self._is_continuous:
            for i in range(n):
                samples[i] = self._lib.unur_sample_cont(self._unuran_gen)
        else:
            for i in range(n):
                samples[i] = float(self._lib.unur_sample_discr(self._unuran_gen))

        return samples

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the sampler's random number generator.

        Parameters
        ----------
        seed : int or None, optional
            Random seed for the generator. Currently not used as UNURAN's basic
            API doesn't provide direct seed setting functionality.

        Notes
        -----
        Currently, this method only reinitializes the sampler if it's not
        initialized. Full seed support would require access to UNURAN's underlying
        RNG object, which is not exposed in the basic API.

        This is a placeholder for future implementation when seed setting
        functionality becomes available.
        """
        # UNURAN doesn't expose RNG seed setting in the basic API
        # This would require access to the underlying RNG object
        # For now, we just reinitialize if needed
        if not self.is_initialized:
            self._initialize_unuran(seed)

    @property
    def method(self) -> UnuranMethod:
        """The sampling method used by this sampler."""
        return self._method

    @property
    def is_initialized(self) -> bool:
        """
        Check whether the sampler has been successfully initialized.

        Returns
        -------
        bool
            True if both the generator and distribution objects are valid
            (non-NULL), False otherwise.

        Notes
        -----
        This property checks that:
        - ``_unuran_gen`` is not None and not NULL
        - ``_unuran_distr`` is not None and not NULL

        The sampler must be initialized before calling ``sample()``.
        """
        return (
            self._unuran_gen is not None
            and self._unuran_gen != self._ffi.NULL
            and self._unuran_distr is not None
            and self._unuran_distr != self._ffi.NULL
        )
