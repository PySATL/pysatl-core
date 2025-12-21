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
from pysatl_core.stats._unuran.bindings._core.helpers import (
    _get_available_characteristics,
    _select_best_method,
)
from pysatl_core.types import (
    CharacteristicName,
    EuclideanDistributionType,
    GenericCharacteristicName,
    Kind,
)

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
        """
        Create PDF callback function for continuous distributions.

        Returns
        -------
        CFFI callback or None
            CFFI callback function that wraps the distribution's PDF computation,
            or None if the distribution is not continuous or PDF is not available.

        Notes
        -----
        The callback signature matches UNURAN's expected format:
        ``double(double, const struct unur_distr*)``

        The callback is stored in ``_callbacks`` list to prevent garbage collection.
        """
        if not self._is_continuous:
            return None

        analytical_comps = self.distr.analytical_computations

        if CharacteristicName.PDF in analytical_comps:
            pdf_func = analytical_comps[CharacteristicName.PDF]

            def pdf_callback(x: float, distr_ptr: Any) -> float:
                return float(pdf_func(x))

            return self._ffi.callback("double(double, const struct unur_distr*)", pdf_callback)

        return None

    def _create_pmf_callback(self) -> Any:
        """
        Create PMF callback function for discrete distributions.

        Returns
        -------
        CFFI callback or None
            CFFI callback function that wraps the distribution's PMF computation,
            or None if the distribution is not discrete or PMF/PDF is not available.

        Notes
        -----
        The callback signature matches UNURAN's expected format:
        ``double(int, const struct unur_distr*)``

        If PMF is not available, falls back to PDF (some systems use "pdf" for
        discrete distributions too). The callback is stored in ``_callbacks``
        list to prevent garbage collection.
        """
        if self._is_continuous:
            return None

        analytical_comps = self.distr.analytical_computations

        # Try PMF first
        if CharacteristicName.PMF in analytical_comps:
            pmf_func = analytical_comps[CharacteristicName.PMF]

            def pmf_callback(k: int, distr_ptr: Any) -> float:
                return float(pmf_func(float(k)))

            return self._ffi.callback("double(int, const struct unur_distr*)", pmf_callback)

        # Fallback to PDF if available (some systems use "pdf" for discrete too)
        if CharacteristicName.PDF in analytical_comps:
            pdf_func = analytical_comps[CharacteristicName.PDF]

            def pmf_callback(k: int, distr_ptr: Any) -> float:
                return float(pdf_func(float(k)))

            return self._ffi.callback("double(int, const struct unur_distr*)", pmf_callback)

        return None

    def _create_cdf_callback(self) -> Any:
        """
        Create CDF callback function for the distribution.

        Returns
        -------
        CFFI callback or None
            CFFI callback function that wraps the distribution's CDF computation,
            or None if CDF is not available.

        Notes
        -----
        The callback signature depends on distribution kind:
        - Continuous: ``double(double, const struct unur_distr*)``
        - Discrete: ``double(int, const struct unur_distr*)``

        The callback is stored in ``_callbacks`` list to prevent garbage collection.
        """
        analytical_comps = self.distr.analytical_computations

        if CharacteristicName.CDF in analytical_comps:
            cdf_func = analytical_comps[CharacteristicName.CDF]

            if self._is_continuous:

                def cdf_callback_cont(x: float, distr_ptr: Any) -> float:
                    return float(cdf_func(x))

                return self._ffi.callback(
                    "double(double, const struct unur_distr*)", cdf_callback_cont
                )
            else:

                def cdf_callback_discr(k: int, distr_ptr: Any) -> float:
                    return float(cdf_func(float(k)))

                return self._ffi.callback(
                    "double(int, const struct unur_distr*)", cdf_callback_discr
                )

        return None

    def _create_dpdf_callback(self) -> Any:
        """
        Create derivative PDF (dPDF) callback function.

        Returns
        -------
        None
            Currently always returns None as derivatives are not computed
            automatically.

        Notes
        -----
        This is a placeholder for future enhancement. Automatic differentiation
        could be used to compute derivatives of PDF functions, which would enable
        additional UNURAN methods like AROU and TDR that require dPDF.
        """
        # For now, we don't compute derivatives automatically
        # This could be enhanced with automatic differentiation
        return None

    def _get_unuran_error_message(self, base_msg: str) -> str:
        """
        Format UNURAN error message with errno and error string.

        Parameters
        ----------
        base_msg : str
            Base error message to format.

        Returns
        -------
        str
            Formatted error message including errno and UNURAN error string
            (if available).

        Notes
        -----
        This helper method retrieves the current UNURAN error state and formats
        a comprehensive error message for debugging purposes.
        """
        errno = self._lib.unur_get_errno()
        error_str = self._lib.unur_get_strerror(errno) if errno != 0 else None
        error_msg = f"{base_msg} (errno: {errno})"
        if error_str:
            error_msg += f": {self._ffi.string(error_str).decode('utf-8')}"
        return error_msg

    def _create_unuran_distribution(self) -> None:
        """
        Create UNURAN distribution object.

        Raises
        ------
        RuntimeError
            If the distribution object creation fails (returns NULL pointer).

        Notes
        -----
        Creates either a continuous or discrete UNURAN distribution object
        based on ``_is_continuous`` flag. The created object is stored in
        ``_unuran_distr`` attribute.
        """
        if self._is_continuous:
            self._unuran_distr = self._lib.unur_distr_cont_new()
        else:
            self._unuran_distr = self._lib.unur_distr_discr_new()

        if self._unuran_distr == self._ffi.NULL:
            raise RuntimeError("Failed to create UNURAN distribution object")

    def _setup_continuous_callbacks(self) -> None:
        """
        Set up callbacks for continuous distributions.

        Configures PDF, dPDF (if available), and CDF callbacks for the UNURAN
        continuous distribution object.

        Raises
        ------
        RuntimeError
            If setting the PDF callback fails (non-zero return code).

        Notes
        -----
        All created callbacks are appended to ``_callbacks`` list to prevent
        garbage collection. Only available callbacks are set (missing
        characteristics are skipped).
        """
        pdf_callback = self._create_pdf_callback()
        if pdf_callback:
            self._callbacks.append(pdf_callback)
            result = self._lib.unur_distr_cont_set_pdf(self._unuran_distr, pdf_callback)
            if result != 0:
                raise RuntimeError(f"Failed to set PDF callback (error code: {result})")

        dpdf_callback = self._create_dpdf_callback()
        if dpdf_callback:
            self._callbacks.append(dpdf_callback)
            self._lib.unur_distr_cont_set_dpdf(self._unuran_distr, dpdf_callback)

        cdf_callback = self._create_cdf_callback()
        if cdf_callback:
            self._callbacks.append(cdf_callback)
            self._lib.unur_distr_cont_set_cdf(self._unuran_distr, cdf_callback)

    def _setup_discrete_callbacks(self) -> None:
        """
        Set up callbacks for discrete distributions.

        Configures PMF and CDF callbacks for the UNURAN discrete distribution
        object.

        Raises
        ------
        RuntimeError
            If setting the PMF callback fails (non-zero return code).

        Notes
        -----
        All created callbacks are appended to ``_callbacks`` list to prevent
        garbage collection. Only available callbacks are set (missing
        characteristics are skipped).
        """
        pmf_callback = self._create_pmf_callback()
        if pmf_callback:
            self._callbacks.append(pmf_callback)
            result = self._lib.unur_distr_discr_set_pmf(self._unuran_distr, pmf_callback)
            if result != 0:
                raise RuntimeError(f"Failed to set PMF callback (error code: {result})")

        cdf_callback = self._create_cdf_callback()
        if cdf_callback:
            self._callbacks.append(cdf_callback)
            self._lib.unur_distr_discr_set_cdf(self._unuran_distr, cdf_callback)

    def _determine_domain_from_support(self) -> tuple[int, int | None] | None:
        """
        Determine domain boundaries from distribution support if available.

        Returns
        -------
        tuple[int, int | None] or None
            Domain as (left, right) if support is bounded, or (left, None) if only
            left boundary is known, or None if support is unavailable/unbounded.
        """
        support = self.distr.support
        if support is None:
            return None

        from pysatl_core.distributions.support import (
            ExplicitTableDiscreteSupport,
            IntegerLatticeDiscreteSupport,
        )

        # Handle ExplicitTableDiscreteSupport
        if isinstance(support, ExplicitTableDiscreteSupport):
            points = support.points
            if points.size == 0:
                return None
            left = int(np.floor(points[0]))
            right = int(np.ceil(points[-1]))
            return (left, right)

        # Handle IntegerLatticeDiscreteSupport
        if isinstance(support, IntegerLatticeDiscreteSupport):
            first = support.first()
            last = support.last()

            if first is not None and last is not None:
                return (first, last)

            if first is not None:
                return (first, None)

            return None

        # For other support types, try to get first and last points
        try:
            first = support.first()  # type: ignore[attr-defined]
            last = support.last()  # type: ignore[attr-defined]
            if first is not None and last is not None:
                return (int(first), int(last))
        except (AttributeError, TypeError):
            pass

        return None

    def _determine_domain_from_pmf(self, domain_left: int | None = None) -> tuple[int, int]:
        """
        Determine domain boundaries by evaluating PMF until probability becomes negligible.

        This heuristic evaluates PMF starting from domain_left (or 0 if None)
        and finds the right boundary where cumulative probability exceeds threshold.

        Parameters
        ----------
        domain_left : int, optional
            Left boundary of domain. If None, starts from 0.

        Returns
        -------
        tuple[int, int]
            Domain as (left, right).
        """
        analytical_comps = self.distr.analytical_computations

        # Get PMF function
        if CharacteristicName.PMF in analytical_comps:
            pmf_func = analytical_comps[CharacteristicName.PMF]
        elif CharacteristicName.PDF in analytical_comps:
            pmf_func = analytical_comps[CharacteristicName.PDF]
        else:
            raise RuntimeError("PMF or PDF is required for domain determination")

        # Determine starting point
        if domain_left is None:
            start_k = 0
            for k in range(-10, 0):
                try:
                    p = float(pmf_func(float(k)))
                    if p > 1e-10:  # Non-negligible probability
                        start_k = k
                        break
                except (ValueError, TypeError):
                    break
            domain_left = start_k
        else:
            domain_left = int(domain_left)

        # Accumulate probability until we reach threshold
        cumulative_prob = 0.0
        threshold = 0.9999  # Cover 99.99% of distribution
        max_iterations = 10000  # Safety limit
        domain_right = domain_left

        for k in range(domain_left, domain_left + max_iterations):
            try:
                p = float(pmf_func(float(k)))
                if p < 0 or np.isnan(p) or np.isinf(p):
                    break
                cumulative_prob += p
                domain_right = k

                if cumulative_prob >= threshold:
                    break
                if k > domain_left + 100 and p < 1e-10:
                    break
            except (ValueError, TypeError):
                break

        return (domain_left, domain_right)

    def _calculate_pmf_sum(self, domain_left: int, domain_right: int) -> float:
        """
        Calculate the sum of PMF over the specified domain.

        Parameters
        ----------
        domain_left : int
            Left boundary of domain.
        domain_right : int
            Right boundary of domain.

        Returns
        -------
        float
            Sum of PMF values over the domain.
        """
        analytical_comps = self.distr.analytical_computations

        # Get PMF function
        if CharacteristicName.PMF in analytical_comps:
            pmf_func = analytical_comps[CharacteristicName.PMF]
        elif CharacteristicName.PDF in analytical_comps:
            pmf_func = analytical_comps[CharacteristicName.PDF]
        else:
            return 1.0  # Fallback: assume normalized

        total = 0.0
        for k in range(domain_left, domain_right + 1):
            try:
                p = float(pmf_func(float(k)))
                if p >= 0 and not (np.isnan(p) or np.isinf(p)):
                    total += p
            except (ValueError, TypeError):
                continue

        return total

    def _setup_dgt_method(self) -> None:
        """
        Set up DGT method specific requirements (domain and probability vector).

        The DGT (Discrete Generation Table) method requires:
        1. A domain to be set for the discrete distribution
        2. A probability vector (PV) to be created from the PMF

        Raises
        ------
        RuntimeError
            If required CFFI functions are not available, if domain setting fails,
            or if PV creation fails.

        Notes
        -----
        - Domain is determined from distribution support if available
        - If support is unbounded or unavailable, domain is determined by
          evaluating PMF until cumulative probability exceeds threshold (0.9999)
        - PMF sum is calculated from the domain if possible, otherwise defaults to 1.0
        - This method should only be called for discrete distributions using
          the DGT method
        """
        if not hasattr(self._lib, "unur_distr_discr_set_domain"):
            raise RuntimeError(
                "unur_distr_discr_set_domain is not available. "
                "Please recompile CFFI module: cd src/pysatl_core/stats/_unuran/bindings "
                "&& python3 _cffi_build.py"
            )

        if not hasattr(self._lib, "unur_distr_discr_make_pv"):
            raise RuntimeError(
                "unur_distr_discr_make_pv is not available. "
                "Please recompile CFFI module: cd src/pysatl_core/stats/_unuran/bindings "
                "&& python3 _cffi_build.py"
            )

        # Determine domain from support or PMF
        domain_info = self._determine_domain_from_support()

        if domain_info is None:
            # Support is unavailable or unbounded, determine from PMF
            domain_left, domain_right = self._determine_domain_from_pmf()
        else:
            domain_left_candidate, domain_right_candidate = domain_info
            if domain_right_candidate is None:
                # Left boundary known, but right is unbounded
                domain_left, domain_right = self._determine_domain_from_pmf(domain_left_candidate)
            else:
                # Both boundaries known from support
                domain_left = domain_left_candidate
                domain_right = domain_right_candidate

        # Set domain in UNURAN
        result = self._lib.unur_distr_discr_set_domain(
            self._unuran_distr, domain_left, domain_right
        )
        if result != 0:
            raise RuntimeError(
                f"Failed to set domain for discrete distribution (error code: {result}). "
                f"Tried domain [{domain_left}, {domain_right}]"
            )

        # Calculate PMF sum to help make_pv work better
        pmf_sum = self._calculate_pmf_sum(domain_left, domain_right)
        if hasattr(self._lib, "unur_distr_discr_set_pmfsum"):
            self._lib.unur_distr_discr_set_pmfsum(self._unuran_distr, pmf_sum)

        # Create PV from PMF
        pv_length = self._lib.unur_distr_discr_make_pv(self._unuran_distr)
        if pv_length <= 0:
            error_msg = (
                f"Failed to create PV from PMF (returned length: {pv_length}). "
                "DGT method requires PV. "
                "The PMF might not be normalized or domain is too large. "
                f"Domain was set to [{domain_left}, {domain_right}], "
                f"PMF sum calculated as {pmf_sum:.6f}. "
                "Try setting a smaller domain or providing PV directly."
            )
            full_error_msg = self._get_unuran_error_message(error_msg)
            raise RuntimeError(full_error_msg)

    def _create_and_init_generator(self) -> None:
        """
        Create UNURAN parameter object and initialize the generator.

        Raises
        ------
        RuntimeError
            If parameter object creation fails or generator initialization fails.

        Notes
        -----
        The parameter object is created based on the selected method (PINV, HINV,
        DGT, etc.). After successful initialization, the parameter object is
        automatically destroyed by UNURAN, so it should not be freed manually.

        The initialized generator is stored in ``_unuran_gen`` attribute.
        """
        # Create parameter object based on method
        self._unuran_par = self._create_parameter_object()
        if self._unuran_par == self._ffi.NULL:
            error_msg = self._get_unuran_error_message("Failed to create UNURAN parameter object")
            raise RuntimeError(error_msg)

        # Initialize generator
        self._unuran_gen = self._lib.unur_init(self._unuran_par)
        if self._unuran_gen == self._ffi.NULL:
            error_msg = self._get_unuran_error_message("Failed to initialize UNURAN generator")
            raise RuntimeError(error_msg)

    def _initialize_unuran(self, seed: int | None) -> None:
        """
        Initialize UNURAN distribution, parameters, and generator.

        This is the main initialization method that orchestrates the setup of
        all UNURAN components.

        Parameters
        ----------
        seed : int or None
            Random seed for the generator. Currently not used as UNURAN's basic
            API doesn't provide direct seed setting functionality.

        Raises
        ------
        RuntimeError
            If any step of initialization fails. All resources are automatically
            cleaned up on error.

        Notes
        -----
        Initialization steps:
        1. Create UNURAN distribution object
        2. Set up callbacks based on distribution type (continuous/discrete)
        3. For discrete distributions with DGT method, set up domain and PV
        4. Create parameter object and initialize generator

        If any step fails, ``_cleanup()`` is called automatically to free
        allocated resources.
        """
        self._create_unuran_distribution()

        try:
            # Set up callbacks based on distribution type
            if self._is_continuous:
                self._setup_continuous_callbacks()
            else:
                self._setup_discrete_callbacks()

                # For DGT method, we need PV (probability vector)
                if self._method == UnuranMethod.DGT:
                    self._setup_dgt_method()

            self._create_and_init_generator()

            # Note: UNURAN doesn't have a direct seed setting function in the basic API
            # Seed would need to be set through the underlying RNG if available
            # For now, we skip seed setting

        except Exception:
            self._cleanup()
            raise

    def _create_parameter_object(self) -> Any:
        """
        Create UNURAN parameter object based on selected method.

        Returns
        -------
        CFFI pointer
            Pointer to the created UNURAN parameter object, or NULL pointer
            if creation fails.

        Raises
        ------
        ValueError
            If the method is SROU (not available in CFFI bindings) or if the
            method is unsupported.

        Notes
        -----
        Maps UNURAN methods to their corresponding CFFI creation functions:
        - AROU: ``unur_arou_new()``
        - TDR: ``unur_tdr_new()``
        - HINV: ``unur_hinv_new()``
        - PINV: ``unur_pinv_new()``
        - NINV: ``unur_ninv_new()``
        - DGT: ``unur_dgt_new()``

        Note that AROU and TDR require dPDF, which may not be available.
        """
        method = self._method

        # Map methods to CFFI functions
        # Note: AROU and TDR require dPDF, which may not be available
        if method == UnuranMethod.AROU:
            return self._lib.unur_arou_new(self._unuran_distr)
        elif method == UnuranMethod.TDR:
            return self._lib.unur_tdr_new(self._unuran_distr)
        elif method == UnuranMethod.HINV:
            return self._lib.unur_hinv_new(self._unuran_distr)
        elif method == UnuranMethod.PINV:
            return self._lib.unur_pinv_new(self._unuran_distr)
        elif method == UnuranMethod.NINV:
            return self._lib.unur_ninv_new(self._unuran_distr)
        elif method == UnuranMethod.DGT:
            return self._lib.unur_dgt_new(self._unuran_distr)
        elif method == UnuranMethod.SROU:
            # SROU is not in CFFI bindings, fallback to PINV
            # This should not happen if _select_method is correct
            raise ValueError("Method SROU is not available in CFFI bindings. Use PINV instead.")
        else:
            raise ValueError(f"Unsupported UNURAN method: {method}")

    def _cleanup(self) -> None:
        """
        Clean up UNURAN resources.

        Frees all allocated UNURAN objects (generator, parameter object,
        distribution object) in the correct order to avoid memory leaks.

        Notes
        -----
        Cleanup order:
        1. Free generator first (this also frees the private copy of distr)
        2. Free parameter object (only if generator was not initialized)
        3. Free the original distribution object

        Important notes:
        - After ``unur_init(par)``, the par object is automatically destroyed
          by UNURAN, so it should NOT be freed after successful initialization
        - UNURAN creates a private copy of distr inside gen, so the original
          distr can be safely freed after freeing gen
        - Callbacks are kept in ``_callbacks`` list to prevent garbage collection
          until the object is deleted
        - This method is idempotent (safe to call multiple times)
        """
        # Prevent double cleanup
        if getattr(self, "_cleaned_up", False):
            return

        self._cleaned_up = True

        # Important: After unur_init(par), the par object is automatically destroyed
        # So we should NOT call unur_par_free(par) after initialization
        # Also, UNURAN creates a private copy of distr inside gen, so we can
        # safely free the original distr after freeing gen

        # Order: free generator first (this also frees the private copy of distr),
        # then free the original distribution
        gen_freed = False
        if hasattr(self, "_unuran_gen") and self._unuran_gen is not None:
            # Check if it's a valid pointer (not NULL)
            # Only free if it's not NULL
            if self._unuran_gen != self._ffi.NULL:
                with contextlib.suppress(Exception):
                    self._lib.unur_free(self._unuran_gen)
                    gen_freed = True
            self._unuran_gen = None

        # Only free par if it exists and gen was not initialized
        # (i.e., if initialization failed before unur_init was called)
        if hasattr(self, "_unuran_par") and self._unuran_par is not None:
            # Check if gen was initialized - if so, par was already destroyed
            if not gen_freed and self._unuran_par != self._ffi.NULL:
                with contextlib.suppress(Exception):
                    self._lib.unur_par_free(self._unuran_par)
            self._unuran_par = None

        # Free the original distribution object
        # This is safe because gen (if it was created) has its own private copy
        if hasattr(self, "_unuran_distr") and self._unuran_distr is not None:
            if self._unuran_distr != self._ffi.NULL:
                with contextlib.suppress(Exception):
                    self._lib.unur_distr_free(self._unuran_distr)
            self._unuran_distr = None

        # IMPORTANT: Keep callbacks in memory!
        # Don't clear _callbacks list - it keeps callback functions alive
        # Callbacks must remain valid as long as the generator might use them
        # They will be garbage collected when the Python object is deleted
        # Only clear if we're sure everything is freed
        if hasattr(self, "_callbacks"):
            # Keep callbacks alive - don't clear the list
            # The list will be garbage collected with the object
            pass

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
