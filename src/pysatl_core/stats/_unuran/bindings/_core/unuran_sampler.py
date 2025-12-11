
from __future__ import annotations

from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig
from pysatl_core.stats._unuran.bindings._core.helpers import _get_available_characteristics, _select_best_method

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pysatl_core.types import EuclideanDistributionType, Kind

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


class DefaultUnuranSampler:
    """
    Default UNU.RAN sampler implementation.

    This sampler uses the default UNU.RAN method selection and parameters.
    """

    def __init__(self, distr: Distribution, config: UnuranMethodConfig | None = None, **override_options: Any):
        self.distr = distr
        self.config = config or UnuranMethodConfig()
        self.override_options = override_options
        
        method = override_options.get("method", self.config.method)
        seed = override_options.get("seed", self.config.seed)
        
        try:
            from pysatl_core.stats._unuran.bindings import _unuran_cffi
        except (ImportError, AttributeError):
            import _unuran_cffi
        
        self._ffi = _unuran_cffi.ffi
        self._lib = _unuran_cffi.lib
    
        distr_type = distr.distribution_type
        if not isinstance(distr_type, EuclideanDistributionType):
            raise RuntimeError(
                f"Unsupported distribution type: {distr_type}. "
                "Only univariate Euclidean distributions are supported."
            )
        
        self._kind = distr_type.kind
        self._is_continuous = self._kind == Kind.CONTINUOUS
        
        available_chars = _get_available_characteristics(distr)
        
        if method == UnuranMethod.AUTO:
            method = self._select_method(available_chars)
        
        self._method = method
        
        self._unuran_distr = None
        self._unuran_par = None
        self._unuran_gen = None
        self._callbacks = []  # Keep callbacks alive
        self._cleaned_up = False  # Flag to prevent double cleanup
        
        self._initialize_unuran(available_chars, seed)
    
    def _select_method(self, available_chars: set[str]) -> UnuranMethod:
        """Select the best method based on available characteristics."""
        return _select_best_method(available_chars, self._kind, self.config)
    
    def _create_pdf_callback(self):
        """Create PDF callback from distribution (for continuous distributions)."""
        if not self._is_continuous:
            return None
        
        analytical_comps = self.distr.analytical_computations
        
        if "pdf" in analytical_comps:                                                                                                                                                                                                           
            pdf_func = analytical_comps["pdf"]
            
            def pdf_callback(x: float, distr_ptr) -> float:
                return float(pdf_func(x))
            
            return self._ffi.callback("double(double, const struct unur_distr*)", pdf_callback)
        
        return None
    
    def _create_pmf_callback(self):
        """Create PMF callback from distribution (for discrete distributions)."""
        if self._is_continuous:
            return None
        
        analytical_comps = self.distr.analytical_computations
        
        # Try PMF first
        if "pmf" in analytical_comps:
            pmf_func = analytical_comps["pmf"]
            
            def pmf_callback(k: int, distr_ptr) -> float:
                return float(pmf_func(float(k)))
            
            return self._ffi.callback("double(int, const struct unur_distr*)", pmf_callback)
        
        # Fallback to PDF if available (some systems use "pdf" for discrete too)
        if "pdf" in analytical_comps:
            pdf_func = analytical_comps["pdf"]
            
            def pmf_callback(k: int, distr_ptr) -> float:
                return float(pdf_func(float(k)))
            
            return self._ffi.callback("double(int, const struct unur_distr*)", pmf_callback)
        
        return None
    
    def _create_cdf_callback(self):
        """Create CDF callback from distribution."""
        analytical_comps = self.distr.analytical_computations
        
        if "cdf" in analytical_comps:
            cdf_func = analytical_comps["cdf"]
            
            if self._is_continuous:
                def cdf_callback(x: float, distr_ptr) -> float:
                    return float(cdf_func(x))
                return self._ffi.callback("double(double, const struct unur_distr*)", cdf_callback)
            else:
                def cdf_callback(k: int, distr_ptr) -> float:
                    return float(cdf_func(float(k)))
                return self._ffi.callback("double(int, const struct unur_distr*)", cdf_callback)
        
        return None
    
    def _create_dpdf_callback(self):
        """Create derivative PDF callback if possible."""
        # For now, we don't compute derivatives automatically
        # This could be enhanced with automatic differentiation
        return None
    
    def _initialize_unuran(self, available_chars: set[str], seed: int | None) -> None:
        """Initialize UNURAN distribution, parameters, and generator."""
        # Create distribution object
        if self._is_continuous:
            self._unuran_distr = self._lib.unur_distr_cont_new()
        else:
            self._unuran_distr = self._lib.unur_distr_discr_new()
        
        if self._unuran_distr == self._ffi.NULL:
            raise RuntimeError("Failed to create UNURAN distribution object")
        
        try:
            # Set up callbacks based on available characteristics
            if self._is_continuous:
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
            else:
                # Discrete distribution
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
                
                # For DGT method, we need PV (probability vector)
                # Try to create PV from PMF if method is DGT
                if self._method == UnuranMethod.DGT:
                    # Set a reasonable domain for geometric distribution
                    # For geometric with p=0.3, most probability is in [1, 20]
                    # We'll set domain to [1, 100] to cover most of the distribution
                    domain_left = 1
                    domain_right = 100
                    
                    # Check if functions are available (CFFI module might need recompilation)
                    if not hasattr(self._lib, 'unur_distr_discr_set_domain'):
                        raise RuntimeError(
                            "unur_distr_discr_set_domain is not available. "
                            "Please recompile CFFI module: cd src/pysatl_core/stats/_unuran/bindings && python3 _cffi_build.py"
                        )
                    
                    if not hasattr(self._lib, 'unur_distr_discr_make_pv'):
                        raise RuntimeError(
                            "unur_distr_discr_make_pv is not available. "
                            "Please recompile CFFI module: cd src/pysatl_core/stats/_unuran/bindings && python3 _cffi_build.py"
                        )
                    
                    # Set domain first
                    result = self._lib.unur_distr_discr_set_domain(
                        self._unuran_distr, domain_left, domain_right
                    )
                    if result != 0:
                        # If domain setting fails, try with larger domain
                        domain_right = 1000
                        result = self._lib.unur_distr_discr_set_domain(
                            self._unuran_distr, domain_left, domain_right
                        )
                        if result != 0:
                            raise RuntimeError(
                                f"Failed to set domain for discrete distribution (error code: {result}). "
                                f"Tried domain [{domain_left}, {domain_right}]"
                            )
                    
                    # Set PMF sum to help make_pv work better
                    # For normalized PMF, sum should be 1.0
                    if hasattr(self._lib, 'unur_distr_discr_set_pmfsum'):
                        # Try to set sum to 1.0 (assuming normalized PMF)
                        # If PMF is not normalized, this might need adjustment
                        self._lib.unur_distr_discr_set_pmfsum(self._unuran_distr, 1.0)
                    
                    # Create PV from PMF
                    # make_pv returns length of PV on success, negative value on failure
                    pv_length = self._lib.unur_distr_discr_make_pv(self._unuran_distr)
                    if pv_length <= 0:
                        errno = self._lib.unur_get_errno()
                        error_str = self._lib.unur_get_strerror(errno) if errno != 0 else None
                        error_msg = (
                            f"Failed to create PV from PMF (returned length: {pv_length}, errno: {errno}). "
                            "DGT method requires PV. "
                            "The PMF might not be normalized or domain is too large. "
                            "Try setting a smaller domain or providing PV directly."
                        )
                        if error_str:
                            error_msg += f" Error: {self._ffi.string(error_str).decode('utf-8')}"
                        raise RuntimeError(error_msg)
            
            # Create parameter object based on method
            self._unuran_par = self._create_parameter_object()
            if self._unuran_par == self._ffi.NULL:
                errno = self._lib.unur_get_errno()
                error_str = self._lib.unur_get_strerror(errno) if errno != 0 else None
                error_msg = f"Failed to create UNURAN parameter object (errno: {errno})"
                if error_str:
                    error_msg += f": {self._ffi.string(error_str).decode('utf-8')}"
                raise RuntimeError(error_msg)
            
            # Initialize generator
            self._unuran_gen = self._lib.unur_init(self._unuran_par)
            if self._unuran_gen == self._ffi.NULL:
                errno = self._lib.unur_get_errno()
                error_str = self._lib.unur_get_strerror(errno) if errno != 0 else None
                error_msg = f"Failed to initialize UNURAN generator (errno: {errno})"
                if error_str:
                    error_msg += f": {self._ffi.string(error_str).decode('utf-8')}"
                raise RuntimeError(error_msg)
            
            # Note: UNURAN doesn't have a direct seed setting function in the basic API
            # Seed would need to be set through the underlying RNG if available
            # For now, we skip seed setting
            
        except Exception:
            # Clean up on error
            self._cleanup()
            raise
    
    def _create_parameter_object(self):
        """Create UNURAN parameter object based on selected method."""
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
            raise ValueError(f"Method SROU is not available in CFFI bindings. Use PINV instead.")
        else:
            raise ValueError(f"Unsupported UNURAN method: {method}")
    
    def _cleanup(self) -> None:
        """Clean up UNURAN resources."""
        # Prevent double cleanup
        if getattr(self, '_cleaned_up', False):
            return
        
        self._cleaned_up = True
        
        # Important: After unur_init(par), the par object is automatically destroyed
        # So we should NOT call unur_par_free(par) after initialization
        # Also, UNURAN creates a private copy of distr inside gen, so we can
        # safely free the original distr after freeing gen
        
        # Order: free generator first (this also frees the private copy of distr),
        # then free the original distribution
        gen_freed = False
        if hasattr(self, '_unuran_gen') and self._unuran_gen is not None:
            # Check if it's a valid pointer (not NULL)
            try:
                # Only free if it's not NULL
                if self._unuran_gen != self._ffi.NULL:
                    self._lib.unur_free(self._unuran_gen)
                    gen_freed = True
            except Exception:
                # Ignore errors during cleanup - might already be freed
                pass
            finally:
                self._unuran_gen = None
        
        # Only free par if it exists and gen was not initialized
        # (i.e., if initialization failed before unur_init was called)
        if hasattr(self, '_unuran_par') and self._unuran_par is not None:
            # Check if gen was initialized - if so, par was already destroyed
            if not gen_freed and self._unuran_par != self._ffi.NULL:
                try:
                    self._lib.unur_par_free(self._unuran_par)
                except Exception:
                    pass  # Ignore errors during cleanup
            self._unuran_par = None
        
        # Free the original distribution object
        # This is safe because gen (if it was created) has its own private copy
        if hasattr(self, '_unuran_distr') and self._unuran_distr is not None:
            if self._unuran_distr != self._ffi.NULL:
                try:
                    self._lib.unur_distr_free(self._unuran_distr)
                except Exception:
                    pass  # Ignore errors during cleanup
            self._unuran_distr = None
        
        # IMPORTANT: Keep callbacks in memory!
        # Don't clear _callbacks list - it keeps callback functions alive
        # Callbacks must remain valid as long as the generator might use them
        # They will be garbage collected when the Python object is deleted
        # Only clear if we're sure everything is freed
        if hasattr(self, '_callbacks'):
            # Keep callbacks alive - don't clear the list
            # The list will be garbage collected with the object
            pass
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self._cleanup()
        except Exception:
            pass  # Ignore all errors during finalization
    
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """Generate n random variates from the distribution."""
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
        
        Note: UNURAN's basic API doesn't provide direct seed setting.
        This method is a placeholder for future implementation.
        """
        # UNURAN doesn't expose RNG seed setting in the basic API
        # This would require access to the underlying RNG object
        # For now, we just reinitialize if needed
        if not self.is_initialized:
            available_chars = _get_available_characteristics(self.distr)
            self._initialize_unuran(available_chars, seed)
    
    @property
    def method(self) -> UnuranMethod:
        """The sampling method used by this sampler."""
        return self._method
    
    @property
    def is_initialized(self) -> bool:
        """Whether the sampler has been successfully initialized."""
        return (
            self._unuran_gen is not None
            and self._unuran_gen != self._ffi.NULL
            and self._unuran_distr is not None
            and self._unuran_distr != self._ffi.NULL
        )
