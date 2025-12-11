from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING

from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution

from pysatl_core.types import Kind


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
    # Note: Only methods available in CFFI bindings are used
    if kind == Kind.CONTINUOUS:
        if "ppf" in available_chars and config.use_ppf:
            return UnuranMethod.PINV
        elif "pdf" in available_chars and config.use_pdf:
            # PINV works with PDF (and optionally CDF/mode)
            # AROU and TDR require dPDF which we may not have
            return UnuranMethod.PINV
        elif "cdf" in available_chars and config.use_cdf:
            return UnuranMethod.NINV
        elif "pdf" in available_chars:
            return UnuranMethod.PINV  # PINV works with PDF
        else:
            raise RuntimeError(
                "No suitable method found. Need at least PDF, CDF, or PPF for continuous distributions."
            )
    else:                                                                                                                                                                                                       
        # Discrete distributions
        if "pmf" in available_chars or "pdf" in available_chars:
            return UnuranMethod.DGT
        else:
            raise RuntimeError(
                "No suitable method found. Need at least PMF for discrete distributions."
            )