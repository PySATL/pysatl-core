"""
UNU.RAN Helper Functions
========================

This module provides helper functions for UNU.RAN integration, including
characteristic detection and automatic method selection for sampling.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING

from pysatl_core.stats._unuran.api import UnuranMethod, UnuranMethodConfig
from pysatl_core.types import CharacteristicName, GenericCharacteristicName, Kind

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution


# TODO: move to registry module
def _get_available_characteristics(distr: Distribution) -> set[GenericCharacteristicName]:
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
    set[GenericCharacteristicName]
        Set of available characteristic names (e.g., {CharacteristicName.PDF,
        CharacteristicName.CDF, CharacteristicName.PPF}).
        Includes both analytical characteristics and those reachable through
        the characteristic graph.

    Notes
    -----
    - If the distribution has no analytical computations, returns an empty set
    - The graph is obtained from the global characteristic registry
    - Characteristics are considered "available" if they can be computed either
      analytically or through graph-based conversions
    """
    from pysatl_core.distributions.registry import characteristic_registry

    analytical_chars = set(distr.analytical_computations.keys())

    if not analytical_chars:
        return set()

    view = characteristic_registry().view(distr)
    available = set(analytical_chars)

    for src_char in analytical_chars:
        if src_char not in view.all_characteristics:
            continue

        # Simple BFS to find all reachable nodes in the view
        visited = {src_char}
        queue = [src_char]
        while queue:
            v = queue.pop(0)
            for w in view.successors_nodes(v):
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
        available.update(visited)

    return available


def _select_best_method(
    available_chars: set[GenericCharacteristicName],
    kind: Kind,
    config: UnuranMethodConfig,
) -> UnuranMethod:
    """
    Select the best UNU.RAN method based on available characteristics.

    This function implements heuristics for method selection when
    ``method=AUTO``.

    Parameters
    ----------
    available_chars : set[GenericCharacteristicName]
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
    else:
        # Discrete distributions
        if CharacteristicName.PMF in available_chars or CharacteristicName.PDF in available_chars:
            return UnuranMethod.DGT
        else:
            raise RuntimeError(
                "No suitable method found. Need at least PMF for discrete distributions."
            )
