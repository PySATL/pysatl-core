"""
DGT (Discrete Guide Table) setup utilities for UNU.RAN bindings.

Encapsulates domain inference, PMF normalization, and probability-vector
construction needed when configuring the UNU.RAN DGT method.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from pysatl_core.distributions.support import (
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.sampling.unuran.core._unuran_sampler.utils import get_unuran_error_message

if TYPE_CHECKING:
    from pysatl_core.distributions.support import Support
    from pysatl_core.sampling.unuran.core._unuran_sampler.domain import UnuranDomain
    from pysatl_core.types import Method


class DGTSetup:
    """
    Helper that orchestrates UNU.RAN DGT (Discrete Generation Table) preparation:
    stores CFFI handles, verifies required callbacks, and derives domain/probability
    vectors for discrete distributions before sampler construction.
    """

    def __init__(
        self,
        lib: Any,
        ffi: Any,
        domain: UnuranDomain,
        unuran_distr: Any,
        support: Support | None,
        pmf: Method[Any, Any] | None,
    ) -> None:
        self._lib = lib
        self._ffi = ffi
        self._domain = domain
        self._unuran_distr = unuran_distr
        self._support = support
        self._pmf = pmf

    def _require_attr(self, attr_name: str) -> None:
        """
        Assert that a required UNU.RAN CFFI function is present on the library handle.

        Parameters
        ----------
        attr_name : str
            Name of the required function on ``self._lib``.

        Raises
        ------
        RuntimeError
            If the attribute is absent, indicating the CFFI module needs recompilation.
        """
        if not hasattr(self._lib, attr_name):
            raise RuntimeError(
                f"{attr_name} is not available. Please recompile CFFI module: poetry build"
            )

    def _domain_points(self, domain_left: int, domain_right: int) -> Iterable[Any]:
        """
        Return an iterable of the support values to evaluate the PMF over.

        - ``ExplicitTableDiscreteSupport``: returns the full points array directly.
          The domain is ``[0, n-1]`` (index space), so ``domain_left``/``domain_right``
          are not meaningful as value bounds here.
        - ``IntegerLatticeDiscreteSupport`` with modulus > 1: align the start
          to the first lattice point ≥ domain_left and step by the modulus.
        - All other cases: dense integer range (step = 1).
        """
        support = self._support

        if isinstance(support, ExplicitTableDiscreteSupport):
            return support.points

        if isinstance(support, IntegerLatticeDiscreteSupport) and support.modulus > 1:
            step = support.modulus
            offset = (domain_left - support.residue) % step
            start = domain_left if offset == 0 else domain_left + (step - offset)
            return range(start, domain_right + 1, step)

        return range(domain_left, domain_right + 1)

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
        if self._pmf is None:
            raise RuntimeError("PMF is unavailable but _calculate_pmf_sum was called")
        pmf_func = self._pmf

        # TODO: use NumPy vectorised evaluation instead of scalar loop
        total = 0.0
        for k in self._domain_points(domain_left, domain_right):
            try:
                p = float(pmf_func(float(k)))
                if p >= 0 and not (np.isnan(p) or np.isinf(p)):
                    total += p
            except (ValueError, TypeError):
                continue

        return total

    def setup_dgt_method(self) -> None:
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
        self._require_attr("unur_distr_discr_set_domain")
        self._require_attr("unur_distr_discr_make_pv")

        domain_info = self._domain.determine_discrete_domain()

        if domain_info is None or domain_info[0] is None or domain_info[1] is None:
            raise RuntimeError(
                "Failed to determine domain for discrete distribution. "
                "Ensure the distribution has a bounded integer-valued support."
            )

        domain_left, domain_right = domain_info

        result = self._lib.unur_distr_discr_set_domain(
            self._unuran_distr, domain_left, domain_right
        )
        if result != 0:
            raise RuntimeError(
                f"Failed to set domain for discrete distribution (error code: {result}). "
                f"Tried domain [{domain_left}, {domain_right}]"
            )

        pmf_sum = self._calculate_pmf_sum(domain_left, domain_right)
        if hasattr(self._lib, "unur_distr_discr_set_pmfsum"):
            self._lib.unur_distr_discr_set_pmfsum(self._unuran_distr, pmf_sum)

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
            full_error_msg = get_unuran_error_message(self._lib, self._ffi, error_msg)
            raise RuntimeError(full_error_msg)
