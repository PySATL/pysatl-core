"""
Default configuration and cached accessor for the global characteristic registry.

- No auto-configuration in constructor.
- Provide `characteristic_registry()` with @lru_cache that builds the singleton
  instance and seeds it with a minimal set of edges (PDF,
  CDF, PPF, PMF).
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail, Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from functools import lru_cache

from pysatl_core.distributions import ComputationMethod
from pysatl_core.distributions.fitters import (
    fit_cdf_to_pdf_1C,
    fit_cdf_to_pmf_1D,
    fit_cdf_to_ppf_1C,
    fit_cdf_to_ppf_1D,
    fit_pdf_to_cdf_1C,
    fit_pmf_to_cdf_1D,
    fit_ppf_to_cdf_1C,
    fit_ppf_to_cdf_1D,
)
from pysatl_core.distributions.registry.constraint import (
    EdgeConstraint,
    NodeConstraint,
    NumericConstraint,
    SetConstraint,
)
from pysatl_core.distributions.registry.graph import CharacteristicRegistry
from pysatl_core.types import Kind

PDF = "pdf"
CDF = "cdf"
PPF = "ppf"
PMF = "pmf"


def _configure(reg: CharacteristicRegistry) -> None:
    pdf_to_cdf_1C = ComputationMethod[float, float](
        target=CDF, sources=[PDF], fitter=fit_pdf_to_cdf_1C
    )
    cdf_to_pdf_1C = ComputationMethod[float, float](
        target=PDF, sources=[CDF], fitter=fit_cdf_to_pdf_1C
    )
    cdf_to_ppf_1C = ComputationMethod[float, float](
        target=PPF, sources=[CDF], fitter=fit_cdf_to_ppf_1C
    )
    ppf_to_cdf_1C = ComputationMethod[float, float](
        target=CDF, sources=[PPF], fitter=fit_ppf_to_cdf_1C
    )

    pmf_to_cdf_1D = ComputationMethod[float, float](
        target=CDF,
        sources=[PMF],
        fitter=fit_pmf_to_cdf_1D,
    )
    cdf_to_pmf_1D = ComputationMethod[float, float](
        target=PMF,
        sources=[CDF],
        fitter=fit_cdf_to_pmf_1D,
    )
    cdf_to_ppf_1D = ComputationMethod[float, float](
        target=PPF, sources=[CDF], fitter=fit_cdf_to_ppf_1D
    )
    ppf_to_cdf_1D = ComputationMethod[float, float](
        target=CDF, sources=[PPF], fitter=fit_ppf_to_cdf_1D
    )

    dim1_constraint = NumericConstraint(allowed=frozenset({1}))

    kind_continuous = SetConstraint(allowed=frozenset({Kind.CONTINUOUS}))
    kind_discrete = SetConstraint(allowed=frozenset({Kind.DISCRETE}))

    reg.add_characteristic(name="cdf", is_definitive=True)
    reg.add_characteristic(name="ppf", is_definitive=True)

    reg.add_characteristic(
        name="pdf",
        is_definitive=True,
        definitive_constraint=NodeConstraint(kinds=kind_continuous),
    )
    reg.add_characteristic(
        name="pmf",
        is_definitive=True,
        definitive_constraint=NodeConstraint(kinds=kind_discrete),
    )

    edge_cont_dim1 = EdgeConstraint(
        kinds=kind_continuous, dims=dim1_constraint, requires_support=False
    )
    edge_disc_dim1 = EdgeConstraint(
        kinds=kind_discrete, dims=dim1_constraint, requires_support=True
    )

    reg.add_computation(pdf_to_cdf_1C, constraint=edge_cont_dim1)
    reg.add_computation(cdf_to_pdf_1C, constraint=edge_cont_dim1)

    reg.add_computation(cdf_to_ppf_1C, constraint=edge_cont_dim1)
    reg.add_computation(ppf_to_cdf_1C, constraint=edge_cont_dim1)

    reg.add_computation(pmf_to_cdf_1D, constraint=edge_disc_dim1)
    reg.add_computation(cdf_to_pmf_1D, constraint=edge_disc_dim1)

    reg.add_computation(ppf_to_cdf_1D, constraint=edge_disc_dim1)
    reg.add_computation(cdf_to_ppf_1D, constraint=edge_disc_dim1)


@lru_cache(maxsize=1)
def characteristic_registry() -> CharacteristicRegistry:
    """
    Return a cached, configured characteristic registry (singleton instance).

    Notes
    -----
    - The singleton is created via CharacteristicRegistry.__new__().
    - Configuration is applied exactly once per process via LRU caching.
    - Users may build and configure a separate (unconfigured) registry by
      instantiating CharacteristicRegistry() directly.
    """
    reg = CharacteristicRegistry()
    _configure(reg)
    return reg


def reset_characteristic_registry() -> None:
    """
    Reset the cached characteristic registry.
    """
    characteristic_registry.cache_clear()
    CharacteristicRegistry._reset()
