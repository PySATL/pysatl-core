"""
Default configuration and cached accessor for the global characteristic registry.

- No auto-configuration in constructor.
- Provide ``characteristic_registry()`` with ``@lru_cache`` that builds the
  singleton instance and seeds it with a set of edges.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from functools import lru_cache

from pysatl_core.distributions.computation import ComputationMethod
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
    GraphPrimitiveConstraint,
    NonNullConstraint,
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
    """Default PySATL configuration for characteristic registry."""
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

    pdf_node_constraint = GraphPrimitiveConstraint(
        distribution_type_feature_constraints={"kind": kind_continuous}
    )
    pmf_node_constraint = GraphPrimitiveConstraint(
        distribution_type_feature_constraints={"kind": kind_discrete}
    )

    reg.add_characteristic(name=CDF, is_definitive=True)
    reg.add_characteristic(name=PPF, is_definitive=True)

    reg.add_characteristic(
        name=PDF,
        is_definitive=True,
        definitive_constraint=pdf_node_constraint,
        # TODO: Maybe it SHOULD be present even in discrete case and every other definitive char
        #  would have constant zero computation method to it
        presence_constraint=pdf_node_constraint,
    )
    reg.add_characteristic(
        name=PMF,
        is_definitive=True,
        definitive_constraint=pmf_node_constraint,
        # TODO: Maybe it SHOULD be present even in continuous case and every other definitive char
        #  would have constant zero computation method to it
        presence_constraint=pmf_node_constraint,
    )

    edge_cont_dim1 = GraphPrimitiveConstraint(
        distribution_type_feature_constraints={
            "kind": kind_continuous,
            "dimension": dim1_constraint,
        },
    )

    edge_disc_dim1 = GraphPrimitiveConstraint(
        distribution_type_feature_constraints={
            "kind": kind_discrete,
            "dimension": dim1_constraint,
        },
        distribution_instance_feature_constraints={
            "support": NonNullConstraint(),
        },
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
      instantiating ``CharacteristicRegistry()`` directly.
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
