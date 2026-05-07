"""
Default configuration and cached accessor for the global characteristic registry.

Notes
-----
No auto-configuration happens in the constructor. The module provides
``characteristic_registry()`` with ``@lru_cache`` that builds the singleton
instance and seeds it with a set of edges.

At configuration time, ``to_computation_method()`` is called on each
``FitterDescriptor`` to build a ``FitterMethod`` (a lightweight wrapper that
holds the fitter callable) and store it as a graph edge. The actual
``fitter(distribution, **options)`` call - the expensive precomputation - happens
on demand when the strategy resolves a path via ``query_method``.

Lookup of fitter descriptors by ``(target, sources, tags)`` is delegated to
``FitterRegistry``: ``configuration`` does not depend on individual descriptor
identifiers but only on the (source, target) pairs and the constraint tag set
they are expected to satisfy.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from functools import lru_cache
from typing import TYPE_CHECKING

from pysatl_core.distributions.computations.registry import fitter_registry
from pysatl_core.distributions.registry.constraint import (
    GraphPrimitiveConstraint,
    NonNullConstraint,
    NumericConstraint,
    SetConstraint,
)
from pysatl_core.distributions.registry.graph import CharacteristicRegistry
from pysatl_core.types import CharacteristicName, Kind

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pysatl_core.types import GenericCharacteristicName


_CONTINUOUS_1D_TAGS: frozenset[str] = frozenset({"continuous", "univariate"})
_DISCRETE_1D_TAGS: frozenset[str] = frozenset({"discrete", "univariate"})


def _add_edges(
    reg: CharacteristicRegistry,
    pairs: Iterable[tuple[GenericCharacteristicName, GenericCharacteristicName]],
    *,
    tags: frozenset[str],
    constraint: GraphPrimitiveConstraint,
) -> None:
    """
    Look up fitters by ``(source, target, tags)`` and attach their computation
    methods to the characteristic graph under the same edge constraint.

    Parameters
    ----------
    reg : CharacteristicRegistry
        Target characteristic graph.
    pairs : Iterable[tuple[str, str]]
        ``(source, target)`` pairs to register.
    tags : frozenset[str]
        Required constraint tags for the descriptor lookup.
    constraint : GraphPrimitiveConstraint
        Edge constraint applied to every added computation.

    Raises
    ------
    RuntimeError
        If no descriptor matches one of the requested ``(source, target, tags)``.
    """
    fitter_reg = fitter_registry()
    for src, tgt in pairs:
        descriptor = fitter_reg.find(tgt, [src], required_tags=tags)
        if descriptor is None:
            raise RuntimeError(
                f"No fitter descriptor registered for {src} -> {tgt} "
                f"with required tags {sorted(tags)}."
            )
        reg.add_computation(
            descriptor.to_computation_method(),
            constraint=constraint,
            options_descriptor=descriptor.to_options_descriptor(),
        )


def _configure(reg: CharacteristicRegistry) -> None:
    """Default PySATL configuration for characteristic registry."""
    dim1_constraint = NumericConstraint(allowed=frozenset({1}))
    kind_continuous = SetConstraint(allowed=frozenset({Kind.CONTINUOUS}))
    kind_discrete = SetConstraint(allowed=frozenset({Kind.DISCRETE}))

    pdf_node_constraint = GraphPrimitiveConstraint(
        distribution_type_feature_constraints={"kind": kind_continuous}
    )
    pmf_node_constraint = GraphPrimitiveConstraint(
        distribution_type_feature_constraints={"kind": kind_discrete}
    )

    reg.add_characteristic(name=CharacteristicName.CDF, is_definitive=True)
    reg.add_characteristic(name=CharacteristicName.PPF, is_definitive=True)

    reg.add_characteristic(
        name=CharacteristicName.PDF,
        is_definitive=True,
        definitive_constraint=pdf_node_constraint,
        # TODO: Maybe it SHOULD be present even in discrete case and every other definitive char
        #  would have constant zero computation method to it
        presence_constraint=pdf_node_constraint,
    )
    reg.add_characteristic(
        name=CharacteristicName.PMF,
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

    _add_edges(
        reg,
        pairs=(
            (CharacteristicName.PDF, CharacteristicName.CDF),
            (CharacteristicName.CDF, CharacteristicName.PDF),
            (CharacteristicName.CDF, CharacteristicName.PPF),
            (CharacteristicName.PPF, CharacteristicName.CDF),
        ),
        tags=_CONTINUOUS_1D_TAGS,
        constraint=edge_cont_dim1,
    )

    _add_edges(
        reg,
        pairs=(
            (CharacteristicName.PMF, CharacteristicName.CDF),
            (CharacteristicName.CDF, CharacteristicName.PMF),
            (CharacteristicName.CDF, CharacteristicName.PPF),
            (CharacteristicName.PPF, CharacteristicName.CDF),
        ),
        tags=_DISCRETE_1D_TAGS,
        constraint=edge_disc_dim1,
    )


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
