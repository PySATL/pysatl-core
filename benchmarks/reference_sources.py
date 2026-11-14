"""Example external reference sources: scipy.stats.norm and mpmath tables.

These are deliberately *outside* the pysatl_core_bench package: the framework
is source-agnostic, and wrappers of external libraries are written by the
bench consumer, not by the framework.  The scipy wrapper is built with the
generic ``ExternalFunctionSource`` — its callable takes the query points as a
pysatl-core ``NumericArray``; ``distribution`` (ред. 40: the scenario-level
``DistributionSpec``) carries the measured distribution, ``participant`` the
adapter knobs and ``query`` the requested characteristic.

``mpmath_table_sources`` loads the mpmath-generated tables from
``reference_tables/`` as lookup-only ``TableLookupSource`` services over
``PrecomputedTable`` data — the production reference: the table provenance
(library, version, dps, ``table_hash``) travels with the data, and
``coverage_tol = 0`` demands an exact point match (no interpolation).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy import __version__ as scipy_version, stats as sps

from pysatl_core.types import CharacteristicName, NumericArray
from pysatl_core_bench import (
    AccuracyQuery,
    DistributionSpec,
    ExternalFunctionSource,
    ParticipantSpec,
    PrecomputedTable,
    Provenance,
    SupportBounds,
    TableLookupSource,
)

_SERVED: tuple[CharacteristicName, ...] = (
    CharacteristicName.PDF,
    CharacteristicName.CDF,
    CharacteristicName.PPF,
)


def _scipy_norm_values(
    distribution: DistributionSpec,
    participant: ParticipantSpec,  # noqa: ARG001 - adapter knobs unused by scipy
    query: AccuracyQuery,
    points: NumericArray,
) -> ArrayLike:
    """Evaluate scipy.stats.norm for one characteristic on the given points."""
    # REV-5 (no silence): the wrapper serves exactly the identity it declares.
    # Missing/wrong family, parametrization or parameters are an honest
    # refusal, never a silent substitution by N(0, 1).
    family = distribution.family
    parametrization = distribution.parametrization
    parameters = distribution.parameters
    if family != "Normal" or parametrization != "meanStd" or not isinstance(parameters, Mapping):
        msg = (
            "scipy.stats.norm serves Normal/meanStd only; the scenario declares "
            f"family={family!r}, parametrization={parametrization!r}, "
            f"parameters={dict(parameters)!r}"
        )
        raise NotImplementedError(msg)
    if "mu" not in parameters or "sigma" not in parameters:
        msg = f"scipy.stats.norm requires explicit 'mu' and 'sigma'; got {sorted(parameters)}"
        raise NotImplementedError(msg)
    mu = float(parameters["mu"])
    sigma = float(parameters["sigma"])
    arr = np.asarray(points, dtype=np.float64)
    if query.characteristic == CharacteristicName.PDF:
        return sps.norm.pdf(arr, loc=mu, scale=sigma)
    if query.characteristic == CharacteristicName.CDF:
        return sps.norm.cdf(arr, loc=mu, scale=sigma)
    if query.characteristic == CharacteristicName.PPF:
        return sps.norm.ppf(arr, loc=mu, scale=sigma)
    msg = f"scipy.stats.norm cannot serve {str(query.characteristic)!r}"
    raise NotImplementedError(msg)


def scipy_norm_source(source_id: str = "scipy-norm") -> ExternalFunctionSource:
    """scipy.stats.norm as a compute-capable source for Normal(mu, sigma)."""
    return ExternalFunctionSource(
        source_id,
        _scipy_norm_values,
        provenance=Provenance(library="scipy", library_version=scipy_version),
        characteristics=_SERVED,
        support=SupportBounds(float("-inf"), float("inf"), False, False),
    )


_TABLE_DIR = Path(__file__).resolve().parent / "reference_tables"


def mpmath_table_sources(table_dir: str | Path | None = None) -> list[TableLookupSource]:
    """Load the mpmath-generated production reference tables.

    Every ``normal_<characteristic>_*.json`` under ``reference_tables/``
    becomes a lookup-only ``TableLookupSource`` registered under
    ``mpmath-normal-<characteristic>`` (ред. 40: the service wraps the
    ``PrecomputedTable`` data).  A missing table directory yields an empty
    list; scenarios referencing the tables then fail with
    ``SourceNotRegistered`` — the honest outcome, never a silent
    substitution by another reference.
    """
    directory = Path(table_dir) if table_dir is not None else _TABLE_DIR
    sources: list[TableLookupSource] = []
    for path in sorted(directory.glob("normal_*.json")):
        table = PrecomputedTable.from_json(path)
        family = (table.family or "table").lower()
        sources.append(TableLookupSource(f"mpmath-{family}-{table.characteristic}", table))
    return sources
