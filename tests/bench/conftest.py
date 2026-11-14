"""Shared fixtures for the accuracy-framework tests.

Mirrors the Core conftest: Core singletons are reset around every test so
runs never pollute each other.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
from scipy import stats

from pysatl_core_bench import (
    AccuracyQuery,
    AccuracyResult,
    AccuracyRunner,
    DistributionSpec,
    ParticipantSpec,
    PrecomputedTable,
    PySATLAdapter,
    ResultsStore,
    ScenarioLoader,
    ScenarioSpec,
    SourceRegistry,
    TableLookupSource,
    ValueSource,
)
from pysatl_core_bench.accuracy import reset_core_state
from pysatl_core_bench.accuracy.computations.protocol import (
    PreparedSubject,
    ValueSource as _VS,  # noqa: F401
)
from pysatl_core_bench.accuracy.dto import Provenance, SupportBounds, SupportVerdict
from pysatl_core_bench.accuracy.scenarios.spec import AccuracyQuery as _AQ  # noqa: F401
from pysatl_core_bench.accuracy.types import Capability, Readiness


@pytest.fixture(autouse=True)
def _fresh_core_registries() -> Generator[None, Any, None]:
    reset_core_state()
    yield
    reset_core_state()


@pytest.fixture()
def registry() -> SourceRegistry:
    reg = SourceRegistry()
    reg.register(PySATLAdapter())
    return reg


def normal_distribution() -> DistributionSpec:
    return DistributionSpec(
        family="Normal", parametrization="meanStd", parameters={"mu": 0.0, "sigma": 1.0}
    )


def normal_table_cdf(source_id: str = "scipy-norm-cdf", n: int = 41) -> TableLookupSource:
    points = np.linspace(-3.0, 3.0, n)
    return TableLookupSource(
        source_id,
        PrecomputedTable(
            characteristic="cdf",
            space="probability",
            points=points,
            values=stats.norm.cdf(points),
            library="scipy",
            library_version="1.15",
            coverage_tol=1e-12,
        ),
    )


def normal_table_pdf(source_id: str = "scipy-norm-pdf", n: int = 41) -> TableLookupSource:
    points = np.linspace(-3.0, 3.0, n)
    return TableLookupSource(
        source_id,
        PrecomputedTable(
            characteristic="pdf",
            space="probability",
            points=points,
            values=stats.norm.pdf(points),
            library="scipy",
            library_version="1.15",
            coverage_tol=1e-12,
        ),
    )


def normal_table_ppf(source_id: str = "scipy-norm-ppf", n: int = 41) -> TableLookupSource:
    """A ppf table on the probability levels axis (quantile levels)."""
    points = np.linspace(0.05, 0.95, n)
    return TableLookupSource(
        source_id,
        PrecomputedTable(
            characteristic="ppf",
            space="probability",
            points=points,
            values=stats.norm.ppf(points),
            library="scipy",
            library_version="1.15",
            coverage_tol=1e-12,
        ),
    )


def normal_participant(source_id: str = "pysatl-core") -> dict[str, Any]:
    return {"source_id": source_id}


def cdf_scenario_raw() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "comparison_group": "normal-cdf-vs-scipy",
        "distribution": {
            "family": "Normal",
            "parametrization": "meanStd",
            "parameters": {"mu": 0.0, "sigma": 1.0},
        },
        "subject": normal_participant(),
        "references": [{"source_id": "scipy-norm-cdf"}],
        "queries": [
            {"characteristic": "cdf", "space": "probability", "grid_hint": {"n_points": 41}}
        ],
        "regions": [{"name": "bulk", "interval": [-3.0, 3.0], "space": "probability"}],
    }


class CrashingSource(ValueSource):
    """A compute source that fails during evaluate (maps to ``failed``)."""

    def __init__(self, source_id: str = "crashing") -> None:
        self._id = source_id

    @property
    def source_id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.COMPUTE})

    @property
    def subject_id(self) -> str:
        return f"{self._id}@test"

    @property
    def provenance(self) -> Provenance:
        return Provenance(library=self._id, library_version="0")

    def support(
        self, participant: ParticipantSpec, distribution: DistributionSpec
    ) -> SupportBounds:
        return SupportBounds(left=-10.0, right=10.0)

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Any = None,
    ) -> SupportVerdict:
        return SupportVerdict(readiness=Readiness.READY, reason="crashing pretends")

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        return ["probability"]

    def prepare(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
    ) -> PreparedSubject:
        return PreparedSubject(
            participant_id=participant.participant_id, query=query, distribution=distribution
        )

    def evaluate(self, prepared: PreparedSubject, points: Any) -> Any:
        msg = "boom: the measured side crashed"
        raise RuntimeError(msg)

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Any,
        distribution: DistributionSpec,
    ) -> Any:
        msg = "boom"
        raise RuntimeError(msg)


class FollowupCrashSource(ValueSource):
    """A compute source serving ppf whose CDF ``compute`` always crashes.

    Role-agnostic: registered as the reference it exercises the
    ``invalid_reference`` mapping of a reference follow-up crash; as the
    subject it exercises ``failed``.  Non-CDF characteristics answer with
    scipy.stats.norm (compute-capable, probability-native).
    """

    def __init__(self, source_id: str = "cdf-boom") -> None:
        self._id = source_id

    @property
    def source_id(self) -> str:
        return self._id

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.COMPUTE})

    @property
    def subject_id(self) -> str:
        return f"{self._id}@test"

    @property
    def provenance(self) -> Provenance:
        return Provenance(library=self._id, library_version="0")

    def support(
        self, participant: ParticipantSpec, distribution: DistributionSpec
    ) -> SupportBounds:
        return SupportBounds(left=-10.0, right=10.0, left_closed=False, right_closed=False)

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Any = None,
    ) -> SupportVerdict:
        return SupportVerdict(readiness=Readiness.READY, reason="cdf-boom pretends")

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        return ["probability"]

    def prepare(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
    ) -> PreparedSubject:
        return PreparedSubject(
            participant_id=participant.participant_id, query=query, distribution=distribution
        )

    def evaluate(self, prepared: PreparedSubject, points: Any) -> Any:
        from pysatl_core.types import CharacteristicName

        if prepared.query.characteristic == CharacteristicName.PPF:
            return stats.norm.ppf(np.asarray(points, dtype=np.float64))
        return stats.norm.cdf(np.asarray(points, dtype=np.float64))

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Any,
        distribution: DistributionSpec,
    ) -> Any:
        from pysatl_core.types import CharacteristicName

        if query.characteristic == CharacteristicName.CDF:
            msg = "cdf boom: the follow-up crashed"
            raise RuntimeError(msg)
        return self.evaluate(PreparedSubject(participant.participant_id, query), points)


def load(raw: dict[str, Any]) -> ScenarioSpec:
    return ScenarioLoader().load(raw)


def run_scenario(
    registry: SourceRegistry,
    raw: dict[str, Any],
    store: ResultsStore | None = None,
) -> list[AccuracyResult]:
    spec = load(raw)
    return AccuracyRunner(registry, store or ResultsStore()).run(spec)


@pytest.fixture()
def registry_with_table() -> SourceRegistry:
    reg = SourceRegistry()
    reg.register(PySATLAdapter())
    reg.register(normal_table_cdf())
    return reg
