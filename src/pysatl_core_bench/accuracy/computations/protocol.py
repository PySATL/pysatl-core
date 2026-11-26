"""The single value-source protocol.

One protocol serves everything one can ask numbers from: the measured
library adapter, precomputed tables, and on-demand computers.  Roles are
separated by registry arguments and ``capabilities``, never by type.

The distribution is a protocol argument (rev. 40): it is scenario-level
state (:class:`~...scenarios.spec.DistributionSpec`), so every method that
needs it takes it explicitly instead of reading it from the participant —
the participant carries only "who" (``source_id``, ``adapter_config``,
``strategy``).

Scale rule (N2): ``evaluate``/``compute`` have *no* scale parameter — a
source always answers in its native output scale; conversion to the question
scale is a runner-side ``OutputScaleConverter`` decision (D8), applied to the
subject and to references alike.  Fixing the scale at the protocol boundary
is what keeps the precision policy in one place: a source that silently
pre-converted its answers would hide a precision loss inside the adapter.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pysatl_core_bench.accuracy.dto import Provenance, SupportBounds, SupportVerdict
from pysatl_core_bench.accuracy.scenarios.spec import (
    AccuracyQuery,
    DistributionSpec,
    ParticipantSpec,
    frozen_mapping,
)
from pysatl_core_bench.accuracy.types import Capability

Float64Array = NDArray[np.float64]

__all__ = ["PreparedSubject", "ValueSource"]


@dataclass(frozen=True)
class PreparedSubject:
    """Prepared state of one participant for one query (cacheable per run).

    The reusable computational state built once per participant by
    :meth:`ValueSource.prepare`; ``payload`` is source-private and opaque
    to the runner.  ``distribution`` records the scenario-level
    distribution the state was built for: shared sources (e.g. the
    on-demand cache) key their caches on it, because the same participant
    can be re-prepared for different distributions across scenarios of a
    sweep.
    """

    participant_id: str
    query: AccuracyQuery
    distribution: DistributionSpec | None = None
    payload: Any = None
    path_dump: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.path_dump is not None:  # REV-20: no post-prepare mutation
            object.__setattr__(self, "path_dump", frozen_mapping(self.path_dump))


class ValueSource(Protocol):
    """Protocol for every source of values (compute and lookup alike)."""

    @property
    def source_id(self) -> str:
        """Registry key (C1)."""
        ...  # pragma: no cover (protocol body)

    @property
    def capabilities(self) -> frozenset[Capability]:
        """``{compute}``, ``{lookup}`` or ``{compute, lookup}``."""
        ...  # pragma: no cover (protocol body)

    @property
    def subject_id(self) -> str:
        """Identity of the computational object behind the source."""
        ...  # pragma: no cover (protocol body)

    @property
    def provenance(self) -> Provenance:
        """Static origin passport of the source."""
        ...  # pragma: no cover (protocol body)

    def support(
        self, participant: ParticipantSpec, distribution: DistributionSpec
    ) -> SupportBounds:
        """Bounds of the distribution support behind ``participant``."""
        ...  # pragma: no cover (protocol body)

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Float64Array | None = None,
    ) -> SupportVerdict:
        """Readiness: with ``points=None`` "can you serve this at all",
        with points "are these points served (and which are missing)"."""
        ...  # pragma: no cover (protocol body)

    def native_output_scales(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        """Output scales this source answers in without any conversion."""
        ...  # pragma: no cover (protocol body)

    def prepare(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
    ) -> PreparedSubject:
        """Build the reusable computational state once per participant."""
        ...  # pragma: no cover (protocol body)

    def evaluate(self, prepared: PreparedSubject, points: Float64Array) -> Float64Array:
        """Values at ``points`` in the *native* scale (no scale parameter)."""
        ...  # pragma: no cover (protocol body)

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Float64Array,
        distribution: DistributionSpec,
    ) -> Float64Array:
        """One-shot path (detector fallback, quantile follow-ups)."""
        ...  # pragma: no cover (protocol body)
