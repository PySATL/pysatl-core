"""Declarative scenario structures (canonical, ready to execute).

Naming rule of the framework: a ``*Spec`` class is a *declarative structure
after canonicalization, ready to be executed*.  Raw scenario data is a plain
``dict`` and never travels past :class:`~pysatl_core_bench.accuracy.
scenarios.scenario.ScenarioLoader`.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pysatl_core.types import CharacteristicName
from pysatl_core_bench.accuracy.canonical import content_hash, frozen_mapping
from pysatl_core_bench.accuracy.types import Role

type Interval = tuple[float | None, float | None]
"""Region interval; ``None`` means "open, resolve against the support"."""

type PairPolicy = str | list[list[int]]
"""``"subject"`` | ``"symmetric"`` | explicit list of participant-index pairs."""

type StrategySpec = str | dict[str, Any]
"""``"Name"`` or ``{"name": ..., "params": {...}}``; ``None`` = first-match.

The dict form is frozen on construction (see :func:`frozen_mapping`)."""


@dataclass(frozen=True)
class DistributionSpec:
    """The distribution under measurement, canonical form (ред. 40).

    Scenario-level, shared by *all* participants of the scenario: "what is
    measured" lives next to the queries, "who measures" (``source_id``,
    ``adapter_config``, ``strategy``) lives in :class:`ParticipantSpec`.
    Every adapter receives this object and encodes it in its own way
    (``ParametricFamilyRegister`` for pysatl-core, ``loc/scale`` for
    scipy, the table identity check for lookup sources).
    """

    family: str
    parametrization: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # REV-20: mapping fields of frozen DTOs are frozen recursively.
        object.__setattr__(self, "parameters", frozen_mapping(self.parameters))

    def identity(self) -> dict[str, Any]:
        """Canonical identity of the distribution (hash/fingerprint input)."""
        return {
            "family": self.family,
            "parametrization": self.parametrization,
            "parameters": dict(self.parameters),
        }

    def fingerprint(self) -> str:
        """Stable content fingerprint; cache keys of shared sources use it."""
        return content_hash(self.identity())


@dataclass(frozen=True)
class ParticipantSpec:
    """One participant of a comparison, canonical form.

    ``participant_id`` is the canonical identity of the participant
    (source + adapter config + strategy); it is computed by the loader and
    used as the pair key component.  ``adapter_config`` is source-specific
    (adapter knobs such as ``table_path``) and validated by the adapter of
    the referenced source, not by the loader.  The distribution itself is
    scenario-level (:class:`DistributionSpec`) and deliberately *not*
    part of the participant identity.
    """

    role: Role
    source_id: str
    participant_id: str
    adapter_config: Mapping[str, Any] = field(default_factory=dict)
    strategy: StrategySpec | None = None

    def __post_init__(self) -> None:
        # REV-20: mapping fields of frozen DTOs are frozen recursively, so
        # the canonical identity cannot be mutated after loading.
        object.__setattr__(self, "adapter_config", frozen_mapping(self.adapter_config))
        if isinstance(self.strategy, Mapping):
            object.__setattr__(self, "strategy", frozen_mapping(self.strategy))

    def identity(self) -> dict[str, Any]:
        """Canonical identity contribution (used for degeneracy checks)."""
        return {
            "source_id": self.source_id,
            "adapter_config": self.adapter_config,
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class Region:
    """A region of the support with the value space of its question (N3)."""

    name: str
    interval: Interval
    space: str = "probability"


@dataclass(frozen=True)
class AccuracyQuery:
    """What is asked, in which space, with which grid hint (ex-``InputSpec``)."""

    characteristic: CharacteristicName
    space: str = "probability"
    grid_hint: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.grid_hint is not None:  # REV-20: no post-load mutation
            object.__setattr__(self, "grid_hint", frozen_mapping(self.grid_hint))


@dataclass(frozen=True)
class ScenarioSpec:
    """Canonical scenario: the only scenario form accepted past the loader.

    ``spec_hash`` is computed by the loader from the canonical form of the
    scenario and has no default: a scenario without it cannot be
    identified in the store, so the loader is the only constructor of this
    type.  The canonical dict itself is a loader-internal intermediate and
    does not travel on the public type; serializable form is the
    ``to_json_dict`` of the result DTO.

    ``invariant_check_on`` names the participants whose values the
    invariant post-check runs on; a name is ``"subject"`` or one of the
    participant's ``source_id``/``participant_id`` (unknown names are
    rejected by the loader — a silently skipped check is forbidden).  An
    empty tuple means no check.
    """

    scenario_id: str
    schema_version: str
    comparison_group: str
    distribution: DistributionSpec
    pair_policy: PairPolicy
    participants: tuple[ParticipantSpec, ...]
    regions: tuple[Region, ...]
    queries: tuple[AccuracyQuery, ...]
    spec_hash: str
    invariant_check_on: tuple[str, ...] = ()
    sort_key: float | str | None = None

    def participant_by_role(self, role: Role) -> tuple[ParticipantSpec, ...]:
        """All participants carrying ``role``."""
        return tuple(p for p in self.participants if p.role == role)

    def subject(self) -> ParticipantSpec | None:
        """The subject participant, if the scenario declares one."""
        subjects = self.participant_by_role(Role.SUBJECT)
        return subjects[0] if subjects else None


__all__ = [
    "AccuracyQuery",
    "DistributionSpec",
    "Interval",
    "PairPolicy",
    "ParticipantSpec",
    "Region",
    "ScenarioSpec",
    "StrategySpec",
    "frozen_mapping",  # re-export (kernel helper, see canonical.py)
]
