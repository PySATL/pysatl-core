"""Domain vocabulary of the accuracy benchmarking model.

Result statuses are exactly the five names from pysatl-core issue #124 and
must never be changed or extended.  Readiness is a deliberately separate
enum: it describes a *source* state and is mapped to a result ``Status``
only by the runner (sources never emit result statuses themselves).
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from enum import StrEnum

SCHEMA_VERSION = "1.0"
"""Version of the benchmark scenario/result JSON schema."""

PROBABILITY_SPACE = "probability"
SURVIVAL_SPACE = "survival"

KNOWN_SPACES = (PROBABILITY_SPACE, SURVIVAL_SPACE)


class Status(StrEnum):
    """Outcome of one participant pair. Exactly five values, names frozen."""

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"
    FAILED = "failed"
    INVALID_REFERENCE = "invalid_reference"


STATUS_PRIORITY: tuple[Status, ...] = (
    Status.FAILED,
    Status.UNSUPPORTED_CONFIGURATION,
    Status.NOT_APPLICABLE,
)
"""Priority used to collapse a doubly-unready pair into a single status.

Higher priority (earlier in the tuple) wins; the explanation must still
list every unready participant.
"""


class Readiness(StrEnum):
    """Readiness verdict of a value source. Never reused as a ``Status``."""

    READY = "ready"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"
    NOT_COVERED = "not_covered"
    UNAVAILABLE = "unavailable"


class Role(StrEnum):
    """Role of a participant inside one scenario."""

    SUBJECT = "subject"
    REFERENCE = "reference"
    PEER = "peer"


class Capability(StrEnum):
    """What a value source can do with respect to points."""

    COMPUTE = "compute"
    LOOKUP = "lookup"


__all__ = [
    "SCHEMA_VERSION",
    "PROBABILITY_SPACE",
    "SURVIVAL_SPACE",
    "KNOWN_SPACES",
    "STATUS_PRIORITY",
    "Capability",
    "Readiness",
    "Role",
    "Status",
]
