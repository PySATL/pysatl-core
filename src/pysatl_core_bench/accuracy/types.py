"""Domain vocabulary of the accuracy benchmarking model.

This module exists so that every stage of the pipeline speaks the same
language: result statuses, readiness verdicts, participant roles and
value scales are named exactly once, here, and every stage imports the
name instead of re-encoding it.  A vocabulary drift between stages would
silently distort statuses and comparisons, so the module is the single
naming authority.

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

SCHEMA_VERSION = "1.2"
"""Version of the benchmark scenario/result JSON schema.

- 1.1 (rev. 60): the value-``space`` grammar field is renamed to
  ``output_scale`` and explicit user ``points`` enter the query grammar;
  a file written in the 1.0 grammar is rejected instead of being silently
  reinterpreted.
- 1.2 (rev. 61): purely additive — optional ``grid_hint.tail_extension``
  (user tail-length override) and ``grid_hint.densify`` (densification
  opt-out) enter the grid hint, and ``MetricRecord`` carries the AP-1
  excluded-points dictionary (``excluded_points``).  1.1-declared files
  stay valid (see the loader's ``_SUPPORTED_SCHEMA_VERSIONS``); results
  written by this generation carry ``1.2``.
"""

PROBABILITY_SCALE = "probability"
SURVIVAL_SCALE = "survival"

KNOWN_OUTPUT_SCALES = (PROBABILITY_SCALE, SURVIVAL_SCALE)
"""The output scales a characteristic answer may be reported in (v1).

An ``output_scale`` is the axis the values live on: ``probability`` for a
CDF/PDF value, ``survival`` for a tail probability ``S = 1 - F``.  The
tuple is the loader's whitelist — a question asked on any other axis is
rejected rather than measured on an undefined scale.  Every graph
characteristic of pysatl-core answers natively on the probability scale;
the survival scale is reached only through the output-scale converter.
"""


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
    "PROBABILITY_SCALE",
    "SURVIVAL_SCALE",
    "KNOWN_OUTPUT_SCALES",
    "STATUS_PRIORITY",
    "Capability",
    "Readiness",
    "Role",
    "Status",
]
