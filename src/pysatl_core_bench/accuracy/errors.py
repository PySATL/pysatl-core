"""Exception hierarchy of the accuracy benchmarking model.

Rules (framework conventions):

- *Pipeline states* (e.g. a rejected scenario) are exceptions; the run never
  starts.  They are never reported as result statuses.
- *Outcome statuses* of a pair are values of :class:`~pysatl_core_bench.accuracy.types.Status`,
  never exceptions.
- Exceptions escaping a source call are treated as data/execution defects
  and mapped to a result status by the runner.  The mapping depends on the
  stage where the exception escaped: at *assess* time the runner cannot yet
  distinguish a broken source from an unservable configuration, so the
  readiness layer collapses both into ``UNAVAILABLE`` and the per-role
  readiness map applies (reference -> ``invalid_reference``, subject ->
  ``unsupported_configuration`` — the crash detail travels in the verdict
  reason and reaches ``status_explanation``).  At *prepare/evaluate* time
  the failure is definitely execution: a crash of the measured (subject)
  side becomes ``Status.FAILED``, a crash of the reference side becomes
  ``Status.INVALID_REFERENCE``.  The same role rule applies to a quantile
  follow-up crashing inside metric evaluation (ред. 46): the subject CDF
  follow-up (``e2_self_consistency``) is ``failed``, the reference CDF
  follow-up (``e2_true``) is ``invalid_reference``.  Genuine framework bugs
  raise :class:`FrameworkDefect` and propagate.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class BenchError(Exception):
    """Base class for all pysatl_core_bench errors."""


class ScenarioRejected(BenchError):
    """A scenario cannot be loaded: the run does not start.

    This is a pipeline state, *not* a result status.  Raised for unreadable
    configurations, unknown fields, or contradictory grammar.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SourceNotRegistered(BenchError):
    """A scenario references a ``source_id`` absent from the registry."""


class SourceOperationError(BenchError):
    """A value source failed while producing values or metadata."""


class AdapterConfigurationError(BenchError):
    """A participant configuration is not supported by its adapter.

    Raised by adapters internally and translated into a
    ``Readiness.UNSUPPORTED_CONFIGURATION`` verdict; never a silent fallback.
    """


class GridMismatchError(BenchError):
    """Pairwise agreement requested over incomparable results.

    The operation is defined only for two results of the same scenario pair
    measured on the same grids; a differing ``grid_hash`` or a differing
    pair is an explicit error, never a silent comparison of quantities that
    are not comparable.
    """


class FrameworkDefect(BenchError):
    """An invariant of the framework itself was violated.

    Unlike a source failure this always indicates a bug in pysatl_core_bench
    (or a misuse of its API), so it propagates instead of becoming a status.
    """


__all__ = [
    "AdapterConfigurationError",
    "BenchError",
    "FrameworkDefect",
    "GridMismatchError",
    "ScenarioRejected",
    "SourceNotRegistered",
    "SourceOperationError",
]
