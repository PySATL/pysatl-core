"""The single source registry (C1).

One registry maps ``source_id`` to a :class:`ValueSource`; participants are
resolved by that id only.  Roles are not types: the same adapter instance
serves a subject, a reference and a peer alike.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_core_bench.accuracy.computations.protocol import ValueSource
from pysatl_core_bench.accuracy.errors import SourceNotRegistered
from pysatl_core_bench.accuracy.types import Capability

__all__ = ["SourceRegistry"]


class SourceRegistry:
    """Registry of value sources keyed by ``source_id``."""

    def __init__(self) -> None:
        self._sources: dict[str, ValueSource] = {}

    def register(self, source: ValueSource) -> None:
        """Register (or replace) a source under its ``source_id``."""
        self._sources[source.source_id] = source

    def resolve(self, source_id: str) -> ValueSource:
        """Return the source for ``source_id`` or raise."""
        try:
            return self._sources[source_id]
        except KeyError as exc:
            msg = f"no value source registered under source_id={source_id!r}"
            raise SourceNotRegistered(msg) from exc

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._sources

    @staticmethod
    def check_role_capabilities(capabilities: frozenset[Capability], *, as_subject: bool) -> bool:
        """A subject must ``compute``; a reference may ``compute`` or ``lookup``."""
        if as_subject:
            return Capability.COMPUTE in capabilities
        return bool(capabilities) and capabilities <= {
            Capability.COMPUTE,
            Capability.LOOKUP,
        }
