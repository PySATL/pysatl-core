from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pysatl_core.distributions.fitters.base import FitterDescriptor
    from pysatl_core.types import GenericCharacteristicName


class FitterRegistry:
    """
    Registry that stores fitter descriptors and selects the best match.

    Examples
    --------
    >>> registry = FitterRegistry()
    >>> registry.register(FITTER_PDF_TO_CDF_1C)
    >>> desc = registry.find("cdf", ["pdf"], required_tags={"continuous", "univariate"})
    """

    def __init__(self) -> None:
        self._by_key: dict[
            tuple[GenericCharacteristicName, tuple[GenericCharacteristicName, ...]],
            list[FitterDescriptor],
        ] = {}
        self._all: list[FitterDescriptor] = []

    def register(self, descriptor: FitterDescriptor) -> None:
        """
        Register a fitter descriptor.

        Parameters
        ----------
        descriptor : FitterDescriptor
            Fitter to register.
        """
        key = (descriptor.target, tuple(descriptor.sources))
        self._by_key.setdefault(key, []).append(descriptor)
        self._all.append(descriptor)

    def register_many(self, descriptors: Sequence[FitterDescriptor]) -> None:
        """Register multiple descriptors at once."""
        for d in descriptors:
            self.register(d)

    def find(
        self,
        target: GenericCharacteristicName,
        sources: Sequence[GenericCharacteristicName],
        *,
        required_tags: frozenset[str] | None = None,
    ) -> FitterDescriptor | None:
        """
        Find the best fitter for the given target and sources.

        Parameters
        ----------
        target : str
            Target characteristic name.
        sources : Sequence[str]
            Source characteristic names.
        required_tags : frozenset[str] | None
            If provided, only fitters whose ``tags`` are a superset of
            *required_tags* are considered.

        Returns
        -------
        FitterDescriptor | None
            Best matching descriptor, or ``None`` if no match.
        """
        key = (target, tuple(sources))
        candidates = self._by_key.get(key)
        if candidates is None:
            return None

        best: FitterDescriptor | None = None
        for desc in candidates:
            if required_tags is not None and not desc.tags.issuperset(required_tags):
                continue
            if best is None or desc.priority > best.priority:
                best = desc
        return best

    def find_all(
        self,
        target: GenericCharacteristicName,
        sources: Sequence[GenericCharacteristicName],
        *,
        required_tags: frozenset[str] | None = None,
    ) -> list[FitterDescriptor]:
        """
        Return *all* matching fitters, sorted by descending priority.

        Parameters
        ----------
        target : str
            Target characteristic name.
        sources : Sequence[str]
            Source characteristic names.
        required_tags : frozenset[str] | None
            Tag filter (same semantics as :meth:`find`).

        Returns
        -------
        list[FitterDescriptor]
            Matching descriptors sorted by priority (highest first).
        """
        key = (target, tuple(sources))
        candidates = self._by_key.get(key, [])
        filtered = [
            d
            for d in candidates
            if required_tags is None or d.tags.issuperset(required_tags)
        ]
        return sorted(filtered, key=lambda d: d.priority, reverse=True)

    def all_descriptors(self) -> list[FitterDescriptor]:
        """Return all registered descriptors in insertion order."""
        return list(self._all)

    def __len__(self) -> int:
        return len(self._all)

    def __contains__(self, name: str) -> bool:
        return any(d.name == name for d in self._all)
