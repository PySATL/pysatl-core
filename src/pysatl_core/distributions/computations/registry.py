"""
Fitter registry for descriptor lookup and matching.

Provides ``FitterRegistry`` for registering, querying, and prioritising
fitter descriptors by target, sources, and constraint tags.

The module-level ``fitter_registry()`` function returns a process-wide
singleton that is populated lazily on first access with all built-in
descriptors from :mod:`pysatl_core.distributions.computations.continuous`
and :mod:`pysatl_core.distributions.computations.discrete`.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pysatl_core.distributions.computations.descriptors import FitterDescriptor
    from pysatl_core.types import GenericCharacteristicName


class FitterRegistry:
    """
    Registry that stores fitter descriptors and selects the best match.

    Examples
    --------
    >>> registry = FitterRegistry()
    >>> registry.register(some_descriptor)
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
        Find the first fitter matching the given target and sources.

        Parameters
        ----------
        target : str
            Target characteristic name.
        sources : Sequence[str]
            Source characteristic names.
        required_tags : frozenset[str] | None
            If provided, only fitters whose ``constraint_tags`` are a superset
            of *required_tags* are considered.

        Returns
        -------
        FitterDescriptor | None
            First matching descriptor, or ``None`` if no match.
        """
        key = (target, tuple(sources))
        candidates = self._by_key.get(key)
        if candidates is None:
            return None

        for desc in candidates:
            if required_tags is not None and not desc.constraint_tags.issuperset(required_tags):
                continue
            return desc
        return None

    def find_all(
        self,
        target: GenericCharacteristicName,
        sources: Sequence[GenericCharacteristicName],
        *,
        required_tags: frozenset[str] | None = None,
    ) -> list[FitterDescriptor]:
        """
        Return *all* matching fitters in registration order.

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
            Matching descriptors in registration order.
        """
        key = (target, tuple(sources))
        candidates = self._by_key.get(key, [])
        return [
            d
            for d in candidates
            if required_tags is None or d.constraint_tags.issuperset(required_tags)
        ]

    def all_descriptors(self) -> list[FitterDescriptor]:
        """Return all registered descriptors in insertion order."""
        return list(self._all)

    def __len__(self) -> int:
        return len(self._all)

    def __contains__(self, name: str) -> bool:
        return any(d.name == name for d in self._all)


@lru_cache(maxsize=1)
def fitter_registry() -> FitterRegistry:
    """
    Return the process-wide singleton ``FitterRegistry``, populated lazily.

    All built-in continuous and discrete fitter descriptors are created and
    registered on the first call.  Subsequent calls return the cached instance.

    Notes
    -----
    - Descriptors are **not** created at import time; they are built here on
      first access, keeping module-level side-effects to a minimum.
    - Users who need a custom registry should instantiate ``FitterRegistry()``
      directly and populate it themselves.
    """
    from pysatl_core.distributions.computations.continuous import (
        _build_continuous_descriptors,
    )
    from pysatl_core.distributions.computations.discrete import (
        _build_discrete_descriptors,
    )

    reg = FitterRegistry()
    reg.register_many(_build_continuous_descriptors())
    reg.register_many(_build_discrete_descriptors())
    return reg


def reset_fitter_registry() -> None:
    """Reset the cached fitter registry (useful in tests)."""
    fitter_registry.cache_clear()


__all__ = [
    "FitterRegistry",
    "fitter_registry",
    "reset_fitter_registry",
]
