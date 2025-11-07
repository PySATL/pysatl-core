"""
Constraint primitives, node/edge applicability constraints.

This module defines two layers of constraints:

1) Value-level constraints (do not depend on a distribution):
   - SetConstraint: membership in a finite set
   - NumericConstraint: finite set and/or inclusive bounds

2) Applicability constraints (depend on a concrete distribution instance):
   - FeatureApplicability (abstract): common base with `_get_feature()` and `allows()`
   - EdgeConstraint: applicability of an edge (uses value-level constraints)
   - NodeConstraint: applicability of a node (currently supports only `kind`)

Design goals
------------
- No hard dependency on a particular distribution-type class.
- Features are read from `distribution.distribution_type.<name>` and, if absent,
  from the `distribution` object itself.
- Numeric domains can be infinite (e.g., constraints like `dim >= 2`).
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from pysatl_core.distributions.distribution import Distribution

# --------------------------------------------------------------------------- #
# Value-level constraints (do not depend on a distribution)
# --------------------------------------------------------------------------- #


class Constraint(Protocol):
    """Protocol for value-level constraints."""

    def allows(self, value: Any) -> bool: ...


@dataclass(frozen=True, slots=True)
class SetConstraint:
    """
    Membership constraint for finite domains.

    Parameters
    ----------
    allowed : frozenset[Any] | None
        If None, the constraint is not applied; otherwise `value ∈ allowed`.
    """

    allowed: frozenset[Any] | None = None

    def allows(self, value: Any) -> bool:
        return True if self.allowed is None else (value in self.allowed)


@dataclass(frozen=True, slots=True)
class NumericConstraint:
    """
    Numeric constraint for (possibly infinite) domains.

    Parameters
    ----------
    allowed : frozenset[int] | None, optional
        Exact allowed values (finite subset). If None, not checked.
    ge : int | None, optional
        Inclusive lower bound (value >= ge).
    le : int | None, optional
        Inclusive upper bound (value <= le).

    Notes
    -----
    Conditions are conjunctive. Examples:
    - dim == 1      -> allowed={1}
    - dim >= 2      -> ge=2
    - dim in {2,3}  -> allowed={2,3}
    - 2 <= dim <= 5 -> ge=2, le=5
    """

    allowed: frozenset[int] | None = None
    ge: int | None = None
    le: int | None = None

    def allows(self, value: Any) -> bool:
        try:
            v = int(value)
        except Exception:
            return False
        if self.allowed is not None and v not in self.allowed:
            return False
        if self.ge is not None and v < self.ge:
            return False
        return not (self.le is not None and v > self.le)


# --------------------------------------------------------------------------- #
# Applicability constraints (depend on a distribution)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _GraphPrimitiveConstraint(ABC):
    """
    Abstract base for applicability constraints that depend on a distribution.
    Subclasses must implement `allows(distr: Distribution) -> bool`.
    """

    kinds: SetConstraint | None = None
    dims: NumericConstraint | None = None
    feature_constraints: Mapping[str, Constraint] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.feature_constraints, MappingProxyType):
            object.__setattr__(
                self,
                "feature_constraints",
                MappingProxyType(dict(self.feature_constraints)),
            )

    @staticmethod
    def _get_feature(distr: Distribution, key: str) -> Any:
        dt = getattr(distr, "distribution_type", None)
        if dt is not None and hasattr(dt, key):
            return getattr(dt, key)
        return getattr(distr, key, None)

    def _base_allows(self, distr: Distribution) -> bool:
        if self.kinds is not None:
            k = self._get_feature(distr, "kind")
            if not self.kinds.allows(k):
                return False
        if self.dims is not None:
            d = self._get_feature(distr, "dim") or self._get_feature(distr, "dimension")
            if not self.dims.allows(d):
                return False
        for key, cons in self.feature_constraints.items():
            v = self._get_feature(distr, key)
            if not cons.allows(v):
                return False
        return True

    @abstractmethod
    def allows(self, distr: Distribution) -> bool:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class NodeConstraint(_GraphPrimitiveConstraint):
    def allows(self, distr: Distribution) -> bool:
        return self._base_allows(distr)


@dataclass(frozen=True, slots=True)
class EdgeConstraint(_GraphPrimitiveConstraint):
    requires_support: bool = False

    def allows(self, distr: Distribution) -> bool:
        if not self._base_allows(distr):
            return False
        return not (self.requires_support and getattr(distr, "_support", None) is None)
