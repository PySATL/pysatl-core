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

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from pysatl_core.distributions.distribution import Distribution


# --------------------------------------------------------------------------- #
# Value-level constraints (do not depend on a distribution)
# --------------------------------------------------------------------------- #


class Constraint(Protocol):
    """Protocol for value-level constraints."""

    def allows(self, value: Any) -> bool: ...


@dataclass(frozen=True, slots=True)
class NonNullConstraint:
    """
    Constraint that accepts any value except None.

    Semantics
    ---------
    - value is None  -> False
    - otherwise      -> True
    """

    def allows(self, value: Any) -> bool:
        return value is not None


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
class GraphPrimitiveConstraint:
    """
    Base for applicability constraints that depend on a distribution.

    All type-level constraints are expressed via `distribution_type_feature_constraints`.
    Keys are feature names obtained from the distribution type.
    """

    distribution_type_feature_constraints: Mapping[str, Constraint] = field(
        default_factory=lambda: MappingProxyType({})
    )

    distribution_instance_feature_constraints: Mapping[str, Constraint] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Wrap provided mappings into read-only proxies for immutability."""
        if not isinstance(self.distribution_type_feature_constraints, MappingProxyType):
            object.__setattr__(
                self,
                "distribution_type_feature_constraints",
                MappingProxyType(dict(self.distribution_type_feature_constraints)),
            )
        if not isinstance(self.distribution_instance_feature_constraints, MappingProxyType):
            object.__setattr__(
                self,
                "distribution_instance_feature_constraints",
                MappingProxyType(dict(self.distribution_instance_feature_constraints)),
            )

    def allows(self, distr: Distribution) -> bool:
        features = distr.distribution_type.registry_features

        for name, cons in self.distribution_type_feature_constraints.items():
            value = features.get(name, None)
            if not cons.allows(value):
                return False

        for name, cons in self.distribution_instance_feature_constraints.items():
            value = getattr(distr, name, None)
            if not cons.allows(value):
                return False

        return True
