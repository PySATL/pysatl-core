"""
Constraint primitives and applicability constraints for distribution registry.

This module defines constraints used to determine whether certain computations
or characteristics can be applied to specific distributions based on their
features like kind, dimension, support, etc.
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


class Constraint(Protocol):
    """Protocol for value-level constraints."""

    def allows(self, value: Any) -> bool:
        """Check if the constraint allows the given value."""
        ...


@dataclass(frozen=True, slots=True)
class NonNullConstraint:
    """Constraint that rejects None values."""

    def allows(self, value: Any) -> bool:
        """Return False if value is None, True otherwise."""
        return value is not None


@dataclass(frozen=True, slots=True)
class SetConstraint:
    """
    Constraint that checks membership in a finite set.

    Parameters
    ----------
    allowed : frozenset[Any] | None
        The set of allowed values. If None, all values are allowed.
    """

    allowed: frozenset[Any] | None = None

    def allows(self, value: Any) -> bool:
        """
        Check if the value is in the allowed set.

        Returns
        -------
        bool
            True if `allowed` is None or `value` is in `allowed`, False otherwise.
        """
        return True if self.allowed is None else (value in self.allowed)


@dataclass(frozen=True, slots=True)
class NumericConstraint:
    """
    Constraint for numeric values with bounds and/or allowed values.

    Parameters
    ----------
    allowed : frozenset[int] | None
        Specific allowed integer values.
    ge : int | None
        Minimum allowed value (inclusive).
    le : int | None
        Maximum allowed value (inclusive).

    Notes
    -----
    All conditions are combined with AND logic. For example, to restrict
    dimension to exactly 1: `allowed=frozenset({1})`. To require dimension
    to be at least 2: `ge=2`. To require dimension between 2 and 5: `ge=2, le=5`.
    """

    allowed: frozenset[int] | None = None
    ge: int | None = None
    le: int | None = None

    def allows(self, value: Any) -> bool:
        """
        Check if the value satisfies all numeric constraints.

        Returns
        -------
        bool
            True if value is an integer satisfying all constraints, False otherwise.
        """
        try:
            v = int(value)
        except Exception:
            return False
        if self.allowed is not None and v not in self.allowed:
            return False
        if self.ge is not None and v < self.ge:
            return False
        return not (self.le is not None and v > self.le)


@dataclass(frozen=True, slots=True)
class GraphPrimitiveConstraint:
    """
    Constraint that checks distribution features at type and instance levels.

    This constraint evaluates whether a Distribution satisfies constraints
    based on both its DistributionType features and its own instance attributes.

    Parameters
    ----------
    distribution_type_feature_constraints : Mapping[str, Constraint]
        Constraints on distribution type features (e.g., kind, dimension).
        Features are read from the DistributionType object's registry_features.
    distribution_instance_feature_constraints : Mapping[str, Constraint]
        Constraints on distribution instance features (e.g., support).
        Features are read directly from the Distribution instance attributes.
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
        """
        Check if the distribution satisfies all constraints.

        Parameters
        ----------
        distr : Distribution
            The distribution to check.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.

        Notes
        -----
        Type features are read from `distr.distribution_type.registry_features`.
        Instance features are read directly from `distr` attributes.
        """
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
