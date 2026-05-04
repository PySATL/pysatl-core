"""
Computation descriptor abstractions.

Provides ``FitterDescriptor`` and ``EvaluatorDescriptor`` for declaring
metadata about fitters and evaluators.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pysatl_core.distributions.computations.options import (
    CharacteristicOption,
    ComputationOption,
    EdgeOptionsDescriptor,
    _BaseOption,
    _resolve_options,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pysatl_core.distributions.computations.computation import (
        EvaluatorMethod,
        FitterMethod,
    )
    from pysatl_core.types import (
        EvaluatorFunc,
        FitterFunc,
        GenericCharacteristicName,
    )


@dataclass(frozen=True, slots=True)
class _BaseDescriptor:
    """
    Abstract base for computation descriptors.

    Holds the common fields and option-resolution methods shared between
    ``FitterDescriptor`` and ``EvaluatorDescriptor``.

    Attributes
    ----------
    name : str
        Unique human-readable identifier.
    target : GenericCharacteristicName
        Characteristic produced by this descriptor.
    sources : Sequence[GenericCharacteristicName]
        Characteristics consumed by this descriptor.
    characteristic_options : tuple[CharacteristicOption, ...]
        Options intrinsic to the characteristic (shared between fitters and
        evaluators, encoded into the cache key).
    computation_options : tuple[ComputationOption, ...]
        Options controlling the numerical algorithm.
    constraint_tags : frozenset[str]
        Constraint tags used for matching.
    description : str
        Human-readable summary.
    """

    name: str
    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    characteristic_options: tuple[CharacteristicOption, ...] = ()
    computation_options: tuple[ComputationOption, ...] = ()
    constraint_tags: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    @property
    def options(self) -> tuple[_BaseOption, ...]:
        """All options (characteristic first, then computation)."""
        return self.characteristic_options + self.computation_options

    def resolve_characteristic_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve only the *characteristic* options from *kwargs*.

        Consumes recognised keys from *kwargs* and returns a dict of
        ``{option_name: resolved_value}``.  Unrecognised keys are left
        in *kwargs* untouched.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Mutable keyword-argument dict from the caller.

        Returns
        -------
        dict[str, Any]
        """
        return _resolve_options(self.characteristic_options, kwargs)

    def resolve_computation_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve only the *computation* options from *kwargs*.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Mutable keyword-argument dict from the caller.

        Returns
        -------
        dict[str, Any]
        """
        return _resolve_options(self.computation_options, kwargs)

    def resolve_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve *all* declared options (characteristic + computation) from *kwargs*.

        Consumes recognised keys from *kwargs* and returns a dict of
        ``{option_name: resolved_value}``.  Unrecognised keys are left
        in *kwargs* untouched.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Mutable keyword-argument dict from the caller.

        Returns
        -------
        dict[str, Any]
            Mapping from option name to resolved (validated, typed) value.
        """
        return _resolve_options(self.options, kwargs)

    def to_options_descriptor(self) -> EdgeOptionsDescriptor:
        """
        Return the :class:`EdgeOptionsDescriptor` projection of this descriptor.

        The returned object carries only the option metadata (and the
        descriptor name for traceability) required by the strategy when
        resolving user-supplied ``**options`` against a specific edge in
        the characteristic graph.  It deliberately omits the heavy callable
        (``fitter`` / ``evaluator``) and the matching metadata
        (``target``, ``sources``, ``constraint_tags``) which are already
        encoded in the graph topology and edge constraints.
        """
        return EdgeOptionsDescriptor(
            name=self.name,
            characteristic_options=self.characteristic_options,
            computation_options=self.computation_options,
        )

    def option_names(self) -> tuple[str, ...]:
        """Return the names of all declared options (characteristic + computation)."""
        return tuple(opt.name for opt in self.options)

    def option_defaults(self) -> dict[str, Any]:
        """Return ``{name: default}`` for every declared option."""
        return {opt.name: opt.default for opt in self.options}

    def characteristic_option_names(self) -> tuple[str, ...]:
        """Return the names of characteristic options only."""
        return tuple(opt.name for opt in self.characteristic_options)

    def computation_option_names(self) -> tuple[str, ...]:
        """Return the names of computation options only."""
        return tuple(opt.name for opt in self.computation_options)


@dataclass(frozen=True, slots=True)
class FitterDescriptor(_BaseDescriptor):
    """
    Complete metadata for a cacheable fitter.

    A fitter performs expensive precomputation and returns a
    ``FittedComputationMethod`` that can be cached and reused.

    Parameters
    ----------
    name : str
        Unique human-readable identifier (e.g. ``"pdf_to_cdf_1C"``).
    target : GenericCharacteristicName
        Characteristic produced by this fitter.
    sources : Sequence[GenericCharacteristicName]
        Characteristics consumed by this fitter (typically length 1).
    fitter : FitterFunc
        The actual fitting callable.
    characteristic_options : tuple[CharacteristicOption, ...]
        Options intrinsic to the characteristic (shared with evaluators,
        encoded into the cache key).
    computation_options : tuple[ComputationOption, ...]
        Options controlling the numerical algorithm (fitter-specific).
    constraint_tags : frozenset[str]
        Constraint tags used for matching (e.g. ``{"continuous", "univariate"}``).
    description : str
        Human-readable summary of what the fitter does.

    Notes
    -----
    The combined ``options`` property returns all options (characteristic
    first, then computation) for backwards-compatible resolution.
    """

    fitter: FitterFunc = field(default=None)  # type: ignore[assignment]

    def to_computation_method(self) -> FitterMethod:
        """
        Build a `FitterMethod` (computation method) from this descriptor.

        Returns
        -------
        FitterMethod
        """
        from pysatl_core.distributions.computations.computation import FitterMethod

        return FitterMethod(
            target=self.target,
            sources=list(self.sources),
            fitter=self.fitter,
        )


@dataclass(frozen=True, slots=True)
class EvaluatorDescriptor(_BaseDescriptor):
    """
    Complete metadata for a non-cacheable evaluator.

    An evaluator is lightweight and called on every query without caching.
    It returns the computed value directly rather than a
    ``FittedComputationMethod``.

    Parameters
    ----------
    name : str
        Unique human-readable identifier.
    target : GenericCharacteristicName
        Characteristic produced by this evaluator.
    sources : Sequence[GenericCharacteristicName]
        Characteristics consumed by this evaluator (typically length 1).
    evaluator : EvaluatorFunc
        The actual evaluator callable.
    characteristic_options : tuple[CharacteristicOption, ...]
        Options intrinsic to the characteristic (shared with fitters).
        These affect the *meaning* of the result.
    computation_options : tuple[ComputationOption, ...]
        Options controlling the numerical algorithm used **on every call**.
        Unlike fitter computation options (used once at fit-time), evaluator
        computation options are applied on each invocation.  Examples:
        integration tolerance, finite-difference step, iteration limit.
    constraint_tags : frozenset[str]
        Constraint tags used for matching.
    description : str
        Human-readable summary of what the evaluator does.
    """

    evaluator: EvaluatorFunc = field(default=None)  # type: ignore[assignment]

    def to_computation_method(self) -> EvaluatorMethod:
        """
        Build an `EvaluatorMethod` (computation method) from this descriptor.

        Returns
        -------
        EvaluatorMethod
        """
        from pysatl_core.distributions.computations.computation import EvaluatorMethod

        return EvaluatorMethod(
            target=self.target,
            sources=list(self.sources),
            evaluator=self.evaluator,
        )


__all__ = [
    "EvaluatorDescriptor",
    "FitterDescriptor",
]
