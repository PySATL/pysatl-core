"""
Computation descriptor abstractions.

Provides ``CharacteristicOption``, ``ComputationOption``,
``FitterDescriptor``, and ``EvaluatorDescriptor`` for declaring metadata
about fitters and evaluators.

Option taxonomy
---------------
``CharacteristicOption``
    Describes a parameter that is *intrinsic to the characteristic itself*
    (e.g. ``eps`` or ``x0`` for PPF).  These options are shared between the
    fitter (pre-computation / caching path) and the evaluator (direct-query
    path).  Because they affect the *meaning* of the result they must be
    encoded into the cache key.

``ComputationOption``
    Describes a parameter that controls the *numerical algorithm* used to
    compute the characteristic (e.g. ``max_iter``, ``x_tol``).  These are
    specific to a particular fitter implementation and do **not** affect the
    evaluator.

Passing options — ``TypedDict`` + ``Unpack`` pattern
-----------------------------------------------------
Concrete fitters and evaluators declare their accepted keyword arguments via
``TypedDict`` subclasses and annotate their signatures with
``**kwargs: Unpack[MyOptionsDict]``.  This gives static type-checkers full
visibility while keeping the runtime interface simple ``**kwargs``.

Example::

    from typing import TypedDict
    from typing_extensions import Unpack

    class PpfCharacteristicOptions(TypedDict, total=False):
        eps: float
        x0: float

    class PpfComputationOptions(TypedDict, total=False):
        max_iter: int
        x_tol: float

    def fit_cdf_to_ppf(
        distribution: Distribution,
        /,
        **kwargs: Unpack[PpfComputationOptions],
    ) -> FittedComputationMethod: ...
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
class _BaseOption:
    """
    Common base for option descriptors.

    Attributes
    ----------
    name : str
        Option name as it appears in keyword arguments.
    type : type
        Expected Python type (``int``, ``float``, …).
    default : Any
        Default value used when the caller does not supply the option.
    description : str
        Human-readable description shown in documentation / introspection.
    validate : Callable[[Any], bool] | None
        Optional predicate.  When not ``None``, the option value is rejected
        (``ValueError``) if ``validate(value)`` returns ``False``.
    """

    name: str
    type: type
    default: Any
    description: str = ""
    validate: Callable[[Any], bool] | None = None

    def resolve(self, kwargs: dict[str, Any]) -> Any:
        """
        Extract and validate the option from *kwargs*.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Caller-supplied keyword arguments.  The key matching
            `name` is consumed (popped) if present.

        Returns
        -------
        Any
            Resolved value (caller-supplied or default), cast to `type`.

        Raises
        ------
        ValueError
            If the resolved value fails the `validate` predicate.
        TypeError
            If the value cannot be cast to `type`.
        """
        raw = kwargs.pop(self.name, self.default)
        try:
            value = self.type(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Option '{self.name}': cannot convert {raw!r} to {self.type.__name__}"
            ) from exc
        if self.validate is not None and not self.validate(value):
            raise ValueError(f"Option '{self.name}': value {value!r} failed validation.")
        return value


@dataclass(frozen=True, slots=True)
class CharacteristicOption(_BaseOption):
    """
    Option that is *intrinsic to the characteristic* being computed.

    Characteristic options are shared between the fitter (pre-computation /
    caching path) and the evaluator (direct-query path).  Because they affect
    the *meaning* of the result they must be encoded into the cache key.

    Examples: ``eps`` (tail threshold for PPF), ``x0`` (starting point for
    bound search), ``excess`` (excess kurtosis flag).

    These options are declared in ``FitterDescriptor.characteristic_options``
    and ``EvaluatorDescriptor.characteristic_options``.
    """


@dataclass(frozen=True, slots=True)
class ComputationOption(_BaseOption):
    """
    Option that controls the *numerical algorithm* used to compute a
    characteristic.

    Computation options are specific to a particular fitter implementation
    and do **not** affect the evaluator.  They influence only the speed /
    accuracy trade-off of the fitting step, not the semantics of the result.

    Examples: ``max_iter`` (bisection iterations), ``x_tol`` (bracket
    tolerance), ``limit`` (quad subdivisions), ``n_q_grid`` (PPF grid size).

    These options are declared in ``FitterDescriptor.computation_options``.
    """


def _resolve_options(
    options: tuple[_BaseOption, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a tuple of option descriptors from *kwargs* (mutates *kwargs*)."""
    return {opt.name: opt.resolve(kwargs) for opt in options}


def _option_names(options: tuple[_BaseOption, ...]) -> tuple[str, ...]:
    return tuple(opt.name for opt in options)


def _option_defaults(options: tuple[_BaseOption, ...]) -> dict[str, Any]:
    return {opt.name: opt.default for opt in options}


@dataclass(frozen=True, slots=True)
class FitterDescriptor:
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

    name: str
    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    fitter: FitterFunc
    characteristic_options: tuple[CharacteristicOption, ...] = ()
    computation_options: tuple[ComputationOption, ...] = ()
    constraint_tags: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    @property
    def options(self) -> tuple[_BaseOption, ...]:
        """All options (characteristic first, then computation)."""
        return (*self.characteristic_options, *self.computation_options)

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

    def option_names(self) -> tuple[str, ...]:
        """Return the names of all declared options (characteristic + computation)."""
        return _option_names(self.options)

    def option_defaults(self) -> dict[str, Any]:
        """Return ``{name: default}`` for every declared option."""
        return _option_defaults(self.options)

    def characteristic_option_names(self) -> tuple[str, ...]:
        """Return the names of characteristic options only."""
        return _option_names(self.characteristic_options)

    def computation_option_names(self) -> tuple[str, ...]:
        """Return the names of computation options only."""
        return _option_names(self.computation_options)


@dataclass(frozen=True, slots=True)
class EvaluatorDescriptor:
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

    name: str
    target: GenericCharacteristicName
    sources: Sequence[GenericCharacteristicName]
    evaluator: EvaluatorFunc
    characteristic_options: tuple[CharacteristicOption, ...] = ()
    computation_options: tuple[ComputationOption, ...] = ()
    constraint_tags: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    @property
    def options(self) -> tuple[_BaseOption, ...]:
        """All options (characteristic first, then computation)."""
        return (*self.characteristic_options, *self.computation_options)

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

    def resolve_characteristic_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve only the *characteristic* options from *kwargs*.

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

        For evaluators these are applied on **every call** (not just at
        fit-time as for fitters).

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

    def option_names(self) -> tuple[str, ...]:
        """Return the names of all declared options (characteristic + computation)."""
        return _option_names(self.options)

    def option_defaults(self) -> dict[str, Any]:
        """Return ``{name: default}`` for every declared option."""
        return _option_defaults(self.options)

    def characteristic_option_names(self) -> tuple[str, ...]:
        """Return the names of characteristic options only."""
        return _option_names(self.characteristic_options)

    def computation_option_names(self) -> tuple[str, ...]:
        """Return the names of computation options only."""
        return _option_names(self.computation_options)


__all__ = [
    "CharacteristicOption",
    "ComputationOption",
    "FitterDescriptor",
    "EvaluatorDescriptor",
]
