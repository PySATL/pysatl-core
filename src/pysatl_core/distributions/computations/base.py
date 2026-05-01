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


@dataclass(frozen=True, slots=True)
class OptionsDescriptor:
    """
    Compact, graph-level form of a computation descriptor.

    A *short* descriptor carries only the metadata required by the strategy
    when resolving caller-supplied ``**options`` against a specific edge in
    the characteristic graph.  It is the graph-level primitive: it is
    immutable, decoupled from the heavy fitter/evaluator callable, and
    cheap to attach to every :class:`ComputationEdgeMeta` so the strategy
    can route options per-edge along a multi-hop conversion path.

    Attributes
    ----------
    name : str
        Descriptor identifier (matches the originating
        :class:`FitterDescriptor` / :class:`EvaluatorDescriptor` name).
        Empty by default for edges that were declared without a descriptor.
    characteristic_options : tuple[CharacteristicOption, ...]
        Options intrinsic to the characteristic.
    computation_options : tuple[ComputationOption, ...]
        Options controlling the numerical algorithm.
    """

    name: str = ""
    characteristic_options: tuple[CharacteristicOption, ...] = ()
    computation_options: tuple[ComputationOption, ...] = ()

    @property
    def options(self) -> tuple[_BaseOption, ...]:
        """All options (characteristic first, then computation)."""
        return self.characteristic_options + self.computation_options

    def resolve_characteristic_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve only the *characteristic* options from *kwargs*."""
        return {opt.name: opt.resolve(kwargs) for opt in self.characteristic_options}

    def resolve_computation_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve only the *computation* options from *kwargs*."""
        return {opt.name: opt.resolve(kwargs) for opt in self.computation_options}

    def resolve_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve *all* declared options (characteristic + computation) from *kwargs*.

        Consumes recognised keys from *kwargs* and returns a dict of
        ``{option_name: resolved_value}``.  Unrecognised keys are left
        in *kwargs* untouched.
        """
        return {opt.name: opt.resolve(kwargs) for opt in self.options}


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
        return {opt.name: opt.resolve(kwargs) for opt in self.characteristic_options}

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
        return {opt.name: opt.resolve(kwargs) for opt in self.computation_options}

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
        return {opt.name: opt.resolve(kwargs) for opt in self.options}

    def to_options_descriptor(self) -> OptionsDescriptor:
        """
        Return the :class:`OptionsDescriptor` projection of this descriptor.

        The returned object carries only the option metadata (and the
        descriptor name for traceability) required by the strategy when
        resolving user-supplied ``**options`` against a specific edge in
        the characteristic graph.  It deliberately omits the heavy callable
        (``fitter`` / ``evaluator``) and the matching metadata
        (``target``, ``sources``, ``constraint_tags``) which are already
        encoded in the graph topology and edge constraints.
        """
        return OptionsDescriptor(
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
    "CharacteristicOption",
    "ComputationOption",
    "EvaluatorDescriptor",
    "FitterDescriptor",
    "OptionsDescriptor",
]
