"""
Option descriptor abstractions.

Provides ``CharacteristicOption``, ``ComputationOption``, and
``EdgeOptionsDescriptor`` for declaring metadata about fitters and evaluators.

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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

type StepOptions = Mapping[int, ResolvedEdgeOptions]
"""Per-step options for a :class:`ComputationPlan`.

A mapping from **step index** (0-based, matching
:attr:`ComputationPlan.steps`) to a :class:`ResolvedEdgeOptions` that
carries validated, resolved option values for that step.

Example::

    plan = distr.explain_computation_path("ppf")
    # step 0: pdf→cdf  (accepts ``tol``)
    # step 1: cdf→ppf  (accepts ``eps``)
    opts: StepOptions = {
        0: plan.steps[0].options_descriptor.with_values(tol=0.1),
        1: plan.steps[1].options_descriptor.with_values(eps=1e-3),
    }
    ppf = distr.query_method("ppf", options=opts)
"""


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


def _resolve_options(options: tuple[_BaseOption, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve declared options from mutable keyword arguments."""
    return {option.name: option.resolve(kwargs) for option in options}


@dataclass(frozen=True, slots=True)
class EdgeOptionsDescriptor:
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
        return _resolve_options(self.characteristic_options, kwargs)

    def resolve_computation_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve only the *computation* options from *kwargs*."""
        return _resolve_options(self.computation_options, kwargs)

    def resolve_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Resolve *all* declared options (characteristic + computation) from *kwargs*.

        Consumes recognised keys from *kwargs* and returns a dict of
        ``{option_name: resolved_value}``.  Unrecognised keys are left
        in *kwargs* untouched.
        """
        return _resolve_options(self.options, kwargs)

    def with_values(self, **kwargs: Any) -> ResolvedEdgeOptions:
        """
        Validate and resolve option values, returning a :class:`ResolvedEdgeOptions`.

        This is the recommended way to build per-step options for
        :meth:`ComputationStrategy.query_method`.  Values are validated
        eagerly (type-cast + predicate check) so errors surface
        immediately rather than deep inside the strategy.

        Parameters
        ----------
        **kwargs : Any
            Option values keyed by option name.  Unrecognised keys are
            silently ignored (they are not consumed by this descriptor).

        Returns
        -------
        ResolvedEdgeOptions
            Immutable container with the validated values.

        Raises
        ------
        TypeError
            If a value cannot be cast to the declared type.
        ValueError
            If a value fails the option's validation predicate.

        Example
        -------
        ::

            plan = distr.explain_computation_path("ppf")
            resolved = plan.steps[0].options_descriptor.with_values(tol=0.1)
        """
        mutable = dict(kwargs)
        values = self.resolve_options(mutable)
        return ResolvedEdgeOptions(descriptor=self, values=values)


@dataclass(frozen=True, slots=True)
class ResolvedEdgeOptions:
    """
    Immutable container holding validated option values for one edge.

    Instances are created via :meth:`EdgeOptionsDescriptor.with_values`
    and passed to the strategy inside a :data:`StepOptions` mapping.

    Attributes
    ----------
    descriptor : EdgeOptionsDescriptor
        The descriptor that produced these values (retained for
        introspection and cache-key generation).
    values : dict[str, Any]
        Resolved ``{option_name: value}`` mapping.
    """

    descriptor: EdgeOptionsDescriptor
    values: dict[str, Any]


__all__ = [
    "CharacteristicOption",
    "ComputationOption",
    "EdgeOptionsDescriptor",
    "ResolvedEdgeOptions",
    "StepOptions",
]
