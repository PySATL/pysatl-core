"""
Computation and Sampling Strategies

This module defines strategies for computing distribution characteristics
and generating random samples.
"""

from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov, Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from pysatl_core.distributions.computations.options import (
    EdgeOptionsDescriptor,
    ResolvedEdgeOptions,
    _BaseOption,
)
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.types import Method, NumericArray

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from pysatl_core.distributions.computations.computation import (
        AnalyticalComputation,
        FittedComputationMethod,
    )
    from pysatl_core.distributions.computations.options import StepOptions
    from pysatl_core.distributions.distribution import Distribution
    from pysatl_core.distributions.registry.graph import RegistryView
    from pysatl_core.distributions.registry.graph_primitives import (
        ComputationEdgeMeta,
        EdgeMeta,
    )
    from pysatl_core.types import GenericCharacteristicName, LabelName


# --------------------------------------------------------------------------- #
# Execution plan (introspection of how a strategy will compute a state)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ComputationStep:
    """
    One step of a strategy's execution plan for a target characteristic.

    Attributes
    ----------
    target : GenericCharacteristicName
        Characteristic produced by this step.
    sources : tuple[GenericCharacteristicName, ...]
        Characteristics consumed by this step.
    edge_kind : str
        Underlying edge kind: ``"analytical_loop"``,
        ``"transformation_loop"`` or ``"computation"``.
    method_name : str
        Human-readable identifier of the underlying method (descriptor
        ``name`` when available, otherwise ``target``).
    options_descriptor : EdgeOptionsDescriptor
        Compact descriptor describing which user-supplied options will
        be consumed at this step.  Empty for self-loop steps.
    """

    target: GenericCharacteristicName
    sources: tuple[GenericCharacteristicName, ...]
    edge_kind: str
    method_name: str
    options_descriptor: EdgeOptionsDescriptor = field(default_factory=EdgeOptionsDescriptor)


@dataclass(frozen=True, slots=True)
class ComputationPlan:
    """
    Plan describing how a strategy will compute ``target`` for a distribution.

    Attributes
    ----------
    target : GenericCharacteristicName
        Characteristic the plan resolves.
    source : GenericCharacteristicName
        Starting characteristic of the plan (a self-loop characteristic
        in :attr:`Distribution.analytical_computations`).
    steps : tuple[ComputationStep, ...]
        Ordered sequence of steps.  For a single-loop plan there is
        exactly one step ``source -> source``; for a conversion plan
        the first step starts at ``source`` and the last step targets
        ``target``.
    """

    target: GenericCharacteristicName
    source: GenericCharacteristicName
    steps: tuple[ComputationStep, ...]

    def required_options(self) -> tuple[str, ...]:
        """
        Return the names of all options that may be consumed by the plan.

        Useful for users that want to know which keyword arguments are
        meaningful for a particular ``query_method`` / ``calculate_characteristic``
        call.
        """
        seen: dict[str, None] = {}
        for step in self.steps:
            for opt in step.options_descriptor.options:
                seen.setdefault(opt.name, None)
        return tuple(seen)

    def required_characteristic_options(self) -> tuple[str, ...]:
        """
        Return the names of all *characteristic* options across all steps.

        These are the options that are intrinsic to the characteristic and
        should be broadcast to every step that declares them.  They also
        affect the cache key.
        """
        seen: dict[str, None] = {}
        for step in self.steps:
            for opt in step.options_descriptor.characteristic_options:
                seen.setdefault(opt.name, None)
        return tuple(seen)

    def required_computation_options(self) -> tuple[str, ...]:
        """
        Return the names of all *computation* options across all steps.

        These are fitter-specific options that control numerical algorithms.
        They do **not** affect the cache key.
        """
        seen: dict[str, None] = {}
        for step in self.steps:
            for opt in step.options_descriptor.computation_options:
                seen.setdefault(opt.name, None)
        return tuple(seen)

    def with_options(self, step_index: int, **kwargs: Any) -> dict[int, ResolvedEdgeOptions]:
        """
        Create a :data:`StepOptions` mapping with validated options for one step.

        This is the recommended way to build the ``options`` parameter for
        :meth:`ComputationStrategy.query_method`.  Call it once per step
        that needs non-default options and merge the results::

            plan = distr.explain_computation_path("ppf")
            opts = plan.with_options(0, tol=0.1) | plan.with_options(1, eps=1e-3)
            ppf = distr.query_method("ppf", options=opts)

        Option values are validated eagerly (type-cast + predicate check)
        so errors surface here rather than deep inside the strategy.

        Parameters
        ----------
        step_index : int
            0-based index into :attr:`steps`.
        **kwargs : Any
            Option values for that step.

        Returns
        -------
        dict[int, ResolvedEdgeOptions]
            A single-entry :data:`StepOptions` mapping that can be
            merged with other such mappings via ``|``.

        Raises
        ------
        IndexError
            If ``step_index`` is out of range.
        TypeError
            If a value cannot be cast to the declared type.
        ValueError
            If a value fails the option's validation predicate.
        """
        if step_index < 0 or step_index >= len(self.steps):
            raise IndexError(
                f"step_index {step_index} out of range for plan with {len(self.steps)} steps."
            )
        resolved = self.steps[step_index].options_descriptor.with_values(**kwargs)
        return {step_index: resolved}


# --------------------------------------------------------------------------- #
# Cached plan (internal — keeps actual edge / loop refs alongside ComputationPlan)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _CachedPlan:
    """
    Internal companion to :class:`ComputationPlan` that retains references
    to the actual graph primitives required for execution.

    Attributes
    ----------
    plan : ComputationPlan
        Public representation of the plan.
    loop_method : Method | None
        Loop method when the plan resolves through a single self-loop;
        ``None`` for multi-edge conversion plans.
    edges : tuple[ComputationEdgeMeta, ...]
        Conversion edges along the plan.  Empty when ``loop_method`` is set.
    """

    plan: ComputationPlan
    loop_method: Method[Any, Any] | None
    edges: tuple[ComputationEdgeMeta, ...]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_hashable(value: Any) -> Hashable:
    """
    Convert a possibly-unhashable option value into a stable hashable key.

    Lists / tuples become tuples of recursively hashable items.  Dicts
    and sets become sorted ``frozenset``s of hashable pairs / items.
    Everything that is already hashable is returned as-is.  As a last
    resort the value is keyed by its ``repr`` so two equal objects
    produce the same key while still keeping the cache safe.
    """
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, dict):
        return frozenset((k, _make_hashable(v)) for k, v in value.items())
    if isinstance(value, list | tuple):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, set | frozenset):
        return frozenset(_make_hashable(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return cast("Hashable", value)


def _freeze_options(resolved: Mapping[str, Any]) -> frozenset[tuple[str, Hashable]]:
    """Freeze a resolved-options mapping into a stable hashable key."""
    return frozenset((name, _make_hashable(val)) for name, val in resolved.items())


def _resolve_option_group(
    options: tuple[_BaseOption, ...],
    per_step_values: Mapping[str, Any],
    fallback_values: Mapping[str, Any],
    *,
    option_kind: str,
) -> dict[str, Any]:
    """Resolve one option group with per-step values taking precedence."""
    resolved: dict[str, Any] = {}
    for option in options:
        if option.name in per_step_values:
            raw = per_step_values[option.name]
        elif option.name in fallback_values:
            raw = fallback_values[option.name]
        else:
            raw = option.default
        try:
            value = option.type(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{option_kind} option '{option.name}': cannot convert "
                f"{raw!r} to {option.type.__name__}"
            ) from exc
        if option.validate is not None and not option.validate(value):
            raise ValueError(
                f"{option_kind} option '{option.name}': value {value!r} failed validation."
            )
        resolved[option.name] = value
    return resolved


def _resolve_step_options(
    edge: ComputationEdgeMeta,
    step_idx: int,
    step_options: StepOptions | None,
    characteristic_options: Mapping[str, Any],
    computation_defaults: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Resolve characteristic and computation options for a single edge/step.

    Resolution order
    ----------------
    *Characteristic options* (affect cache key, broadcast across all steps):
        1. ``step_options[step_idx].values`` for keys that are characteristic options
        2. ``characteristic_options`` shared dict
        3. Declared ``CharacteristicOption.default``

    *Computation options* (do NOT affect cache key, fitter-specific):
        1. ``step_options[step_idx].values`` for keys that are computation options
        2. ``computation_defaults`` dict
        3. Declared ``ComputationOption.default``

    Parameters
    ----------
    edge : ComputationEdgeMeta
        The edge being resolved.
    step_idx : int
        0-based step index (used to look up per-step overrides).
    step_options : StepOptions | None
        Per-step caller overrides (keyed by step index).
    characteristic_options : Mapping[str, Any]
        Shared characteristic options broadcast to every step.
    computation_defaults : Mapping[str, Any]
        Strategy/call-level computation defaults (between hardcoded and per-step).

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        ``(char_resolved, comp_resolved)`` — characteristic and computation
        option dicts respectively.
    """
    descriptor = edge.options_descriptor
    step_resolved = step_options.get(step_idx) if step_options else None
    per_step_values: dict[str, Any] = step_resolved.values if step_resolved is not None else {}

    char_resolved = _resolve_option_group(
        descriptor.characteristic_options,
        per_step_values,
        characteristic_options,
        option_kind="Characteristic",
    )
    comp_resolved = _resolve_option_group(
        descriptor.computation_options,
        per_step_values,
        computation_defaults,
        option_kind="Computation",
    )

    return char_resolved, comp_resolved


# --------------------------------------------------------------------------- #
# Strategy protocol & default implementation
# --------------------------------------------------------------------------- #


class ComputationStrategy(Protocol):
    """
    Protocol for strategies that resolve computation methods for characteristics.

    Attributes
    ----------
    enable_caching : bool
        Whether to cache fitted computation methods.
    """

    def query_method(
        self,
        state: GenericCharacteristicName,
        distr: Distribution,
        options: StepOptions | None = None,
        *,
        characteristic_options: Mapping[str, Any] | None = None,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> Method[Any, Any]: ...

    def explain_computation_path(
        self, state: GenericCharacteristicName, distr: Distribution
    ) -> ComputationPlan:
        """
        Describe how this strategy will compute ``state`` for ``distr``.

        Returns an :class:`ComputationPlan` that lists every step the
        strategy will perform along with the option descriptors it will
        consult at each step.  Implementations are expected to *fix* the
        returned plan internally so that a subsequent call to
        :meth:`query_method` for the same ``(distr, state)`` follows
        the same path even if the strategy makes non-deterministic
        choices.
        """
        ...


class DefaultComputationStrategy:
    """
    Default strategy for resolving characteristic computation methods.

    This strategy first checks for analytical implementations provided by
    the distribution. If none exists, it walks the characteristic graph
    to find a conversion path from an analytical characteristic to the
    target characteristic.

    Parameters
    ----------
    enable_caching : bool, default=False
        If True, cache fitted conversions to avoid repeated fitting.
    computation_defaults : Mapping[str, Any] | None, default=None
        Strategy-level defaults for computation options.  These sit between
        the hardcoded ``ComputationOption.default`` and any per-step caller
        override.  Resolution order (highest priority first):

        1. Per-step caller override (``options`` argument to
           :meth:`query_method`).
        2. ``computation_defaults`` supplied here.
        3. Hardcoded ``ComputationOption.default`` on the descriptor.

        Example::

            strategy = DefaultComputationStrategy(
                enable_caching=True,
                computation_defaults={"max_iter": 100, "limit": 50},
            )

    Attributes
    ----------
    _enable_caching : bool
        Whether caching is enabled.
    _computation_defaults : dict[str, Any]
        Strategy-level computation option defaults.
    _cache : dict
        Cache of fitted computation methods keyed by
        ``(distr_id, edge_id, target, frozen_all_options)`` so that
        different option sets (both characteristic and computation) produce
        independent cache entries.  Characteristic options affect the
        *meaning* of the result; computation options affect the *accuracy*
        of the fitted callable — both must be part of the key.
    _path_cache : dict
        Cache of resolved execution plans keyed by ``(distr_id, target)``.
        Lets repeated ``query_method`` calls reuse the path produced by a
        previous ``explain_computation_path`` / ``query_method`` and keeps both methods in
        sync for non-deterministic strategies.
    _resolving : dict[int, set[str]]
        Tracking of currently resolving characteristics to detect cycles.
    _char_options_stack : list[dict[str, Any]]
        Stack of characteristic-options dicts, one entry per active
        query_method call.  When a fitter on step N calls
        distribution.query_method(intermediate) recursively, the strategy
        picks up the characteristic options from the top of this stack so
        they are propagated automatically without the fitter needing to
        forward them explicitly.
    """

    def __init__(
        self,
        enable_caching: bool = False,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self._enable_caching = enable_caching
        self._computation_defaults: dict[str, Any] = dict(computation_defaults or {})
        self._cache: dict[
            tuple[int, int, GenericCharacteristicName, frozenset[tuple[str, Hashable]]],
            FittedComputationMethod[Any, Any],
        ] = {}
        self._path_cache: dict[tuple[int, GenericCharacteristicName], _CachedPlan] = {}
        self._resolving: dict[int, set[GenericCharacteristicName]] = {}
        self._char_options_stack: list[dict[str, Any]] = []

    @property
    def is_caching_enabled(self) -> bool:
        return self._enable_caching

    # ------------------------------------------------------------------ #
    # Cycle detection helpers
    # ------------------------------------------------------------------ #

    def _push_guard(self, distr: Distribution, state: GenericCharacteristicName) -> None:
        """
        Push a characteristic onto the resolution stack to detect cycles.

        Raises
        ------
        RuntimeError
            If a cycle is detected during resolution.
        """
        key = id(distr)
        seen = self._resolving.setdefault(key, set())
        if state in seen:
            raise RuntimeError(
                f"Cycle detected while resolving '{state}'. "
                "Provide at least one analytical base characteristic in the distribution."
            )
        seen.add(state)

    def _pop_guard(self, distr: Distribution, state: GenericCharacteristicName) -> None:
        """Pop a characteristic from the resolution stack."""
        key = id(distr)
        seen = self._resolving.get(key)
        if seen is not None:
            seen.discard(state)
            if not seen:
                self._resolving.pop(key, None)

    # ------------------------------------------------------------------ #
    # Method picking helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_analytical_method(
        state: GenericCharacteristicName,
        methods: Mapping[LabelName, AnalyticalComputation[Any, Any]],
    ) -> AnalyticalComputation[Any, Any]:
        """
        Pick the first available analytical method for a characteristic.

        Raises
        ------
        RuntimeError
            If no labeled analytical methods are available for the characteristic.
        """
        try:
            return next(iter(methods.values()))
        except StopIteration as exc:
            raise RuntimeError(
                f"Characteristic '{state}' provides no labeled analytical computations."
            ) from exc

    @staticmethod
    def _pick_loop_edge(
        state: GenericCharacteristicName,
        view: RegistryView,
    ) -> EdgeMeta | None:
        """
        Pick the first available self-loop edge for a characteristic in a view.
        """
        loops = view.variants(state, state)
        if not loops:
            return None
        return next(iter(loops.values()))

    @staticmethod
    def _loop_method(loop_edge: EdgeMeta) -> Method[Any, Any]:
        """Return the loop-edge underlying method as a :data:`Method`."""
        return cast(Method[Any, Any], loop_edge.method)

    # ------------------------------------------------------------------ #
    # Plan building (introspection + caching of paths)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _step_for_loop(state: GenericCharacteristicName, loop_edge: EdgeMeta) -> ComputationStep:
        return ComputationStep(
            target=state,
            sources=(state,),
            edge_kind=loop_edge.edge_kind(),
            method_name=getattr(loop_edge.method, "target", state),
            options_descriptor=EdgeOptionsDescriptor(),
        )

    @staticmethod
    def _step_for_edge(edge: ComputationEdgeMeta) -> ComputationStep:
        method = edge.method
        return ComputationStep(
            target=method.target,
            sources=tuple(method.sources),
            edge_kind=edge.edge_kind(),
            method_name=edge.options_descriptor.name or method.target,
            options_descriptor=edge.options_descriptor,
        )

    def _build_plan(self, distr: Distribution, state: GenericCharacteristicName) -> _CachedPlan:
        """
        Resolve ``state`` against the registry and build a :class:`_CachedPlan`.

        This method does **not** perform any fitting -- it only chooses the
        loop / conversion path the strategy will use.  The plan is cached
        so subsequent ``query_method`` calls for the same ``(distr, state)``
        follow exactly the same edges.
        """
        cache_key = (id(distr), state)
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return cached

        if not distr.analytical_computations:
            raise RuntimeError(
                "Distribution provides no analytical computations to ground conversions."
            )

        registry = characteristic_registry()

        # Non-registry characteristics: use the distribution-supplied analytical method.
        if state not in registry.declared_characteristics:
            if state in distr.analytical_computations:
                method = self._pick_analytical_method(state, distr.analytical_computations[state])
                step = ComputationStep(
                    target=state,
                    sources=(state,),
                    edge_kind="analytical_loop",
                    method_name=getattr(method, "target", state),
                    options_descriptor=EdgeOptionsDescriptor(),
                )
                plan = _CachedPlan(
                    plan=ComputationPlan(target=state, source=state, steps=(step,)),
                    loop_method=cast(Method[Any, Any], method),
                    edges=(),
                )
                self._path_cache[cache_key] = plan
                return plan
            raise RuntimeError(
                f"Characteristic '{state}' is not declared in the registry and has no "
                "analytical implementation in the distribution."
            )

        view = registry.view(distr)

        # Direct self-loop hit first.
        loop_edge = self._pick_loop_edge(state, view)
        if loop_edge is not None:
            step = self._step_for_loop(state, loop_edge)
            plan = _CachedPlan(
                plan=ComputationPlan(target=state, source=state, steps=(step,)),
                loop_method=self._loop_method(loop_edge),
                edges=(),
            )
            self._path_cache[cache_key] = plan
            return plan

        # Otherwise: try each loop characteristic as a source.
        for src in distr.analytical_computations:
            if not view.variants(src, src):
                continue

            path = view.find_path(src, state)
            if not path:
                continue

            steps = tuple(self._step_for_edge(edge) for edge in path)
            plan = _CachedPlan(
                plan=ComputationPlan(target=state, source=src, steps=steps),
                loop_method=None,
                edges=tuple(path),
            )
            self._path_cache[cache_key] = plan
            return plan

        raise RuntimeError(
            "No conversion path from any characteristic in "
            f"analytical_computations to '{state}'."
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def explain_computation_path(
        self, state: GenericCharacteristicName, distr: Distribution
    ) -> ComputationPlan:
        """
        Describe and pin the plan that :meth:`query_method` will follow.

        The returned :class:`ComputationPlan` lists every step (loop or
        conversion edge) and the :class:`EdgeOptionsDescriptor` consulted at
        that step.  The plan is cached per ``(distr, state)`` so that a
        subsequent :meth:`query_method` call goes through the very same
        edges -- this matters for non-deterministic strategy variants
        and is also what allows the second call to skip the BFS pass.
        """
        return self._build_plan(distr, state).plan

    def query_method(
        self,
        state: GenericCharacteristicName,
        distr: Distribution,
        options: StepOptions | None = None,
        *,
        characteristic_options: Mapping[str, Any] | None = None,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> Method[Any, Any]:
        """
        Resolve a computation method for the target characteristic.

        Resolution order:
        1. Cached fitted method (if caching enabled)
        2. Analytical implementation for non-registry characteristics
        3. First self-loop from the registry view
        4. Conversion path from loop characteristics via the graph

        Parameters
        ----------
        state : str
            Target characteristic name (e.g., "pdf", "cdf").
        distr : Distribution
            Distribution to compute the characteristic for.
        options : StepOptions | None, default=None
            Per-step options built via
            :meth:`ComputationPlan.with_options`.  Each key is a 0-based
            step index and each value is a :class:`ResolvedEdgeOptions`
            produced by :meth:`EdgeOptionsDescriptor.with_values`.
            When ``None``, every edge uses its declared defaults.
        characteristic_options : Mapping[str, Any] | None, default=None
            Shared characteristic options broadcast to **every step** that
            declares a matching :class:`CharacteristicOption`.  These are
            intrinsic to the characteristic (e.g. ``eps``, ``x0`` for PPF)
            and affect the *meaning* of the result.  Per-step overrides in
            ``options`` take precedence over this dict; the dict takes
            precedence over the hardcoded ``CharacteristicOption.default``.
        computation_defaults : Mapping[str, Any] | None, default=None
            Per-call computation option defaults.  These override the
            strategy-level ``computation_defaults`` set at construction time
            and the hardcoded ``ComputationOption.default``, but are
            overridden by per-step values in ``options``.

        Returns
        -------
        Method
            Callable that computes the characteristic.

        Raises
        ------
        RuntimeError
            If no analytical base exists, no conversion path is found,
            or a cycle is detected.
        """
        cached_plan = self._build_plan(distr, state)

        if cached_plan.loop_method is not None:
            return cached_plan.loop_method

        # Merge computation defaults: call-level overrides strategy-level.
        effective_comp_defaults: dict[str, Any] = dict(self._computation_defaults)
        if computation_defaults:
            effective_comp_defaults.update(computation_defaults)

        inherited_char_options: dict[str, Any] = dict(
            self._char_options_stack[-1] if self._char_options_stack else {}
        )
        if characteristic_options:
            inherited_char_options.update(characteristic_options)
        effective_char_options: Mapping[str, Any] = inherited_char_options

        self._push_guard(distr, state)
        self._char_options_stack.append(dict(effective_char_options))
        try:
            last_fitted: FittedComputationMethod[Any, Any] | None = None
            injected_keys: list[tuple[int, GenericCharacteristicName]] = []
            for step_idx, edge in enumerate(cached_plan.edges):
                method = edge.method

                char_resolved, comp_resolved = _resolve_step_options(
                    edge,
                    step_idx,
                    options,
                    effective_char_options,
                    effective_comp_defaults,
                )

                all_resolved = {**char_resolved, **comp_resolved}
                cache_key = (
                    id(distr),
                    id(edge),
                    method.target,
                    _freeze_options(all_resolved),
                )
                cached_fitted: FittedComputationMethod[Any, Any] | None = None
                if self._enable_caching:
                    cached_fitted = self._cache.get(cache_key)

                if cached_fitted is not None:
                    fitted = cached_fitted
                else:
                    fitted = method.prepare(distr, **all_resolved)
                    if self._enable_caching and method.cacheable:
                        self._cache[cache_key] = fitted

                last_fitted = fitted

                # Expose the fitted result as a loop plan for the intermediate
                # target so that fitters on subsequent edges can retrieve it
                # via distribution.query_method(method.target).
                intermediate_key = (id(distr), method.target)
                self._path_cache[intermediate_key] = _CachedPlan(
                    plan=ComputationPlan(
                        target=method.target,
                        source=method.target,
                        steps=(
                            ComputationStep(
                                target=method.target,
                                sources=(method.target,),
                                edge_kind="analytical_loop",
                                method_name=method.target,
                            ),
                        ),
                    ),
                    loop_method=fitted,
                    edges=(),
                )
                injected_keys.append(intermediate_key)

            # Remove the temporary loop plans injected for intermediate targets.
            for key in injected_keys:
                self._path_cache.pop(key, None)

            if last_fitted is None:
                raise RuntimeError(f"Empty path when resolving '{state}'.")
            return last_fitted
        finally:
            self._char_options_stack.pop()
            self._pop_guard(distr, state)


class SamplingStrategy(Protocol):
    """Protocol for strategies that generate samples from distributions."""

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray: ...
