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

from pysatl_core.distributions.computations.base import OptionsDescriptor
from pysatl_core.distributions.registry import characteristic_registry
from pysatl_core.types import Method, NumericArray

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from pysatl_core.distributions.computations.computation import (
        AnalyticalComputation,
        FittedComputationMethod,
    )
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
class ExecutionStep:
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
    options_descriptor : OptionsDescriptor
        Compact descriptor describing which user-supplied options will
        be consumed at this step.  Empty for self-loop steps.
    """

    target: GenericCharacteristicName
    sources: tuple[GenericCharacteristicName, ...]
    edge_kind: str
    method_name: str
    options_descriptor: OptionsDescriptor = field(default_factory=OptionsDescriptor)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """
    Plan describing how a strategy will compute ``target`` for a distribution.

    Attributes
    ----------
    target : GenericCharacteristicName
        Characteristic the plan resolves.
    source : GenericCharacteristicName
        Starting characteristic of the plan (a self-loop characteristic
        in :attr:`Distribution.analytical_computations`).
    steps : tuple[ExecutionStep, ...]
        Ordered sequence of steps.  For a single-loop plan there is
        exactly one step ``source -> source``; for a conversion plan
        the first step starts at ``source`` and the last step targets
        ``target``.
    """

    target: GenericCharacteristicName
    source: GenericCharacteristicName
    steps: tuple[ExecutionStep, ...]

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


# --------------------------------------------------------------------------- #
# Cached plan (internal — keeps actual edge / loop refs alongside ExecutionPlan)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _CachedPlan:
    """
    Internal companion to :class:`ExecutionPlan` that retains references
    to the actual graph primitives required for execution.

    Attributes
    ----------
    plan : ExecutionPlan
        Public representation of the plan.
    loop_method : Method | None
        Loop method when the plan resolves through a single self-loop;
        ``None`` for multi-edge conversion plans.
    edges : tuple[ComputationEdgeMeta, ...]
        Conversion edges along the plan.  Empty when ``loop_method`` is set.
    """

    plan: ExecutionPlan
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
        self, state: GenericCharacteristicName, distr: Distribution, **options: Any
    ) -> Method[Any, Any]: ...

    def explain(self, state: GenericCharacteristicName, distr: Distribution) -> ExecutionPlan:
        """
        Describe how this strategy will compute ``state`` for ``distr``.

        Returns an :class:`ExecutionPlan` that lists every step the
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

    Attributes
    ----------
    _enable_caching : bool
        Whether caching is enabled.
    _cache : dict
        Cache of fitted computation methods keyed by
        ``(distr_id, edge_id, target, frozen_resolved_options)`` so that
        different option sets produce independent cache entries.
    _path_cache : dict
        Cache of resolved execution plans keyed by ``(distr_id, target)``.
        Lets repeated ``query_method`` calls reuse the path produced by a
        previous ``explain`` / ``query_method`` and keeps both methods in
        sync for non-deterministic strategies.
    _resolving : dict[int, set[str]]
        Tracking of currently resolving characteristics to detect cycles.
    """

    def __init__(self, enable_caching: bool = False) -> None:
        self._enable_caching = enable_caching
        self._cache: dict[
            tuple[int, int, GenericCharacteristicName, frozenset[tuple[str, Hashable]]],
            FittedComputationMethod[Any, Any],
        ] = {}
        self._path_cache: dict[tuple[int, GenericCharacteristicName], _CachedPlan] = {}
        self._resolving: dict[int, set[GenericCharacteristicName]] = {}

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
    def _step_for_loop(state: GenericCharacteristicName, loop_edge: EdgeMeta) -> ExecutionStep:
        return ExecutionStep(
            target=state,
            sources=(state,),
            edge_kind=loop_edge.edge_kind(),
            method_name=getattr(loop_edge.method, "target", state),
            options_descriptor=OptionsDescriptor(),
        )

    @staticmethod
    def _step_for_edge(edge: ComputationEdgeMeta) -> ExecutionStep:
        method = edge.method
        return ExecutionStep(
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
                step = ExecutionStep(
                    target=state,
                    sources=(state,),
                    edge_kind="analytical_loop",
                    method_name=getattr(method, "target", state),
                    options_descriptor=OptionsDescriptor(),
                )
                plan = _CachedPlan(
                    plan=ExecutionPlan(target=state, source=state, steps=(step,)),
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
                plan=ExecutionPlan(target=state, source=state, steps=(step,)),
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
                plan=ExecutionPlan(target=state, source=src, steps=steps),
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

    def explain(self, state: GenericCharacteristicName, distr: Distribution) -> ExecutionPlan:
        """
        Describe and pin the plan that :meth:`query_method` will follow.

        The returned :class:`ExecutionPlan` lists every step (loop or
        conversion edge) and the :class:`OptionsDescriptor` consulted at
        that step.  The plan is cached per ``(distr, state)`` so that a
        subsequent :meth:`query_method` call goes through the very same
        edges -- this matters for non-deterministic strategy variants
        and is also what allows the second call to skip the BFS pass.
        """
        return self._build_plan(distr, state).plan

    def query_method(
        self, state: GenericCharacteristicName, distr: Distribution, **options: Any
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
        **options : Any
            Additional options passed to fitters.

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

        # Loop-only plan -- nothing to fit, return the underlying method.
        if cached_plan.loop_method is not None:
            return cached_plan.loop_method

        # Cycle guard wraps the entire execution, so a fitter that
        # recurses into ``query_method`` for the same ``state`` is
        # detected even when the plan came from ``_path_cache``.
        self._push_guard(distr, state)
        try:
            # Conversion plan -- walk the cached edges, fitting (with
            # caching) along the way.  Per-edge options are extracted
            # from the caller-supplied ``options`` using the short
            # descriptor attached to that specific edge, so each fitter
            # receives only its own declared options *and* the cache
            # key includes those options too.
            last_fitted: FittedComputationMethod[Any, Any] | None = None
            for edge in cached_plan.edges:
                method = edge.method
                edge_kwargs = dict(options)
                resolved = edge.options_descriptor.resolve_options(edge_kwargs)

                cache_key = (id(distr), id(edge), method.target, _freeze_options(resolved))
                cached_fitted: FittedComputationMethod[Any, Any] | None = None
                if self._enable_caching:
                    cached_fitted = self._cache.get(cache_key)

                if cached_fitted is not None:
                    fitted = cached_fitted
                else:
                    fitted = method.prepare(distr, **resolved)
                    if self._enable_caching and method.cacheable:
                        self._cache[cache_key] = fitted

                last_fitted = fitted

            if last_fitted is None:
                raise RuntimeError(f"Empty path when resolving '{state}'.")
            return last_fitted
        finally:
            self._pop_guard(distr, state)


class SamplingStrategy(Protocol):
    """Protocol for strategies that generate samples from distributions."""

    def sample(self, n: int, distr: Distribution, **options: Any) -> NumericArray: ...
