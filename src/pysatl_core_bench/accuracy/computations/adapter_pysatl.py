"""Adapter for pysatl-core: the boundary of library knowledge.

The adapter owns the catalog of computation-strategy names of *its* library
version, constructs strategy objects and installs them through the public
``distr.with_computation_strategy(...)`` during prepare.  There is no
separate strategy registry (М-3): a second one would duplicate the C1
pattern without benefit.  Unknown or inapplicable strategies produce an
explicit ``unsupported_configuration`` verdict — never a silent fallback
(М-6).

Strategy catalog (v1):

- ``DefaultComputationStrategy`` — the library first-match walker;
- ``ForcedGraphStrategy`` (М-2, params ``{"from": ..., "labels": [...]?}``)
  — forces the route to start at the given characteristic.  The forced plan
  is: analytical loop at ``from`` + graph path to the target.  A start that
  is not a graph node (today: ``cf``, a source-only characteristic — the
  characteristic-graph invariants forbid one-way source nodes) is served
  through a direct fitter-registry lookup of the ``from -> target`` edge;
  the resulting plan is dumped into provenance (М-7) exactly as any other.

The library graph ships no ``cf -> pdf`` edge (a source-only characteristic
can never be a graph node), and the benchmark never modifies pysatl-core
files.  The Gil-Pelaez ``cf -> pdf`` fitter therefore lives *here* and is
installed into the library's fitter registry through its public ``register``
API, idempotently, before every distribution construction.  Only the public
pysatl-core API is used: ``ParametricFamilyRegister``, ``family.
distribution(...)``, ``distr.support``, ``distr.with_computation_strategy``,
``distr.calculate_characteristic``, ``distr.explain_computation_path``,
``fitter_registry().register``.  Private core helpers, ``_path_cache`` and
sampler internals are never imported.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

import pysatl_core
from pysatl_core.distributions.computations.computation import (
    AnalyticalComputation,
    FittedComputationMethod,
)
from pysatl_core.distributions.computations.descriptors import FitterDescriptor
from pysatl_core.distributions.computations.options import (
    CharacteristicOption,
    ComputationOption,
    ResolvedEdgeOptions,
    StepOptions,
)
from pysatl_core.distributions.computations.registry import fitter_registry
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.distributions.registry import RegistryView, characteristic_registry
from pysatl_core.distributions.registry.graph_primitives import ComputationEdgeMeta, EdgeMeta
from pysatl_core.distributions.strategies import (
    ComputationPlan,
    ComputationStep,
    DefaultComputationStrategy,
)
from pysatl_core.distributions.support import Support
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import (
    CharacteristicName,
    GenericCharacteristicName,
    Interval1D,
    LabelName,
    Method,
    NumericArray,
)
from pysatl_core_bench.accuracy.computations.protocol import PreparedSubject
from pysatl_core_bench.accuracy.dto import Provenance, SupportBounds, SupportVerdict
from pysatl_core_bench.accuracy.errors import AdapterConfigurationError, FrameworkDefect
from pysatl_core_bench.accuracy.scenarios.spec import (
    AccuracyQuery,
    DistributionSpec,
    ParticipantSpec,
    StrategySpec,
)
from pysatl_core_bench.accuracy.types import Capability, Readiness

Float64Array = NDArray[np.float64]

__all__ = ["PySATLAdapter"]

_STRATEGY_CATALOG: tuple[str, ...] = ("DefaultComputationStrategy", "ForcedGraphStrategy")
_CONTINUOUS_1D_TAGS: frozenset[str] = frozenset({"continuous", "univariate"})

_SCALAR_CHARACTERISTICS: frozenset[CharacteristicName] = frozenset(
    {
        CharacteristicName.MEAN,
        CharacteristicName.VAR,
        CharacteristicName.SKEW,
        CharacteristicName.KURT,
        CharacteristicName.MOMENT,
        CharacteristicName.CENTRAL_MOMENT,
        CharacteristicName.STANDARD_MOMENT,
    }
)
"""Characteristics that are *scalars*, not grid functions (08 §2.3).

The scenario grammar measures a characteristic over a region's point grid;
asking a moment "on a grid" is a category error, not a library crash.
Without this check such a query assessed READY (the graph can name a plan)
and then blew up in evaluate (scalar methods have no array semantics) —
an avoidable ``failed``.  Refusing here is adapter-owned library knowledge
(М-6: explicit unsupported_configuration, no silent fallback); a future
core that serves moments with array semantics extends the catalog instead.
"""

_FITTER_REGISTRATION_LOCK = threading.Lock()
"""Serializes the idempotent check-and-register of the bench fitter (А-9)."""

_FAMILIES_CONFIGURE_LOCK = threading.Lock()
"""Serializes the first core-family configuration inside one process.

``configure_families_register()`` is ``@lru_cache``-wrapped in core, but a
cache miss is *not* atomic: two threads calling it before the cache entry
exists each execute the full configuration (families re-register, the
characteristic graph gets seeded twice -- hence core's "already has a
presence rule" warnings -- and a concurrent ``get`` can observe a half-built
family register and refuse the participant).  One lock turns the race into a
well-defined "first caller configures, the rest take the cache hit".
"""

_GIL_PELAEZ_T_TRUNC: float = 40.0
_GIL_PELAEZ_N_STEPS: int = 8192


# -- bench-provided cf -> pdf edge -------------------------------------------------


def _fit_cf_to_pdf_1C(
    distribution: Distribution,
    /,
    t_trunc: float = _GIL_PELAEZ_T_TRUNC,
    n_steps: int = _GIL_PELAEZ_N_STEPS,
) -> FittedComputationMethod[NumericArray, NumericArray]:
    """Fit a ``cf -> pdf`` conversion via the Gil-Pelaez inversion integral.

    The density is recovered from the characteristic function as

    ``f(x) = (1 / pi) * Integral_0^T Re[exp(-i t x) phi(t)] dt``

    evaluated by the trapezoidal rule on ``[0, T]``.  The characteristic
    function is sampled once at fit time; evaluation is a vectorised dot
    product per point.

    Applicability: the trapezoid rule converges to machine precision for
    densities whose characteristic function decays rapidly (e. g. the
    normal family).  For slowly decaying ``phi`` (polynomial decay, e. g.
    exponential-type densities with a support-boundary jump) the integral
    converges only conditionally and the truncation error decays like
    ``1 / (x * T)`` — such densities need an oscillatory-quadrature variant
    of this edge instead of a larger ``n_steps``.

    Parameters
    ----------
    distribution : Distribution
        Must expose an array-semantic ``cf``.
    t_trunc : float
        *(Characteristic option)* Truncation bound ``T`` of the inversion
        integral.  Different values yield a different density (truncation
        error trade-off), hence the value is part of the cache key.
    n_steps : int
        *(Computation option)* Number of trapezoid intervals on ``[0, T]``.
        Controls quadrature accuracy only.

    Returns
    -------
    FittedComputationMethod[NumericArray, NumericArray]
        Array-semantic ``pdf`` callable.
    """
    cf = distribution.query_method(CharacteristicName.CF)

    t_grid = np.linspace(0.0, t_trunc, num=n_steps + 1, dtype=float)
    step = t_trunc / n_steps
    weights = np.full(n_steps + 1, step, dtype=float)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    phi = np.asarray(cf(t_grid), dtype=complex)
    phi_real = np.asarray(phi.real, dtype=float)
    phi_imag = np.asarray(phi.imag, dtype=float)

    def _pdf(*args: Any, **options: Any) -> NumericArray:
        x_arr = np.atleast_1d(np.asarray(args[0], dtype=float))
        result = np.empty(x_arr.shape, dtype=float)
        chunk_size = 256
        for start in range(0, x_arr.size, chunk_size):
            block = x_arr[start : start + chunk_size]
            phase = np.exp(-1j * np.outer(block, t_grid))
            integrand = phase.real * phi_real - phase.imag * phi_imag
            result[start : start + chunk_size] = integrand @ weights / np.pi
        return np.clip(result, 0.0, None)

    return FittedComputationMethod[NumericArray, NumericArray](
        target=CharacteristicName.PDF,
        sources=[CharacteristicName.CF],
        func=_pdf,
    )


def _build_cf_to_pdf_1C() -> FitterDescriptor:
    """Describe the bench-provided ``cf -> pdf`` edge for the fitter registry."""
    return FitterDescriptor(
        name="cf_to_pdf_1C",
        target=CharacteristicName.PDF,
        sources=[CharacteristicName.CF],
        fitter=_fit_cf_to_pdf_1C,
        characteristic_options=(
            CharacteristicOption(
                name="t_trunc",
                type=float,
                default=_GIL_PELAEZ_T_TRUNC,
                description=(
                    "Truncation bound T of the Gil-Pelaez inversion integral "
                    "over [0, T].  Different values yield a different result "
                    "near the tails — the value is part of the cache key."
                ),
                validate=lambda v: v > 0,
            ),
        ),
        computation_options=(
            ComputationOption(
                name="n_steps",
                type=int,
                default=_GIL_PELAEZ_N_STEPS,
                description="Number of trapezoid intervals on [0, T].",
                validate=lambda v: v >= 16,
            ),
        ),
        constraint_tags=frozenset({"continuous", "univariate"}),
        description=(
            "CF -> PDF via Gil-Pelaez inversion (trapezoid rule on [0, T]). "
            "Machine-precision for rapidly decaying cf (smooth densities); "
            "slowly decaying cf (support jumps) needs an oscillatory-quadrature "
            "variant.  Registered by the benchmark through the public registry "
            "API; pysatl-core files are not modified."
        ),
    )


def _ensure_cf_to_pdf_edge() -> None:
    """Register the bench ``cf -> pdf`` fitter into core's registry (idempotent).

    The characteristic-graph invariants of pysatl-core forbid one-way source
    nodes, so the library itself ships no ``cf -> pdf`` edge; the benchmark
    supplies its own Gil-Pelaez fitter through the public ``register`` API.
    ``reset_core_state()`` (called between scenarios) clears the registry,
    so registration is repeated lazily before each distribution construction.
    The check-then-register window is guarded by a lock: parallel participants
    (А-9) build distributions concurrently and must not double-register.
    """
    with _FITTER_REGISTRATION_LOCK:
        registry = fitter_registry()
        if (
            registry.find(
                CharacteristicName.PDF,
                [CharacteristicName.CF],
                required_tags=_CONTINUOUS_1D_TAGS,
            )
            is None
        ):
            registry.register(_build_cf_to_pdf_1C())


@dataclass(frozen=True)
class _ForcedPlan:
    """Internal companion of a forced plan: public plan + fit handles."""

    computation: ComputationPlan
    loop_method: Method[NumericArray, NumericArray] | None
    edges: tuple[ComputationEdgeMeta, ...]


class _ForcedGraphStrategy:
    """Computation strategy that forces the route to start at a characteristic (М-2).

    Implements the public ``ComputationStrategy`` protocol of pysatl-core.
    The forced plan is: analytical loop at ``from`` followed by the graph path
    to the requested target.  Targets equal to ``from`` resolve to the plain
    analytical loop.  A start that is not a graph node (source-only
    characteristics such as ``cf``) is served through a direct fitter-registry
    lookup of a single ``(from -> target)`` edge.

    No caching (v1): routes are deterministic lookups, and the adapter
    prepares every participant once per run (C2.3).
    """

    def __init__(
        self, start: GenericCharacteristicName, labels: tuple[LabelName, ...] = ()
    ) -> None:
        self._start = start
        self._labels = labels

    def explain_computation_path(
        self, state: GenericCharacteristicName, distr: Distribution
    ) -> ComputationPlan:
        """Describe (and thereby fix) the forced plan for ``state``."""
        return self._forced_plan(state, distr).computation

    def query_method(
        self,
        state: GenericCharacteristicName,
        distr: Distribution,
        options: StepOptions | None = None,
        *,
        characteristic_options: Mapping[str, Any] | None = None,
        computation_defaults: Mapping[str, Any] | None = None,
    ) -> Method[NumericArray, NumericArray]:
        """Fit the forced plan edges in order and return the last fitted method."""
        plan = self._forced_plan(state, distr)
        if plan.loop_method is not None:
            return plan.loop_method
        fitted: FittedComputationMethod[NumericArray, NumericArray] | None = None
        char_opts = dict(characteristic_options or {})
        comp_opts = dict(computation_defaults or {})
        for idx, edge in enumerate(plan.edges):
            resolved = self._resolve_options(edge, idx, options, char_opts, comp_opts)
            fitted = edge.method.prepare(distr, **resolved)
        if fitted is None:
            msg = f"forced plan for {str(state)!r} has no edges"
            raise AdapterConfigurationError(msg)
        return fitted

    # -- plan construction ------------------------------------------------------

    def _forced_plan(self, state: GenericCharacteristicName, distr: Distribution) -> _ForcedPlan:
        if state == self._start:
            loop = self._loop_method(distr)
            step = ComputationStep(
                target=state,
                sources=(state,),
                edge_kind="analytical_loop",
                method_name=state,
            )
            return _ForcedPlan(
                computation=ComputationPlan(target=state, source=state, steps=(step,)),
                loop_method=loop,
                edges=(),
            )
        view = characteristic_registry().view(distr)
        path = self._find_graph_path(view, state)
        if path is not None:
            loop_edge = self._loop_edge(view)
            loop_step = ComputationStep(
                target=self._start,
                sources=(self._start,),
                edge_kind=(loop_edge.edge_kind() if loop_edge is not None else "analytical_loop"),
                method_name=self._start,
            )
            steps = (loop_step, *(self._step_for_edge(edge) for edge in path))
            return _ForcedPlan(
                computation=ComputationPlan(target=state, source=self._start, steps=steps),
                loop_method=None,
                edges=tuple(path),
            )
        edge = self._direct_edge(state)
        if edge is not None:
            loop_step = ComputationStep(
                target=self._start,
                sources=(self._start,),
                edge_kind="analytical_loop",
                method_name=self._start,
            )
            return _ForcedPlan(
                computation=ComputationPlan(
                    target=state,
                    source=self._start,
                    steps=(loop_step, self._step_for_edge(edge)),
                ),
                loop_method=None,
                edges=(edge,),
            )
        msg = (
            f"ForcedGraphStrategy: no computation route from {str(self._start)!r} to {str(state)!r}"
        )
        raise AdapterConfigurationError(msg)

    def _loop_edge(self, view: RegistryView) -> EdgeMeta | None:
        loops = view.variants(self._start, self._start)
        if not loops:
            return None
        return next(iter(loops.values()))

    def _loop_method(self, distr: Distribution) -> Method[NumericArray, NumericArray]:
        view = characteristic_registry().view(distr)
        loops = view.variants(self._start, self._start)
        if loops:
            return self._loop_edge_method(next(iter(loops.values())), distr)
        labeled = distr.analytical_computations.get(self._start)
        if isinstance(labeled, Mapping):
            return next(iter(labeled.values()))
        if labeled is not None:
            return labeled
        msg = (
            f"ForcedGraphStrategy: start {str(self._start)!r} is not available "
            f"analytically on this distribution"
        )
        raise AdapterConfigurationError(msg)

    @staticmethod
    def _loop_edge_method(
        edge: EdgeMeta, distr: Distribution
    ) -> Method[NumericArray, NumericArray]:
        method = edge.method
        if isinstance(method, AnalyticalComputation):
            return method
        return method.prepare(distr)

    def _find_graph_path(
        self, view: RegistryView, state: GenericCharacteristicName
    ) -> list[ComputationEdgeMeta] | None:
        """Labels are preferences; the first matching route wins (М-2)."""
        for label in self._labels:
            path = view.find_path(self._start, state, prefer_label=label)
            if path:
                return path
        return view.find_path(self._start, state)

    def _direct_edge(self, state: GenericCharacteristicName) -> ComputationEdgeMeta | None:
        """Single-edge fallback for source-only starts (e. g. ``cf``).

        Source-only characteristics are not graph nodes: the characteristic
        graph invariants require every non-definitive node to be reachable
        from a definitive one and forbid any path from a non-definitive node
        back to a definitive one, while no computation ever returns to
        ``cf``.  The fitter registry still knows the edge (the benchmark
        registers its Gil-Pelaez fitter through the public API), so the
        forced plan can carry it and the dump stays a faithful fact (М-7).
        """
        descriptor = fitter_registry().find(state, [self._start], required_tags=_CONTINUOUS_1D_TAGS)
        if descriptor is None:
            return None
        return ComputationEdgeMeta(
            method=descriptor.to_computation_method(),
            options_descriptor=descriptor.to_options_descriptor(),
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

    @staticmethod
    def _resolve_options(
        edge: ComputationEdgeMeta,
        step_idx: int,
        options: StepOptions | None,
        characteristic_options: Mapping[str, Any],
        computation_defaults: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Resolve one edge's options: defaults < broadcast < per-step (core order)."""
        descriptor = edge.options_descriptor
        per_step: Mapping[str, Any] = {}
        if options is not None:
            resolved: ResolvedEdgeOptions | None = options.get(step_idx)
            if resolved is not None:
                per_step = resolved.values
        out: dict[str, Any] = {}
        declared_groups: tuple[tuple[CharacteristicOption, ...], tuple[ComputationOption, ...]] = (
            descriptor.characteristic_options,
            descriptor.computation_options,
        )
        broadcasts: tuple[Mapping[str, Any], ...] = (characteristic_options, computation_defaults)
        for broadcast, declared in zip(broadcasts, declared_groups, strict=True):
            for option in declared:
                if option.name in per_step:
                    raw = per_step[option.name]
                elif option.name in broadcast:
                    raw = broadcast[option.name]
                else:
                    raw = option.default
                try:
                    value = option.type(raw)
                except (TypeError, ValueError) as exc:
                    msg = f"option {option.name!r}: cannot convert {raw!r}: {exc}"
                    raise AdapterConfigurationError(msg) from exc
                if option.validate is not None and not option.validate(value):
                    msg = f"option {option.name!r}: value {value!r} failed validation"
                    raise AdapterConfigurationError(msg)
                out[option.name] = value
        return out


class PySATLAdapter:
    """Value source for the measured pysatl-core library (always compute)."""

    def __init__(self, source_id: str = "pysatl-core") -> None:
        self._source_id = source_id

    # -- protocol: static surface -------------------------------------------

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.COMPUTE})

    @property
    def subject_id(self) -> str:
        return f"pysatl-core@{pysatl_core.__version__}"

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            library="pysatl-core",
            library_version=pysatl_core.__version__,
        )

    # -- protocol: readiness --------------------------------------------------

    def support(
        self, participant: ParticipantSpec, distribution: DistributionSpec
    ) -> SupportBounds:
        distr = self._build_distribution(distribution, participant)
        support = distr.support
        if support is None:
            msg = "distribution does not expose a support"
            raise AdapterConfigurationError(msg)
        return self._bounds(support)

    def assess(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
        points: Float64Array | None = None,
    ) -> SupportVerdict:
        if query.characteristic in _SCALAR_CHARACTERISTICS:
            return SupportVerdict(
                readiness=Readiness.UNSUPPORTED_CONFIGURATION,
                reason=(
                    f"{str(query.characteristic)!r} is a scalar characteristic: it is "
                    "not measurable on a region point grid (scenario-grammar "
                    "mismatch, deferred until core serves moments with array "
                    "semantics; see the smoke taxonomy 08 §2.3)"
                ),
            )
        try:
            distr = self._build_distribution(distribution, participant)
        except AdapterConfigurationError as exc:
            return SupportVerdict(readiness=Readiness.UNSUPPORTED_CONFIGURATION, reason=str(exc))
        try:
            distr.explain_computation_path(query.characteristic)
        except AdapterConfigurationError as exc:
            # Unknown strategy, inapplicable start, route not found (М-6).
            return SupportVerdict(
                readiness=Readiness.UNSUPPORTED_CONFIGURATION,
                reason=f"strategy inapplicable: {exc}",
            )
        except FrameworkDefect:
            raise
        except Exception as exc:
            # Any other refusal means the library cannot serve the
            # characteristic at all — an outcome, not a crash (М-6).
            return SupportVerdict(
                readiness=Readiness.NOT_APPLICABLE,
                reason=(
                    f"no computation route for {str(query.characteristic)!r}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        if points is not None:
            support = distr.support
            bounds = self._bounds(support) if support is not None else None
            outside = (
                [float(x) for x in np.asarray(points).ravel() if not self._in(x, bounds)]
                if bounds is not None
                else []
            )
            if outside:
                return SupportVerdict(
                    readiness=Readiness.UNSUPPORTED_CONFIGURATION,
                    reason=(
                        f"{len(outside)} requested points lie outside the distribution support"
                    ),
                    missing=tuple(outside[:16]),
                )
        return SupportVerdict(readiness=Readiness.READY, reason="compute source")

    def native_spaces(self, participant: ParticipantSpec, query: AccuracyQuery) -> list[str]:
        # v1: every graph characteristic answers in probability space; the
        # survival space is reached only through SpaceConverter.
        return ["probability"]

    # -- protocol: two-phase evaluation ---------------------------------------

    def prepare(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        distribution: DistributionSpec,
    ) -> PreparedSubject:
        distr = self._build_distribution(distribution, participant)
        plan = distr.explain_computation_path(query.characteristic)
        return PreparedSubject(
            participant_id=participant.participant_id,
            query=query,
            distribution=distribution,
            payload=distr,
            path_dump=self._path_dump(plan),
        )

    def evaluate(self, prepared: PreparedSubject, points: Float64Array) -> Float64Array:
        distr = self._payload_distribution(prepared)
        values = distr.calculate_characteristic(
            prepared.query.characteristic, np.asarray(points, dtype=np.float64)
        )
        return np.asarray(values, dtype=np.float64)

    def compute(
        self,
        participant: ParticipantSpec,
        query: AccuracyQuery,
        points: Float64Array,
        distribution: DistributionSpec,
    ) -> Float64Array:
        distr = self._build_distribution(distribution, participant)
        values = distr.calculate_characteristic(
            query.characteristic, np.asarray(points, dtype=np.float64)
        )
        return np.asarray(values, dtype=np.float64)

    # -- construction ----------------------------------------------------------

    def _build_distribution(
        self, distribution: DistributionSpec, participant: ParticipantSpec
    ) -> Distribution:
        # ред. 40: the measured distribution is scenario-level state; the
        # participant carries only adapter knobs.  The v1 pysatl-core
        # adapter accepts no adapter_config (no silent knobs).
        if participant.adapter_config:
            msg = (
                "pysatl-core adapter accepts no adapter_config in v1; got "
                f"{dict(participant.adapter_config)!r}"
            )
            raise AdapterConfigurationError(msg)
        family_name = distribution.family
        parameters = distribution.parameters
        parametrization = distribution.parametrization
        with _FAMILIES_CONFIGURE_LOCK:
            configure_families_register()
        _ensure_cf_to_pdf_edge()
        try:
            family = ParametricFamilyRegister.get(family_name)
        except ValueError as exc:
            msg = f"family {family_name!r} is not registered in pysatl-core"
            raise AdapterConfigurationError(msg) from exc
        try:
            distr = family.distribution(
                parametrization_name=parametrization,
                **dict(parameters),
            )
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"cannot construct {family_name!r} distribution: {exc}"
            raise AdapterConfigurationError(msg) from exc
        return self._with_strategy(distr, participant)

    def _with_strategy(
        self,
        distr: Distribution,
        participant: ParticipantSpec,
    ) -> Distribution:
        strategy = participant.strategy
        if strategy is None:
            return distr
        name, params = self._strategy_parts(strategy)
        if name not in _STRATEGY_CATALOG:
            msg = (
                f"unknown computation strategy {name!r} for pysatl-core; "
                f"known names: {list(_STRATEGY_CATALOG)}"
            )
            raise AdapterConfigurationError(msg)
        if name == "ForcedGraphStrategy":
            return distr.with_computation_strategy(self._forced_graph_strategy(params))
        if params:
            msg = f"strategy {name!r} accepts no params in the v1 adapter catalog; got {params!r}"
            raise AdapterConfigurationError(msg)
        return distr.with_computation_strategy(DefaultComputationStrategy())

    @staticmethod
    def _forced_graph_strategy(params: Mapping[str, Any]) -> _ForcedGraphStrategy:
        """Build ``ForcedGraphStrategy`` from its params (М-2; validated, no fallback)."""
        unknown = set(params) - {"from", "labels"}
        if unknown:
            msg = (
                f"unknown ForcedGraphStrategy params fields: {sorted(unknown)}; "
                f"known: ['from', 'labels']"
            )
            raise AdapterConfigurationError(msg)
        start_value = params.get("from")
        if not isinstance(start_value, str) or not start_value:
            msg = (
                "strategy 'ForcedGraphStrategy' requires params {'from': "
                "<characteristic name>, 'labels'?: [names]}"
            )
            raise AdapterConfigurationError(msg)
        try:
            start = CharacteristicName(start_value)
        except ValueError as exc:
            msg = f"ForcedGraphStrategy: unknown start characteristic {start_value!r}"
            raise AdapterConfigurationError(msg) from exc
        labels_raw = params.get("labels", [])
        if not isinstance(labels_raw, list) or not all(
            isinstance(label, str) and label for label in labels_raw
        ):
            msg = "ForcedGraphStrategy params.labels must be a list of non-empty strings"
            raise AdapterConfigurationError(msg)
        return _ForcedGraphStrategy(start, tuple(labels_raw))

    @staticmethod
    def _strategy_parts(strategy: StrategySpec) -> tuple[str, Mapping[str, Any]]:
        if isinstance(strategy, str):
            return strategy, {}
        if isinstance(strategy, Mapping):
            # REV-20: canonical strategies arrive frozen (MappingProxyType).
            name = strategy.get("name")
            params = strategy.get("params", {})
            if isinstance(name, str) and isinstance(params, Mapping):
                return name, params
        msg = "strategy must be a string or {name, params}"
        raise AdapterConfigurationError(msg)

    @staticmethod
    def _bounds(support: Support) -> SupportBounds:
        if isinstance(support, Interval1D):
            return SupportBounds(
                left=float(support.left),
                right=float(support.right),
                left_closed=bool(support.left_closed),
                right_closed=bool(support.right_closed),
            )
        msg = f"unsupported support type {type(support).__name__}"
        raise AdapterConfigurationError(msg)

    @staticmethod
    def _in(x: float, bounds: SupportBounds) -> bool:
        left_ok = bounds.left < x or (bounds.left_closed and bounds.left == x)
        right_ok = x < bounds.right or (bounds.right_closed and bounds.right == x)
        return left_ok and right_ok

    @staticmethod
    def _payload_distribution(prepared: PreparedSubject) -> Distribution:
        payload = prepared.payload
        if not isinstance(payload, Distribution):
            msg = "prepared payload is not a pysatl-core Distribution"
            raise AdapterConfigurationError(msg)
        return payload

    @staticmethod
    def _path_dump(plan: ComputationPlan) -> dict[str, Any]:
        """Serialize the actual ComputationPlan (М-7: the dump is the fact)."""
        return {
            "target": str(plan.target),
            "source": str(plan.source),
            "steps": [
                {
                    "target": str(step.target),
                    "sources": [str(s) for s in step.sources],
                    "edge_kind": str(step.edge_kind),
                    "method_name": str(step.method_name),
                }
                for step in plan.steps
            ],
        }
