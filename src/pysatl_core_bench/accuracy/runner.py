"""The measurement service: orchestration of the whole accuracy cycle.

The runner is a plain per-``run`` object (never a singleton): parallel runs
in one process work because every prepared state lives in a local,
run-scoped cache.  Call order inside ``run`` is fixed:

resolve participants -> support (a single provider, or the intersection of
the participants' supports when a lookup-only member is present — решение
по 7.2, ред. 46) -> detect special points -> generate grids
-> assess subject and references -> prepare (once per participant) -> values
in native spaces -> SpaceConverter (only if needed) -> metrics per region ->
group-wide spread of invariant verdicts (N7) -> write one result per pair
-> Reporter.

Prepare is orchestrated here too (R41-1): :meth:`run_group_raw` takes a raw
scenario dict (a JSON file payload, possibly a sweep/chain file), expands it
into N ordinary raw scenarios and loads each one immediately — the variant
list is a local buffer of the call, nothing is persisted between the
expander and the loader, and raw dicts never travel past the loader (D2,
invariant No. 1).  :meth:`run_group` receives canonical specs only.

Write scope (REV-16): the store is written once per *comparison group*,
after all scenarios of the group have run — :meth:`run_group` is the
primary entry point for a file's scenarios (sweep/chain variants share one
group), :meth:`run` is its single-scenario degenerate case.  Only after
the whole group is measured can the spread verdicts of one scenario reach
the pair rows of the others; storing earlier would leave rows in the store
that disagree with the report (А-5).

Parallelism contract (А-9).  Participants are independent by construction:
all per-participant state (prepare + evaluate) flows through the run-scoped
``_RunCache`` and no step couples two participants, so measuring several
participants concurrently (e.g. the subject via pysatl-core and a reference
via an external library) is an executor-level change, not a redesign: the
per-participant work item is ``run_cache.prepared`` / ``run_cache.values``.
Framework-side state is safe for concurrency (run-scoped cache; the store
guards itself with a lock; the adapter registers its fitter atomically).
The one process-global resource is the pysatl-core registry set:
``reset_core_state()`` must run between scenario batches, never concurrently
with a live run that resolves families/fitters.

Status rules encoded here:

- subject not ready -> *every* pair of the group receives the subject
  status, with every unready reference merged into the explanation;
- one result per pair, pair key sorted (``{A, B} == {B, A}``);
- double unavailability collapses by priority
  ``failed > unsupported_configuration > not_applicable`` while the
  explanation still lists every unready participant;
- partial reference coverage is still ``measured`` (holes are counted);
  nothing covered at all is ``invalid_reference``;
- a reference crash is ``invalid_reference`` (unusable baseline), a subject
  crash is ``failed`` (defect of the measured side);
- ``y_true`` keeps the report language: the reference values in the
  question space of the region.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pysatl_core.types import CharacteristicName
from pysatl_core_bench.accuracy.canonical import values_hash
from pysatl_core_bench.accuracy.computations.metrics import CdfFollowup, MetricsEvaluator
from pysatl_core_bench.accuracy.computations.protocol import PreparedSubject, ValueSource
from pysatl_core_bench.accuracy.computations.registry import SourceRegistry
from pysatl_core_bench.accuracy.computations.spaces import SpaceConverter
from pysatl_core_bench.accuracy.dto import (
    AccuracyResult,
    MetricRecord,
    PointwiseRef,
    PointwiseSegment,
    Provenance,
    Report,
    SpecialPoint,
    SupportBounds,
    SupportVerdict,
    ValidatorNote,
    Violation,
)
from pysatl_core_bench.accuracy.errors import (
    AdapterConfigurationError,
    FrameworkDefect,
    SourceNotRegistered,
)
from pysatl_core_bench.accuracy.reporting.compare import spread_invariant_verdicts
from pysatl_core_bench.accuracy.reporting.store import ResultsStore
from pysatl_core_bench.accuracy.scenarios.detection import SpecialPointDetector
from pysatl_core_bench.accuracy.scenarios.grid import Grid, GridGenerator
from pysatl_core_bench.accuracy.scenarios.invariants import InvariantChecker
from pysatl_core_bench.accuracy.scenarios.scenario import (
    ScenarioExpander,
    ScenarioLoader,
    ScenarioValidator,
)
from pysatl_core_bench.accuracy.scenarios.spec import (
    AccuracyQuery,
    DistributionSpec,
    PairPolicy,
    ParticipantSpec,
    ScenarioSpec,
)
from pysatl_core_bench.accuracy.types import (
    PROBABILITY_SPACE,
    SCHEMA_VERSION,
    STATUS_PRIORITY,
    Capability,
    Readiness,
    Role,
    Status,
)

Float64Array = NDArray[np.float64]

__all__ = ["AccuracyRunner"]

_SUBJECT_CHECK_NAME = "subject"


class _FollowupCrash(Exception):
    """A quantile follow-up crashed inside metric evaluation.

    Internal to the runner: carries the side role so the pair-status
    mapping follows the docstring rule (a crash of the measured side is
    ``failed``, a crash of the reference side is ``invalid_reference``)
    instead of blaming the pair wholesale.
    """

    def __init__(self, role: str, inner: Exception) -> None:
        super().__init__(f"{role} cdf follow-up failed: {type(inner).__name__}: {inner}")
        self.role = role
        self.inner = inner


class _RunCache:
    """Run-scoped cache: prepared states, values and assessments.

    Everything here lives exactly one run (C2.3/А-9); the assessment slot
    (REV-18) keeps the readiness verdicts of one participant stable across
    the pairs of a group — symmetric mode used to re-assess every side for
    every pair, and a reference was re-assessed per pair in subject mode.
    The scenario distribution (ред. 40) travels with the cache: one run
    measures exactly one distribution, so every prepare call forwards it.
    """

    def __init__(
        self, sources: Mapping[str, ValueSource | None], distribution: DistributionSpec
    ) -> None:
        self._sources = sources
        self._distribution = distribution
        self._prepared: dict[tuple[str, str], PreparedSubject] = {}
        self._values: dict[tuple[str, str, str, str], Float64Array] = {}
        self._assessments: dict[tuple[str, str], Mapping[str, SupportVerdict]] = {}

    def source(self, participant: ParticipantSpec) -> ValueSource | None:
        return self._sources.get(participant.participant_id)

    def prepared(self, participant: ParticipantSpec, query: AccuracyQuery) -> PreparedSubject:
        # The query participates in the key (REV-5): two queries sharing a
        # characteristic but differing in space/grid_hint are different
        # prepare calls.
        key = (participant.participant_id, repr(query))
        if key not in self._prepared:
            source = self._require_source(participant)
            self._prepared[key] = source.prepare(participant, query, self._distribution)
        return self._prepared[key]

    def values(
        self, participant: ParticipantSpec, query: AccuracyQuery, grid: Grid
    ) -> Float64Array:
        # The grid identity is in the key (REV-5, invariant 16 defense in
        # depth): values are grid-bound, not region-name-bound.
        key = (
            participant.participant_id,
            str(query.characteristic),
            grid.region.name,
            grid.grid_hash,
        )
        if key not in self._values:
            source = self._require_source(participant)
            prepared = self.prepared(participant, query)
            self._values[key] = source.evaluate(prepared, grid.points)
        return self._values[key]

    def prepared_for_provenance(
        self, participant: ParticipantSpec, query: AccuracyQuery
    ) -> PreparedSubject | None:
        return self._prepared.get((participant.participant_id, repr(query)))

    def assessments_key(self, participant_id: str, spec: ScenarioSpec) -> tuple[str, str]:
        """Cache key slot of one participant's readiness verdicts (REV-18)."""
        return (participant_id, repr(spec.queries))

    def store_assessments(
        self, key: tuple[str, str], verdicts: dict[str, SupportVerdict]
    ) -> Mapping[str, SupportVerdict]:
        frozen = MappingProxyType(dict(verdicts))
        self._assessments[key] = frozen
        return frozen

    def cached_assessments(self, key: tuple[str, str]) -> Mapping[str, SupportVerdict] | None:
        return self._assessments.get(key)

    def _require_source(self, participant: ParticipantSpec) -> ValueSource:
        source = self._sources.get(participant.participant_id)
        if source is None:
            msg = f"no value source registered for {participant.participant_id!r}"
            raise SourceNotRegistered(msg)
        return source


@dataclass
class _PairSides:
    subject: ParticipantSpec
    reference: ParticipantSpec
    role_labels: Mapping[str, str]


class AccuracyRunner:
    """Orchestrates one scenario run and authors every pair result."""

    def __init__(
        self,
        registry: SourceRegistry,
        store: ResultsStore,
        *,
        detector: SpecialPointDetector | None = None,
        grid_generator: GridGenerator | None = None,
        validator: ScenarioValidator | None = None,
        metrics_evaluator: MetricsEvaluator | None = None,
        converter: SpaceConverter | None = None,
        invariant_checker: InvariantChecker | None = None,
        expander: ScenarioExpander | None = None,
        loader: ScenarioLoader | None = None,
    ) -> None:
        self._registry = registry
        self._store = store
        self._detector = detector or SpecialPointDetector()
        self._grid_generator = grid_generator or GridGenerator()
        self._validator = validator or ScenarioValidator()
        self._metrics = metrics_evaluator or MetricsEvaluator()
        self._converter = converter or SpaceConverter()
        self._invariants = invariant_checker or InvariantChecker()
        # Prepare orchestration lives here since R41-1: the raw-file entry
        # (run_group_raw) owns the expand -> load pipeline, so the diagram's
        # ``AccuracyRunner --> ScenarioExpander/Loader`` edges are true and
        # the raw dict buffer never leaves the orchestrator (invariant 1:
        # raw dicts go no further than the loader).
        self._expander = expander or ScenarioExpander()
        self._loader = loader or ScenarioLoader()

    def run_group_raw(
        self,
        raw: dict[str, Any],
        *,
        before_scenario: Callable[[], None] | None = None,
    ) -> list[AccuracyResult]:
        """Expand, load and run one raw scenario file as a group.

        The Prepare pipeline of a raw dict (a JSON scenario, possibly a
        sweep/chain file) is orchestrated here, not by the caller: the raw
        dict is expanded into N ordinary raw scenarios and every variant is
        loaded immediately (nothing is persisted between the expander and
        the loader — the variant list is a local buffer of this call).
        Invariant No. 1 holds by construction: raw dicts go no further than
        the loader, everything past it is a canonical :class:`ScenarioSpec`.

        :class:`~pysatl_core_bench.accuracy.errors.ScenarioRejected` from
        expansion or load propagates to the caller (a file rejected in
        Prepare is a scenario defect, reported and skipped by the entry
        point — not a measurement outcome).
        """
        expanded = self._expander.expand(raw)
        specs = [self._loader.load(one_variant) for one_variant in expanded]
        return self.run_group(specs, before_scenario=before_scenario)

    def run_group(
        self,
        specs: Sequence[ScenarioSpec],
        *,
        before_scenario: Callable[[], None] | None = None,
    ) -> list[AccuracyResult]:
        """Execute the scenarios of one comparison group and store once.

        REV-16: the store write happens *after* the group-wide spread
        (N7), so invariant verdicts of any scenario of the group reach
        every pair row of the group before anything is persisted — the
        stored rows agree with the report by construction (А-5), not by
        an idempotent re-spread at read time.  All ``specs`` must share
        one ``comparison_group`` (a misuse raises :class:`FrameworkDefect`).

        ``before_scenario`` is an optional caller hook (e.g. resetting the
        core singletons between scenarios, as the core test-suite does);
        the runner itself never touches library state.
        """
        if not specs:
            return []
        groups = {spec.comparison_group for spec in specs}
        if len(groups) > 1:
            msg = f"run_group requires one comparison_group per call; got {sorted(groups)}"
            raise FrameworkDefect(msg)
        results: list[AccuracyResult] = []
        for spec in specs:
            if before_scenario is not None:
                before_scenario()
            results.extend(self._execute(spec))
        results = spread_invariant_verdicts(results)
        for result in results:
            self._store.write_result(result)
        self._report(specs[0].comparison_group)
        return results

    def run(self, spec: ScenarioSpec) -> list[AccuracyResult]:
        """Execute one canonical scenario end-to-end and return its results.

        Single-scenario degenerate case of :meth:`run_group` (REV-16): the
        spread is group-wide over one scenario, then the rows are written
        and the report rendered — the store still only ever sees rows that
        agree with the report (А-5).

        Participants and pairs come only from the canonical spec (В-7): the
        old optional ``participants``/``pair_policy`` arguments were a dead
        input that existed only to be rejected (А-8) and are removed.
        """
        return self.run_group([spec])

    def _execute(self, spec: ScenarioSpec) -> list[AccuracyResult]:
        """Measure one scenario without touching the store (REV-16)."""
        participants = spec.participants
        pair_policy = spec.pair_policy

        pairs = self._pairs(participants, pair_policy, spec)
        # One source of truth for the degenerate rule (М-5/А-4): the runner
        # orchestrates the call and supplies *its* pairs; the validator owns
        # the rule and the DEGENERATE_NOTE constant.
        degenerate = self._validator.notes_by_pair(
            spec, [(sides.subject, sides.reference) for sides in pairs]
        )
        sources = self._resolve(participants)
        run_cache = _RunCache(sources, spec.distribution)

        # Support rule (ред. 46, решение по 7.2): a lookup-only participant
        # answers only on its own range, so a grid built from another
        # member's support would be silently truncated by its coverage.
        # When the group contains a lookup-only participant, the grids are
        # built on the intersection of the participants' supports and the
        # deviation is written as a warning note.  Otherwise the support is
        # the subject's (subject mode) or the first declared participant's
        # (symmetric mode) — the all-compute case has no conflict.
        lookup_only = [
            p
            for p in participants
            if (source := sources.get(p.participant_id)) is not None
            and source.capabilities == frozenset({Capability.LOOKUP})
        ]
        if lookup_only:
            support, support_notes = self._intersected_support(
                sources, participants, spec.distribution, lookup_only
            )
        else:
            primary = spec.subject() or participants[0]
            bounds, note = self._support_of(
                sources.get(primary.participant_id), primary, spec.distribution
            )
            support, support_notes = bounds, ((note,) if note else ())
        env_notes: tuple[ValidatorNote, ...] = support_notes
        special_points = tuple(self._detector.detect(spec.distribution, support))
        grids = self._grid_generator.generate(
            list(spec.queries), list(spec.regions), list(special_points), support
        )

        if pair_policy == "subject":
            results = self._run_subject_mode(
                spec, run_cache, grids, degenerate, special_points, env_notes
            )
        else:
            results = self._run_symmetric_mode(
                spec, pairs, run_cache, grids, degenerate, special_points, env_notes
            )
        return results

    # -- pair construction -------------------------------------------------------

    @staticmethod
    def _pairs(
        participants: tuple[ParticipantSpec, ...],
        pair_policy: PairPolicy,
        spec: ScenarioSpec,
    ) -> list[_PairSides]:
        by_index = list(participants)
        if pair_policy == "subject":
            subject = spec.subject()
            if subject is None:
                msg = "pair_policy 'subject' requires exactly one subject participant"
                raise FrameworkDefect(msg)
            return [
                _PairSides(
                    subject=subject,
                    reference=p,
                    role_labels={
                        "subject": subject.participant_id,
                        "reference": p.participant_id,
                    },
                )
                for p in by_index
                if p.role == Role.REFERENCE and p is not subject
            ]
        if pair_policy == "symmetric":
            return [
                _PairSides(
                    subject=left,
                    reference=right,
                    role_labels={
                        "side_a": left.participant_id,
                        "side_b": right.participant_id,
                    },
                )
                for left, right in combinations(by_index, 2)
            ]
        if isinstance(pair_policy, list):
            return [
                _PairSides(
                    subject=by_index[i],
                    reference=by_index[j],
                    role_labels={
                        "side_a": by_index[i].participant_id,
                        "side_b": by_index[j].participant_id,
                    },
                )
                for i, j in pair_policy
            ]
        msg = f"unknown pair_policy {pair_policy!r}"
        raise FrameworkDefect(msg)

    @staticmethod
    def _pair_key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    # -- resolution and environment ----------------------------------------------

    def _intersected_support(
        self,
        sources: Mapping[str, ValueSource | None],
        participants: tuple[ParticipantSpec, ...],
        distribution: DistributionSpec,
        lookup_only: Sequence[ParticipantSpec],
    ) -> tuple[SupportBounds | None, tuple[ValidatorNote, ...]]:
        """Intersection of the participants' supports (ред. 46, решение по 7.2).

        A lookup-only member makes a single-provider support arbitrary:
        a table answers only on its data range, so grids built from
        another member's support would be silently truncated by the
        table's coverage — "we measured something else than we wanted"
        would be indistinguishable from an honest narrow support.  The
        honest grid lives where *every* participant answers; the deviation
        from the single-provider rule is reported as a warning note,
        never silently.  Participants whose ``support()`` fails do not
        constrain the intersection (their probe failure is reported by
        its own note); an empty intersection simply resolves every region
        to an empty grid — an honest outcome (REV-2).
        """
        probed: list[tuple[ParticipantSpec, SupportBounds]] = []
        notes: list[ValidatorNote] = []
        for participant in participants:
            source = sources.get(participant.participant_id)
            if source is None:
                continue
            try:
                probed.append((participant, source.support(participant, distribution)))
            except FrameworkDefect:
                raise
            except Exception as exc:
                notes.append(
                    ValidatorNote(
                        code="support_probe_failed",
                        detail=(
                            f"support() of {participant.participant_id} failed: "
                            f"{type(exc).__name__}: {exc}; the participant does not "
                            "constrain the support intersection"
                        ),
                    )
                )
        if not probed:
            return None, tuple(notes)
        left = max(b.left for _, b in probed)
        right = min(b.right for _, b in probed)
        left_closed = next(b.left_closed for _, b in probed if b.left == left)
        right_closed = next(b.right_closed for _, b in probed if b.right == right)
        lookup_ids = ", ".join(sorted(p.participant_id for p in lookup_only))
        notes.append(
            ValidatorNote(
                code="support_intersection",
                detail=(
                    f"lookup-only participant(s) [{lookup_ids}] present; grids are built "
                    f"on the intersection of the participants' supports [{left!r}, {right!r}] "
                    f"({len(probed)} of {len(participants)} participants probed)"
                ),
            )
        )
        support = SupportBounds(
            left=left, right=right, left_closed=left_closed, right_closed=right_closed
        )
        return support, tuple(notes)

    def _resolve(self, participants: tuple[ParticipantSpec, ...]) -> dict[str, ValueSource | None]:
        resolved: dict[str, ValueSource | None] = {}
        for participant in participants:
            if participant.participant_id in resolved:
                continue
            try:
                resolved[participant.participant_id] = self._registry.resolve(participant.source_id)
            except SourceNotRegistered:
                resolved[participant.participant_id] = None
        return resolved

    @staticmethod
    def _support_of(
        source: ValueSource | None,
        participant: ParticipantSpec,
        distribution: DistributionSpec,
    ) -> tuple[SupportBounds | None, ValidatorNote | None]:
        """Ask the source for its support; the runner never computes it.

        The support is *reported* by the source itself (ValueSource
        protocol) for the scenario distribution (ред. 40).  A source that
        fails to answer is not swallowed silently: the failure comes back
        as a validator note and reaches the report, while the grid falls
        back to the documented heuristic.
        """
        if source is None:
            return None, None
        try:
            return source.support(participant, distribution), None
        except FrameworkDefect:
            raise
        except Exception as exc:
            note = ValidatorNote(
                code="support_probe_failed",
                detail=(
                    f"support() of {participant.participant_id} failed: "
                    f"{type(exc).__name__}: {exc}; the grid was built with "
                    f"the fallback heuristic"
                ),
            )
            return None, note

    # -- subject mode ----------------------------------------------------------------

    def _run_subject_mode(
        self,
        spec: ScenarioSpec,
        run_cache: _RunCache,
        grids: list[Grid],
        degenerate: Mapping[tuple[str, str], list[ValidatorNote]],
        special_points: tuple[SpecialPoint, ...],
        env_notes: Sequence[ValidatorNote] = (),
    ) -> list[AccuracyResult]:
        subject = spec.subject()
        if subject is None:  # guarded by run(); for the type checker
            msg = "subject mode without a subject participant"
            raise FrameworkDefect(msg)
        subject_verdicts = self._assess_cached(run_cache, subject, spec, grids)
        subject_ready = all(v.readiness == Readiness.READY for v in subject_verdicts.values())

        references = [p for p in spec.participants if p.role == Role.REFERENCE and p is not subject]
        if not subject_ready:
            return self._subject_unready_results(
                spec,
                subject,
                references,
                subject_verdicts,
                run_cache,
                grids,
                degenerate,
                special_points,
                env_notes=env_notes,
            )

        # Prepare and evaluate the subject once; reused by every pair (C2.3).
        try:
            for query in spec.queries:
                run_cache.prepared(subject, query)
                for grid in self._grids_for(grids, query):
                    run_cache.values(subject, query, grid)
        except FrameworkDefect:
            raise
        except Exception as exc:
            # Subject crash -> failed for every pair (В-3a); the explanation
            # still lists every unready reference (К-4).
            crashed = SupportVerdict(
                readiness=Readiness.UNAVAILABLE,
                reason=(f"subject failed during prepare/evaluate: {type(exc).__name__}: {exc}"),
            )
            subject_verdicts = {str(q.characteristic): crashed for q in spec.queries}
            return self._subject_unready_results(
                spec,
                subject,
                references,
                subject_verdicts,
                run_cache,
                grids,
                degenerate,
                special_points,
                subject_status=Status.FAILED,
                env_notes=env_notes,
            )

        results: list[AccuracyResult] = []
        for ref in references:
            results.append(
                self._measure_pair(
                    spec,
                    sides=_PairSides(
                        subject=subject,
                        reference=ref,
                        role_labels={
                            "subject": subject.participant_id,
                            "reference": ref.participant_id,
                        },
                    ),
                    run_cache=run_cache,
                    grids=grids,
                    degenerate=degenerate,
                    special_points=special_points,
                    env_notes=env_notes,
                )
            )
        return results

    def _subject_unready_results(
        self,
        spec: ScenarioSpec,
        subject: ParticipantSpec,
        references: list[ParticipantSpec],
        subject_verdicts: Mapping[str, SupportVerdict],
        run_cache: _RunCache,
        grids: list[Grid],
        degenerate: Mapping[tuple[str, str], list[ValidatorNote]],
        special_points: tuple[SpecialPoint, ...],
        subject_status: Status | None = None,
        env_notes: Sequence[ValidatorNote] = (),
    ) -> list[AccuracyResult]:
        # К-4: the base status belongs to the subject and is computed once;
        # every pair collapses it with *its own* unready reference, so the
        # status never leaks from one pair into the next.
        base_status = subject_status or self._subject_status(subject_verdicts)
        results: list[AccuracyResult] = []
        for ref in references:
            ref_verdicts = self._assess_cached(run_cache, ref, spec, grids)
            status, explanation = self._merge_unready(subject_verdicts, base_status, ref_verdicts)
            results.append(
                self._unready_result(
                    spec,
                    _PairSides(
                        subject=subject,
                        reference=ref,
                        role_labels={
                            "subject": subject.participant_id,
                            "reference": ref.participant_id,
                        },
                    ),
                    status,
                    explanation,
                    degenerate,
                    run_cache,
                    spec.queries[0] if spec.queries else None,
                    special_points=special_points,
                    env_notes=env_notes,
                )
            )
        return results

    # -- symmetric / explicit mode -------------------------------------------------

    def _run_symmetric_mode(
        self,
        spec: ScenarioSpec,
        pairs: list[_PairSides],
        run_cache: _RunCache,
        grids: list[Grid],
        degenerate: Mapping[tuple[str, str], list[ValidatorNote]],
        special_points: tuple[SpecialPoint, ...],
        env_notes: Sequence[ValidatorNote] = (),
    ) -> list[AccuracyResult]:
        results: list[AccuracyResult] = []
        for sides in pairs:
            left_verdicts = self._assess_cached(run_cache, sides.subject, spec, grids)
            right_verdicts = self._assess_cached(run_cache, sides.reference, spec, grids)
            left_ready = all(v.readiness == Readiness.READY for v in left_verdicts.values())
            right_ready = all(v.readiness == Readiness.READY for v in right_verdicts.values())
            if not left_ready or not right_ready:
                status, explanation = self._merge_symmetric_unready(
                    left_verdicts, right_verdicts, left_ready, right_ready
                )
                results.append(
                    self._unready_result(
                        spec,
                        sides,
                        status,
                        explanation,
                        degenerate,
                        run_cache,
                        spec.queries[0] if spec.queries else None,
                        special_points=special_points,
                        env_notes=env_notes,
                    )
                )
                continue
            results.append(
                self._measure_pair(
                    spec,
                    sides=sides,
                    run_cache=run_cache,
                    grids=grids,
                    degenerate=degenerate,
                    special_points=special_points,
                    env_notes=env_notes,
                )
            )
        return results

    # -- the pair measurement ----------------------------------------------------------

    def _measure_pair(
        self,
        spec: ScenarioSpec,
        *,
        sides: _PairSides,
        run_cache: _RunCache,
        grids: list[Grid],
        degenerate: Mapping[tuple[str, str], list[ValidatorNote]],
        special_points: tuple[SpecialPoint, ...],
        env_notes: Sequence[ValidatorNote] = (),
    ) -> AccuracyResult:
        key = AccuracyRunner._pair_key(sides.subject.participant_id, sides.reference.participant_id)
        notes = list(degenerate.get(key, ()))

        ref_verdicts = self._assess_cached(run_cache, sides.reference, spec, grids)
        not_ready = {
            char: verdict
            for char, verdict in ref_verdicts.items()
            if verdict.readiness != Readiness.READY
        }
        if not_ready:
            status = self._reference_status(not_ready)
            explanation = "reference side not ready: " + "; ".join(
                f"{char}: {v.readiness.value} ({v.reason})" for char, v in sorted(not_ready.items())
            )
            return self._unready_result(
                spec,
                sides,
                status,
                explanation,
                degenerate,
                run_cache,
                spec.queries[0] if spec.queries else None,
                special_points=special_points,
            )

        metrics: list[MetricRecord] = []
        subject_facts: list[dict[str, str]] = []
        reference_facts: list[dict[str, str]] = []
        y_true: list[float] = []
        subject_pointwise: list[float] = []
        reference_pointwise: list[float] = []
        pointwise_segments: list[PointwiseSegment] = []
        space_computed: str | None = None
        space_reported: str | None = None
        ref_provenance_prepared: PreparedSubject | None = None
        subj_provenance_prepared: PreparedSubject | None = None
        probe_notes_acc: list[ValidatorNote] = []

        for query in spec.queries:
            subject_native, subject_probe = self._native_spaces(
                run_cache.source(sides.subject), sides.subject, query
            )
            reference_native, reference_probe = self._native_spaces(
                run_cache.source(sides.reference), sides.reference, query
            )
            probe_notes = [n for n in (subject_probe, reference_probe) if n is not None]
            # REV-5 (А-6/no silence): notes of *every* query reach the
            # result; the per-query list used to be overwritten.
            probe_notes_acc.extend(probe_notes)
            if space_computed is None and subject_native:
                space_computed = subject_native[0]
            try:
                ref_prepared = run_cache.prepared(sides.reference, query)
                if ref_provenance_prepared is None:
                    ref_provenance_prepared = ref_prepared
                if subj_provenance_prepared is None:
                    subj_provenance_prepared = run_cache.prepared_for_provenance(
                        sides.subject, query
                    )
            except FrameworkDefect:
                raise
            except Exception as exc:
                return self._unready_result(
                    spec,
                    sides,
                    Status.INVALID_REFERENCE,
                    (
                        f"reference {sides.reference.participant_id} failed during "
                        f"prepare: {type(exc).__name__}: {exc}"
                    ),
                    degenerate,
                    run_cache,
                    query,
                    special_points=special_points,
                    env_notes=(*env_notes, *probe_notes_acc),
                )
            for grid in self._grids_for(grids, query):
                if grid.points.size == 0:
                    # REV-2/REV-5: a region that resolves to nothing against
                    # the runtime support is an outcome, not a crash; it is
                    # reported and simply carries no metric record.
                    continue
                try:
                    s_values = run_cache.values(sides.subject, query, grid)
                except FrameworkDefect:
                    raise
                except Exception as exc:
                    # The measured side crashed -> failed (docstring rule:
                    # a reference crash is invalid_reference, a subject
                    # crash is failed).  Both calls used to share one
                    # except-block, blaming the reference even when the
                    # subject crashed (А-2).
                    return self._unready_result(
                        spec,
                        sides,
                        Status.FAILED,
                        (
                            f"subject {sides.subject.participant_id} failed during "
                            f"evaluate: {type(exc).__name__}: {exc}"
                        ),
                        degenerate,
                        run_cache,
                        query,
                        special_points=special_points,
                        env_notes=(*env_notes, *probe_notes_acc),
                    )
                try:
                    r_values = run_cache.values(sides.reference, query, grid)
                except FrameworkDefect:
                    raise
                except Exception as exc:
                    return self._unready_result(
                        spec,
                        sides,
                        Status.INVALID_REFERENCE,
                        (
                            f"reference {sides.reference.participant_id} failed during "
                            f"evaluate: {type(exc).__name__}: {exc}"
                        ),
                        degenerate,
                        run_cache,
                        query,
                        special_points=special_points,
                        env_notes=(*env_notes, *probe_notes_acc),
                    )
                if probe_notes and grid.region.space != PROBABILITY_SPACE:
                    # REV-5 (D8): a failed native_spaces() probe means the
                    # conversion decision is made on missing data; for a
                    # non-probability question space that is not allowed to
                    # pass silently as "no conversion needed".
                    return self._unready_result(
                        spec,
                        sides,
                        Status.UNSUPPORTED_CONFIGURATION,
                        (
                            "the native_spaces() probe failed for this pair; the "
                            f"space conversion required by region {grid.region.name!r} "
                            f"({grid.region.space}) cannot be decided safely"
                        ),
                        degenerate,
                        run_cache,
                        query,
                        special_points=special_points,
                        env_notes=(*env_notes, *probe_notes_acc),
                    )
                s_conv, s_fact = self._to_question_space(
                    s_values, subject_native, grid.region.space
                )
                r_conv, r_fact = self._to_question_space(
                    r_values, reference_native, grid.region.space
                )
                if s_fact:
                    subject_facts.append(s_fact)
                if r_fact:
                    reference_facts.append(r_fact)
                if space_reported is None:
                    space_reported = grid.region.space

                missing = self._missing_mask(
                    grid.points, ref_verdicts[str(query.characteristic)], r_conv
                )
                subject_followup, reference_followup, followup_notes = self._ppf_followups(
                    spec,
                    sides,
                    run_cache,
                    query,
                    subject_native,
                    subject_probe,
                    reference_native,
                    reference_probe,
                    subject_facts,
                    reference_facts,
                )
                probe_notes_acc.extend(n for n in followup_notes if n not in probe_notes_acc)
                try:
                    record = self._metrics.evaluate(
                        query=query,
                        grid=grid,
                        subject_values=s_conv,
                        reference_values=r_conv,
                        missing_reference=missing,
                        subject_cdf_followup=subject_followup,
                        reference_cdf_followup=reference_followup,
                    )
                except FrameworkDefect:
                    raise
                except _FollowupCrash as exc:
                    # REV-5/ред. 46: the follow-up is an evaluate-stage call
                    # of one side; its crash maps by the role rule (subject
                    # -> failed, reference -> invalid_reference), not by the
                    # blanket pair failure.
                    status = Status.FAILED if exc.role == "subject" else Status.INVALID_REFERENCE
                    return self._unready_result(
                        spec,
                        sides,
                        status,
                        (
                            f"{exc.role} cdf follow-up failed during metric evaluation: "
                            f"{type(exc.inner).__name__}: {exc.inner}"
                        ),
                        degenerate,
                        run_cache,
                        query,
                        special_points=special_points,
                        env_notes=(*env_notes, *probe_notes_acc),
                    )
                except Exception as exc:
                    # REV-5: a crash inside metric evaluation is a pair
                    # outcome (failed), never the fall of the whole run.
                    return self._unready_result(
                        spec,
                        sides,
                        Status.FAILED,
                        (f"metric evaluation failed for the pair: {type(exc).__name__}: {exc}"),
                        degenerate,
                        run_cache,
                        query,
                        special_points=special_points,
                        env_notes=(*env_notes, *probe_notes_acc),
                    )
                metrics.append(record)
                start = len(y_true)
                y_true.extend(float(x) for x in r_conv)
                subject_pointwise.extend(float(x) for x in s_conv)
                reference_pointwise.extend(float(x) for x in r_conv)
                # D1 joinability (А-11): record which slice of the flat
                # sidecar belongs to this (characteristic, region) block.
                pointwise_segments.append(
                    PointwiseSegment(
                        characteristic=str(query.characteristic),
                        region=grid.region.name,
                        space=grid.region.space,
                        start=start,
                        stop=len(y_true),
                    )
                )

        measured = any(m.n_compared > 0 for m in metrics)
        if measured:
            status = Status.MEASURED
            explanation = (
                f"metrics computed for {sum(m.n_compared for m in metrics)} points; "
                f"subject native space {space_computed!r}, question space {space_reported!r}"
            )
        elif not metrics:
            # REV-2/REV-5: every region resolved to an empty grid against the
            # runtime support — a scenario-vs-support mismatch, an honest
            # outcome, not an invalid reference.
            status = Status.NOT_APPLICABLE
            explanation = (
                "all regions resolved to empty grids against the runtime "
                "support; no points were measured"
            )
        else:
            status = Status.INVALID_REFERENCE
            explanation = "no point of the declared grid produced comparable values"

        violations, invariant_notes = self._maybe_check_invariants(spec, sides, run_cache, grids)
        if violations:
            status = Status.INVALID_REFERENCE
            explanation += "; invariant check disproved the participant values"

        # REV-5 (М-7): in symmetric/explicit mode the subject is prepared
        # lazily inside the grid loop; its prepared state exists by now.
        if subj_provenance_prepared is None and spec.queries:
            subj_provenance_prepared = run_cache.prepared_for_provenance(
                sides.subject, spec.queries[0]
            )

        pointwise_subject = self._sidecar(
            subject_pointwise, space_reported, tuple(pointwise_segments)
        )
        pointwise_reference = self._sidecar(
            reference_pointwise, space_reported, tuple(pointwise_segments)
        )

        return AccuracyResult(
            schema_version=SCHEMA_VERSION,
            scenario_id=spec.scenario_id,
            spec_hash=spec.spec_hash,
            comparison_group_id=spec.comparison_group,
            sort_key=spec.sort_key,
            pair=key,
            subject_id=sides.subject.participant_id,
            reference=sides.reference.participant_id,
            role_labels=dict(sides.role_labels),
            status=status,
            status_explanation=explanation,
            validator_notes=(*notes, *env_notes, *probe_notes_acc, *invariant_notes),
            invariant_violations=tuple(violations),
            metrics=tuple(metrics),
            space_computed=space_computed,
            space_reported=space_reported,
            y_true=tuple(y_true) if y_true else None,
            pointwise_subject=pointwise_subject,
            pointwise_reference=pointwise_reference,
            reference_provenance=self._provenance_of(
                run_cache.source(sides.reference),
                sides.reference,
                ref_provenance_prepared,
                tuple(reference_facts),
            ),
            subject_provenance=self._provenance_of(
                run_cache.source(sides.subject),
                sides.subject,
                subj_provenance_prepared,
                tuple(subject_facts),
            ),
            reference_hash=self._hash_of(y_true),
            special_points=special_points,
        )

    # -- readiness plumbing ----------------------------------------------------------------

    def _assess_cached(
        self,
        run_cache: _RunCache,
        participant: ParticipantSpec,
        spec: ScenarioSpec,
        grids: list[Grid],
    ) -> Mapping[str, SupportVerdict]:
        """Readiness verdicts of one participant, cached per run (REV-18).

        The verdict depends only on the run's own grids and the source
        configuration, so a group with k participants measured k(k-1)/2
        pairs assesses every participant once instead of once per pair.
        The cached mapping is frozen (REV-20) — callers only read it.
        """
        key = run_cache.assessments_key(participant.participant_id, spec)
        cached = run_cache.cached_assessments(key)
        if cached is None:
            cached = run_cache.store_assessments(
                key,
                self._assess_queries(run_cache.source(participant), participant, spec, grids),
            )
        return cached

    def _assess_queries(
        self,
        source: ValueSource | None,
        participant: ParticipantSpec,
        spec: ScenarioSpec,
        grids: list[Grid],
    ) -> dict[str, SupportVerdict]:
        verdicts: dict[str, SupportVerdict] = {}
        for query in spec.queries:
            name = str(query.characteristic)
            if source is None:
                verdicts[name] = SupportVerdict(
                    readiness=Readiness.UNAVAILABLE,
                    reason=f"no value source registered under {participant.source_id!r}",
                )
                continue
            if not self._capability_ok(source, participant):
                verdicts[name] = SupportVerdict(
                    readiness=Readiness.UNSUPPORTED_CONFIGURATION,
                    reason=(
                        f"source {participant.source_id!r} capabilities "
                        f"{sorted(c.value for c in source.capabilities)} do not "
                        f"allow the {participant.role.value} role here"
                    ),
                )
                continue
            points = self._concat_points(grids, query)
            try:
                verdicts[name] = source.assess(participant, query, spec.distribution, points)
            except FrameworkDefect:
                raise
            except AdapterConfigurationError as exc:
                verdicts[name] = SupportVerdict(
                    readiness=Readiness.UNSUPPORTED_CONFIGURATION,
                    reason=f"assessment refused: {exc}",
                )
            except Exception as exc:
                # REV-5: SourceOperationError is a *crash* of the source, not
                # a configuration verdict — it maps to UNAVAILABLE, and the
                # per-role readiness map turns it into the pair status
                # (reference -> invalid_reference; subject ->
                # unsupported_configuration, the same column as an
                # unregistered source — at assess time the runner cannot yet
                # distinguish a broken source from an unservable
                # configuration, and the crash detail stays in the reason).
                verdicts[name] = SupportVerdict(
                    readiness=Readiness.UNAVAILABLE,
                    reason=f"assessment crashed: {type(exc).__name__}: {exc}",
                )
        return verdicts

    @staticmethod
    def _concat_points(grids: list[Grid], query: AccuracyQuery) -> Float64Array | None:
        points: Float64Array | None = None
        for grid in grids:
            if grid.query != query:
                continue
            points = grid.points.copy() if points is None else np.concatenate([points, grid.points])
        return points

    @staticmethod
    def _capability_ok(source: ValueSource, participant: ParticipantSpec) -> bool:
        if participant.role == Role.SUBJECT:
            return Capability.COMPUTE in source.capabilities
        return bool(source.capabilities) and source.capabilities <= {
            Capability.COMPUTE,
            Capability.LOOKUP,
        }

    @staticmethod
    def _subject_status(verdicts: Mapping[str, SupportVerdict]) -> Status:
        """Collapse the subject's verdicts by the contract priority (К-4).

        Single code point of the priority: ``STATUS_PRIORITY`` from types
        (А-7); FAILED is covered by the priority list, there is no separate
        branch for it (А-13: the old branch was unreachable).
        """
        mapped = [AccuracyRunner._subject_readiness_map(v.readiness) for v in verdicts.values()]
        in_priority = [m for m in mapped if m in STATUS_PRIORITY]
        if in_priority:
            return min(in_priority, key=STATUS_PRIORITY.index)
        return Status.NOT_APPLICABLE

    @staticmethod
    def _subject_readiness_map(readiness: Readiness) -> Status:
        if readiness == Readiness.READY:
            return Status.MEASURED
        if readiness == Readiness.NOT_APPLICABLE:
            return Status.NOT_APPLICABLE
        if readiness == Readiness.UNSUPPORTED_CONFIGURATION:
            return Status.UNSUPPORTED_CONFIGURATION
        if readiness == Readiness.UNAVAILABLE:
            # e.g. source_id not registered: the run configuration cannot
            # be served at all -> configuration status, not an execution crash
            return Status.UNSUPPORTED_CONFIGURATION
        return Status.FAILED  # NOT_COVERED on the subject side

    @staticmethod
    def _reference_status(not_ready: Mapping[str, SupportVerdict]) -> Status:
        mapped: list[Status] = []
        for verdict in not_ready.values():
            if verdict.readiness == Readiness.UNSUPPORTED_CONFIGURATION:
                mapped.append(Status.UNSUPPORTED_CONFIGURATION)
            elif verdict.readiness in (Readiness.NOT_COVERED, Readiness.UNAVAILABLE):
                mapped.append(Status.INVALID_REFERENCE)
            else:
                mapped.append(Status.NOT_APPLICABLE)
        in_priority = [m for m in mapped if m in STATUS_PRIORITY]
        if in_priority:
            return min(in_priority, key=STATUS_PRIORITY.index)
        return Status.INVALID_REFERENCE

    @staticmethod
    def _merge_unready(
        subject_verdicts: Mapping[str, SupportVerdict],
        subject_status: Status,
        ref_verdicts: Mapping[str, SupportVerdict],
    ) -> tuple[Status, str]:
        parts = [f"subject: {v.readiness.value} ({v.reason})" for v in subject_verdicts.values()]
        parts.extend(f"reference: {v.readiness.value} ({v.reason})" for v in ref_verdicts.values())
        candidates = [subject_status]
        if any(v.readiness != Readiness.READY for v in ref_verdicts.values()):
            candidates.append(
                AccuracyRunner._reference_status(
                    {k: v for k, v in ref_verdicts.items() if v.readiness != Readiness.READY}
                )
            )
        in_priority = [c for c in candidates if c in STATUS_PRIORITY]
        status = min(in_priority, key=STATUS_PRIORITY.index) if in_priority else candidates[0]
        return status, "; ".join(parts)

    @staticmethod
    def _merge_symmetric_unready(
        left_verdicts: Mapping[str, SupportVerdict],
        right_verdicts: Mapping[str, SupportVerdict],
        left_ready: bool,
        right_ready: bool,
    ) -> tuple[Status, str]:
        parts: list[str] = []
        mapped: list[Status] = []
        if not left_ready:
            parts.extend(
                f"side_a: {v.readiness.value} ({v.reason})" for v in left_verdicts.values()
            )
            mapped.extend(
                AccuracyRunner._subject_readiness_map(v.readiness) for v in left_verdicts.values()
            )
        if not right_ready:
            parts.extend(
                f"side_b: {v.readiness.value} ({v.reason})" for v in right_verdicts.values()
            )
            mapped.extend(
                AccuracyRunner._subject_readiness_map(v.readiness) for v in right_verdicts.values()
            )
        in_priority = [m for m in mapped if m in STATUS_PRIORITY]
        status = (
            min(in_priority, key=STATUS_PRIORITY.index) if in_priority else Status.INVALID_REFERENCE
        )
        return status, "; ".join(parts)

    def _unready_result(
        self,
        spec: ScenarioSpec,
        sides: _PairSides,
        status: Status,
        explanation: str,
        degenerate: Mapping[tuple[str, str], list[ValidatorNote]],
        run_cache: _RunCache,
        query: AccuracyQuery | None,
        special_points: tuple[SpecialPoint, ...] = (),
        env_notes: Sequence[ValidatorNote] = (),
    ) -> AccuracyResult:
        key = AccuracyRunner._pair_key(sides.subject.participant_id, sides.reference.participant_id)
        first_query = query or (spec.queries[0] if spec.queries else None)
        return AccuracyResult(
            schema_version=SCHEMA_VERSION,
            scenario_id=spec.scenario_id,
            spec_hash=spec.spec_hash,
            comparison_group_id=spec.comparison_group,
            sort_key=spec.sort_key,
            pair=key,
            subject_id=sides.subject.participant_id,
            reference=sides.reference.participant_id,
            role_labels=dict(sides.role_labels),
            status=status,
            status_explanation=explanation,
            validator_notes=(*degenerate.get(key, ()), *env_notes),
            special_points=special_points,
            subject_provenance=self._provenance_of(
                run_cache.source(sides.subject),
                sides.subject,
                run_cache.prepared_for_provenance(sides.subject, first_query)
                if first_query
                else None,
            ),
            reference_provenance=self._provenance_of(
                run_cache.source(sides.reference),
                sides.reference,
                run_cache.prepared_for_provenance(sides.reference, first_query)
                if first_query
                else None,
            ),
        )

    # -- values, spaces, metrics ------------------------------------------------------------

    @staticmethod
    def _grids_for(grids: list[Grid], query: AccuracyQuery) -> list[Grid]:
        return [g for g in grids if g.query == query]

    def _to_question_space(
        self, values: Float64Array, native_spaces: list[str], question_space: str
    ) -> tuple[Float64Array, dict[str, str] | None]:
        if question_space in native_spaces or not native_spaces:
            return values, None
        converted = self._converter.convert(values, native_spaces[0], question_space)
        return converted, self._converter.conversion_fact(native_spaces[0], question_space)

    @staticmethod
    def _native_spaces(
        source: ValueSource | None, participant: ParticipantSpec, query: AccuracyQuery
    ) -> tuple[list[str], ValidatorNote | None]:
        """Ask the source for its native spaces; failures reach the report.

        An empty answer used to be indistinguishable from a crashed answer,
        which could hide a missing space conversion (D8).  A crash now
        returns an explicit validator note instead of silence.
        """
        if source is None:
            return [], None
        try:
            return list(source.native_spaces(participant, query)), None
        except FrameworkDefect:
            raise
        except Exception as exc:
            note = ValidatorNote(
                code="native_spaces_probe_failed",
                detail=(
                    f"native_spaces() of {participant.participant_id} failed: "
                    f"{type(exc).__name__}: {exc}; the space conversion decision "
                    f"was made without this information"
                ),
            )
            return [], note

    @staticmethod
    def _missing_mask(
        points: Float64Array,
        verdict: SupportVerdict,
        reference_values: Float64Array,
    ) -> NDArray[np.bool_]:
        mask = ~np.isfinite(np.asarray(reference_values, dtype=np.float64))
        if verdict.missing:
            declared = np.asarray(verdict.missing, dtype=np.float64)
            for point in declared:
                mask |= np.asarray(points) == point
        return mask

    def _ppf_followups(
        self,
        spec: ScenarioSpec,
        sides: _PairSides,
        run_cache: _RunCache,
        query: AccuracyQuery,
        subject_native: list[str],
        subject_probe: ValidatorNote | None,
        reference_native: list[str],
        reference_probe: ValidatorNote | None,
        subject_facts: list[dict[str, str]],
        reference_facts: list[dict[str, str]],
    ) -> tuple[CdfFollowup | None, CdfFollowup | None, list[ValidatorNote]]:
        """CDF follow-ups of the two sides for the ppf metrics (ред. 46).

        ``e2_self_consistency`` plugs the found quantiles into the subject
        CDF (a diagnostic, not accuracy); ``e2_true`` plugs them into the
        reference CDF (accuracy, issue #124 §3.5).  Preconditions are
        checked explicitly and their violation is reported, never silent:

        - the side's ``native_spaces()`` probe must have succeeded —
          otherwise the follow-up values cannot be converted to
          probability safely and the number is not computed (D8 rule);
        - ``e2_true`` additionally requires a compute-capable reference:
          a lookup-only table has no CDF beyond its stored points, and
          found quantiles are arbitrary floats.

        Returns the two closures (``None`` = the metric/diagnostic is not
        computed for this query) and the notes explaining every skip.
        """
        notes: list[ValidatorNote] = []
        if query.characteristic != CharacteristicName.PPF:
            return None, None, notes
        subject_followup: CdfFollowup | None = None
        reference_followup: CdfFollowup | None = None
        if subject_probe is None:
            subject_followup = self._cdf_followup(
                sides.subject,
                run_cache,
                spec.distribution,
                subject_native,
                sink=subject_facts,
                role="subject",
            )
        else:
            notes.append(
                ValidatorNote(
                    code="e2_self_consistency_skipped",
                    detail=(
                        f"native_spaces() probe of {sides.subject.participant_id} failed; "
                        "the subject CDF follow-up is not computed because its values "
                        "could not be converted to probability safely"
                    ),
                )
            )
        ref_source = run_cache.source(sides.reference)
        if ref_source is None:
            # Unreachable by construction (the reference was assessed and
            # evaluated by now); kept for the type checker only.
            msg = "reference source vanished during measurement"
            raise FrameworkDefect(msg)
        if Capability.COMPUTE not in ref_source.capabilities:
            notes.append(
                ValidatorNote(
                    code="e2_true_skipped",
                    detail=(
                        f"reference {sides.reference.participant_id} is lookup-only; "
                        "e2_true requires the reference CDF via compute, which a "
                        "precomputed table cannot serve at the found quantiles"
                    ),
                )
            )
        elif reference_probe is None:
            reference_followup = self._cdf_followup(
                sides.reference,
                run_cache,
                spec.distribution,
                reference_native,
                sink=reference_facts,
                role="reference",
            )
        else:
            notes.append(
                ValidatorNote(
                    code="e2_true_skipped",
                    detail=(
                        f"native_spaces() probe of {sides.reference.participant_id} failed; "
                        "e2_true is not computed because the follow-up values could not "
                        "be converted to probability safely"
                    ),
                )
            )
        return subject_followup, reference_followup, notes

    def _cdf_followup(
        self,
        participant: ParticipantSpec,
        run_cache: _RunCache,
        distribution: DistributionSpec,
        native_spaces: list[str],
        *,
        sink: list[dict[str, str]],
        role: str,
    ) -> CdfFollowup | None:
        """Closure computing one side's CDF at the found quantiles (ред. 46).

        One-shot protocol path (quantile follow-ups): ``compute`` with a
        probability-space CDF query.  The values arrive in the side's
        native space (N2: no space parameter) and are converted to
        probability here — the same last-resort ``SpaceConverter`` rule as
        for the grid values (D8); the conversion fact is recorded in the
        side's provenance facts.  Crashes are wrapped into
        ``_FollowupCrash`` so the runner maps them by role instead of
        blaming the pair wholesale; ``FrameworkDefect`` still propagates.
        """
        source = run_cache.source(participant)
        if source is None:
            return None
        cdf_query = AccuracyQuery(characteristic=CharacteristicName.CDF, space=PROBABILITY_SPACE)

        def followup(quantiles: Float64Array) -> Float64Array:
            try:
                raw = source.compute(
                    participant,
                    cdf_query,
                    np.asarray(quantiles, dtype=np.float64),
                    distribution,
                )
                values = np.asarray(raw, dtype=np.float64).ravel()
                if native_spaces and native_spaces[0] != PROBABILITY_SPACE:
                    fact = self._converter.conversion_fact(native_spaces[0], PROBABILITY_SPACE)
                    if fact not in sink:
                        sink.append(fact)
                    return self._converter.convert(values, native_spaces[0], PROBABILITY_SPACE)
                return values
            except FrameworkDefect:
                raise
            except Exception as exc:
                raise _FollowupCrash(role, exc) from exc

        return followup

    def _maybe_check_invariants(
        self,
        spec: ScenarioSpec,
        sides: _PairSides,
        run_cache: _RunCache,
        grids: list[Grid],
    ) -> tuple[list[Violation], list[ValidatorNote]]:
        """Run the requested invariant checks; skips are reported, not silent."""
        if not spec.invariant_check_on:
            return [], []
        violations: list[Violation] = []
        skip_notes: list[ValidatorNote] = []
        for participant in self._invariant_targets(spec, sides):
            source = run_cache.source(participant)
            if source is None:
                skip_notes.append(
                    ValidatorNote(
                        code="invariant_check_skipped",
                        detail=(
                            f"no value source registered for {participant.participant_id!r}; "
                            "the requested invariant check did not run"
                        ),
                    )
                )
                continue
            for query in spec.queries:
                for grid in self._grids_for(grids, query):
                    try:
                        found = run_cache.values(participant, query, grid)
                    except FrameworkDefect:
                        raise
                    except Exception as exc:
                        # REV-5: a skipped check is visible in the report.
                        skip_notes.append(
                            ValidatorNote(
                                code="invariant_check_skipped",
                                detail=(
                                    f"values for {query.characteristic}/{grid.region.name} "
                                    f"of {participant.participant_id} could not be obtained "
                                    f"({type(exc).__name__}: {exc}); the requested "
                                    "invariant check did not run"
                                ),
                            )
                        )
                        continue
                    for violation in self._invariants.check(query, found):
                        violations.append(
                            Violation(
                                participant_id=participant.participant_id,
                                code=violation.code,
                                detail=(
                                    f"{query.characteristic}/{grid.region.name}: {violation.detail}"
                                ),
                            )
                        )
        return violations, skip_notes

    @staticmethod
    def _invariant_targets(spec: ScenarioSpec, sides: _PairSides) -> list[ParticipantSpec]:
        targets: list[ParticipantSpec] = []
        for name in spec.invariant_check_on:
            if name == _SUBJECT_CHECK_NAME or name in (
                sides.subject.source_id,
                sides.subject.participant_id,
            ):
                targets.append(sides.subject)
            elif name in (sides.reference.source_id, sides.reference.participant_id):
                targets.append(sides.reference)
        return targets

    def _sidecar(
        self,
        values: list[float],
        space: str | None,
        segments: tuple[PointwiseSegment, ...] = (),
    ) -> PointwiseRef | None:
        if not values:
            return None
        uri, digest = self._store.put_sidecar(
            np.asarray(values, dtype=np.float64), space or "probability"
        )
        return PointwiseRef(
            uri=uri,
            hash=digest,
            space=space or "probability",
            n_points=len(values),
            segments=segments,
        )

    @staticmethod
    def _hash_of(values: list[float]) -> str | None:
        if not values:
            return None
        return values_hash(values)

    def _provenance_of(
        self,
        source: ValueSource | None,
        participant: ParticipantSpec,
        prepared: PreparedSubject | None,
        conversion_facts: tuple[Mapping[str, str], ...] = (),
    ) -> Provenance | None:
        if source is None:
            return None
        return replace(
            source.provenance,
            strategy=participant.strategy,
            path_dump=dict(prepared.path_dump) if prepared and prepared.path_dump else None,
            conversion_facts=conversion_facts,
        )

    def _report(self, group_id: str) -> Report:
        from pysatl_core_bench.accuracy.reporting.report import Reporter

        return Reporter(self._store).report(group_id)
