"""Scenario pipeline: expand -> load -> validate.

The runner calls these steps (orchestrator model); they never call each
other.  Raw scenario data is a plain ``dict`` and goes no further than the
loader: past it, everything is a canonical :class:`ScenarioSpec` with a
``spec_hash``.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from pysatl_core.types import CharacteristicName
from pysatl_core_bench.accuracy.canonical import content_hash
from pysatl_core_bench.accuracy.dto import ValidatorNote
from pysatl_core_bench.accuracy.errors import ScenarioRejected
from pysatl_core_bench.accuracy.scenarios.spec import (
    AccuracyQuery,
    DistributionSpec,
    Interval,
    PairPolicy,
    ParticipantSpec,
    Region,
    ScenarioSpec,
    StrategySpec,
)
from pysatl_core_bench.accuracy.types import (
    KNOWN_SPACES,
    PROBABILITY_SPACE,
    SCHEMA_VERSION,
    Role,
)

_ALLOWED_TOP_KEYS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "comparison_group",
        "distribution",
        "participants",
        "subject",
        "references",
        "pair_policy",
        "queries",
        "regions",
        "invariant_check_on",
        "sweep_spec",
        "chain_spec",
        "sort_key",
    }
)

DEGENERATE_NOTE = "degenerate_comparison"


class ScenarioExpander:
    """Unfolds sweep/chain scenario files into ordinary scenarios (К-1/К-5).

    The result model never learns the words "sweep" or "chain": expansion
    produces plain scenarios sharing a ``comparison_group`` and carrying a
    ``sort_key``; the comparator later builds curves/slices from that key.
    The sweep axis may address any nested field — including the
    scenario-level ``distribution`` (e.g. ``distribution.parameters.mu``)
    and a participant ``strategy`` (М-8); a chain stage may carry a
    per-stage ``distribution`` override (ред. 40).
    """

    def expand(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand one raw file into a list of raw scenario dicts."""
        self._require_dict(raw, "scenario file")
        sweep = raw.get("sweep_spec")
        chain = raw.get("chain_spec")
        if sweep is not None and chain is not None:
            msg = "sweep_spec and chain_spec are mutually exclusive"
            raise ScenarioRejected(msg)
        if sweep is not None:
            return self._expand_sweep(raw, sweep)
        if chain is not None:
            return self._expand_chain(raw, chain)
        return [dict(raw)]

    def _expand_sweep(self, raw: dict[str, Any], sweep: Any) -> list[dict[str, Any]]:
        if not isinstance(sweep, Mapping):
            msg = "sweep_spec must be an object {param, values}"
            raise ScenarioRejected(msg)
        unknown = set(sweep) - {"param", "values"}
        if unknown:
            msg = f"unknown sweep_spec fields: {sorted(unknown)}"
            raise ScenarioRejected(msg)
        param = sweep.get("param")
        values = sweep.get("values")
        if not isinstance(param, str) or not param:
            msg = "sweep_spec.param must be a non-empty dotted path"
            raise ScenarioRejected(msg)
        if not isinstance(values, list) or not values:
            msg = "sweep_spec.values must be a non-empty list"
            raise ScenarioRejected(msg)
        group = self._group(raw)
        out: list[dict[str, Any]] = []
        for index, value in enumerate(values):
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                msg = (
                    f"sweep_spec.values[{index}] is not a scalar; a sweep axis "
                    "must carry scalars (REV-3: a silent sort_key=None is forbidden)"
                )
                raise ScenarioRejected(msg)
            scenario = copy.deepcopy(raw)
            self._assign(scenario, param.split("."), value, param)
            scenario.pop("sweep_spec", None)
            scenario.pop("chain_spec", None)
            scenario["sort_key"] = value
            scenario["scenario_id"] = f"{group}/{index}"
            out.append(scenario)
        return out

    def _expand_chain(self, raw: dict[str, Any], chain: Any) -> list[dict[str, Any]]:
        if not isinstance(chain, Mapping) or not isinstance(chain.get("stages"), list):
            msg = "chain_spec must be an object {stages: [...]}"
            raise ScenarioRejected(msg)
        unknown = set(chain) - {"stages"}
        if unknown:
            msg = f"unknown chain_spec fields: {sorted(unknown)}"
            raise ScenarioRejected(msg)
        group = self._group(raw)
        out: list[dict[str, Any]] = []
        for index, stage in enumerate(chain["stages"]):
            if not isinstance(stage, Mapping):
                msg = f"chain stage {index} must be an object"
                raise ScenarioRejected(msg)
            unknown = set(stage) - {"id", "participant", "reference", "distribution"}
            if unknown:
                msg = f"unknown chain stage fields: {sorted(unknown)}"
                raise ScenarioRejected(msg)
            for required in ("participant", "reference"):
                if stage.get(required) is None:
                    msg = f"chain stage {index} is missing {required!r}"
                    raise ScenarioRejected(msg)
                if not isinstance(stage[required], Mapping):
                    msg = f"chain stage {index}.{required} must be an object"
                    raise ScenarioRejected(msg)
            scenario = copy.deepcopy(raw)
            scenario["participants"] = [
                {**copy.deepcopy(stage["participant"]), "role": Role.SUBJECT.value},
                {**copy.deepcopy(stage["reference"]), "role": Role.REFERENCE.value},
            ]
            if stage.get("distribution") is not None:
                # ред. 40: the distribution is scenario-level, so a chain
                # that walks over distributions overrides it per stage (the
                # stage object must be a full {family, parametrization?,
                # parameters?} declaration; its shape is validated by the
                # loader, not here).
                if not isinstance(stage["distribution"], Mapping):
                    msg = f"chain stage {index}.distribution must be an object"
                    raise ScenarioRejected(msg)
                scenario["distribution"] = copy.deepcopy(dict(stage["distribution"]))
            scenario.pop("subject", None)
            scenario.pop("references", None)
            scenario.pop("sweep_spec", None)
            scenario.pop("chain_spec", None)
            stage_id = stage.get("id", str(index))
            scenario["scenario_id"] = f"{group}/{stage_id}"
            scenario["sort_key"] = float(index)
            out.append(scenario)
        return out

    @staticmethod
    def _group(raw: dict[str, Any]) -> str:
        """The group of a sweep/chain file; falls back to ``scenario_id``."""
        group = raw.get("comparison_group")
        if group is None:
            group = raw.get("scenario_id")
        if not isinstance(group, str) or not group:
            msg = (
                "comparison_group must be a non-empty string (or inject the "
                "file name before expansion, see run_accuracy.py)"
            )
            raise ScenarioRejected(msg)
        return group

    def _assign(self, container: Any, path: list[str], value: Any, full_path: str) -> None:
        if not path:
            return
        key = self._segment(path[0], container, full_path)
        if len(path) == 1:
            container[key] = value
            return
        nxt = container[key]
        self._assign(nxt, path[1:], value, full_path)

    def _segment(self, raw_key: str, container: Any, full_path: str) -> str | int:
        if isinstance(container, dict):
            if raw_key not in container:
                msg = f"sweep path {full_path!r}: unknown key {raw_key!r}"
                raise ScenarioRejected(msg)
            return raw_key
        if isinstance(container, list):
            try:
                index = int(raw_key)
            except ValueError as exc:
                msg = f"sweep path {full_path!r}: {raw_key!r} is not an index"
                raise ScenarioRejected(msg) from exc
            if not 0 <= index < len(container):
                msg = f"sweep path {full_path!r}: index {index} out of range"
                raise ScenarioRejected(msg)
            return index
        msg = f"sweep path {full_path!r} traverses a scalar"
        raise ScenarioRejected(msg)

    @staticmethod
    def _require_dict(raw: Any, what: str) -> None:
        if not isinstance(raw, dict):
            msg = f"{what} must be a JSON object"
            raise ScenarioRejected(msg)


class ScenarioLoader:
    """Loads a raw dict into the canonical, hashed :class:`ScenarioSpec` (D2)."""

    def load(self, raw: dict[str, Any]) -> ScenarioSpec:
        """Read, normalize, canonicalize and hash one ordinary scenario.

        Mandatory fields are kept to the irreducible minimum (ред. 35):
        the distribution, the participants and the queries (ред. 40: the
        distribution is scenario-level — one declaration serves every
        participant).  Everything else has a documented default that the
        loader materializes (D2):

        - ``schema_version`` — defaults to the current ``SCHEMA_VERSION``;
          a *present* but wrong value is still ``Rejected``;
        - ``comparison_group`` — defaults to ``scenario_id``; if neither is
          given, the file runner (``run_accuracy.py``) injects the file
          stem as the group before expansion.  Loading a nameless dict
          programmatically is ``Rejected`` — an unnamed result row is a
          defect, not an outcome.
        """
        ScenarioExpander._require_dict(raw, "scenario")
        unknown = set(raw) - _ALLOWED_TOP_KEYS
        if unknown:
            msg = f"unknown scenario fields: {sorted(unknown)}"
            raise ScenarioRejected(msg)
        schema_version = raw.get("schema_version", SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            msg = f"schema_version must be {SCHEMA_VERSION!r}, got {schema_version!r}"
            raise ScenarioRejected(msg)
        for pre_spec in ("sweep_spec", "chain_spec"):
            if raw.get(pre_spec) is not None:
                msg = f"{pre_spec} must be expanded before load (see ScenarioExpander)"
                raise ScenarioRejected(msg)

        sort_key = self._sort_key(raw.get("sort_key"))
        scenario_id = self._scenario_id(raw, None, sort_key)
        group = self._group_or_default(raw, scenario_id)
        distribution = self._distribution(raw.get("distribution"))
        participants = self._participants(raw)
        pair_policy = self._pair_policy(raw.get("pair_policy"), participants)
        queries = self._queries(raw.get("queries"))
        regions = self._regions(raw.get("regions"))
        self._cross_check(queries, regions)
        invariants = self._invariants(raw.get("invariant_check_on"), participants)
        if scenario_id is None:
            # group came from the field or the file-name injection; the
            # canonical row identity collapses to the group.
            scenario_id = group

        canonical = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario_id,
            "comparison_group": group,
            "distribution": distribution.identity(),
            "pair_policy": self._policy_canonical(pair_policy),
            "participants": [
                {
                    "role": p.role.value,
                    "source_id": p.source_id,
                    "strategy": p.strategy,
                    "adapter_config": dict(p.adapter_config),
                }
                for p in participants
            ],
            "queries": [
                {
                    "characteristic": str(q.characteristic),
                    "space": q.space,
                    "grid_hint": dict(q.grid_hint) if q.grid_hint else None,
                }
                for q in queries
            ],
            "regions": [
                {
                    "name": r.name,
                    "interval": list(r.interval),
                    "space": r.space,
                }
                for r in regions
            ],
            "invariant_check_on": list(invariants),
            "sort_key": sort_key,
        }
        # The canonical form is a loader-internal intermediate: its hash
        # travels on the spec, the dict itself does not leak into the
        # public type.
        return ScenarioSpec(
            scenario_id=scenario_id,
            schema_version=SCHEMA_VERSION,
            comparison_group=group,
            distribution=distribution,
            pair_policy=pair_policy,
            participants=tuple(participants),
            regions=tuple(regions),
            queries=tuple(queries),
            spec_hash=content_hash(canonical),
            invariant_check_on=invariants,
            sort_key=sort_key,
        )

    # -- participants -------------------------------------------------------

    def _participants(self, raw: Mapping[str, Any]) -> list[ParticipantSpec]:
        has_list = "participants" in raw and raw["participants"] is not None
        has_synonyms = "subject" in raw or "references" in raw
        if has_list and has_synonyms:
            msg = "use either 'participants' or 'subject'/'references', not both"
            raise ScenarioRejected(msg)
        entries: list[tuple[Role, Mapping[str, Any]]] = []
        if has_list:
            if not isinstance(raw["participants"], list) or not raw["participants"]:
                msg = "participants must be a non-empty list"
                raise ScenarioRejected(msg)
            for entry in raw["participants"]:
                self._require_mapping(entry, "participant")
                role_value = entry.get("role")
                if role_value is None:
                    msg = "participant.role is required in the participants list"
                    raise ScenarioRejected(msg)
                entries.append((self._role(role_value), entry))
        else:
            if "subject" in raw:
                self._require_mapping(raw["subject"], "subject")
                entries.append((Role.SUBJECT, raw["subject"]))
            refs = raw.get("references")
            if refs is not None:
                if not isinstance(refs, list) or not refs:
                    msg = "references must be a non-empty list"
                    raise ScenarioRejected(msg)
                for entry in refs:
                    self._require_mapping(entry, "reference")
                    entries.append((Role.REFERENCE, entry))
        if not entries:
            msg = "scenario needs at least one participant"
            raise ScenarioRejected(msg)

        participants: list[ParticipantSpec] = []
        for role, entry in entries:
            source_id = entry.get("source_id")
            if not isinstance(source_id, str) or not source_id:
                msg = "participant.source_id must be a non-empty string"
                raise ScenarioRejected(msg)
            unknown = set(entry) - {"role", "source_id", "strategy", "adapter_config"}
            if unknown:
                msg = f"unknown participant fields: {sorted(unknown)}"
                raise ScenarioRejected(msg)
            adapter_config = entry.get("adapter_config", {})
            self._require_mapping(adapter_config, "participant.adapter_config")
            strategy = self._strategy(entry.get("strategy"))
            participant = ParticipantSpec(
                role=role,
                source_id=source_id,
                participant_id=self._participant_id(source_id, adapter_config, strategy),
                adapter_config=self._canonical_adapter_config(adapter_config),
                strategy=strategy,
            )
            participants.append(participant)
        # An explicit null and an absent key must behave the same (А-12):
        # both mean the default policy, so the role check must not be
        # skipped for null.
        policy_raw = raw.get("pair_policy", "subject")
        self._check_policy_roles("subject" if policy_raw is None else policy_raw, participants)
        return participants

    @staticmethod
    def _participant_id(
        source_id: str, adapter_config: Mapping[str, Any], strategy: StrategySpec | None
    ) -> str:
        digest = content_hash(
            {"source_id": source_id, "adapter_config": adapter_config, "strategy": strategy}
        )
        return f"{source_id}#{digest.removeprefix('sha256:')[:12]}"

    @staticmethod
    def _canonical_adapter_config(adapter_config: Mapping[str, Any]) -> dict[str, Any]:
        """Validate JSON-safety and fix the number representation."""
        try:
            canonical: dict[str, Any] = _json_safe(dict(adapter_config))
        except ValueError as exc:
            msg = f"participant.adapter_config is not JSON-safe: {exc}"
            raise ScenarioRejected(msg) from exc
        return canonical

    # -- distribution ---------------------------------------------------------

    @staticmethod
    def _distribution(value: Any) -> DistributionSpec:
        """The mandatory scenario-level distribution (ред. 40)."""
        if not isinstance(value, Mapping):
            msg = (
                "distribution is required: declare {family, parametrization?, "
                "parameters?} of the measured distribution at the scenario level"
            )
            raise ScenarioRejected(msg)
        unknown = set(value) - {"family", "parametrization", "parameters"}
        if unknown:
            msg = f"unknown distribution fields: {sorted(unknown)}"
            raise ScenarioRejected(msg)
        family = value.get("family")
        if not isinstance(family, str) or not family:
            msg = "distribution.family must be a non-empty string"
            raise ScenarioRejected(msg)
        parametrization = value.get("parametrization")
        if parametrization is not None and not isinstance(parametrization, str):
            msg = "distribution.parametrization must be a string or null"
            raise ScenarioRejected(msg)
        parameters = value.get("parameters", {})
        if not isinstance(parameters, Mapping):
            msg = "distribution.parameters must be an object"
            raise ScenarioRejected(msg)
        try:
            parameters = _json_safe(dict(parameters))
        except ValueError as exc:
            msg = f"distribution.parameters is not JSON-safe: {exc}"
            raise ScenarioRejected(msg) from exc
        return DistributionSpec(
            family=family, parametrization=parametrization, parameters=parameters
        )

    @staticmethod
    def _role(value: Any) -> Role:
        try:
            return Role(value)
        except ValueError as exc:
            msg = f"unknown participant role: {value!r}"
            raise ScenarioRejected(msg) from exc

    @staticmethod
    def _strategy(value: Any) -> StrategySpec | None:
        if value is None:
            return None
        if isinstance(value, str):
            if not value:
                msg = "strategy name must be non-empty"
                raise ScenarioRejected(msg)
            return value
        if isinstance(value, Mapping):
            unknown = set(value) - {"name", "params"}
            if unknown:
                msg = f"unknown strategy fields: {sorted(unknown)}"
                raise ScenarioRejected(msg)
            name = value.get("name")
            if not isinstance(name, str) or not name:
                msg = "strategy.name must be a non-empty string"
                raise ScenarioRejected(msg)
            params = value.get("params", {})
            if not isinstance(params, dict):
                msg = "strategy.params must be an object"
                raise ScenarioRejected(msg)
            try:
                params = _json_safe(dict(params))
            except ValueError as exc:
                msg = f"strategy.params is not JSON-safe: {exc}"
                raise ScenarioRejected(msg) from exc
            return {"name": name, "params": params}
        msg = "strategy must be a string or {name, params}"
        raise ScenarioRejected(msg)

    # -- pair policy ---------------------------------------------------------

    def _pair_policy(self, value: Any, participants: list[ParticipantSpec]) -> PairPolicy:
        policy = value if value is not None else "subject"
        if isinstance(policy, str):
            if policy not in ("subject", "symmetric"):
                msg = "pair_policy must be 'subject', 'symmetric' or a pair list"
                raise ScenarioRejected(msg)
            return policy
        if not isinstance(policy, list) or not policy:
            msg = "pair_policy list must be a non-empty list of [i, j] pairs"
            raise ScenarioRejected(msg)
        normalized: list[list[int]] = []
        for pair in policy:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(i, int) and not isinstance(i, bool) for i in pair)
            ):
                msg = "pair_policy entries must be [index, index] integer pairs"
                raise ScenarioRejected(msg)
            if pair[0] == pair[1] or not all(0 <= i < len(participants) for i in pair):
                msg = f"pair_policy pair {pair} is invalid for given participants"
                raise ScenarioRejected(msg)
            # REV-3 (invariant 3): one result per pair — normalize direction,
            # then reject duplicates instead of emitting two results under
            # the same sorted pair key.
            normalized.append(sorted(pair))
        seen: set[tuple[int, int]] = set()
        for pair in normalized:
            key = (pair[0], pair[1])
            if key in seen:
                msg = f"pair_policy contains a duplicate pair {pair}"
                raise ScenarioRejected(msg)
            seen.add(key)
        normalized.sort()
        return normalized

    @staticmethod
    def _check_policy_roles(value: Any, participants: list[ParticipantSpec]) -> None:
        if value == "subject":
            subjects = [p for p in participants if p.role == Role.SUBJECT]
            if len(subjects) != 1 or len(participants) < 2:
                msg = "pair_policy 'subject' needs exactly one subject and a counterpart"
                raise ScenarioRejected(msg)
            peers = [p for p in participants if p.role == Role.PEER]
            if peers:
                msg = (
                    "pair_policy 'subject' does not measure peer-role participants "
                    f"({len(peers)} declared); use pair_policy 'symmetric' or an "
                    "explicit pair list"
                )
                raise ScenarioRejected(msg)
            # REV-3 (invariant 3): two references with the same canonical
            # identity produce two results under one pair key.
            ref_ids = [p.participant_id for p in participants if p.role == Role.REFERENCE]
            if len(ref_ids) != len(set(ref_ids)):
                msg = (
                    "duplicate reference identity: the same participant would be "
                    "measured twice under one pair key"
                )
                raise ScenarioRejected(msg)
        elif value == "symmetric" and len(participants) < 2:
            msg = "pair_policy 'symmetric' needs at least two participants"
            raise ScenarioRejected(msg)

    @staticmethod
    def _policy_canonical(policy: PairPolicy) -> str | list[list[int]]:
        if isinstance(policy, str):
            return policy
        # _pair_policy already normalizes (sorted pairs, deduplicated,
        # ascending); equivalent policies hash identically (REV-3).
        return [list(pair) for pair in policy]

    # -- queries / regions ----------------------------------------------------

    def _queries(self, value: Any) -> list[AccuracyQuery]:
        if not isinstance(value, list) or not value:
            msg = "queries must be a non-empty list"
            raise ScenarioRejected(msg)
        queries: list[AccuracyQuery] = []
        for entry in value:
            self._require_mapping(entry, "query")
            unknown = set(entry) - {"characteristic", "space", "grid_hint"}
            if unknown:
                msg = f"unknown query fields: {sorted(unknown)}"
                raise ScenarioRejected(msg)
            name = entry.get("characteristic")
            try:
                characteristic = CharacteristicName(name)
            except ValueError as exc:
                msg = f"unknown characteristic: {name!r}"
                raise ScenarioRejected(msg) from exc
            space = entry.get("space", "probability")
            if space not in KNOWN_SPACES:
                msg = f"unknown query space: {space!r}"
                raise ScenarioRejected(msg)
            hint = entry.get("grid_hint")
            if hint is not None:
                self._require_mapping(hint, "query.grid_hint")
                unknown = set(hint) - {"n_points"}
                if unknown:
                    msg = f"unknown grid_hint fields: {sorted(unknown)}"
                    raise ScenarioRejected(msg)
                n = hint.get("n_points")
                if not isinstance(n, int) or isinstance(n, bool) or n < 4:
                    msg = "grid_hint.n_points must be an integer >= 4"
                    raise ScenarioRejected(msg)
                hint = dict(hint)
            queries.append(
                AccuracyQuery(characteristic=characteristic, space=space, grid_hint=hint)
            )
        # REV-3: duplicate (characteristic, space) queries would produce two
        # identical MetricRecords without a distinguishing key.
        seen: set[tuple[str, str]] = set()
        for query in queries:
            key = (str(query.characteristic), query.space)
            if key in seen:
                msg = f"duplicate query {key[0]!r} in space {key[1]!r}"
                raise ScenarioRejected(msg)
            seen.add(key)
        return queries

    def _regions(self, value: Any) -> list[Region]:
        if value is None:
            value = [{"name": "bulk", "interval": [None, None], "space": "probability"}]
        if not isinstance(value, list) or not value:
            msg = "regions must be a non-empty list"
            raise ScenarioRejected(msg)
        regions: list[Region] = []
        names: set[str] = set()
        for entry in value:
            self._require_mapping(entry, "region")
            unknown = set(entry) - {"name", "interval", "space"}
            if unknown:
                msg = f"unknown region fields: {sorted(unknown)}"
                raise ScenarioRejected(msg)
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                msg = "region.name must be a non-empty string"
                raise ScenarioRejected(msg)
            if name in names:
                msg = f"duplicate region name: {name!r}"
                raise ScenarioRejected(msg)
            names.add(name)
            interval = self._interval(entry.get("interval"))
            space = entry.get("space", "probability")
            if space not in KNOWN_SPACES:
                msg = f"unknown region space: {space!r}"
                raise ScenarioRejected(msg)
            regions.append(Region(name=name, interval=interval, space=space))
        return regions

    @staticmethod
    def _interval(value: Any) -> Interval:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            msg = "region.interval must be a two-element [left, right] list"
            raise ScenarioRejected(msg)
        bounds: list[float | None] = []
        for bound in value:
            if bound is None:
                bounds.append(None)
                continue
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                msg = "region.interval bounds must be numbers or null"
                raise ScenarioRejected(msg)
            try:
                number = float(bound)
            except OverflowError as exc:
                msg = f"region.interval bound {bound!r} is too large for a float"
                raise ScenarioRejected(msg) from exc
            if number != number or number in (float("inf"), float("-inf")):
                msg = "region.interval bounds must be finite numbers or null"
                raise ScenarioRejected(msg)
            bounds.append(number)
        lo, hi = bounds
        if lo is not None and hi is not None and not lo < hi:
            msg = "region.interval must satisfy left < right"
            raise ScenarioRejected(msg)
        return lo, hi

    # -- small helpers --------------------------------------------------------

    @staticmethod
    def _group_or_default(raw: Mapping[str, Any], scenario_id: str | None) -> str:
        """The comparison group, defaulting to ``scenario_id`` (ред. 35)."""
        group = raw.get("comparison_group")
        if group is None:
            if scenario_id is not None:
                return scenario_id
            msg = (
                "an unnamed scenario: declare 'scenario_id'/'comparison_group' "
                "or let run_accuracy.py inject the file name as the group"
            )
            raise ScenarioRejected(msg)
        if not isinstance(group, str) or not group:
            msg = "comparison_group must be a non-empty string"
            raise ScenarioRejected(msg)
        return group

    def _invariants(self, value: Any, participants: list[ParticipantSpec]) -> tuple[str, ...]:
        """Validate the invariant targets (grammar: ``"subject"``, or one of
        the participant's ``source_id``/``participant_id``).

        REV-3 (no silence): a name that matches no participant would
        silently disable the requested check.
        """
        if value is None:
            return ()
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            msg = "invariant_check_on must be a list of non-empty strings"
            raise ScenarioRejected(msg)
        known = (
            {"subject"}
            | {p.source_id for p in participants}
            | {p.participant_id for p in participants}
        )
        unknown = [item for item in value if item not in known]
        if unknown:
            msg = (
                "invariant_check_on names match no participant: "
                f"{sorted(set(unknown))}; known: {sorted(known)}"
            )
            raise ScenarioRejected(msg)
        return tuple(value)

    @staticmethod
    def _cross_check(queries: list[AccuracyQuery], regions: list[Region]) -> None:
        """Grammar contradictions knowable without a library (REV-3)."""
        ppf = any(q.characteristic == CharacteristicName.PPF for q in queries)
        if ppf:
            bad = [r.name for r in regions if r.space != PROBABILITY_SPACE]
            if bad:
                msg = (
                    "ppf queries measure quantile levels in probability space; "
                    f"regions {bad} declare another space"
                )
                raise ScenarioRejected(msg)
            # The level axis is (0, 1) by the definition of quantiles —
            # framework knowledge, no library needed.  A region that does not
            # intersect it resolves to an empty level set at grid generation
            # (which used to raise FrameworkDefect and kill the whole batch);
            # it is a grammar defect of the file, rejected here instead.
            # Partially overlapping intervals stay legal: the generator clamps
            # them to (0, 1) — mixed pdf+ppf scenarios share wide bulk regions.
            empty = [
                r.name
                for r in regions
                if not (0.0 if r.interval[0] is None else max(r.interval[0], 0.0))
                < (1.0 if r.interval[1] is None else min(r.interval[1], 1.0))
            ]
            if empty:
                msg = (
                    "ppf regions must intersect the level axis (0, 1); "
                    f"{empty} resolve to no quantile levels at all"
                )
                raise ScenarioRejected(msg)

    @staticmethod
    def _require_mapping(value: Any, what: str) -> None:
        if not isinstance(value, Mapping):
            msg = f"{what} must be an object"
            raise ScenarioRejected(msg)

    @staticmethod
    def _sort_key(value: Any) -> float | str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            msg = "sort_key must be a number or string"
            raise ScenarioRejected(msg)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return value
        msg = "sort_key must be a number or string"
        raise ScenarioRejected(msg)

    @staticmethod
    def _scenario_id(
        raw: Mapping[str, Any], group: str | None, sort_key: float | str | None
    ) -> str | None:
        """Explicit scenario_id, or ``group``-derived, or ``None`` (ред. 35).

        The three-way default: scenario_id > ``group[<sort_key>]`` > None;
        when both end up None the group defaults back to the scenario_id
        slot (both None together -> ``Rejected`` by ``_group_or_default``).
        """
        value = raw.get("scenario_id")
        if value is not None:
            if not isinstance(value, str) or not value:
                msg = "scenario_id must be a non-empty string"
                raise ScenarioRejected(msg)
            return value
        if group is None:
            return None
        return group if sort_key is None else f"{group}#{sort_key}"


def _json_safe(value: Any) -> Any:
    """Validate JSON-safety and normalize numbers (ints -> float)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        try:
            return float(value)
        except OverflowError as exc:
            msg = f"integer {value} is too large for a float"
            raise ValueError(msg) from exc
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            msg = f"non-finite number {value!r}"
            raise ValueError(msg)
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if key in out:
                msg = f"duplicate key {key!r} after string coercion"
                raise ValueError(msg)
            out[key] = _json_safe(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    msg = f"value of type {type(value).__name__} is not JSON-safe"
    raise ValueError(msg)


class ScenarioValidator:
    """Degenerate-comparison check on canonical configurations (B1 + М-5).

    The single code point of the rule "identical canonical identities ->
    degenerate comparison, visible as a validator note" (М-5); the runner
    calls it instead of re-implementing the rule (А-4).
    """

    def validate(self, spec: ScenarioSpec) -> list[ValidatorNote]:
        """Notes are attached by the runner to *both* results of a pair."""
        return [note for notes in self.notes_by_pair(spec).values() for note in notes]

    def notes_by_pair(
        self,
        spec: ScenarioSpec,
        pairs: Sequence[tuple[ParticipantSpec, ParticipantSpec]] | None = None,
    ) -> dict[tuple[str, str], list[ValidatorNote]]:
        """Degenerate notes keyed by the sorted pair key.

        ``pairs`` lets the orchestrator supply *its* pairs, so pairing is
        decided in one place; without it the validator derives the pairs
        from the spec policy.
        """
        sides = pairs if pairs is not None else self._pairs(spec)
        notes: dict[tuple[str, str], list[ValidatorNote]] = {}
        for left, right in sides:
            if left.identity() == right.identity():
                detail = (
                    f"participants {left.participant_id} and {right.participant_id} "
                    f"have identical canonical configurations; the comparison is "
                    f"degenerate by design and runs anyway"
                )
                key = (left.participant_id, right.participant_id)
                key = key if key[0] <= key[1] else (key[1], key[0])
                notes.setdefault(key, []).append(ValidatorNote(code=DEGENERATE_NOTE, detail=detail))
        return notes

    @staticmethod
    def _pairs(spec: ScenarioSpec) -> list[tuple[ParticipantSpec, ParticipantSpec]]:
        participants = list(spec.participants)
        policy = spec.pair_policy
        if policy == "subject":
            subject = spec.subject()
            if subject is None:
                return []
            # Same filter as the runner (REV-3): peer-role participants are
            # not measured in subject mode.
            others = [p for p in participants if p.role == Role.REFERENCE and p is not subject]
            return [(subject, other) for other in others]
        if policy == "symmetric":
            return [
                (participants[i], participants[j])
                for i in range(len(participants))
                for j in range(i + 1, len(participants))
            ]
        return [
            (participants[i], participants[j])
            for i, j in (pair for pair in policy if isinstance(pair, list))
        ]


__all__ = [
    "ScenarioExpander",
    "ScenarioLoader",
    "ScenarioValidator",
    "DEGENERATE_NOTE",
]
