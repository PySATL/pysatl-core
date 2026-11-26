"""Special point detection and grid generation (N1, D7 order, B-7)."""

from __future__ import annotations

import numpy as np
import pytest

from pysatl_core.types import CharacteristicName
from pysatl_core_bench import (
    AccuracyQuery,
    DistributionSpec,
    GridGenerator,
    Region,
    SpecialPoint,
    SpecialPointDetector,
    SupportBounds,
)
from pysatl_core_bench.accuracy.errors import FrameworkDefect


def distribution(family: str = "Normal", **params: float) -> DistributionSpec:
    return DistributionSpec(family=family, parametrization="meanStd", parameters=dict(params))


class TestSpecialPointDetector:
    def test_finite_support_boundaries_are_predicted(self) -> None:
        support = SupportBounds(left=-1.0, right=2.0)
        points = SpecialPointDetector().detect(distribution(), support)
        xs = sorted(p.x for p in points)
        assert xs == [-1.0, 2.0]

    def test_exponential_boundary_jump(self) -> None:
        support = SupportBounds(left=0.0, right=float("inf"), right_closed=False)
        d = DistributionSpec(family="Exponential", parametrization="rate", parameters={"rate": 1.5})
        points = SpecialPointDetector().detect(d, support)
        kinds = {pt.x: pt.kind for pt in points}
        assert kinds[0.0] == "pdf_jump"

    def test_gamma_singularity_for_shape_below_one(self) -> None:
        support = SupportBounds(left=0.0, right=float("inf"), right_closed=False)
        d = DistributionSpec(
            family="Gamma",
            parametrization="shapeScale",
            parameters={"shape": 0.5, "scale": 1.0},
        )
        points = SpecialPointDetector().detect(d, support)
        assert any(pt.kind == "singularity" and pt.x == 0.0 for pt in points)

    def test_normal_has_no_interior_points(self) -> None:
        points = SpecialPointDetector().detect(distribution(mu=0.0, sigma=1.0), None)
        assert points == []

    def test_gamma_without_shape_defaults_to_pdf_jump(self) -> None:
        # _shape() finds no shape key: the rule falls back to the jump kind
        # (the singularity kind is reserved for shape < 1).
        support = SupportBounds(left=0.0, right=float("inf"), right_closed=False)
        d = DistributionSpec("Gamma", "shapeScale", {"theta": 1.0})
        points = SpecialPointDetector().detect(d, support)
        kinds = {pt.x: pt.kind for pt in points}
        assert kinds[0.0] == "pdf_jump"

    def test_uniform_needs_no_curated_points(self) -> None:
        # A bounded-support family: its pdf jumps coincide with the support
        # boundaries, which the support rule already predicts.
        support = SupportBounds(left=0.0, right=1.0)
        points = SpecialPointDetector().detect(
            DistributionSpec("ContinuousUniform", None, {}), support
        )
        assert [p for p in points if p.source == "curated_rules"] == []

    def test_never_takes_points_from_subject_output(self) -> None:
        # the detector API has no channel for subject values at all:
        # it accepts only the scenario distribution and the support bounds.
        import inspect

        signature = inspect.signature(SpecialPointDetector.detect)
        assert list(signature.parameters) == ["self", "distribution", "support"]


class TestGridGenerator:
    def queries(self, characteristic: str = "cdf", hint: int | None = None) -> list[AccuracyQuery]:
        grid_hint = {"n_points": hint} if hint else None
        return [
            AccuracyQuery(characteristic=CharacteristicName(characteristic), grid_hint=grid_hint)
        ]

    def test_grid_belongs_to_region(self) -> None:
        grids = GridGenerator().generate(
            self.queries(),
            [
                Region("bulk", (-3.0, 3.0)),
                Region("tail", (3.0, None)),
            ],
            special_points=[],
            support=SupportBounds(left=float("-inf"), right=float("inf")),
        )
        assert [g.region.name for g in grids] == ["bulk", "tail"]
        assert all(g.query.characteristic == CharacteristicName.CDF for g in grids)

    def test_densifies_near_special_points(self) -> None:
        plain = GridGenerator().generate(
            self.queries(hint=8), [Region("bulk", (-3.0, 3.0))], [], None
        )[0]
        special = [SpecialPoint(x=0.0, kind="pdf_jump", source="curated_rules")]
        densified = GridGenerator().generate(
            self.queries(hint=8),
            [Region("bulk", (-3.0, 3.0))],
            special,
            None,
        )[0]
        assert densified.points.size > plain.points.size
        near = np.min(np.abs(densified.points - 0.0))
        assert near < np.min(np.abs(plain.points - 0.0))

    def test_points_respect_support(self) -> None:
        support = SupportBounds(left=0.0, right=float("inf"), left_closed=True)
        grids = GridGenerator().generate(
            self.queries(hint=33), [Region("bulk", (-5.0, 5.0))], [], support
        )
        assert grids[0].points.min() >= 0.0

    def test_unbounded_side_uses_tail_extension(self) -> None:
        grids = GridGenerator().generate(
            self.queries(hint=9), [Region("tail", (3.0, None))], [], None
        )
        lo, hi = grids[0].points.min(), grids[0].points.max()
        assert lo == 3.0
        assert 12.9 <= hi <= 13.1

    def test_tail_extension_parameter_scales_the_resolved_side(self) -> None:
        grids = GridGenerator().generate(
            self.queries(hint=9), [Region("tail", (3.0, None))], [], None, tail_extension=5.0
        )
        lo, hi = grids[0].points.min(), grids[0].points.max()
        assert lo == 3.0
        assert 7.9 <= hi <= 8.1

    def test_query_hint_overrides_the_resolved_tail_extension(self) -> None:
        # The per-query override wins over the runner-resolved per-family
        # value: the user asked for a concrete tail geometry.
        query = AccuracyQuery(
            characteristic=CharacteristicName.CDF,
            grid_hint={"n_points": 9, "tail_extension": 2.0},
        )
        grids = GridGenerator().generate(
            [query], [Region("tail", (3.0, None))], [], None, tail_extension=50.0
        )
        lo, hi = grids[0].points.min(), grids[0].points.max()
        assert lo == 3.0
        assert 4.9 <= hi <= 5.1

    def test_tail_extension_hint_must_be_a_positive_finite_number(self) -> None:
        for bad in (0.0, -1.0, "10", True, float("inf"), float("nan")):
            query = AccuracyQuery(
                characteristic=CharacteristicName.CDF,
                grid_hint={"n_points": 9, "tail_extension": bad},
            )
            with pytest.raises(FrameworkDefect, match="tail_extension"):
                GridGenerator().generate([query], [Region("tail", (3.0, None))], [], None)

    def test_left_unbounded_side_uses_tail_extension_too(self) -> None:
        grids = GridGenerator().generate(
            self.queries(hint=9), [Region("tail", (None, -3.0))], [], None, tail_extension=5.0
        )
        lo, hi = grids[0].points.min(), grids[0].points.max()
        assert hi == -3.0
        assert -8.1 <= lo <= -7.9

    def test_densify_false_keeps_exactly_the_uniform_grid(self) -> None:
        # AP-3/rev. 61 opt-out: the generated part is exactly the linspace
        # — no special-point offsets, no inserted special points — so a
        # user who wants the untouched uniform geometry gets exactly
        # n_points points.
        special = [SpecialPoint(0.0, "pdf_jump", "curated_rules")]
        support = SupportBounds(left=float("-inf"), right=float("inf"))
        query = AccuracyQuery(
            characteristic=CharacteristicName.CDF,
            grid_hint={"n_points": 21, "densify": False},
        )
        grids = GridGenerator().generate([query], [Region("bulk", (-3.0, 3.0))], special, support)
        assert grids[0].points.size == 21
        assert 0.0 in np.asarray(grids[0].points)  # linspace midpoint, not densification

        default = GridGenerator().generate(
            [AccuracyQuery(characteristic=CharacteristicName.CDF, grid_hint={"n_points": 21})],
            [Region("bulk", (-3.0, 3.0))],
            special,
            support,
        )
        assert default[0].points.size > 21  # the default still densifies

    def test_densify_hint_must_be_a_boolean(self) -> None:
        for bad in (0, 1, "false"):
            query = AccuracyQuery(
                characteristic=CharacteristicName.CDF,
                grid_hint={"n_points": 9, "densify": bad},
            )
            with pytest.raises(FrameworkDefect, match="densify"):
                GridGenerator().generate([query], [Region("bulk", (-3.0, 3.0))], [], None)

    def test_densify_absent_means_enabled(self) -> None:
        # An explicit null keeps the default: a hint field the user did
        # not set must not silently disable anything.
        special = [SpecialPoint(0.0, "pdf_jump", "curated_rules")]
        support = SupportBounds(left=float("-inf"), right=float("inf"))
        query = AccuracyQuery(
            characteristic=CharacteristicName.CDF,
            grid_hint={"n_points": 21, "densify": None},
        )
        grids = GridGenerator().generate([query], [Region("bulk", (-3.0, 3.0))], special, support)
        assert grids[0].points.size > 21

    def test_removed_out_of_support_count_is_carried_by_the_grid(self) -> None:
        # The support filter never fires on generator-produced geometry
        # (the resolved interval lives inside the support) — but the count
        # it would report is part of the Grid so the runner can say
        # "removed", per AP-1.
        support = SupportBounds(left=0.0, right=1.0)
        grids = GridGenerator().generate(
            self.queries(hint=11), [Region("bulk", (-1.0, 2.0))], [], support
        )
        assert grids[0].points.min() >= 0.0
        assert grids[0].n_removed_out_of_support == 0

    def test_open_support_boundary_shifts_the_first_point_inward(self) -> None:
        # _inner_bound moves an open boundary one ulp inside the support:
        # the grid never touches the excluded boundary value itself.
        support = SupportBounds(left=0.0, right=1.0, left_closed=False)
        grids = GridGenerator().generate(
            self.queries(hint=11), [Region("bulk", (-1.0, 1.0))], [], support
        )
        points = np.asarray(grids[0].points)
        assert points.min() > 0.0
        assert points.min() < 1e-6

    def test_bad_n_points_hint_is_a_framework_defect_in_the_generator(self) -> None:
        # The loader rejects a bad n_points; the generator check is the
        # same defense-in-depth as the tail/densify hint checks.
        query = AccuracyQuery(characteristic=CharacteristicName.CDF, grid_hint={"n_points": 3})
        with pytest.raises(FrameworkDefect, match="n_points"):
            GridGenerator().generate([query], [Region("bulk", (-1.0, 1.0))], [], None)

    def test_empty_ppf_region_is_a_framework_defect(self) -> None:
        # A ppf region that resolves empty on (0, 1) cannot be measured;
        # the loader rejects it at load, the generator check is defensive.
        query = AccuracyQuery(characteristic=CharacteristicName.PPF, grid_hint={"n_points": 9})
        with pytest.raises(FrameworkDefect, match="empty on"):
            GridGenerator().generate([query], [Region("levels", (1.0, 0.0))], [], None)

    def test_ppf_levels_live_in_probability_space(self) -> None:
        grids = GridGenerator().generate(
            self.queries("ppf", hint=11),
            [Region("levels", (0.9, 1.0))],
            [],
            None,
        )
        points = grids[0].points
        assert points.min() > 0.0 and points.max() < 1.0
        assert points.max() > 0.9

    def test_grid_hash_is_stable_bit_for_bit(self) -> None:
        support = SupportBounds(left=float("-inf"), right=float("inf"))
        a = GridGenerator().generate(
            self.queries(hint=17), [Region("bulk", (-1.0, 1.0))], [], support
        )
        b = GridGenerator().generate(
            self.queries(hint=17), [Region("bulk", (-1.0, 1.0))], [], support
        )
        assert a[0].grid_hash == b[0].grid_hash
        np.testing.assert_array_equal(a[0].points, b[0].points)

    def test_region_outside_support_resolves_to_empty_grid(self) -> None:
        """REV-2: a region against the runtime support is an outcome, not a crash."""
        grids = GridGenerator().generate(
            self.queries(hint=9),
            [Region("bulk", (-5.0, -4.0))],
            [],
            SupportBounds(left=0.0, right=1.0),
        )
        assert len(grids) == 1
        assert grids[0].points.size == 0

    def test_ppf_grid_ignores_argument_support(self) -> None:
        """REV-2: quantile levels live on the probability axis, not the x-axis."""
        grids = GridGenerator().generate(
            self.queries(characteristic="ppf", hint=9),
            [Region("levels", (0.0, 1.0))],
            [],
            SupportBounds(left=2.0, right=5.0),
        )
        assert grids[0].points.size == 9


class TestExplicitPoints:
    """The ``points`` branch of the generator (rev. 60).

    Explicit points are a contract, not a hint: they enter the grid as
    declared (order kept, no densification), the only reductions are cell
    membership, the runtime support (non-ppf) and the first-occurrence
    deduplication that the runner reports with a note.
    """

    def query(self, points: tuple[float, ...] | None, hint: int | None = None) -> AccuracyQuery:
        return AccuracyQuery(
            characteristic=CharacteristicName.CDF,
            grid_hint={"n_points": hint} if hint else None,
            points=points,
        )

    def test_points_only_query_measures_exactly_the_declared_points(self) -> None:
        points = (3.5, 25.0)
        grid = GridGenerator().generate(
            [self.query(points)], [Region("tail_right", (3.0, None))], [], None
        )[0]
        # No grid_hint -> no default grid diluting the answer: the grid is
        # exactly the user's array, including a point the generated tail
        # could never reach (its linspace stops at 3 + TAIL_EXTENSION = 13).
        assert list(grid.points) == [3.5, 25.0]

    def test_points_and_hint_combine_generated_first(self) -> None:
        points = (0.25, -2.0)
        grid = GridGenerator().generate(
            [self.query(points, hint=9)], [Region("bulk", (-3.0, 3.0))], [], None
        )[0]
        generated = (
            GridGenerator()
            .generate([self.query(None, hint=9)], [Region("bulk", (-3.0, 3.0))], [], None)[0]
            .points
        )
        assert grid.points.size == generated.size + 2
        assert list(grid.points[: generated.size]) == list(generated)
        assert list(grid.points[generated.size :]) == [0.25, -2.0]  # user order kept

    def test_points_are_filtered_by_the_region_interval(self) -> None:
        grid = GridGenerator().generate(
            [self.query((0.1, 5.0))], [Region("bulk", (-3.0, 3.0))], [], None
        )[0]
        # 5.0 belongs to another region's cell (orphan checks happen at
        # load time against ALL regions); in this cell only 0.1 survives.
        assert list(grid.points) == [0.1]

    def test_points_respect_the_support_without_clipping(self) -> None:
        support = SupportBounds(left=0.0, right=2.0)
        grid = GridGenerator().generate(
            [self.query((-1.0, 0.5, 1.5, 25.0))], [Region("bulk", (-3.0, 3.0))], [], support
        )[0]
        # Filter, not clip: out-of-support points are dropped whole and the
        # runner reports them; the survivors keep their declared values.
        assert list(grid.points) == [0.5, 1.5]

    def test_duplicate_points_are_measured_once(self) -> None:
        grid = GridGenerator().generate(
            [self.query((0.5, 0.5, 0.5, 1.5))], [Region("bulk", (-3.0, 3.0))], [], None
        )[0]
        assert list(grid.points) == [0.5, 1.5]

    def test_explicit_points_are_not_densified(self) -> None:
        special = [SpecialPoint(x=0.0, kind="pdf_jump", source="curated_rules")]
        points_only = GridGenerator().generate(
            [self.query((0.0,))], [Region("bulk", (-3.0, 3.0))], special, None
        )[0]
        # Offsets around a user-declared point would answer a question
        # nobody asked; the point itself is measured normally.
        assert list(points_only.points) == [0.0]

    def test_ppf_levels_never_touch_the_support(self) -> None:
        # REV-2: the support bounds the argument axis, not the probability
        # axis; a ppf level is filtered by its region interval only.
        query = AccuracyQuery(
            characteristic=CharacteristicName.PPF,
            grid_hint=None,
            points=(0.001, 0.999),
        )
        support = SupportBounds(left=-1.0, right=1.0)
        grid = GridGenerator().generate([query], [Region("levels", (0.0, 1.0))], [], support)[0]
        assert list(grid.points) == [0.001, 0.999]


class TestCuratedShapeRules:
    """Gamma shape lookup (08 §2.3): the v1 heuristic ignores
    ``parametrization`` and falls back to pdf_jump when no shape can be
    read — the detection must never guess a singularity it cannot justify."""

    def test_gamma_shape_below_one_is_a_singularity(self) -> None:
        points = SpecialPointDetector().detect(
            distribution("Gamma", shape=0.5),
            SupportBounds(left=0.0, right=float("inf")),
        )
        kinds = [p.kind for p in points if p.source == "curated_rules"]
        assert kinds == ["singularity"]

    def test_gamma_shape_at_or_above_one_is_a_jump(self) -> None:
        points = SpecialPointDetector().detect(
            distribution("Gamma", shape=1.5),
            SupportBounds(left=0.0, right=float("inf")),
        )
        kinds = [p.kind for p in points if p.source == "curated_rules"]
        assert kinds == ["pdf_jump"]

    def test_gamma_without_a_readable_shape_falls_back_to_jump(self) -> None:
        # No shape/alpha/k key, or a non-numeric one: the rule degrades to
        # the jump instead of inventing a singularity.
        no_shape = DistributionSpec(
            family="Gamma", parametrization="rate", parameters={"rate": 2.0}
        )
        text_shape = DistributionSpec(
            family="Gamma", parametrization="shape", parameters={"shape": "big"}
        )
        for spec in (no_shape, text_shape):
            points = SpecialPointDetector().detect(
                spec, SupportBounds(left=0.0, right=float("inf"))
            )
            kinds = [p.kind for p in points if p.source == "curated_rules"]
            assert kinds == ["pdf_jump"]

    def test_gamma_rules_silently_skip_a_nonzero_left_boundary(self) -> None:
        # Documented v1 boundary (AO-2): the curated rule is written for a
        # support anchored at 0; elsewhere it stays quiet — the
        # special_point_prediction option is the owner-confirmed future fix.
        points = SpecialPointDetector().detect(
            distribution("Gamma", shape=0.5),
            SupportBounds(left=-1.0, right=float("inf")),
        )
        assert [p.source for p in points] == ["support"]

    def test_exponential_jump_only_at_zero_left_boundary(self) -> None:
        points = SpecialPointDetector().detect(
            distribution("Exponential", rate=1.0),
            SupportBounds(left=0.0, right=float("inf")),
        )
        assert any(p.kind == "pdf_jump" and p.source == "curated_rules" for p in points)

    def test_detection_deduplicates_identical_points(self) -> None:
        support = SupportBounds(left=0.0, right=2.0)
        # Gamma shape=1 at left=0: the curated jump coincides with the
        # support boundary — one manifest entry, not two.
        points = SpecialPointDetector().detect(distribution("Gamma", shape=1.0), support)
        keys = [(p.x, p.kind, p.source) for p in points]
        assert len(keys) == len(set(keys))
