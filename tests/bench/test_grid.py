"""Special point detection and grid generation (N1, D7 order, В-7)."""

from __future__ import annotations

import numpy as np

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
