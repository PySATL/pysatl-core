from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from math import inf

import numpy as np
import pytest

from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
    IntegerLatticeDiscreteSupport,
)
from pysatl_core.types import ContinuousSupportShape1D


class TestContinuousSupport:
    support_example = ContinuousSupport(left=0.0, right=1.0, left_closed=True, right_closed=False)

    @pytest.mark.parametrize(
        "point, expected_result",
        [
            (0, True),
            (1, False),
            (0.5, True),
            (-0.1, False),
            (inf, False),
            (-inf, False),
        ],
        ids=[
            "left_bound_closed",
            "right_bound_open",
            "inside_interval",
            "outside_interval",
            "+inf",
            "-inf",
        ],
    )
    def test_continuous_support_contains_scalar(self, point, expected_result):
        assert (point in self.support_example) is expected_result
        assert self.support_example.contains(point) is expected_result

    @pytest.mark.parametrize("infinity", [-inf, inf])
    def test_continuous_support_doesnt_contain_inf(self, infinity):
        # inf isn't considered as a number
        # but as a limit so support doesn't contain it even if it's a real line
        support = ContinuousSupport()
        assert infinity not in support
        assert support.contains(infinity) is False

    @pytest.mark.parametrize(
        "points,expected_result",
        [
            (np.array([-1.0, 0.0, 0.5, 1.0]), [False, True, True, False]),
            (np.array([]), []),
        ],
    )
    def test_continuous_support_contains_array(self, points, expected_result):
        # np.array doesn't have `in`(__contains__) syntax
        result = self.support_example.contains(points)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == expected_result
        if len(points) > 0:
            assert np.array_equal(result, np.array(expected_result))

    @pytest.mark.parametrize(
        "support, expected_shape",
        [
            (ContinuousSupport(1, 0), ContinuousSupportShape1D.EMPTY),
            (ContinuousSupport(0, 1), ContinuousSupportShape1D.BOUNDED_INTERVAL),
            (ContinuousSupport(left=0), ContinuousSupportShape1D.RAY_RIGHT),
            (ContinuousSupport(right=0), ContinuousSupportShape1D.RAY_LEFT),
            (ContinuousSupport(), ContinuousSupportShape1D.REAL_LINE),
            (ContinuousSupport(1, 1), ContinuousSupportShape1D.SINGLE_POINT),
        ],
        ids=["empty", "bounded", "ray_left", "ray_right", "real_line", "single_point"],
    )
    def test_continuous_support_is_empty_and_shape_variants(self, support, expected_shape):
        assert support.shape == expected_shape

    def test_inf_bound_is_not_closed(self):
        assert ContinuousSupport().left_closed is False
        assert ContinuousSupport().right_closed is False


class TestExplicitTableDiscreteSupport:
    points_example = [3, 1, 2, 2, 5]
    support_example = ExplicitTableDiscreteSupport(points_example)
    x_examples = [0, 1, 1.5, 2, 4.9, 5, 10]

    def test_table_is_sorted_and_deduplicated(self):
        np.testing.assert_array_equal(self.support_example.points, np.array([1, 2, 3, 5]))

    @pytest.mark.parametrize(
        "point, expected_result",
        [
            (2, True),
            (4, False),
            (2.0, True),
            (2.5, False),
        ],
    )
    def test_contains_scalar(self, point, expected_result):
        assert (point in self.support_example) is expected_result
        assert self.support_example.contains(point) is expected_result

    @pytest.mark.parametrize(
        "points, expected_result",
        [
            (np.array([0, 1, 2, 3, 4, 5]), [False, True, True, True, False, True]),
            (np.array([]), []),
            (np.array([1.0, 1.5, 2.0, 2.5]), [True, False, True, False]),
            ([0, 1, 2, 3, 4, 5], [False, True, True, True, False, True]),
        ],
    )
    def test_contains_array(self, points, expected_result):
        result = self.support_example.contains(points)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == expected_result
        if len(points) > 0:
            assert np.array_equal(result, np.array(expected_result))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            ExplicitTableDiscreteSupport([], assume_sorted=True)

    def test_assume_sorted_but_unsorted(self):
        unsorted_points = [5, 1, 3, 2]
        support = ExplicitTableDiscreteSupport(unsorted_points, assume_sorted=True)
        np.testing.assert_array_equal(support.points, np.array([5, 1, 3, 2]))

    @pytest.mark.parametrize(
        "x, expected_points",
        zip(
            x_examples, [[], [1], [1], [1, 2], [1, 2, 3], [1, 2, 3, 5], [1, 2, 3, 5]], strict=False
        ),
    )
    def test_iter_leq(self, x, expected_points):
        result = list(self.support_example.iter_leq(x))
        assert result == expected_points

    @pytest.mark.parametrize(
        "x, expected_prev", zip(x_examples, [None, None, 1, 1, 3, 3, 5], strict=False)
    )
    def test_prev(self, x, expected_prev):
        result = self.support_example.prev(x)
        assert result == expected_prev

    def test_first(self):
        assert self.support_example.first() == 1

    @pytest.mark.parametrize(
        "x, expected_next",
        zip(x_examples, [1, 2, 2, 3, 5, None, None], strict=False),
    )
    def test_next(self, x, expected_next):
        assert self.support_example.next(x) == expected_next

    def test_iter_points_and_iter(self):
        assert list(self.support_example.iter_points()) == [1, 2, 3, 5]
        assert list(iter(self.support_example)) == [1, 2, 3, 5]

    def test_points_property_returns_copy(self):
        pts_copy = self.support_example.points
        pts_copy[0] = -999
        np.testing.assert_array_equal(self.support_example.points, np.array([1, 2, 3, 5]))


class TestIntegerLatticeDiscreteSupport:
    support_examples = {
        "boundless": IntegerLatticeDiscreteSupport(residue=0, modulus=1),
        "bounded_left": IntegerLatticeDiscreteSupport(residue=1, modulus=2, min_k=5),
        "bounded_right": IntegerLatticeDiscreteSupport(residue=1, modulus=3, max_k=10),
        "full_bounded": IntegerLatticeDiscreteSupport(residue=0, modulus=2, min_k=0, max_k=10),
    }

    def test_invalid_modulus_raises(self):
        with pytest.raises(ValueError):
            IntegerLatticeDiscreteSupport(residue=0, modulus=0)

    @pytest.mark.parametrize(
        "support_name, point, expected_result",
        [
            ("boundless", 1, True),
            ("boundless", 1.5, False),
            ("bounded_left", 1, False),
            ("bounded_left", 5, True),
            ("bounded_right", 1, True),
            ("bounded_right", 13, False),
            ("full_bounded", 0, True),
            ("full_bounded", -2, False),
        ],
    )
    def test_contains_scalar(self, support_name, point, expected_result):
        support = self.support_examples[support_name]
        assert (point in support) is expected_result
        assert support.contains(point) is expected_result

    @pytest.mark.parametrize(
        "support_name, points, expected_result",
        [
            (
                "bounded_left",
                np.array([3, 4, 5, 6]),
                [False, False, True, False],
            ),
            (
                "bounded_right",
                np.array([-2, 1, 4, 7]),
                [True, True, True, True],
            ),
            (
                "full_bounded",
                np.array([-2, 0, 10, 12]),
                [False, True, True, False],
            ),
            (
                "boundless",
                np.array([]),
                [],
            ),
        ],
    )
    def test_contains_array(self, support_name, points, expected_result):
        support = self.support_examples[support_name]
        result = support.contains(points)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == expected_result
        if len(points) > 0:
            assert np.array_equal(result, np.array(expected_result))

    def test_iter_points(self):
        support = self.support_examples["boundless"]
        with pytest.raises(RuntimeError):
            list(support.iter_points())

        support = self.support_examples["bounded_left"]
        it = support.iter_points()
        first_five = [next(it) for _ in range(5)]
        assert first_five == [5, 7, 9, 11, 13]

        support = self.support_examples["bounded_right"]
        it = support.iter_points()
        first_five = [next(it) for _ in range(5)]
        assert first_five == [10, 7, 4, 1, -2]

        support = self.support_examples["full_bounded"]
        assert list(support.iter_points()) == [0, 2, 4, 6, 8, 10]

    @pytest.mark.parametrize(
        "support_name, x, expected_points_or_error",
        [
            ("bounded_left", 8, [5, 7]),
            ("bounded_left", 9, [5, 7, 9]),
            ("bounded_right", 8, RuntimeError),
            ("bounded_right", -1, RuntimeError),
            ("full_bounded", 5, [0, 2, 4]),
            ("full_bounded", 10, [0, 2, 4, 6, 8, 10]),
            ("full_bounded", -1, []),
        ],
    )
    def test_iter_leq(self, support_name, x, expected_points_or_error):
        support = self.support_examples[support_name]
        if expected_points_or_error is RuntimeError:
            with pytest.raises(RuntimeError):
                list(support.iter_leq(x))
        else:
            result = list(support.iter_leq(x))
            assert result == expected_points_or_error

    @pytest.mark.parametrize(
        "support_name, x, expected_prev",
        [
            ("bounded_left", 6, 5),
            ("bounded_left", 8, 7),
            ("bounded_left", 5, None),
            ("bounded_left", 4, None),
            ("bounded_right", 4, 1),
            ("bounded_right", 7, 4),
            ("bounded_right", 10, 7),
            ("full_bounded", 5, 4),
            ("full_bounded", 4, 2),
            ("full_bounded", 2, 0),
            ("full_bounded", 0, None),
            ("full_bounded", 12, 10),
            ("full_bounded", -1, None),
            ("boundless", 0, -1),
        ],
    )
    def test_prev(self, support_name, x, expected_prev):
        support = self.support_examples[support_name]
        result = support.prev(x)
        assert result == expected_prev

    @pytest.mark.parametrize(
        "support_name, expected_first",
        [
            ("boundless", None),
            ("bounded_left", 5),
            ("bounded_right", None),
            ("full_bounded", 0),
        ],
    )
    def test_first(self, support_name, expected_first):
        support = self.support_examples[support_name]
        result = support.first()
        assert result == expected_first

    @pytest.mark.parametrize(
        "support_name, current, expected_next",
        [
            ("bounded_left", 5, 7),
            ("bounded_left", 7, 9),
            ("bounded_left", 9, 11),
            ("bounded_right", 1, 4),
            ("bounded_right", 4, 7),
            ("bounded_right", 7, 10),
            ("bounded_right", 10, None),
            ("full_bounded", 0, 2),
            ("full_bounded", 2, 4),
            ("full_bounded", 4, 6),
            ("full_bounded", 6, 8),
            ("full_bounded", 8, 10),
            ("full_bounded", 10, None),
        ],
    )
    def test_next(self, support_name, current, expected_next):
        support = self.support_examples[support_name]
        result = support.next(current)
        assert result == expected_next

    def test_iter(self):
        support = self.support_examples["full_bounded"]
        result = []
        for point in support:
            result.append(point)
            if len(result) >= 4:
                break
        assert result == [0, 2, 4, 6]

    @pytest.mark.parametrize(
        "support_name, expected_left_bounded, expected_right_bounded",
        [
            ("boundless", False, False),
            ("bounded_left", True, False),
            ("bounded_right", False, True),
            ("full_bounded", True, True),
        ],
    )
    def test_is_bounded_properties(
        self, support_name, expected_left_bounded, expected_right_bounded
    ):
        support = self.support_examples[support_name]
        assert support.is_left_bounded == expected_left_bounded
        assert support.is_right_bounded == expected_right_bounded

    def test_iter_points_raises_when_no_points_in_bounds(self):
        support = IntegerLatticeDiscreteSupport(residue=0, modulus=2, min_k=10, max_k=5)
        with pytest.raises(RuntimeError):
            list(support.iter_points())
        assert support.first() is None
