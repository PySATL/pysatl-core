"""Tail extension resolution: curated per-family scales and the fallback.

Pins the rev. 61 decisions: the tail length is derived from the
distribution parameters (N1 — pure arithmetic, no library calls), every
family carries its own multiplier constant, and the fixed constant is the
*reported* fallback for unknown families — never a silent default.
"""

from __future__ import annotations

from typing import Any

import pytest

from pysatl_core_bench import DistributionSpec
from pysatl_core_bench.accuracy import TAIL_EXTENSION
from pysatl_core_bench.accuracy.scenarios.tail import (
    TAIL_ORIGIN_CURATED,
    TAIL_ORIGIN_FALLBACK,
    resolve_tail_extension,
)


class TestCuratedFamilies:
    def test_normal_mean_std_uses_sigma_times_normal_multiplier(self) -> None:
        tail = resolve_tail_extension(
            DistributionSpec("Normal", "meanStd", {"mu": 0.0, "sigma": 2.0})
        )
        assert (tail.value, tail.origin) == (16.0, TAIL_ORIGIN_CURATED)
        assert tail.value == 8.0 * 2.0  # the Normal multiplier is 8 sigma

    def test_normal_mean_var_takes_square_root_of_variance(self) -> None:
        tail = resolve_tail_extension(
            DistributionSpec("Normal", "meanVar", {"mu": 0.0, "var": 9.0})
        )
        assert tail.value == 8.0 * 3.0

    def test_normal_mean_prec_reciprocal_precision(self) -> None:
        tail = resolve_tail_extension(
            DistributionSpec("Normal", "meanPrec", {"mu": 0.0, "tau": 0.5})
        )
        assert tail.value == 8.0 * 2.0

    def test_exponential_rate_is_reciprocal_scale(self) -> None:
        tail = resolve_tail_extension(DistributionSpec("Exponential", "rate", {"lambda_": 0.25}))
        assert tail.value == 40.0 * 4.0

    def test_exponential_scale_is_the_scale_itself(self) -> None:
        tail = resolve_tail_extension(DistributionSpec("Exponential", "scale", {"beta": 0.1}))
        assert tail.value == 40.0 * 0.1

    def test_gamma_shape_scale_uses_theta(self) -> None:
        tail = resolve_tail_extension(
            DistributionSpec("Gamma", "shapeScale", {"k": 2.0, "theta": 3.0})
        )
        assert tail.value == 40.0 * 3.0

    def test_gamma_shape_rate_reciprocal(self) -> None:
        tail = resolve_tail_extension(
            DistributionSpec("Gamma", "shapeRate", {"k": 2.0, "beta": 0.5})
        )
        assert tail.value == 40.0 * 2.0

    def test_gamma_mean_var_derives_theta(self) -> None:
        tail = resolve_tail_extension(DistributionSpec("Gamma", "meanVar", {"m": 3.0, "v": 6.0}))
        assert tail.value == 40.0 * 2.0

    def test_absent_parametrization_uses_the_documented_try_order(self) -> None:
        # No parametrization declared: the family's known keys are tried in
        # order (sigma -> var -> tau), the first readable scale wins.
        tail = resolve_tail_extension(DistributionSpec("Normal", None, {"mu": 1.0, "sigma": 4.0}))
        assert (tail.value, tail.origin) == (32.0, TAIL_ORIGIN_CURATED)
        var_only = resolve_tail_extension(DistributionSpec("Normal", None, {"mu": 1.0, "var": 4.0}))
        assert (var_only.value, var_only.origin) == (16.0, TAIL_ORIGIN_CURATED)


class TestFallbackIsVisible:
    def test_unknown_family_falls_back_to_the_fixed_constant(self) -> None:
        tail = resolve_tail_extension(DistributionSpec("Cauchy", None, {"gamma": 1.0}))
        assert (tail.value, tail.origin) == (TAIL_EXTENSION, TAIL_ORIGIN_FALLBACK)

    def test_unknown_parametrization_is_not_guessed_around(self) -> None:
        # An explicitly declared but unknown parametrization must not be
        # "rescued" by trying other keys: the honest answer is the
        # fallback, and the runner reports it.
        tail = resolve_tail_extension(DistributionSpec("Normal", "weird", {"sigma": 1.0}))
        assert (tail.value, tail.origin) == (TAIL_EXTENSION, TAIL_ORIGIN_FALLBACK)

    @pytest.mark.parametrize(
        "parameters",
        [
            {},
            {"mu": 0.0},  # no scale parameter
            {"sigma": 0.0},  # non-positive
            {"sigma": -1.0},  # non-positive
            {"sigma": "1.0"},  # not a number
            {"sigma": None},
        ],
    )
    def test_unreadable_parameters_fall_back(self, parameters: dict[str, Any]) -> None:
        tail = resolve_tail_extension(DistributionSpec("Normal", "meanStd", parameters))
        assert tail.origin == TAIL_ORIGIN_FALLBACK
        assert tail.value == TAIL_EXTENSION

    def test_bool_parameter_is_not_a_number(self) -> None:
        tail = resolve_tail_extension(DistributionSpec("Normal", "meanStd", {"sigma": True}))
        assert tail.origin == TAIL_ORIGIN_FALLBACK

    def test_no_readable_scale_without_parametrization_falls_back(self) -> None:
        # The try-order extractor exhausts the family's known keys without
        # finding a readable scale: the visible fallback, not a guess.
        tail = resolve_tail_extension(DistributionSpec("Normal", None, {"mu": 0.0}))
        assert (tail.value, tail.origin) == (TAIL_EXTENSION, TAIL_ORIGIN_FALLBACK)


class TestDeterminism:
    def test_replay_yields_the_same_pair(self) -> None:
        # Reconstructibility contract (rev. 61): the resolved pair is pure
        # arithmetic over the spec — two replays agree exactly.
        spec = DistributionSpec("Exponential", "rate", {"lambda_": 0.5})
        first = resolve_tail_extension(spec)
        second = resolve_tail_extension(spec)
        assert (first.value, first.origin) == (second.value, second.origin)
        assert first.value == 80.0
