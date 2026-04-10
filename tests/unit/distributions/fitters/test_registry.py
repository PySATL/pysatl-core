"""
Tests for FitterRegistry: matching, selection, and priority logic.
"""

from __future__ import annotations

__author__ = "Irina Sergeeva"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_core.distributions.fitters.base import FitterDescriptor
from pysatl_core.distributions.fitters.registry import FitterRegistry
from pysatl_core.types import CharacteristicName


def _dummy_fitter(distribution: Any, /, **kwargs: Any) -> Any:
    return None


def _make_desc(
    name: str,
    target: str = CharacteristicName.CDF,
    sources: list[str] | None = None,
    tags: frozenset[str] | None = None,
    priority: int = 0,
) -> FitterDescriptor:
    return FitterDescriptor(
        name=name,
        target=target,
        sources=sources or [CharacteristicName.PDF],
        fitter=_dummy_fitter,
        options=(),
        tags=tags or frozenset(),
        priority=priority,
    )


class TestFitterRegistry:
    """Tests for the FitterRegistry class."""

    def test_register_and_find(self) -> None:
        reg = FitterRegistry()
        desc = _make_desc("pdf_to_cdf")
        reg.register(desc)
        found = reg.find(CharacteristicName.CDF, [CharacteristicName.PDF])
        assert found is desc

    def test_find_returns_none_when_no_match(self) -> None:
        reg = FitterRegistry()
        found = reg.find(CharacteristicName.CDF, [CharacteristicName.PDF])
        assert found is None

    def test_find_returns_none_when_tags_dont_match(self) -> None:
        reg = FitterRegistry()
        desc = _make_desc("pdf_to_cdf", tags=frozenset({"continuous"}))
        reg.register(desc)
        found = reg.find(
            CharacteristicName.CDF,
            [CharacteristicName.PDF],
            required_tags=frozenset({"discrete"}),
        )
        assert found is None

    def test_find_matches_with_tag_superset(self) -> None:
        reg = FitterRegistry()
        desc = _make_desc("pdf_to_cdf", tags=frozenset({"continuous", "univariate"}))
        reg.register(desc)
        found = reg.find(
            CharacteristicName.CDF,
            [CharacteristicName.PDF],
            required_tags=frozenset({"continuous"}),
        )
        assert found is desc

    def test_find_selects_highest_priority(self) -> None:
        reg = FitterRegistry()
        low = _make_desc("low_prio", priority=0)
        high = _make_desc("high_prio", priority=10)
        reg.register(low)
        reg.register(high)
        found = reg.find(CharacteristicName.CDF, [CharacteristicName.PDF])
        assert found is high

    def test_find_all_returns_sorted_by_priority(self) -> None:
        reg = FitterRegistry()
        d1 = _make_desc("p0", priority=0)
        d2 = _make_desc("p5", priority=5)
        d3 = _make_desc("p3", priority=3)
        reg.register(d1)
        reg.register(d2)
        reg.register(d3)
        result = reg.find_all(CharacteristicName.CDF, [CharacteristicName.PDF])
        assert [d.name for d in result] == ["p5", "p3", "p0"]

    def test_find_all_filters_by_tags(self) -> None:
        reg = FitterRegistry()
        cont = _make_desc("cont", tags=frozenset({"continuous"}))
        disc = _make_desc("disc", tags=frozenset({"discrete"}))
        reg.register(cont)
        reg.register(disc)
        result = reg.find_all(
            CharacteristicName.CDF,
            [CharacteristicName.PDF],
            required_tags=frozenset({"continuous"}),
        )
        assert len(result) == 1
        assert result[0].name == "cont"

    def test_find_all_returns_empty_when_no_match(self) -> None:
        reg = FitterRegistry()
        result = reg.find_all(CharacteristicName.CDF, [CharacteristicName.PDF])
        assert result == []

    def test_register_many(self) -> None:
        reg = FitterRegistry()
        descs = [_make_desc(f"d{i}") for i in range(5)]
        reg.register_many(descs)
        assert len(reg) == 5

    def test_all_descriptors_preserves_insertion_order(self) -> None:
        reg = FitterRegistry()
        d1 = _make_desc("first")
        d2 = _make_desc("second")
        d3 = _make_desc("third")
        reg.register(d1)
        reg.register(d2)
        reg.register(d3)
        assert [d.name for d in reg.all_descriptors()] == ["first", "second", "third"]

    def test_len(self) -> None:
        reg = FitterRegistry()
        assert len(reg) == 0
        reg.register(_make_desc("a"))
        assert len(reg) == 1
        reg.register(_make_desc("b"))
        assert len(reg) == 2

    def test_contains(self) -> None:
        reg = FitterRegistry()
        reg.register(_make_desc("pdf_to_cdf"))
        assert "pdf_to_cdf" in reg
        assert "nonexistent" not in reg

    def test_different_source_target_pairs_are_independent(self) -> None:
        reg = FitterRegistry()
        pdf_cdf = _make_desc(
            "pdf_cdf",
            target=CharacteristicName.CDF,
            sources=[CharacteristicName.PDF],
        )
        cdf_ppf = _make_desc(
            "cdf_ppf",
            target=CharacteristicName.PPF,
            sources=[CharacteristicName.CDF],
        )
        reg.register(pdf_cdf)
        reg.register(cdf_ppf)

        assert reg.find(CharacteristicName.CDF, [CharacteristicName.PDF]) is pdf_cdf
        assert reg.find(CharacteristicName.PPF, [CharacteristicName.CDF]) is cdf_ppf
        assert reg.find(CharacteristicName.PPF, [CharacteristicName.PDF]) is None

    def test_all_fitter_descriptors_registered(self) -> None:
        """Verify that ALL_FITTER_DESCRIPTORS contains all 8 built-in fitters."""
        from pysatl_core.distributions.fitters import ALL_FITTER_DESCRIPTORS

        assert len(ALL_FITTER_DESCRIPTORS) == 8
        names = {d.name for d in ALL_FITTER_DESCRIPTORS}
        expected = {
            "pdf_to_cdf_1C",
            "cdf_to_pdf_1C",
            "cdf_to_ppf_1C",
            "ppf_to_cdf_1C",
            "pmf_to_cdf_1D",
            "cdf_to_pmf_1D",
            "cdf_to_ppf_1D",
            "ppf_to_cdf_1D",
        }
        assert names == expected
