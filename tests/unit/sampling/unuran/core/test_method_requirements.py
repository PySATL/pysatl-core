"""
Unit tests for pysatl_core.sampling.unuran.core.method_requirements

Tests:
  - MethodCharacteristics dataclass (required, optional, requires_support, __str__)
  - METHOD_CHARACTERISTIC_REQUIREMENTS mapping completeness and correctness
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_core.sampling.unuran.core.method_requirements import (
    METHOD_CHARACTERISTIC_REQUIREMENTS,
    MethodCharacteristics,
)
from pysatl_core.sampling.unuran.method_config import UnuranMethod
from pysatl_core.types import CharacteristicName


class TestMethodCharacteristics:
    """Tests for the MethodCharacteristics dataclass."""

    def test_required_chars_are_stored(self) -> None:
        """required is stored as a frozenset."""
        req = frozenset({CharacteristicName.PDF})
        mc = MethodCharacteristics(required=req)

        assert mc.required == req

    def test_optional_defaults_to_empty_frozenset(self) -> None:
        """optional defaults to an empty frozenset when not supplied."""
        mc = MethodCharacteristics(required=frozenset({CharacteristicName.PDF}))

        assert mc.optional == frozenset()

    def test_requires_support_defaults_to_false(self) -> None:
        """requires_support defaults to False."""
        mc = MethodCharacteristics(required=frozenset())

        assert mc.requires_support is False

    def test_requires_support_can_be_set_true(self) -> None:
        """requires_support can be explicitly set to True."""
        mc = MethodCharacteristics(required=frozenset(), requires_support=True)

        assert mc.requires_support is True

    def test_str_contains_required_and_optional_and_support_info(self) -> None:
        """__str__ includes mandatory requirements, optional requirements, and support flag."""
        mc = MethodCharacteristics(
            required=frozenset({CharacteristicName.PDF}),
            optional=frozenset({CharacteristicName.CDF}),
            requires_support=True,
        )
        text = str(mc)

        assert "Mandatory requirements" in text
        assert "Optional requirements" in text
        assert "Requires support" in text

    def test_frozen_raises_on_mutation(self) -> None:
        """MethodCharacteristics is frozen — post-creation assignment raises."""
        mc = MethodCharacteristics(required=frozenset())

        with pytest.raises((AttributeError, TypeError)):
            mc.requires_support = True  # type: ignore[misc]


class TestMethodCharacteristicRequirementsMapping:
    """Tests for the METHOD_CHARACTERISTIC_REQUIREMENTS constant mapping."""

    @pytest.mark.parametrize("method", list(UnuranMethod))
    def test_auto_method_is_not_in_requirements(self, method: UnuranMethod) -> None:
        """AUTO is not a concrete method and should not appear in the requirements map."""
        if method == UnuranMethod.AUTO:
            assert UnuranMethod.AUTO not in METHOD_CHARACTERISTIC_REQUIREMENTS

    def test_all_concrete_methods_have_requirements(self) -> None:
        """Every non-AUTO method has an entry in the requirements map."""
        concrete_methods = [m for m in UnuranMethod if m != UnuranMethod.AUTO]
        for method in concrete_methods:
            assert (
                method in METHOD_CHARACTERISTIC_REQUIREMENTS
            ), f"Method {method} is missing from METHOD_CHARACTERISTIC_REQUIREMENTS"

    def test_pinv_requires_pdf(self) -> None:
        """PINV requires PDF as mandatory characteristic."""
        req = METHOD_CHARACTERISTIC_REQUIREMENTS[UnuranMethod.PINV]

        assert CharacteristicName.PDF in req.required

    def test_pinv_does_not_require_support(self) -> None:
        """PINV does not mandate a finite support."""
        req = METHOD_CHARACTERISTIC_REQUIREMENTS[UnuranMethod.PINV]

        assert req.requires_support is False

    def test_hinv_requires_ppf_and_finite_support(self) -> None:
        """HINV requires PPF and a bounded support."""
        req = METHOD_CHARACTERISTIC_REQUIREMENTS[UnuranMethod.HINV]

        assert CharacteristicName.PPF in req.required
        assert req.requires_support is True

    def test_ninv_requires_cdf(self) -> None:
        """NINV requires CDF as mandatory characteristic."""
        req = METHOD_CHARACTERISTIC_REQUIREMENTS[UnuranMethod.NINV]

        assert CharacteristicName.CDF in req.required

    def test_dgt_requires_pmf_and_finite_support(self) -> None:
        """DGT requires PMF and a bounded support."""
        req = METHOD_CHARACTERISTIC_REQUIREMENTS[UnuranMethod.DGT]

        assert CharacteristicName.PMF in req.required
        assert req.requires_support is True

    def test_arou_requires_pdf_and_dpdf(self) -> None:
        """AROU requires both PDF and dPDF."""
        req = METHOD_CHARACTERISTIC_REQUIREMENTS[UnuranMethod.AROU]

        assert CharacteristicName.PDF in req.required
        assert CharacteristicName.DPDF in req.required

    def test_tdr_requires_pdf_and_dpdf(self) -> None:
        """TDR requires both PDF and dPDF."""
        req = METHOD_CHARACTERISTIC_REQUIREMENTS[UnuranMethod.TDR]

        assert CharacteristicName.PDF in req.required
        assert CharacteristicName.DPDF in req.required

    def test_all_required_sets_are_frozensets(self) -> None:
        """Every required/optional field is a frozenset (immutable)."""
        for method, req in METHOD_CHARACTERISTIC_REQUIREMENTS.items():
            assert isinstance(req.required, frozenset), f"{method}.required is not frozenset"
            assert isinstance(req.optional, frozenset), f"{method}.optional is not frozenset"
