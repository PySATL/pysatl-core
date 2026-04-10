"""Mapping UNU.RAN methods to the distribution characteristics they require."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from pysatl_core.sampling.unuran.method_config import UnuranMethod
from pysatl_core.types import CharacteristicName


@dataclass(frozen=True, slots=True)
class MethodCharacteristics:
    """
    Describe which analytical characteristics and distribution properties a UNU.RAN
    method depends on.

    requires_support indicates the method needs a finite, known support (e.g., HINV
    requires a right boundary).
    """

    required: frozenset[CharacteristicName]
    optional: frozenset[CharacteristicName] = frozenset()
    requires_support: bool = False

    def __str__(self) -> str:
        return f"""
            Mandatory requirements: {self.required}.
            Optional requirements: {self.optional}.
            Requires support: {self.requires_support}.
            """


METHOD_CHARACTERISTIC_REQUIREMENTS: Final[Mapping[UnuranMethod, MethodCharacteristics]] = {
    UnuranMethod.AROU: MethodCharacteristics(
        required=frozenset({CharacteristicName.PDF, CharacteristicName.DPDF})
    ),
    UnuranMethod.TDR: MethodCharacteristics(
        required=frozenset({CharacteristicName.PDF, CharacteristicName.DPDF})
    ),
    UnuranMethod.PINV: MethodCharacteristics(
        required=frozenset({CharacteristicName.PDF}),
        optional=frozenset({CharacteristicName.CDF, CharacteristicName.PPF}),
    ),
    UnuranMethod.HINV: MethodCharacteristics(
        required=frozenset({CharacteristicName.PPF}),
        optional=frozenset({CharacteristicName.CDF}),
        requires_support=True,
    ),
    UnuranMethod.NINV: MethodCharacteristics(
        required=frozenset({CharacteristicName.CDF}),
        optional=frozenset({CharacteristicName.PDF}),
    ),
    UnuranMethod.DGT: MethodCharacteristics(
        required=frozenset({CharacteristicName.PMF}),
        requires_support=True,
    ),
}
