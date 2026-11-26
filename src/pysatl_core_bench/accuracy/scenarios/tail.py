"""Tail extension: how far an unbounded region side reaches into the tail.

A region with an open side (``interval = [x, None]``) on an infinite
support must be measured over a finite segment; the length of that segment
decides *which part of the tail is measured at all*.  A length that ignores
the scale of the distribution measures the wrong thing in both directions:
for ``Normal(0, sigma=0.01)`` a fixed 10-unit tail is a thousand sigmas of
trivially-zero density, for ``Normal(0, sigma=100)`` it is a tenth of one
sigma with ~4% of the probability mass left behind the grid.

The length is therefore *derived from the distribution parameters* by
curated per-family arithmetic — never by calling the measured library
(N1: the runner holds no distribution object and grid geometry must not
depend on measured answers).  Each family carries its own multiplier
constant: tail shapes differ (Gaussian mass decays super-exponentially in
sigmas, exponential-family mass decays like ``exp(-x/scale)``), so a
single shared constant would either waste points on short-tailed families
or starve long-tailed ones.  Families the curated table does not know fall
back to the fixed constant — and the fallback is reported, never silent
(no-silence rule; the reader must know the tail length is the arbitrary
default, not a scale-aware choice).

A scenario can override the resolved length per query
(``grid_hint.tail_extension``): the user asked for a concrete geometry,
the framework measures exactly that geometry.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pysatl_core_bench.accuracy.scenarios.spec import DistributionSpec

__all__ = [
    "TAIL_EXTENSION",
    "TAIL_ORIGIN_CURATED",
    "TAIL_ORIGIN_FALLBACK",
    "TailExtension",
    "resolve_tail_extension",
]

TAIL_EXTENSION = 10.0
"""Fallback extension for an unbounded region side on an infinite support.

Used when the curated table has no linear scale for the scenario family
(or the parameters are unreadable); the runner reports every fallback.
The value is an absolute length in argument units — deliberately the last
resort, not the policy (see the module docstring).
"""

TAIL_ORIGIN_CURATED = "curated"
"""Origin tag: the length is ``multiplier(family) * curated scale``."""

TAIL_ORIGIN_FALLBACK = "fallback"
"""Origin tag: no curated scale — the fixed constant applies (reported)."""


@dataclass(frozen=True)
class TailExtension:
    """Resolved tail length and where it came from.

    ``origin`` distinguishes the curated per-family arithmetic from the
    fixed fallback: the same number with different origins means different
    things to a report reader (a deliberate scale-aware choice vs an
    arbitrary default), so the pair is resolved once and stays together.
    """

    value: float
    origin: str


def _positive(value: Any) -> float | None:
    """The parameter as a finite positive float, or ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _reciprocal(value: Any) -> float | None:
    """``1 / parameter`` as a finite positive float, or ``None``."""
    number = _positive(value)
    return None if number is None else 1.0 / number


def _sqrt(value: Any) -> float | None:
    """``sqrt(parameter)`` as a finite positive float, or ``None``."""
    number = _positive(value)
    return None if number is None else math.sqrt(number)


# The linear scale of the distribution, per family and parametrization —
# pure arithmetic over the scenario parameters (N1: no library calls).
_SCALE_EXTRACTORS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "Normal": MappingProxyType(
            {
                "meanStd": lambda p: _positive(p.get("sigma")),
                "meanVar": lambda p: _sqrt(p.get("var")),
                "meanPrec": lambda p: _reciprocal(p.get("tau")),
            }
        ),
        "Exponential": MappingProxyType(
            {
                "rate": lambda p: _reciprocal(p.get("lambda_")),
                "scale": lambda p: _positive(p.get("beta")),
            }
        ),
        "Gamma": MappingProxyType(
            {
                "shapeScale": lambda p: _positive(p.get("theta")),
                "shapeRate": lambda p: _reciprocal(p.get("beta")),
                "meanVar": lambda p: (
                    None
                    if (v := _positive(p.get("v"))) is None or (m := _positive(p.get("m"))) is None
                    else v / m
                ),
            }
        ),
    }
)

# Parametrization try-order when the scenario declares no parametrization
# (documented v1 heuristic: the first readable scale wins; the fallback is
# visible either way).
_PARAM_TRY_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "Normal": ("meanStd", "meanVar", "meanPrec"),
        "Exponential": ("rate", "scale"),
        "Gamma": ("shapeScale", "shapeRate", "meanVar"),
    }
)

# The multiplier constant is unique per family (owner decision): the same
# relative length serves families with the same tail-decay *shape*, and
# every family may carry its own number.  Rationale per family:
# - Normal: 8 sigma — the mass beyond is ~6.7e-16, at the float64
#   significance limit (LRE_CAP = 16); a longer tail measures nothing.
# - Exponential / Gamma: 40 scale units — the mass beyond decays like
#   exp(-40) ~ 4.3e-18, below anything float64 metrics can resolve.
# Families absent from the table (or with bounded support, where no
# unbounded side can occur) fall back to ``TAIL_EXTENSION`` + a report.
_FAMILY_MULTIPLIERS: Mapping[str, float] = MappingProxyType(
    {
        "Normal": 8.0,
        "Exponential": 40.0,
        "Gamma": 40.0,
    }
)


def resolve_tail_extension(distribution: DistributionSpec) -> TailExtension:
    """Resolve the tail length of one scenario distribution (deterministic).

    Curated per-family arithmetic over the scenario parameters — no
    library calls, no measured output (N1); replaying this function on the
    same spec always yields the same pair, so the effective tail geometry
    is reconstructible from the scenario alone.  Unknown family, unknown
    parametrization or unreadable parameters resolve to the fixed
    fallback; the runner reports every fallback (no silence).
    """
    family = distribution.family
    if family not in _FAMILY_MULTIPLIERS:
        return TailExtension(TAIL_EXTENSION, TAIL_ORIGIN_FALLBACK)
    extractors = _SCALE_EXTRACTORS[family]
    extractor = _extractor_for(family, extractors, distribution.parametrization)
    if extractor is None:
        # An explicitly declared but unknown parametrization lands here
        # too: it is not guessed around, the honest answer is the
        # visible fallback.
        return TailExtension(TAIL_EXTENSION, TAIL_ORIGIN_FALLBACK)
    scale = extractor(distribution.parameters)
    if scale is None:
        return TailExtension(TAIL_EXTENSION, TAIL_ORIGIN_FALLBACK)
    return TailExtension(_FAMILY_MULTIPLIERS[family] * float(scale), TAIL_ORIGIN_CURATED)


def _extractor_for(family: str, extractors: Mapping[str, Any], parametrization: str | None) -> Any:
    """The scale extractor of one (family, parametrization) pair.

    A declared parametrization must be in the table (no key-guessing
    around a declared name); a missing parametrization tries the
    family's documented try-order.
    """
    if parametrization is not None:
        return extractors.get(parametrization)
    return _first_scale_extractor(family, extractors)


def _first_scale_extractor(family: str, extractors: Mapping[str, Any]) -> Any:
    """Try-order extractor for a family declared without a parametrization."""
    candidates = tuple(extractors[name] for name in _PARAM_TRY_ORDER[family])

    def extractor(parameters: Mapping[str, Any]) -> float | None:
        for candidate in candidates:
            scale = candidate(parameters)
            if scale is not None:
                return float(scale)
        return None

    return extractor
