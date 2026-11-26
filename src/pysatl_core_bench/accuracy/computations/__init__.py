"""Compute stage of the accuracy model: value sources, scales, metrics.

Everything one can ask *numbers* from lives here: the ``ValueSource``
protocol, the source registry, the pysatl-core adapter (the library
boundary), reference sources (precomputed tables, external library
functions, on-demand caching), the output-scale converter and the per-region
error metrics.  Scenario grammar and comparison/report stages belong to
the sibling subpackages ``scenarios`` and ``reporting``; the orchestrator
is ``accuracy.runner``.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .adapter_pysatl import PySATLAdapter
from .metrics import LRE_CAP, MetricsEvaluator
from .output_scale import OutputScaleConverter
from .protocol import PreparedSubject, ValueSource
from .registry import SourceRegistry
from .sources import (
    ExternalFunction,
    ExternalFunctionSource,
    OnDemandComputer,
    PrecomputedTable,
    TableLookupSource,
)

__all__ = [
    "ExternalFunction",
    "PreparedSubject",
    "ExternalFunctionSource",
    "LRE_CAP",
    "MetricsEvaluator",
    "OnDemandComputer",
    "PrecomputedTable",
    "PySATLAdapter",
    "SourceRegistry",
    "OutputScaleConverter",
    "TableLookupSource",
    "ValueSource",
]
