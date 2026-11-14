"""Compare and report stages of the accuracy model.

The persistence sink (``ResultsStore``), the comparison stage
(``Comparator``: the group table, invariant-verdict spread inside the
group, pairwise agreement, the sweep curve over ``sort_key``) and the
report stage (``Reporter``: a deterministic text report in which every
status is visible).  Results DTOs live in ``accuracy.dto``; the
orchestrator is ``accuracy.runner``.
"""

from __future__ import annotations

__author__ = "Fedor Myznikov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .compare import Comparator
from .report import Reporter
from .store import ResultsStore

__all__ = ["Comparator", "Reporter", "ResultsStore"]
