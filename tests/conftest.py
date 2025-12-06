from __future__ import annotations

__author__ = "Leonid Elkin, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Generator
from typing import Any

import pytest

from pysatl_core.distributions.registry import reset_characteristic_registry
from pysatl_core.families.configuration import reset_families_register

pytest.importorskip("scipy")


@pytest.fixture(autouse=True)
def _fresh_registries() -> Generator[None, Any, None]:
    reset_characteristic_registry()
    reset_families_register()
    yield
