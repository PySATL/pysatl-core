from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from pysatl_core.distributions.registry import reset_characteristic_registry
from pysatl_core.families.registry import _reset_families_register_for_tests

pytest.importorskip("scipy")


@pytest.fixture(autouse=True)
def _fresh_registries() -> Generator[None, Any, None]:
    reset_characteristic_registry()
    _reset_families_register_for_tests()
    yield
