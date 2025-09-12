from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from pysatl_core.distributions.registry import _reset_distribution_type_register_for_tests

pytest.importorskip("scipy")


@pytest.fixture(autouse=True)
def _fresh_registry() -> Generator[None, Any, None]:
    _reset_distribution_type_register_for_tests()
    yield
    _reset_distribution_type_register_for_tests()
