"""
Golden test: fixed samples, fixed expected values, checked in.

The rest of this suite asserts that pysatl-core agrees with SciPy *today*.
That is a useful property and a fragile one: it holds just as well if both
sides drift together, and it says nothing at all on a machine where SciPy is
a different version.  This test pins the answer instead.  The samples are
stored in ``data/golden_mle.json`` rather than redrawn from a seed, because a
seed pins the random stream and not the sample; and every expected value in
that file was computed from SciPy or from a closed-form expression at
generation time, never read back from pysatl-core, so the file records what
the answer is rather than what this library printed once.

Regenerate with ``python experiments/mle_validation/make_golden.py``.  A diff
in that file is a change in the library's numerical behaviour and has to be
explained, not accepted.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pysatl_core.families.configuration import configure_families_register
from pysatl_core.types import FamilyName

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_mle.json"

FAMILY_NAMES = {
    "Normal": FamilyName.NORMAL,
    "Exponential": FamilyName.EXPONENTIAL,
    "ContinuousUniform": FamilyName.CONTINUOUS_UNIFORM,
    "Gamma": FamilyName.GAMMA,
}


def _load() -> dict[str, Any]:
    """Read the golden file."""
    return json.loads(GOLDEN_PATH.read_text())


GOLDEN = _load()
CASES = GOLDEN["cases"]
CASE_IDS = [case["name"] for case in CASES]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def case(request: pytest.FixtureRequest) -> dict[str, Any]:
    """One golden case."""
    return request.param


def _fit(case: dict[str, Any]) -> Any:
    """Run the fit the case describes, through a view if it pins anything."""
    family = configure_families_register().get(FAMILY_NAMES[case["family"]])
    sample = np.asarray(case["sample"], dtype=float)
    if case["fixed"]:
        family = family.view(**case["fixed"])
    return family.fit(sample)


class TestGoldenMLE:
    """Every fit reproduces the stored reference."""

    def test_sample_is_intact(self, case: dict[str, Any]) -> None:
        # The samples are the input the expected values were computed from.
        # If the file is ever reformatted by a tool that rounds floats, every
        # other assertion here would start failing for a reason that has
        # nothing to do with the library, so the length and the extremes are
        # checked first and the failure says so.
        sample = np.asarray(case["sample"], dtype=float)
        assert sample.shape == (GOLDEN["n"],)
        assert np.isfinite(sample).all()

    def test_parameters(self, case: dict[str, Any]) -> None:
        result = _fit(case)
        tolerance = case["rel_tolerance"]
        for name, expected in case["expected"].items():
            actual = getattr(result.params, name)
            assert actual == pytest.approx(expected, rel=tolerance), (
                f"{case['name']}: parameter '{name}' drifted from the reference "
                f"({case['reference']})"
            )

    def test_log_likelihood(self, case: dict[str, Any]) -> None:
        # The reference log-likelihood is SciPy's ``logpdf`` summed over the
        # stored sample at the stored estimate.  It checks the density, which
        # the parameter assertions do not: an ``lpdf`` off by a constant would
        # leave every estimate correct and every criterion wrong.
        result = _fit(case)
        assert result.log_likelihood == pytest.approx(
            case["log_likelihood"], rel=max(case["rel_tolerance"], 1e-9)
        )

    def test_information_criteria(self, case: dict[str, Any]) -> None:
        result = _fit(case)
        assert result.aic == pytest.approx(case["aic"], rel=1e-9)
        assert result.bic == pytest.approx(case["bic"], rel=1e-9)

    def test_route_and_shape(self, case: dict[str, Any]) -> None:
        # Which branch produced the estimate is part of the pinned behaviour:
        # a family quietly losing its closed form would still pass every
        # numerical assertion above while becoming an order of magnitude
        # slower.
        result = _fit(case)
        assert result.route == case["route"]
        assert result.n_params == case["n_params"]
        assert result.n_observations == GOLDEN["n"]
        assert result.success

    def test_reproducible_across_calls(self, case: dict[str, Any]) -> None:
        # Two fits of the same sample must agree bit for bit. An optimizer
        # seeded from anything ambient -- the clock, a hash, an uninitialised
        # buffer -- would show up here and nowhere else in this suite.
        first, second = _fit(case), _fit(case)
        for name in case["expected"]:
            assert getattr(first.params, name) == getattr(second.params, name)
        assert first.log_likelihood == second.log_likelihood
