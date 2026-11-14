"""The bench-provided ``cf -> pdf`` edge (Gil-Pelaez inversion, forced route).

pysatl-core files are never modified by the benchmark; the fitter lives in
``pysatl_core_bench.accuracy.computations.adapter_pysatl`` and is registered into the
library's fitter registry through its public ``register`` API.
"""

from __future__ import annotations

import numpy as np
import pytest

from pysatl_core.distributions.computations.registry import fitter_registry
from pysatl_core.distributions.distribution import Distribution
from pysatl_core.families.configuration import configure_families_register
from pysatl_core.families.registry import ParametricFamilyRegister
from pysatl_core.types import CharacteristicName, FamilyName, NumericArray
from pysatl_core_bench.accuracy.computations.adapter_pysatl import (
    _build_cf_to_pdf_1C,
    _ensure_cf_to_pdf_edge,
    _fit_cf_to_pdf_1C,
    _ForcedGraphStrategy,
)
from pysatl_core_bench.accuracy.core_state import reset_core_state


def _normal() -> Distribution:
    """Standard normal distribution from the builtin family register."""
    configure_families_register()
    family = ParametricFamilyRegister.get(FamilyName.NORMAL)
    return family.distribution(parametrization_name="meanStd", mu=0.0, sigma=1.0)


def _pdf_ref(x: NumericArray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.exp(-(arr**2) / 2) / np.sqrt(2 * np.pi)


class TestFitCfToPdf1C:
    """Tests for _fit_cf_to_pdf_1C (Gil-Pelaez inversion, trapezoid on [0, T])."""

    def test_normal_pdf_machine_precision(self) -> None:
        """Gil-Pelaez trapezoid reaches machine precision for a smooth density."""
        fitted = _fit_cf_to_pdf_1C(_normal())
        xs = np.linspace(-6.0, 6.0, 65)
        values = np.asarray(fitted(xs), dtype=np.float64)
        np.testing.assert_allclose(values, _pdf_ref(xs), atol=1e-12)

    def test_truncation_option_changes_result(self) -> None:
        """``t_trunc`` is a characteristic option: T=4 truncates, T=40 does not."""
        short = _fit_cf_to_pdf_1C(_normal(), t_trunc=4.0, n_steps=2048)
        full = _fit_cf_to_pdf_1C(_normal())
        xs = np.array([2.0, 4.0])
        ref = _pdf_ref(xs)
        short_vals = np.asarray(short(xs), dtype=np.float64)
        full_vals = np.asarray(full(xs), dtype=np.float64)
        assert np.abs(short_vals - ref).max() > 1e-6
        assert np.abs(full_vals - ref).max() < 1e-12

    def test_non_negative_output(self) -> None:
        """Clipped at zero: the fitted pdf never returns negative values."""
        fitted = _fit_cf_to_pdf_1C(_normal())
        values = np.asarray(fitted(np.linspace(8.0, 30.0, 20)), dtype=np.float64)
        assert np.all(values >= 0.0)

    def test_empty_input(self) -> None:
        """An empty points array yields an empty result, not an error."""
        fitted = _fit_cf_to_pdf_1C(_normal())
        result = np.asarray(fitted(np.array([])), dtype=np.float64)
        assert result.size == 0

    def test_descriptor_metadata(self) -> None:
        """The descriptor exposes the option taxonomy and constraint tags."""
        desc = _build_cf_to_pdf_1C()
        assert desc.name == "cf_to_pdf_1C"
        assert desc.target == CharacteristicName.PDF
        assert desc.sources == [CharacteristicName.CF]
        assert {o.name for o in desc.characteristic_options} == {"t_trunc"}
        assert {o.name for o in desc.computation_options} == {"n_steps"}


class TestRegistration:
    """The edge is installed through the public registry API, idempotently."""

    def test_register_then_find(self) -> None:
        reset_core_state()
        assert fitter_registry().find(CharacteristicName.PDF, [CharacteristicName.CF]) is None
        _ensure_cf_to_pdf_edge()
        descriptor = fitter_registry().find(
            CharacteristicName.PDF,
            [CharacteristicName.CF],
            required_tags=frozenset({"continuous", "univariate"}),
        )
        assert descriptor is not None
        assert descriptor.name == "cf_to_pdf_1C"

    def test_register_is_idempotent(self) -> None:
        reset_core_state()
        _ensure_cf_to_pdf_edge()
        descriptors = fitter_registry().all_descriptors()
        _ensure_cf_to_pdf_edge()
        assert len(fitter_registry().all_descriptors()) == len(descriptors)

    def test_reset_clears_registration(self) -> None:
        _ensure_cf_to_pdf_edge()
        reset_core_state()
        assert fitter_registry().find(CharacteristicName.PDF, [CharacteristicName.CF]) is None

    def test_forced_strategy_serves_pdf_from_cf(self) -> None:
        """End-to-end: ForcedGraphStrategy(from=cf) computes the pdf via the edge."""
        _ensure_cf_to_pdf_edge()
        configure_families_register()
        family = ParametricFamilyRegister.get(FamilyName.NORMAL)
        distr = family.distribution(parametrization_name="meanStd", mu=0.0, sigma=1.0)
        distr = distr.with_computation_strategy(_ForcedGraphStrategy(CharacteristicName.CF))
        plan = distr.explain_computation_path(CharacteristicName.PDF)
        assert str(plan.source) == "cf"
        xs = np.linspace(-4.0, 4.0, 33)
        values = np.asarray(
            distr.calculate_characteristic(CharacteristicName.PDF, xs), dtype=np.float64
        )
        np.testing.assert_allclose(values, _pdf_ref(xs), atol=1e-12)


@pytest.fixture(autouse=True)
def _fresh_core_state() -> None:
    reset_core_state()
