from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pysatl_core.stats._unuran.bindings._core.unuran_sampler import DefaultUnuranSampler
from pysatl_core.types import CharacteristicName


class TestDistributionSetup:
    """Tests for distribution object creation, callbacks, domains and DGT setup."""

    def test_create_unuran_distribution_continuous(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Continuous distributions should produce a UNURAN distribution via cont constructor."""
        sampler_stub._is_continuous = True
        created = []

        def cont_new() -> str:
            created.append("cont")
            return "CONT"

        sampler_stub._lib.unur_distr_cont_new = cont_new
        sampler_stub._create_unuran_distribution()
        assert sampler_stub._unuran_distr == "CONT"

    def test_create_unuran_distribution_failure_raises(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Creation should raise if UNURAN factory returns NULL."""
        sampler_stub._is_continuous = False
        sampler_stub._ffi = SimpleNamespace(NULL=None)
        sampler_stub._lib.unur_distr_discr_new = lambda: None
        with pytest.raises(RuntimeError, match="Failed to create UNURAN distribution"):
            sampler_stub._create_unuran_distribution()

    def test_setup_continuous_callbacks_registers_pdf_and_cdf(self) -> None:
        """Continuous callback setup should register PDF and CDF when provided."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._callbacks = []
        sampler._unuran_distr = "DIST"
        sampler._is_continuous = True
        sampler._lib = SimpleNamespace(
            unur_distr_cont_set_pdf=lambda dist, cb: 0,
            unur_distr_cont_set_dpdf=lambda dist, cb: 0,
            unur_distr_cont_set_cdf=lambda dist, cb: 0,
        )

        with (
            patch.object(DefaultUnuranSampler, "_create_pdf_callback", autospec=True) as mock_pdf,
            patch.object(DefaultUnuranSampler, "_create_dpdf_callback", autospec=True) as mock_dpdf,
            patch.object(DefaultUnuranSampler, "_create_cdf_callback", autospec=True) as mock_cdf,
            patch.object(DefaultUnuranSampler, "_create_ppf_callback", autospec=True) as mock_ppf,
        ):
            mock_pdf.return_value = "PDF"
            mock_dpdf.return_value = None
            mock_cdf.return_value = "CDF"
            mock_ppf.return_value = None

            sampler._setup_continuous_callbacks()

        mock_pdf.assert_called_once_with(sampler)
        mock_cdf.assert_called_once_with(sampler)
        assert sampler._callbacks == ["PDF", "CDF"]

    def test_setup_discrete_callbacks_registers_pmf_and_cdf(self) -> None:
        """Discrete callback setup should register PMF and CDF."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._callbacks = []
        sampler._unuran_distr = "DIST"
        sampler._is_continuous = False
        sampler._lib = SimpleNamespace(
            unur_distr_discr_set_pmf=lambda dist, cb: 0,
            unur_distr_discr_set_cdf=lambda dist, cb: 0,
        )

        with (
            patch.object(DefaultUnuranSampler, "_create_pmf_callback", autospec=True) as mock_pmf,
            patch.object(DefaultUnuranSampler, "_create_cdf_callback", autospec=True) as mock_cdf,
        ):
            mock_pmf.return_value = "PMF"
            mock_cdf.return_value = "CDF"

            sampler._setup_discrete_callbacks()

        mock_pmf.assert_called_once_with(sampler)
        mock_cdf.assert_called_once_with(sampler)
        assert sampler._callbacks == ["PMF", "CDF"]

    def test_determine_domain_from_support_explicit_table(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Explicit discrete support should map to (min,max) integer domain."""
        from pysatl_core.distributions.support import ExplicitTableDiscreteSupport

        support = ExplicitTableDiscreteSupport([1, 2, 3])
        sampler_stub.distr = SimpleNamespace(support=support)
        assert sampler_stub._determine_domain_from_support() == (1, 3)

    def test_determine_domain_from_support_custom_methods(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Custom support with first/last methods should be used."""
        support = SimpleNamespace(first=lambda: 5, last=lambda: 8)
        sampler_stub.distr = SimpleNamespace(support=support)
        assert sampler_stub._determine_domain_from_support() == (5, 8)

    def test_determine_domain_from_pmf_without_left_bound(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Domain should expand via PMF evaluation when support absent."""
        counts = {0: 0.5, 1: 0.4, 2: 0.1}

        def pmf(x: float) -> float:
            return counts.get(int(x), 0.0)

        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.PMF: pmf},
        )
        left, right = sampler_stub._determine_domain_from_pmf()
        assert left <= 0 <= right and right >= 1

    def test_determine_domain_from_pmf_with_left_bound(
        self, sampler_stub: DefaultUnuranSampler
    ) -> None:
        """Providing left boundary should seed the PMF scan."""
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.PMF: lambda x: 0.5 if x in (5, 6) else 0.0},
        )
        assert sampler_stub._determine_domain_from_pmf(5) == (5, 6)

    def test_calculate_pmf_sum(self, sampler_stub: DefaultUnuranSampler) -> None:
        """PMF sum should aggregate values across inclusive domain."""
        sampler_stub.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.PMF: lambda x: 0.1 * (x + 1)},
        )
        total = sampler_stub._calculate_pmf_sum(0, 2)
        assert pytest.approx(total, rel=1e-9) == 0.6

    def test_setup_dgt_method_configures_domain_and_pv(self) -> None:
        """DGT setup should set domain, pmf sum, and build PV without errors."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._ffi = SimpleNamespace(NULL=None)
        sampler._unuran_distr = "DIST"
        sampler.distr = SimpleNamespace(
            analytical_computations={CharacteristicName.PMF: lambda x: 0.5 if x in (0, 1) else 0.0},
            support=SimpleNamespace(first=lambda: 0, last=lambda: 1),
        )

        recorded: dict[str, object] = {}

        def set_domain(distr: object, left: int, right: int) -> int:
            recorded["domain"] = (left, right)
            return 0

        def set_pmfsum(distr: object, value: float) -> None:
            recorded["pmfsum"] = value

        def make_pv(distr: object) -> int:
            recorded["make_pv"] = True
            return 2

        sampler._lib = SimpleNamespace(
            unur_distr_discr_set_domain=set_domain,
            unur_distr_discr_set_pmfsum=set_pmfsum,
            unur_distr_discr_make_pv=make_pv,
        )

        sampler._setup_dgt_method()
        assert recorded["domain"] == (0, 1)
        assert pytest.approx(recorded["pmfsum"], rel=1e-9) == 1.0
        assert recorded.get("make_pv") is True
