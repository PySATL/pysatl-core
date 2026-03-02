"""
Unit tests for pysatl_core.sampling.unuran.core.unuran_sampler

Tests DefaultUnuranSampler:
  - _select_best_method: method selection heuristics
  - _resolve_available_chars: characteristic probing
  - _check_method_suitability: pre-flight validation
  - Integration tests using mocked CFFI bindings
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
)
from pysatl_core.sampling.unuran.core.unuran_sampler import DefaultUnuranSampler
from pysatl_core.sampling.unuran.method_config import UnuranMethod, UnuranMethodConfig
from pysatl_core.types import CharacteristicName, Kind
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


def _continuous_distr(
    chars: list[CharacteristicName] | None = None,
    support: Any = None,
) -> StandaloneEuclideanUnivariateDistribution:
    """Build a minimal continuous distribution."""
    from collections.abc import Callable
    from typing import Any

    from mypy_extensions import KwArg

    chars = chars or [CharacteristicName.PDF]
    acs = [
        AnalyticalComputation[float, float](
            target=c,
            func=cast(Callable[[float, KwArg(Any)], float], lambda x, **_: 1.0),
        )
        for c in chars
    ]
    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=acs,
        support=support or ContinuousSupport(0.0, 1.0),
    )


def _discrete_distr(
    chars: list[CharacteristicName] | None = None,
) -> StandaloneEuclideanUnivariateDistribution:
    """Build a minimal discrete distribution."""
    from collections.abc import Callable
    from typing import Any

    from mypy_extensions import KwArg

    chars = chars or [CharacteristicName.PMF]
    masses = {0: 0.5, 1: 0.5}
    pmf_func = cast(
        Callable[[float, KwArg(Any)], float],
        lambda x, **_: masses.get(int(x), 0.0),
    )
    acs = []
    for c in chars:
        if c == CharacteristicName.PMF:
            acs.append(AnalyticalComputation[float, float](target=c, func=pmf_func))
        else:
            acs.append(
                AnalyticalComputation[float, float](
                    target=c,
                    func=cast(Callable[[float, KwArg(Any)], float], lambda x, **_: 0.5),
                )
            )
    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.DISCRETE,
        analytical_computations=acs,
        support=ExplicitTableDiscreteSupport([0, 1]),
    )


class _Handle:
    """Non-null CFFI-like handle."""


class _NULL:
    """Null sentinel for CFFI comparisons."""


def _mock_cffi_module() -> MagicMock:
    """Return a fake _unuran_cffi module with all required attributes."""
    null = _NULL()
    mock_lib = MagicMock()
    mock_lib.unur_distr_cont_new.return_value = _Handle()
    mock_lib.unur_distr_discr_new.return_value = _Handle()
    mock_lib.unur_pinv_new.return_value = _Handle()
    mock_lib.unur_ninv_new.return_value = _Handle()
    mock_lib.unur_dgt_new.return_value = _Handle()
    mock_lib.unur_init.return_value = _Handle()
    mock_lib.unur_get_errno.return_value = 0
    mock_lib.unur_distr_cont_set_pdf.return_value = 0
    mock_lib.unur_distr_cont_set_dpdf.return_value = 0
    mock_lib.unur_distr_cont_set_cdf.return_value = 0
    mock_lib.unur_distr_cont_set_invcdf.return_value = 0
    mock_lib.unur_distr_discr_set_pmf.return_value = 0
    mock_lib.unur_distr_discr_set_cdf.return_value = 0
    mock_lib.unur_distr_discr_set_domain.return_value = 0
    mock_lib.unur_distr_discr_make_pv.return_value = 5
    mock_lib.unur_distr_cont_set_domain.return_value = 0
    mock_lib.unur_set_default_urng.return_value = None
    mock_lib.unur_set_default_urng_aux.return_value = None
    mock_lib.unur_urng_new.return_value = _Handle()
    mock_lib.unur_sample_cont.return_value = 0.5
    mock_lib.unur_sample_discr.return_value = 1
    mock_lib.unur_free.return_value = None
    mock_lib.unur_distr_free.return_value = None

    mock_ffi = MagicMock()
    mock_ffi.NULL = null
    mock_ffi.callback.side_effect = lambda sig, func: func
    mock_ffi.new_handle.return_value = _Handle()
    mock_ffi.from_handle.return_value = MagicMock(random=MagicMock(return_value=0.5))
    mock_ffi.string.return_value = b"error"

    mock_module = MagicMock()
    mock_module.ffi = mock_ffi
    mock_module.lib = mock_lib
    return mock_module


class TestSelectBestMethod:
    """Tests for DefaultUnuranSampler._select_best_method static method."""

    def test_continuous_ppf_available_selects_pinv(self) -> None:
        """PPF available for continuous dist → PINV selected (fastest inversion)."""
        chars = {CharacteristicName.PPF, CharacteristicName.PDF}
        config = UnuranMethodConfig(use_ppf=True)

        result = DefaultUnuranSampler._select_best_method(chars, Kind.CONTINUOUS, config)

        assert result == UnuranMethod.PINV

    def test_continuous_pdf_only_selects_pinv(self) -> None:
        """PDF available without PPF → PINV (works with PDF alone)."""
        chars = {CharacteristicName.PDF}
        config = UnuranMethodConfig(use_ppf=False, use_pdf=True)

        result = DefaultUnuranSampler._select_best_method(chars, Kind.CONTINUOUS, config)

        assert result == UnuranMethod.PINV

    def test_continuous_cdf_only_selects_ninv(self) -> None:
        """CDF only, no PDF → NINV (numerical inversion)."""
        chars = {CharacteristicName.CDF}
        config = UnuranMethodConfig(use_ppf=False, use_pdf=False, use_cdf=True)

        result = DefaultUnuranSampler._select_best_method(chars, Kind.CONTINUOUS, config)

        assert result == UnuranMethod.NINV

    def test_continuous_no_chars_raises_runtime_error(self) -> None:
        """No usable characteristics for continuous distribution → RuntimeError."""
        chars: set[CharacteristicName] = set()
        config = UnuranMethodConfig(use_ppf=False, use_pdf=False, use_cdf=False)

        with pytest.raises(RuntimeError, match="No suitable method"):
            DefaultUnuranSampler._select_best_method(chars, Kind.CONTINUOUS, config)

    def test_discrete_pmf_available_selects_dgt(self) -> None:
        """PMF available for discrete dist → DGT."""
        chars = {CharacteristicName.PMF}
        config = UnuranMethodConfig()

        result = DefaultUnuranSampler._select_best_method(chars, Kind.DISCRETE, config)

        assert result == UnuranMethod.DGT

    def test_discrete_no_pmf_raises_runtime_error(self) -> None:
        """Discrete distribution with no PMF → RuntimeError."""
        chars: set[CharacteristicName] = set()
        config = UnuranMethodConfig()

        with pytest.raises(RuntimeError, match="PMF"):
            DefaultUnuranSampler._select_best_method(chars, Kind.DISCRETE, config)

    def test_unknown_kind_raises_runtime_error(self) -> None:
        """An unsupported distribution kind raises RuntimeError."""
        config = UnuranMethodConfig()

        with pytest.raises(RuntimeError):
            DefaultUnuranSampler._select_best_method(set(), "unknown_kind", config)  # type: ignore[arg-type]

    def test_ppf_skipped_when_use_ppf_false(self) -> None:
        """With use_ppf=False, PDF is preferred over PPF for continuous distributions."""
        chars = {CharacteristicName.PPF, CharacteristicName.PDF}
        config = UnuranMethodConfig(use_ppf=False, use_pdf=True)

        result = DefaultUnuranSampler._select_best_method(chars, Kind.CONTINUOUS, config)

        # PINV is selected because PDF is available and use_pdf=True
        assert result == UnuranMethod.PINV


class TestResolveAvailableChars:
    """Tests for DefaultUnuranSampler._resolve_available_chars static method."""

    def test_returns_chars_present_in_analytical_computations(self) -> None:
        """Without registry, only characteristics in analytical_computations are returned."""
        distr = _continuous_distr(chars=[CharacteristicName.PDF, CharacteristicName.CDF])

        result = DefaultUnuranSampler._resolve_available_chars(
            distr, use_registry_characteristics=False
        )

        assert CharacteristicName.PDF in result
        assert CharacteristicName.CDF in result

    def test_returns_empty_set_when_no_analytical_computations(self) -> None:
        """A distribution with no analytical computations returns an empty set."""
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS, analytical_computations=[], support=None
        )

        result = DefaultUnuranSampler._resolve_available_chars(
            distr, use_registry_characteristics=False
        )

        assert result == set()

    def test_uses_query_method_when_registry_chars_enabled(self) -> None:
        """With use_registry_characteristics=True, query_method is used for probing."""
        mock_distr = MagicMock()
        mock_distr.analytical_computations = {}

        def query_method(char):
            if char == CharacteristicName.PDF:
                return MagicMock()
            raise RuntimeError("not available")

        mock_distr.query_method = query_method

        result = DefaultUnuranSampler._resolve_available_chars(
            mock_distr, use_registry_characteristics=True
        )

        assert CharacteristicName.PDF in result
        assert CharacteristicName.CDF not in result

    def test_runtime_error_from_query_method_is_silently_skipped(self) -> None:
        """Characteristics that raise RuntimeError from query_method are excluded."""
        mock_distr = MagicMock()
        mock_distr.analytical_computations = {}
        mock_distr.query_method.side_effect = RuntimeError("not available")

        result = DefaultUnuranSampler._resolve_available_chars(
            mock_distr, use_registry_characteristics=True
        )

        assert result == set()


class TestCheckMethodSuitability:
    """Tests for DefaultUnuranSampler._check_method_suitability."""

    def test_raises_for_unsupported_method(self) -> None:
        """An unsupported method name raises RuntimeError before CFFI is used."""
        mock_cffi = _mock_cffi_module()
        _continuous_distr()

        with (
            patch("pysatl_core.sampling.unuran.bindings._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._DEFAULT_URNG", _Handle()),
        ):
            config = UnuranMethodConfig(method=UnuranMethod.AUTO)
            # Manually set method to unsupported value by building sampler normally
            # then testing the private method
            sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
            sampler._method = "unsupported"  # type: ignore[assignment]
            sampler._config = config
            sampler.available_chars = {CharacteristicName.PDF}

            with pytest.raises(RuntimeError, match="Unsupported"):
                sampler._check_method_suitability()

    def test_raises_when_required_chars_missing(self) -> None:
        """Raises RuntimeError when the chosen method's required chars are absent."""
        _mock_cffi_module()
        _continuous_distr(chars=[CharacteristicName.PDF])

        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._method = UnuranMethod.AROU  # AROU requires PDF + DPDF
        sampler._config = UnuranMethodConfig()
        sampler.available_chars = {CharacteristicName.PDF}  # DPDF is missing

        with pytest.raises(RuntimeError, match="requires"):
            sampler._check_method_suitability()

    def test_passes_when_all_required_chars_present(self) -> None:
        """No error is raised when all required characteristics are available."""
        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._method = UnuranMethod.PINV
        sampler._config = UnuranMethodConfig()
        sampler.available_chars = {CharacteristicName.PDF}

        # Should not raise
        sampler._check_method_suitability()


class TestDefaultUnuranSamplerInit:
    """Integration tests for DefaultUnuranSampler.__init__ with mocked CFFI."""

    def test_raises_when_cffi_not_available(self) -> None:
        """Raises RuntimeError when _unuran_cffi is None."""
        distr = _continuous_distr()

        with (
            patch("pysatl_core.sampling.unuran.bindings._unuran_cffi", None),
            pytest.raises(RuntimeError, match="UNURAN CFFI bindings"),
        ):
            DefaultUnuranSampler(distr)

    def test_raises_for_non_euclidean_distribution(self) -> None:
        """Raises RuntimeError for distributions that aren't EuclideanDistributionType."""
        mock_cffi = _mock_cffi_module()
        mock_distr = MagicMock()
        mock_distr.distribution_type = "not euclidean"

        with (
            patch("pysatl_core.sampling.unuran.bindings._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._DEFAULT_URNG", _Handle()),
            pytest.raises(RuntimeError, match="Unsupported distribution type"),
        ):
            DefaultUnuranSampler(mock_distr)

    def test_raises_for_multivariate_distribution(self) -> None:
        """Raises RuntimeError for distributions with dimension != 1."""
        from pysatl_core.types import EuclideanDistributionType

        mock_cffi = _mock_cffi_module()
        mock_distr = MagicMock()
        mock_distr.distribution_type = EuclideanDistributionType(Kind.CONTINUOUS, 2)

        with (
            patch("pysatl_core.sampling.unuran.bindings._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._DEFAULT_URNG", _Handle()),
            pytest.raises(RuntimeError, match="dimension"),
        ):
            DefaultUnuranSampler(mock_distr)

    def test_method_property_returns_selected_method(self) -> None:
        """The method property reflects the method chosen during initialization."""
        mock_cffi = _mock_cffi_module()
        distr = _continuous_distr(chars=[CharacteristicName.PDF])
        config = UnuranMethodConfig(method=UnuranMethod.PINV)

        with (
            patch("pysatl_core.sampling.unuran.bindings._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._DEFAULT_URNG", _Handle()),
        ):
            sampler = DefaultUnuranSampler(distr, config)

        assert sampler.method == UnuranMethod.PINV

    def test_fallback_sampler_used_when_init_fails_and_fallback_enabled(self) -> None:
        """Fallback to DefaultSamplingUnivariateStrategy on init failure when use_fallback=True."""
        mock_cffi = _mock_cffi_module()
        # Make unur_init return NULL to force failure
        mock_cffi.lib.unur_init.return_value = mock_cffi.ffi.NULL
        distr = _continuous_distr(chars=[CharacteristicName.PDF])
        config = UnuranMethodConfig(method=UnuranMethod.PINV, use_fallback_sampler=True)

        with (
            patch("pysatl_core.sampling.unuran.bindings._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._DEFAULT_URNG", _Handle()),
        ):
            sampler = DefaultUnuranSampler(distr, config)

        assert sampler._fallback_sampling_method is not None

    def test_no_fallback_re_raises_when_fallback_disabled(self) -> None:
        """RuntimeError is re-raised when use_fallback_sampler=False and init fails."""
        mock_cffi = _mock_cffi_module()
        mock_cffi.lib.unur_init.return_value = mock_cffi.ffi.NULL
        distr = _continuous_distr(chars=[CharacteristicName.PDF])
        config = UnuranMethodConfig(method=UnuranMethod.PINV, use_fallback_sampler=False)

        with (
            patch("pysatl_core.sampling.unuran.bindings._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._unuran_cffi", mock_cffi),
            patch("pysatl_core.sampling.unuran.core._unuran_sampler.urng._DEFAULT_URNG", _Handle()),
            pytest.raises(RuntimeError),
        ):
            DefaultUnuranSampler(distr, config)


class TestDefaultUnuranSamplerSample:
    """Tests for DefaultUnuranSampler.sample."""

    def test_negative_n_raises_value_error(self) -> None:
        """Requesting a negative number of samples raises ValueError."""
        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._fallback_sampling_method = None
        sampler._unuran_gen = _Handle()
        sampler._lib = MagicMock()
        sampler._kind = Kind.CONTINUOUS

        with pytest.raises(ValueError, match="non-negative"):
            sampler.sample(-1)

    def test_zero_samples_returns_empty_array(self) -> None:
        """Requesting 0 samples returns a zero-length float64 array."""
        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._fallback_sampling_method = None
        sampler._unuran_gen = _Handle()
        sampler._lib = MagicMock()
        sampler._kind = Kind.CONTINUOUS

        result = sampler.sample(0)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.dtype == np.float64

    def test_uses_fallback_strategy_when_set(self) -> None:
        """When a fallback method is active, sample delegates to it."""
        mock_fallback = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._fallback_sampling_method = mock_fallback

        sampler.sample(3)

        mock_fallback.assert_called_once_with(3)

    def test_raises_when_generator_is_none_and_no_fallback(self) -> None:
        """RuntimeError is raised when neither a generator nor a fallback is available."""
        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._fallback_sampling_method = None
        sampler._unuran_gen = None
        sampler._lib = None
        sampler._kind = Kind.CONTINUOUS

        with pytest.raises(RuntimeError, match="not initialized"):
            sampler.sample(5)

    def test_continuous_sampling_calls_unur_sample_cont(self) -> None:
        """Continuous sampling uses unur_sample_cont in a loop."""
        mock_lib = MagicMock()
        mock_lib.unur_sample_cont.return_value = 0.42

        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._fallback_sampling_method = None
        sampler._unuran_gen = _Handle()
        sampler._lib = mock_lib
        sampler._kind = Kind.CONTINUOUS

        result = sampler.sample(3)

        assert mock_lib.unur_sample_cont.call_count == 3
        assert len(result) == 3

    def test_discrete_sampling_calls_unur_sample_discr(self) -> None:
        """Discrete sampling uses unur_sample_discr and casts to float."""
        mock_lib = MagicMock()
        mock_lib.unur_sample_discr.return_value = 2

        sampler = DefaultUnuranSampler.__new__(DefaultUnuranSampler)
        sampler._fallback_sampling_method = None
        sampler._unuran_gen = _Handle()
        sampler._lib = mock_lib
        sampler._kind = Kind.DISCRETE
        sampler._index_remap_points = None

        result = sampler.sample(4)

        assert mock_lib.unur_sample_discr.call_count == 4
        assert result.dtype == np.float64
        assert np.all(result == 2.0)
