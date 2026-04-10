"""
Unit tests for pysatl_core.sampling.unuran.core._unuran_sampler.initialization

Tests the UnuranSamplerInitializer class that orchestrates UNU.RAN distribution
object creation, callback setup, and generator initialization.
"""

from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import (
    ContinuousSupport,
    ExplicitTableDiscreteSupport,
)
from pysatl_core.sampling.unuran.core._unuran_sampler.initialization import (
    UnuranSamplerInitializer,
)
from pysatl_core.sampling.unuran.method_config import UnuranMethod
from pysatl_core.types import CharacteristicName, Kind
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


class _Handle:
    """Non-null CFFI-like handle for testing (identity comparison only)."""


class _NULL:
    """Null sentinel for CFFI-like comparisons."""


def _build_mock_cffi() -> tuple[MagicMock, MagicMock, _NULL]:
    """Return (mock_lib, mock_ffi, null_sentinel) with sensible defaults."""
    null = _NULL()

    mock_lib = MagicMock()
    mock_lib.unur_distr_cont_new.return_value = _Handle()
    mock_lib.unur_distr_discr_new.return_value = _Handle()
    mock_lib.unur_pinv_new.return_value = _Handle()
    mock_lib.unur_ninv_new.return_value = _Handle()
    mock_lib.unur_dgt_new.return_value = _Handle()
    mock_lib.unur_arou_new.return_value = _Handle()
    mock_lib.unur_tdr_new.return_value = _Handle()
    mock_lib.unur_hinv_new.return_value = _Handle()
    mock_lib.unur_init.return_value = _Handle()
    mock_lib.unur_get_errno.return_value = 0
    mock_lib.unur_distr_cont_set_domain.return_value = 0
    mock_lib.unur_distr_cont_set_pdf.return_value = 0
    mock_lib.unur_distr_cont_set_cdf.return_value = 0
    mock_lib.unur_distr_cont_set_invcdf.return_value = 0

    mock_ffi = MagicMock()
    mock_ffi.NULL = null
    mock_ffi.callback.side_effect = lambda sig, func: func
    mock_ffi.string.return_value = b"error text"

    return mock_lib, mock_ffi, null


def _make_continuous_distr(
    chars: list[CharacteristicName] | None = None,
    support: Any = None,
) -> StandaloneEuclideanUnivariateDistribution:
    """Create a continuous distribution with optional characteristics and support."""
    chars = chars or [CharacteristicName.PDF]
    from collections.abc import Callable
    from typing import Any

    from mypy_extensions import KwArg

    acs = []
    for char in chars:
        func = cast(Callable[[float, KwArg(Any)], float], lambda x, **_: 1.0)
        acs.append(AnalyticalComputation[float, float](target=char, func=func))

    return StandaloneEuclideanUnivariateDistribution(
        kind=Kind.CONTINUOUS,
        analytical_computations=acs,
        support=support or ContinuousSupport(0.0, 1.0),
    )


class TestUnuranSamplerInitializerInit:
    """Tests for UnuranSamplerInitializer.__init__ validation."""

    def test_raises_for_non_euclidean_distribution_type(self) -> None:
        """Initializer raises RuntimeError for non-Euclidean distribution types."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        mock_distr = MagicMock()
        mock_distr.distribution_type = "not euclidean"

        with pytest.raises(RuntimeError, match="Unsupported distribution type"):
            UnuranSamplerInitializer(mock_distr, UnuranMethod.PINV, mock_lib, mock_ffi)

    def test_stores_method_lib_ffi_on_valid_input(self) -> None:
        """Initializer correctly stores method, lib, and ffi for valid distributions."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr()

        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        assert init._method == UnuranMethod.PINV
        assert init._lib is mock_lib
        assert init._ffi is mock_ffi

    def test_kind_is_set_from_distribution_type(self) -> None:
        """The _kind attribute is taken from the distribution's distribution_type.kind."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr()

        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        assert init._kind == Kind.CONTINUOUS

    def test_domain_helper_is_created(self) -> None:
        """An UnuranDomain helper is created and stored during initialisation."""
        from pysatl_core.sampling.unuran.core._unuran_sampler.domain import UnuranDomain

        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr()

        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        assert isinstance(init._domain, UnuranDomain)


class TestRequiresFiniteSupport:
    """Tests for UnuranSamplerInitializer._requires_finite_support."""

    def test_pinv_does_not_require_finite_support(self) -> None:
        """PINV method does not mandate a bounded support."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr()
        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        assert init._requires_finite_support() is False

    def test_hinv_requires_finite_support(self) -> None:
        """HINV method mandates a bounded support."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr(chars=[CharacteristicName.PPF])
        init = UnuranSamplerInitializer(distr, UnuranMethod.HINV, mock_lib, mock_ffi)

        assert init._requires_finite_support() is True

    def test_dgt_requires_finite_support(self) -> None:
        """DGT method mandates a bounded support."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        support = ExplicitTableDiscreteSupport([0, 1, 2])

        from collections.abc import Callable
        from typing import Any

        from mypy_extensions import KwArg

        pmf_func = cast(Callable[[float, KwArg(Any)], float], lambda x, **_: 0.33)
        discrete_distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations=[
                AnalyticalComputation[float, float](target=CharacteristicName.PMF, func=pmf_func)
            ],
            support=support,
        )
        init = UnuranSamplerInitializer(discrete_distr, UnuranMethod.DGT, mock_lib, mock_ffi)

        assert init._requires_finite_support() is True


class TestApplyContinuousDomainConstraints:
    """Tests for _apply_continuous_domain_constraints."""

    def test_skipped_for_discrete_kind(self) -> None:
        """Continuous domain constraints are not applied to discrete distributions."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        support = ExplicitTableDiscreteSupport([0, 1])

        from collections.abc import Callable
        from typing import Any

        from mypy_extensions import KwArg

        pmf_func = cast(Callable[[float, KwArg(Any)], float], lambda x, **_: 0.5)
        distr = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.DISCRETE,
            analytical_computations=[
                AnalyticalComputation[float, float](target=CharacteristicName.PMF, func=pmf_func)
            ],
            support=support,
        )
        init = UnuranSamplerInitializer(distr, UnuranMethod.DGT, mock_lib, mock_ffi)

        # Should not raise even though unuran_distr is None
        init._apply_continuous_domain_constraints()

        mock_lib.unur_distr_cont_set_domain.assert_not_called()

    def test_skipped_when_method_does_not_require_support(self) -> None:
        """Domain constraints are skipped for methods that don't need bounded support."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr(support=ContinuousSupport(0.0, 1.0))
        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        init._apply_continuous_domain_constraints()

        mock_lib.unur_distr_cont_set_domain.assert_not_called()

    def test_raises_when_support_missing_for_requiring_method(self) -> None:
        """Raises RuntimeError when HINV method is used but support is missing."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr(
            chars=[CharacteristicName.PPF],
            support=ContinuousSupport(),  # unbounded, returns (-inf, inf)
        )
        init = UnuranSamplerInitializer(distr, UnuranMethod.HINV, mock_lib, mock_ffi)

        with pytest.raises(RuntimeError, match="finite support"):
            init._apply_continuous_domain_constraints()

    def test_raises_when_set_domain_fails(self) -> None:
        """Raises RuntimeError when unur_distr_cont_set_domain returns non-zero."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        mock_lib.unur_distr_cont_set_domain.return_value = -5
        distr = _make_continuous_distr(
            chars=[CharacteristicName.PPF],
            support=ContinuousSupport(0.0, 1.0),
        )
        init = UnuranSamplerInitializer(distr, UnuranMethod.HINV, mock_lib, mock_ffi)
        init._unuran_distr = _Handle()

        with pytest.raises(RuntimeError, match="domain"):
            init._apply_continuous_domain_constraints()

    def test_raises_when_left_equals_right(self) -> None:
        """Raises RuntimeError when the support has zero width (left == right)."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr(
            chars=[CharacteristicName.PPF],
            support=ContinuousSupport(1.0, 1.0),
        )
        init = UnuranSamplerInitializer(distr, UnuranMethod.HINV, mock_lib, mock_ffi)
        init._unuran_distr = _Handle()

        with pytest.raises(RuntimeError, match="bounds"):
            init._apply_continuous_domain_constraints()


class TestCleanup:
    """Tests for UnuranSamplerInitializer.cleanup."""

    def test_cleanup_frees_generator_when_initialized(self) -> None:
        """cleanup() calls unur_free on the generator handle when it was created."""
        mock_lib, mock_ffi, null = _build_mock_cffi()
        distr = _make_continuous_distr()
        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        gen_handle = _Handle()
        init._unuran_gen = gen_handle

        init.cleanup()

        mock_lib.unur_free.assert_called_once_with(gen_handle)
        assert init._unuran_gen is None

    def test_cleanup_frees_distribution_when_no_generator(self) -> None:
        """cleanup() calls unur_distr_free when there is no generator to free."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr()
        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        distr_handle = _Handle()
        init._unuran_gen = None
        init._unuran_distr = distr_handle

        init.cleanup()

        mock_lib.unur_distr_free.assert_called_once_with(distr_handle)
        assert init._unuran_distr is None

    def test_cleanup_sets_all_handles_to_none(self) -> None:
        """After cleanup, all three handles are None."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr()
        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        init._unuran_gen = _Handle()
        init._unuran_par = _Handle()
        init._unuran_distr = _Handle()

        init.cleanup()

        assert init._unuran_gen is None
        assert init._unuran_par is None
        assert init._unuran_distr is None

    def test_cleanup_is_idempotent(self) -> None:
        """Calling cleanup twice does not raise an error."""
        mock_lib, mock_ffi, _ = _build_mock_cffi()
        distr = _make_continuous_distr()
        init = UnuranSamplerInitializer(distr, UnuranMethod.PINV, mock_lib, mock_ffi)

        init._unuran_gen = _Handle()

        init.cleanup()
        init.cleanup()  # second call should be a no-op
