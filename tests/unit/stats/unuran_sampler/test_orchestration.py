from __future__ import annotations

__author__ = "Artem Romanyuk"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pysatl_core.stats._unuran.api import UnuranMethod
from pysatl_core.stats._unuran.bindings._core.unuran_sampler import DefaultUnuranSampler


class TestOrchestration:
    """Tests covering generator creation, initialization, and parameter mapping."""

    def test_create_and_init_generator_success(self) -> None:
        """_create_and_init_generator should fill _unuran_par/_unuran_gen on success."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._ffi = SimpleNamespace(NULL=None)
        sampler._lib = SimpleNamespace(unur_init=lambda par: "GEN")
        sampler._unuran_distr = "DIST"
        sampler._method = UnuranMethod.PINV

        with patch.object(
            DefaultUnuranSampler, "_create_parameter_object", autospec=True
        ) as mock_create_par:
            mock_create_par.return_value = "PAR"
            sampler._create_and_init_generator()

        mock_create_par.assert_called_once_with(sampler)
        assert sampler._unuran_par == "PAR"
        assert sampler._unuran_gen == "GEN"

    def test_initialize_unuran_calls_components(self) -> None:
        """Continuous initialization should run distribution, callbacks, and generator setup."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._is_continuous = True
        sequence: list[str] = []

        sampler._method = UnuranMethod.PINV

        with (
            patch.object(
                DefaultUnuranSampler, "_create_unuran_distribution", autospec=True
            ) as mock_create_distr,
            patch.object(
                DefaultUnuranSampler, "_setup_continuous_callbacks", autospec=True
            ) as mock_setup_cont,
            patch.object(
                DefaultUnuranSampler, "_create_and_init_generator", autospec=True
            ) as mock_create_gen,
        ):
            mock_create_distr.side_effect = lambda self: sequence.append("distr")
            mock_setup_cont.side_effect = lambda self: sequence.append("callbacks")
            mock_create_gen.side_effect = lambda self: sequence.append("generator")

            sampler._initialize_unuran(seed=None)

        mock_create_distr.assert_called_once_with(sampler)
        mock_setup_cont.assert_called_once_with(sampler)
        mock_create_gen.assert_called_once_with(sampler)
        assert sequence == ["distr", "callbacks", "generator"]

    def test_initialize_unuran_discrete_calls_dgt(self) -> None:
        """Discrete initialization should run DGT setup after callbacks."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._is_continuous = False
        sampler._method = UnuranMethod.DGT
        sequence: list[str] = []

        with (
            patch.object(
                DefaultUnuranSampler, "_create_unuran_distribution", autospec=True
            ) as mock_create_distr,
            patch.object(
                DefaultUnuranSampler, "_setup_discrete_callbacks", autospec=True
            ) as mock_setup_discrete,
            patch.object(DefaultUnuranSampler, "_setup_dgt_method", autospec=True) as mock_dgt,
            patch.object(
                DefaultUnuranSampler, "_create_and_init_generator", autospec=True
            ) as mock_create_gen,
        ):
            mock_create_distr.side_effect = lambda self: sequence.append("distr")
            mock_setup_discrete.side_effect = lambda self: sequence.append("discrete")
            mock_dgt.side_effect = lambda self: sequence.append("dgt")
            mock_create_gen.side_effect = lambda self: sequence.append("generator")

            sampler._initialize_unuran(seed=None)

        mock_create_distr.assert_called_once_with(sampler)
        mock_setup_discrete.assert_called_once_with(sampler)
        mock_dgt.assert_called_once_with(sampler)
        mock_create_gen.assert_called_once_with(sampler)
        assert sequence == ["distr", "discrete", "dgt", "generator"]

    def test_create_parameter_object_maps_methods(self) -> None:
        """_create_parameter_object should dispatch to each UNURAN factory and reject SROU."""
        sampler = object.__new__(DefaultUnuranSampler)
        sampler._unuran_distr = "DIST"

        def make(method: UnuranMethod) -> str:
            sampler._method = method
            sampler._lib = SimpleNamespace(
                **{f"unur_{method.value.lower()}_new": lambda dist: method.value}
            )
            return sampler._create_parameter_object()

        assert make(UnuranMethod.AROU) == "arou"
        assert make(UnuranMethod.TDR) == "tdr"
        assert make(UnuranMethod.HINV) == "hinv"
        assert make(UnuranMethod.PINV) == "pinv"
        assert make(UnuranMethod.NINV) == "ninv"
        assert make(UnuranMethod.DGT) == "dgt"
        sampler._method = UnuranMethod.SROU
        sampler._lib = SimpleNamespace()
        with pytest.raises(ValueError, match="Method SROU"):
            sampler._create_parameter_object()
