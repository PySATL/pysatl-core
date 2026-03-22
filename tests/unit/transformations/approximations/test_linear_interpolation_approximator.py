from __future__ import annotations

__author__ = "Leonid Elkin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np
import pytest

from pysatl_core.distributions.computation import AnalyticalComputation
from pysatl_core.distributions.support import ContinuousSupport
from pysatl_core.transformations.approximations import (
    CDFMonotoneSplineApproximation,
    PDFLinearInterpolationApproximation,
    PPFMonotoneSplineApproximation,
)
from pysatl_core.transformations.operations import affine
from pysatl_core.types import CharacteristicName, ComputationFunc, Kind
from tests.unit.distributions.test_basic import DistributionTestBase
from tests.utils.mocks import StandaloneEuclideanUnivariateDistribution


class TestInterpolationApproximator(DistributionTestBase):
    @staticmethod
    def _make_vectorized_uniform_pdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
        def _pdf(x: np.ndarray, **_kwargs: Any) -> np.ndarray:
            array = np.asarray(x, dtype=float)
            return np.where((array >= 0.0) & (array <= 1.0), 1.0, 0.0)

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.PDF: AnalyticalComputation[np.ndarray, np.ndarray](
                    target=CharacteristicName.PDF,
                    func=cast(ComputationFunc[np.ndarray, np.ndarray], _pdf),
                )
            },
            support=ContinuousSupport(0.0, 1.0),
        )

    @staticmethod
    def _make_vectorized_logistic_cdf_distribution() -> StandaloneEuclideanUnivariateDistribution:
        def _cdf(x: np.ndarray, **_kwargs: Any) -> np.ndarray:
            array = np.asarray(x, dtype=float)
            clipped = np.clip(array, -700.0, 700.0)
            return 1.0 / (1.0 + np.exp(-clipped))

        return StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.CDF: AnalyticalComputation[np.ndarray, np.ndarray](
                    target=CharacteristicName.CDF,
                    func=cast(ComputationFunc[np.ndarray, np.ndarray], _cdf),
                )
            },
            support=ContinuousSupport(),
        )

    def test_pdf_is_non_negative_and_normalized_after_approximation(self) -> None:
        source = self._make_vectorized_uniform_pdf_distribution()
        transformed = affine(source, scale=2.0, shift=1.0)

        approximated = transformed.approximate(
            methods={
                CharacteristicName.PDF: PDFLinearInterpolationApproximation(
                    n_grid=513,
                    lower_limit=1.0,
                    upper_limit=3.0,
                )
            },
        )
        pdf = approximated.query_method(CharacteristicName.PDF)

        grid = np.linspace(1.0, 3.0, 4001, dtype=float)
        values = np.asarray(pdf(grid), dtype=float)

        assert float(values.min()) >= -1e-12
        assert np.trapezoid(values, grid) == pytest.approx(1.0, abs=5e-3)
        assert float(pdf(0.0)) == pytest.approx(0.0)
        assert float(pdf(4.0)) == pytest.approx(0.0)

    def test_cdf_is_monotone_and_bounded(self) -> None:
        source = self._make_vectorized_logistic_cdf_distribution()
        transformed = affine(source, scale=1.0)

        approximated = transformed.approximate(
            methods={
                CharacteristicName.CDF: CDFMonotoneSplineApproximation(
                    n_grid=513,
                    lower_limit_prob=1e-6,
                    upper_limit_prob=1e-6,
                )
            },
        )
        cdf = approximated.query_method(CharacteristicName.CDF)

        grid = np.linspace(-10.0, 10.0, 1025, dtype=float)
        values = np.asarray(cdf(grid), dtype=float)

        assert np.all(np.diff(values) >= -1e-12)
        assert float(values[0]) == pytest.approx(0.0, abs=1e-3)
        assert float(values[-1]) == pytest.approx(1.0, abs=1e-3)
        assert float(cdf(-100.0)) == pytest.approx(0.0, abs=1e-12)
        assert float(cdf(100.0)) == pytest.approx(1.0, abs=1e-12)

    def test_ppf_exists_on_full_unit_interval(self) -> None:
        source = self.make_uniform_ppf_distribution()
        transformed = affine(source, scale=2.0, shift=1.0)

        approximated = transformed.approximate(
            methods={
                CharacteristicName.PPF: PPFMonotoneSplineApproximation(
                    n_grid=513,
                    lower_limit=0.0,
                    upper_limit=1.0,
                )
            },
        )
        ppf = approximated.query_method(CharacteristicName.PPF)

        probabilities = np.linspace(0.0, 1.0, 513, dtype=float)
        values = np.asarray(ppf(probabilities), dtype=float)

        assert np.all(np.isfinite(values))
        assert np.all(np.diff(values) >= -1e-12)
        assert float(ppf(0.0)) == pytest.approx(1.0, abs=1e-8)
        assert float(ppf(1.0)) == pytest.approx(3.0, abs=1e-8)

    def test_rejects_empty_methods_mapping(self) -> None:
        source = self.make_uniform_ppf_distribution()
        transformed = affine(source, scale=1.0)

        with pytest.raises(ValueError, match="At least one characteristic approximation method"):
            transformed.approximate(methods={})

    def test_approximates_only_characteristics_from_mapping(self) -> None:
        source = self._make_vectorized_logistic_cdf_distribution()
        transformed = affine(source, scale=1.0)

        approximated = transformed.approximate(
            methods={
                CharacteristicName.CDF: CDFMonotoneSplineApproximation(),
            },
        )

        assert set(approximated.analytical_computations) == {CharacteristicName.CDF}

    def test_raises_when_distribution_cannot_provide_requested_characteristic(self) -> None:
        def _mean(**_kwargs: object) -> float:
            return 0.0

        source = StandaloneEuclideanUnivariateDistribution(
            kind=Kind.CONTINUOUS,
            analytical_computations={
                CharacteristicName.MEAN: AnalyticalComputation[object, float](
                    target=CharacteristicName.MEAN,
                    func=_mean,
                )
            },
        )
        transformed = affine(source, scale=1.0)

        with pytest.raises(ValueError, match="requires an analytical method"):
            transformed.approximate(
                methods={
                    CharacteristicName.PDF: PDFLinearInterpolationApproximation(),
                },
            )
