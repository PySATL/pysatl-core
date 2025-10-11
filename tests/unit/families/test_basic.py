from pysatl_core.families import ParametricFamily, Parametrization
from pysatl_core.types import (
    GenericCharacteristicName,
    UnivariateContinuous,
)
from tests.utils.mocks import MockSamplingStrategy


class TestBaseFamily:
    PDF: GenericCharacteristicName = "pdf"
    CDF: GenericCharacteristicName = "cdf"
    PPF: GenericCharacteristicName = "mean"

    def make_default_family(self, distr_characteristics=None) -> ParametricFamily:
        if distr_characteristics is None:
            distr_characteristics = {
                self.PDF: {"base": lambda p, x: x},
                self.CDF: {"alt": lambda p, x: x, "base": lambda p, x: x},
                self.PPF: {"base": lambda p, x: x},
            }
        fam = ParametricFamily(
            name="Default",
            distr_type=UnivariateContinuous,
            distr_parametrizations=["base", "alt"],
            distr_characteristics=distr_characteristics,
            sampling_strategy=MockSamplingStrategy(),
        )

        @fam.parametrization(name="base")
        class Base(Parametrization):
            value: float

        @fam.parametrization(name="alt")
        class Alt(Parametrization):
            value: float

            def transform_to_base_parametrization(self) -> Parametrization:
                return Base(value=self.value)

        return fam
