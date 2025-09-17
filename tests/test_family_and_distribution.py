from pysatl_core.families import (
    ParametricFamily,
    ParametricFamilyDistribution,
    ParametricFamilyRegister,
    Parametrization,
)

PDF = "pdf"


class TestParametricFamily:
    """Test the ParametricFamily class."""

    def test_family_creation(self):
        """Test creating a parametric family."""

        # Mock strategies
        class MockSamplingStrategy:
            pass

        class MockComputationStrategy:
            pass

        # Mock characteristic function
        def mock_pdf(parameters, x):
            return x * 2

        # Create family
        family = ParametricFamily(
            name="TestFamily",
            distr_type="Continuous",
            distr_parametrizations=["mock"],
            distr_characteristics={PDF: mock_pdf},
            sampling_strategy=MockSamplingStrategy(),
            computation_strategy=MockComputationStrategy(),
        )
        register = ParametricFamilyRegister()
        register.register(family)

        # Check properties
        assert family.name == "TestFamily"
        assert family._distr_type == "Continuous"
        assert PDF in family.distr_characteristics
        assert isinstance(family.sampling_strategy, MockSamplingStrategy)
        assert isinstance(family.computation_strategy, MockComputationStrategy)


class TestParametricFamilyDistribution:
    """Test the ParametricFamilyDistribution class."""

    def test_distribution_creation(self):
        """Test creating a distribution instance."""

        # Create a mock family
        class MockSamplingStrategy:
            def sample(self, n, distr, **options):
                return [1, 2, 3]  # Mock samples

        class MockComputationStrategy:
            pass

        def mock_pdf(parameters, x):
            return x * parameters.value

        family = ParametricFamily(
            name="MockFamily",
            distr_type="Continuous",
            distr_parametrizations=["mock"],
            distr_characteristics={PDF: mock_pdf},
            sampling_strategy=MockSamplingStrategy(),
            computation_strategy=MockComputationStrategy(),
        )

        # Create a mock parametrization
        class MockParametrization(Parametrization):
            def __init__(self, value):
                self.value = value

            @property
            def name(self):
                return "mock"

            @property
            def parameters(self):
                return {"value": self.value}

        # Add to family
        family.parametrizations.add_parametrization("mock", MockParametrization)

        # Register family
        register = ParametricFamilyRegister()
        register.register(family)

        # Create distribution
        params = MockParametrization(2.0)

        dist = ParametricFamilyDistribution("MockFamily", "Continuous", params)

        # Check properties
        assert dist.distr_name == "MockFamily"
        assert dist.distribution_type == "Continuous"
        assert dist.parameters is params
        assert dist.family is family

        # Test sampling
        samples = dist.sample(3)
        assert samples == [1, 2, 3]

        # Test analytical computations
        computations = dist.analytical_computations
        assert PDF in computations
        assert computations[PDF].func(5.0) == 10.0  # 5.0 * 2.0
