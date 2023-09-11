import unittest
from langtest.transform.sensitivity import *
from langtest.utils.custom_types.sample import SensitivitySample
from langtest.transform import TestFactory

class SensitivityTestCase(unittest.TestCase):
    """
    A test case class for testing sensitivity samples on sensitivity classes.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.

        Returns:
            None
        """
        test_scenarios = TestFactory.test_scenarios()
        self.available_tests = {
            test: list(scenarios.keys()) for test, scenarios in test_scenarios.items()
        }

        self.perturbations_list = self.available_tests["sensitivity"]
        self.supported_tests = self.available_test()
        self.samples = {
            "sensitivity-test": [
                SensitivitySample(
                    original="she is going to school?",
                )
            ]
        }