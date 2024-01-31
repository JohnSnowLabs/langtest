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
            "sensitivity": [
                SensitivitySample(
                    original="she is going to school?",
                    options="-",
                )
            ]
        }

    def available_test(self) -> dict:
        """
        Get a dictionary of available sensitivity tests.

        Returns:
            dict: A dictionary containing available sensitivity tests.
        """
        tests = {
            j: i
            for i in BaseSensitivity.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests

    def test(self) -> None:
        """
        Test sensitivity sample for sensitivity classes.

        Returns:
            None
        """
        for test in self.perturbations_list:
            for task in self.samples:
                sample = self.samples[task][-1]
                test_func = self.supported_tests[test].transform

                if test != "add_toxic_words":
                    sample.transform(test_func, {})

                else:
                    sample.transform(
                        test_func,
                        {
                            "ending_context": ["fuck u ", "u fuck"],
                            "starting_context": [
                                "Hi fuck",
                                "Good morning fuck",
                                "hello fuck",
                            ],
                        },
                    )
            assert sample.test_case is not None
