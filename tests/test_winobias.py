import unittest
from langtest import Harness


class WinoBiasTestCase(unittest.TestCase):
    """
    Test case for wino-bias functionality using the langtest harness.
    """

    def setUp(self) -> None:
        """
        Set up the test case by initializing the langtest Harness and configure it.
        """
        self.harness = Harness(
            task="wino-bias",
            model={"model": "bert-base-uncased", "hub": "huggingface"},
            data={"data_source": "Wino-test"},
        )

        # configure the harness
        self.harness.configure(
            {
                "tests": {
                    "defaults": {"min_pass_rate": 1.0},
                    "wino-bias": {
                        "gender-occupational-stereotype": {"min_pass_rate": 0.7},
                    },
                },
            }
        )
        self.harness.data = self.harness.data[:5]

    def test_wino_bias_workflow(self):
        """
        Test the wino-bias workflow by generating and running test and obtaining a report.
        """
        self.harness.generate().run().report()
