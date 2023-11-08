import unittest
from langtest import Harness


class CrowsPairsTestCase(unittest.TestCase):
    """
    Test case for crows-pairs functionality using the langtest harness.
    """

    def setUp(self) -> None:
        """
        Set up the test case by initializing the langtest Harness and configure it.
        """
        self.harness = Harness(
            task={"task": "fill-mask", "category": "crows-pairs"},
            model={"model": "bert-base-uncased", "hub": "huggingface"},
            data={"data_source": "Crows-Pairs"},
        )

        # configure the harness
        self.harness.configure(
            {
                "tests": {
                    "defaults": {"min_pass_rate": 1.0},
                    "stereotype": {
                        "crows-pairs": {"min_pass_rate": 0.7},
                    },
                },
            }
        )
        self.harness.data = self.harness.data[:5]

    def test_crows_pairs_workflow(self):
        """
        Test the crows-pairs workflow by generating and running test and obtaining a report.
        """
        self.harness.generate().run().report()
