import unittest
from langtest import Harness


class StereoSetTestCase(unittest.TestCase):
    """
    Test case for stereoset functionality using the langtest harness.
    """

    def setUp(self) -> None:
        """
        Set up the test case by initializing the langtest Harness and configure it.
        """
        self.harness = Harness(
            task="stereoset",
            model={"model": "bert-base-uncased", "hub": "huggingface"},
            data={"data_source": "StereoSet"},
        )

        self.harness.data = self.harness.data[:50]

    def test_stereoset_workflow(self):
        """
        Test the stereoset workflow by generating and running test and obtaining a report.
        """
        self.harness.generate().run().report()
