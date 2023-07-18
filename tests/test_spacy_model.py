import unittest
from langtest import Harness
from langtest.modelhandler.modelhandler import ModelFactory


class SpacyTestCase(unittest.TestCase):
    """
    Test case for the Spacy integration in the langtest module.
    """

    def setUp(self) -> None:
        """
        Set up the test case.

        Initializes the parameters for the Harness class.
        """
        self.params = {
            "task": "ner",
            "model": "en_core_web_sm",
            "data": "langtest/data/conll/sample.conll",
            "config": "tests/fixtures/config_ner.yaml",
            "hub": "spacy",
        }

    def test_Harness(self):
        """
        Testing Instance after in Harness Class
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness, Harness)

    def test_attributes(self):
        """
        Testing Attributes of Harness Class
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness.task, str)
        self.assertIsInstance(harness.model, (str, ModelFactory))
        # self.assertIsInstance(self.harness.data, (str, DataFactory))
        self.assertIsInstance(harness._config, (str, dict))

    def test_generate(self):
        """
        Test the generate method of the Harness class.

        Checks if the generate method returns a Harness instance.
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness.generate(), Harness)

    def test_run(self):
        """
        Test the run method of the Harness class.

        Checks if the run method returns a Harness instance after generating.
        """
        harness = Harness(**self.params)
        harness.generate()
        self.assertIsInstance(harness.run(), Harness)
