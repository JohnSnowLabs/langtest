
import unittest
from nlptest import Harness
from nlptest.modelhandler.modelhandler import ModelFactory


class SpacyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.params = {
            "task": 'ner',
            "model": "en_core_web_sm",
            "data": "nlptest/data/conll/sample.conll",
            "config": "tests/fixtures/config_ner.yaml",
            "hub": "spacy"
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
        harness = Harness(**self.params)
        self.assertIsInstance(
            harness.generate(),
            Harness
        )

    def test_run(self):
        harness = Harness(**self.params)
        harness.generate()
        self.assertIsInstance(
            harness.run(),
            Harness
        )
