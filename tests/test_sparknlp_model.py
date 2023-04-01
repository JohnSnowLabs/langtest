import os
import unittest

from nlptest import Harness
from nlptest.modelhandler import ModelFactory


class SparkNLPTestCase(unittest.TestCase):

    def setUp(self) -> None:
        print(os.getcwd())
        self.params = {
            "task": 'ner',
            "model": "ner.dl",
            "data": "nlptest/demo/data/test.conll",
            "config": "nlptest/demo/data/config.yml"
        }

    def test_predict(self):
        """
        Testing Instance after in Harness Class
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness, Harness)

    def test_outputCol(self):
        """
        Testing Attributes of Harness Class
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness.model, (str, ModelFactory))
        self.assertIsInstance(harness.model.output_col, str)

    def test_confidence(self):
        """
        Testing Attributes of Harness Class
        """
        harness = Harness(**self.params)
        self.assertIsInstance(harness.model.model.getIncludeAllConfidence(), bool)