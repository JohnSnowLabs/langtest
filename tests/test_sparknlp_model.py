import os
import unittest

from nlptest import Harness
from nlptest.modelhandler import ModelFactory


class SparkNLPTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.params = {
            "task": 'ner',
            "model": "ner_dl_bert",
            "data": "demo/data/test.conll",
            "config": "demo/data/config.yml",
            "hub": "johnsnowlabs"
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
        self.assertIsInstance(harness.model.model_class.output_col, str)
