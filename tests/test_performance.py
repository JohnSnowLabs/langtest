import unittest
from langtest.langtest import Harness
from langtest.transform.performance import Speed
from langtest.utils.custom_types.sample import SpeedTestSample


class TestPerformance(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {
            "spacy_ner": {
                "task": "ner",
                "model": {"model": "en_core_web_sm", "hub": "spacy"},
                "data": {"data_source": "tests/fixtures/test.conll"},
                "config": "tests/fixtures/config_performance.yaml",
            },
            "huggingface_ner": {
                "task": "ner",
                "model": {"model": "dslim/bert-base-NER", "hub": "huggingface"},
                "data": {"data_source": "tests/fixtures/test.conll"},
                "config": "tests/fixtures/config_performance.yaml",
            },
            "huggingface_textclassification": {
                "task": "text-classification",
                "model": {"model": "distilbert-base-uncased", "hub": "huggingface"},
                "data": {"data_source": "tests/fixtures/text_classification.csv"},
                "config": "tests/fixtures/config_performance.yaml",
            },
        }

    def test_speed_spacy(self):
        """
        Test speed measure for spacy model.
        """
        harness = Harness(**self.params["spacy_ner"])
        harness.generate().run().report()
        self.assertIsInstance(harness._testcases[-1], SpeedTestSample)

    def test_speed_huggingface(self):
        """
        Test speed measure for huggingface model.
        """
        harness = Harness(**self.params["huggingface_ner"])
        harness.generate().run().report()
        self.assertIsInstance(harness._testcases[-1], SpeedTestSample)

    def test_speed_huggingface_textclassification(self):
        """
        Test speed measure for huggingface model.
        """
        harness = Harness(**self.params["huggingface_textclassification"])
        harness.generate().run().report()
        self.assertIsInstance(harness._testcases[-1], SpeedTestSample)
