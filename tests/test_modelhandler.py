import unittest
import nlptest
from nlptest.modelhandler.modelhandler import *


class ModelHandlerTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_unsupported_task(self) -> None:
        with self.assertRaises(AssertionError) as _:
            ModelFactory.load_model(
                task="unsupported_task", hub="spacy", path="en_core_web_sm")

    def test_unsupported_hub(self) -> None:
        with self.assertRaises(AssertionError) as _:
            ModelFactory.load_model(
                task="ner", hub="invalid_hub", path="en_core_web_sm")

    def test_ner_transformers_model(self) -> None:
        model = ModelFactory.load_model(
            "ner", "huggingface", "dslim/bert-base-NER")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(
            model.model_class, nlptest.modelhandler.transformers_modelhandler.PretrainedModelForNER)

    def test_ner_spacy_model(self) -> None:
        model = ModelFactory.load_model("ner", "spacy", "en_core_web_sm")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(
            model.model_class, nlptest.modelhandler.spacy_modelhandler.PretrainedModelForNER)