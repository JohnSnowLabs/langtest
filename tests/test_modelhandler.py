import unittest
import langtest
from langtest.modelhandler.modelhandler import *


class ModelHandlerTestCase(unittest.TestCase):
    """
    Test case for the ModelHandler class.
    """
    def setUp(self):
        pass

    def test_unsupported_task(self) -> None:
        """
        Test loading an unsupported task.
        """
        with self.assertRaises(AssertionError) as _:
            ModelFactory.load_model(
                task="unsupported_task", hub="spacy", path="en_core_web_sm")

    def test_unsupported_hub(self) -> None:
        """
        Test loading an unsupported hub.
        """
        with self.assertRaises(AssertionError) as _:
            ModelFactory.load_model(
                task="ner", hub="invalid_hub", path="en_core_web_sm")

    def test_ner_transformers_model(self) -> None:
        """
        Test loading an NER model from Hugging Face Transformers.
        """
        model = ModelFactory.load_model(
            "ner", "huggingface", "dslim/bert-base-NER")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(
            model.model_class, langtest.modelhandler.transformers_modelhandler.PretrainedModelForNER)

    def test_ner_spacy_model(self) -> None:
        """
        Test loading an NER model from spaCy.
        """
        model = ModelFactory.load_model("ner", "spacy", "en_core_web_sm")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(
            model.model_class, langtest.modelhandler.spacy_modelhandler.PretrainedModelForNER)