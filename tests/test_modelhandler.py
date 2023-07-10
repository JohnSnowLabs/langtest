import unittest
import langtest
from langtest.modelhandler import ModelFactory
from langtest.modelhandler.spacy_modelhandler import PretrainedModelForNER as SpacyNER
from langtest.modelhandler.llm_modelhandler import ConfigError


class ModelHandlerTestCase(unittest.TestCase):
    """
    Test case for the ModelHandler class.

    Note: we mainly want to check hubs access and loading mechanism
    """

    def setUp(self):
        pass

    def test_unsupported_task(self) -> None:
        """
        Test loading an unsupported task.
        """
        with self.assertRaises(AssertionError) as _:
            ModelFactory.load_model(
                task="unsupported_task", hub="spacy", path="en_core_web_sm"
            )

    def test_unsupported_hub(self) -> None:
        """
        Test loading an unsupported hub.
        """
        with self.assertRaises(AssertionError) as _:
            ModelFactory.load_model(task="ner", hub="invalid_hub", path="en_core_web_sm")

    def test_ner_transformers_model(self) -> None:
        """
        Test loading an NER model from Hugging Face Transformers.
        """
        model = ModelFactory.load_model("ner", "huggingface", "dslim/bert-base-NER")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(
            model.model_class,
            langtest.modelhandler.transformers_modelhandler.PretrainedModelForNER,
        )

    def test_ner_spacy_model(self) -> None:
        """
        Test loading an NER model from spaCy.
        """
        model = ModelFactory.load_model("ner", "spacy", "en_core_web_sm")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(
            model.model_class,
            SpacyNER,
        )

    def test_ai21_model(self) -> None:
        """
        Test loading a model from the Ai21 hub
        """
        with self.assertRaises(ConfigError) as _:
            ModelFactory.load_model(
                task="question-answering", hub="ai21", path="j2-jumbo-instruct"
            )

    def test_openai_model(self) -> None:
        """
        Test loading a model from the OpenAI hub
        """
        with self.assertRaises(ConfigError) as _:
            ModelFactory.load_model(
                task="question-answering", hub="openai", path="gpt-3.5-turbo"
            )

    def test_cohere_model(self) -> None:
        """
        Test loading a model from Cohere
        """
        with self.assertRaises(ConfigError) as _:
            ModelFactory.load_model(
                task="question-answering", hub="cohere", path="command-xlarge-nightly"
            )
