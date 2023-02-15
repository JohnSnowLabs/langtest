import unittest

from nlptest.modelhandler.modelhandler import *

class ModelHandlerTestCase(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_unsupported_task(self) -> None:
        self.assertRaises(
            ValueError,
            ModelFactory.load_model(task="unsupported_task", hub="spacy", path="en_core_web_sm")
        )

    def test_unsupported_hub(self) -> None:
        self.assertRaises(
            ValueError,
            ModelFactory.load_model(task="unsupported_task", hub="invalid_hub", path="en_core_web_sm")
        )
    
    def test_ner_transformers_model(self) -> None:
        model = ModelFactory.load_model("ner","huggingface","dslim/bert-base-NER")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(model.model_class, NERTransformersPretrainedModel)
        
    def test_ner_spacy_model(self) -> None:
        model = ModelFactory.load_model("ner","spacy","en_core_web_sm")
        self.assertIsInstance(model, ModelFactory)
        self.assertIsInstance(model.model_class, NERSpaCyPretrainedModel)
