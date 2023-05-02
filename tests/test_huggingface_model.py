import unittest
from nlptest.modelhandler import ModelFactory


class HuggingFaceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.models = [
            'dslim/bert-base-NER',
            'Jean-Baptiste/camembert-ner',
            'd4data/biomedical-ner-all'
        ]

        self.tasks = ["ner", "text-classifier"]

    def test_transformers_models(self):
        model = ModelFactory.load_model(task=self.tasks[0], hub="huggingface", path=self.models[0])
        self.assertIsInstance(model, ModelFactory)

    def test_unsupported_task(self):
        # Raises with unsupported task to model Factory
        with self.assertRaises(AssertionError):
            ModelFactory(self.models[0], self.tasks[1], hub="huggingface")
