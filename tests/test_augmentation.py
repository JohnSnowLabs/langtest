import unittest
import pandas as pd
import pathlib as pl

from nlptest.augmentation.fix_robustness import AugmentRobustness
from nlptest.modelhandler.modelhandler import ModelFactory
from nlptest.nlptest import Harness


class AugmentRobustnessTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.params = {
            "spacy_ner": {
                "task": 'ner',
                "model": "en_core_web_sm",
                "data": "tests/fixtures/test.conll",
                "config": "tests/fixtures/config_ner.yaml",
                "hub": "spacy"
            },
            "huggingface_ner": {
                "task": 'ner',
                "model": "dslim/bert-base-NER",
                "data": "tests/fixtures/test.conll",
                "config": "tests/fixtures/config_ner.yaml",
                "hub": "huggingface"
            },
            "huggingface_textclassification": {
                "task": 'text-classification',
                "model": "distilbert-base-uncased",
                "data": "tests/fixtures/test.conll",
                "config": "tests/fixtures/config_ner.yaml",
                "hub": "huggingface"
            }
        }

    def test_hf_ner_augmentation(self):
        harness = Harness(**self.params['huggingface_ner'])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        harness.augment('tests/fixtures/train.conll',
                        'tests/fixtures/augmentated_train.conll', inplace=True)
        is_file_exist = pl.Path(
            'tests/fixtures/augmentated_train.conll').is_file()
        self.assertTrue(is_file_exist)

    def test_spacy_ner_augmentation(self):
        harness = Harness(**self.params['spacy_ner'])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        harness.augment('tests/fixtures/train.conll',
                        'tests/fixtures/augmentated_train.conll', inplace=True)
        is_file_exist = pl.Path(
            'tests/fixtures/augmentated_train.conll').is_file()
        self.assertTrue(is_file_exist)
