import pathlib as pl
import unittest

import pandas as pd
import yaml

from nlptest.augmentation import AugmentRobustness
from nlptest.modelhandler.modelhandler import ModelFactory
from nlptest.nlptest import Harness


class AugmentRobustnessTestCase(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        """"""
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

    def test_augment_robustness(self):
        """"""
        temp_df = pd.DataFrame({
            'test_type': ['replace_to_female_pronouns', 'replace_to_male_pronouns', 'lowercase', 'uppercase'],
            'category': ['bias', 'bias', 'robustness', 'robustness'],
            'fail_count': [3, 0, 82, 43],
            'pass_count': [88, 91, 9, 48],
            'pass_rate': [97, 100, 10, 53],
            'minimum_pass_rate': [65, 65, 65, 65],
            'pass': [True, True, False, False]
        })

        augment = AugmentRobustness(
            task='ner',
            h_report=temp_df,
            config=yaml.safe_load('tests/fixtures/config_ner.yaml')
        )
        augment.fix('tests/fixtures/train.conll',
                    'tests/fixtures/augmentated_train.conll')
        self.assertIsInstance(augment, AugmentRobustness)
        self.assertIsInstance(augment.suggestions(temp_df), pd.DataFrame)

        is_file_exist = pl.Path(
            'tests/fixtures/augmentated_train.conll').is_file()
        self.assertTrue(is_file_exist)

        temp_df = pd.DataFrame({
            'test_type': ['replace_to_female_pronouns', 'replace_to_male_pronouns', 'lowercase', 'uppercase'],
            'category': ['bias', 'bias', 'robustness', 'robustness'],
            'fail_count': [3, 0, 82, 43],
            'pass_count': [88, 91, 9, 48],
            'pass_rate': [97, 100, 10, 53],
            'minimum_pass_rate': [65, 65, 65, 65],
            'pass': [True, True, False, False]
        })


        augment = AugmentRobustness(
            task='ner',
            h_report=temp_df,
            config='tests/fixtures/config_ner.yaml'
        )
        augment.fix('tests/fixtures/train.conll',
                    'tests/fixtures/augmentated_train.conll')
        self.assertIsInstance(augment, AugmentRobustness)
        self.assertIsInstance(augment.suggestions(temp_df), pd.DataFrame)

        is_file_exist = pl.Path(
            'tests/fixtures/augmentated_train.conll').is_file()
        self.assertTrue(is_file_exist)

    def test_hf_ner_augmentation(self):
        """"""
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
