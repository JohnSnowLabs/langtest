import os
import unittest
import pandas as pd 
import pathlib as pl

import yaml 

from nlptest.augmentation.fix_robustness import AugmentRobustness
from nlptest.modelhandler.modelhandler import ModelFactory
from nlptest.nlptest import Harness


class AugmentRobustnessTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.params = {
            "spacy_ner":{
                "task": 'ner',
                "model": "en_core_web_sm",
                "data": "demo/data/test.conll",
                "config": "demo/data/config.yml",
                "hub": "spacy"
            },
            "huggingface_ner":{
                "task": 'ner',
                "model": "dslim/bert-base-NER",
                "data": "demo/data/test.conll",
                "config": "demo/data/config.yml",
                "hub": "huggingface"
            },
            "huggingface_textclassification":{
                "task": 'text-classification',
                "model": "distilbert-base-uncased",
                "data": "demo/data/test.conll",
                "config": "demo/data/config.yml",
                "hub": "huggingface"
            }
        }
    
    def test_augmentrobustness(self):
        temp_df = pd.DataFrame({
            'test_type': ['replace_to_female_pronouns', 'replace_to_male_pronouns', 'lowercase', 'swap_entities', 'uppercase', 'add_context'],
            'category': ['Bias', 'Bias', 'Robustness', 'Robustness', 'Robustness', 'Robustness'],
            'fail_count': [3, 0, 82, 43, 84, 91],
            'pass_count': [88, 91, 9, 48, 7, 0],
            'pass_rate': [97, 100, 10, 53, 8, 0],
            'minimum_pass_rate': [65, 65, 65, 65, 65, 65],
            'pass': [True, True, False, False, False, False]
        })

        model = ModelFactory.load_model(
            task='ner',
            hub="huggingface",
            path='dslim/bert-base-NER')

        augment = AugmentRobustness(
            task='ner',
            h_report=temp_df,
            config='tests/fixtures/config_ner.yaml',
            model=model
        )
        augment.fix('tests/fixtures/train.conll', 'tests/fixtures/augmentated_train.conll')
        self.assertIsInstance(augment, AugmentRobustness)
        self.assertIsInstance(augment.suggestions(temp_df), pd.DataFrame)

        is_file_exist = pl.Path('tests/fixtures/augmentated_train.conll').is_file()
        self.assertTrue(is_file_exist)
        

    def test_hf_ner_augmentation(self):
        harness = Harness(**self.params['huggingface_ner'])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        harness.augment('tests/fixtures/train.conll', 'tests/fixtures/augmentated_train.conll', inplace=True)
        is_file_exist = pl.Path('tests/fixtures/augmentated_train.conll').is_file()
        self.assertTrue(is_file_exist)

    def test_spacy_ner_augmentation(self):
        harness = Harness(**self.params['spacy_ner'])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        harness.augment('tests/fixtures/train.conll', 'tests/fixtures/augmentated_train.conll', inplace=True)
        is_file_exist = pl.Path('tests/fixtures/augmentated_train.conll').is_file()
        self.assertTrue(is_file_exist)
