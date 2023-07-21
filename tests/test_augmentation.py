import pathlib as pl
import unittest

import pandas as pd
import yaml

from langtest.augmentation import AugmentRobustness, TemplaticAugment
from langtest.langtest import Harness


class AugmentWorkflowTestCase(unittest.TestCase):
    """
    Test case for the AugmentRobustness class.
    """

    def setUp(self) -> None:
        """"""
        self.params = {
            "spacy_ner": {
                "task": "ner",
                "model": "en_core_web_sm",
                "data": "tests/fixtures/test.conll",
                "config": "tests/fixtures/config_ner.yaml",
                "hub": "spacy",
            },
            "huggingface_ner": {
                "task": "ner",
                "model": "dslim/bert-base-NER",
                "data": "tests/fixtures/test.conll",
                "config": "tests/fixtures/config_ner.yaml",
                "hub": "huggingface",
            },
            "huggingface_textclassification": {
                "task": "text-classification",
                "model": "distilbert-base-uncased",
                "data": "tests/fixtures/test.conll",
                "config": "tests/fixtures/config_ner.yaml",
                "hub": "huggingface",
            },
        }

    def test_augment_robustness(self):
        """
        Test augmenting data for robustness.
        """
        temp_df = pd.DataFrame(
            {
                "test_type": [
                    "replace_to_female_pronouns",
                    "replace_to_male_pronouns",
                    "lowercase",
                    "uppercase",
                    "swap_entities",
                ],
                "category": ["bias", "bias", "robustness", "robustness", "robustness"],
                "fail_count": [3, 0, 82, 43, 43],
                "pass_count": [88, 91, 9, 48, 48],
                "pass_rate": [97, 100, 10, 53, 53],
                "minimum_pass_rate": [65, 65, 65, 65, 65],
                "pass": [True, True, False, False, False],
            }
        )

        augment = AugmentRobustness(
            task="ner",
            h_report=temp_df,
            config=yaml.safe_load("tests/fixtures/config_ner.yaml"),
        )
        augment.fix(
            "tests/fixtures/train.conll", "tests/fixtures/augmentated_train.conll"
        )
        self.assertIsInstance(augment, AugmentRobustness)
        self.assertIsInstance(augment.suggestions(temp_df), pd.DataFrame)

        is_file_exist = pl.Path("tests/fixtures/augmentated_train.conll").is_file()
        self.assertTrue(is_file_exist)

    def test_hf_ner_augmentation(self):
        """
        Test augmentation using Hugging Face NER model.
        """
        harness = Harness(**self.params["huggingface_ner"])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        harness.augment(
            "tests/fixtures/train.conll",
            "tests/fixtures/augmentated_train.conll",
            export_mode="inplace",
        )
        is_file_exist = pl.Path("tests/fixtures/augmentated_train.conll").is_file()
        self.assertTrue(is_file_exist)

    def test_spacy_ner_augmentation(self):
        """
        Test augmentation using spaCy NER model.
        """
        harness = Harness(**self.params["spacy_ner"])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        harness.augment(
            "tests/fixtures/train.conll",
            "tests/fixtures/augmentated_train.conll",
            export_mode="inplace",
        )
        is_file_exist = pl.Path("tests/fixtures/augmentated_train.conll").is_file()
        self.assertTrue(is_file_exist)

    def test_custom_proportions_augment_harness(self):
        """
        Test augmentation with custom proportions using Hugging Face NER model.
        """
        harness = Harness(**self.params["huggingface_ner"])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        proportions = {"uppercase": 0.5, "lowercase": 0.5}

        harness.augment(
            "tests/fixtures/train.conll",
            "tests/fixtures/augmentated_train.conll",
            custom_proportions=proportions,
            export_mode="inplace",
        )

        is_file_exist = pl.Path("tests/fixtures/augmentated_train.conll").is_file()
        self.assertTrue(is_file_exist)

    def test_templatic_augmentation(self):
        """
        Test augmentation using templatic augmentation.
        """
        generator = TemplaticAugment(
            templates=["I living in {LOC}", "you are working in {ORG}"],
            task="ner",
        )
        self.assertIsInstance(generator, TemplaticAugment)
        generator.fix(
            "tests/fixtures/train.conll",
            "tests/fixtures/augmentated_train.conll",
        )
        is_file_exist = pl.Path("tests/fixtures/augmentated_train.conll").is_file()
        self.assertTrue(is_file_exist)

    def test_spacy_templatic_augmentation(self):
        """
        Test augmentation using templatic augmentation with spaCy NER model.
        """
        harness = Harness(**self.params["spacy_ner"])
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

        harness.augment(
            "tests/fixtures/train.conll",
            "tests/fixtures/augmentated_train.conll",
            templates=["I living in {LOC}", "you are working in {ORG}"],
        )
        is_file_exist = pl.Path("tests/fixtures/augmentated_train.conll").is_file()
        self.assertTrue(is_file_exist)


class TestTemplaticAugmentation(unittest.TestCase):
    """Test case for the TemplaticAugment class"""

    def setUp(self):
        """Set up the test case"""
        self.generator = TemplaticAugment(
            templates=[
                "{PERSON} is {AGE} years old",
                "The {ANIMAL} jumped over the {OBJECT}",
            ],
            task="ner",
        )
        self.conll_path = "tests/fixtures/conll_for_augmentation.conll"

    def test_extract_variable_names(self):
        """Test extracting variable names from a template"""
        self.assertEqual(
            self.generator.extract_variable_names("{PERSON} is {AGE} years old"),
            ["PERSON", "AGE"],
        )
        self.assertEqual(
            self.generator.extract_variable_names(
                "The {ANIMAL} jumped over the {OBJECT}"
            ),
            ["ANIMAL", "OBJECT"],
        )

    def test_str_to_sample(self):
        """Test converting a template to a Sample object"""
        sample = self.generator.str_to_sample("{PERSON} is {AGE} years old")
        self.assertEqual(sample.original, "{PERSON} is {AGE} years old")
        self.assertEqual(len(sample.expected_results.predictions), 5)

    def test_add_spaces_around_punctuation(self):
        """Test adding spaces around punctuation"""
        text = "Hello,world!"
        self.assertEqual(
            self.generator.add_spaces_around_punctuation(text), "Hello , world !"
        )

    def test_change_templates(self):
        """Test changing templates"""
        changed_template = [
            "I was going to {LOCATION} in {COUNTRY}",
            "He is {AGE} years old",
        ]
        self.generator.templates = changed_template
        self.assertEqual(len(self.generator.templates), len(changed_template))
        self.assertEqual(self.generator.templates[0].original, changed_template[0])
        self.assertEqual(self.generator.templates[1].original, changed_template[1])

    def test_fix(self):
        """Test the augmentation workflow"""
        expected_result = [
            "My -X- -X- O",
            "name -X- -X- O",
            "is -X- -X- O",
            "Jean NN NN B-PER",
            "- NN NN I-PER",
            "Pierre NN NN I-PER",
            "and -X- -X- O",
            "I -X- -X- O",
            "am -X- -X- O",
            "from -X- -X- O",
            "New NN NN B-LOC",
            "York NN NN I-LOC",
            "City NN NN I-LOC",
        ]
        generator = TemplaticAugment(
            templates=["My name is {PER} and I am from {LOC}"], task="ner"
        )
        generator.fix(
            input_path=self.conll_path, output_path="/tmp/augmented_conll.conll"
        )
        with open("/tmp/augmented_conll.conll", "r") as reader:
            lines = [line.strip() for line in reader.readlines() if line.strip() != ""]

        self.assertListEqual(lines, expected_result)
