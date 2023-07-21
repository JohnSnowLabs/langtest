import unittest
import pandas as pd
from langtest import Harness


class RepresentationTestCase(unittest.TestCase):
    """Test case class for representation tests.

    This test case class evaluates the representation quality of various NLP models.
    It checks the representation counts and proportions of gender, ethnicity names, labels, religions,
    and country economic data in the model outputs.

    Attributes:
        params (dict): A dictionary containing model configuration parameters.

    Methods:
        setUp(self): Initializes the test case by setting up the model parameters.
        configure_tests(self): Configures the tests by setting the minimum count and proportion thresholds for
                              representation evaluation.
        test_representation_hf_ner(self): Tests the representation quality of the Huggingface NER model.
        test_representation_spacy_ner(self): Tests the representation quality of the Spacy NER model.
        test_representation_jsl_ner(self): Tests the representation quality of the John Snow Labs NER model.
        test_representation_jsl_test_classification(self): Tests the representation quality of the John Snow Labs
                                                           Text Classification model.
        test_representation_spacy_test_classification(self): Tests the representation quality of the Spacy Text
                                                             Classification model.
        test_representation_huggingface_test_classification(self): Tests the representation quality of the Huggingface
                                                                   Text Classification model.

    """

    def setUp(self) -> None:
        """Sets up the test case by initializing parameters."""

        self.params = {
            "spacy_ner": {
                "task": "ner",
                "model": "en_core_web_sm",
                "hub": "spacy",
            },
            "huggingface_ner": {
                "task": "ner",
                "model": "dslim/bert-base-NER",
                "hub": "huggingface",
            },
            "jsl_ner": {
                "task": "ner",
                "model": "ner.dl",
                "hub": "johnsnowlabs",
            },
            "jsl_text_classification": {
                "task": "text-classification",
                "model": "en.sentiment.imdb.glove",
                "hub": "johnsnowlabs",
            },
            "spacy_text_classification": {
                "task": "text-classification",
                "model": "textcat_imdb",
                "hub": "spacy",
            },
            "huggingface_text_classification": {
                "task": "text-classification",
                "model": "lvwerra/distilbert-imdb",
                "hub": "huggingface",
            },
        }

    def configure_tests(self) -> None:
        """Configures the test settings for the harness."""

        self.harness.configure(
            {
                "tests": {
                    "defaults": {"min_pass_rate": 0.65},
                    "representation": {
                        "min_gender_representation_count": {"min_count": 10},
                        "min_ethnicity_name_representation_count": {"min_count": 10},
                        "min_label_representation_count": {"min_count": 10},
                        "min_religion_name_representation_count": {"min_count": 10},
                        "min_country_economic_representation_count": {"min_count": 10},
                        "min_gender_representation_proportion": {"min_proportion": 0.1},
                        "min_ethnicity_name_representation_proportion": {
                            "min_proportion": 0.1
                        },
                        "min_label_representation_proportion": {"min_proportion": 0.1},
                        "min_religion_name_representation_proportion": {
                            "min_proportion": 0.1
                        },
                        "min_country_economic_representation_proportion": {
                            "min_proportion": 0.1
                        },
                    },
                }
            }
        )

    def test_representation_hf_ner(self):
        """Test representation for Hugging Face NER model."""

        harness = Harness(**self.params["huggingface_ner"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

    def test_representation_spacy_ner(self):
        """Test representation for spaCy NER model."""

        harness = Harness(**self.params["spacy_ner"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

    def test_representation_jsl_ner(self):
        """Test representation for John Snow Labs NER model."""

        harness = Harness(**self.params["jsl_ner"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

    def test_representation_jsl_test_classification(self):
        """Test representation for John Snow Labs Text Classification model."""

        harness = Harness(**self.params["jsl_text_classification"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

    def test_representation_spacy_test_classification(self):
        """Test representation for spaCy Text Classification model."""

        harness = Harness(**self.params["spacy_text_classification"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

    def test_representation_huggingface_test_classification(self):
        """Test representation for Hugging Face Text Classification model."""

        harness = Harness(**self.params["huggingface_text_classification"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)
