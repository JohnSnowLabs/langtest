import unittest
import pandas as pd
from langtest import Harness


class RepresentationTestCase(unittest.TestCase):
    def setUp(self) -> None:
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
        }

    def configure_tests(self) -> None:
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
        harness = Harness(**self.params["huggingface_ner"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)

    def test_representation_spacy_ner(self):
        harness = Harness(**self.params["spacy_ner"])
        self.harness = harness
        self.configure_tests()
        harness.data = harness.data[:50]
        self.assertIsInstance(harness, Harness)
        report = harness.generate().run().report()
        self.assertIsInstance(report, pd.DataFrame)
