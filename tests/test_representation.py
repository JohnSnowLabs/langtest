import pandas as pd
import pytest
from langtest import Harness


params = {
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


class TestRepresentation:
    """Test case class for representation tests.

    This test case class evaluates the representation quality of various NLP models.
    It checks the representation counts and proportions of gender, ethnicity names, labels, religions,
    and country economic data in the model outputs.
    """

    def configure_tests(self, harness):
        """Configures the test settings for the harness."""
        harness.configure(
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

    @pytest.mark.parametrize(
        "model_params",
        [
            params["jsl_text_classification"],
            params["spacy_text_classification"],
            params["huggingface_text_classification"],
        ],
    )
    def test_representation_text_classification(self, model_params):
        """Test representation for Text Classification models."""

        harness = Harness(**model_params)
        self.configure_tests(harness)
        harness.data = harness.data[:50]
        assert isinstance(harness, Harness)
        report = harness.generate().run().report()
        assert isinstance(report, pd.DataFrame)

    @pytest.mark.parametrize(
        "model_params",
        [params["spacy_ner"], params["huggingface_ner"], params["jsl_ner"]],
    )
    def test_representation_ner(self, model_params):
        """Test representation for NER models."""

        harness = Harness(**model_params)
        self.configure_tests(harness)
        harness.data = harness.data[:50]
        assert isinstance(harness, Harness)
        report = harness.generate().run().report()
        assert isinstance(report, pd.DataFrame)
