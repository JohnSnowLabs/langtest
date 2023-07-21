import pandas as pd
import pytest
from langtest import Harness


class TestRepresentation:
    """
    TestRepresentation class for evaluating the representation quality of various NLP models.

    This class checks the representation counts and proportions of gender, ethnicity names, labels, religions,
    and country economic data in the model outputs.

    Attributes:
        params (dict): Dictionary containing different NLP models and their configurations.

    Methods:
        configure_tests(harness): Configures the test settings for the harness.
        test_representation(model_params): Generic test representation for NER and Text Classification models.
    """

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

    def configure_tests(self, harness):
        """
        Configures the test settings for the harness.

        Args:
            harness (Harness): The Harness instance to configure the tests.
        """

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

    @pytest.mark.parametrize("model_params", params.values())
    def test_representation(self, model_params):
        """
        Generic test representation for NER and Text Classification models.

        Args:
            model_params (dict): Dictionary containing the parameters for the NLP model.

        Raises:
            AssertionError: If the generated report is not an instance of pd.DataFrame.
        """

        harness = Harness(**model_params)
        self.configure_tests(harness)
        harness.data = harness.data[:50]
        assert isinstance(harness, Harness)
        report = harness.generate().run().report()
        assert isinstance(report, pd.DataFrame)
