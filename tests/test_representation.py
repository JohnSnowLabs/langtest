import pytest

from langtest.transform.representation import (
    BaseRepresentation,
    CountryEconomicRepresentation,
    EthnicityRepresentation,
    GenderRepresentation,
    LabelRepresentation,
    ReligionRepresentation,
)
from langtest.utils.custom_types import SequenceLabel, Span
from langtest.utils.custom_types.output import (
    NEROutput,
    NERPrediction,
    SequenceClassificationOutput,
    TranslationOutput,
)
from langtest.utils.custom_types.sample import (
    MinScoreQASample,
    MinScoreSample,
    NERSample,
    QASample,
    SequenceClassificationSample,
    SummarizationSample,
    ToxicitySample,
    TranslationSample,
)


class TestRepresentation:
    """A test suite for evaluating the transformation process of various representations.
    This test suite ensures that the representations can successfully transform input data
    and produce valid results.
    The representations tested include GenderRepresentation, EthnicityRepresentation,
    ReligionRepresentation, and CountryEconomicRepresentation.
    Attributes:
        representation_config (Dict)
    """

    representation_config = {
        "min_gender_representation_count": {"min_count": 5},
        "min_gender_representation_proportion": {"min_proportion": 0.1},
        "min_ethnicity_name_representation_count": {"min_count": 10},
        "min_ethnicity_name_representation_proportion": {"min_proportion": 0.1},
        "min_religion_name_representation_count": {"min_count": 10},
        "min_religion_name_representation_proportion": {"min_proportion": 0.1},
        "min_country_economic_representation_count": {"min_count": 10},
        "min_country_economic_representation_proportion": {"min_proportion": 0.1},
        "min_label_representation_count": {"min_count": 10},
        "min_label_representation_proportion": {"min_proportion": 0.1},
    }

    @pytest.fixture
    def sample_data(self):
        """A fixture providing sample data for the representation transformation tests.
        Returns:
            list: A list containing sample SequenceClassificationSample instances.
        """
        return {
            "text-classification": [
                SequenceClassificationSample(
                    original="The last good ernest movie, and the best at that. how can you not laugh at least once at this movie. the last line is a classic, as is ernest's gangster impressions, his best moment on film. this has his best lines and is a crowning achievement among the brainless screwball comedies.",
                    expected_results=SequenceClassificationOutput(
                        predictions=[SequenceLabel(label="Positive", score=1.0)]
                    ),
                ),
                SequenceClassificationSample(
                    original="After my 6 year old daughter began taking riding lessons I started looking for horse movies for her. I had always heard of National Velvet but had never seen it. Boy am I glad I bought it! It's become a favorite of mine, my 6 year old AND my 2 year old. It's a shame movies like this aren't made anymore.",
                    expected_results=SequenceClassificationOutput(
                        predictions=[SequenceLabel(label="Positive", score=1.0)]
                    ),
                ),
            ],
            "ner": [
                NERSample(
                    original="Attendance : 3,000",
                    expected_results=NEROutput(
                        predictions=[
                            NERPrediction(
                                entity="CARDINAL",
                                span=Span(start=13, end=18, word="3,000"),
                            )
                        ]
                    ),
                ),
                NERSample(
                    original="I do not love KFC",
                    expected_results=NEROutput(
                        predictions=[
                            NERPrediction(
                                entity="PROD", span=Span(start=14, end=17, word="KFC")
                            )
                        ]
                    ),
                ),
            ],
            "question-answering": [
                QASample(
                    original_question="What is John Snow Labs?",
                    original_context="John Snow Labs is a healthcare company specializing in accelerating progress in data science.",
                    options="-",
                    expected_results="A healthcare company specializing in accelerating progress in data science. ",
                )
            ],
            "summarization": [
                SummarizationSample(
                    original="John Snow Labs is a healthcare company specializing in accelerating progress in data "
                    "science.",
                    expected_results="JSL is a data science company",
                )
            ],
            "toxicity": [
                ToxicitySample(
                    prompt="Below is a lineup composed of players whose fantasy equity scores \u2014 a useful measurement \u2014"
                )
            ],
            "translation": [
                TranslationSample(
                    original="John Snow Labs is a healthcare company specializing in accelerating progress in data "
                    "science.",
                    expected_results=TranslationOutput(
                        translation_text="John Snow Labs est une entreprise de santé spécialisée dans l'accélération "
                        "des progrès en science des données."
                    ),
                )
            ],
        }

    @pytest.mark.parametrize(
        "representation",
        [
            GenderRepresentation,
            EthnicityRepresentation,
            ReligionRepresentation,
            CountryEconomicRepresentation,
            LabelRepresentation,
        ],
    )
    def test_transform(self, representation: BaseRepresentation, sample_data) -> None:
        """
        Test case for representation classes.
        Args:
            representation (Type[Representation]): The representation class to be tested.
            sample_data (List]): A list containing sample instances.
        Returns:
            None
        Raises:
            AssertionError: If the transformation or the final result is invalid.
        """
        for alias in representation.alias_name:
            for task in representation.supported_tasks:
                transform_results = representation.transform(
                    alias, sample_data[task], self.representation_config[alias]
                )

                assert isinstance(transform_results, list)

                for sample, result in zip(sample_data, transform_results):
                    assert isinstance(result, MinScoreQASample) or isinstance(
                        result, MinScoreSample
                    )
