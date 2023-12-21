import pytest

from langtest.transform.fairness import (
    BaseFairness,
    MinGenderF1Score,
    MaxGenderF1Score,
    MinGenderRougeScore,
    MaxGenderRougeScore,
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
    MaxScoreQASample,
    MaxScoreSample,
    MinScoreSample,
    NERSample,
    QASample,
    SequenceClassificationSample,
    SummarizationSample,
    ToxicitySample,
    TranslationSample,
)


class Testfairness:
    """A test suite for evaluating the transformation process of various fairnesss.

    This test suite ensures that the fairnesss can successfully transform input data
    and produce valid results.

    The fairnesss tested include Genderfairness, Ethnicityfairness,
    Religionfairness, and CountryEconomicfairness.

    Attributes:
        fairness_config (Dict)
    """

    fairness_config = {
        "min_gender_f1_score": {"min_score": 0.66},
        "max_gender_f1_score": {"max_score": 0.60},
        "min_gender_rouge1_score": {"min_score": 0.66},
        "min_gender_rouge2_score": {"min_score": 0.60},
        "min_gender_rougeL_score": {"min_score": 0.66},
        "min_gender_rougeLsum_score": {"min_score": 0.66},
        "max_gender_rouge1_score": {"max_score": 0.66},
        "max_gender_rouge2_score": {"max_score": 0.60},
        "max_gender_rougeL_score": {"max_score": 0.66},
        "max_gender_rougeLsum_score": {"max_score": 0.66},
    }

    @pytest.fixture
    def sample_data(self):
        """A fixture providing sample data for the fairness transformation tests.

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
        }

    @pytest.mark.parametrize(
        "fairness",
        [
            MinGenderF1Score,
            MaxGenderF1Score,
            MinGenderRougeScore,
            MaxGenderRougeScore,
        ],
    )
    def test_transform(self, fairness: BaseFairness, sample_data) -> None:
        """
        Test case for fairness classes.

        Args:
            fairness (Type[fairness]): The fairness class to be tested.
            sample_data (List]): A list containing sample instances.

        Returns:
            None

        Raises:
            AssertionError: If the transformation or the final result is invalid.
        """
        for alias in fairness.alias_name:
            for task in fairness.supported_tasks:
                transform_results = fairness.transform(
                    alias, sample_data[task], self.fairness_config[alias]
                )
                assert isinstance(transform_results, list)

                for _, result in zip(sample_data, transform_results):
                    assert isinstance(result, MaxScoreSample) or isinstance(
                        result, MinScoreSample
                    )
