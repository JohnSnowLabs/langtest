import pytest
import pandas as pd
from langtest.transform.accuracy import (
    BaseAccuracy,
    MinPrecisionScore,
    MinF1Score,
    MinMicroF1Score,
    MinMacroF1Score,
    MinWeightedF1Score,
    MinEMcore,
    MinBLEUcore,
    MinROUGEcore,
)
from langtest.utils.custom_types import SequenceLabel, Span
from langtest.utils.custom_types.output import (
    NEROutput,
    NERPrediction,
    SequenceClassificationOutput,
)
from langtest.utils.custom_types.sample import (
    MinScoreSample,
    NERSample,
    QASample,
    SequenceClassificationSample,
    SummarizationSample,
)


class TestAccuracy:
    """
    A test suite for evaluating accuracy classes.
    """

    accuracy_config = {
        "min_precision_score": {"min_score": 0.66},
        "min_recall_score": {"min_score": 0.60},
        "min_f1_score": {"min_score": 0.60},
        "min_micro_f1_score": {"min_score": 0.60},
        "min_macro_f1_score": {"min_score": 0.60},
        "min_weighted_f1_score": {"min_score": 0.60},
        "min_bleu_score": {"min_score": 0.66},
        "min_exact_match_score": {"min_score": 0.60},
        "min_rouge1_score": {"min_score": 0.60},
        "min_rouge2_score": {"min_score": 0.60},
        "min_rougeL_score": {"min_score": 0.60},
        "min_rougeLsum_score": {"min_score": 0.60},
    }

    @pytest.fixture
    def sample_data(self):
        """A fixture providing sample data for testing.

        Returns:
            dict: A dictionary containing sample data for different tasks.
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
        "accuracy",
        [
            MinPrecisionScore,
            MinF1Score,
            MinMicroF1Score,
            MinMacroF1Score,
            MinWeightedF1Score,
            MinEMcore,
            MinBLEUcore,
            MinROUGEcore,
        ],
    )
    def test_transform(self, accuracy: BaseAccuracy, sample_data) -> None:
        """Test the transform method of accuracy-related classes.

        Args:
            accuracy (BaseAccuracy): An accuracy-related class to test.
            sample_data (dict): Sample data for different tasks.

        Returns:
            None
        """
        for alias in accuracy.alias_name:
            for task in accuracy.supported_tasks:
                if task == "text-classification":
                    y_true = (
                        pd.Series(sample_data["text-classification"])
                        .apply(
                            lambda x: [y.label for y in x.expected_results.predictions]
                        )
                        .explode()
                    )
                elif task == "ner":
                    y_true = pd.Series(sample_data["ner"]).apply(
                        lambda x: [y.entity for y in x.expected_results.predictions]
                    )
                    y_true = y_true.explode().apply(
                        lambda x: x.split("-")[-1] if isinstance(x, str) else x
                    )

                else:
                    y_true = (
                        pd.Series(sample_data[task])
                        .apply(lambda x: x.expected_results)
                        .explode()
                    )
                transform_results = accuracy.transform(
                    alias, y_true, self.accuracy_config[alias]
                )
                assert isinstance(transform_results, list)

                for _, result in zip(y_true, transform_results):
                    assert isinstance(result, MinScoreSample)
