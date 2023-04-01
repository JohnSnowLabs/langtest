from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from nlptest.utils.custom_types import Sample, MinScoreOutput
from nlptest.modelhandler import ModelFactory


class BaseAccuracy(ABC):
    """
    Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented accuracy measure.
    """

    @staticmethod
    @abstractmethod
    def transform(y_true, y_pred):
        """
        Abstract method that implements the accuracy measure.

        Args:
            y_true: True values
            y_pred: Predicted values
            model (ModelFactory): Model to be evaluted.

        Returns:
            Any: The transformed data based on the implemented accuracy measure.
        """

        return NotImplementedError

    alias_name = None


class MinPrecisionScore(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name "min_precision_score" for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_precision_score"

    @staticmethod
    def transform(y_true, y_pred, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            List[Sample]: Precision test results.
        """
        labels = set(y_true).union(set(y_pred))

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                label: params["min_score"] for label in labels
            }

        df_metrics = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")

        precision_samples = []
        for k, v in df_metrics.items():
            if k not in min_scores.keys():
                continue
            sample = Sample(
                original="-",
                category="accuracy",
                test_type="min_precision_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k]),
                actual_results=MinScoreOutput(min_score=v["precision"]),
                state="done"
            )
            precision_samples.append(sample)
        return precision_samples


class MinRecallScore(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name "min_precision_score" for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_recall_score"

    @staticmethod
    def transform(y_true, y_pred, params):
        """
        Computes the minimum recall score for the given data.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            List[Sample]: Precision recall results.
        """

        labels = set(y_true).union(set(y_pred))

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                label: params["min_score"] for label in labels
            }

        df_metrics = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")

        rec_samples = []
        for k, v in df_metrics.items():
            if k not in min_scores.keys():
                continue
            sample = Sample(
                original="-",
                category="accuracy",
                test_type="min_recall_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k]),
                actual_results=MinScoreOutput(min_score=v["recall"]),
                state="done"
            )
            rec_samples.append(sample)
        return rec_samples


class MinF1Score(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name "min_precision_score" for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_f1_score"

    @staticmethod
    def transform(y_true, y_pred, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            List[Sample]: F1 score test results.
        """

        labels = set(y_true).union(set(y_pred))

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                label: params["min_score"] for label in labels
            }

        df_metrics = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")

        f1_samples = []
        for k, v in df_metrics.items():
            if k not in min_scores.keys():
                continue
            sample = Sample(
                original="-",
                category="accuracy",
                test_type="min_f1_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k]),
                actual_results=MinScoreOutput(min_score=v["f1-score"]),
                state="done"
            )
            f1_samples.append(sample)
        return f1_samples


class MinMicroF1Score(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_micro_f1_score"

    @staticmethod
    def transform(y_true, y_pred, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Any: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]

        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

        sample = Sample(
            original="-",
            category="accuracy",
            test_type="min_micro_f1_score",
            test_case="micro",
            expected_results=MinScoreOutput(min_score=min_score),
            actual_results=MinScoreOutput(min_score=f1),

            state="done"
        )

        return [sample]


class MinMacroF1Score(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name "min_precision_score" for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_macro_f1_score"

    @staticmethod
    def transform(y_true, y_pred, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Any: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        sample = Sample(
            original="-",
            category="accuracy",
            test_type="min__macro_f1_score",
            test_case="macro",
            expected_results=MinScoreOutput(min_score=min_score),
            actual_results=MinScoreOutput(min_score=f1),
            state="done"
        )

        return [sample]


class MinWeightedF1Score(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum weighted f1 score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_weighted_f1_score"

    @staticmethod
    def transform(y_true, y_pred, params):
        """
        Computes the minimum weighted F1 score for the given data.

        Args:
            y_true: True values
            y_pred: Predicted values   

        Returns:
            Any: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        sample = Sample(
            original="-",
            category="accuracy",
            test_type="min_weighted_f1_score",
            test_case="weighted",
            expected_results=MinScoreOutput(min_score=min_score),
            actual_results=MinScoreOutput(min_score=f1),
            state="done"
        )

        return [sample]
