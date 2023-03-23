from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from nlptest.utils.custom_types import Sample, MinScoreOutput, MaxScoreOutput
from nlptest.utils.gender_classifier import GenderClassifier

class BaseFairness(ABC):
    """
    Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented accuracy measure.
    """

    @staticmethod
    @abstractmethod
    def transform(data, model, params):

        """
        Abstract method that implements the accuracy measure.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented accuracy measure.
        """

        return NotImplementedError
    
    alias_name = None


class MinGenderF1Score(BaseFairness):

    """
    Subclass of BaseFairness that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_f1" identifying the minimum F1 score.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the minimum F1 score.
    """

    alias_name = "min_gender_f1_score"


    @staticmethod
    def transform(data: List[Sample], model, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the minimum F1 score.
        """
        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"]
            }

        gendered_data = get_gendered_data(data)

        samples = []

        for key, val in gendered_data.items():
            if key not in min_scores.keys():
                continue
            val = pd.Series(val, dtype="object")
            try:
                y_true = val.apply(lambda x: [y.entity for y in x.expected_results.predictions])
            except:
                y_true = val.apply(lambda x: [y.label for y in x.expected_results.predictions])
            X_test = val.apply(lambda x: x.original)

            y_pred = X_test.apply(model.predict_raw)
            
            valid_indices = y_true.apply(len) == y_pred.apply(len)
            # length_mismatch = valid_indices.count()-valid_indices.sum()
            # if length_mismatch > 0:
                # print(f"{length_mismatch} predictions have different lenghts than dataset and will be ignored.\nPlease make sure dataset and model uses same tokenizer.")
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]

            y_true = y_true.explode().apply(lambda x: x.split("-")[-1])
            y_pred = y_pred.explode().apply(lambda x: x.split("-")[-1])
            
            if len(y_true)>0:
                macro_f1_score = f1_score(y_true, y_pred, average="macro", zero_division=0)
            else:
                macro_f1_score = 1

            sample = Sample(
                original = "-",
                category = "fairness",
                test_type = "min_gender_f1_score",
                test_case = key,
                expected_results = MinScoreOutput(min_score=min_scores[key]),
                actual_results = MinScoreOutput(min_score=macro_f1_score),
                state = "done"
            )

            samples.append(sample)
        return samples

class MaxGenderF1Score(BaseFairness):

    """
    Subclass of BaseFairness that implements the maximum F1 score.

    Attributes:
        alias_name (str): The name identifying the test.

    Methods:
        transform(data: List[Sample]) -> List[Sample]: Transforms the input data into an output samples.
    """

    alias_name = "max_gender_f1_score"


    @staticmethod
    def transform(data: List[Sample], model, params):
        """
        Computes the gendered max F1 score tests for the given data.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            List[Sample]: The transformed samples.
        """
        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"]
            }
        
        gendered_data = get_gendered_data(data)

        samples = []
        for key, val in gendered_data.items():
            if key not in max_scores.keys():
                continue
            val = pd.Series(val, dtype="object")

            try:
                y_true = val.apply(lambda x: [y.entity for y in x.expected_results.predictions])
            except:
                y_true = val.apply(lambda x: [y.label for y in x.expected_results.predictions])
            X_test = val.apply(lambda x: x.original)
            y_true = pd.Series(val).apply(lambda x: [y.entity for y in x.expected_results.predictions])
            X_test = pd.Series(val).apply(lambda x: x.original)
            y_pred = X_test.apply(model.predict_raw)
            
            valid_indices = y_true.apply(len) == y_pred.apply(len)
            # length_mismatch = valid_indices.count()-valid_indices.sum()
            # if length_mismatch > 0:
                # print(f"{length_mismatch} predictions have different lenghts than dataset and will be ignored.\nPlease make sure dataset and model uses same tokenizer.")
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]

            y_true = y_true.explode().apply(lambda x: x.split("-")[-1])
            y_pred = y_pred.explode().apply(lambda x: x.split("-")[-1])

            if len(y_true)>0:
                macro_f1_score = f1_score(y_true, y_pred, average="macro", zero_division=0)
            else:
                macro_f1_score = 0
            
            sample = Sample(
                original = "-",
                category = "fairness",
                test_type = "max_gender_f1_score",
                test_case = key,
                expected_results = MaxScoreOutput(max_score=max_scores[key]),
                actual_results = MaxScoreOutput(max_score=macro_f1_score),
                state = "done"
            )

            samples.append(sample)
        return samples

def get_gendered_data(data):
    """Split list of samples into gendered lists."""
    data = pd.Series(data)
    sentences = pd.Series([x.original for x in data])
    classifier = GenderClassifier()
    genders = sentences.apply(classifier.predict)
    gendered_data = {
        "male": data[genders == "male"].tolist(),
        "female": data[genders == "female"].tolist(),
        "unknown": data[genders == "unknown"].tolist(),
    }
    return gendered_data
    