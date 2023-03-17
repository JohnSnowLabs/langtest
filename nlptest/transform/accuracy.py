
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from nlptest.utils.custom_types import Sample, AccuracyOutput
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
    def transform(data, model_handler):

        """
        Abstract method that implements the accuracy measure.

        Args:
            data (List[Sample]): The input data to be transformed.
            model (ModelFactory): Model to be evaluted.

        Returns:
            Any: The transformed data based on the implemented accuracy measure.
        """

        return NotImplementedError
    
    alias_name = None


class MinPrecisionScore(BaseAccuracy):

    """
    Subclass of BaseAccuracy that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_f1" identifying the minimum F1 score.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the minimum F1 score.
    """

    alias_name = "min_precision_score"

    @staticmethod
    def transform(data: List[Sample], model_handler: ModelFactory):
        """
        Computes the minimum F1 score for the given data.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the minimum F1 score.
        """

        y_true = pd.Series(data).apply(lambda x: [y.entity for y in x.expected_results.predictions])
        X_test = pd.Series(data).apply(lambda x: x.original)
        y_pred = X_test.apply(model_handler.predict_raw)

        valid_indices = y_true.apply(len) == y_pred.apply(len)
        # length_mismatch = valid_indices.count() - valid_indices.sum()
        # if length_mismatch > 0:
        #   print(
        #       f"{length_mismatch} predictions have different lenghts than dataset and will be ignored.\nPlease make sure dataset and model uses same tokenizer.")
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

        y_true = y_true.explode().apply(lambda x: x.split("-")[-1])
        y_pred = y_pred.explode().apply(lambda x: x.split("-")[-1])

        # y_pred = [x.split("-")[-1] for x in y_pred.tolist()]

        # if(len(y_pred) != len(y_true)):
        # raise ValueError("Please use the dataset used to train/test the model. Model and dataset has different tokenizers.")

        df_metrics = classification_report(y_true, y_pred, output_dict=True)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")

        precision_samples = []
        for k, v in df_metrics.items():
            sample = Sample(
                original = "-",
                category = "Accuracy",
                test_type = "min_precision_score",
                test_case = k,
                expected_results = AccuracyOutput(score=0),
                actual_results = AccuracyOutput(score=v["precision"]),
                state = "done"
            )
            precision_samples.append(sample)
            
        return precision_samples
    

