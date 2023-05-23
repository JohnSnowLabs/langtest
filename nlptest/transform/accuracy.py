from abc import ABC, abstractmethod
from typing import Any, Dict, List

import asyncio
import logging
import evaluate

from sklearn.metrics import classification_report, f1_score
from nlptest.utils.custom_types import MinScoreOutput, MinScoreSample

class BaseAccuracy(ABC):
    """
    Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented accuracy measure.
    """
    alias_name = None
    supported_tasks = ["ner", "text-classification"]

    @staticmethod
    @abstractmethod
    def transform(y_true: List[Any]):
        """
        Abstract method that implements the accuracy measure.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the implemented accuracy measure.
        """

        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs) -> List[MinScoreSample]:
        return NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[MinScoreSample],  y_true, y_pred, **kwargs):
        """ 
        Creates a task to run the accuracy measure.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        created_task = asyncio.create_task(
            cls.run(sample_list, y_true, y_pred, **kwargs))
        return created_task


class MinPrecisionScore(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_precision_score"

    @staticmethod
    def transform(y_true: List[Any], params: Dict):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration


        Returns:
            List[MinScoreSample]: Precision test results.
        """
        labels = set(y_true)  # .union(set(y_pred))

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                label: params["min_score"] for label in labels
            }

        precision_samples = []
        for k in labels:
            if k not in min_scores.keys():
                continue
            sample = MinScoreSample(
                original="-",
                category="accuracy",
                test_type="min_precision_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k])
            )
            precision_samples.append(sample)
        return precision_samples

    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):
        """
        Computes the minimum precision score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)
        df_metrics = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")

        for idx, sample in enumerate(sample_list):
            if progress:
                progress.update(1)
            if sample.test_case not in df_metrics:
                sample_list.pop(idx)
                
                continue
            precision = df_metrics.get(sample.test_case)
            sample.actual_results = MinScoreOutput(
                min_score=precision['precision'])
            sample.state = "done"

        return sample_list


class MinRecallScore(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_recall_score"

    @staticmethod
    def transform(y_true: List[Any], params: Dict):
        """
        Computes the minimum recall score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: Precision recall results.
        """

        labels = set(y_true)  # .union(set(y_pred))

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                label: params["min_score"] for label in labels
            }

        rec_samples = []
        for k in labels:
            if k not in min_scores.keys():
                continue
            sample = MinScoreSample(
                original="-",
                category="accuracy",
                test_type="min_recall_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k])
            )
            rec_samples.append(sample)
        return rec_samples

    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):
        """
        Computes the minimum recall score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        df_metrics = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")

        for idx, sample in enumerate(sample_list):
            if progress:
                progress.update(1)
            if sample.test_case not in df_metrics:
                sample_list.pop(idx)
                
                continue
            precision = df_metrics.get(sample.test_case)
            sample.actual_results = MinScoreOutput(
                min_score=precision['recall'])
            sample.state = "done"

        return sample_list


class MinF1Score(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_f1_score"

    @staticmethod
    def transform(y_true: List[Any], params: Dict):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: F1 score test results.
        """

        labels = set(y_true)

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                label: params["min_score"] for label in labels
            }

        f1_samples = []
        for k in labels:
            if k not in min_scores.keys():
                continue
            sample = MinScoreSample(
                original="-",
                category="accuracy",
                test_type="min_f1_score",
                test_case=k,
                expected_results=MinScoreOutput(min_score=min_scores[k])
            )
            f1_samples.append(sample)
        return f1_samples

    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):
        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        df_metrics = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")

        for idx, sample in enumerate(sample_list):
            if progress:
                progress.update(1)

            if sample.test_case not in df_metrics:
                sample_list.pop(idx)
                continue
            f1_scores = df_metrics.get(sample.test_case)
            sample.actual_results = MinScoreOutput(
                min_score=f1_scores['f1-score'])
            sample.state = "done"

        return sample_list


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
    def transform(y_true: List[Any], params: Dict):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]

        sample = MinScoreSample(
            original="-",
            category="accuracy",
            test_type="min_micro_f1_score",
            test_case="micro",
            expected_results=MinScoreOutput(min_score=min_score)
        )

        return [sample]

    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):
        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=f1)
            sample.state = "done"
            if progress:
                progress.update(1)

        return sample_list


class MinMacroF1Score(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_macro_f1_score"

    @staticmethod
    def transform(y_true, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]

        sample = MinScoreSample(
            original="-",
            category="accuracy",
            test_type="min_macro_f1_score",
            test_case="macro",
            expected_results=MinScoreOutput(min_score=min_score)
        )

        return [sample]

    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):

        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=f1)
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list


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
    def transform(y_true: List[Any], params: Dict):
        """
        Computes the minimum weighted F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]

        sample = MinScoreSample(
            original="-",
            category="accuracy",
            test_type="min_weighted_f1_score",
            test_case="weighted",
            expected_results=MinScoreOutput(min_score=min_score)
        )

        return [sample]

    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):
        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=f1)
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

class MinEMcore(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_exact_match_score"
    supported_tasks = ["question-answering", "summarization"]

    @staticmethod
    def transform(y_true, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]

        sample = MinScoreSample(
            category="accuracy",
            test_type="min_macro_f1_score",
            expected_results=MinScoreOutput(min_score=min_score)
        )

        return [sample]

    @staticmethod
    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):

        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)

        em = evaluate.load("exact_match")
        y_true = [x[0] for x in y_true]
        result = em.compute(references=y_true, predictions=y_pred)["exact_match"]
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=result)
            sample.state = "done"
            if progress:
                progress.update(1)
                
        return sample_list

class MinBLEUcore(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = "min_bleu_score"
    supported_tasks = ["question-answering", "summarization"]

    @staticmethod
    def transform(y_true, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]

        sample = MinScoreSample(
            category="accuracy",
            test_type="min_bleu_score",
            expected_results=MinScoreOutput(min_score=min_score)
        )

        return [sample]

    @staticmethod
    async def run(sample_list: List[MinScoreSample], y_true, y_pred, **kwargs):

        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)
        em = evaluate.load("bleu")
        result = em.compute(references=y_true, predictions=y_pred)
        y_true = [[f'The answer is {y}' for y in x] for x in y_true]
        y_pred = [f'The answer is {x}' for x in y_pred]
        
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=result["bleu"])
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

class MinROUGEcore(BaseAccuracy):
    """
    Subclass of BaseAccuracy that implements the minimum precision score.

    Attributes:
        alias_name (str): The name for config.

    Methods:
        transform(y_true, y_pred) -> Any: Creates accuracy test results.
    """

    alias_name = ["min_rouge1_score","min_rouge2_score","min_rougeL_score","min_rougeLsum_score"]
    supported_tasks = ["question-answering", "summarization"]

    @staticmethod
    def transform(y_true, params):
        """
        Computes the minimum F1 score for the given data.

        Args:
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values
            params (Dict): parameters for tests configuration

        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        min_score = params["min_score"]

        sample = MinScoreSample(
            category="accuracy",
            test_type=params["test_name"],
            expected_results=MinScoreOutput(min_score=min_score)
        )

        return [sample]

    @staticmethod
    async def run(sample_list: List[MinScoreSample], y_true, y_pred,**kwargs):

        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): List of samples to be transformed.
            y_true (List[Any]): True values
            y_pred (List[Any]): Predicted values

        """
        progress = kwargs.get("progress_bar", False)
        em = evaluate.load("rouge")
        result = em.compute(references=y_true, predictions=y_pred)
        
        for sample in sample_list:
            sample.actual_results = MinScoreOutput(min_score=result[sample.test_type.split('_')[1]])
            sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list
