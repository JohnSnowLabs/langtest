from abc import ABC, abstractmethod
import asyncio
from typing import List

import evaluate
import pandas as pd
from sklearn.metrics import f1_score
from nlptest.modelhandler.modelhandler import ModelFactory

from nlptest.utils.custom_types import MaxScoreOutput, MaxScoreSample, MinScoreOutput, MinScoreSample, Sample
from nlptest.utils.custom_types.sample import QASample


class BaseFairness(ABC):
    """
    Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an 
        output based on the implemented accuracy measure.
    """
    alias_name = None
    supported_tasks = ["ner", "text-classification", "question-answering", "summarization"]

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

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[MinScoreSample], categorised_data, **kwargs) -> List[Sample]:
        return NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """
        Creates a task for the run method.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.

        Returns:
            asyncio.Task: The task for the run method.
            
        """
        created_task = asyncio.create_task(
            cls.run(sample_list, model, **kwargs))
        return created_task


class MinGenderF1Score(BaseFairness):
    """
    Subclass of BaseFairness that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_f1" identifying the minimum F1 score.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into 
        an output based on the minimum F1 score.
    """

    alias_name = "min_gender_f1_score"

    @staticmethod
    def transform(data: List[Sample], params):
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

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type="min_gender_f1_score",
                test_case=key,
                expected_results=MinScoreOutput(min_score=val)
            )

            samples.append(sample)
        return samples
    
    @staticmethod
    async def run(sample_list: List[MinScoreSample], gendered_data, **kwargs) -> List[MinScoreSample]:
        """
        Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.
        
            
        Returns:
            List[MinScoreSample]: The transformed samples.

        """
        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = gendered_data[sample.test_case]
            if len(data[0]) > 0:
                if task == QASample:
                    em = evaluate.load("f1")
                    macro_f1_score = em.compute(references=data[0], predictions=data[1], average="macro")
                else:
                    macro_f1_score = f1_score([x[0] for x in data[0]], data[1], average="macro", zero_division=0)
            else:
                macro_f1_score = 1

            sample.actual_results = MinScoreOutput(min_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderF1Score(BaseFairness):
    """
    Subclass of BaseFairness that implements the maximum F1 score.

    Attributes:
        alias_name (str): The name to be used in config.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into 
        an output based on the maximum F1 score.
    """

    alias_name = "max_gender_f1_score"

    @staticmethod
    def transform(data: List[Sample], params):
        """
        Computes the maximum F1 score for the given data.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the maximum F1 score.
        """
        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"]
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type="max_gender_f1_score",
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val)
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(sample_list: List[MaxScoreSample], gendered_data, **kwargs) -> List[MaxScoreSample]:
        """
        Computes the maximum F1 score for the given data.

        Args:
            sample_list (List[MaxScoreSample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.
        
            
        Returns:
            List[MaxScoreSample]: The transformed samples.

        """
        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = gendered_data[sample.test_case]
            if len(data[0]) > 0:
                if task == QASample:
                    em = evaluate.load("f1")
                    macro_f1_score = em.compute(references=data[0], predictions=data[1], average="macro")
                else:
                    macro_f1_score = f1_score(data[0], data[1], average="macro", zero_division=0)
            else:
                macro_f1_score = 1

            sample.actual_results = MaxScoreOutput(max_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list