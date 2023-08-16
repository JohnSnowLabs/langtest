import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Union

from langtest.modelhandler.modelhandler import ModelFactory
from langtest.utils.custom_types import (
    MaxScoreOutput,
    MaxScoreSample,
    MinScoreOutput,
    MinScoreSample,
    Sample,
)
from langtest.utils.util_metrics import calculate_f1_score


class BaseFairness(ABC):
    """Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample], params: Dict) -> Union[List[MinScoreSample], List[MaxScoreSample]]:
            Transforms the input data into an output based on the implemented accuracy measure.
    """

    alias_name = None
    supported_tasks = ["ner", "text-classification"]

    @staticmethod
    @abstractmethod
    def transform(
        data: List[Sample], params: Dict
    ) -> Union[List[MinScoreSample], List[MaxScoreSample]]:
        """Abstract method that implements the computation of the given measure.

        Args:
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration
        Returns:
            Union[List[MinScoreSample], List[MaxScoreSample]]: The transformed data based on the implemented measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(
        sample_list: List[MinScoreSample], categorised_data, **kwargs
    ) -> List[Sample]:
        """Computes the score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.

        Returns:
            List[MinScoreSample]: The transformed samples.
        """
        raise NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """Creates a task for the run method.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.

        Returns:
            asyncio.Task: The task for the run method.

        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


class MinGenderF1Score(BaseFairness):
    """Subclass of BaseFairness that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_f1" identifying the minimum F1 score.

    Methods:
        transform(data: List[Sample], params: Dict) -> List[MinScoreSample]:
            Transforms the input data into an output based on the minimum F1 score.
    """

    alias_name = ["min_gender_f1_score"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the minimum F1 score for the given data.

        Args:
            test (str): name of the test
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration.
        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """

        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type="min_gender_f1_score",
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], gendered_data, **kwargs
    ) -> List[MinScoreSample]:
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.

        Returns:
            List[MinScoreSample]: The transformed samples.

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            data = gendered_data[sample.test_case]
            if len(data[0]) > 0:
                macro_f1_score = calculate_f1_score(
                    data[0].to_list(), data[1].to_list(), average="macro", zero_division=0
                )
            else:
                macro_f1_score = 1

            sample.actual_results = MinScoreOutput(min_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderF1Score(BaseFairness):
    """Subclass of BaseFairness that implements the maximum F1 score.

    Attributes:
        alias_name (str): The name to be used in config.

    Methods:
        transform(data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms the input data into an output based on the maximum F1 score.
    """

    alias_name = ["max_gender_f1_score"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """Computes the maximum F1 score for the given data.

        Args:
            test (str): name of the test.
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration
        Returns:
            List[MaxScoreSample]: The transformed data based on the maximum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"
        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type="max_gender_f1_score",
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], gendered_data, **kwargs
    ) -> List[MaxScoreSample]:
        """Computes the maximum F1 score for the given data.

        Args:
            sample_list (List[MaxScoreSample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.


        Returns:
            List[MaxScoreSample]: The transformed samples.

        """
        progress = kwargs.get("progress_bar", False)

        for sample in sample_list:
            data = gendered_data[sample.test_case]
            if len(data[0]) > 0:
                macro_f1_score = calculate_f1_score(
                    data[0].to_list(), data[1].to_list(), average="macro", zero_division=0
                )
            else:
                macro_f1_score = 1

            sample.actual_results = MaxScoreOutput(max_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MinGenderRougeScore(BaseFairness):
    """Subclass of BaseFairness that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_f1" identifying the minimum F1 score.

    Methods:
        transform(data: List[Sample], params: Dict) -> List[MinScoreSample]:
            Transforms the input data into an output based on the minimum F1 score.
    """

    alias_name = [
        "min_gender_rouge1_score",
        "min_gender_rouge2_score",
        "min_gender_rougeL_score",
        "min_gender_rougeLsum_score",
    ]
    supported_tasks = ["question-answering", "summarization"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MinScoreSample]:
        """Computes the min rouge score for the given data.

        Args:
            test (str): name of the test.
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration
        Returns:
            List[MinScoreSample]: The transformed data based on the minimum F1 score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["min_score"], dict):
            min_scores = params["min_score"]
        elif isinstance(params["min_score"], float):
            min_scores = {
                "male": params["min_score"],
                "female": params["min_score"],
                "unknown": params["min_score"],
            }

        samples = []
        for key, val in min_scores.items():
            sample = MinScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MinScoreOutput(min_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MinScoreSample], gendered_data, **kwargs
    ) -> List[MinScoreSample]:
        """Computes the minimum F1 score for the given data.

        Args:
            sample_list (List[MinScoreSample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.

        Returns:
            List[MinScoreSample]: The transformed samples.

        """
        import evaluate

        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = gendered_data[sample.test_case]
            if len(data[0]) > 0:
                if task == "question-answering" or task == "summarization":
                    em = evaluate.load("rouge")
                    macro_f1_score = em.compute(references=data[0], predictions=data[1])[
                        sample.test_type.split("_")[2]
                    ]
                else:
                    macro_f1_score = calculate_f1_score(
                        [x[0] for x in data[0]], data[1], average="macro", zero_division=0
                    )
            else:
                macro_f1_score = 1

            sample.actual_results = MinScoreOutput(min_score=macro_f1_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list


class MaxGenderRougeScore(BaseFairness):
    """Subclass of BaseFairness that implements the rouge score.

    Attributes:
        alias_name (str): The name to be used in config.

    Methods:
        transform(data: List[Sample], params: Dict) -> List[MaxScoreSample]:
            Transforms the input data into an output based on the rouge score.
    """

    alias_name = [
        "max_gender_rouge1_score",
        "max_gender_rouge2_score",
        "max_gender_rougeL_score",
        "max_gender_rougeLsum_score",
    ]
    supported_tasks = ["question-answering", "summarization"]

    @classmethod
    def transform(
        cls, test: str, data: List[Sample], params: Dict
    ) -> List[MaxScoreSample]:
        """Computes the rouge score for the given data.

        Args:
            test (str): name of the test.
            data (List[Sample]): The input data to be transformed.
            params (Dict): parameters for tests configuration
        Returns:
            List[MaxScoreSample]: The transformed data based on the rouge score.
        """
        assert (
            test in cls.alias_name
        ), f"Parameter 'test' should be in: {cls.alias_name}, got '{test}'"

        if isinstance(params["max_score"], dict):
            max_scores = params["max_score"]
        elif isinstance(params["max_score"], float):
            max_scores = {
                "male": params["max_score"],
                "female": params["max_score"],
                "unknown": params["max_score"],
            }

        samples = []
        for key, val in max_scores.items():
            sample = MaxScoreSample(
                original=None,
                category="fairness",
                test_type=test,
                test_case=key,
                expected_results=MaxScoreOutput(max_score=val),
            )

            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[MaxScoreSample], gendered_data, **kwargs
    ) -> List[MaxScoreSample]:
        """Computes the maximum rouge score for the given data.

        Args:
            sample_list (List[MaxScoreSample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the computation.

        Returns:
            List[MaxScoreSample]: The transformed samples.

        """
        import evaluate

        progress = kwargs.get("progress_bar", False)
        task = kwargs.get("task", None)

        for sample in sample_list:
            data = gendered_data[sample.test_case]
            if len(data[0]) > 0:
                if task == "question-answering" or task == "summarization":
                    em = evaluate.load("rouge")
                    rouge_score = em.compute(references=data[0], predictions=data[1])[
                        sample.test_type.split("_")[2]
                    ]
                else:
                    rouge_score = calculate_f1_score(
                        [x[0] for x in data[0]], data[1], average="macro", zero_division=0
                    )
            else:
                rouge_score = 1

            sample.actual_results = MaxScoreOutput(max_score=rouge_score)
            sample.state = "done"

            if progress:
                progress.update(1)
        return sample_list
