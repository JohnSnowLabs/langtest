import asyncio
import time
from collections import defaultdict
from typing import List, Dict
from langtest.errors import Errors
from abc import ABC, abstractmethod
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
from langtest.utils.custom_types.sample import Sample, SpeedTestSample


class PerformanceTestFactory(ITests):
    """Factory class for the model performance

    This class implements the model performance The robustness measure is the number of test cases that the model fails to run on.

    """

    alias_name = "performance"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the model performance"""

        self.supported_tests = self.available_tests()
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Transforms the sample data based on the implemented tests measure.

        Args:
            sample (Sample): The input data to be transformed.
            **kwargs: Additional arguments to be passed to the tests measure.

        Returns:
            Sample: The transformed data based on the implemented
            tests measure.

        """
        all_samples = []
        for test_name, params in self.tests.items():
            transformed_samples = self.supported_tests[test_name].transform(
                params=params, **self.kwargs
            )
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the model performance

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the model performance

        Returns:
            List[Sample]: The transformed data based on the implemented model performance

        """
        supported_tests = cls.available_tests()
        tasks = []
        for test_name, samples in sample_list.items():
            out = await supported_tests[test_name].async_run(samples, model, **kwargs)
            if isinstance(out, list):
                tasks.extend(out)
            else:
                tasks.append(out)

        return tasks

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the available model performance

        Returns:
            Dict[str, str]: The available model performance

        """
        return BasePerformance.test_types


class BasePerformance(ABC):
    """Abstract base class for implementing a model performance.

    This class defines the interface for implementing a model performance.

    Attributes:
        None
    """

    test_types = defaultdict(lambda: BasePerformance)
    alias_name = None
    TOKENS = 0

    @staticmethod
    @abstractmethod
    def transform():
        """Abstract method that transforms the sample data based on the implemented model performance.

        Args:
            params (dict): The input data to be transformed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Sample]: The transformed data based on the implemented model performance.

        Raises:
            NotImplementedError: This method must be implemented in the derived class.
        """
        raise NotImplementedError(Errors.E063())

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Abstract method that implements the model performance.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the model performance.

        Returns:
            List[Sample]: The transformed data based on the implemented model performance.

        """
        progress = kwargs.get("progress_bar", False)
        BasePerformance.TOKENS = 0
        for sample in kwargs.get("raw_data", []):
            if hasattr(sample, "run"):
                sample_status = sample.run(model, **kwargs)
                if sample_status:
                    BasePerformance.TOKENS += sample_status
                    sample.state = "done"
            else:
                BasePerformance.TOKENS += len(sample.original.split())
                _ = model(sample.original)
        if progress:
            progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Creates a task to run the model performance.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the model performance.

        Returns:
            asyncio.Task: The task that runs the model performance.

        """
        start_time = time.time_ns()
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        created_task.add_done_callback(
            lambda x: cls.time_measure(start_time, sample_list, cls.TOKENS)
        )
        return await created_task

    @staticmethod
    def time_measure(start_time, sample_list, tokens):
        end_time = time.time_ns()
        time_taken = end_time - start_time  # in nanoseconds

        for sample in sample_list:
            sample.total_time(time_taken, tokens)

        return sample_list

    def __init_subclass__(cls) -> None:
        """Registers the test types"""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]

        for alias_name in alias:
            BasePerformance.test_types[alias_name] = cls


class Speed(BasePerformance):
    """Speed measure class.

    This class implements the speed measure. The speed measure is the time it takes for the model to run on the test case.

    """

    alias_name = ["speed"]
    supported_tasks = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
    ]

    @staticmethod
    def transform(params: dict, *args, **kwargs) -> List[Sample]:
        """Transforms the sample data based on the implemented tests measure.

        Args:
            sample (Sample): The input data to be transformed.
            **kwargs: Additional arguments to be passed to the tests measure.

        Returns:
            Sample: The transformed data based on the implemented tests measure.

        """
        sample = SpeedTestSample()
        sample.expected_results = "{min_pass_rate} {unit}".format(**params)
        return [sample]
