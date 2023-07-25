import asyncio
from typing import List
from abc import ABC, abstractmethod
from langtest.modelhandler.modelhandler import ModelFactory
from langtest.utils.custom_types.sample import Sample, SpeedTestSample


class BaseMeasure(ABC):
    """Abstract base class for implementing a robustness measure.

    This class defines the interface for implementing a robustness measure.

    Attributes:
        None
    """

    @staticmethod
    @abstractmethod
    def transform():
        """Abstract method that transforms the sample data based on the implemented robustness measure.

        Args:
            params (dict): The input data to be transformed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Sample]: The transformed data based on the implemented robustness measure.

        Raises:
            NotImplementedError: This method must be implemented in the derived class.
        """
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    @abstractmethod
    async def run(
        sample_list: List[Sample], model: ModelFactory, **kwargs
    ) -> List[Sample]:
        """Abstract method that implements the robustness measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the robustness measure.

        Returns:
            List[Sample]: The transformed data based on the implemented robustness measure.

        """
        progress = kwargs.get("progress_bar", False)
        for sample in kwargs.get('raw_data', []):
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
                else:
                    sample.expected_results = model(sample.original)
                    sample.actual_results = model(sample.test_case)
                    sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """Creates a task to run the robustness measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the robustness measure.

        Returns:
            asyncio.Task: The task that runs the robustness measure.

        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


class Speed(BaseMeasure):
    """Speed measure class.

    This class implements the speed measure. The speed measure is the time it takes for the model to run on the test case.

    """

    alias_name = "speed"
    supported_tasks = ["ner", "text-classification"]

    @staticmethod
    def transform(params: dict, *args, **kwargs) -> List[Sample]:
        """Transforms the sample data based on the implemented tests measure.

        Args:
            sample (Sample): The input data to be transformed.
            **kwargs: Additional arguments to be passed to the tests measure.

        Returns:
            Sample: The transformed data based on the implemented tests measure.

        """
        speed_samples = []
        for test_name, value in params.items():
            sample = SpeedTestSample()
            sample.category = "measure"
            sample.test_type = "speed"
            sample.expected_results = value
            speed_samples.append(sample)
        return speed_samples