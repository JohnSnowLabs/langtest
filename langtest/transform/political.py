import asyncio
from typing import List

from langtest.modelhandler.modelhandler import ModelFactory
from .constants import political_compass_questions
from ..utils.custom_types import LLMAnswerSample, Sample
from abc import ABC, abstractmethod


class BasePolitical(ABC):
    """Abstract base class for implementing political measures.

    Attributes:
        alias_name (str): A name or list of names that identify the political measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented political measure.
    """

    alias_name = None
    supported_tasks = ["political"]

    @abstractmethod
    def transform(self, sample_list: List[Sample], *args, **kwargs) -> List[Sample]:
        """Abstract method that implements the political measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.

        Returns:
            List[Sample]: The transformed data based on the implemented political measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(
        sample_list: List[Sample], model: ModelFactory, **kwargs
    ) -> List[Sample]:
        """Abstract method that implements the political measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the political measure.

        Returns:
            List[Sample]: The transformed data based on the implemented political measure.

        """
        raise NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """Abstract method that implements the creation of an asyncio task for the political measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the political measure.

        Returns:
            asyncio.Task: The asyncio task for the political measure.
        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


class PoliticalCompass(BasePolitical):
    """Class for religious politicales"""

    alias_name = "political_compass"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Replace  names to check the religion political

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            names_to_substitute (List[str]): list of names that need to be substituted.
            chosen_names (List[str]): list of names to replace with.

        Returns:
            List[Sample]: List of sentences with replaced names
        """

        samples = []
        for q in political_compass_questions:
            sample = LLMAnswerSample(
                question=q["question"],
                answer="",
                category="political",
                test_case=q["type"],
                test_type="political_compass",
            )
            samples.append(sample)
        return samples

    @staticmethod
    async def run(
        sample_list: List[Sample], model: ModelFactory, **kwargs
    ) -> List[Sample]:
        """Abstract method that implements the political measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the political measure.

        Returns:
            List[Sample]: The transformed data based on the implemented political measure.

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
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
