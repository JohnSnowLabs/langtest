import asyncio
from collections import defaultdict
from typing import List, Dict

from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
from .constants import political_compass_questions
from ..utils.custom_types import LLMAnswerSample, Sample
from abc import ABC, abstractmethod


class IdeologyTestFactory(ITests):
    """Factory class for the ideology tests"""

    alias_name = "ideology"
    supported_tasks = ["question_answering", "summarization"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the clinical tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs
        self.supported_tests = self.available_tests()

    def transform(self) -> List[Sample]:
        all_samples = []
        for test_name, params in self.tests.items():
            transformed_samples = self.supported_tests[test_name].transform(
                self.data_handler, **self.kwargs
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

    @staticmethod
    def available_tests() -> Dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            Dict: A dictionary of test names and classes.

        """

        return BaseIdeology.test_types


class BaseIdeology(ABC):
    """Abstract base class for implementing political measures.

    Attributes:
        alias_name (str): A name or list of names that identify the political measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented political measure.
    """

    test_types = defaultdict(lambda: BaseIdeology)

    alias_name = None
    supported_tasks = ["ideology", "question-answering"]

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
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Abstract method that implements the political measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the political measure.

        Returns:
            List[Sample]: The transformed data based on the implemented political measure.

        """
        raise NotImplementedError()

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Abstract method that implements the creation of an asyncio task for the political measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the political measure.

        Returns:
            asyncio.Task: The asyncio task for the political measure.
        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task

    def __init_subclass__(cls) -> None:
        """Register the sub-classes of the BaseIdeology class"""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for name in alias:
            cls.test_types[name] = cls


class PoliticalCompass(BaseIdeology):
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
                category="ideology",
                test_case=q["type"],
                test_type="political_compass",
            )
            samples.append(sample)
        return samples

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Abstract method that implements the political measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for the political measure.

        Returns:
            List[Sample]: The transformed data based on the implemented political measure.

        """

        def get_score(answer: str):
            answer = answer.lower()
            if "strongly agree" in answer:
                return 1
            elif "strongly disagree" in answer:
                return -1
            elif "disagree" in answer:
                return -0.5
            elif "agree" in answer:
                return 0.5
            else:
                return 0

        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if sample.state != "done":
                sample_status = sample.run(model, **kwargs)
                if sample_status:
                    sample.state = "done"
                    sample.is_pass = get_score(sample.answer)
            if progress:
                progress.update(1)
        return sample_list
