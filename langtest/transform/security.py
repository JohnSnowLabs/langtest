from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
from typing import List, Dict, TypedDict
from langtest.errors import Errors
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
from langtest.utils.custom_types.sample import Sample


class SecurityTestFactory(ITests):

    """Factory class for the security tests"""

    alias_name = "security"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        self.supported_tests = self.available_tests()
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

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
        """
        Register the sub-classes of the BaseSecurity class
        """
        return BaseSecurity.test_types


class BaseSecurity(ABC):
    """Abstract base class for implementing a model security.

    This class defines the interface for implementing a model security.

    Attributes:
        None
    """

    test_types = defaultdict(lambda: BaseSecurity)
    alias_name = None

    # TestConfig
    TestConfig = TypedDict(
        "TestConfig",
        min_pass_rate=float,
    )

    @staticmethod
    @abstractmethod
    def transform():
        """Abstract method that transforms the sample data based on the implemented model security."""
        raise NotImplementedError(Errors.E063)

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Abstract method that implements the model security."""
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
                else:
                    sample.actual_results = model(sample.prompt)
                    sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Abstract method that implements the model security."""
        created_task = await asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task

    def __init_subclass__(cls) -> None:
        """Registers the test types"""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]

        for alias_name in alias:
            BaseSecurity.test_types[alias_name] = cls


class PromptInjection(BaseSecurity):
    """
    PromptInjection is a class that implements the model security for prompt injection.
    """

    alias_name = ["prompt_injection_attack"]
    supported_tasks = [
        "security",
        "text-generation",
    ]

    def transform(sample_list: List[Sample], *args, **kwargs):
        """"""
        for sample in sample_list:
            sample.test_type = "prompt_injection_attack"
            sample.category = "security"

        return sample_list
