import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List

from langtest.errors import Errors
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
from langtest.utils.custom_types.sample import Sample


class SafetyTestFactory(ITests):
    alias_name = "safety"
    supported_tasks = ["text-classification", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initialize a new SafetyTestFactory instance."""

        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """Execute the Safety test and return resulting `Sample` objects."""
        all_samples = []
        no_transformation_applied_tests = {}
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            if test_name not in self.supported_tests:
                no_transformation_applied_tests[test_name] = params
                continue
            test = self.supported_tests[test_name](self._data_handler, **params)
            all_samples.extend(test.transform())
        return all_samples

    @abstractmethod
    def available_tests(self) -> Dict:
        """Return a dictionary of available tests."""
        pass


class BaseSafetyTest(ABC):
    """Base class for Safety tests."""

    def __init__(self, data_handler: List[Sample], **kwargs) -> None:
        """Initialize a new BaseSafetyTest instance."""
        self._data_handler = data_handler
        self.kwargs = kwargs

    @abstractmethod
    def transform(self) -> List[Sample]:
        """Execute the Safety test and return resulting `Sample` objects."""
        pass

    @classmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Run the Safety test."""
        pass

    @classmethod
    async def async_run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Run the Safety test asynchronously."""
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))

        return await created_task


class Misuse(BaseSafetyTest):
    alias_name = "misuse"
    supported_tasks = ["text-classification", "text-generation", "question-answering"]
    """ Misuse test.
    """

    def transform(self) -> List[Sample]:
        """Execute the Misuse test and return resulting `Sample` objects."""
        pass

    @classmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Run the Misuse test."""
        pass

    @classmethod
    async def async_run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Run the Misuse test asynchronously."""
        pass

    def available_tests(self) -> Dict:
        """Return a dictionary of available tests."""
        pass
