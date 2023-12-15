import asyncio
from typing import List, Dict
from langtest.utils.custom_types.sample import Sample
from abc import ABC, abstractmethod
from langtest.errors import Errors
from langtest.transform import ITests, TestFactory
from langtest.modelhandler.modelhandler import ModelAPI


class GrammerTestFactory(ITests):
    alias_name = "grammer"
    supported_tasks = ["text-classification", "text-generation", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initialize a new SycophancyTestFactory instance.

        Args:
            data_handler (List[Sample]): A list of `Sample` objects representing the input dataset.
            tests (Optional[Dict]): A dictionary of test names and corresponding parameters (default is None).
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the `tests` argument is not a dictionary.

        """
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048)

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049.format(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """Execute the Sycophancy test and return resulting `Sample` objects.

        Returns:
            List[Sample]: A list of `Sample` objects representing the resulting dataset
            after conducting the Sycophancy test.

        """
        all_samples = []
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            if TestFactory.is_augment:
                data_handler_copy = [x.copy() for x in self._data_handler]
            else:
                data_handler_copy = [x.copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform

            _ = [
                sample.transform(
                    test_func,
                    params.get("parameters", {}),
                )
                if hasattr(sample, "transform")
                else sample
                for sample in data_handler_copy
            ]
            transformed_samples = data_handler_copy

            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @staticmethod
    def available_tests() -> dict:
        """
        Retrieve a dictionary of all available tests, with their names as keys
        and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """
        tests = {
            j: i
            for i in BaseGrammer.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class BaseGrammer(ABC):
    @staticmethod
    @abstractmethod
    def transform():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def run(sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Abstract method that implements the model testing for to check the grammer issues."""

        progress_bar = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if hasattr(sample, "run"):
                sample.run(model, **kwargs)
            else:
                sample.expected_results = model.predict(sample.original)
                sample.actual_results = model.predict(sample.test_case)
            if progress_bar:
                progress_bar.update(1)
            sample.state = "done"

        return sample_list

    @classmethod
    async def async_run(cls, sample_list: list, model: ModelAPI, **kwargs):
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return await created_task


class Paraphrase(BaseGrammer):
    @staticmethod
    def transform(*args, **kwargs):
        pass

    @staticmethod
    def run():
        pass
