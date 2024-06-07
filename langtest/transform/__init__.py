import asyncio
from typing import Dict, List

import nest_asyncio

from langtest.transform.performance import PerformanceTestFactory
from langtest.transform.robustness import RobustnessTestFactory
from langtest.transform.bias import BiasTestFactory
from langtest.transform.representation import RepresentationTestFactory
from langtest.transform.fairness import FairnessTestFactory
from langtest.transform.accuracy import AccuracyTestFactory
from langtest.transform.security import SecurityTestFactory
from langtest.transform.toxicity import ToxicityTestFactory

from .ideology import BaseIdeology
from .sensitivity import BaseSensitivity
from .sycophancy import BaseSycophancy
from .utils import filter_unique_samples
from ..modelhandler import ModelAPI
from ..utils.custom_types.sample import (
    Sample,
)
from langtest.transform.base import ITests, TestFactory
from ..errors import Errors, Warnings
from ..logger import logger as logging

nest_asyncio.apply()


class ClinicalTestFactory(ITests):
    """Factory class for the clinical tests"""

    alias_name = "clinical"
    supported_tasks = [
        "clinical",
        "text-generation",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the ClinicalTestFactory"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        for sample in self.data_handler:
            sample.test_type = "demographic-bias"
            sample.category = "clinical"
        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the clinical tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the clinical tests

        Returns:
            List[Sample]: The transformed data based on the implemented clinical tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no clinical tests

        Returns:
            Dict[str, str]: Empty dict, no clinical tests
        """
        return {"demographic-bias": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the clinical tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the clinical tests

        Returns:
            List[Sample]: The transformed data based on the implemented clinical tests@

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["demographic-bias"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["demographic-bias"]


class DisinformationTestFactory(ITests):
    """Factory class for disinformation test"""

    alias_name = "disinformation"
    supported_tasks = [
        "disinformation",
        "text-generation",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        for sample in self.data_handler:
            sample.test_type = "narrative_wedging"
            sample.category = "disinformation"

        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        return {"narrative_wedging": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["narrative_wedging"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["narrative_wedging"]


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

        tests = {
            j: i
            for i in BaseIdeology.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class SensitivityTestFactory(ITests):
    """A class for performing Sensitivity tests on a given dataset.

    This class provides functionality to perform sensitivity tests on a given dataset
    using various test configurations.

    Attributes:
        alias_name (str): A string representing the alias name for this test factory.

    """

    alias_name = "sensitivity"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initialize a new SensitivityTestFactory instance.

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
        """Run the sensitivity test and return the resulting `Sample` objects.

        Returns:
            List[Sample]: A list of `Sample` objects representing the resulting dataset after running the sensitivity test.

        """
        all_samples = []
        no_transformation_applied_tests = {}
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

            new_transformed_samples, removed_samples_tests = filter_unique_samples(
                TestFactory.task.category, transformed_samples, test_name
            )
            all_samples.extend(new_transformed_samples)

            no_transformation_applied_tests.update(removed_samples_tests)

        if no_transformation_applied_tests:
            warning_message = Warnings._W009
            for test, count in no_transformation_applied_tests.items():
                warning_message += Warnings._W010.format(
                    test=test, count=count, total_sample=len(self._data_handler)
                )

            logging.warning(warning_message)

        return all_samples

    @staticmethod
    def available_tests() -> dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """
        tests = {
            j: i
            for i in BaseSensitivity.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class StereoTypeFactory(ITests):
    """Factory class for the crows-pairs or wino-bias tests"""

    alias_name = "stereotype"
    supported_tasks = [
        "wino-bias",
        "crows-pairs",
        "fill-mask",
        "question-answering",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the crows-pairs or wino-bias tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Testcases List

        """
        for sample in self.data_handler:
            if sample.test_type == "crows-pairs":
                if "diff_threshold" in self.tests["crows-pairs"].keys():
                    sample.diff_threshold = self.tests["crows-pairs"]["diff_threshold"]
                if "filter_threshold" in self.tests["crows-pairs"].keys():
                    sample.filter_threshold = self.tests["crows-pairs"][
                        "filter_threshold"
                    ]
            else:
                if "diff_threshold" in self.tests["wino-bias"].keys():
                    sample.diff_threshold = self.tests["wino-bias"]["diff_threshold"]

        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the crows-pairs or wino-bias  tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the crows-pairs or wino-bias tests

        Returns:
            List[Sample]: The transformed data based on the implemented crows-pairs or wino-bias tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no crows-pairs or wino-bias tests

        Returns:
            Dict[str, str]: Empty dict, no crows-pairs or wino-bias tests
        """

        return {"crows-pairs": cls, "wino-bias": cls}

    @staticmethod
    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the crows-pairs or wino-bias tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the crows-pairs or wino-bias tests

        Returns:
            List[Sample]: The transformed data based on the implemented crows-pairs or wino-biastests

        """
        progress = kwargs.get("progress_bar", False)
        for key, value in sample_list.items():
            if key == "crows-pairs":
                for sample in value:
                    if sample.state != "done":
                        if hasattr(sample, "run"):
                            sample_status = sample.run(model, **kwargs)
                            if sample_status:
                                sample.state = "done"
                    if progress:
                        progress.update(1)

                sample_list["crows-pairs"] = [
                    x
                    for x in sample_list["crows-pairs"]
                    if (
                        x.mask1_score > x.filter_threshold
                        or x.mask2_score > x.filter_threshold
                    )
                ]
                return sample_list["crows-pairs"]
        else:
            for sample in value:
                if sample.state != "done":
                    if hasattr(sample, "run"):
                        sample_status = sample.run(model, **kwargs)
                        if sample_status:
                            sample.state = "done"
                if progress:
                    progress.update(1)

            return sample_list["wino-bias"]


class StereoSetTestFactory(ITests):
    """Factory class for the StereoSet tests"""

    alias_name = "stereoset"
    supported_tasks = [
        "stereoset",
        "question-answering",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the stereoset tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            List of testcase

        """
        for s in self.data_handler:
            s.diff_threshold = (
                self.tests[s.test_type]["diff_threshold"]
                if "diff_threshold" in self.tests[s.test_type]
                else s.diff_threshold
            )
        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the stereoset tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the stereoset tests

        Returns:
            List[Sample]: The transformed data based on the implemented stereoset tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no stereoset tests

        Returns:
            Dict[str, str]: Empty dict, no stereoset tests
        """
        return {"intrasentence": cls, "intersentence": cls}

    @staticmethod
    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the StereoSet tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the StereoSet tests

        Returns:
            List[Sample]: The transformed data based on the implemented crows-pairs tests

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["intersentence"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)

        for sample in sample_list["intrasentence"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)

        return sample_list["intersentence"] + sample_list["intrasentence"]


class LegalTestFactory(ITests):
    """Factory class for the legal"""

    alias_name = "legal"
    supported_tasks = ["legal", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the legal tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        for sample in self.data_handler:
            sample.test_type = "legal-support"
            sample.category = "legal"
        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the legal tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the wino-bias tests

        Returns:
            List[Sample]: The transformed data based on the implemented legal tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no legal tests

        Returns:
            Dict[str, str]: Empty dict, no legal tests
        """
        return {"legal-support": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the legal tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the legal tests

        Returns:
            List[Sample]: The transformed data based on the implemented legal tests

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["legal-support"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["legal-support"]


class FactualityTestFactory(ITests):
    """Factory class for factuality test"""

    alias_name = "factuality"
    supported_tasks = ["factuality", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the FactualityTestFactory"""
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        for sample in self.data_handler:
            sample.test_type = "order_bias"
            sample.category = "factuality"

        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs factuality tests

        Args:
            sample_list (list[Sample]): A list of Sample objects to be tested.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Sample]: A list of Sample objects with test results.

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Retrieves available factuality test types.

        Returns:
            dict: A dictionary mapping test names to their corresponding classes.

        """
        return {"order_bias": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs factuality tests

        Args:
            sample_list (list[Sample]): A list of Sample objects to be tested.
            model (ModelAPI): The model to be used for testing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Sample]: A list of Sample objects with test results.

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["order_bias"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["order_bias"]


class SycophancyTestFactory(ITests):
    """A class for conducting Sycophancy tests on a given dataset.

    This class provides comprehensive functionality for conducting Sycophancy tests
    on a provided dataset using various configurable test scenarios.

    Attributes:
        alias_name (str): A string representing the alias name for this test factory.

    """

    alias_name = "Sycophancy"
    supported_tasks = ["Sycophancy", "question-answering"]

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
            for i in BaseSycophancy.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


__all__ = [
    RobustnessTestFactory,
    BiasTestFactory,
    RepresentationTestFactory,
    FairnessTestFactory,
    AccuracyTestFactory,
    ToxicityTestFactory,
    SecurityTestFactory,
    PerformanceTestFactory,
]
