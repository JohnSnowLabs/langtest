import asyncio
from typing import Dict, List

import nest_asyncio
import pandas as pd

from langtest.transform.performance import BasePerformance
from langtest.transform.robustness import RobustnessTestFactory
from langtest.transform.bias import BiasTestFactory
from langtest.transform.representation import RepresentationTestFactory
from langtest.transform.fairness import FairnessTestFactory
from langtest.transform.security import BaseSecurity

from .accuracy import BaseAccuracy
from .toxicity import BaseToxicity
from .ideology import BaseIdeology
from .sensitivity import BaseSensitivity
from .sycophancy import BaseSycophancy
from .utils import filter_unique_samples
from ..modelhandler import ModelAPI
from ..utils.custom_types.sample import (
    NERSample,
    SequenceClassificationSample,
    Sample,
)
from ..utils.custom_types.helpers import default_user_prompt
from langtest.transform.base import ITests, TestFactory
from ..errors import Errors, Warnings
from ..logger import logger as logging

nest_asyncio.apply()


class AccuracyTestFactory(ITests):
    """
    A class for performing accuracy tests on a given dataset.
    """

    alias_name = "accuracy"
    model_result = None

    def __init__(self, data_handler: List[Sample], tests: Dict, **kwargs) -> None:
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
        """
        Runs the accuracy test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the accuracy test.
        """
        all_samples = []

        if self._data_handler[0].expected_results is None:
            raise RuntimeError(Errors.E052(var="accuracy"))

        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]

            if data_handler_copy[0].task == "ner":
                y_true = pd.Series(data_handler_copy).apply(
                    lambda x: [y.entity for y in x.expected_results.predictions]
                )
                y_true = y_true.explode().apply(
                    lambda x: x.split("-")[-1] if isinstance(x, str) else x
                )
            elif data_handler_copy[0].task == "text-classification":
                y_true = (
                    pd.Series(data_handler_copy)
                    .apply(lambda x: [y.label for y in x.expected_results.predictions])
                    .explode()
                )
            elif (
                data_handler_copy[0].task == "question-answering"
                or data_handler_copy[0].task == "summarization"
            ):
                y_true = (
                    pd.Series(data_handler_copy)
                    .apply(lambda x: x.expected_results)
                    .explode()
                )

            y_true = y_true.dropna()
            transformed_samples = self.supported_tests[test_name].transform(
                test_name, y_true, params
            )

            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)

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
            for i in BaseAccuracy.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests

    @classmethod
    def run(
        cls,
        sample_list: Dict[str, List[Sample]],
        model: ModelAPI,
        raw_data: List[Sample],
        **kwargs,
    ):
        """
        Runs the accuracy tests on the given model and dataset.

        Args:
            sample_list (Dict[str, List[Sample]]): A dictionary of test names and corresponding `Sample` objects.
            model (ModelAPI): The model to be tested.
            raw_data (List[Sample]): The raw dataset.

        """
        raw_data_copy = [x.copy() for x in raw_data]

        if isinstance(raw_data_copy[0], NERSample):

            def predict_ner(sample):
                prediction = model.predict(sample.original)
                sample.actual_results = prediction
                return prediction

            X_test = pd.Series(raw_data_copy)
            X_test.apply(predict_ner)
            y_true = pd.Series(raw_data_copy).apply(
                lambda x: [y.entity for y in x.expected_results.predictions]
            )
            y_pred = pd.Series(raw_data_copy).apply(
                lambda x: [y.entity for y in x.actual_results.predictions]
            )
            valid_indices = y_true.apply(len) == y_pred.apply(len)
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            y_true = y_true.explode()
            y_pred = y_pred.explode()
            y_pred = y_pred.apply(lambda x: x.split("-")[-1])
            y_true = y_true.apply(lambda x: x.split("-")[-1])

        elif isinstance(raw_data_copy[0], SequenceClassificationSample):

            def predict_text_classification(sample):
                prediction = model.predict(sample.original)
                sample.actual_results = prediction
                return prediction

            X_test = pd.Series(raw_data_copy)
            X_test.apply(predict_text_classification)

            y_true = pd.Series(raw_data_copy).apply(
                lambda x: [y.label for y in x.expected_results.predictions]
            )
            y_pred = pd.Series(raw_data_copy).apply(
                lambda x: [y.label for y in x.actual_results.predictions]
            )
            y_true = y_true.apply(lambda x: x[0])
            y_pred = y_pred.apply(lambda x: x[0])

            y_true = y_true.explode()
            y_pred = y_pred.explode()

        elif raw_data_copy[0].task == "question-answering":
            from ..utils.custom_types.helpers import build_qa_input, build_qa_prompt

            if raw_data_copy[0].dataset_name is None:
                dataset_name = "default_question_answering_prompt"
            else:
                dataset_name = raw_data_copy[0].dataset_name.split("-")[0].lower()

            def predict_question_answering(sample):
                input_data = build_qa_input(
                    context=sample.original_context,
                    question=sample.original_question,
                    options=sample.options,
                )
                prompt = build_qa_prompt(input_data, dataset_name, **kwargs)
                server_prompt = kwargs.get("server_prompt", " ")
                prediction = model(
                    text=input_data, prompt=prompt, server_prompt=server_prompt
                ).strip()
                sample.actual_results = prediction
                return prediction

            y_true = pd.Series(raw_data_copy).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data_copy)

            y_pred = X_test.apply(predict_question_answering)

        elif raw_data_copy[0].task == "summarization":
            if raw_data_copy[0].dataset_name is None:
                dataset_name = "default_summarization_prompt"
            else:
                dataset_name = raw_data_copy[0].dataset_name.split("-")[0].lower()
            prompt_template = kwargs.get(
                "user_prompt", default_user_prompt.get(dataset_name, "")
            )

            def predict_summarization(sample):
                prediction = model(
                    text={"context": sample.original},
                    prompt={"template": prompt_template, "input_variables": ["context"]},
                ).strip()
                sample.actual_results = prediction
                return prediction

            y_true = pd.Series(raw_data_copy).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data_copy)
            y_pred = X_test.apply(predict_summarization)

        if kwargs["is_default"]:
            y_pred = y_pred.apply(
                lambda x: "1"
                if x in ["pos", "LABEL_1", "POS"]
                else ("0" if x in ["neg", "LABEL_0", "NEG"] else x)
            )

        supported_tests = cls.available_tests()

        tasks = []

        from ..utils.custom_types.helpers import TestResultManager

        cls.model_result = TestResultManager().prepare_model_response(raw_data_copy)

        for test_name, samples in sample_list.items():
            tasks.append(
                supported_tests[test_name].async_run(
                    samples, y_true, y_pred, X_test=X_test, **kwargs
                )
            )
        return tasks


class ToxicityTestFactory(ITests):
    """
    A class for performing toxicity tests on a given dataset.
    """

    alias_name = "toxicity"

    def __init__(self, data_handler: List[Sample], tests: Dict, **kwargs) -> None:
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
        """
        Runs the toxicity test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the toxicity test.
        """
        all_samples = []

        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform
            transformed_samples = test_func(
                data_handler_copy, test_name=test_name, **params.get("parameters", {})
            )

            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)

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
            for i in BaseToxicity.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


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
        tests = {
            j: i
            for i in BasePerformance.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


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
        tests = {
            j: i
            for i in BaseSecurity.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


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
]
