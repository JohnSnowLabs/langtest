from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
import random
from typing import List, Dict, Union

import importlib_resources
from langtest.errors import Errors
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests, TestFactory
from langtest.utils.custom_types.helpers import HashableDict
from langtest.utils.custom_types.sample import QASample, Sample


class ClinicalTestFactory(ITests):
    """Factory class for the clinical tests"""

    alias_name = "clinical"
    supported_tasks = [
        "clinical",
        "text-generation",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the ClinicalTestFactory"""

        self.supported_tests = self.available_tests()
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        # check if configured tests are supported
        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        all_samples = []
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            test_func = self.supported_tests[test_name].transform
            data_handler_copy = [sample.copy() for sample in self.data_handler]
            samples = test_func(data_handler_copy, **params)

            all_samples.extend(samples)

        return all_samples

    @classmethod
    def available_tests(cls) -> Dict[str, Union["BaseClincial", "ClinicalTestFactory"]]:
        """Returns the empty dict, no clinical tests

        Returns:
            Dict[str, str]: Empty dict, no clinical tests
        """
        test_types = BaseClincial.available_tests()
        # test_types.update({"demographic-bias": cls})
        return test_types


class BaseClincial(ABC):
    """
    Baseclass for the clinical tests
    """

    test_types = defaultdict(lambda: BaseClincial)
    alias_name = None
    supported_tasks = [
        "question-answering",
    ]

    @staticmethod
    @abstractmethod
    def transform(*args, **kwargs):
        """Transform method for the clinical tests"""

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the clinical tests"""

        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, *args, **kwargs):
        """Async run method for the clinical tests"""
        created_task = asyncio.create_task(cls.run(*args, **kwargs))
        return await created_task

    @classmethod
    def available_tests(cls) -> Dict[str, "BaseClincial"]:
        """Available tests for the clinical tests"""

        return cls.test_types

    def __init_subclass__(cls) -> None:
        """Initializes the subclass for the clinical tests"""
        alias = cls.alias_name if isinstance(cls.alias_name, list) else [cls.alias_name]
        for name in alias:
            BaseClincial.test_types[name] = cls


class DemographicBias(BaseClincial):
    """
    DemographicBias class for the clinical tests
    """

    alias_name = "demographic-bias"
    supported_tasks = ["question-answering", "text-generation"]

    @staticmethod
    def transform(sample_list: List[Sample], *args, **kwargs):
        """Transform method for the DemographicBias class"""
        for sample in sample_list:
            sample.test_type = "demographic-bias"
            sample.category = "clinical"
        return sample_list

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Run method for the DemographicBias class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress_bar:
                progress_bar.update(1)
        return sample_list


class Generic2Brand(BaseClincial):
    """
    GenericBrand class for the clinical tests
    """

    alias_name = "drug_generic_to_brand"

    @staticmethod
    def transform(*args, **kwargs):
        """Transform method for the GenericBrand class"""
        import pandas as pd

        task = TestFactory.task
        data = []
        if task == "ner":
            dataset_path = "ner_g2b.jsonl"
        elif task == "question-answering":
            dataset_path = "qa_generic_to_brand_v2.jsonl"
            file_path = (
                importlib_resources.files("langtest") / "data" / "DrugSwap" / dataset_path
            )
            df = pd.read_json(file_path, lines=True)
            sample_df = df.sample(50, random_state=42, replace=True)
            for index, row in sample_df.iterrows():
                sample = QASample(
                    original_context="-",
                    original_question=row["original_question"],
                    perturbed_question=row["perturbed_question"],
                )
                # sample.expected_results = row["expected_results"]
                # sample.actual_results = row["actual_results"]
                sample.test_type = "drug_generic_to_brand"
                sample.category = "clinical"
                data.append(sample)

        # data: List[Sample] = DataFactory(
        #     file_path={"data_source": "DrugSwap", "split": dataset_path},
        #     task=task,
        # ).load()
        # for sample in data[:50]:
        #     sample.test_type = "drug_generic_to_brand"
        #     sample.category = "clinical"

        return random.choices(data, k=50)

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the GenericBrand class"""

        progress_bar = kwargs.get("progress_bar", False)

        for sample in sample_list:
            # if hasattr(sample, "run"):
            #     sample.run(model, **kwargs)
            # else:
            if isinstance(sample, QASample):
                sample.expected_results = model.predict(
                    text=HashableDict(
                        {
                            "text": sample.original_question,
                        }
                    ),
                    prompt=HashableDict(
                        {
                            "template": TEMPLATE,
                            "input_variables": ["text"],
                        }
                    ),
                    server_prompt="Perform the task to the best of your ability:",
                )
                sample.actual_results = model.predict(
                    text=HashableDict(
                        {
                            "text": sample.perturbed_question,
                        }
                    ),
                    prompt=HashableDict(
                        {
                            "template": TEMPLATE,
                            "input_variables": ["text"],
                        }
                    ),
                    server_prompt="Perform the task to the best of your ability:",
                )
            else:
                sample.expected_results = model.predict(sample.original)
                sample.actual_results = model.predict(sample.test_case)
            if progress_bar:
                progress_bar.update(1)
            sample.state = "done"
        return sample_list


class Brand2Generic(BaseClincial):
    """
    BrandGeneric class for the clinical tests
    """

    alias_name = "drug_brand_to_generic"

    @staticmethod
    def transform(*args, **kwargs):
        """Transform method for the BrandGeneric class"""
        from langtest import DataFactory

        task = TestFactory.task

        if task == "ner":
            dataset_path = "ner_b2g.jsonl"
        elif task == "question-answering":
            dataset_path = "qa_generic_to_brand.jsonl"

        data: List[Sample] = DataFactory(
            file_path={"data_source": "DrugSwap", "split": dataset_path},
            task=task,
        )
        for sample in data:
            sample.test_type = "drug_brand_to_generic"
            sample.category = "clinical"

        return data

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Run method for the BrandGeneric class"""

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


TEMPLATE = """
You are an AI bot specializing in providing accurate and concise answers to questions. You will be presented with a medical question and multiple-choice answer options. The following are examples for how to answer.

EXAMPLE 1:
<|user|>
Question: What is the most common cause of death in the United States?
A) Cancer
B) Heart disease
C) Stroke
D) Diabetes
<|end|>
Answer(only A, B, C, or D):
<|assistant|> B <|end|>

EXAMPLE 2:
<|user|>
Question: what is the purpose of paracetamol tablet?
A) To reduce fever
B) To reduce pain
C) To reduce inflammation
D) All of the above
Answer(only A, B, C, or D):
<|end|>
<|assistant|>D<|end|>

Your task is to choose the correct answer (only A, B, C, or D) from the options of mcq question.
{text}
Answer(only A, B, C, or D):
<|end|>
<|assistant|>
"""
