from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
from typing import List, Dict
from langtest.datahandler.datasource import DataFactory
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests, TestFactory
from langtest.utils.custom_types.sample import Sample


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
        test_types = BaseClincial.available_tests()
        test_types.update({"demographic-bias": cls})
        return test_types

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
    async def run(*args, **kwargs):
        """Run method for the clinical tests"""

        raise NotImplementedError

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


class Generic2Brand(BaseClincial):
    """
    GenericBrand class for the clinical tests
    """

    alias_name = "drug_generic_to_brand"

    @staticmethod
    def transform(*args, **kwargs):
        """Transform method for the GenericBrand class"""

        task = TestFactory.task

        if task == "ner":
            dataset_path = "ner_g2b.jsonl"
        elif task == "question-answering":
            dataset_path = "qa_g2b.jsonl"

        data: List[Sample] = DataFactory(
            file_path={"data_source": "DrugSwap", "split": dataset_path},
            task=task,
        )
        for sample in data:
            sample.test_type = "drug_generic_to_brand"
            sample.category = "clinical"

        return data

    @staticmethod
    async def run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Run method for the GenericBrand class"""

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


class Brand2Generic(BaseClincial):
    """
    BrandGeneric class for the clinical tests
    """

    alias_name = "drug_brand_to_generic"

    @staticmethod
    def transform(*args, **kwargs):
        """Transform method for the BrandGeneric class"""

        task = TestFactory.task

        if task == "ner":
            dataset_path = "ner_b2g.jsonl"
        elif task == "question-answering":
            dataset_path = "qa_b2g.jsonl"

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
