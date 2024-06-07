import asyncio
from typing import List, Dict
from langtest.errors import Errors
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
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
