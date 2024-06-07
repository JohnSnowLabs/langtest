import asyncio
from typing import List, Dict
from langtest.transform.base import ITests
from langtest.utils.custom_types.sample import Sample
from langtest.modelhandler.modelhandler import ModelAPI


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
