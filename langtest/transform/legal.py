import asyncio
from typing import List, Dict
from langtest.transform.base import ITests
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.utils.custom_types.sample import Sample


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
