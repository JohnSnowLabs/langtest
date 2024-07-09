import asyncio
from typing import List, Dict
from langtest.transform.base import ITests
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.utils.custom_types.sample import Sample


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
