from typing import List, Dict, Union
import asyncio
from langtest.transform.base import ITests
from langtest.utils.custom_types.sample import Sample
from langtest.modelhandler.modelhandler import ModelAPI


class StereoTypeTestFactory(ITests):
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
    async def async_run(
        sample_list: Union[Dict[str, List[Sample]], List[Sample]],
        model: ModelAPI,
        *args,
        **kwargs
    ):
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
