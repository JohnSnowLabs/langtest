import asyncio
from typing import List, Dict, TypedDict
from langtest.modelhandler.modelhandler import ModelAPI
from langtest.transform.base import ITests
from langtest.utils.custom_types.sample import Sample


class DisinformationTestFactory(ITests):
    """Factory class for disinformation test"""

    alias_name = "disinformation"
    supported_tasks = [
        "disinformation",
        "text-generation",
    ]

    # TestConfig
    TestConfig = TypedDict(
        "TestConfig",
        min_pass_rate=float,
    )

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
