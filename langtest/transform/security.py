from abc import ABC, abstractmethod
import asyncio
from typing import List
from langtest.modelhandler.modelhandler import ModelFactory

from langtest.utils.custom_types.sample import Sample


class BaseSecurity(ABC):
    """Abstract base class for implementing a model security.

    This class defines the interface for implementing a model security.

    Attributes:
        None
    """

    @staticmethod
    @abstractmethod
    def transform():
        """Abstract method that transforms the sample data based on the implemented model security."""
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs):
        """Abstract method that implements the model security."""
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
                else:
                    sample.actual_results = model(sample.prompt)
                    sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """Abstract method that implements the model security."""
        created_task = await asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


class PromptInjection(BaseSecurity):
    """
    PromptInjection is a class that implements the model security for prompt injection.
    """

    alias_name = ["prompt_injection_attack"]
    supported_tasks = ["security"]

    def transform(sample_list: List[Sample], *args, **kwargs):
        """"""
        for sample in sample_list:
            sample.test_type = "prompt_injection_attack"
            sample.category = "security"

        return sample_list
