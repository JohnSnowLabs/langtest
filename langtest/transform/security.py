from abc import ABC, abstractmethod
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
        """Abstract method that transforms the sample data based on the implemented model security.

        """
        raise NotImplementedError("Please Implement this method")
    
    @staticmethod
    @abstractmethod
    async def run():
        """Abstract method that implements the model security.

        """
        raise NotImplementedError("Please Implement this method")
    
    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """Abstract method that implements the model security.

        """
        raise NotImplementedError("Please Implement this method")
    

class PromptInjection(BaseSecurity):
    """
    PromptInjection is a class that implements the model security for prompt injection.
    """

    alias_name = ["prompt_injection_attack"]
    supported_task = [
        "text-classification",
        "question-answering",
        "summarization"]

    def transform(sample_list):
        pass