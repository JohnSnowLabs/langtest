
from abc import ABC, abstractmethod
from typing import List

from nlptest.utils.custom_types import Sample


class BaseAccuracy(ABC):

    """
    Abstract base class for implementing accuracy measures.

    Attributes:
        alias_name (str): A name or list of names that identify the accuracy measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented accuracy measure.
    """

    @staticmethod
    @abstractmethod
    def transform(self):

        """
        Abstract method that implements the accuracy measure.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented accuracy measure.
        """

        return NotImplementedError
    
    alias_name = None


class Minimum_F1(BaseAccuracy):

    """
    Subclass of BaseAccuracy that implements the minimum F1 score.

    Attributes:
        alias_name (str): The name "min_f1" identifying the minimum F1 score.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the minimum F1 score.
    """

    alias_name = "min_f1"

    def transform(data: List[Sample]):
        """
        Computes the minimum F1 score for the given data.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the minimum F1 score.
        """
        
        return super().transform()