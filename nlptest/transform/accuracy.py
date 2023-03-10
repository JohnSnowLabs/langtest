
from abc import ABC, abstractmethod
from typing import List

from nlptest.utils.custom_types import Sample


class BaseAccuracy(ABC):

    @staticmethod
    @abstractmethod
    def transform(self):
        return NotImplementedError
    
    alias_name = None


class Minimum_F1(BaseAccuracy):

    alias_name = "min_f1"

    def transform(data: List[Sample]):
        return super().transform()