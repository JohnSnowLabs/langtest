

from abc import ABC, abstractmethod
from typing import List

from nlptest.utils.custom_types import Sample


class BaseRepresentation(ABC):
    @staticmethod
    @abstractmethod
    def transform(self):
        return NotImplementedError


class GenderReprestation(BaseRepresentation):

    def transform(data: List[Sample]):
        return super().transform()