

from abc import ABC, abstractmethod
from typing import List

from nlptest.utils.custom_types import Sample


class BaseRepresentation(ABC):

    @staticmethod
    @abstractmethod
    def transform(self):
        return NotImplementedError
    
    alias_name = None


class GenderReprestation(BaseRepresentation):

    alias_name = "gender_represtation"

    def transform(data: List[Sample]):
        return super().transform()