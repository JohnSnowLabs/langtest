

from abc import ABC, abstractmethod
from typing import List

from nlptest.utils.custom_types import Sample


class BaseRepresentation(ABC):
<<<<<<< HEAD

=======
>>>>>>> e4be3befcb71af4921610a51797e81c520be6edd
    @staticmethod
    @abstractmethod
    def transform(self):
        return NotImplementedError


class GenderReprestation(BaseRepresentation):

    def transform(data: List[Sample]):
        return super().transform()