

from abc import ABC, abstractmethod


class BaseBias(ABC):

    @abstractmethod
    def transform(self):
        return NotImplementedError

    alias_name = None

    

class GenderBias(BaseBias):

    alias_name = "gender_bias"

    def transform(self):
        return super().transform()