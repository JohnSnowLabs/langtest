

from abc import ABC, abstractmethod


class BaseBias(ABC):

    @abstractmethod
    def transform(self):
        pass

    

class Geneder(BaseBias):

    def transform(self):
        return super().transform()