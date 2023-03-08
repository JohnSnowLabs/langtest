

from abc import ABC, abstractmethod


class BaseBias(ABC):

    @abstractmethod
    def transform(self):
        pass

    @property
    def tests():
        return BaseBias.__subclasses__()
    

class Geneder(BaseBias):

    def transform(self):
        return super().transform()