from abc import ABC, abstractmethod
from langtest.transform import ITests


class GrammerTestFactory(ITests):
    pass


class BaseGrammer(ABC):
    @staticmethod
    @abstractmethod
    def transform():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def run():
        raise NotImplementedError


class Paraphrase(BaseGrammer):
    @staticmethod
    def transform(*args, **kwargs):
        pass

    @staticmethod
    def run():
        pass
