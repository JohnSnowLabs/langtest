from abc import ABC, abstractmethod
from typing import List

from .perturbation import PerturbationFactory
from .perturbation import BasePerturbation

from .bias import BaseBias
from .representation import BaseRepresentation
from .robustness import BaseRobustness
from ..utils.custom_types import Sample
from multiprocessing import Pool

class TestFactory:

    @staticmethod
    def transform(data: List[Sample], test_types: dict):
        all_results = []
        all_categories = TestFactory.test_catgories()
        process = []
        for each in list(test_types.keys()):
            values = test_types[each]
            all_results.append(
                all_categories[each]().transform(data, values)
            )

    @staticmethod
    def test_catgories():
        return {cls.__name__.lower(): cls for cls in ITests.__subclasses__()}


class ITests(ABC):

    @abstractmethod
    def transform(self):
        return NotImplementedError


class Robustness(ITests):

    @staticmethod
    def transform(data: List[Sample], test_types: dict):
        all_results = []
        all_tests = Robustness.get_tests()
        for each in list(test_types.keys()):
            values = test_types[each]
            all_results.append(
                all_tests[each]().transform(data)
            )

    @staticmethod
    def get_tests() -> dict:
        return {cls.__name__.lower(): cls for cls in BaseRobustness.__subclasses__()}

class Bias(ITests):

    @staticmethod
    def transform(data: List[Sample], test_types: dict):
        all_results = []
        all_tests = Bias.get_tests()
        for each in list(test_types.keys()):
            values = test_types[each]
            all_results.extend(
                all_tests[each]().transform(data, values)
            )
        return all_results
    
    @staticmethod
    def get_tests(self) -> dict:
        return {cls.__name__.lower(): cls for cls in BaseBias.__subclasses__()}
    

class Representation(ITests):

    @staticmethod
    def transform(data: List[Sample], test_types: dict):
        all_results = []
        all_tests = Representation.get_tests()
        for each in list(test_types.keys()):
            values = test_types[each]
            all_results.extend(
                all_tests[each]().transform(data, values)
            )
        return all_results
    

    def get_tests(self) -> dict:
        return {cls.__name__.lower(): cls for cls in BaseRepresentation.__subclasses__()}

