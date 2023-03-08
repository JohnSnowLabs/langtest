from abc import ABC, abstractmethod
from typing import List

from nlptest.transform.bias import BaseBias
from nlptest.transform.representation import BaseRepresentation
from nlptest.transform.robustness import BaseRobustness
from nlptest.utils.custom_types import Sample


class TestFactory:

    @staticmethod
    def transform(data: List[Sample], test_types: dict):
        all_results = []
        all_categories = TestFactory.test_catgories()
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
                all_tests[each]().transform(data, values)
            )
    
    def get_tests(self) -> dict:
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

