from abc import ABC, abstractclassmethod, abstractmethod
from typing import List
from functools import reduce
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import nltk

from .perturbation import PerturbationFactory
from .perturbation import BasePerturbation

from .bias import BaseBias
from .representation import BaseRepresentation
from .robustness import BaseRobustness
from ..utils.custom_types import Sample
from multiprocessing import Pool

from .utils import (A2B_DICT, CONTRACTION_MAP, DEFAULT_PERTURBATIONS, PERTURB_CLASS_MAP, TYPO_FREQUENCY, create_terminology, male_pronouns, female_pronouns, neutral_pronouns)
from ..utils.custom_types import Sample, Span, Transformation

class TestFactory:

    @staticmethod
    def transform(data: List[Sample], test_types: dict):
        all_results = []
        all_categories = TestFactory.test_catgories()
        # process = []
        for each in list(test_types.keys()):
            values = test_types[each]
            all_results.extend(
                all_categories[each](data, values).transform()
            )
        return all_results

    @classmethod
    def test_catgories(cls):
        return {cls.__name__.lower(): cls for cls in ITests.__subclasses__()}
    
    @classmethod
    def test_scenarios(cls):
        return {cls.__name__.lower(): cls.available_tests() for cls in ITests.__subclasses__()}


class ITests(ABC):

    @abstractmethod
    def transform(self):
        return NotImplementedError
    
    @abstractclassmethod
    def available_tests(cls):
        return NotImplementedError


# class Robustness(ITests):

#     @staticmethod
#     def transform(data: List[Sample], test_types: dict):
#         all_results = []
#         all_tests = Robustness.get_tests()
#         for each in list(test_types.keys()):
#             values = test_types[each]
#             all_results.append(
#                 all_tests[each]().transform(data)
#             )

#     @staticmethod
#     def get_tests() -> dict:
#         return {cls.__name__.lower(): cls for cls in BaseRobustness.__subclasses__()}
class Robustness(ITests):
    """"""

    def __init__(
            self,
            data_handler: List[Sample],
            tests=None
    ) -> None:
        
        self.supported_tests = self.available_tests()

        if tests is []:
            tests = self.supported_tests

        self._tests = dict()
        for test in tests:

            if isinstance(test, str):
                if test not in self.supported_tests:
                    raise ValueError(
                        f'Invalid test specification: {test}. Available tests are: {self.supported_tests}')
                self._tests[test] = dict()
            elif isinstance(test, dict):
                test_name = list(test.keys())[0]
                if test_name not in self.supported_tests:
                    raise ValueError(
                        f'Invalid test specification: {test_name}. Available tests are: {self.supported_tests}')
                self._tests[test_name] = reduce(lambda x, y: {**x, **y}, test[test_name])
            else:
                raise ValueError(
                    f'Invalid test configuration! Tests can be '
                    f'[1] test name as string or '
                    f'[2] dictionary of test name and corresponding parameters.'
                )

        if 'swap_entities' in self._tests:
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                   'label': [[i.entity for i in sample.expected_results.predictions]
                             for sample in data_handler]})
            self._tests['swap_entities']['terminology'] = create_terminology(df)
            self._tests['swap_entities']['labels'] = df.label.tolist()


        if "american_to_british" in self._tests:
            self._tests['american_to_british']['accent_map'] = A2B_DICT

        if "british_to_american" in self._tests:
            self._tests['british_to_american']['accent_map'] = {v: k for k, v in A2B_DICT.items()}

        if 'swap_cohyponyms' in self._tests:
            nltk.download('omw-1.4')
            nltk.download('wordnet')
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                   'label': [[i.entity for i in sample.expected_results.predictions]
                         for sample in data_handler]})
            self._tests['swap_cohyponyms']['labels'] = df.label.tolist()

        self._data_handler = data_handler

    def transform(self) -> List[Sample]:
        """"""
        # NOTE: I don't know if we need to work with a dataframe of if we can keep it as a List[Sample]
        all_samples = []
        for test_name, params in self._tests.items():
            print(test_name)
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, **params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples
    
    @classmethod
    def available_tests(cls) -> dict:
        tests = {
            j: i for i in BaseRobustness.__subclasses__() 
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
            }
        return tests

class Bias(ITests):

    def __init__(
            self,
            data_handler: List[Sample],
            tests=None
     ) -> None:
        
        self.supported_tests = self.available_tests()

        if tests is []:
            tests = self.supported_tests

        self._tests = dict()
        for test in tests:

            if isinstance(test, str):
                if test not in self.supported_tests:
                    raise ValueError(
                        f'Invalid test specification: {test}. Available tests are: {self.supported_tests}')
                self._tests[test] = dict()
            elif isinstance(test, dict):
                test_name = list(test.keys())[0]
                if test_name not in self.supported_tests:
                    raise ValueError(
                        f'Invalid test specification: {test_name}. Available tests are: {self.supported_tests}')
                self._tests[test_name] = reduce(lambda x, y: {**x, **y}, test[test_name])
            else:
                raise ValueError(
                    f'Invalid test configuration! Tests can be '
                    f'[1] test name as string or '
                    f'[2] dictionary of test name and corresponding parameters.'
                )


        if 'replace_to_male_pronouns' in self._tests:
          self._tests['replace_to_male_pronouns']['pronouns_to_substitute'] = [item for sublist in list(female_pronouns.values()) for item in sublist] +[item for sublist in list(neutral_pronouns.values()) for item in sublist] 
          self._tests['replace_to_male_pronouns']['pronoun_type'] = 'male'
        
        if 'replace_to_female_pronouns' in self._tests:
          self._tests['replace_to_female_pronouns']['pronouns_to_substitute'] = [item for sublist in list(male_pronouns.values()) for item in sublist] +[item for sublist in list(neutral_pronouns.values()) for item in sublist] 
          self._tests['replace_to_female_pronouns']['pronoun_type'] = 'female'

        if 'replace_to_neutral_pronouns' in self._tests:
          self._tests['replace_to_neutral_pronouns']['pronouns_to_substitute'] = [item for sublist in list(female_pronouns.values()) for item in sublist] +[item for sublist in list(male_pronouns.values()) for item in sublist] 
          self._tests['replace_to_neutral_pronouns']['pronoun_type'] = 'neutral'

        self._data_handler = data_handler

    def transform(self):
        all_samples = []
        for test_name, params in self._tests.items():
            print(test_name)
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, **params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples
    
    @classmethod
    def available_tests(cls) -> dict:
        tests = {
            j: i for i in BaseBias.__subclasses__() 
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
            }
        return tests
    

class Representation(ITests):

    def __init__(
        self,
        data_handler: List[Sample],
        tests=None
    ) -> None:
    
        self.supported_tests = self.available_tests()

        if tests is []:
            tests = self.supported_tests

        self._tests = dict()
        for test in tests:

            if isinstance(test, str):
                if test not in self.supported_tests:
                    raise ValueError(
                        f'Invalid test specification: {test}. Available tests are: {self.supported_tests}')
                self._tests[test] = dict()
            elif isinstance(test, dict):
                test_name = list(test.keys())[0]
                if test_name not in self.supported_tests:
                    raise ValueError(
                        f'Invalid test specification: {test_name}. Available tests are: {self.supported_tests}')
                self._tests[test_name] = reduce(lambda x, y: {**x, **y}, test[test_name])
            else:
                raise ValueError(
                    f'Invalid test configuration! Tests can be '
                    f'[1] test name as string or '
                    f'[2] dictionary of test name and corresponding parameters.'
                )


        self._data_handler = data_handler


    
    def transform(self):
        all_samples = []
        for test_name, params in self._tests.items():
            print(test_name)
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, **params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples
    

    @classmethod
    def available_tests(cls) -> dict:
        tests = {
            j: i for i in BaseRepresentation.__subclasses__() 
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
            }
        return tests

