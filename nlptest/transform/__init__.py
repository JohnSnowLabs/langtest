from abc import ABC, abstractclassmethod, abstractmethod
from typing import List
from functools import reduce
from typing import Dict, List, Optional
import pandas as pd
import nltk

from nlptest.transform.accuracy import BaseAccuracy

from .bias import BaseBias
from .representation import BaseRepresentation
from .robustness import BaseRobustness
from ..utils.custom_types import Sample
from multiprocessing import Pool

from .utils import (A2B_DICT, create_terminology, male_pronouns, female_pronouns, neutral_pronouns)
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


class Robustness(ITests):
    """"""

    def __init__(
            self,
            data_handler: List[Sample],
            tests=None
    ) -> None:
        
        self.supported_tests = self.available_tests()
        self.tests = tests
        
        if not isinstance(self.tests, dict):
            raise ValueError(
                f'Invalid test configuration! Tests can be '
                f'[1] dictionary of test name and corresponding parameters.'
            )

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = (set(self.tests) - set(self.supported_tests))
        if len(not_supported_tests) > 0:
            raise ValueError(
                f'Invalid test specification: {not_supported_tests}. Available tests are: {list(self.supported_tests.keys())}')
        

        if 'swap_entities' in self.tests:
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                   'label': [[i.entity for i in sample.expected_results.predictions]
                             for sample in data_handler]})
            self.tests['swap_entities']['parameters'] = {}
            self.tests['swap_entities']['parameters']['terminology'] = create_terminology(df)
            self.tests['swap_entities']['parameters']['labels'] = df.label.tolist()


        if "american_to_british" in self.tests:
            self.tests['american_to_british']['parameters'] = {}
            self.tests['american_to_british']['parameters']['accent_map'] = A2B_DICT

        if "british_to_american" in self.tests:
            self.tests['british_to_american']['parameters']['accent_map'] = {v: k for k, v in A2B_DICT.items()}

        if 'swap_cohyponyms' in self.tests:
            nltk.download('omw-1.4',quiet=True)
            nltk.download('wordnet', quiet=True)
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                   'label': [[i.entity for i in sample.expected_results.predictions]
                         for sample in data_handler]})
            self.tests['swap_cohyponyms']['parameters'] = {}
            self.tests['swap_cohyponyms']['parameters']['labels'] = df.label.tolist()

        self._data_handler = data_handler

    def transform(self) -> List[Sample]:
        """"""
        # NOTE: I don't know if we need to work with a dataframe of if we can keep it as a List[Sample]
        all_samples = []
        for test_name, params in self.tests.items():
            print(test_name)
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, **params.get('parameters', {}))
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
        self._data_handler = data_handler
        self.tests = tests

        if not isinstance(self.tests, dict):
            raise ValueError(
                f'Invalid test configuration! Tests can be '
                f'[1] dictionary of test name and corresponding parameters.'
            )

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = (set(self.tests) - set(self.supported_tests))
        if len(not_supported_tests) > 0:
            raise ValueError(
                f'Invalid test specification: {not_supported_tests}. Available tests are: {list(self.supported_tests.keys())}')
        
        

        if 'replace_to_male_pronouns' in self.tests:
          self.tests['replace_to_male_pronouns']['parameters'] = {} 
          self.tests['replace_to_male_pronouns']['parameters']['pronouns_to_substitute'] = [item for sublist in list(female_pronouns.values()) for item in sublist] +[item for sublist in list(neutral_pronouns.values()) for item in sublist] 
          self.tests['replace_to_male_pronouns']['parameters']['pronoun_type'] = 'male'
        
        if 'replace_to_female_pronouns' in self.tests:
          self.tests['replace_to_female_pronouns']['parameters'] = {} 
          self.tests['replace_to_female_pronouns']['parameters']['pronouns_to_substitute'] = [item for sublist in list(male_pronouns.values()) for item in sublist] +[item for sublist in list(neutral_pronouns.values()) for item in sublist] 
          self.tests['replace_to_female_pronouns']['parameters']['pronoun_type'] = 'female'

        if 'replace_to_neutral_pronouns' in self.tests:
          self.tests['replace_to_neutral_pronouns']['parameters'] = {} 
          self.tests['replace_to_neutral_pronouns']['parameters']['pronouns_to_substitute'] = [item for sublist in list(female_pronouns.values()) for item in sublist] +[item for sublist in list(male_pronouns.values()) for item in sublist] 
          self.tests['replace_to_neutral_pronouns']['parameters']['pronoun_type'] = 'neutral'
   

    def transform(self):
        all_samples = []
        for test_name, params in self.tests.items():
            print(test_name)
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, **params.get('parameters', {}))
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
        self._data_handler = data_handler
        self.tests = tests

        if not isinstance(self.tests, dict):
            raise ValueError(
                f'Invalid test configuration! Tests can be '
                f'[1] dictionary of test name and corresponding parameters.'
            )

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = (set(self.tests) - set(self.supported_tests))
        if len(not_supported_tests) > 0:
            raise ValueError(
                    f'Invalid test specification: {not_supported_tests}. Available tests are: {list(self.supported_tests.keys())}')
    

    def transform(self):
        all_samples = []
        for test_name, params in self.tests.items():
            print(test_name)
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, **params.get('parameters', {}))
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


class Accuracy(ITests):

    def __init__(
        self,
        data_handler: List[Sample],
        tests=None
    ) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests

        if not isinstance(self.tests, dict):
            raise ValueError(
                f'Invalid test configuration! Tests can be '
                f'[1] dictionary of test name and corresponding parameters.'
            )

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = (set(self.tests) - set(self.supported_tests))
        if len(not_supported_tests) > 0:
            raise ValueError(
                    f'Invalid test specification: {not_supported_tests}. Available tests are: {list(self.supported_tests.keys())}')
    

    def transform(self):
        all_samples = []
        for test_name, params in self.tests.items():
            print(test_name)
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, **params.get('parameters', {}))
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples
    
    @classmethod
    def available_tests(cls) -> dict:
        tests = {
            j: i for i in BaseAccuracy.__subclasses__() 
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
            }
        return tests

