from abc import ABC, abstractclassmethod, abstractmethod
from typing import List

import nltk
import pandas as pd
from nlptest.modelhandler import ModelFactory

from nlptest.transform.accuracy import BaseAccuracy
from .bias import BaseBias
from .representation import BaseRepresentation
from .robustness import BaseRobustness
from .utils import (A2B_DICT, create_terminology,get_substitution_names, male_pronouns, female_pronouns, neutral_pronouns, country_economic_dict, white_names, black_names, hispanic_names, asian_names, native_american_names, inter_racial_names, religion_wise_names)
from ..utils.custom_types import Sample, Span, Transformation

class TestFactory:
    """
    A factory class for creating and running different types of tests on data.

    ...

    Methods
    -------
    transform(data: List[Sample], test_types: dict) -> List[Results]:
        Runs the specified tests on the given data and returns a list of results.

    test_categories() -> dict:
        Returns a dictionary mapping test category names to the corresponding test classes.

    test_scenarios() -> dict:
        Returns a dictionary mapping test class names to the available test scenarios for each class.
    """

    @staticmethod
    def transform(data: List[Sample], test_types: dict, model: ModelFactory):
        """
        Runs the specified tests on the given data and returns a list of results.

        Parameters
        ----------
        data : List[Sample]
            The data to be tested.
        test_types : dict
            A dictionary mapping test category names to lists of test scenario names.
        model: ModelFactory
            Model to be tested.

        Returns
        -------
        List[Results]
            A list of results from running the specified tests on the data.
        """

        all_results = []
        all_categories = TestFactory.test_catgories()
        # process = []
        for each in list(test_types.keys()):
            values = test_types[each]
            all_results.extend(
                all_categories[each](data, values, model).transform()
            )
        return all_results

    @classmethod
    def test_catgories(cls):
        """
        Returns a dictionary mapping test category names to the corresponding test classes.

        Returns
        -------
        dict
            A dictionary mapping test category names to the corresponding test classes.
        """
        return {cls.alias_name.lower(): cls for cls in ITests.__subclasses__()}

    @classmethod
    def test_scenarios(cls):
        """
        Returns a dictionary mapping test class names to the available test scenarios for each class.

        Returns
        -------
        dict
            A dictionary mapping test class names to the available test scenarios for each class.
        """

        return {cls.alias_name.lower(): cls.available_tests() for cls in ITests.__subclasses__()}


class ITests(ABC):
    """
    An abstract base class for defining different types of tests.

    ...

    Methods
    -------
    transform() -> List[Results]:
        Runs the test and returns the results.

    available_tests() -> List[str]:
        Returns a list of available test scenarios for the test class.
    """

    @abstractmethod
    def transform(self):
        """
        Runs the test and returns the results.

        Returns
        -------
        List[Results]
            A list of results from running the test.
        """

        return NotImplementedError

    @abstractclassmethod
    def available_tests(cls):
        """
        Returns a list of available test scenarios for the test class.

        Returns
        -------
        List[str]
            A list of available test scenarios for the test class.
        """

        return NotImplementedError


class RobustnessTestFactory(ITests):
    alias_name = "robustness"

    """
    A class for performing robustness tests on a given dataset.

    ...

    Attributes
    ----------
    supported_tests : dict
        A dictionary of supported robustness test scenarios.
    tests : dict
        A dictionary of test names and corresponding parameters.
    _data_handler : List[Sample]
        A list of `Sample` objects representing the input dataset.

    Methods
    -------
    transform() -> List[Sample]:
        Runs the robustness test and returns the resulting `Sample` objects.

    available_tests() -> dict:
        Returns a dictionary of available test scenarios for the `Robustness` class.
    """

    def __init__(
            self,
            data_handler: List[Sample],
            tests=None,
            model: ModelFactory = None
    ) -> None:

        """
        Initializes a new instance of the `Robustness` class.

        Parameters
        ----------
        data_handler : List[Sample]
            A list of `Sample` objects representing the input dataset.
        tests : dict, optional
            A dictionary of test names and corresponding parameters (default is None).
        """

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
            nltk.download('omw-1.4', quiet=True)
            nltk.download('wordnet', quiet=True)
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                               'label': [[i.entity for i in sample.expected_results.predictions]
                                         for sample in data_handler]})
            self.tests['swap_cohyponyms']['parameters'] = {}
            self.tests['swap_cohyponyms']['parameters']['labels'] = df.label.tolist()

        self._data_handler = data_handler

    def transform(self) -> List[Sample]:
        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns
        -------
        List[Sample]
            A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """
        # NOTE: I don't know if we need to work with a dataframe of if we can keep it as a List[Sample]
        all_samples = []
        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy,
                                                                            **params.get('parameters', {}))
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    def available_tests(cls) -> dict:

        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.

        """

        tests = {
            j: i for i in BaseRobustness.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class BiasTestFactory(ITests):
    alias_name = "bias"
    """
    A class for performing bias tests on a given dataset.

    ...

    Attributes
    ----------
    supported_tests : dict
        A dictionary of supported bias test scenarios.
    tests : dict
        A dictionary of test names and corresponding parameters.
    _data_handler : List[Sample]
        A list of `Sample` objects representing the input dataset.

    Methods
    -------
    transform() -> List[Sample]:
        Runs the bias test and returns the resulting `Sample` objects.

    available_tests() -> dict:
        Returns a dictionary of available test scenarios for the `bias` class.
    """

    def __init__(
            self,
            data_handler: List[Sample],
            tests=None,
            model: ModelFactory = None
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
            self.tests['replace_to_male_pronouns']['parameters']['pronouns_to_substitute'] = [item for sublist in list(
                female_pronouns.values()) for item in sublist] + [item for sublist in list(neutral_pronouns.values())
                                                                  for item in sublist]
            self.tests['replace_to_male_pronouns']['parameters']['pronoun_type'] = 'male'

        if 'replace_to_female_pronouns' in self.tests:
            self.tests['replace_to_female_pronouns']['parameters'] = {}
            self.tests['replace_to_female_pronouns']['parameters']['pronouns_to_substitute'] = [item for sublist in
                                                                                                list(
                                                                                                    male_pronouns.values())
                                                                                                for item in sublist] + [
                                                                                                   item for sublist in
                                                                                                   list(
                                                                                                       neutral_pronouns.values())
                                                                                                   for item in sublist]
            self.tests['replace_to_female_pronouns']['parameters']['pronoun_type'] = 'female'

        if 'replace_to_neutral_pronouns' in self.tests:
          self.tests['replace_to_neutral_pronouns']['parameters'] = {} 
          self.tests['replace_to_neutral_pronouns']['parameters']['pronouns_to_substitute'] = [item for sublist in list(female_pronouns.values()) for item in sublist] +[item for sublist in list(male_pronouns.values()) for item in sublist] 
          self.tests['replace_to_neutral_pronouns']['parameters']['pronoun_type'] = 'neutral'
          
        for income_level in ['Low-income', 'Lower-middle-income', 'Upper-middle-income', 'High-income']:
            economic_level = income_level.replace("-","_").lower()
            if f'replace_to_{economic_level}_country' in self.tests:
                countries_to_exclude = [v for k, v in country_economic_dict.items() if k != income_level]
                self.tests[f"replace_to_{economic_level}_country"]['parameters'] = {}
                self.tests[f"replace_to_{economic_level}_country"]['parameters']['country_names_to_substitute'] = get_substitution_names(countries_to_exclude)
                self.tests[f"replace_to_{economic_level}_country"]['parameters']['chosen_country_names'] = country_economic_dict[income_level]
                
        for religion in religion_wise_names.keys():
            if f"replace_to_{religion.lower()}_names" in self.tests:
                religion_to_exclude = [v for k, v in religion_wise_names.items() if k != religion]
                self.tests[f"replace_to_{religion.lower()}_names"]['parameters'] = {}
                self.tests[f"replace_to_{religion.lower()}_names"]['parameters']['names_to_substitute'] = get_substitution_names(religion_to_exclude)
                self.tests[f"replace_to_{religion.lower()}_names"]['parameters']['chosen_names'] = religion_wise_names[religion]
        
        ethnicity_first_names = {'white': white_names['first_names'], 'black': black_names['first_names'], 'hispanic': hispanic_names['first_names'], 'asian': asian_names['first_names']}
        for ethnicity in ['white', 'black', 'hispanic','asian']:
            test_key = f'replace_to_{ethnicity}_firstnames'
            if test_key in self.tests:
                self.tests[test_key]['parameters'] = {}
                self.tests[test_key]['parameters'] = {
                    'names_to_substitute': sum([ethnicity_first_names[e] for e in ethnicity_first_names if e != ethnicity], []),
                    'chosen_ethnicity_names': ethnicity_first_names[ethnicity]
                }
        
        ethnicity_last_names = {'white': white_names['last_names'], 'black': black_names['last_names'], 'hispanic': hispanic_names['last_names'], 'asian': asian_names['last_names']}
        for ethnicity in ['white', 'black', 'hispanic','asian','native_american','inter_racial']:
            test_key = f'replace_to_{ethnicity}_lastnames'
            if test_key in self.tests:
                self.tests[test_key]['parameters'] = {}
                self.tests[test_key]['parameters'] = {
                    'names_to_substitute': sum([ethnicity_last_names[e] for e in ethnicity_last_names if e != ethnicity], []),
                    'chosen_ethnicity_names': ethnicity_last_names[ethnicity]
                }
 
        
    def transform(self):

        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns
        -------
        List[Sample]
            A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """

        all_samples = []
        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy,
                                                                            **params.get('parameters', {}))
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    def available_tests(cls) -> dict:

        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.

        """

        tests = {
            j: i for i in BaseBias.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class RepresentationTestFactory(ITests):
    alias_name = "representation"

    """
    A class for performing representation tests on a given dataset.

    ...

    Attributes
    ----------
    supported_tests : dict
        A dictionary of supported representation test scenarios.
    tests : dict
        A dictionary of test names and corresponding parameters.
    _data_handler : List[Sample]
        A list of `Sample` objects representing the input dataset.

    Methods
    -------
    transform() -> List[Sample]:
        Runs the representation test and returns the resulting `Sample` objects.

    available_tests() -> dict:
        Returns a dictionary of available test scenarios for the `representation` class.
    """

    def __init__(
            self,
            data_handler: List[Sample],
            tests=None,
            model: ModelFactory = None
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

        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns
        -------
        List[Sample]
            A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """
        
        all_samples = [] 
        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(test_name,data_handler_copy, params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        
        return all_samples

    @classmethod
    def available_tests(cls) -> dict:

        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.

        """

        tests = {
            j: i for i in BaseRepresentation.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class AccuracyTestFactory(ITests):
    alias_name = "accuracy"
    """
    A class for performing accuracy tests on a given dataset.

    ...

    Attributes
    ----------
    supported_tests : dict
        A dictionary of supported accuracy test scenarios.
    tests : dict
        A dictionary of test names and corresponding parameters.
    _data_handler : List[Sample]
        A list of `Sample` objects representing the input dataset.

    Methods
    -------
    transform() -> List[Sample]:
        Runs the accuracy test and returns the resulting `Sample` objects.

    available_tests() -> dict:
        Returns a dictionary of available test scenarios for the `accuracy` class.
    """

    def __init__(
            self,
            data_handler: List[Sample],
            tests,
            model: ModelFactory
    ) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self._model_handler = model

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

        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns
        -------
        List[Sample]
            A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """

        all_samples = []
        for test_name, params in self.tests.items():            
            data_handler_copy = [x.copy() for x in self._data_handler]

            y_true = pd.Series(data_handler_copy).apply(lambda x: [y.entity for y in x.expected_results.predictions])
            X_test = pd.Series(data_handler_copy).apply(lambda x: x.original)
            y_pred = X_test.apply(self._model_handler.predict_raw)

            valid_indices = y_true.apply(len) == y_pred.apply(len)
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]

            y_true = y_true.explode().apply(lambda x: x.split("-")[-1])
            y_pred = y_pred.explode().apply(lambda x: x.split("-")[-1])

            transformed_samples = self.supported_tests[test_name].transform(y_true, y_pred, params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    def available_tests(cls) -> dict:

        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.

        """

        tests = {
            j: i for i in BaseAccuracy.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests
