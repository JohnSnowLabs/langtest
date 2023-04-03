from abc import ABC, abstractmethod
from typing import Dict, List
from tqdm import tqdm

import nltk
import pandas as pd

from nlptest.modelhandler import ModelFactory
from nlptest.transform.accuracy import BaseAccuracy
from nlptest.transform.fairness import BaseFairness
from .bias import BaseBias
from .representation import BaseRepresentation
from .robustness import BaseRobustness
from .utils import (A2B_DICT, asian_names, black_names, country_economic_dict, create_terminology, female_pronouns,
                    get_substitution_names, hispanic_names, inter_racial_names, male_pronouns, native_american_names,
                    neutral_pronouns, religion_wise_names, white_names)
from ..utils.custom_types import Result, Sample


class TestFactory:
    """
    A factory class for creating and running different types of tests on data.
    """
    is_augment = False

    @staticmethod
    def transform(data: List[Sample], test_types: dict, model: ModelFactory) -> List[Result]:
        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            data : List[Sample]
                The data to be tested.
            test_types : dict
                A dictionary mapping test category names to lists of test scenario names.
            model: ModelFactory
                Model to be tested.

        Returns:
            List[Results]
                A list of results from running the specified tests on the data.
        """
        all_results = []
        all_categories = TestFactory.test_categories()
        tests = tqdm(test_types.keys(), desc="Generating testcases...", disable=TestFactory.is_augment)
        for each in tests:
            tests.set_description(f"Generating testcases... ({each})")
            values = test_types[each]
            all_results.extend(
                all_categories[each](data, values, model).transform()
            )
        return all_results

    @classmethod
    def test_categories(cls) -> Dict:
        """
        Returns a dictionary mapping test category names to the corresponding test classes.

        Returns:
            Dict:
                A dictionary mapping test category names to the corresponding test classes.
        """
        return {cls.alias_name.lower(): cls for cls in ITests.__subclasses__()}

    @classmethod
    def test_scenarios(cls):
        """
        Returns a dictionary mapping test class names to the available test scenarios for each class.

        Returns:
            Dict:
                A dictionary mapping test class names to the available test scenarios for each class.
        """
        return {cls.alias_name.lower(): cls.available_tests() for cls in ITests.__subclasses__()}


class ITests(ABC):
    """
    An abstract base class for defining different types of tests.
    """
    alias_name = None

    @abstractmethod
    def transform(cls):
        """
        Runs the test and returns the results.

        Returns:
            List[Results]:
                A list of results from running the test.
        """
        return NotImplementedError

    @abstractmethod
    def available_tests(cls):
        """
        Returns a list of available test scenarios for the test class.

        Returns:
            List[str]:
                A list of available test scenarios for the test class.
        """
        return NotImplementedError


class RobustnessTestFactory(ITests):
    """
    A class for performing robustness tests on a given dataset.
    """
    alias_name = "robustness"

    def __init__(
            self,
            data_handler: List[Sample],
            tests: Dict = None,
            model: ModelFactory = None
    ) -> None:

        """
        Initializes a new instance of the `Robustness` class.

        Args:
            data_handler (List[Sample]):
                A list of `Sample` objects representing the input dataset.
            tests Optional[Dict]:
                A dictionary of test names and corresponding parameters (default is None).
        """

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

        if 'swap_entities' in self.tests:
            # TODO: check if we can get rid of pandas here
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                               'label': [[i.entity for i in sample.expected_results.predictions]
                                         for sample in data_handler]})
            params = self.tests['swap_entities']
            if len(params.get('parameters', {}).get('terminology', {})) == 0:
                params['parameters'] = {}
                params['parameters']['terminology'] = create_terminology(df)
                params['parameters']['labels'] = df.label.tolist()

        if "american_to_british" in self.tests:
            self.tests['american_to_british']['parameters'] = {}
            self.tests['american_to_british']['parameters']['accent_map'] = A2B_DICT

        if "british_to_american" in self.tests:
            self.tests['british_to_american']['parameters'] = {}
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

        Returns:
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
                if not TestFactory.is_augment:
                    sample.expected_results = self._model_handler(sample.original)
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
    """
    A class for performing bias tests on a given dataset.
    """
    alias_name = "bias"

    def __init__(
            self,
            data_handler: List[Sample],
            tests: Dict = None,
            model: ModelFactory = None
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

        if 'replace_to_male_pronouns' in self.tests:
            self.tests['replace_to_male_pronouns']['parameters'] = {}
            self.tests['replace_to_male_pronouns']['parameters']['pronouns_to_substitute'] = [item for sublist in list(
                female_pronouns.values()) for item in sublist] + [item for sublist in list(neutral_pronouns.values())
                                                                  for item in sublist]
            self.tests['replace_to_male_pronouns']['parameters']['pronoun_type'] = 'male'

        if 'replace_to_female_pronouns' in self.tests:
            self.tests['replace_to_female_pronouns']['parameters'] = {}
            self.tests['replace_to_female_pronouns']['parameters']['pronouns_to_substitute'] = \
                [item for sublist in list(male_pronouns.values()) for item in sublist] + \
                [item for sublist in list(neutral_pronouns.values()) for item in sublist]
            self.tests['replace_to_female_pronouns']['parameters']['pronoun_type'] = 'female'

        if 'replace_to_neutral_pronouns' in self.tests:
            self.tests['replace_to_neutral_pronouns']['parameters'] = {}
            self.tests['replace_to_neutral_pronouns']['parameters']['pronouns_to_substitute'] = \
                [item for sublist in list(female_pronouns.values()) for item in sublist] + \
                [item for sublist in list(male_pronouns.values()) for item in sublist]
            self.tests['replace_to_neutral_pronouns']['parameters']['pronoun_type'] = 'neutral'

        for income_level in ['Low-income', 'Lower-middle-income', 'Upper-middle-income', 'High-income']:
            economic_level = income_level.replace("-", "_").lower()
            if f'replace_to_{economic_level}_country' in self.tests:
                countries_to_exclude = [v for k, v in country_economic_dict.items() if k != income_level]
                self.tests[f"replace_to_{economic_level}_country"]['parameters'] = {}
                self.tests[f"replace_to_{economic_level}_country"]['parameters'][
                    'country_names_to_substitute'] = get_substitution_names(countries_to_exclude)
                self.tests[f"replace_to_{economic_level}_country"]['parameters']['chosen_country_names'] = \
                    country_economic_dict[income_level]

        for religion in religion_wise_names.keys():
            if f"replace_to_{religion.lower()}_names" in self.tests:
                religion_to_exclude = [v for k, v in religion_wise_names.items() if k != religion]
                self.tests[f"replace_to_{religion.lower()}_names"]['parameters'] = {}
                self.tests[f"replace_to_{religion.lower()}_names"]['parameters'][
                    'names_to_substitute'] = get_substitution_names(religion_to_exclude)
                self.tests[f"replace_to_{religion.lower()}_names"]['parameters']['chosen_names'] = religion_wise_names[
                    religion]

        ethnicity_first_names = {'white': white_names['first_names'], 'black': black_names['first_names'],
                                 'hispanic': hispanic_names['first_names'], 'asian': asian_names['first_names']}
        for ethnicity in ['white', 'black', 'hispanic', 'asian']:
            test_key = f'replace_to_{ethnicity}_firstnames'
            if test_key in self.tests:
                self.tests[test_key]['parameters'] = {}
                self.tests[test_key]['parameters'] = {
                    'names_to_substitute': sum(
                        [ethnicity_first_names[e] for e in ethnicity_first_names if e != ethnicity], []),
                    'chosen_ethnicity_names': ethnicity_first_names[ethnicity]
                }

        ethnicity_last_names = {'white': white_names['last_names'], 'black': black_names['last_names'],
                                'hispanic': hispanic_names['last_names'], 'asian': asian_names['last_names']}
        for ethnicity in ['white', 'black', 'hispanic', 'asian', 'native_american', 'inter_racial']:
            test_key = f'replace_to_{ethnicity}_lastnames'
            if test_key in self.tests:
                self.tests[test_key]['parameters'] = {}
                self.tests[test_key]['parameters'] = {
                    'names_to_substitute': sum(
                        [ethnicity_last_names[e] for e in ethnicity_last_names if e != ethnicity], []),
                    'chosen_ethnicity_names': ethnicity_last_names[ethnicity]
                }

    def transform(self) -> List[Sample]:
        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns:
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
                if not TestFactory.is_augment:
                    sample.expected_results = self._model_handler(sample.original)
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    def available_tests(cls) -> Dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            Dict: A dictionary of test names and classes.

        """

        tests = {
            j: i for i in BaseBias.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class RepresentationTestFactory(ITests):
    """
    A class for performing representation tests on a given dataset.
    """
    alias_name = "representation"

    def __init__(
            self,
            data_handler: List[Sample],
            tests: Dict = None,
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

    def transform(self) -> List[Sample]:
        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """
        all_samples = []
        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(test_name, data_handler_copy, params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)

        return all_samples

    @classmethod
    def available_tests(cls) -> Dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            Dict: A dictionary of test names and classes.
        """
        tests = {
            j: i for i in BaseRepresentation.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class FairnessTestFactory(ITests):
    """
    A class for performing fairness tests on a given dataset.
    """
    alias_name = "fairness"

    def __init__(
            self,
            data_handler: List[Sample],
            tests: Dict,
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

    def transform(self) -> List[Sample]:
        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """

        all_samples = []
        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy, self._model_handler,
                                                                            params)
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
            j: i for i in BaseFairness.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class AccuracyTestFactory(ITests):
    """
    A class for performing accuracy tests on a given dataset.
    """
    alias_name = "accuracy"

    def __init__(
            self,
            data_handler: List[Sample],
            tests: Dict,
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

    def transform(self) -> List[Sample]:
        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """
        # TODO: get rid of pandas
        all_samples = []
        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]

            try:
                y_true = pd.Series(data_handler_copy).apply(
                    lambda x: [y.entity for y in x.expected_results.predictions])
            except:
                y_true = pd.Series(data_handler_copy).apply(lambda x: [y.label for y in x.expected_results.predictions])

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
