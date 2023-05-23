from nlptest.utils.custom_types.sample import QASample, SequenceClassificationSample, NERSample
from nlptest.utils.gender_classifier import GenderClassifier
from ..utils.custom_types import Result, Sample
from .utils import ( default_user_prompt, A2B_DICT, asian_names, black_names, country_economic_dict, create_terminology, female_pronouns,
                    get_substitution_names, hispanic_names, inter_racial_names, male_pronouns, native_american_names,
                    neutral_pronouns, religion_wise_names, white_names)
from .robustness import BaseRobustness
from .representation import BaseRepresentation
from .bias import BaseBias
from .fairness import BaseFairness
from .accuracy import BaseAccuracy
from ..modelhandler import ModelFactory
import pandas as pd
import logging
from tqdm.asyncio import tqdm
from typing import Dict, List
from abc import ABC, abstractmethod
import asyncio
import nest_asyncio
nest_asyncio.apply()


class TestFactory:
    """
    A factory class for creating and running different types of tests on data.
    """
    is_augment = False

    @staticmethod
    def transform(task: str, data: List[Sample], test_types: dict, *args, **kwargs) -> List[Result]:
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
        test_names = list(test_types.keys())
        if 'defaults' in test_names:
            test_names.pop(test_names.index("defaults"))
        tests = tqdm(test_names, desc="Generating testcases...",
                     disable=TestFactory.is_augment)
        m_data = kwargs.get('m_data', None)

        # Check test-task supportance
        for test_category in tests:
            if test_category in all_categories.keys():
                sub_test_types = test_types[test_category]
                for sub_test in sub_test_types:
                    supported = all_categories[test_category].available_tests()[sub_test].supported_tasks
                    if task not in supported:
                        raise ValueError(f"The test type \"{sub_test}\" is not supported for the task \"{task}\". \"{sub_test}\" only supports {supported}.")
            elif test_category != "defaults":
                raise ValueError(f"The test category {test_category} does not exist. Available categories are: {all_categories.keys()}.")
                
        # Generate testcases
        for each in tests:
            tests.set_description(f"Generating testcases... ({each})")
            if each in all_categories:
                sub_test_types = test_types[each]
                all_results.extend(
                    all_categories[each](m_data, sub_test_types,
                                        raw_data=data).transform()
                    if each in ["robustness", "bias"] and m_data
                    else all_categories[each](data, sub_test_types).transform()
                )
        tests.close()
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

    @staticmethod
    def run(samples_list: List[Sample], model_handler: ModelFactory,  **kwargs):
        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            samples_list : List[Sample]
            model_handler : ModelFactory

        """
        async_tests = TestFactory.async_run(
            samples_list, model_handler, **kwargs)
        temp_res = asyncio.run(async_tests)
        results = []
        for each in temp_res:
            results.extend(each.result())
        return results

    @classmethod
    async def async_run(cls, samples_list: List[Sample], model_handler: ModelFactory, **kwargs):

        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            samples_list : List[Sample]
            model_handler : ModelFactory

        """
        hash_samples = {}
        for sample in samples_list:
            if sample.category not in hash_samples:
                hash_samples[sample.category] = {}
            if sample.test_type not in hash_samples[sample.category]:
                hash_samples[sample.category][sample.test_type] = [sample]
            else:
                hash_samples[sample.category][sample.test_type].append(sample)

        all_categories = TestFactory.test_categories()
        tests = tqdm(total=len(samples_list), desc="Running testcases... ",
                     position=0, disable=TestFactory.is_augment)
        all_results = []
        for each in hash_samples:
            values = hash_samples[each]
            category_output = all_categories[each].run(
                values, model_handler, progress_bar=tests, **kwargs)
            if type(category_output) == list:
                all_results.extend(category_output)
            else:
                all_results.append(category_output)
        return await asyncio.gather(*all_results)


class ITests(ABC):
    """
    An abstract base class for defining different types of tests.
    """
    alias_name = None

    @abstractmethod
    def transform(self):
        """
        Runs the test and returns the results.

        Returns:
            List[Results]:
                A list of results from running the test.
        """
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def available_tests():
        """
        Returns a list of available test scenarios for the test class.

        Returns:
            List[str]:
                A list of available test scenarios for the test class.
        """
        return NotImplementedError

    @classmethod
    def run(cls, sample_list: Dict[str, List[Sample]], model: ModelFactory, **kwargs) -> List[Sample]:
        """
        Runs the specified tests on the given data and returns a list of results.

        Args:
            sample_list (Dict[str, List[Sample]]):
                A dictionary mapping test scenario names to a list of `Sample` objects.
            model (ModelFactory):
                A `ModelFactory` object representing the model to be tested.

        Returns:
            List[Sample]: A list of `Sample` objects with the test results.

        """
        supported_tests = cls.available_tests()
        tasks = []
        for test_name, samples in sample_list.items():
            test_output = supported_tests[test_name].async_run(
                samples, model, **kwargs)
            if type(test_output) == list:
                tasks.extend(test_output)
            else:
                tasks.append(test_output)

        return tasks


class RobustnessTestFactory(ITests):
    """
    A class for performing robustness tests on a given dataset.
    """
    alias_name = "robustness"

    def __init__(
            self,
            data_handler: List[Sample],
            tests: Dict = None,
            **kwargs
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
        self.kwargs = kwargs

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
            raw_data = self.kwargs.get('raw_data', self._data_handler)
            df = pd.DataFrame({'text': [sample.original for sample in raw_data],
                               'label': [[i.entity for i in sample.expected_results.predictions]
                                         for sample in raw_data]})
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
            self.tests['british_to_american']['parameters']['accent_map'] = {
                v: k for k, v in A2B_DICT.items()}

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
            if TestFactory.is_augment:
                data_handler_copy = [x.copy() for x in self._data_handler]
            elif test_name in ["swap_entities"]:
                data_handler_copy = [x.copy()
                                     for x in self.kwargs.get('raw_data', [])]
            else:
                data_handler_copy = [x.copy() for x in self._data_handler]

            transformed_samples = self.supported_tests[test_name].transform(data_handler_copy,
                                                                            **params.get('parameters', {}))
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @staticmethod
    def available_tests() -> dict:
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
            **kwargs
    ) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

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
                [item for sublist in list(neutral_pronouns.values())
                 for item in sublist]
            self.tests['replace_to_female_pronouns']['parameters']['pronoun_type'] = 'female'

        if 'replace_to_neutral_pronouns' in self.tests:
            self.tests['replace_to_neutral_pronouns']['parameters'] = {}
            self.tests['replace_to_neutral_pronouns']['parameters']['pronouns_to_substitute'] = \
                [item for sublist in list(female_pronouns.values()) for item in sublist] + \
                [item for sublist in list(male_pronouns.values())
                 for item in sublist]
            self.tests['replace_to_neutral_pronouns']['parameters']['pronoun_type'] = 'neutral'

        for income_level in ['Low-income', 'Lower-middle-income', 'Upper-middle-income', 'High-income']:
            economic_level = income_level.replace("-", "_").lower()
            if f'replace_to_{economic_level}_country' in self.tests:
                countries_to_exclude = [
                    v for k, v in country_economic_dict.items() if k != income_level]
                self.tests[f"replace_to_{economic_level}_country"]['parameters'] = {
                }
                self.tests[f"replace_to_{economic_level}_country"]['parameters'][
                    'country_names_to_substitute'] = get_substitution_names(countries_to_exclude)
                self.tests[f"replace_to_{economic_level}_country"]['parameters']['chosen_country_names'] = \
                    country_economic_dict[income_level]

        for religion in religion_wise_names.keys():
            if f"replace_to_{religion.lower()}_names" in self.tests:
                religion_to_exclude = [
                    v for k, v in religion_wise_names.items() if k != religion]
                self.tests[f"replace_to_{religion.lower()}_names"]['parameters'] = {
                }
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
                                'hispanic': hispanic_names['last_names'], 'asian': asian_names['last_names'],
                                'native_american': native_american_names['last_names'], 'inter_racial': inter_racial_names['last_names']}
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
            all_samples.extend(transformed_samples)
        return all_samples

    @staticmethod
    def available_tests() -> Dict:
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
            **kwargs
    ) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

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
            transformed_samples = self.supported_tests[test_name].transform(
                test_name, data_handler_copy, params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)

        return all_samples

    @staticmethod
    def available_tests() -> Dict:
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
            **kwargs
    ) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

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

            if isinstance(data_handler_copy[0], NERSample):
                y_true = pd.Series(data_handler_copy).apply(lambda x: [y.entity for y in x.expected_results.predictions])
            elif isinstance(data_handler_copy[0], SequenceClassificationSample):
                y_true = pd.Series(data_handler_copy).apply(lambda x: [y.label for y in x.expected_results.predictions])
            elif data_handler_copy[0].task in ["question-answering", "summarization"]:
                y_true = pd.Series(data_handler_copy).apply(lambda x: x.expected_results)

            y_true = y_true.explode().apply(lambda x: x.split("-")
                                            [-1] if isinstance(x, str) else x)
            y_true = y_true.dropna()
            params["test_name"] = test_name
            transformed_samples = self.supported_tests[test_name].transform(
                y_true, params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @staticmethod
    def available_tests() -> dict:
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

    @classmethod
    def run(cls, sample_list: Dict[str, List[Sample]], model: ModelFactory, raw_data: List[Sample], **kwargs):
        """
        Runs the fairness tests on the given model and dataset.

        Args:
            sample_list (Dict[str, List[Sample]]): A dictionary of test names and corresponding `Sample` objects.
            model (ModelFactory): The model to be tested.
            raw_data (List[Sample]): The raw dataset.

        """

        grouped_data = cls.get_gendered_data(raw_data)

        for gender, data in grouped_data.items():
            if len(data) == 0:
                grouped_data[gender] = [[],[]]
            else:
                if isinstance(data[0], NERSample):
                    y_true = pd.Series(data).apply(lambda x: [y.entity for y in x.expected_results.predictions])
                    X_test = pd.Series(data).apply(lambda sample: sample.original)
                    y_pred = X_test.apply(model.predict_raw)
                    valid_indices = y_true.apply(len) == y_pred.apply(len)
                    y_true = y_true[valid_indices]
                    y_pred = y_pred[valid_indices]
                    y_true = y_true.explode()
                    y_pred = y_pred.explode()
                    y_pred = y_pred.apply(lambda x: x.split("-")[-1])
                    y_true = y_true.apply(lambda x: x.split("-")[-1])
                
                elif isinstance(data[0], SequenceClassificationSample):
                    y_true = pd.Series(data).apply(lambda x: [y.label for y in x.expected_results.predictions])
                    y_true = y_true.apply(lambda x: x[0])
                    X_test = pd.Series(data).apply(lambda sample: sample.original)
                    y_pred = X_test.apply(model.predict_raw)
                    y_true = y_true.explode()
                    y_pred = y_pred.explode()
                
                elif data[0].task == "question-answering":
                    dataset_name = data[0].dataset_name.split('-')[0].lower()
                    user_prompt = kwargs.get('user_prompt', default_user_prompt.get(dataset_name, ""))
                    prompt_template = """Context: {context}\nQuestion: {question}\n """ + user_prompt
                    
                    if data[0].expected_results is None:
                        raise RuntimeError(f'The dataset {dataset_name} does not contain labels and fairness tests cannot be run with it. Skipping the fairness tests.')
                    y_true = pd.Series(data).apply(lambda x: x.expected_results)
                    X_test = pd.Series(data)
                    y_pred = X_test.apply(lambda sample: model(text={'context':sample.original_context, 'question': sample.original_question}, prompt={"template":prompt_template, 'input_variables':["context", "question"]}))
                    y_pred = y_pred.apply(lambda x: x.strip())

                elif data[0].task == "summarization":
                    dataset_name = data[0].dataset_name.split('-')[0].lower()
                    user_prompt = kwargs.get('user_prompt', default_user_prompt.get(dataset_name, ""))
                    prompt_template =  user_prompt + """Context: {context}\n\n Summary: """
                    if data[0].expected_results is None:
                        raise RuntimeError(f'The dataset {dataset_name} does not contain labels and fairness tests cannot be run with it. Skipping the fairness tests.')
                        
                    y_true = pd.Series(data).apply(lambda x: x.expected_results)
                    X_test = pd.Series(data)
                    y_pred = X_test.apply(lambda sample: model(text={'context':sample.original}, prompt={"template":prompt_template, 'input_variables':["context"]}))
                    y_pred = y_pred.apply(lambda x: x.strip())
                
                if kwargs['is_default']:
                    y_pred = y_pred.apply(lambda x: '1' if x in ['pos', 'LABEL_1', 'POS'] else (
                        '0' if x in ['neg', 'LABEL_0', 'NEG'] else x))
                
                grouped_data[gender] = [y_true, y_pred]

        supported_tests = cls.available_tests()
        
        kwargs["task"] = type(QASample)
        tasks = []
        for test_name, samples in sample_list.items():
            tasks.append(
                supported_tests[test_name].async_run(samples, grouped_data, **kwargs))
        return tasks
    
    @staticmethod
    def get_gendered_data(data):
        """Split list of samples into gendered lists."""
        data = pd.Series(data)
        if isinstance(data[0], QASample):
            sentences = data.apply(lambda x: f"{x.original_context} {x.original_question}")
        else:
            sentences = data.apply(lambda x: x.original)
        classifier = GenderClassifier()
        genders = sentences.apply(classifier.predict)
        gendered_data = {
            "male": data[genders == "male"].tolist(),
            "female": data[genders == "female"].tolist(),
            "unknown": data[genders == "unknown"].tolist(),
        }
        return gendered_data


class AccuracyTestFactory(ITests):
    """
    A class for performing accuracy tests on a given dataset.
    """
    alias_name = "accuracy"

    def __init__(
            self,
            data_handler: List[Sample],
            tests: Dict,
            **kwargs
    ) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

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

            if data_handler_copy[0].task=="ner":
                y_true = pd.Series(data_handler_copy).apply(lambda x: [y.entity for y in x.expected_results.predictions])
                y_true = y_true.explode().apply(lambda x: x.split("-")
                                                [-1] if isinstance(x, str) else x)
            elif data_handler_copy[0].task=="text-classification":
                y_true = pd.Series(data_handler_copy).apply(lambda x: [y.label for y in x.expected_results.predictions]).explode()
            elif data_handler_copy[0].task=="question-answering" or data_handler_copy[0].task=="summarization":
                y_true = pd.Series(data_handler_copy).apply(lambda x: x.expected_results).explode()

            y_true = y_true.dropna()
            params["test_name"] = test_name
            transformed_samples = self.supported_tests[test_name].transform(
                y_true, params)
            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @staticmethod
    def available_tests() -> dict:
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

    @classmethod
    def run(cls, sample_list: Dict[str, List[Sample]], model: ModelFactory, raw_data: List[Sample], **kwargs):
        """
        Runs the accuracy tests on the given model and dataset.

        Args:
            sample_list (Dict[str, List[Sample]]): A dictionary of test names and corresponding `Sample` objects.
            model (ModelFactory): The model to be tested.
            raw_data (List[Sample]): The raw dataset.

        """

        if isinstance(raw_data[0], NERSample):
            y_true = pd.Series(raw_data).apply(lambda x: [y.entity for y in x.expected_results.predictions])
            X_test = pd.Series(raw_data).apply(lambda sample: sample.original)
            y_pred = X_test.apply(model.predict_raw)
            valid_indices = y_true.apply(len) == y_pred.apply(len)
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            y_true = y_true.explode()
            y_pred = y_pred.explode()
            y_pred = y_pred.apply(lambda x: x.split("-")[-1])
            y_true = y_true.apply(lambda x: x.split("-")[-1])
        
        elif isinstance(raw_data[0], SequenceClassificationSample):
            y_true = pd.Series(raw_data).apply(lambda x: [y.label for y in x.expected_results.predictions])
            y_true = y_true.apply(lambda x: x[0])
            X_test = pd.Series(raw_data).apply(lambda sample: sample.original)
            y_pred = X_test.apply(model.predict_raw)
            y_true = y_true.explode()
            y_pred = y_pred.explode()
        
        elif raw_data[0].task=="question-answering":
            dataset_name = raw_data[0].dataset_name.split('-')[0].lower()
            user_prompt = kwargs.get('user_prompt', default_user_prompt.get(dataset_name, ""))
            prompt_template = """Context: {context}\nQuestion: {question}\n """ + user_prompt
            
            if raw_data[0].expected_results is None:
                        raise RuntimeError(f'The dataset {dataset_name} does not contain labels and fairness tests cannot be run with it. Skipping the fairness tests.')
            y_true = pd.Series(raw_data).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data)
            y_pred = X_test.apply(lambda sample: model(text={'context':sample.original_context, 'question': sample.original_question}, prompt={"template":prompt_template, 'input_variables':["context", "question"]}))
            y_pred = y_pred.apply(lambda x: x.strip())
        
        elif raw_data[0].task=="summarization":
            dataset_name = raw_data[0].dataset_name.split('-')[0].lower()
            user_prompt = kwargs.get('user_prompt', default_user_prompt.get(dataset_name, ""))
            prompt_template =  user_prompt + """Context: {context}\n\n Summary: """
            if raw_data[0].expected_results is None:
                raise RuntimeError(f'The dataset {dataset_name} does not contain labels and fairness tests cannot be run with it. Skipping the fairness tests.')
                        
            y_true = pd.Series(raw_data).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data)
            y_pred = X_test.apply(lambda sample: model(text={'context':sample.original}, prompt={"template":prompt_template, 'input_variables':["context"]}))
            y_pred = y_pred.apply(lambda x: x.strip())
        
        if kwargs['is_default']:
            y_pred = y_pred.apply(lambda x: '1' if x in ['pos', 'LABEL_1', 'POS'] else (
                '0' if x in ['neg', 'LABEL_0', 'NEG'] else x))


        supported_tests = cls.available_tests()
        
        tasks = []
        for test_name, samples in sample_list.items():
            tasks.append(
                supported_tests[test_name].async_run(samples, y_true, y_pred, **kwargs))
        return tasks
