import asyncio
import copy
from typing import Dict, List

import nest_asyncio
import pandas as pd

from langtest.transform.performance import BasePerformance
from langtest.transform.security import BaseSecurity

from .accuracy import BaseAccuracy
from .bias import BaseBias
from .fairness import BaseFairness
from .representation import BaseRepresentation
from .robustness import BaseRobustness
from .toxicity import BaseToxicity
from .ideology import BaseIdeology
from .sensitivity import BaseSensitivity
from .sycophancy import BaseSycophancy
from .constants import (
    A2B_DICT,
    asian_names,
    black_names,
    country_economic_dict,
    female_pronouns,
    hispanic_names,
    inter_racial_names,
    male_pronouns,
    native_american_names,
    neutral_pronouns,
    religion_wise_names,
    white_names,
)
from .utils import get_substitution_names, create_terminology, filter_unique_samples
from ..modelhandler import ModelAPI
from ..utils.custom_types.sample import (
    NERSample,
    QASample,
    SequenceClassificationSample,
    Sample,
)
from ..utils.custom_types.helpers import default_user_prompt
from langtest.transform.base import ITests, TestFactory
from langtest.transform.grammar import GrammarTestFactory
from ..errors import Errors, Warnings
from ..logger import logger as logging

nest_asyncio.apply()


class RobustnessTestFactory(ITests):
    """
    A class for performing robustness tests on a given dataset.
    """

    alias_name = "robustness"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
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
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

        if "swap_entities" in self.tests:
            # TODO: check if we can get rid of pandas here
            raw_data = self.kwargs.get("raw_data", self._data_handler)
            df = pd.DataFrame(
                {
                    "text": [sample.original for sample in raw_data],
                    "label": [
                        [i.entity for i in sample.expected_results.predictions]
                        for sample in raw_data
                    ],
                }
            )
            params = self.tests["swap_entities"]
            if len(params.get("parameters", {}).get("terminology", {})) == 0:
                params["parameters"] = {}
                params["parameters"]["terminology"] = create_terminology(df)
                params["parameters"]["labels"] = df.label.tolist()

        if "american_to_british" in self.tests:
            self.tests["american_to_british"]["parameters"] = {}
            self.tests["american_to_british"]["parameters"]["accent_map"] = A2B_DICT

        if "british_to_american" in self.tests:
            self.tests["british_to_american"]["parameters"] = {}
            self.tests["british_to_american"]["parameters"]["accent_map"] = {
                v: k for k, v in A2B_DICT.items()
            }

        self._data_handler = data_handler

    def transform(self) -> List[Sample]:
        """
        Runs the robustness test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]
                A list of `Sample` objects representing the resulting dataset after running the robustness test.
        """
        all_samples = []
        no_transformation_applied_tests = {}
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            if TestFactory.is_augment:
                data_handler_copy = [x.copy() for x in self._data_handler]
            elif test_name in ["swap_entities"]:
                data_handler_copy = [x.copy() for x in self.kwargs.get("raw_data", [])]
            else:
                data_handler_copy = [x.copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform

            if (
                TestFactory.task in ("question-answering", "summarization")
                and test_name != "multiple_perturbations"
            ):
                _ = [
                    sample.transform(
                        test_func,
                        params.get("parameters", {}),
                        prob=params.pop("prob", 1.0),
                    )
                    if hasattr(sample, "transform")
                    else sample
                    for sample in data_handler_copy
                ]
                transformed_samples = data_handler_copy

            elif test_name == "multiple_perturbations" and TestFactory.task in (
                "question-answering",
                "summarization",
            ):
                transformed_samples = []
                prob = params.pop("prob", 1.0)
                for key, perturbations in params.items():
                    if key.startswith("perturbations"):
                        perturbation_number = key[len("perturbations") :]

                        if "american_to_british" in perturbations:
                            self.tests.setdefault("american_to_british", {})[
                                "parameters"
                            ] = {"accent_map": A2B_DICT}

                        if "british_to_american" in perturbations:
                            self.tests.setdefault("british_to_american", {})[
                                "parameters"
                            ] = {"accent_map": {v: k for k, v in A2B_DICT.items()}}
                        _ = [
                            sample.transform(
                                func=test_func,
                                params=self.tests,
                                prob=prob,
                                perturbations=perturbations,
                            )
                            if hasattr(sample, "transform")
                            else sample
                            for sample in data_handler_copy
                        ]
                        transformed_samples_perturbation = copy.deepcopy(
                            data_handler_copy
                        )  # Create a deep copy
                        if perturbation_number != "":
                            test_type = "-".join(
                                str(perturbation)
                                if not isinstance(perturbation, dict)
                                else next(iter(perturbation))
                                for perturbation in perturbations
                            )
                            for sample in transformed_samples_perturbation:
                                sample.test_type = test_type

                        transformed_samples.extend(transformed_samples_perturbation)
                    elif key != "min_pass_rate":
                        raise ValueError(Errors.E050(key=key))

            elif (
                test_name == "multiple_perturbations"
                and TestFactory.task == "text-classification"
            ):
                transformed_samples = []
                prob = params.pop("prob", 1.0)
                for key, perturbations in params.items():
                    if key.startswith("perturbations"):
                        perturbation_number = key[len("perturbations") :]

                        if "american_to_british" in perturbations:
                            self.tests.setdefault("american_to_british", {})[
                                "parameters"
                            ] = {"accent_map": A2B_DICT}

                        if "british_to_american" in perturbations:
                            self.tests.setdefault("british_to_american", {})[
                                "parameters"
                            ] = {"accent_map": {v: k for k, v in A2B_DICT.items()}}

                        transformed_samples_perturbation = test_func(
                            data_handler_copy,
                            perturbations,
                            prob=prob,
                            config=self.tests,
                        )

                        if perturbation_number != "":
                            test_type = "-".join(
                                str(perturbation)
                                if not isinstance(perturbation, dict)
                                else next(iter(perturbation))
                                for perturbation in perturbations
                            )
                            for sample in transformed_samples_perturbation:
                                sample.test_type = test_type
                        transformed_samples.extend(transformed_samples_perturbation)

                    elif key not in ("min_pass_rate", "prob"):
                        raise ValueError(Errors.E050(key=key))

            elif test_name == "multiple_perturbations" and TestFactory.task == "ner":
                raise ValueError(Errors.E051())

            else:
                transformed_samples = test_func(
                    data_handler_copy,
                    **params.get("parameters", {}),
                    prob=params.pop("prob", 1.0),
                )
            new_transformed_samples, removed_samples_tests = filter_unique_samples(
                TestFactory.task, transformed_samples, test_name
            )
            all_samples.extend(new_transformed_samples)

            no_transformation_applied_tests.update(removed_samples_tests)

        if no_transformation_applied_tests:
            warning_message = Warnings._W009
            for test, count in no_transformation_applied_tests.items():
                warning_message += Warnings._W010.format(
                    test=test, count=count, total_sample=len(self._data_handler)
                )

            logging.warning(warning_message)

        return all_samples

    @staticmethod
    def available_tests() -> dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """
        tests = {
            j: i
            for i in BaseRobustness.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class BiasTestFactory(ITests):
    """
    A class for performing bias tests on a given dataset.
    """

    alias_name = "bias"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

        if "replace_to_male_pronouns" in self.tests:
            self.tests["replace_to_male_pronouns"]["parameters"] = {}
            self.tests["replace_to_male_pronouns"]["parameters"][
                "pronouns_to_substitute"
            ] = [
                item for sublist in list(female_pronouns.values()) for item in sublist
            ] + [
                item for sublist in list(neutral_pronouns.values()) for item in sublist
            ]
            self.tests["replace_to_male_pronouns"]["parameters"]["pronoun_type"] = "male"

        if "replace_to_female_pronouns" in self.tests:
            self.tests["replace_to_female_pronouns"]["parameters"] = {}
            self.tests["replace_to_female_pronouns"]["parameters"][
                "pronouns_to_substitute"
            ] = [item for sublist in list(male_pronouns.values()) for item in sublist] + [
                item for sublist in list(neutral_pronouns.values()) for item in sublist
            ]
            self.tests["replace_to_female_pronouns"]["parameters"][
                "pronoun_type"
            ] = "female"

        if "replace_to_neutral_pronouns" in self.tests:
            self.tests["replace_to_neutral_pronouns"]["parameters"] = {}
            self.tests["replace_to_neutral_pronouns"]["parameters"][
                "pronouns_to_substitute"
            ] = [
                item for sublist in list(female_pronouns.values()) for item in sublist
            ] + [
                item for sublist in list(male_pronouns.values()) for item in sublist
            ]
            self.tests["replace_to_neutral_pronouns"]["parameters"][
                "pronoun_type"
            ] = "neutral"

        for income_level in [
            "Low-income",
            "Lower-middle-income",
            "Upper-middle-income",
            "High-income",
        ]:
            economic_level = income_level.replace("-", "_").lower()
            if f"replace_to_{economic_level}_country" in self.tests:
                countries_to_exclude = [
                    v for k, v in country_economic_dict.items() if k != income_level
                ]
                self.tests[f"replace_to_{economic_level}_country"]["parameters"] = {}
                self.tests[f"replace_to_{economic_level}_country"]["parameters"][
                    "country_names_to_substitute"
                ] = get_substitution_names(countries_to_exclude)
                self.tests[f"replace_to_{economic_level}_country"]["parameters"][
                    "chosen_country_names"
                ] = country_economic_dict[income_level]

        for religion in religion_wise_names.keys():
            if f"replace_to_{religion.lower()}_names" in self.tests:
                religion_to_exclude = [
                    v for k, v in religion_wise_names.items() if k != religion
                ]
                self.tests[f"replace_to_{religion.lower()}_names"]["parameters"] = {}
                self.tests[f"replace_to_{religion.lower()}_names"]["parameters"][
                    "names_to_substitute"
                ] = get_substitution_names(religion_to_exclude)
                self.tests[f"replace_to_{religion.lower()}_names"]["parameters"][
                    "chosen_names"
                ] = religion_wise_names[religion]

        ethnicity_first_names = {
            "white": white_names["first_names"],
            "black": black_names["first_names"],
            "hispanic": hispanic_names["first_names"],
            "asian": asian_names["first_names"],
        }
        for ethnicity in ["white", "black", "hispanic", "asian"]:
            test_key = f"replace_to_{ethnicity}_firstnames"
            if test_key in self.tests:
                self.tests[test_key]["parameters"] = {}
                self.tests[test_key]["parameters"] = {
                    "names_to_substitute": sum(
                        [
                            ethnicity_first_names[e]
                            for e in ethnicity_first_names
                            if e != ethnicity
                        ],
                        [],
                    ),
                    "chosen_ethnicity_names": ethnicity_first_names[ethnicity],
                }

        ethnicity_last_names = {
            "white": white_names["last_names"],
            "black": black_names["last_names"],
            "hispanic": hispanic_names["last_names"],
            "asian": asian_names["last_names"],
            "native_american": native_american_names["last_names"],
            "inter_racial": inter_racial_names["last_names"],
        }
        for ethnicity in [
            "white",
            "black",
            "hispanic",
            "asian",
            "native_american",
            "inter_racial",
        ]:
            test_key = f"replace_to_{ethnicity}_lastnames"
            if test_key in self.tests:
                self.tests[test_key]["parameters"] = {}
                self.tests[test_key]["parameters"] = {
                    "names_to_substitute": sum(
                        [
                            ethnicity_last_names[e]
                            for e in ethnicity_last_names
                            if e != ethnicity
                        ],
                        [],
                    ),
                    "chosen_ethnicity_names": ethnicity_last_names[ethnicity],
                }

    def transform(self) -> List[Sample]:
        """
        Runs the bias test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]
                A list of `Sample` objects representing the resulting dataset after running the bias test.
        """
        all_samples = []
        no_transformation_applied_tests = {}
        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]

            transformed_samples = self.supported_tests[test_name].transform(
                data_handler_copy, **params.get("parameters", {})
            )

            new_transformed_samples, removed_samples_tests = filter_unique_samples(
                TestFactory.task, transformed_samples, test_name
            )
            all_samples.extend(new_transformed_samples)

            no_transformation_applied_tests.update(removed_samples_tests)

        if no_transformation_applied_tests:
            warning_message = Warnings._W009
            for test, count in no_transformation_applied_tests.items():
                warning_message += Warnings._W010.format(
                    test=test, count=count, total_sample=len(self._data_handler)
                )

            logging.warning(warning_message)

        return all_samples

    @staticmethod
    def available_tests() -> Dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            Dict: A dictionary of test names and classes.

        """

        tests = {
            j: i
            for i in BaseBias.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class RepresentationTestFactory(ITests):
    """
    A class for performing representation tests on a given dataset.
    """

    alias_name = "representation"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """
        Runs the representation test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the representation test.
        """
        all_samples = []

        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]

            transformed_samples = self.supported_tests[test_name].transform(
                test_name, data_handler_copy, params
            )

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
            j: i
            for i in BaseRepresentation.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class FairnessTestFactory(ITests):
    """
    A class for performing fairness tests on a given dataset.
    """

    alias_name = "fairness"
    model_result = None

    def __init__(self, data_handler: List[Sample], tests: Dict, **kwargs) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """
        Runs the fairness test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the fairness test.
        """
        all_samples = []

        if self._data_handler[0].expected_results is None:
            raise RuntimeError(Errors.E052(var="fairness"))

        for test_name, params in self.tests.items():
            transformed_samples = self.supported_tests[test_name].transform(
                test_name, None, params
            )

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
            j: i
            for i in BaseFairness.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests

    @classmethod
    def run(
        cls,
        sample_list: Dict[str, List[Sample]],
        model: ModelAPI,
        raw_data: List[Sample],
        **kwargs,
    ):
        """
        Runs the fairness tests on the given model and dataset.

        Args:
            sample_list (Dict[str, List[Sample]]): A dictionary of test names and corresponding `Sample` objects.
            model (ModelAPI): The model to be tested.
            raw_data (List[Sample]): The raw dataset.

        """
        raw_data_copy = [x.copy() for x in raw_data]
        grouped_label = {}
        grouped_data = cls.get_gendered_data(raw_data_copy)
        for gender, data in grouped_data.items():
            if len(data) == 0:
                grouped_label[gender] = [[], []]
            else:
                if isinstance(data[0], NERSample):

                    def predict_ner(sample: Sample):
                        prediction = model.predict(sample.original)
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    X_test = pd.Series(data)
                    X_test.apply(predict_ner)
                    y_true = pd.Series(data).apply(
                        lambda x: [y.entity for y in x.expected_results.predictions]
                    )
                    y_pred = pd.Series(data).apply(
                        lambda x: [y.entity for y in x.actual_results.predictions]
                    )
                    valid_indices = y_true.apply(len) == y_pred.apply(len)
                    y_true = y_true[valid_indices]
                    y_pred = y_pred[valid_indices]
                    y_true = y_true.explode()
                    y_pred = y_pred.explode()
                    y_pred = y_pred.apply(lambda x: x.split("-")[-1]).reset_index(
                        drop=True
                    )
                    y_true = y_true.apply(lambda x: x.split("-")[-1]).reset_index(
                        drop=True
                    )

                elif isinstance(data[0], SequenceClassificationSample):

                    def predict_text_classification(sample: Sample):
                        prediction = model.predict(sample.original)
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    X_test = pd.Series(data)
                    X_test.apply(predict_text_classification)

                    y_true = pd.Series(data).apply(
                        lambda x: [y.label for y in x.expected_results.predictions]
                    )
                    y_pred = pd.Series(data).apply(
                        lambda x: [y.label for y in x.actual_results.predictions]
                    )
                    y_true = y_true.apply(lambda x: x[0])
                    y_pred = y_pred.apply(lambda x: x[0])

                    y_true = y_true.explode()
                    y_pred = y_pred.explode()

                elif data[0].task == "question-answering":
                    from ..utils.custom_types.helpers import (
                        build_qa_input,
                        build_qa_prompt,
                    )

                    if data[0].dataset_name is None:
                        dataset_name = "default_question_answering_prompt"
                    else:
                        dataset_name = data[0].dataset_name.split("-")[0].lower()

                    if data[0].expected_results is None:
                        raise RuntimeError(Errors.E053(dataset_name=dataset_name))

                    def predict_question_answering(sample: Sample):
                        input_data = build_qa_input(
                            context=sample.original_context,
                            question=sample.original_question,
                            options=sample.options,
                        )
                        prompt = build_qa_prompt(input_data, dataset_name, **kwargs)
                        server_prompt = kwargs.get("server_prompt", " ")
                        prediction = model(
                            text=input_data, prompt=prompt, server_prompt=server_prompt
                        ).strip()
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    y_true = pd.Series(data).apply(lambda x: x.expected_results)
                    X_test = pd.Series(data)

                    y_pred = X_test.apply(predict_question_answering)
                elif data[0].task == "summarization":
                    if data[0].dataset_name is None:
                        dataset_name = "default_summarization_prompt"
                    else:
                        dataset_name = data[0].dataset_name.split("-")[0].lower()
                    prompt_template = kwargs.get(
                        "user_prompt", default_user_prompt.get(dataset_name, "")
                    )
                    if data[0].expected_results is None:
                        raise RuntimeError(Errors.E053(dataset_name=dataset_name))

                    def predict_summarization(sample: Sample):
                        prediction = model(
                            text={"context": sample.original},
                            prompt={
                                "template": prompt_template,
                                "input_variables": ["context"],
                            },
                        ).strip()
                        sample.actual_results = prediction
                        sample.gender = gender
                        return prediction

                    y_true = pd.Series(data).apply(lambda x: x.expected_results)
                    X_test = pd.Series(data)
                    y_pred = X_test.apply(predict_summarization)

                if kwargs["is_default"]:
                    y_pred = y_pred.apply(
                        lambda x: "1"
                        if x in ["pos", "LABEL_1", "POS"]
                        else ("0" if x in ["neg", "LABEL_0", "NEG"] else x)
                    )

                grouped_label[gender] = [y_true, y_pred]

        supported_tests = cls.available_tests()
        from ..utils.custom_types.helpers import TestResultManager

        cls.model_result = TestResultManager().prepare_model_response(raw_data_copy)
        kwargs["task"] = raw_data[0].task
        tasks = []
        for test_name, samples in sample_list.items():
            tasks.append(
                supported_tests[test_name].async_run(
                    samples, grouped_label, grouped_data=grouped_data, **kwargs
                )
            )
        return tasks

    @staticmethod
    def get_gendered_data(data: List[Sample]) -> Dict[str, List[Sample]]:
        """Split list of samples into gendered lists."""
        from langtest.utils.gender_classifier import GenderClassifier

        classifier = GenderClassifier()

        data = pd.Series(data)
        if isinstance(data[0], QASample):
            sentences = data.apply(
                lambda x: f"{x.original_context} {x.original_question}"
            )
        else:
            sentences = data.apply(lambda x: x.original)

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
    model_result = None

    def __init__(self, data_handler: List[Sample], tests: Dict, **kwargs) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """
        Runs the accuracy test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the accuracy test.
        """
        all_samples = []

        if self._data_handler[0].expected_results is None:
            raise RuntimeError(Errors.E052(var="accuracy"))

        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]

            if data_handler_copy[0].task == "ner":
                y_true = pd.Series(data_handler_copy).apply(
                    lambda x: [y.entity for y in x.expected_results.predictions]
                )
                y_true = y_true.explode().apply(
                    lambda x: x.split("-")[-1] if isinstance(x, str) else x
                )
            elif data_handler_copy[0].task == "text-classification":
                y_true = (
                    pd.Series(data_handler_copy)
                    .apply(lambda x: [y.label for y in x.expected_results.predictions])
                    .explode()
                )
            elif (
                data_handler_copy[0].task == "question-answering"
                or data_handler_copy[0].task == "summarization"
            ):
                y_true = (
                    pd.Series(data_handler_copy)
                    .apply(lambda x: x.expected_results)
                    .explode()
                )

            y_true = y_true.dropna()
            transformed_samples = self.supported_tests[test_name].transform(
                test_name, y_true, params
            )

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
            j: i
            for i in BaseAccuracy.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests

    @classmethod
    def run(
        cls,
        sample_list: Dict[str, List[Sample]],
        model: ModelAPI,
        raw_data: List[Sample],
        **kwargs,
    ):
        """
        Runs the accuracy tests on the given model and dataset.

        Args:
            sample_list (Dict[str, List[Sample]]): A dictionary of test names and corresponding `Sample` objects.
            model (ModelAPI): The model to be tested.
            raw_data (List[Sample]): The raw dataset.

        """
        raw_data_copy = [x.copy() for x in raw_data]

        if isinstance(raw_data_copy[0], NERSample):

            def predict_ner(sample):
                prediction = model.predict(sample.original)
                sample.actual_results = prediction
                return prediction

            X_test = pd.Series(raw_data_copy)
            X_test.apply(predict_ner)
            y_true = pd.Series(raw_data_copy).apply(
                lambda x: [y.entity for y in x.expected_results.predictions]
            )
            y_pred = pd.Series(raw_data_copy).apply(
                lambda x: [y.entity for y in x.actual_results.predictions]
            )
            valid_indices = y_true.apply(len) == y_pred.apply(len)
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            y_true = y_true.explode()
            y_pred = y_pred.explode()
            y_pred = y_pred.apply(lambda x: x.split("-")[-1])
            y_true = y_true.apply(lambda x: x.split("-")[-1])

        elif isinstance(raw_data_copy[0], SequenceClassificationSample):

            def predict_text_classification(sample):
                prediction = model.predict(sample.original)
                sample.actual_results = prediction
                return prediction

            X_test = pd.Series(raw_data_copy)
            X_test.apply(predict_text_classification)

            y_true = pd.Series(raw_data_copy).apply(
                lambda x: [y.label for y in x.expected_results.predictions]
            )
            y_pred = pd.Series(raw_data_copy).apply(
                lambda x: [y.label for y in x.actual_results.predictions]
            )
            y_true = y_true.apply(lambda x: x[0])
            y_pred = y_pred.apply(lambda x: x[0])

            y_true = y_true.explode()
            y_pred = y_pred.explode()

        elif raw_data_copy[0].task == "question-answering":
            from ..utils.custom_types.helpers import build_qa_input, build_qa_prompt

            if raw_data_copy[0].dataset_name is None:
                dataset_name = "default_question_answering_prompt"
            else:
                dataset_name = raw_data_copy[0].dataset_name.split("-")[0].lower()

            def predict_question_answering(sample):
                input_data = build_qa_input(
                    context=sample.original_context,
                    question=sample.original_question,
                    options=sample.options,
                )
                prompt = build_qa_prompt(input_data, dataset_name, **kwargs)
                server_prompt = kwargs.get("server_prompt", " ")
                prediction = model(
                    text=input_data, prompt=prompt, server_prompt=server_prompt
                ).strip()
                sample.actual_results = prediction
                return prediction

            y_true = pd.Series(raw_data_copy).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data_copy)

            y_pred = X_test.apply(predict_question_answering)

        elif raw_data_copy[0].task == "summarization":
            if raw_data_copy[0].dataset_name is None:
                dataset_name = "default_summarization_prompt"
            else:
                dataset_name = raw_data_copy[0].dataset_name.split("-")[0].lower()
            prompt_template = kwargs.get(
                "user_prompt", default_user_prompt.get(dataset_name, "")
            )

            def predict_summarization(sample):
                prediction = model(
                    text={"context": sample.original},
                    prompt={"template": prompt_template, "input_variables": ["context"]},
                ).strip()
                sample.actual_results = prediction
                return prediction

            y_true = pd.Series(raw_data_copy).apply(lambda x: x.expected_results)
            X_test = pd.Series(raw_data_copy)
            y_pred = X_test.apply(predict_summarization)

        if kwargs["is_default"]:
            y_pred = y_pred.apply(
                lambda x: "1"
                if x in ["pos", "LABEL_1", "POS"]
                else ("0" if x in ["neg", "LABEL_0", "NEG"] else x)
            )

        supported_tests = cls.available_tests()

        tasks = []

        from ..utils.custom_types.helpers import TestResultManager

        cls.model_result = TestResultManager().prepare_model_response(raw_data_copy)

        for test_name, samples in sample_list.items():
            tasks.append(
                supported_tests[test_name].async_run(
                    samples, y_true, y_pred, X_test=X_test, **kwargs
                )
            )
        return tasks


class ToxicityTestFactory(ITests):
    """
    A class for performing toxicity tests on a given dataset.
    """

    alias_name = "toxicity"

    def __init__(self, data_handler: List[Sample], tests: Dict, **kwargs) -> None:
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """
        Runs the toxicity test and returns the resulting `Sample` objects.

        Returns:
            List[Sample]:
                A list of `Sample` objects representing the resulting dataset after running the toxicity test.
        """
        all_samples = []

        for test_name, params in self.tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform
            transformed_samples = test_func(
                data_handler_copy, test_name=test_name, **params.get("parameters", {})
            )

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
            j: i
            for i in BaseToxicity.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class PerformanceTestFactory(ITests):
    """Factory class for the model performance

    This class implements the model performance The robustness measure is the number of test cases that the model fails to run on.

    """

    alias_name = "performance"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the model performance"""

        self.supported_tests = self.available_tests()
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Transforms the sample data based on the implemented tests measure.

        Args:
            sample (Sample): The input data to be transformed.
            **kwargs: Additional arguments to be passed to the tests measure.

        Returns:
            Sample: The transformed data based on the implemented
            tests measure.

        """
        all_samples = []
        for test_name, params in self.tests.items():
            transformed_samples = self.supported_tests[test_name].transform(
                params=params, **self.kwargs
            )
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the model performance

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the model performance

        Returns:
            List[Sample]: The transformed data based on the implemented model performance

        """
        supported_tests = cls.available_tests()
        tasks = []
        for test_name, samples in sample_list.items():
            out = await supported_tests[test_name].async_run(samples, model, **kwargs)
            if isinstance(out, list):
                tasks.extend(out)
            else:
                tasks.append(out)

        return tasks

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the available model performance

        Returns:
            Dict[str, str]: The available model performance

        """
        tests = {
            j: i
            for i in BasePerformance.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class SecurityTestFactory(ITests):

    """Factory class for the security tests"""

    alias_name = "security"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        self.supported_tests = self.available_tests()
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        all_samples = []
        for test_name, params in self.tests.items():
            transformed_samples = self.supported_tests[test_name].transform(
                self.data_handler, **self.kwargs
            )
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        supported_tests = cls.available_tests()
        tasks = []
        for test_name, samples in sample_list.items():
            out = await supported_tests[test_name].async_run(samples, model, **kwargs)
            if isinstance(out, list):
                tasks.extend(out)
            else:
                tasks.append(out)

        return tasks

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        tests = {
            j: i
            for i in BaseSecurity.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class ClinicalTestFactory(ITests):
    """Factory class for the clinical tests"""

    alias_name = "clinical"
    supported_tasks = [
        "clinical",
        "text-generation",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the ClinicalTestFactory"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        for sample in self.data_handler:
            sample.test_type = "demographic-bias"
            sample.category = "clinical"
        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the clinical tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the clinical tests

        Returns:
            List[Sample]: The transformed data based on the implemented clinical tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no clinical tests

        Returns:
            Dict[str, str]: Empty dict, no clinical tests
        """
        return {"demographic-bias": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the clinical tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the clinical tests

        Returns:
            List[Sample]: The transformed data based on the implemented clinical tests@

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["demographic-bias"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["demographic-bias"]


class DisinformationTestFactory(ITests):
    """Factory class for disinformation test"""

    alias_name = "disinformation"
    supported_tasks = [
        "disinformation",
        "text-generation",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        for sample in self.data_handler:
            sample.test_type = "narrative_wedging"
            sample.category = "disinformation"

        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        return {"narrative_wedging": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["narrative_wedging"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["narrative_wedging"]


class IdeologyTestFactory(ITests):
    """Factory class for the ideology tests"""

    alias_name = "ideology"
    supported_tasks = ["question_answering", "summarization"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the clinical tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs
        self.supported_tests = self.available_tests()

    def transform(self) -> List[Sample]:
        all_samples = []
        for test_name, params in self.tests.items():
            transformed_samples = self.supported_tests[test_name].transform(
                self.data_handler, **self.kwargs
            )
            all_samples.extend(transformed_samples)
        return all_samples

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the model performance

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the model performance

        Returns:
            List[Sample]: The transformed data based on the implemented model performance

        """
        supported_tests = cls.available_tests()
        tasks = []
        for test_name, samples in sample_list.items():
            out = await supported_tests[test_name].async_run(samples, model, **kwargs)
            if isinstance(out, list):
                tasks.extend(out)
            else:
                tasks.append(out)
        return tasks

    @staticmethod
    def available_tests() -> Dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            Dict: A dictionary of test names and classes.

        """

        tests = {
            j: i
            for i in BaseIdeology.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class SensitivityTestFactory(ITests):
    """A class for performing Sensitivity tests on a given dataset.

    This class provides functionality to perform sensitivity tests on a given dataset
    using various test configurations.

    Attributes:
        alias_name (str): A string representing the alias name for this test factory.

    """

    alias_name = "sensitivity"

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initialize a new SensitivityTestFactory instance.

        Args:
            data_handler (List[Sample]): A list of `Sample` objects representing the input dataset.
            tests (Optional[Dict]): A dictionary of test names and corresponding parameters (default is None).
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the `tests` argument is not a dictionary.

        """

        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """Run the sensitivity test and return the resulting `Sample` objects.

        Returns:
            List[Sample]: A list of `Sample` objects representing the resulting dataset after running the sensitivity test.

        """
        all_samples = []
        no_transformation_applied_tests = {}
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            if TestFactory.is_augment:
                data_handler_copy = [x.copy() for x in self._data_handler]
            else:
                data_handler_copy = [x.copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform

            _ = [
                sample.transform(
                    test_func,
                    params.get("parameters", {}),
                )
                if hasattr(sample, "transform")
                else sample
                for sample in data_handler_copy
            ]
            transformed_samples = data_handler_copy

            new_transformed_samples, removed_samples_tests = filter_unique_samples(
                TestFactory.task.category, transformed_samples, test_name
            )
            all_samples.extend(new_transformed_samples)

            no_transformation_applied_tests.update(removed_samples_tests)

        if no_transformation_applied_tests:
            warning_message = Warnings._W009
            for test, count in no_transformation_applied_tests.items():
                warning_message += Warnings._W010.format(
                    test=test, count=count, total_sample=len(self._data_handler)
                )

            logging.warning(warning_message)

        return all_samples

    @staticmethod
    def available_tests() -> dict:
        """
        Get a dictionary of all available tests, with their names as keys and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """
        tests = {
            j: i
            for i in BaseSensitivity.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


class StereoTypeFactory(ITests):
    """Factory class for the crows-pairs or wino-bias tests"""

    alias_name = "stereotype"
    supported_tasks = [
        "wino-bias",
        "crows-pairs",
        "fill-mask",
        "question-answering",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the crows-pairs or wino-bias tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Testcases List

        """
        for sample in self.data_handler:
            if sample.test_type == "crows-pairs":
                if "diff_threshold" in self.tests["crows-pairs"].keys():
                    sample.diff_threshold = self.tests["crows-pairs"]["diff_threshold"]
                if "filter_threshold" in self.tests["crows-pairs"].keys():
                    sample.filter_threshold = self.tests["crows-pairs"][
                        "filter_threshold"
                    ]
            else:
                if "diff_threshold" in self.tests["wino-bias"].keys():
                    sample.diff_threshold = self.tests["wino-bias"]["diff_threshold"]

        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the crows-pairs or wino-bias  tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the crows-pairs or wino-bias tests

        Returns:
            List[Sample]: The transformed data based on the implemented crows-pairs or wino-bias tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no crows-pairs or wino-bias tests

        Returns:
            Dict[str, str]: Empty dict, no crows-pairs or wino-bias tests
        """

        return {"crows-pairs": cls, "wino-bias": cls}

    @staticmethod
    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the crows-pairs or wino-bias tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the crows-pairs or wino-bias tests

        Returns:
            List[Sample]: The transformed data based on the implemented crows-pairs or wino-biastests

        """
        progress = kwargs.get("progress_bar", False)
        for key, value in sample_list.items():
            if key == "crows-pairs":
                for sample in value:
                    if sample.state != "done":
                        if hasattr(sample, "run"):
                            sample_status = sample.run(model, **kwargs)
                            if sample_status:
                                sample.state = "done"
                    if progress:
                        progress.update(1)

                sample_list["crows-pairs"] = [
                    x
                    for x in sample_list["crows-pairs"]
                    if (
                        x.mask1_score > x.filter_threshold
                        or x.mask2_score > x.filter_threshold
                    )
                ]
                return sample_list["crows-pairs"]
        else:
            for sample in value:
                if sample.state != "done":
                    if hasattr(sample, "run"):
                        sample_status = sample.run(model, **kwargs)
                        if sample_status:
                            sample.state = "done"
                if progress:
                    progress.update(1)

            return sample_list["wino-bias"]


class StereoSetTestFactory(ITests):
    """Factory class for the StereoSet tests"""

    alias_name = "stereoset"
    supported_tasks = [
        "stereoset",
        "question-answering",
    ]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the stereoset tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            List of testcase

        """
        for s in self.data_handler:
            s.diff_threshold = (
                self.tests[s.test_type]["diff_threshold"]
                if "diff_threshold" in self.tests[s.test_type]
                else s.diff_threshold
            )
        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the stereoset tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the stereoset tests

        Returns:
            List[Sample]: The transformed data based on the implemented stereoset tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no stereoset tests

        Returns:
            Dict[str, str]: Empty dict, no stereoset tests
        """
        return {"intrasentence": cls, "intersentence": cls}

    @staticmethod
    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the StereoSet tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the StereoSet tests

        Returns:
            List[Sample]: The transformed data based on the implemented crows-pairs tests

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["intersentence"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)

        for sample in sample_list["intrasentence"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)

        return sample_list["intersentence"] + sample_list["intrasentence"]


class LegalTestFactory(ITests):
    """Factory class for the legal"""

    alias_name = "legal"
    supported_tasks = ["legal", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the legal tests"""

        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        for sample in self.data_handler:
            sample.test_type = "legal-support"
            sample.category = "legal"
        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs the legal tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the wino-bias tests

        Returns:
            List[Sample]: The transformed data based on the implemented legal tests

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Returns the empty dict, no legal tests

        Returns:
            Dict[str, str]: Empty dict, no legal tests
        """
        return {"legal-support": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs the legal tests

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the legal tests

        Returns:
            List[Sample]: The transformed data based on the implemented legal tests

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["legal-support"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["legal-support"]


class FactualityTestFactory(ITests):
    """Factory class for factuality test"""

    alias_name = "factuality"
    supported_tasks = ["factuality", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initializes the FactualityTestFactory"""
        self.data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

    def transform(self) -> List[Sample]:
        """Nothing to use transform for no longer to generating testcases.

        Returns:
            Empty list

        """
        for sample in self.data_handler:
            sample.test_type = "order_bias"
            sample.category = "factuality"

        return self.data_handler

    @classmethod
    async def run(
        cls, sample_list: List[Sample], model: ModelAPI, **kwargs
    ) -> List[Sample]:
        """Runs factuality tests

        Args:
            sample_list (list[Sample]): A list of Sample objects to be tested.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Sample]: A list of Sample objects with test results.

        """
        task = asyncio.create_task(cls.async_run(sample_list, model, **kwargs))
        return task

    @classmethod
    def available_tests(cls) -> Dict[str, str]:
        """Retrieves available factuality test types.

        Returns:
            dict: A dictionary mapping test names to their corresponding classes.

        """
        return {"order_bias": cls}

    async def async_run(sample_list: List[Sample], model: ModelAPI, *args, **kwargs):
        """Runs factuality tests

        Args:
            sample_list (list[Sample]): A list of Sample objects to be tested.
            model (ModelAPI): The model to be used for testing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            list[Sample]: A list of Sample objects with test results.

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list["order_bias"]:
            if sample.state != "done":
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list["order_bias"]


class SycophancyTestFactory(ITests):
    """A class for conducting Sycophancy tests on a given dataset.

    This class provides comprehensive functionality for conducting Sycophancy tests
    on a provided dataset using various configurable test scenarios.

    Attributes:
        alias_name (str): A string representing the alias name for this test factory.

    """

    alias_name = "Sycophancy"
    supported_tasks = ["Sycophancy", "question-answering"]

    def __init__(self, data_handler: List[Sample], tests: Dict = None, **kwargs) -> None:
        """Initialize a new SycophancyTestFactory instance.

        Args:
            data_handler (List[Sample]): A list of `Sample` objects representing the input dataset.
            tests (Optional[Dict]): A dictionary of test names and corresponding parameters (default is None).
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the `tests` argument is not a dictionary.

        """
        self.supported_tests = self.available_tests()
        self._data_handler = data_handler
        self.tests = tests
        self.kwargs = kwargs

        if not isinstance(self.tests, dict):
            raise ValueError(Errors.E048())

        if len(self.tests) == 0:
            self.tests = self.supported_tests

        not_supported_tests = set(self.tests) - set(self.supported_tests)
        if len(not_supported_tests) > 0:
            raise ValueError(
                Errors.E049(
                    not_supported_tests=not_supported_tests,
                    supported_tests=list(self.supported_tests.keys()),
                )
            )

    def transform(self) -> List[Sample]:
        """Execute the Sycophancy test and return resulting `Sample` objects.

        Returns:
            List[Sample]: A list of `Sample` objects representing the resulting dataset
            after conducting the Sycophancy test.

        """
        all_samples = []
        tests_copy = self.tests.copy()
        for test_name, params in tests_copy.items():
            if TestFactory.is_augment:
                data_handler_copy = [x.copy() for x in self._data_handler]
            else:
                data_handler_copy = [x.copy() for x in self._data_handler]

            test_func = self.supported_tests[test_name].transform

            _ = [
                sample.transform(
                    test_func,
                    params.get("parameters", {}),
                )
                if hasattr(sample, "transform")
                else sample
                for sample in data_handler_copy
            ]
            transformed_samples = data_handler_copy

            for sample in transformed_samples:
                sample.test_type = test_name
            all_samples.extend(transformed_samples)
        return all_samples

    @staticmethod
    def available_tests() -> dict:
        """
        Retrieve a dictionary of all available tests, with their names as keys
        and their corresponding classes as values.

        Returns:
            dict: A dictionary of test names and classes.
        """
        tests = {
            j: i
            for i in BaseSycophancy.__subclasses__()
            for j in (i.alias_name if isinstance(i.alias_name, list) else [i.alias_name])
        }
        return tests


__all__ = [
    RobustnessTestFactory,
    BiasTestFactory,
    RepresentationTestFactory,
    FairnessTestFactory,
    AccuracyTestFactory,
    ToxicityTestFactory,
    PerformanceTestFactory,
    SecurityTestFactory,
    ClinicalTestFactory,
    DisinformationTestFactory,
    IdeologyTestFactory,
    SensitivityTestFactory,
    StereoTypeFactory,
    StereoSetTestFactory,
    LegalTestFactory,
    FactualityTestFactory,
    SycophancyTestFactory,
    GrammarTestFactory,
]
