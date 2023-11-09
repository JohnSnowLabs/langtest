from collections import defaultdict
from typing import Dict, List
from typing import Union
import re
import pandas as pd
from ..errors import Errors
from langtest.utils.custom_types import (
    NERPrediction,
    Sample,
    SequenceLabel,
)
from .constants import (
    asian_names,
    black_names,
    country_economic_dict,
    entity_types as default_entity_types,
    hispanic_names,
    inter_racial_names,
    native_american_names,
    religion_wise_names,
    white_names,
    bad_word_list,
)
from .custom_data import add_custom_data


class RepresentationOperation:
    """This class provides operations for analyzing and evaluating different representations in data.

    Methods:
        - add_custom_representation(data, name, append, check):
            Adds custom representation to the given data.
        - get_label_representation_dict(data):
            Retrieves the label representation information from the data.
        - get_country_economic_representation_dict(data):
            Retrieves the country economic representation information from the data.
        - get_religion_name_representation_dict(data):
            Retrieves the religion representation information from the data.
        - get_ethnicity_representation_dict(data):
            Retrieves the ethnicity representation information from the data.
        - get_entity_representation_proportions(entity_representation):
            Calculates the proportions of each entity in the representation.
    Attributes:
        - entity_types: A list of default entity types.
    """

    entity_types = default_entity_types.copy()

    @staticmethod
    def add_custom_representation(
        data: Union[list, dict], name: str, append: bool, check: str
    ) -> None:
        """Add custom representation to the given data.

        Args:
            data (Union[list, dict]): The data to which the custom representation will be added.
            name (str): The name of the custom representation.
            append (bool): Indicates whether to append the custom representation or replace the existing representation.
            check (str): The check parameter is used for 'Label-Representation' because it is only supported for NER.

        Returns:
            None
        """
        if name != "Label-Representation":
            add_custom_data(data, name, append)
        else:
            if not isinstance(data, list):
                raise ValueError(Errors.E068)

            if check != "ner":
                raise ValueError(Errors.E069)

            if append:
                RepresentationOperation.entity_types = list(
                    set(RepresentationOperation.entity_types) | set(data)
                )
            else:
                RepresentationOperation.entity_types = data

    @staticmethod
    def get_label_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the label representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            dict: a dictionary containing label representation information.
        """
        label_representation = defaultdict(int)
        for sample in data:
            for prediction in sample.expected_results.predictions:
                if isinstance(prediction, SequenceLabel):
                    label_representation[prediction.label] += 1
                elif isinstance(prediction, NERPrediction):
                    if prediction.entity == "O":
                        label_representation[prediction.entity] += 1
                    elif prediction.entity in RepresentationOperation.entity_types:
                        label_representation[prediction.entity.split("-")[1]] += 1

        return label_representation

    @staticmethod
    def get_country_economic_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the country economic representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing country economic representation information.
        """
        country_economic_representation = {
            "high_income": 0,
            "low_income": 0,
            "lower_middle_income": 0,
            "upper_middle_income": 0,
        }

        income_mapping = {
            "High-income": "high_income",
            "Lower-middle-income": "low_income",
            "Low-income": "low_income",
            "Upper-middle-income": "upper_middle_income",
        }

        for sample in data:
            if sample.task == "ner":
                words = [x.span.word.lower() for x in sample.expected_results.predictions]
            elif sample.task == "text-classification":
                words = set(sample.original.replace(".", "").lower().split())
            elif sample.task == "question-answering":
                if "perturbed_context" in sample.__annotations__:
                    words = set(sample.original_context.replace(".", "").lower().split())
                else:
                    words = set(sample.original_question.replace(".", "").lower().split())
            elif sample.task == "summarization":
                words = set(sample.original.replace(".", "").lower().split())
            else:
                raise ValueError(Errors.E070.format(var=sample.task))

            for income, countries in country_economic_dict.items():
                for country in countries:
                    country_words = set(country.lower().split())
                    if country_words.issubset(words):
                        country_economic_representation[income_mapping[income]] += 1

        return country_economic_representation

    @staticmethod
    def get_religion_name_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the religion representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing religion representation information.
        """
        religion_representation = {
            "muslim": 0,
            "hindu": 0,
            "sikh": 0,
            "christian": 0,
            "jain": 0,
            "buddhist": 0,
            "parsi": 0,
        }
        religions = [religion.capitalize() for religion in religion_representation]

        for sample in data:
            if sample.task == "ner":
                words = [x.span.word for x in sample.expected_results.predictions]
            elif sample.task == "text-classification":
                words = sample.original.split()
            elif sample.task == "question-answering":
                if "perturbed_context" in sample.__annotations__:
                    words = sample.original_context.split()
                else:
                    words = sample.original_question.split()
            elif sample.task == "summarization":
                words = sample.original.split()
            else:
                raise ValueError(Errors.E070.format(var=sample.task))

            for word in words:
                for religion in religions:
                    if check_name(word, [religion_wise_names[religion]]):
                        religion_representation[religion.lower()] += 1

        return religion_representation

    @staticmethod
    def get_ethnicity_representation_dict(data: List[Sample]) -> Dict[str, int]:
        """Retrieves the ethnicity representation information from the data.

        Args:
            data (List[Sample]): The input data to be evaluated for representation test.

        Returns:
            Dict[str, int]: a dictionary containing ethnicity representation information.
        """
        ethnicity_representation = {
            "black": 0,
            "asian": 0,
            "white": 0,
            "native_american": 0,
            "hispanic": 0,
            "inter_racial": 0,
        }

        for sample in data:
            if sample.task == "ner":
                words = [x.span.word for x in sample.expected_results.predictions]
            elif sample.task == "text-classification":
                words = sample.original.split()
            elif sample.task == "question-answering":
                if "perturbed_context" in sample.__annotations__:
                    words = sample.original_context.split()
                else:
                    words = sample.original_question.split()
            elif sample.task == "summarization":
                words = sample.original.split()
            else:
                raise ValueError(Errors.E070.format(var=sample.task))

            for word in words:
                if check_name(
                    word, [white_names["first_names"], white_names["last_names"]]
                ):
                    ethnicity_representation["white"] += 1
                if check_name(
                    word, [black_names["first_names"], black_names["last_names"]]
                ):
                    ethnicity_representation["black"] += 1
                if check_name(
                    word, [hispanic_names["first_names"], hispanic_names["last_names"]]
                ):
                    ethnicity_representation["hispanic"] += 1
                if check_name(
                    word, [asian_names["first_names"], asian_names["last_names"]]
                ):
                    ethnicity_representation["asian"] += 1
                if check_name(word, [inter_racial_names["last_names"]]):
                    ethnicity_representation["inter_racial"] += 1
                if check_name(word, [native_american_names["last_names"]]):
                    ethnicity_representation["native_american"] += 1

        return ethnicity_representation

    @staticmethod
    def get_entity_representation_proportions(
        entity_representation: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculates the proportions of each entity in the representation.

        Args:
            entity_representation (dict): a dictionary containing representation information.

        Returns:
            Dict[str, float]: a dictionary with proportions of each entity.
        """
        total_entities = sum(entity_representation.values())
        entity_representation_proportion = {}
        for entity, count in entity_representation.items():
            if total_entities == 0:
                entity_representation_proportion[entity] = 0
            else:
                entity_representation_proportion[entity] = count / total_entities

        return entity_representation_proportion


def get_substitution_names(values_list: List[List[str]]) -> List[str]:
    """Helper function to get list of substitution names

    Args:
         values_list (List[List[str]]):
            list of substitution lists.

    Returns:
         List[str]:
            List of substitution names
    """
    substitution_names = []
    for lst in values_list:
        substitution_names.extend(lst)

    return substitution_names


def create_terminology(ner_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Iterate over the DataFrame to create terminology from the predictions. IOB format converted to the IO.

    Args:
        ner_data: Pandas DataFrame that has 2 column, 'text' as string and 'label' as list of labels

    Returns:
        Dictionary of entities and corresponding list of words.
    """
    terminology = {}

    chunk = list()
    ent_type = None
    for i, row in ner_data.iterrows():
        sent_labels = row.label
        for token_indx, label in enumerate(sent_labels):
            try:
                if label.startswith("B"):
                    if chunk:
                        if terminology.get(ent_type, None):
                            terminology[ent_type].append(" ".join(chunk))
                        else:
                            terminology[ent_type] = [" ".join(chunk)]

                    sent_tokens = row.text.split(" ")
                    chunk = [sent_tokens[token_indx]]
                    ent_type = label[2:]

                elif label.startswith("I"):
                    sent_tokens = row.text.split(" ")
                    chunk.append(sent_tokens[token_indx])

                else:
                    if chunk:
                        if terminology.get(ent_type, None):
                            terminology[ent_type].append(" ".join(chunk))
                        else:
                            terminology[ent_type] = [" ".join(chunk)]

                    chunk = None
                    ent_type = None

            except AttributeError:
                continue

    return terminology


def check_name(word: str, name_lists: List[List[str]]) -> bool:
    """
    Checks if a word is in a list of list of strings

    Args:
        word (str):
            string to look for
        name_lists (List[List[str]]):
            list of lists of potential candidates
    """
    return any(
        word.lower() in [name.lower() for name in name_list] for name_list in name_lists
    )


def filter_unique_samples(task: str, transformed_samples: list, test_name: str):
    """
    Filter and remove samples with no applied transformations from the list of transformed_samples.

    Args:
        task (str): The type of task.
        transformed_samples (list): List of transformed samples to be filtered.
        test_name (str): Name of the test.

    Returns:
        new_transformed_samples (list): List of filtered samples with unique transformations.
        no_transformation_applied_tests (dict): A dictionary where keys are test names and
            values are the number of samples removed from each test.
    """
    no_transformation_applied_tests = {}
    new_transformed_samples = []
    if task == "question-answering":
        for sample in transformed_samples:
            if (
                sample.original_question.replace(" ", "")
                != sample.perturbed_question.replace(" ", "")
            ) or (
                sample.original_context.replace(" ", "")
                != sample.perturbed_context.replace(" ", "")
            ):
                if test_name != "multiple_perturbations":
                    sample.test_type = test_name
                new_transformed_samples.append(sample)
            else:
                if test_name == "multiple_perturbations":
                    if sample.test_type in no_transformation_applied_tests:
                        no_transformation_applied_tests[sample.test_type] += 1
                    else:
                        no_transformation_applied_tests[sample.test_type] = 1
                else:
                    if test_name in no_transformation_applied_tests:
                        no_transformation_applied_tests[test_name] += 1
                    else:
                        no_transformation_applied_tests[test_name] = 1
    else:
        for sample in transformed_samples:
            if sample.original.replace(" ", "") != sample.test_case.replace(" ", ""):
                if test_name != "multiple_perturbations":
                    sample.test_type = test_name
                new_transformed_samples.append(sample)
            else:
                if test_name == "multiple_perturbations":
                    if sample.test_type in no_transformation_applied_tests:
                        no_transformation_applied_tests[sample.test_type] += 1
                    else:
                        no_transformation_applied_tests[sample.test_type] = 1
                else:
                    if test_name in no_transformation_applied_tests:
                        no_transformation_applied_tests[test_name] += 1
                    else:
                        no_transformation_applied_tests[test_name] = 1

    return new_transformed_samples, no_transformation_applied_tests


def compare_generations_overlap(phrase) -> int:
    """
    Compare a given phrase against a list of offensive words and count the number of offensive words present in the phrase.

    Parameters:
        phrase (str): The input phrase to be checked for offensive words or the middle finger emoji.

    Returns:
        int: The count of offensive words or the middle finger emoji found in the phrase.
    """

    sorted_bad_word_list = sorted(bad_word_list, key=len, reverse=True)
    pattern = (
        r"\b(?:" + "|".join(re.escape(word) for word in sorted_bad_word_list) + r")\b|🖕"
    )
    matches = re.findall(pattern, phrase, flags=re.UNICODE | re.IGNORECASE)
    count = len(matches)

    return count
