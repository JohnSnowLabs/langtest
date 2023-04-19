import asyncio
import random
import re
from abc import ABC, abstractmethod
from typing import List

from nlptest.modelhandler.modelhandler import ModelFactory

from .utils import female_pronouns, male_pronouns, neutral_pronouns
from ..utils.custom_types import Sample, Span, Transformation


class BaseBias(ABC):
    """
    Abstract base class for implementing bias measures.

    Attributes:
        alias_name (str): A name or list of names that identify the bias measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented bias measure.
    """
    alias_name = None

    @abstractmethod
    def transform(self, sample_list: List[Sample], *args, **kwargs) -> List[Sample]:
        """
        Abstract method that implements the bias measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.

        Returns:
            List[Sample]: The transformed data based on the implemented bias measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:
        """
        Abstract method that implements the bias measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the bias measure.

        Returns:
            List[Sample]: The transformed data based on the implemented bias measure.

        """
        progress = kwargs.get("progress_bar", False)
        for sample in sample_list:
            if sample.state != "done":
                sample.expected_results = model(sample.original)
                sample.actual_results = model(sample.test_case)
                sample.state = "done"
            if progress:
                progress.update(1)
        return sample_list

    @classmethod
    async def async_run(cls, sample_list: List[Sample], model: ModelFactory, **kwargs):
        """
        Abstract method that implements the creation of an asyncio task for the bias measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for the bias measure.

        Returns:
            asyncio.Task: The asyncio task for the bias measure.        
        """
        created_task = asyncio.create_task(
            cls.run(sample_list, model, **kwargs))
        return created_task


class GenderPronounBias(BaseBias):
    alias_name = [
        "replace_to_male_pronouns",
        "replace_to_female_pronouns",
        "replace_to_neutral_pronouns"
    ]

    @staticmethod
    def transform(sample_list: List[Sample], pronouns_to_substitute: List[str], pronoun_type: str) -> List[Sample]:
        """Replace pronouns to check the gender bias

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            pronouns_to_substitute (List[str]): list of pronouns that need to be substituted.
            pronoun_type (str): replacing pronoun type string ('male', 'female' or 'neutral')

        Returns:
            List[Sample]: List of sentences with replaced pronouns
        """

        for sample in sample_list:
            sample.category = "bias"
            transformations = []
            replaced_string = sample.original

            tokens_to_substitute = [token for token in sample.original.split(' ') if
                                    token.lower() in pronouns_to_substitute]

            for replace_token in tokens_to_substitute:
                if pronoun_type == "female":
                    combined_dict = {
                        k: male_pronouns[k] + neutral_pronouns[k] for k in male_pronouns.keys()}
                    chosen_dict = female_pronouns
                elif pronoun_type == "male":
                    combined_dict = {
                        k: female_pronouns[k] + neutral_pronouns[k] for k in female_pronouns.keys()}
                    chosen_dict = male_pronouns
                elif pronoun_type == "neutral":
                    combined_dict = {
                        k: female_pronouns[k] + male_pronouns[k] for k in female_pronouns.keys()}
                    chosen_dict = neutral_pronouns

                for key, value in combined_dict.items():
                    if replace_token.lower() in value:
                        type_of_pronoun = str(key)
                        break

                chosen_token = random.choice(chosen_dict[type_of_pronoun])
                regex = r'\b{}\b'.format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1)
                    transformations.append(
                        Transformation(
                            original_span=Span(
                                start=span.start(), end=span.end(), word=replace_token),
                            new_span=Span(start=span.start(), end=span.end(
                            ) + diff_len, word=chosen_token),
                            ignore=False
                        )
                    )

            sample.test_case = replaced_string
            sample.transformations = transformations

        return sample_list


class CountryEconomicBias(BaseBias):
    alias_name = [
        "replace_to_high_income_country",
        "replace_to_low_income_country",
        "replace_to_upper_middle_income_country",
        "replace_to_lower_middle_income_country"
    ]

    @staticmethod
    def transform(sample_list: List[Sample], country_names_to_substitute: List[str], chosen_country_names: List[str]) -> \
            List[Sample]:
        """Replace country names to check the ethnicity bias


        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            country_names_to_substitute (List[str]): list of country names that need to be substituted.
            chosen_country_names (List[str]): list of country names to replace with.

        Returns:
            List[Sample]: List of sentences with replaced names
        """

        for sample in sample_list:
            transformations = []
            replaced_string = sample.original

            tokens_to_substitute = [token for token in sample.original.split(' ') if
                                    token.lower() in [name.lower() for name in country_names_to_substitute]]

            for replace_token in tokens_to_substitute:
                chosen_token = random.choice(chosen_country_names)
                regex = r'\b{}\b'.format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1)
                    transformations.append(
                        Transformation(
                            original_span=Span(
                                start=span.start(), end=span.end(), word=replace_token),
                            new_span=Span(start=span.start(), end=span.end(
                            ) + diff_len, word=chosen_token),
                            ignore=False
                        )
                    )

            sample.test_case = replaced_string
            sample.transformations = transformations
            sample.category = "bias"

        return sample_list


class EthnicityNameBias(BaseBias):
    alias_name = [
        "replace_to_white_firstnames",
        "replace_to_black_firstnames",
        "replace_to_hispanic_firstnames",
        "replace_to_asian_firstnames",
        "replace_to_white_lastnames",
        "replace_to_black_lastnames",
        "replace_to_hispanic_lastnames",
        "replace_to_asian_lastnames",
        "replace_to_native_american_lastnames",
        "replace_to_inter_racial_lastnames"
    ]

    @staticmethod
    def transform(sample_list: List[Sample], names_to_substitute: List[str], chosen_ethnicity_names: List[str]) \
            -> List[Sample]:
        """Replace names to check the ethnicity bias
        Ethnicity Dataset Curated from the United States Census Bureau surveys

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            names_to_substitute (List[str]): list of ethnicity names that need to be substituted.
            chosen_ethnicity_names (List[str]): list of ethnicity names to replace with.

        Returns:
            List[Sample]: List of sentences with replaced names
        """

        for sample in sample_list:
            transformations = []
            replaced_string = sample.original
            tokens_to_substitute = [token for token in sample.original.split(' ') if
                                    any(name.lower() == token.lower() for name in names_to_substitute)]

            for replace_token in tokens_to_substitute:
                chosen_token = random.choice(chosen_ethnicity_names)
                regex = r'\b{}\b'.format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1)
                    transformations.append(
                        Transformation(
                            original_span=Span(
                                start=span.start(), end=span.end(), word=replace_token),
                            new_span=Span(start=span.start(), end=span.end(
                            ) + diff_len, word=chosen_token),
                            ignore=False
                        )
                    )
            sample.test_case = replaced_string
            sample.transformations = transformations
            sample.category = "bias"

        return sample_list


class ReligionBias(BaseBias):
    alias_name = [
        "replace_to_muslim_names",
        "replace_to_hindu_names",
        "replace_to_christian_names",
        "replace_to_sikh_names",
        "replace_to_jain_names",
        "replace_to_parsi_names",
        "replace_to_buddhist_names"
    ]

    @staticmethod
    def transform(sample_list: List[Sample], names_to_substitute: List[str], chosen_names: List[str]) -> List[Sample]:
        """
        Replace  names to check the religion bias

        Args:
            sample_list (List[Sample]): List of sentences to apply perturbation.
            names_to_substitute (List[str]): list of names that need to be substituted.
            chosen_names (List[str]): list of names to replace with.

        Returns:
            List[Sample]: List of sentences with replaced names
        """

        for sample in sample_list:
            transformations = []
            replaced_string = sample.original
            tokens_to_substitute = [token for token in sample.original.split(' ') if
                                    any(name.lower() == token.lower() for name in names_to_substitute)]

            for replace_token in tokens_to_substitute:
                chosen_token = random.choice(chosen_names)
                regex = r'\b{}\b'.format(replace_token)
                diff_len = len(chosen_token) - len(replace_token)
                nb_occurrences = len(re.findall(regex, replaced_string))
                for c in range(nb_occurrences):
                    span = re.search(regex, replaced_string)
                    replaced_string = re.sub(
                        regex, chosen_token, replaced_string, count=1)
                    transformations.append(
                        Transformation(
                            original_span=Span(
                                start=span.start(), end=span.end(), word=replace_token),
                            new_span=Span(start=span.start(), end=span.end(
                            ) + diff_len, word=chosen_token),
                            ignore=False
                        )
                    )

            sample.test_case = replaced_string
            sample.transformations = transformations
            sample.category = "bias"

        return sample_list
