import random
import re
import logging

from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import nltk

from .utils import (A2B_DICT, CONTRACTION_MAP, DEFAULT_PERTURBATIONS, PERTURB_CLASS_MAP, TYPO_FREQUENCY,
                    create_terminology, male_pronouns, female_pronouns, neutral_pronouns)
from ..utils.custom_types import Sample, Span, Transformation


class BasePerturbation(ABC):

    @staticmethod
    @abstractmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        return NotImplementedError()


class PerturbationFactory:
    """"""

    def __init__(
            self,
            data_handler: List[Sample],
            tests=None
    ) -> None:

        if tests is []:
            tests = DEFAULT_PERTURBATIONS

        self._tests = dict()
        for test in tests:

            if isinstance(test, str):
                if test not in DEFAULT_PERTURBATIONS:
                    raise ValueError(
                        f'Invalid test specification: {test}. Available tests are: {DEFAULT_PERTURBATIONS}')
                self._tests[test] = dict()
            elif isinstance(test, dict):
                test_name = list(test.keys())[0]
                if test_name not in DEFAULT_PERTURBATIONS:
                    raise ValueError(
                        f'Invalid test specification: {test_name}. Available tests are: {DEFAULT_PERTURBATIONS}')
                self._tests[test_name] = reduce(
                    lambda x, y: {**x, **y}, test[test_name])
            else:
                raise ValueError(
                    '''Invalid test configuration! Tests can be
                    [1] test name as string or
                    [2] dictionary of test name and corresponding parameters.'''
                )

        if 'swap_entities' in self._tests:
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                               'label': [[i.entity for i in sample.expected_results.predictions]
                                         for sample in data_handler]})
            self._tests['swap_entities']['terminology'] = create_terminology(
                df)
            self._tests['swap_entities']['labels'] = df.label.tolist()

        if "american_to_british" in self._tests:
            self._tests['american_to_british']['accent_map'] = A2B_DICT

        if "british_to_american" in self._tests:
            self._tests['british_to_american']['accent_map'] = {
                v: k for k, v in A2B_DICT.items()}

        if 'swap_cohyponyms' in self._tests:
            nltk.download('omw-1.4', quiet=True)
            nltk.download('wordnet', quiet=True)
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                               'label': [[i.entity for i in sample.expected_results.predictions]
                                         for sample in data_handler]})
            self._tests['swap_cohyponyms']['labels'] = df.label.tolist()

        if 'replace_to_male_pronouns' in self._tests:
            neutral_pronouns_list = [item for sublist in list(
                neutral_pronouns.values()) for item in sublist]
            female_pronouns_list = [item for sublist in list(
                female_pronouns.values()) for item in sublist]
            self._tests['replace_to_male_pronouns']['pronouns_to_substitute'] = female_pronouns_list + \
                neutral_pronouns_list
            self._tests['replace_to_male_pronouns']['pronoun_type'] = 'male'

        if 'replace_to_female_pronouns' in self._tests:
            neutral_pronouns_list = [item for sublist in list(
                neutral_pronouns.values()) for item in sublist]
            male_pronouns_list = [item for sublist in list(
                male_pronouns.values()) for item in sublist]
            self._tests['replace_to_female_pronouns']['pronouns_to_substitute'] = male_pronouns_list + \
                neutral_pronouns_list
            self._tests['replace_to_female_pronouns']['pronoun_type'] = 'female'

        if 'replace_to_neutral_pronouns' in self._tests:
            female_pronouns_list = [item for sublist in list(
                female_pronouns.values()) for item in sublist]
            male_pronouns_list = [item for sublist in list(
                male_pronouns.values()) for item in sublist]
            self._tests['replace_to_neutral_pronouns']['pronouns_to_substitute'] = male_pronouns_list + \
                female_pronouns_list
            self._tests['replace_to_neutral_pronouns']['pronoun_type'] = 'neutral'

        self._data_handler = data_handler

    def transform(self) -> List[Sample]:
        """"""
        # NOTE: I don't know if we need to work with a dataframe of if we can keep it as a List[Sample]
        all_samples = []
        for test_name, params in self._tests.items():
            data_handler_copy = [x.copy() for x in self._data_handler]
            transformed_samples = globals()[PERTURB_CLASS_MAP[test_name]].transform(
                data_handler_copy, **params)
            for sample in transformed_samples:
                sample.test_type = test_name

            # Check for number of perturbed sentences
            transformed_samples = [
                x for x in transformed_samples if x.original != x.test_case]
            if len(transformed_samples) == 0:
                logging.warning(
                    "%s did not create any test cases. Test will be removed from results.", test_name)
            elif len(transformed_samples) < 10:
                logging.warning("%s has perturbed %s sample(s). Results may not be reliable.", test_name, len(
                    transformed_samples))

            all_samples.extend(transformed_samples)
        return all_samples


class UpperCase(BasePerturbation):
    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Transform a list of strings with uppercase perturbation
        Args:
            sample_list: List of sentences to apply perturbation.
        Returns:
            List of sentences that uppercase perturbation is applied.
        """
        for sample in sample_list:
            sample.test_case = sample.original.upper()
        return sample_list


class LowerCase(BasePerturbation):
    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Transform a list of strings with lowercase perturbation
        Args:
            sample_list: List of sentences to apply perturbation.
        Returns:
            List of sentences that lowercase perturbation is applied.
        """
        for sample in sample_list:
            sample.test_case = sample.original.lower()
        return sample_list


class TitleCase(BasePerturbation):
    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Transform a list of strings with titlecase perturbation
        Args:
            sample_list: List of sentences to apply perturbation.
        Returns:
            List of sentences that titlecase perturbation is applied.
        """
        for sample in sample_list:
            sample.test_case = sample.original.title()
        return sample_list


class AddPunctuation(BasePerturbation):
    @staticmethod
    def transform(sample_list: List[Sample], whitelist: Optional[List[str]] = None) -> List[Sample]:
        """Add punctuation at the end of the string, if there is punctuation at the end skip it
        Args:
            sample_list: List of sentences to apply perturbation.
            whitelist: Whitelist for punctuations to add to sentences.
        Returns:
            List of sentences that have punctuation at the end.
        """

        if whitelist is None:
            whitelist = ['!', '?', ',', '.', '-', ':', ';']

        for sample in sample_list:
            if sample.original[-1] not in whitelist:
                chosen_punc = random.choice(whitelist)
                sample.test_case = sample.original + chosen_punc
                sample.transformations = [
                    Transformation(
                        original_span=Span(
                            start=len(sample.original),
                            end=len(sample.original),
                            word=""
                        ),
                        new_span=Span(
                            start=len(sample.original),
                            end=len(sample.test_case),
                            word=chosen_punc
                        ),
                        ignore=True
                    )
                ]
            else:
                sample.test_case = sample.original
        return sample_list


class StripPunctuation(BasePerturbation):
    @staticmethod
    def transform(sample_list: List[Sample], whitelist: Optional[List[str]] = None) -> List[Sample]:
        """Add punctuation from the string, if there isn't punctuation at the end skip it

        Args:
            sample_list: List of sentences to apply perturbation.
            whitelist: Whitelist for punctuations to strip from sentences.
        Returns:
            List of sentences that punctuation is stripped.
        """

        if whitelist is None:
            whitelist = ['!', '?', ',', '.', '-', ':', ';']

        for sample in sample_list:
            if sample.original[-1] in whitelist:
                sample.test_case = sample.original[:-1]
                sample.transformations = [
                    Transformation(
                        original_span=Span(
                            start=len(sample.original) - 1,
                            end=len(sample.original),
                            word=sample.original[-1:]
                        ),
                        new_span=Span(
                            start=len(sample.test_case),
                            end=len(sample.test_case),
                            word=""
                        ),
                        ignore=True
                    )
                ]
            else:
                sample.test_case = sample.original
        return sample_list


class AddTypo(BasePerturbation):
    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Add typo to the sentences using keyboard typo and swap typo.
        Args:
            sample_list: List of sentences to apply perturbation.
        Returns:
            List of sentences that typo introduced.
        """
        for sample in sample_list:
            if len(sample.original) < 5:
                sample.test_case = sample.original
                continue

            string = list(sample.original)
            if random.random() > 0.1:
                idx_list = list(range(len(TYPO_FREQUENCY)))
                char_list = list(TYPO_FREQUENCY.keys())

                counter, idx = 0, -1
                while counter < 10 and idx == -1:
                    idx = random.randint(0, len(string) - 1)
                    char = string[idx]
                    if TYPO_FREQUENCY.get(char.lower(), None):
                        char_frequency = TYPO_FREQUENCY[char.lower()]

                        if sum(char_frequency) > 0:
                            chosen_char = random.choices(idx_list, weights=char_frequency)
                            difference = ord(char.lower()) - ord(char_list[chosen_char[0]])
                            char = chr(ord(char) - difference)
                            string[idx] = char
                    else:
                        idx = -1
                        counter += 1
            else:
                string = list(string)
                swap_idx = random.randint(0, len(string) - 2)
                tmp = string[swap_idx]
                string[swap_idx] = string[swap_idx + 1]
                string[swap_idx + 1] = tmp

            sample.test_case = "".join(string)
        return sample_list


class SwapEntities(BasePerturbation):
    @staticmethod
    def transform(
            sample_list: List[Sample],
            labels: List[List[str]] = None,
            terminology: Dict[str, List[str]] = None
    ) -> List[Sample]:
        """Swaps named entities with the new one from the terminology extracted from passed data.

        Args:
            sample_list: List of sentences to process.
            labels: Corresponding labels to make changes according to sentences.
            terminology: Dictionary of entities and corresponding list of words.
        Returns:
            List of sentences that entities swapped with the terminology.
        """

        if terminology is None:
            raise ValueError('In order to generate test cases for swap_entities, terminology should be passed!')

        if labels is None:
            raise ValueError('In order to generate test cases for swap_entities, labels should be passed!')

        assert len(sample_list) == len(labels), f"'labels' and 'sample_list' must have same lengths."

        for sample, sample_labels in zip(sample_list, labels):
            if all([label == "O" for label in sample_labels]):
                sample.test_case = sample.original
                continue

            sent_tokens = sample.original.split(' ')

            ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sample_labels])
            ent_idx, = np.where(ent_start_pos == 1)

            replace_idx = np.random.choice(ent_idx)
            ent_type = sample_labels[replace_idx][2:]
            replace_idxs = [replace_idx]
            if replace_idx < len(sample_labels) - 1:
                for i, label in enumerate(sample_labels[replace_idx + 1:]):
                    if label == f'I-{ent_type}':
                        replace_idxs.append(i + replace_idx + 1)
                    else:
                        break

            replace_token = sent_tokens[replace_idx: replace_idx + len(replace_idxs)]
            token_length = len(replace_token)
            replace_token = " ".join(replace_token)

            chosen_ent = random.choice(terminology[ent_type])
            replace_token_pos = re.search(replace_token, sample.original)

            sample.test_case = sample.original.replace(replace_token, chosen_ent)
            sample.transformations = [
                Transformation(
                    original_span=Span(
                        start=replace_token_pos.start(),
                        end=replace_token_pos.end(),
                        word=replace_token
                    ),
                    new_span=Span(
                        start=replace_token_pos.start(),
                        end=replace_token_pos.start() + len(chosen_ent),
                        word=chosen_ent
                    ),
                    ignore=False
                )
            ]

        return sample_list


def get_cohyponyms_wordnet(word: str) -> str:
    """
    Retrieve co-hyponym of the input string using WordNet when a hit is found.

    Args:
        word: input string to retrieve co-hyponym
    Returns:
        Cohyponym of the input word if exists, else original word.
    """

    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        raise ImportError("WordNet is not available!\n"
                          "Please install WordNet via pip install wordnet to use swap_cohyponyms")

    orig_word = word
    word = word.lower()
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    syns = wn.synsets(word)

    if len(syns) == 0:
        return orig_word
    else:
        hypernym = syns[0].hypernyms()
        if len(hypernym) == 0:
            return orig_word
        else:
            hypos = hypernym[0].hyponyms()
            hypo_len = len(hypos)
            if hypo_len == 1:
                name = str(hypos[0].lemmas()[0])
            else:
                ind = random.sample(range(hypo_len), k=1)[0]
                name = str(hypos[ind].lemmas()[0])
                while name == word:
                    ind = random.sample(range(hypo_len), k=1)[0]
                    name = str(hypos[ind].lemmas()[0])
            return name.replace("_", " ").split(".")[0][7:]


class GenderPronounBias(BasePerturbation):
    @staticmethod
    def transform(sample_list: List[Sample], pronouns_to_substitute: List[str], pronoun_type: str) -> List[Sample]:
        """Replace pronouns to check the gender bias

        Args:
            sample_list: List of sentences to apply perturbation.
            pronouns_to_substitute: list of pronouns that need to be substituted.
            pronoun_type: replacing pronoun type string ('male', 'female' or 'neutral')

        Returns:
            List of sentences with replaced pronouns
        """

        for sample in sample_list:
            tokens_to_substitute = [token for token in sample.original.split(' ') if
                                    token.lower() in pronouns_to_substitute]

            if len(tokens_to_substitute) != 0:
                replace_token = random.choice(tokens_to_substitute)
                if pronoun_type == "female":
                    combined_dict = {k: male_pronouns[k] + neutral_pronouns[k] for k in male_pronouns.keys()}
                    chosen_dict = female_pronouns
                elif pronoun_type == "male":
                    combined_dict = {k: female_pronouns[k] + neutral_pronouns[k] for k in female_pronouns.keys()}
                    chosen_dict = male_pronouns
                elif pronoun_type == "neutral":
                    combined_dict = {k: female_pronouns[k] + male_pronouns[k] for k in female_pronouns.keys()}
                    chosen_dict = neutral_pronouns

                for key, value in chosen_dict.items():
                    if replace_token.lower() in value:
                        type_of_pronoun = str(key)
                        break

                chosen_token = random.choice(combined_dict[type_of_pronoun])
                diff_len = len(chosen_token) - len(replace_token)

                replaced_string = sample.original
                transformations = []
                for _ in range(len(re.findall(replace_token, sample.original))):
                    span = re.search(replace_token, replaced_string)
                    replaced_string = re.sub(replace_token, chosen_token, replaced_string, count=1)
                    transformations.append(
                        Transformation(
                            original_span=Span(start=span.start(), end=span.end(), word=replace_token),
                            new_span=Span(start=span.start(), end=span.end() + diff_len, word=chosen_token),
                            ignore=False
                        )
                    )

                sample.test_case = replaced_string
                sample.transformations = transformations
            else:
                sample.test_case = sample.original

        return sample_list


class SwapCohyponyms(BasePerturbation):

    @staticmethod
    def transform(
            sample_list: List[Sample],
            labels: List[List[str]] = None,
    ) -> List[Sample]:
        """Swaps named entities with the new one from the terminology extracted from passed data.

        Args:
            sample_list: List of sentences to process.
            labels: Corresponding labels to make changes according to sentences.

        Returns:
            List sample indexes and corresponding augmented sentences, tags and labels if provided.
        """

        if labels is None:
            raise ValueError('In order to generate test cases for swap_entities, terminology should be passed!')

        assert len(sample_list) == len(labels), f"'labels' and 'sample_list' must have same lengths."

        for sample, sample_labels in zip(sample_list, labels):
            if all([label == "O" for label in sample_labels]):
                sample.test_case = sample.original
                continue

            sent_tokens = sample.original.split(' ')

            ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sample_labels])
            ent_idx, = np.where(ent_start_pos == 1)

            replace_idx = np.random.choice(ent_idx)
            ent_type = sample_labels[replace_idx][2:]
            replace_idxs = [replace_idx]
            if replace_idx < len(sample_labels) - 1:
                for i, label in enumerate(sample_labels[replace_idx + 1:]):
                    if label == f'I-{ent_type}':
                        replace_idxs.append(i + replace_idx + 1)
                    else:
                        break

            replace_token = sent_tokens[replace_idx: replace_idx + len(replace_idxs)]
            token_length = len(replace_token)
            replace_token = " ".join(replace_token)

            chosen_ent = get_cohyponyms_wordnet(replace_token)
            replace_token_pos = re.search(replace_token, sample.original)

            sample.test_case = sample.original.replace(replace_token, chosen_ent)
            sample.transformations = [
                Transformation(
                    original_span=Span(
                        start=replace_token_pos.start(),
                        end=replace_token_pos.end(),
                        word=replace_token
                    ),
                    new_span=Span(
                        start=replace_token_pos.start(),
                        end=replace_token_pos.start() + len(chosen_ent),
                        word=chosen_ent
                    ),
                    ignore=False
                )
            ]

        return sample_list


class ConvertAccent(BasePerturbation):

    @staticmethod
    def transform(sample_list: List[Sample], accent_map: Dict[str, str] = None) -> List[Sample]:
        """Converts input sentences using a conversion dictionary
        Args:
            sample_list: List of sentences to process.
            accent_map: Dictionary with conversion terms.
        Returns:
            List of sentences that perturbed with accent conversion.
        """
        for sample in sample_list:
            tokens = set(sample.original.split(' '))
            replaced_string = sample.original
            transformations = []

            for token in tokens:
                new_token = accent_map.get(token.lower(), token)
                if new_token != token:
                    diff_len = len(new_token) - len(token)
                    nb_occurrences = len(re.findall(token, replaced_string))

                    for c in range(nb_occurrences):
                        span = re.search(token, replaced_string)
                        replaced_string = re.sub(token, new_token, replaced_string, count=1)
                        transformations.append(
                            Transformation(
                                original_span=Span(start=span.start(), end=span.end(), word=token),
                                new_span=Span(start=span.start(), end=span.end() + diff_len, word=new_token),
                                ignore=False
                            )
                        )
            sample.test_case = replaced_string
            sample.transformations = transformations

        return sample_list


class AddContext(BasePerturbation):

    @staticmethod
    def transform(
            sample_list: List[Sample],
            starting_context: Optional[List[str]] = None,
            ending_context: Optional[List[str]] = None,
            strategy: str = None,
    ) -> List[Sample]:
        """Converts input sentences using a conversion dictionary
        Args:
            sample_list: List of sentences to process.
            strategy: Config method to adjust where will context tokens added. start, end or combined.
            starting_context: list of terms (context) to input at start of sentences.
            ending_context: list of terms (context) to input at end of sentences.
        Returns:
            List of sentences that context added at to begging, end or both, randomly.
        """
        possible_methods = ['start', 'end', 'combined']

        for sample in sample_list:
            if strategy is None:
                strategy = random.choice(possible_methods)
            elif strategy not in possible_methods:
                raise ValueError(
                    f"Add context strategy must be one of 'start', 'end', 'combined'. Cannot be {strategy}."
                )

            transformations = []
            if strategy == "start" or strategy == "combined":
                add_tokens = random.choice(starting_context)
                add_string = " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens
                string = add_string + ' ' + sample.original
                transformations.append(
                    Transformation(
                        original_span=Span(start=0, end=0, word=""),
                        new_span=Span(start=0, end=len(add_string) + 1, word=add_string),
                        ignore=True
                    )
                )
            else:
                string = sample.original

            if strategy == "end" or strategy == "combined":
                add_tokens = random.choice(ending_context)
                add_string = " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens

                if sample.original[-1].isalnum():
                    from_start, from_end = len(string), len(string)
                    to_start = from_start + 1
                    to_end = to_start + len(add_string) + 1
                    string = string + " " + add_string
                else:
                    from_start, from_end = len(string[:-1]), len(string[:-1])
                    to_start = from_start
                    to_end = to_start + len(add_string) + 1
                    string = string[:-1] + add_string + " " + string[-1]

                transformations.append(
                    Transformation(
                        original_span=Span(start=from_start, end=from_end, word=""),
                        new_span=Span(start=to_start, end=to_end, word=string[to_start:to_end]),
                        ignore=True
                    )
                )

            sample.test_case = string
            sample.transformations = transformations
        return sample_list


class AddContraction(BasePerturbation):

    @staticmethod
    def transform(
            sample_list: List[Sample],
    ) -> List[Sample]:
        """Converts input sentences using a conversion dictionary
        Args:
            sample_list: List of sentences to process.
        Returns:
            List[Sample]:
                transformed samples
        """

        def custom_replace(match):
            """
              regex replace for contraction.
            """
            token = match.group(0)
            contracted_token = CONTRACTION_MAP.get(
                token, CONTRACTION_MAP.get(token.lower()))

            is_upper_case = token[0]
            expanded_contraction = is_upper_case + contracted_token[1:]
            return expanded_contraction

        for sample in sample_list:
            replaced_string = sample.original
            transformations = []

            for contraction in CONTRACTION_MAP:
                search = re.search(contraction, sample.original, flags=re.IGNORECASE | re.DOTALL)
                if search:
                    new_string = CONTRACTION_MAP.get(search.group(), CONTRACTION_MAP.get(search.group()))
                    diff_len = len(new_string) - len(search.group())
                    replaced_string = re.sub(contraction, custom_replace, replaced_string,
                                             flags=re.IGNORECASE | re.DOTALL)
                    transformations.append(
                        Transformation(
                            original_span=Span(start=search.start(), end=search.end(), word=search.group()),
                            new_span=Span(start=search.start(), end=search.end() + diff_len, word=new_string),
                            ignore=False
                        )
                    )
            sample.test_case = replaced_string
            sample.transformations = transformations
        return sample_list
