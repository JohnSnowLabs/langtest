import numpy as np
import random
import re
import pandas as pd

from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional, Dict

from .utils import (
    TYPO_FREQUENCY,
    PERTURB_CLASS_MAP,
    DEFAULT_PERTURBATIONS,
    CONTRACTION_MAP,
    A2B_DICT,
    create_terminology
)


class BasePerturbation(ABC):

    @staticmethod
    @abstractmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        return NotImplementedError


class PerturbationFactory:

    def __init__(self, data_handler, tests=None) -> None:

        if tests is []:
            tests = DEFAULT_PERTURBATIONS

        self._tests = dict()
        for test in tests:
            if isinstance(test, str):
                if test not in DEFAULT_PERTURBATIONS:
                     raise ValueError(f'Invalid test specification: {test}. Available tests are: {DEFAULT_PERTURBATIONS}')
                self._tests[test] = dict()
            elif isinstance(test, dict):
                test_name = list(test.keys())[0]
                if test_name not in DEFAULT_PERTURBATIONS:
                     raise ValueError(f'Invalid test specification: {test_name}. Available tests are: {DEFAULT_PERTURBATIONS}')

                self._tests[test_name] = reduce(lambda x, y: {**x, **y}, test[test_name])
            else:
                raise ValueError(
                    f'Invalid test configuration! Tests can be '
                    f'[1] test name as string or '
                    f'[2] dictionary of test name and corresponding parameters.'
                )

        if 'swap_entities' in self._tests:
            self._tests['swap_entities']['terminology'] = create_terminology(data_handler)
            self._tests['swap_entities']['labels'] = data_handler.label

        if 'swap_cohyponyms' in self._tests:
            self._tests['swap_cohyponyms']['labels'] = data_handler.label

        if "american_to_british" in self._tests:
            self._tests['american_to_british']['accent_map'] = A2B_DICT

        if "british_to_american" in self._tests:
            self._tests['american_to_british']['accent_map'] = {v: k for k, v in A2B_DICT.items()}

        self._data_handler = data_handler

    def transform(self):

        generated_results_df = pd.DataFrame()
        for test_name, params in self._tests.items():
            print(test_name)
            res = globals()[PERTURB_CLASS_MAP[test_name]].transform(list(self._data_handler.text), **params)
            result_df = pd.DataFrame()
            result_df['Original'] = self._data_handler.text
            result_df['Test_Case'] = res
            result_df['Test_type'] = test_name
            generated_results_df = pd.concat([generated_results_df, result_df], ignore_index=True)

        return generated_results_df


class UpperCase(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """Transform a list of strings with uppercase perturbation
        Args:
            list_of_strings: List of sentences to apply perturbation.
        Returns:
            List of sentences that uppercase perturbation is applied.
        """
        return [string.upper() for string in list_of_strings]


class LowerCase(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """Transform a list of strings with lowercase perturbation
        Args:
            list_of_strings: List of sentences to apply perturbation.
        Returns:
            List of sentences that lowercase perturbation is applied.
        """
        return [string.lower() for string in list_of_strings]


class TitleCase(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """Transform a list of strings with titlecase perturbation
        Args:
            list_of_strings: List of sentences to apply perturbation.
        Returns:
            List of sentences that titlecase perturbation is applied.
        """
        return [string.title() for string in list_of_strings]


class AddPunctuation(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str], whitelist: Optional[List[str]] = None) -> List[str]:
        """Add punctuation at the end of the string, if there is punctuation at the end skip it
        Args:
            list_of_strings: List of sentences to apply perturbation.
            whitelist: Whitelist for punctuations to add to sentences.
        Returns:
            List of sentences that have punctuation at the end.
        """

        if whitelist is None:
            whitelist = ['!', '?', ',', '.', '-', ':', ';']

        perturb_list = list()
        for string in list_of_strings:

            if string[-1] not in whitelist:
                perturb_list.append(string[-1] + random.choice(whitelist))
            else:
                perturb_list.append(string)

        return perturb_list


class StripPunctuation(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str], whitelist: Optional[List[str]] = None) -> List[str]:
        """Add punctuation from the string, if there isn't punctuation at the end skip it

        Args:
            list_of_strings: List of sentences to apply perturbation.
            whitelist: Whitelist for punctuations to strip from sentences.
        Returns:
            List of sentences that punctuation is stripped.
        """

        if whitelist is None:
            whitelist = '!?,.-:;'

        return [s.strip(whitelist) for s in list_of_strings]


class AddTypo(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """Add typo to the sentences using keyboard typo and swap typo.
        Args:
            list_of_strings: List of sentences to apply perturbation.
        Returns:
            List of sentences that typo introduced.
        """
        perturb_list = []
        for string in list_of_strings:

            if len(string) < 5:
                perturb_list.append(string)
                continue

            string = list(string)
            if random.random() > 0.1:

                indx_list = list(range(len(TYPO_FREQUENCY)))
                char_list = list(TYPO_FREQUENCY.keys())

                counter = 0
                indx = -1
                while counter < 10 and indx == -1:
                    indx = random.randint(0, len(string) - 1)
                    char = string[indx]
                    if TYPO_FREQUENCY.get(char.lower(), None):

                        char_frequency = TYPO_FREQUENCY[char.lower()]

                        if sum(char_frequency) > 0:
                            chosen_char = random.choices(indx_list, weights=char_frequency)
                            difference = ord(char.lower()) - ord(char_list[chosen_char[0]])
                            char = chr(ord(char) - difference)
                            string[indx] = char

                    else:
                        indx = -1
                        counter += 1

            else:
                string = list(string)
                swap_indx = random.randint(0, len(string) - 2)
                tmp = string[swap_indx]
                string[swap_indx] = string[swap_indx + 1]
                string[swap_indx + 1] = tmp

            perturb_list.append("".join(string))

        return perturb_list


class SwapEntities(BasePerturbation):
    @staticmethod
    def transform(
            list_of_strings: List[str],
            labels: List[List[str]] = None,
            terminology: Dict[str, List[str]] = None
    ) -> List[str]:
        """Swaps named entities with the new one from the terminology extracted from passed data.

        Args:
            list_of_strings: List of sentences to process.
            labels: Corresponding labels to make changes according to sentences.
            terminology: Dictionary of entities and corresponding list of words.
        Returns:
            List of sentences that entities swapped with the terminology.
        """

        if terminology is None:
            raise ValueError('In order to generate test cases for swap_entities, terminology should be passed!')

        if labels is None:
            raise ValueError('In order to generate test cases for swap_entities, labels should be passed!')

        perturb_sent = []
        for indx, string in enumerate(list_of_strings):

            sent_tokens = string.split(' ')
            sent_labels = labels[indx]

            ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sent_labels])
            ent_indx, = np.where(ent_start_pos == 1)

            #  no swaps since there is no entity in the sentence
            if len(ent_indx) == 0:
                perturb_sent.append(string)
                continue

            replace_indx = np.random.choice(ent_indx)
            ent_type = sent_labels[replace_indx][2:]
            replace_indxs = [replace_indx]
            if replace_indx < len(sent_labels) - 1:
                for i, label in enumerate(sent_labels[replace_indx + 1:]):
                    if label == f'I-{ent_type}':
                        replace_indxs.append(i + replace_indx + 1)
                    else:
                        break

            replace_token = sent_tokens[replace_indx: replace_indx + len(replace_indxs)]
            token_length = len(replace_token)
            replace_token = " ".join(replace_token)
            replace_token = replace_token.replace("-","").replace(".","").replace("\'s","").replace("\'S","")

            proper_entities = [ent for ent in terminology[ent_type] if len(ent.split(' ')) == token_length]
            chosen_ent = random.choice(proper_entities)
            chosen_ent = chosen_ent.replace("-","").replace(".","").replace("\'s","").replace("\'S","")
            replaced_string = string.replace(replace_token, chosen_ent)
            perturb_sent.append(replaced_string)

        return perturb_sent


def get_cohyponyms_wordnet(word: str) -> str:
    """
    Retrieve co-hyponym of the input string using WordNet when a hit is found.

    Args:
        word: input string to retrieve co-hyponym
    Returns:
        Cohyponym of the input word if exist, else original word.
    """

    try:
        import wn
    except ImportError:
        raise ImportError("WordNet is not available!\n"
                          "Please install WordNet via pip install wordnet to use swap_coyphonyms")

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
                name = hypos[0].lemmas()[0]
            else:
                ind = random.sample(range(hypo_len), k=1)[0]
                name = hypos[ind].lemmas()[0]
                while name == word:
                    ind = random.sample(range(hypo_len), k=1)[0]
                    name = hypos[ind].lemmas()[0]
            return name.replace("_", " ")


class SwapCohyponyms(BasePerturbation):

    @staticmethod
    def transform(
            list_of_strings: List[str],
            labels: List[List[str]] = None,
    ) -> List[str]:
        """Swaps named entities with the new one from the terminology extracted from passed data.

        Args:
            list_of_strings: List of sentences to process.
            labels: Corresponding labels to make changes according to sentences.

        Returns:
            List sample indexes and corresponding augmented sentences, tags and labels if provided.
        """

        if labels is None:
            raise ValueError('In order to generate test cases for swap_entities, terminology should be passed!')

        perturb_sent = []
        for indx, string in enumerate(list_of_strings):

            sent_tokens = string.split(' ')
            sent_labels = labels[indx]

            ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sent_labels])
            ent_indx, = np.where(ent_start_pos == 1)

            #  no swaps since there is no entity in the sentence
            if len(ent_indx) == 0:
                perturb_sent.append(string)
                continue

            replace_indx = np.random.choice(ent_indx)
            ent_type = sent_labels[replace_indx][2:]
            replace_indxs = [replace_indx]
            if replace_indx < len(sent_labels) - 1:
                for i, label in enumerate(sent_labels[replace_indx + 1:]):
                    if label == f'I-{ent_type}':
                        replace_indxs.append(i + replace_indx + 1)
                    else:
                        break

            replace_token = sent_tokens[replace_indx: replace_indx + len(replace_indxs)]

            replace_token = " ".join(replace_token)
            chosen_ent = get_cohyponyms_wordnet(replace_token)
            replaced_string = string.replace(replace_token, chosen_ent)
            perturb_sent.append(replaced_string)

        return perturb_sent


class ConvertAccent(BasePerturbation):

    @staticmethod
    def transform(list_of_strings: List[str], accent_map: Dict[str, str] = None) -> List[str]:
        """Converts input sentences using a conversion dictionary
        Args:
            list_of_strings: List of sentences to process.
            accent_map: Dictionary with conversion terms.
        Returns:
            List of sentences that perturbed with accent conversion.
        """
        perturb_sent = []
        for string in list_of_strings:
            tokens = string.split(' ')
            tokens = [accent_map[t.lower()] if accent_map.get(t.lower(), None) else t for t in tokens]
            perturb_sent.append(' '.join(tokens))

        return perturb_sent


class AddContext(BasePerturbation):

    @staticmethod
    def transform(
            list_of_strings: List[str],
            starting_context: Optional[List[str]] = None,
            ending_context: Optional[List[str]] = None,
            strategy: List[str] = None,
    ) -> List[str]:
        """Converts input sentences using a conversion dictionary
        Args:
            list_of_strings: List of sentences to process.
            strategy: Config method to adjust where will context tokens added. start, end or combined.
            starting_context: list of terms (context) to input at start of sentences.
            ending_context: list of terms (context) to input at end of sentences.
        Returns:
            List of sentences that context added at to begging, end or both, randomly.
        """

        possible_methods = ['start', 'end', 'combined']
        perturb_sent = []
        for string in list_of_strings:

            if strategy is None:
                strategy = random.choice(possible_methods)
            elif strategy not in possible_methods:
                raise ValueError(
                    f"Add context strategy must be one of 'start', 'end', 'combined'. Can not be {strategy}."
                )

            if strategy == "start" or strategy == "combined":
                add_tokens = random.choice(starting_context)

                #   join tokens
                add_string = " ".join(add_tokens)
                string = add_string + ' ' + string

            if strategy == "end" or strategy == "combined":
                add_tokens = random.choice(ending_context)

                #   join tokens
                add_string = " ".join(add_tokens)

                if string[-1].isalnum():
                    string = string + ' ' + add_string
                else:
                    string = string[:-1] + add_string + " " + string[-1]

            perturb_sent.append(string)

        return perturb_sent


class AddContraction(BasePerturbation):

    @staticmethod
    def transform(
            list_of_strings: List[str],
    ) -> List[str]:
        """Converts input sentences using a conversion dictionary
        Args:
            list_of_strings: List of sentences to process.
        """

        def custom_replace(match):
            """
              regex replace for contraction.
            """
            token = match.group(0)
            contracted_token = CONTRACTION_MAP.get(token, CONTRACTION_MAP.get(token.lower()))

            is_upper_case = token[0]
            expanded_contraction = is_upper_case + contracted_token[1:]
            return expanded_contraction

        perturb_sent = []
        for string in list_of_strings:
            for contraction in CONTRACTION_MAP:
                if re.search(contraction, string, flags=re.IGNORECASE | re.DOTALL):
                    string = re.sub(contraction, custom_replace, string, flags=re.IGNORECASE | re.DOTALL)

            perturb_sent.append(string)

        return perturb_sent
