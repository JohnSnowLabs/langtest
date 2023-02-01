from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional, Dict
import numpy as np
import random
import pandas as pd

from .utils import _TYPO_FREQUENCY, _PERTURB_CLASS_MAP, _DEFAULT_PERTURBATIONS, create_terminology


class BasePerturbation(ABC):

    @staticmethod
    @abstractmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        return NotImplementedError


class PerturbationFactory:

    def __init__(self, data_handler, tests=None) -> None:

        if tests is []:
            tests = _DEFAULT_PERTURBATIONS

        self._tests = dict()
        for test in tests:
            if isinstance(test, str):
                self._tests[test] = dict()
            elif isinstance(test, dict):
                test_name = list(test.keys())[0]
                self._tests[test_name] = reduce(lambda x, y: {**x, **y}, test[test_name])
            else:
                raise ValueError(
                    f'Invalid test configuration! Tests can be '
                    f'[1] test name as string or '
                    f'[2] dictionary of test name and corresponding parameters.'
                )

        if 'swap_entities' in self._tests:
            self._tests['swap_entities']['terminology'] = create_terminology(data_handler)

        self._data_handler = data_handler

    def transform(self):

        generated_results_df = pd.DataFrame()
        for test_name, params in self._tests.items():
            res = self._class_map[test_name](self._data_handler, **params)
            result_df = pd.DataFrame()
            result_df['Original'] = self._data_handler[:30]
            result_df['Test_Case'] = res[:30]
            result_df['Test_type'] = test_name
            generated_results_df = pd.concat([generated_results_df, result_df])

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
            whitelist = ['!', '?', ',', '.', '-', ':', ';']

        perturb_list = list()
        for string in list_of_strings:

            if string[-1] in whitelist:
                perturb_list.append(string[-1])
            else:
                perturb_list.append(string)

        return perturb_list


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

            if len(string) == 1:
                perturb_list.append(string)

            string = list(string)
            if random.random() > 0.1:

                indx_list = list(range(len(_TYPO_FREQUENCY)))
                char_list = list(_TYPO_FREQUENCY.keys())

                counter = 0
                indx = -1
                while counter < 10 and indx == -1:
                    indx = random.randint(0, len(string) - 1)
                    char = string[indx]
                    if _TYPO_FREQUENCY.get(char.lower(), None):

                        char_frequency = _TYPO_FREQUENCY[char.lower()]

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

            proper_entities = [ent for ent in terminology[ent_type] if len(ent.split(' ')) == token_length]
            chosen_ent = random.choice(proper_entities)
            replaced_string = string.replace(replace_token, chosen_ent)
            perturb_sent.append(replaced_string)

        return perturb_sent


class SwapCoyphonyms(BasePerturbation):

    @staticmethod
    def get_cohyponyms_wordnet(word: str) -> str:
        """
        Retrieve co-hyponym of the input string using WordNet when a hit is found.
        :param word: input string to retrieve co-hyponym
        :return: Cohyponym of the input word if exist, else original word.
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

    def transform(
            self,
            list_of_strings: List[str],
            labels: List[List[str]] = None,
            terminology: Dict[str, List[str]] = None
    ) -> List[str]:
        """Swaps named entities with the new one from the terminology extracted from passed data.

        Args:
            list_of_strings: List of sentences to process.
            labels: Corresponding labels to make changes according to sentences.
            terminology: Dictionary of entities and corresponding list of words.
        """

        if terminology is None:
            raise ValueError('In order to generate test cases for swap_entities, terminology should be passed!')

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
            chosen_ent = self.get_cohyponyms_wordnet(replace_token)
            replaced_string = string.replace(replace_token, chosen_ent)
            perturb_sent.append(replaced_string)

        return perturb_sent


