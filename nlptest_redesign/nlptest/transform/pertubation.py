from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional
import numpy as np
import random
import pandas as pd

from .utils import TYPO_FREQUENCY
_DEFAULT_PERTURBATIONS = ["uppercase", "lowercase"]


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

        self._data_handler = data_handler

    def transform(self):

        generated_results_df = pd.DataFrame()
        for test_name, params in self._tests.items():
            res = self.__getattribute__(f"generate_{test_name}")(self._data_handler, **params)
            result_df = pd.DataFrame()
            result_df['Original'] = self._data_handler[:30]
            result_df['Test_Case'] = res[:30]
            result_df['Test_type'] = test_name
            generated_results_df = pd.concat([generated_results_df, result_df])

        return generated_results_df


class UpperCase(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """Transform a list of strings with uppercase perturbation"""
        return [string.upper() for string in list_of_strings]


class LowerCase(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """Transform a list of strings with lowercase perturbation"""
        return [string.lower() for string in list_of_strings]


class TitleCase(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """Transform a list of strings with titlecase perturbation"""
        return [string.title() for string in list_of_strings]


class Add_Punctuation(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str], whitelist: Optional[List[str]] = None) -> List[str]:
        """Add punctuation at the end of the string, if there is punctuation at the end skip it

        Args:
            list_of_strings: List of sentences to apply perturbation.
            whitelist: Whitelist for punctuations to add to sentences.
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


class Strip_Punctuation(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str], whitelist: Optional[List[str]] = None) -> List[str]:
        """Add punctuation from the string, if there isn't punctuation at the end skip it

        Args:
            list_of_strings: List of sentences to apply perturbation.
            whitelist: Whitelist for punctuations to strip from sentences.
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


class Add_Typo(BasePerturbation):
    @staticmethod
    def transform(list_of_strings: List[str]) -> List[str]:
        """"""

        perturb_list = []
        for string in list_of_strings:

            if len(string) == 1:
                perturb_list.append(string)

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


