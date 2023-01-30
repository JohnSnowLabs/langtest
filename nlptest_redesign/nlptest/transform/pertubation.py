from functools import reduce
from typing import List, Optional
import numpy as np
import random
import pandas as pd

_DEFAULT_PERTURBATIONS = ["uppercase", "lowercase"]


class PerturbationFactory:

    def __init__(self, data_handler, tests=None) -> None:
        if tests is None:
            tests = _DEFAULT_PERTURBATIONS
        self._tests = tests
        self._data_handler = data_handler

    def transform(self):

        generated_results_df = pd.DataFrame()
        # result > List returned by func

        if len(self._tests) != 0:
            for test in self._tests:
                tmp_df = pd.DataFrame()

                if type(test) == str:
                    res = self.__getattribute__(f"generate_{test}")(self._data_handler)
                    tmp_df['Orginal'] = self._data_handler[:30]
                    tmp_df['Test_Case'] = res[:30]
                    tmp_df['Test_type'] = test  # "Add_Context" or "Uppercase"
                    if generated_results_df.empty:
                        generated_results_df = tmp_df
                    else:
                        generated_results_df = pd.concat([generated_results_df, tmp_df])

        return generated_results_df

    @staticmethod
    def generate_uppercase(self, list_of_strings: List[str]):
        return [i.upper() for i in list_of_strings]

    @staticmethod
    def generate_lowercase(self, list_of_strings: List[str]):
        return [i.lower() for i in list_of_strings]

    @staticmethod
    def generate_add_context(self, list_of_strings: List[str],
                             method: str = "Start",
                             starting_context: Optional[List[str]] = None,
                             ending_context: Optional[List[str]] = None,
                             noise_prob: float = 1) -> List:
        """Adds tokens at the beginning and/or at the end of strings
        :param list_of_strings: list of sentences to process
        :param method: 'Start' adds context only at the beginning, 'End' adds it at the end, 'Combined' adds context both
        at the beginning and at the end, 'Random' means method for each string is randomly assigned.
        :param starting_context: list of terms (context) to input at start of sentences.
        :param ending_context: list of terms (context) to input at end of sentences.
        :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
        # """

        np.random.seed(7)
        outcome_list_of_strings = []
        for string in list_of_strings:
            if random.random() > noise_prob:
                outcome_list_of_strings.append(string)
                continue
            if method == 'Start':
                outcome_list_of_strings.append(random.choice(starting_context) + ' ' + string)
            if method == 'End':
                if string[-1].isalnum() or string[-1] == '.' or string[-1] == "'" or string[-1] == '"':
                    outcome_list_of_strings.append(string + ' ' + random.choice(ending_context))
                else:
                    outcome_list_of_strings.append(string[:-1] + ' ' + random.choice(ending_context))
            elif method == 'Combined':
                if string[-1].isalnum() or string[-1] == '.' or string[-1] == "'" or string[-1] == '"':
                    outcome_list_of_strings.append(
                        random.choice(starting_context) + ' ' + string + ' ' + random.choice(ending_context))
                else:
                    outcome_list_of_strings.append(
                        random.choice(starting_context) + ' ' + string[:-1] + ' ' + random.choice(ending_context))
            elif method == 'Random':
                if string[-1].isalnum():
                    list_of_possibilities = [(random.choice(starting_context) + ' ' + string),
                                             (string + ' ' + random.choice(ending_context))]
                    outcome_list_of_strings.append(random.choice(list_of_possibilities))
                else:
                    list_of_possibilities = [(random.choice(starting_context) + ' ' + string),
                                             (string[:-1] + ' ' + random.choice(ending_context))]
                    outcome_list_of_strings.append(random.choice(list_of_possibilities))
        return outcome_list_of_strings
