"""Functions to test NER model robustness against different kinds of perturbations"""
import json
import random
import numpy as np
import pandas as pd
from pandas import DataFrame

from ...utils.imports import is_module_importable

if not is_module_importable('wn'):
    raise ImportError(f'Please run <pip install wn> to use this module.  ')

import wn
from .utils import _CONTRACTION_MAP, _A2B_DICT, _TYPO_FREQUENCY
from pyspark.sql import SparkSession
from sparknlp.base import PipelineModel
from typing import Optional, List, Dict, Tuple, Any
from sklearn.metrics import classification_report


def strip_punctuation(list_of_strings: List[str],
                      keep_numeric_punctuation: bool = True,
                      noise_prob: float = 0.5) -> List[str]:
    """Strips punctuation from list of sentences

    :param list_of_strings: list of sentences to process
    :param keep_numeric_punctuation: whether to keep punctuation related to numeric characters, ie 40,000 or 2.5,
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    defaults to True
    """
    outcome_list_of_strings = []

    for string in list_of_strings:

        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        stripped_string = []

        for idx, char in enumerate(string):

            if idx == 0 and keep_numeric_punctuation:
                if char.isspace() or char.isalnum() or string[idx + 1].isdigit():
                    stripped_string.append(char)

            elif idx == (len(string) - 1) and keep_numeric_punctuation:
                if char.isspace() or char.isalnum() or string[idx - 1].isdigit():
                    stripped_string.append(char)

            elif not keep_numeric_punctuation:
                if char.isspace() or char.isalnum():
                    stripped_string.append(char)

            else:
                if char.isspace() or char.isalnum() or string[idx - 1].isdigit() or string[idx + 1].isdigit():
                    stripped_string.append(char)

        outcome_list_of_strings.append(''.join(stripped_string).replace('  ', ' '))

    return outcome_list_of_strings


def add_punctuation(list_of_strings: List[str],
                    noise_prob: float = 0.5) -> List[str]:
    """Adds a special character at the end of strings, if last character is a special character replace it

    :param list_of_strings: list of sentences to process
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """

    list_of_characters = ['!', '?', ',', '.', '-', ':', ';']

    np.random.seed(7)

    outcome_list_of_strings = []

    for string in list_of_strings:

        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        if string[-1].isalnum():
            outcome_list_of_strings.append(string + random.choice(list_of_characters))

        else:
            outcome_list_of_strings.append(string[0:-1] + random.choice(list_of_characters))

    return outcome_list_of_strings


def modify_capitalization(list_of_strings: List[str], method: str = 'Combined',
                          noise_prob: float = 0.5) -> List[str]:
    """Changes the casing of the input sentences

    :param list_of_strings: list of sentences to process
    :param method: the casing method to use for modifying the list of sentences, defaults to 'Combined'
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """
    np.random.seed(7)

    outcome_list_of_strings = []

    for string in list_of_strings:

        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        if method == 'Uppercase':
            outcome_list_of_strings.append(string.upper())

        elif method == 'Lowercase':
            outcome_list_of_strings.append(string.lower())

        elif method == 'Title':
            outcome_list_of_strings.append(string.title())

        elif method == 'Combined':
            list_of_possibilities = [string.upper(), string.lower(), string.title()]
            outcome_list_of_strings.append(random.choice(list_of_possibilities))

    return outcome_list_of_strings


def add_typo_to_sentence(
        sentence: str,
        frequency: Dict[str, List[int]]
) -> str:

    if len(sentence) == 1:
        return sentence

    sentence = list(sentence)

    if random.random() > 0.1:

        indx_list = list(range(len(frequency)))
        char_list = list(frequency.keys())

        counter = 0
        indx = -1
        while counter < 10 and indx == -1:
            indx = random.randint(0, len(sentence) - 1)
            char = sentence[indx]
            if frequency.get(char.lower(), None):

                char_frequency = frequency[char.lower()]

                if sum(char_frequency) > 0:
                    chosen_char = random.choices(indx_list, weights=char_frequency)
                    difference = ord(char.lower()) - ord(char_list[chosen_char[0]])
                    char = chr(ord(char) - difference)
                    sentence[indx] = char

            else:
                indx = -1
                counter += 1

    else:
        sentence = list(sentence)
        swap_indx = random.randint(0, len(sentence) - 2)
        tmp = sentence[swap_indx]
        sentence[swap_indx] = sentence[swap_indx + 1]
        sentence[swap_indx + 1] = tmp

    return "".join(sentence)


def introduce_typos(list_of_strings: List[str],
                    noise_prob: float = 0.5) -> List[str]:
    """Introduces typos in input sentences

    :param list_of_strings: list of sentences to process
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """
    outcome_list_of_strings = []

    for string in list_of_strings:
        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        outcome_list_of_strings.append(add_typo_to_sentence(string, _TYPO_FREQUENCY))

    return outcome_list_of_strings


def add_contractions(list_of_strings: List[str], noise_prob: float = 0.5) -> List[str]:
    """Adds contractions in input sentences

    :param list_of_strings: list of sentences to process
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """
    outcome_list_of_strings = []

    for string in list_of_strings:

        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        for token in _CONTRACTION_MAP:
            if token in string:
                string = string.replace(token, _CONTRACTION_MAP[token])

        outcome_list_of_strings.append(string)

    return outcome_list_of_strings


def add_context(list_of_strings: List[str], method: str = 'Random',
                starting_context: Optional[List[str]] = None,
                ending_context: Optional[List[str]] = None,
                noise_prob: float = 0.5) -> List[str]:
    """Adds tokens at the beginning and/or at the end of strings

    :param list_of_strings: list of sentences to process
    :param method: 'Start' adds context only at the beginning, 'End' adds it at the end, 'Combined' adds context both
    at the beginning and at the end, 'Random' means method for each string is randomly assigned.
    :param starting_context: list of terms (context) to input at start of sentences.
    :param ending_context: list of terms (context) to input at end of sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """

    np.random.seed(7)

    outcome_list_of_strings = []

    for string in list_of_strings:

        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        if method == 'Start':
            outcome_list_of_strings.append(random.choice(starting_context) + ' ' + string)

        elif method == 'End':
            if string[-1].isalnum():
                outcome_list_of_strings.append(string + ' ' + random.choice(ending_context))

            else:
                outcome_list_of_strings.append(string[:-1] + ' ' + random.choice(ending_context))

        elif method == 'Combined':
            if string[-1].isalnum():
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


def create_terminology(df: DataFrame) -> Dict[str, List[str]]:
    """
    Iterate over the DataFrame to create terminology from the predictions. IOB format converted to the IO.

    :param df: annotation DataFrame got from SparkNLP NER Pipeline.
    :return: dictionary of entities and corresponding list of words.
    """

    chunk = None
    ent_type = None

    terminology = dict()
    for indx, row in df.iterrows():

        for prediction in row.ner:
            ent = prediction.result

            if ent.startswith('B'):

                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]

                chunk = [prediction.metadata['word']]
                ent_type = ent[2:]  #  drop B-

            elif ent.startswith('I'):

                chunk.append(prediction.metadata['word'])

            else:

                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]

                chunk = None
                ent_type = None

    return terminology


def swap_named_entities_from_terminology(
        list_of_strings: List[str],
        annotations: DataFrame,
        terminology: dict,
        noise_prob: float = 0.5
) -> List[str]:
    """
    This function swap named entities with the chosen same entity with same token count from the terminology.

    :param list_of_strings: List of sentences to process
    :param annotations: Corresponding NER results for given list_of_strings.
    :param terminology: Dictionary of entities and corresponding list of words.
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    :return:
    """
    outcome_list_of_strings = []
    for indx, string in enumerate(list_of_strings):

        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        ner_result = [token.result for token in annotations.iloc[indx].ner]
        ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in ner_result])
        ent_indx, = np.where(ent_start_pos == 1)

        #  if there is no entity in the sentence, skip
        if len(ent_indx) == 0:
            outcome_list_of_strings.append(string)
            continue

        replace_indx = np.random.choice(ent_indx)
        ent_type = ner_result[replace_indx][2:]
        replace_indxs = [replace_indx]

        if replace_indx < len(ner_result) - 1:
            for indx, label in enumerate(ner_result[replace_indx + 1:]):
                if label == f'I-{ent_type}':
                    replace_indxs.append(indx + replace_indx + 1)
                else:
                    break

        token_list = string.split(' ')
        replace_token = token_list[replace_indx: replace_indx + len(replace_indxs)]
        replace_token = " ".join(replace_token)

        ent_terminology = []
        for ent in terminology[ent_type]:
            if len(ent.split(' ')) == len(replace_indxs):
                ent_terminology.append(ent)

        if len(ent_terminology) > 0:
            chosen_ent = random.choice(ent_terminology)
            string = string.replace(replace_token, chosen_ent)
            outcome_list_of_strings.append(string)

        else:
            outcome_list_of_strings.append(string)

    return outcome_list_of_strings


def american_to_british(list_of_strings: List[str], lang_dict: Dict[str, str],
                        noise_prob: float = 0.5) -> List[str]:
    """Converts input sentences from american english to british english using a conversion dictionary

    :param list_of_strings: list of sentences to process
    :param lang_dict: dictionary with conversion terms
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """
    new_list = []

    for string in list_of_strings:

        if random.random() > noise_prob:
            new_list.append(string)
            continue

        for american_spelling, british_spelling in lang_dict.items():
            string = string.replace(american_spelling, british_spelling)

        new_list.append(string)

    return new_list


def british_to_american(list_of_strings: List[str], lang_dict: Dict[str, str],
                        noise_prob: float = 0.5) -> List[str]:
    """Converts input sentences from british english to american english using a conversion dictionary

    :param list_of_strings: list of sentences to process
    :param lang_dict: dictionary with conversion terms
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """
    new_list = []

    for string in list_of_strings:

        if random.random() > noise_prob:
            new_list.append(string)
            continue

        for american_spelling, british_spelling in lang_dict.items():
            string = string.replace(british_spelling, american_spelling)

        new_list.append(string)

    return new_list


def remove_punctuation_tokens(column: List[str]) -> List[str]:
    """Removes all punctuation tokens from input sentences

    :param column: list of sentences to process
    """
    outcome_list = []

    punc_list = ['!', '?', ',', '.', '-', ':', ';']

    for token_list in column:

        lst = []

        for token in token_list:
            if token.metadata['word'] not in punc_list:
                lst.append(token)

        outcome_list.append(lst)

    return outcome_list


def remove_context_tokens(column: List[str], starting_context_tokens: List[str],
                          ending_context_tokens: List[str]) -> List[str]:
    """Removes user-defined context tokens from strings

    :param column: list of sentences to process
    :param starting_context_tokens: list of starting context tokens to remove
    :param ending_context_tokens: list of ending context tokens to remove
    """

    def match_starting_context(token_list):

        for context_token in starting_context_tokens:
            length_context = len(context_token.split())
            token_string = " ".join([token.metadata['word'] for token in token_list[:length_context]])
            if token_string == context_token:
                return length_context

        return 0

    def match_ending_context(token_list):

        for context_token in ending_context_tokens:
            length_context = len(context_token.split())
            token_string = " ".join([token.metadata['word'] for token in token_list[-length_context:]])
            if token_string == context_token:
                return len(token_list) - length_context

        return len(token_list)

    outcome_list = []
    for token_list in column:
        starting_indx = match_starting_context(token_list)
        ending_indx = match_ending_context(token_list)
        outcome_list.append(token_list[starting_indx:ending_indx])

    return outcome_list


def remove_contraction_tokens(list_with_contractions: List[str], list_without_contractions: List[str]) -> None:
    """Removes contraction tokens

    :param list_with_contractions: list of sentences with contractions
    :param list_without_contractions: list of sentences without contractions
    """
    idx = []

    for i, element in enumerate(list_with_contractions):

        if list_with_contractions[i].metadata['word'] != list_without_contractions[i].metadata['word']:
            idx.append(i)

    if len(idx) > 0:
        contraction_idx = idx[0]

        del list_with_contractions[contraction_idx]

        del list_without_contractions[contraction_idx:contraction_idx + 2]


def get_cohyponyms_wordnet(word: str) -> str:
    """Retrieve co-hyponym of the input string using WordNet when a hit is found.

    :param word: input string to retrieve co-hyponym
    """
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


def swap_with_cohyponym(list_of_strings: List[str],
                        annotations: DataFrame,
                        noise_prob: float = 0.5
                        ) -> List[str]:
    """This function swap named entities with a co-hyponym from the WordNet database when a hit is found.

    :param list_of_strings: List of sentences to process
    :param annotations: Corresponding NER results for given list_of_strings.
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    """
    #  download WordNet DB
    print('\nDownloading WordNet database to execute co-hyponym swapping.\n')
    wn.download('ewn:2020')
    outcome_list_of_strings = []
    for indx, string in enumerate(list_of_strings):

        if random.random() > noise_prob:
            outcome_list_of_strings.append(string)
            continue

        ner_result = [token.result for token in annotations.iloc[indx].ner]
        ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in ner_result])
        ent_indx, = np.where(ent_start_pos == 1)

        #  if there is no entity in the sentence, skip
        if len(ent_indx) == 0:
            outcome_list_of_strings.append(string)
            continue

        replace_indx = np.random.choice(ent_indx)
        ent_type = ner_result[replace_indx][2:]
        replace_indxs = [replace_indx]

        if replace_indx < len(ner_result) - 1:
            for indx, label in enumerate(ner_result[replace_indx + 1:]):
                if label == f'I-{ent_type}':
                    replace_indxs.append(indx + replace_indx + 1)
                else:
                    break

        token_list = string.split(' ')
        replace_token = token_list[replace_indx: replace_indx + len(replace_indxs)]
        replace_token = " ".join(replace_token)

        #  replace by cohyponym
        chosen_ent = get_cohyponyms_wordnet(replace_token)
        string = string.replace(replace_token, chosen_ent)
        outcome_list_of_strings.append(string)

    return outcome_list_of_strings


def calculate_metrics(filtered_df: DataFrame, method: str = 'strict') -> Dict[str, Any]:
    """Calculates comparison metrics for robustness

    :param filtered_df: dataframe created during robustness tests
    :param method: 'strict' calculates metrics for IOB2 format, 'flex' calculates for IO format
    """
    comparison_df = pd.DataFrame()

    comparison_df['original_token'] = filtered_df['ner'].apply(lambda x: x.metadata['word'] if pd.notnull(x) else x)

    comparison_df['original_label'] = filtered_df['ner'].apply(lambda x: x.result if pd.notnull(x) else x)

    comparison_df['original_flex_label'] = filtered_df['ner'].apply(
        lambda x: x.result.split("-")[-1] if pd.notnull(x) else x)

    comparison_df['noisy_token'] = filtered_df['noisy_ner'].apply(lambda x: x.metadata['word'] if pd.notnull(x) else x)

    comparison_df['noisy_label'] = filtered_df['noisy_ner'].apply(lambda x: x.result if pd.notnull(x) else x)

    comparison_df['noisy_flex_label'] = filtered_df['noisy_ner'].apply(
        lambda x: x.result.split("-")[-1] if pd.notnull(x) else x)

    def round_metrics(m):
        for ent, metric_dict in m.items():
            if ent == 'accuracy':
                m['accuracy'] = round(m['accuracy'], 2)
                continue
            for k, v in metric_dict.items():
                if k != 'support':
                    m[ent][k] = round(v, 2)
        return m

    if method == 'strict':
        metrics = classification_report(comparison_df['original_label'].astype(str),
                                        comparison_df['noisy_label'].astype(str),
                                        output_dict=True)
        metrics = round_metrics(metrics)

    if method == 'flex':
        metrics = classification_report(comparison_df['original_flex_label'].astype(str),
                                        comparison_df['noisy_flex_label'].astype(str), output_dict=True)
        metrics = round_metrics(metrics)

    outcome = {'metrics': metrics, 'comparison_df': comparison_df}

    return outcome


def run_test(spark: SparkSession, noise_type: str, noise_description: str, pipeline_model: PipelineModel,
             test_set: List[str], total_amount: int, original_annotations_df: DataFrame, noisy_test_set: List[str],
             metric_type: str, token_filter_function: Optional[str] = None,
             starting_context_token_list: Optional[List[str]] = None,
             ending_context_token_list: Optional[List[str]] = None,
             ) -> Tuple[DataFrame, str, DataFrame]:
    """Runs comparisons between original list of sentences and noisy list of sentences, returning metrics and dataframe
    for comparison
    :param spark: An active Spark Session to create spark DataFrame
    :param noise_type: type of noise to introduce in sentences for running tests on 'modify_capitalization_upper',
    'modify_capitalization_lower', 'modify_capitalization_title', 'add_punctuation', 'strip_punctuation',
    'introduce_typos', 'add_contractions', 'add_context', 'american_to_british', 'british_to_american',
    'swap_entities', 'swap_cohyponyms'
    :param noise_description: description of the type of noise for user awareness
    :param pipeline_model: PipelineModel with document assembler, sentence detector, tokenizer, word embeddings if
    applicable, NER model with 'ner' output name, NER converter with 'ner_chunk' output name
    :param test_set: list of original sentences to process
    :param total_amount: length of the list of original sentences to process
    :param original_annotations_df: DataFrame containing LightPipeline model fullAnnotate results on original list of
    sentences
    :param noisy_test_set: list of sentences with perturbations
    :param metric_type: 'strict' calculates metrics for IOB2 format, 'flex' calculates for IO format
    :param metrics_output_format: 'dictionary' to get a dictionary report, 'dataframe' to get a dataframe report
    :param token_filter_function: function to filter tokens for appropriate comparison when applying perturbations
    which disrupt token match-up between original test set and noisy test set, options are None,
    'remove_context_tokens', 'remove_contraction_tokens', 'remove_punctuation_tokens'
    :param starting_context_token_list: list of starting context tokens to add when applying the 'add_context' noise type
    :param ending_context_token_list: list of ending context tokens to add when applying the 'add_context' noise type
    """
    report_text = '\n\n' + noise_type + '\nGenerated noise: ' + noise_description

    noisy_test_data = [[i] for i in noisy_test_set]
    noisy_test_data = spark.createDataFrame(noisy_test_data).toDF("text")
    noisy_annotations = pipeline_model.transform(noisy_test_data)
    noisy_annotations_df = noisy_annotations.select('ner').toPandas()

    noisy_lst = []
    for i in range(len(test_set)):
        noisy_lst.append(test_set[i] != noisy_test_set[i])

    report_text = report_text + '\nGenerated noise affects ' + str(sum(noisy_lst)) + ' sentences (' + str(
        (round(100 * (sum(noisy_lst) / total_amount), 2))) + '% of the test set).\n'

    if token_filter_function == 'remove_punctuation_tokens':

        noisy_annotations_df['ner'] = remove_punctuation_tokens(column=noisy_annotations_df['ner'])

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        reduced_original_annotations_df = original_annotations_df.copy()

        reduced_original_annotations_df['ner'] = remove_punctuation_tokens(
            column=reduced_original_annotations_df['ner'])

        reduced_original_annotations_df['token_count'] = reduced_original_annotations_df['ner'].apply(
            lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        joined_df = reduced_original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

    elif token_filter_function == 'remove_contraction_tokens':

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        joined_df = original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

        for index in range(len(joined_df)):
            remove_contraction_tokens(list_with_contractions=joined_df['noisy_ner'][index],
                                      list_without_contractions=joined_df['ner'][index])

        joined_df['token_count'] = joined_df['ner'].apply(lambda x: len(x))

        joined_df['noisy_token_count'] = joined_df['noisy_ner'].apply(lambda x: len(x))

    elif token_filter_function == 'remove_context_tokens':

        noisy_annotations_df['ner'] = remove_context_tokens(column=noisy_annotations_df['ner'],
                                                            starting_context_tokens=starting_context_token_list,
                                                            ending_context_tokens=ending_context_token_list)

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        reduced_original_annotations_df = original_annotations_df.copy()

        for indx, (noisy, original) in enumerate(zip(
                noisy_annotations_df['noisy_ner'], reduced_original_annotations_df['ner'])):
            if len(noisy) == len(original) - 1 and not original[-1].metadata['word'][-1].isalnum():
                reduced_original_annotations_df['ner'].iloc[indx] = original[:-1]

        reduced_original_annotations_df['token_count'] = reduced_original_annotations_df['ner'].apply(
            lambda x: len(x))

        joined_df = reduced_original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

    else:

        noisy_annotations_df['token_count'] = noisy_annotations_df['ner'].apply(lambda x: len(x))

        noisy_annotations_df = noisy_annotations_df.rename(
            columns={'ner': 'noisy_ner', 'sentence': 'noisy_sentence', 'token_count': 'noisy_token_count'})

        joined_df = original_annotations_df[['token_count', 'ner']].join(
            noisy_annotations_df[['noisy_token_count', 'noisy_ner']])

    filtered_df = joined_df[joined_df['token_count'] == joined_df['noisy_token_count']][['ner', 'noisy_ner']]

    filtered_df = filtered_df[
        filtered_df['ner'].apply(lambda x: len(x)) == filtered_df['noisy_ner'].apply(lambda x: len(x))]

    filtered_sentences = total_amount - len(filtered_df)

    report_text = report_text + '\nA total amount of ' + str(filtered_sentences) + \
                  " were filtered out due to mismatching tokenization (" + \
                  str((round(100 * (filtered_sentences / total_amount), 2))) + "% of the test set)."

    filtered_df = filtered_df.apply(pd.Series.explode).reset_index()

    test_outcomes = calculate_metrics(filtered_df=filtered_df, method=metric_type)

    test_metrics = test_outcomes['metrics']

    test_metrics['test'] = noise_type

    comparison_df = test_outcomes['comparison_df'][
        ['original_token', 'noisy_token', 'original_label', 'noisy_label']].copy()

    comparison_df['test'] = noise_type

    report_text = report_text + "\nf1 - macro average: " + str(
        round(test_metrics['macro avg']['f1-score'], 2)
    )

    report_text = report_text + "\nf1 - weighted average: " + str(
        round(test_metrics['weighted avg']['f1-score'], 2)
    )

    return test_metrics, report_text, comparison_df


def conll_sentence_reader(conll_path: str) -> List[str]:
    """
    Read CoNLL file and convert it to the list of sentences.
    :param conll_path: CoNLL file path.
    :return: list of sentences in the conll data.
    """
    with open(conll_path) as f:
        data = []
        content = f.read()
        docs = [i.strip() for i in content.strip().split('-DOCSTART- -X- -X- O') if i != '']
        for doc in docs:

            #  file content to sentence split
            sentences = doc.strip().split('\n\n')

            if sentences == ['']:
                data.append(([''], [''], ['']))
                continue

            for sent in sentences:
                sentence_data = []

                #  sentence string to token level split
                tokens = sent.strip().split('\n')

                # get annotations from token level split
                token_list = [t.split() for t in tokens]

                #  get token and labels from the split
                for split in token_list:
                    sentence_data.append(split[0])

                data.append(" ".join(sentence_data))

    return data


def test_robustness(spark: SparkSession, pipeline_model: PipelineModel, test_file_path: str,
                    metric_type: Optional[str] = 'flex',
                    metrics_output_format: str = 'dictionary',
                    log_path: str = 'robustness_test_results.json',
                    noise_prob: float = 0.5,
                    sample_sentence_count: int = None,
                    test: Optional[List[str]] = None,
                    starting_context: Optional[List[str]] = None,
                    ending_context: Optional[List[str]] = None) -> Dict[str, Any]:
    """Tests robustness of a NER model by applying different types of noise generating functions to a list of sentences.
    Metrics are calculated by comparing model's extractions in the original list of sentences set with the extractions
    done in the noisy list of sentences.

    :param spark: An active SparkSession to create Spark DataFrame
    :param pipeline_model: PipelineModel with document assembler, sentence detector, tokenizer, word embeddings if
    applicable, NER model with 'ner' output name, NER converter with 'ner_chunk' output name
    :param test_file_path: Path to test file to test robustness. Can be .txt or .conll file in CoNLL format or
    .csv file with just one column (text) with series of test samples.
    :param test: type of robustness test implemented by the function, options include
    'modify_capitalization_upper': capitalization of the test set is turned into uppercase
    'modify_capitalization_lower': capitalization of the test set is turned into lowercase
    'modify_capitalization_title': capitalization of the test set is turned into title case
    'add_punctuation': special characters at end of each sentence are replaced by other special characters, if no
    special character at the end, one is added
    'strip_punctuation': special characters are removed from the sentences (except if found in numbers, such as '2.5')
    'introduce_typos': typos are introduced in sentences
    'add_contractions': contractions are added where possible (e.g. 'do not' contracted into 'don't')
    'add_context': tokens are added at the beginning and at the end of the sentences.
    'swap_entities': named entities replaced with same entity type with same token count from terminology.
    'swap_cohyponyms': Named entities replaced with co-hyponym from the WordNet database.
    'american_to_british': American English will be changed to British English, test_set_language should be set to
    'American English' (default)
    'british_to_american': British English will be changed to American English, test_set_language should be set to
    'British English'
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
    'american_to_british' or 'british_to_american' will be run depending on test_set_language value
    (by default, 'american_to_british')
    :param sample_sentence_count: Number of sentence that will be sampled from the test_data.
    :param metric_type: 'strict' calculates metrics for IOB2 format, 'flex' (default) calculates for IO format
    :param metrics_output_format: 'dictionary' to get a dictionary report, 'dataframe' to get a dataframe report
    :param log_path: Path to log file, False to avoid saving test results. Default 'robustness_test_results.json'
    :param starting_context: list of context tokens to add to beginning when running the 'add_context' test
    :param ending_context: list of context tokens to add to end when running the 'add_context' test
    """

    outcome = {}
    perturb_metrics = dict()
    complete_comparison_df = pd.DataFrame(
        columns=['original_token', 'noisy_token', 'original_label', 'noisy_label', 'test'])

    lang_dict = _A2B_DICT

    if test_file_path.endswith('.txt') or test_file_path.endswith('.conll'):
        test_set = conll_sentence_reader(test_file_path)
    elif test_file_path.endswith('.csv'):
        test_df = pd.read_csv(test_file_path)
        test_set = test_df['text'].tolist()
    else:
        raise ValueError("'test_file_path' could not be read! It should be txt, conll or csv file.")

    test_set = [sent for sent in test_set if not len(sent.split()) < 2]
    if sample_sentence_count:
        if sample_sentence_count < len(test_set):
            test_set = random.sample(test_set, sample_sentence_count)
        else:
            raise ValueError(
                f"sample_sentence_count ({sample_sentence_count}) must be greater than test_set length ({len(test_set)}).")
    total_amount = len(test_set)

    if test is None:
        test = ['modify_capitalization_upper', 'modify_capitalization_lower', 'modify_capitalization_title',
                'add_punctuation', 'strip_punctuation', 'introduce_typos', 'add_contractions', 'add_context',
                'american_to_british', 'swap_entities', 'swap_cohyponyms']

    report_text = 'Test type: ' + ', '.join(test) + '\nTest set size: ' + str(total_amount) + ' sentences\n'

    test_data = [[i] for i in test_set]
    test_data = spark.createDataFrame(test_data).toDF('text')

    original_annotations = pipeline_model.transform(test_data)
    original_annotations_df = original_annotations.select('ner').toPandas()

    original_annotations_df['token_count'] = original_annotations_df['ner'].apply(lambda x: len(x))

    if 'modify_capitalization_upper' in test:
        noise_type = 'modify_capitalization_upper'

        noise_description = 'text capitalization turned into uppercase'

        noisy_test_set = modify_capitalization(list_of_strings=test_set, method='Uppercase', noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(
            spark=spark, noise_type=noise_type, noise_description=noise_description,
            pipeline_model=pipeline_model, test_set=test_set,
            total_amount=total_amount,
            original_annotations_df=original_annotations_df,
            noisy_test_set=noisy_test_set, metric_type=metric_type,
            token_filter_function=None
        )

        perturb_metrics['modify_capitalization_upper'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'modify_capitalization_lower' in test:
        noise_type = 'modify_capitalization_lower'

        noise_description = 'text capitalization turned into lowercase'

        noisy_test_set = modify_capitalization(list_of_strings=test_set, method='Lowercase', noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function=None)

        perturb_metrics['modify_capitalization_lower'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'modify_capitalization_title' in test:
        noise_type = 'modify_capitalization_title'

        noise_description = 'text capitalization turned into title type (first letter capital)'

        noisy_test_set = modify_capitalization(list_of_strings=test_set, method='Title', noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function=None)

        perturb_metrics['modify_capitalization_title'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'add_punctuation' in test:
        noise_type = 'add_punctuation'

        noise_description = 'special character at the end of the sentence is modified'

        noisy_test_set = add_punctuation(list_of_strings=test_set,
                                         noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function='remove_punctuation_tokens')

        perturb_metrics['add_punctuation'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if "swap_cohyponyms" in test:
        noise_type = 'swap_cohyponyms'

        noise_description = 'named entities in the sentences are replaced with same entity type'

        noisy_test_set = swap_with_cohyponym(test_set, original_annotations_df, noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type)

        perturb_metrics['swap_cohyponyms'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if "swap_entities" in test:
        terminology = create_terminology(original_annotations_df)

        noise_type = 'swap_entities'

        noise_description = 'named entities in the sentences are replaced with same entity type'

        noisy_test_set = swap_named_entities_from_terminology(test_set, original_annotations_df, terminology,
                                                              noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type)

        perturb_metrics['swap_entities'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'strip_punctuation' in test:
        noise_type = 'strip_punctuation'

        noise_description = 'special character at the end of the sentence is removed'

        noisy_test_set = strip_punctuation(list_of_strings=test_set,
                                           keep_numeric_punctuation=True,
                                           noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function='remove_punctuation_tokens')

        perturb_metrics['strip_punctuation'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'introduce_typos' in test:
        noise_type = 'introduce_typos'

        noise_description = 'typos introduced in sentences'

        noisy_test_set = introduce_typos(test_set, noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function=None)

        perturb_metrics['introduce_typos'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'add_contractions' in test:
        noise_type = 'add_contractions'

        noise_description = 'contractions added in sentences'

        noisy_test_set = add_contractions(test_set, noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function='remove_contraction_tokens')

        perturb_metrics['add_contractions'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'add_context' in test:

        if starting_context is None:
            starting_context = ["Description:", "MEDICAL HISTORY:", "FINDINGS:", "RESULTS: ",
                                "Report: ", "Conclusion is that "]

        if ending_context is None:
            ending_context = ["according to the patient's family", "as stated by the patient",
                              "due to various circumstances", "confirmed by px"]

        starting_context_token_list = [[i] for i in starting_context]
        starting_context_token_list = spark.createDataFrame(starting_context_token_list).toDF('text')
        starting_context_token_list = pipeline_model.transform(starting_context_token_list)

        starting_context_tokens = []
        for context_token in starting_context_token_list.select('token').collect():
            starting_context_tokens.append(" ".join([token.result for token in context_token[0]]))

        ending_context_token_list = [[i] for i in ending_context]
        ending_context_token_list = spark.createDataFrame(ending_context_token_list).toDF('text')
        ending_context_token_list = pipeline_model.transform(ending_context_token_list)

        ending_context_tokens = []
        for context_token in ending_context_token_list.select('token').collect():
            ending_context_tokens.append(" ".join([token.result for token in context_token[0]]))

        noise_type = 'add_context'

        noise_description = 'words added at the beginning and end of sentences'

        noisy_test_set = add_context(test_set, method='Combined', starting_context=starting_context_tokens,
                                     ending_context=ending_context_tokens,
                                     noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function='remove_context_tokens',
                                                     starting_context_token_list=starting_context_tokens,
                                                     ending_context_token_list=ending_context_tokens
                                                     )

        perturb_metrics['add_context'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'american_to_british' in test:
        noise_type = 'american_to_british'

        noise_description = 'American spelling turned into British spelling'

        noisy_test_set = american_to_british(test_set, lang_dict, noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function=None)

        perturb_metrics['american_to_british'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    if 'british_to_american' in test:
        noise_type = 'british_to_american'

        noise_description = 'British spelling turned into American spelling'

        noisy_test_set = british_to_american(test_set, lang_dict, noise_prob=noise_prob)

        test_metrics, text, comparison_df = run_test(spark=spark, noise_type=noise_type,
                                                     noise_description=noise_description,
                                                     pipeline_model=pipeline_model, test_set=test_set,
                                                     total_amount=total_amount,
                                                     original_annotations_df=original_annotations_df,
                                                     noisy_test_set=noisy_test_set, metric_type=metric_type,
                                                     token_filter_function=None)

        perturb_metrics['british_to_american'] = test_metrics
        report_text = report_text + text
        complete_comparison_df = pd.concat([complete_comparison_df, comparison_df]).reset_index(drop=True)

    outcome['metrics'] = perturb_metrics
    outcome['comparison_df'] = complete_comparison_df
    outcome['test_details'] = report_text

    if log_path:
        test_results = outcome.copy()
        complete_comparison = dict()
        groups = outcome['comparison_df'].groupby('test')
        for name, group in groups:
            group = group.drop('test', axis=1)
            complete_comparison[name] = group.to_dict('list')
        test_results['comparison_df'] = complete_comparison

        try:
            with open(log_path, 'w') as f:
                try:
                    f.write(json.dumps(test_results))

                except (IOError, OSError) as e:
                    print(f"Error while writing to the {log_path}. Log file will not be written.")
                    print(e)

        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error while opening the {log_path}. Log file will be ignored.")
            print(e)

    if metrics_output_format == 'dataframe':
        metrics_df = pd.DataFrame(columns=['entity', 'precision', 'recall', 'f1-score', 'support', 'test'])

        for perturb_type, metrics in outcome['metrics'].items():
            metrics = pd.DataFrame.from_dict(metrics) \
                .transpose() \
                .reset_index() \
                .rename({'index': 'entity'},
                        axis='columns')
            metrics['test'] = perturb_type
            metrics_df = pd.concat([metrics_df, metrics]).reset_index(drop=True)

        outcome['metrics'] = metrics_df

    return outcome
