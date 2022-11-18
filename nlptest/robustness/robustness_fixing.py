"""Functions to fix robustness of NER model with different kinds of perturbations in CoNLL data"""
import re
import random
import logging
import itertools
import numpy as np

from ...utils.imports import is_module_importable

if not is_module_importable('wn'):
    raise ImportError(f'Please run <pip install wn> to use this module.  ')

import wn
from .robustness_testing import test_robustness
from pyspark.sql import SparkSession, DataFrame
from sparknlp.base import PipelineModel
from typing import Optional, List, Dict, Tuple, Any
from .utils import (
    _DF_SCHEMA,
    _CONTRACTION_MAP,
    _A2B_DICT,
    _TYPO_FREQUENCY,
    suggest_perturbations,
    get_augmentation_proportions
)


def create_dataframe(
        spark: SparkSession,
        data: List[str],
        pos_sync_tags: List[str],
        labels: List[str]
) -> DataFrame:
    """
    Create Spark Dataframe from given data and labels extracted from the CoNLL file.
    :param spark: An active spark session to create the dataframe.
    :param data: List of sentences.
    :param pos_sync_tags: List of pos and synthetic tags seperated by '-' and whitespace.
    :param labels: Corresponding NER labels of the data.
    :return: Spark DataFrame consisting of 'text', 'document', 'sentence', 'token', 'label', 'pos' columns.
    """
    from pyspark.sql.types import Row

    spark_data = []
    for sentence, sentence_tags, sentence_label in zip(data, pos_sync_tags, labels):

        token_rows = []
        label_rows = []
        tags_rows = []

        begin = end = 0
        for token, tag, label in zip(sentence.split(), sentence_tags.split(), sentence_label.split()):
            end += len(token) - 1

            token_rows.append(Row(
                annotatorType='token',
                begin=begin,
                end=end,
                result=token,
                metadata={'sentence': '0'},
                embeddings=[]
            )),

            label_rows.append(Row(
                annotatorType='named_entity',
                begin=begin,
                end=end,
                result=label,
                metadata={'sentence': '0', 'word': token},
                embeddings=[]
            ))

            tags_rows.append(Row(
                annotatorType='pos',
                begin=begin,
                end=end,
                result=tag.split('*')[0],
                metadata={'sentence': '0', 'word': token},
                embeddings=[]
            ))

            end = begin = end + 2

        spark_data.append(
            {
                'text': sentence,
                'document': [Row(
                    annotatorType='document',
                    begin=0,
                    end=len(sentence) - 1,
                    result=sentence,
                    metadata={'training': 'true'},
                    embeddings=[]
                )],
                'sentence': [Row(
                    annotatorType='document',
                    begin=0,
                    end=len(sentence) - 1,
                    result=sentence,
                    metadata={'sentence': '0'},
                    embeddings=[]
                )],
                'pos': tags_rows,
                'token': token_rows,
                'label': label_rows
            }
        )

    return spark.createDataFrame(spark_data, _DF_SCHEMA)


def conll_reader(conll_path: str) -> List[tuple]:
    """
    Read CoNLL file and convert it to the list of labels and sentences.
    :param conll_path: CoNLL file path.
    :return: data and labels which have sentences and labels joined with the single space.
    """
    with open(conll_path) as f:
        data = []
        content = f.read()
        docs = [i.strip() for i in content.strip().split('-DOCSTART- -X- -X- O') if i != '']
        for doc in docs:
            doc_sent = []
            pos_sync_tag = []
            labels = []

            #  file content to sentence split
            sentences = doc.strip().split('\n\n')

            if sentences == ['']:
                data.append(([''], [''], ['']))
                continue

            for sent in sentences:
                sentence_data = []
                sentence_labels = []
                sentence_tags = []

                #  sentence string to token level split
                tokens = sent.strip().split('\n')

                # get annotations from token level split
                token_list = [t.split() for t in tokens]

                #  get token and labels from the split
                for split in token_list:
                    sentence_data.append(split[0])
                    sentence_labels.append(split[-1])
                    sentence_tags.append("*".join([split[1], split[2]]))

                doc_sent.append(" ".join(sentence_data))
                labels.append(" ".join(sentence_labels))
                pos_sync_tag.append(" ".join(sentence_tags))

            data.append((doc_sent, pos_sync_tag, labels))

    return data


def filter_by_entity_type(data: List[str], pos_sync_tag: List[str], labels: List[str], ent_type: str) \
        -> (List[str], List[str], List[str]):
    """
    A function to filter data by the entity type.
    :param data: List of sentences to be filtered by the entity type.
    :param pos_sync_tag: List of tags extracted from CoNLL file.
    :param labels: List of labels to be filtered by the entity type.
    :param ent_type: Entity type without IOB2 suffix to filter the data and the labels.
    :return: Filtered data and labels by the entity type.
    """
    filtered_labels = []
    filtered_data = []
    filtered_tags = []
    for indx, label in enumerate(labels):
        if 'B-' + ent_type in label:
            filtered_labels.append(label)
            filtered_data.append(data[indx])
            filtered_tags.append(pos_sync_tag[indx])

    assert len(filtered_data) == len(filtered_labels) == len(filtered_tags), \
        "Length of the labels and the data should be the same."
    return filtered_data, filtered_tags, filtered_labels


def get_sample(k: int, *args: List[Any]) -> List[List[str], ]:
    """
    print(sample_data[0
    print(ent_type, sample_data[0
    A function to get sample from passed lists.
    :param k: Number of sample for each list.
    :param args: List to get sample. Every list passed should be exactly same length.
    :return: Sampled lists.
    """
    random_indxs = None
    filtered_arrays = []
    for arr in args:

        if random_indxs is None:

            #  Get first sample from the input and create indx for the given sample
            arr_indx = random.sample(list(enumerate(arr)), k=k)

            arr = []
            random_indxs = []
            for indx, elem in arr_indx:
                arr.append(elem)
                random_indxs.append(indx)
            filtered_arrays.append(arr)

        else:
            filtered_arr = [arr[indx] for indx in random_indxs]
            filtered_arrays.append(filtered_arr)

    return filtered_arrays


def create_terminology(data: List[str], labels: List[str]):
    """
    Iterate over the DataFrame to create terminology from the predictions. IOB format converted to the IO.

    :param data: list of sentences to process
    :param labels: corresponding labels to make changes according to data.
    :return: dictionary of entities and corresponding list of words.
    """
    terminology = {}

    chunk = None
    ent_type = None
    for sent_indx, sent_labels in enumerate(labels):

        sent_labels = sent_labels.split(' ')
        for token_indx, label in enumerate(sent_labels):

            if label.startswith('B'):

                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]

                sent_tokens = data[sent_indx].split(' ')
                chunk = [sent_tokens[token_indx]]
                ent_type = label[2:]

            elif label.startswith('I'):

                sent_tokens = data[sent_indx].split(' ')
                chunk.append(sent_tokens[token_indx])

            else:

                if chunk:
                    if terminology.get(ent_type, None):
                        terminology[ent_type].append(" ".join(chunk))
                    else:
                        terminology[ent_type] = [" ".join(chunk)]

                chunk = None
                ent_type = None

    return terminology


def modify_capitalization_upper(
        data: List[str],
        tags: List[str],
        labels: List[str],
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Convert every sentence in the data by uppercase.
    :param data: list of sentences to process
    :param tags: corresponding tags to make changes according to data.
    :param labels: corresponding labels to make changes according to data.
    :return: List of augmented sentences with uppercase
    """

    aug_data = []
    aug_tags = []
    aug_labels = []
    aug_indx = []

    for i, sent in enumerate(data):

        upper_sent = sent.upper()
        if sent != sent.upper():
            aug_data.append(upper_sent)
            aug_tags.append(tags[i])
            aug_labels.append(labels[i])
            aug_indx.append(i)

    return aug_data, aug_tags, aug_labels, aug_indx


def modify_capitalization_lower(
        data: List[str],
        tags: List[str],
        labels: List[str]
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Convert every sentence in the data by lowercase.
    :param data: list of sentences to process
    :param tags: corresponding tags to make changes according to data.
    :param labels: corresponding labels to make changes according to data.
    :return: List of augmented sentences with lowercase
    """

    aug_data = []
    aug_tags = []
    aug_labels = []
    aug_indx = []

    for i, sent in enumerate(data):

        sent_lower = sent.lower()
        if sent_lower != sent:
            aug_data.append(sent_lower)
            aug_tags.append(tags[i])
            aug_labels.append(labels[i])
            aug_indx.append(i)

    return aug_data, aug_tags, aug_labels, aug_indx


def modify_capitalization_title(
        data: List[str],
        tags: List[str],
        labels: List[str]
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Convert every sentence in the data by title case.
    :param data: list of sentences to process
    :param tags: corresponding tags to make changes according to data.
    :param labels: corresponding labels to make changes according to data.
    :return: List of augmented sentences with title case
    """

    aug_data = []
    aug_tags = []
    aug_labels = []
    aug_indx = []

    for i, sent in enumerate(data):

        title_sent = sent.title()
        if title_sent != sent:
            aug_data.append(title_sent)
            aug_tags.append(tags[i])
            aug_labels.append(labels[i])
            aug_indx.append(i)

    return aug_data, aug_tags, aug_labels, aug_indx


def add_punctuation_to_data(
        data: List[str],
        tags: List[str],
        labels: List[str]
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Adds a punctuation at the end of strings.
    :param data: list of sentences to process
    :param tags: corresponding tags to make changes according to data.
    :param labels: corresponding labels to make changes according to data.
    :return: List of augmented sentences with punctuation added.
    """

    list_of_characters = ['!', '?', ',', '.', '-', ':', ';']

    punc_sentences = []
    punc_labels = []
    punc_tags = []
    punc_indx = []
    for indx, sentence in enumerate(data):

        if sentence[-1].isalnum():
            punc_sentences.append(sentence + " " + random.choice(list_of_characters))
            punc_tags.append(tags[indx] + ' NN*NN')
            punc_labels.append(labels[indx] + ' O')
            punc_indx.append(indx)

    return punc_sentences, punc_tags, punc_labels, punc_indx


def strip_punctuation_from_data(
        data: List[str],
        tags: List[str],
        labels: List[str]
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    Strips punctuations from the senctences.
    :param data: list of sentences to process
    :param tags: corresponding tags to make changes according to data.
    :param labels: corresponding labels to make changes according to data.List[str]
    """

    punc_sentences = []
    punc_labels = []
    punc_tags = []
    punc_indx = []
    for indx, sentence in enumerate(data):

        if not sentence[-1].isalnum():
            punc_sentences.append(" ".join(sentence.split()[:-1]))
            punc_tags.append(" ".join(tags[indx].split()[:-1]))
            punc_labels.append(" ".join(labels[indx].split()[:-1]))
            punc_indx.append(indx)

    return punc_sentences, punc_tags, punc_labels, punc_indx


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


def introduce_typos(
        data: List[str],
        tags: List[str],
        labels: List[str]
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Introduces typos in input sentences
    :param data: list of sentences to process
    :param tags: corresponding tags of sentences to align with.
    :param labels: corresponding labels of sentences to align with.
    :return proceed data with random typos
    """

    outcome_data = []
    outcome_labels = []
    outcome_tags = []
    outcome_indx = []
    for indx, sentence in enumerate(data):
        typo_sent = add_typo_to_sentence(sentence, _TYPO_FREQUENCY)
        if len(typo_sent.split()) == len(sentence.split()):
            outcome_data.append(typo_sent)
            outcome_tags.append(tags[indx])
            outcome_labels.append(labels[indx])
            outcome_indx.append(indx)

    return outcome_data, outcome_tags, outcome_labels, outcome_indx


def swap_entities_with_terminology(
        data: List[str],
        pos_sync_tag,
        labels: List[str],
        terminology: dict
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """
    :param data: list of sentences to process
    :param pos_sync_tag: corresponding tags of sentences to align with.
    :param labels: corresponding labels of sentences to align with.
    :param terminology: dictionary of entities and corresponding list of words
    :return: given data, tags and labels after entity swapping
    """

    output_data = []
    output_tags = []
    output_labels = []
    output_indx = []
    for sent_indx, string in enumerate(data):

        sent_tokens = string.split(' ')
        sent_tags = pos_sync_tag[sent_indx].split(' ')
        sent_labels = labels[sent_indx].split(' ')

        ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sent_labels])
        ent_indx, = np.where(ent_start_pos == 1)

        #  if there is no named entity in the sentence, skip
        if len(ent_indx) == 0:
            continue

        replace_indx = np.random.choice(ent_indx)
        ent_type = sent_labels[replace_indx][2:]
        replace_indxs = [replace_indx]
        if replace_indx < len(sent_labels) - 1:
            for indx, label in enumerate(sent_labels[replace_indx + 1:]):
                if label == f'I-{ent_type}':
                    replace_indxs.append(indx + replace_indx + 1)
                else:
                    break

        replace_token = sent_tokens[replace_indx: replace_indx + len(replace_indxs)]
        replace_token = " ".join(replace_token)

        chosen_ent = random.choice(terminology[ent_type])
        chosen_ent_length = len(chosen_ent.split(' '))
        replaced_string = string.replace(replace_token, chosen_ent)

        new_tags = ["-X-*-X-"] * chosen_ent_length
        new_tags = sent_tags[:replace_indx] + new_tags + sent_tags[replace_indx + len(replace_indxs):]

        new_labels = ['B-' + ent_type] + ['I-' + ent_type] * (chosen_ent_length - 1)
        new_labels = sent_labels[:replace_indx] + new_labels + sent_labels[replace_indx + len(replace_indxs):]

        output_data.append(replaced_string)
        output_tags.append(" ".join(new_tags))
        output_labels.append(" ".join(new_labels))
        output_indx.append(sent_indx)

    return output_data, output_tags, output_labels, output_indx


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


def swap_with_cohyponym(
        data: List[str],
        pos_sync_tag: List[str],
        labels: List[str],
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """This function swap named entities with a co-hyponym from the WordNet database when a hit is found.

    :param data: list of sentences to process
    :param pos_sync_tag: corresponding tags of sentences to align with.
    :param labels: corresponding labels of sentences to align with.
    """
    #  download WordNet DB
    print('\nDownloading WordNet database to execute co-hyponym swapping.\n')
    wn.download('ewn:2020')
    output_data = []
    output_tags = []
    output_labels = []
    output_indx = []
    for sent_indx, string in enumerate(data):

        sent_tokens = string.split(' ')
        sent_tags = pos_sync_tag[sent_indx].split(' ')
        sent_labels = labels[sent_indx].split(' ')

        ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sent_labels])
        ent_indx, = np.where(ent_start_pos == 1)

        #  if there is no named entity in the sentence, skip
        if len(ent_indx) == 0:
            continue

        replace_indx = np.random.choice(ent_indx)
        ent_type = sent_labels[replace_indx][2:]
        replace_indxs = [replace_indx]
        if replace_indx < len(sent_labels) - 1:
            for indx, label in enumerate(sent_labels[replace_indx + 1:]):
                if label == f'I-{ent_type}':
                    replace_indxs.append(indx + replace_indx + 1)
                else:
                    break

        replace_token = sent_tokens[replace_indx: replace_indx + len(replace_indxs)]
        replace_token = " ".join(replace_token)

        #  replace by cohyponym
        chosen_ent = get_cohyponyms_wordnet(replace_token)
        chosen_ent_length = len(chosen_ent.split(' '))
        replaced_string = string.replace(replace_token, chosen_ent)

        new_tags = ["-X-*-X-"] * chosen_ent_length
        new_tags = sent_tags[:replace_indx] + new_tags + sent_tags[replace_indx + len(replace_indxs):]

        new_labels = ['B-' + ent_type] + ['I-' + ent_type] * (chosen_ent_length - 1)
        new_labels = sent_labels[:replace_indx] + new_labels + sent_labels[replace_indx + len(replace_indxs):]

        output_data.append(replaced_string)
        output_tags.append(" ".join(new_tags))
        output_labels.append(" ".join(new_labels))
        output_indx.append(sent_indx)

    return output_data, output_tags, output_labels, output_indx


def convert_accent(
        data: List[str],
        tags: List[str],
        labels: List[str],
        lang_dict: Dict[str, str]
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Converts input sentences using a conversion dictionary
    :param data: list of sentences to process
    :param lang_dict: dictionary with conversion terms
    :param tags: corresponding tags of sentences to align with.
    :param labels: corresponding labels of sentences to align with.
    """

    accent_data = []
    accent_tags = []
    accent_labels = []
    accent_indx = []
    for indx, sentence in enumerate(data):

        old_sentence = sentence
        for token in sentence.split():
            if lang_dict.get(token, None):
                sentence = sentence.replace(token, lang_dict[token])

        if old_sentence != sentence:
            accent_data.append(sentence)
            accent_tags.append(tags[indx])
            accent_labels.append(labels[indx])
            accent_indx.append(indx)

    return accent_data, accent_tags, accent_labels, accent_indx


def add_context_to_data(
        data: List[str],
        tags: List[str],
        labels: List[str],
        regex_pattern: str = r"\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])",
        starting_context: Optional[List[str]] = None,
        ending_context: Optional[List[str]] = None
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Adds tokens at the beginning and/or at the end of strings
    :param data: list of sentences to process.
    :param tags: corresponding tags of sentences to align with.
    :param labels: corresponding labels of sentences to align with.
    :param regex_pattern: Regex pattern to split string into tokens.
    :param method: 'Start' adds context only at the beginning, 'End' adds it at the end, 'Combined' adds context both
    at the beginning and at the end, 'Random' means method for each string is randomly assigned.
    :param starting_context: list of terms (context) to input at start of sentences.
    :param ending_context: list of terms (context) to input at end of sentences.
        Ending context should not be ending with punctuation.
        It will be added if normal sentence is added with punctuation.
    """

    if starting_context is None:
        starting_context = ["Description:", "MEDICAL HISTORY:", "FINDINGS:", "RESULTS: ",
                            "Report: ", "Conclusion is that "]
    if ending_context is None:
        ending_context = ["according to the patient's family", "as stated by the patient",
                          "due to various circumstances", "confirmed by px"]

    starting_context = [re.split(regex_pattern, i) for i in starting_context if i != '']
    ending_context = [re.split(regex_pattern, i) for i in ending_context if i != '']

    context_added_sentences = []
    context_added_tags = []
    context_added_labels = []
    context_added_indx = []
    for indx, sentence in enumerate(data):

        method = random.choice(['start', 'end', 'combined'])

        if method == 'start':
            #   choose randomly from list of starting context
            add_tokens = random.choice(starting_context)

            #   join tokens
            add_labels = " ".join(['O'] * len(add_tokens))
            add_tags = " ".join(['NN*NN'] * len(add_tokens))
            add_string = " ".join(add_tokens)

            context_added_sentences.append(add_string + ' ' + sentence)
            context_added_tags.append(add_tags + ' ' + tags[indx])
            context_added_labels.append(add_labels + ' ' + labels[indx])
            context_added_indx.append(indx)

        elif method == 'end':
            #   choose randomly from list of ending context
            add_tokens = random.choice(ending_context)

            #   join tokens
            add_labels = " ".join(['O'] * len(add_tokens))
            add_tags = " ".join(['NN*NN'] * len(add_tokens))
            add_string = " ".join(add_tokens)

            if sentence[-1].isalnum():
                context_added_sentences.append(sentence + ' ' + add_string)
                context_added_tags.append(tags[indx] + ' ' + add_tags)
                context_added_labels.append(labels[indx] + ' ' + add_labels)
                context_added_indx.append(indx)

            else:
                context_added_sentences.append(sentence[:-1] + add_string + " " + sentence[-1])
                tags_splitted = tags[indx].split()
                context_added_tags.append(" ".join(tags_splitted[:-1]) + ' ' + add_tags + " " + tags_splitted[-1])
                context_added_labels.append(labels[indx][:-1] + add_labels + " O")
                context_added_indx.append(indx)

        elif method == 'combined':

            #   choose randomly from list of starting and ending context
            add_ending_tokens = random.choice(ending_context)
            add_starting_tokens = random.choice(starting_context)

            #   join starting tokens
            add_starting_labels = " ".join(['O'] * len(add_starting_tokens))
            add_starting_tags = " ".join(['NN*NN'] * len(add_starting_tokens))
            add_starting_string = " ".join(add_starting_tokens)

            #   join ending tokens
            add_ending_labels = " ".join(['O'] * len(add_ending_tokens))
            add_ending_tags = " ".join(['NN*NN'] * len(add_ending_tokens))
            add_ending_string = " ".join(add_ending_tokens)

            if sentence[-1].isalnum():
                context_added_sentences.append(sentence + ' ' + add_ending_string)
                context_added_tags.append(tags[indx] + ' ' + add_ending_tags)
                context_added_labels.append(labels[indx] + ' ' + add_ending_labels)
                context_added_indx.append(indx)

            else:
                #   add context to sentence
                context_added_sentences.append(add_starting_string + " " + sentence[:-1] +
                                               add_ending_string + " " + sentence[-1])
                #   align tags
                tags_splitted = tags[indx].split()
                context_added_tags.append(add_starting_tags + " " + " ".join(tags_splitted[:-1]) +
                                          ' ' + add_ending_tags + " " + tags_splitted[-1])

                #   align labels
                context_added_labels.append(add_starting_labels + " " + labels[indx][:-1] + add_ending_labels + " O")
                context_added_indx.append(indx)

    return context_added_sentences, context_added_tags, context_added_labels, context_added_indx


def add_contractions(
        sentences: List[str],
        tags: List[str],
        labels: List[str],
        regex_pattern: str = r"\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])"
) -> Tuple[List[str], List[str], List[str], List[int]]:
    """Adds contractions in input sentences
    :param sentences: list of sentences to contract.
    :param tags: corresponding tags of sentences to align with.
    :param labels: corresponding labels of sentences to align with.
    :param regex_pattern: regex pattern to split sentence into tokens.
    """
    out_sentences = []
    out_tags = []
    out_labels = []
    out_indx = []
    for sent_indx, sentence in enumerate(sentences):

        sent_tokens = []
        sent_tags = []
        sent_labels = []

        #   iterating token by token
        tokens = sentence.split()
        tags_split = tags[sent_indx].split()
        label_split = labels[sent_indx].split()

        is_continue = False
        is_contracted = False
        for indx, token in enumerate(tokens):

            #   in order to avoid index out of range error.
            if indx == len(tokens) - 1:
                sent_tokens.append(token)
                sent_tags.append(tags_split[indx])
                sent_labels.append(label_split[indx])

            elif is_continue:
                is_continue = False
                continue

            else:

                next_token = tokens[indx + 1]
                if _CONTRACTION_MAP.get(f"{token} {next_token}", None):

                    is_contracted = True

                    #   do contraction
                    contracted_token = re.split(regex_pattern, _CONTRACTION_MAP[f"{token} {next_token}"])

                    sent_tokens.append(" ".join(contracted_token))

                    #   contraction tokens could be inside entity in some cases.
                    if label_split[indx][0] == 'B':
                        sent_tags.append(tags_split[indx])
                        sent_labels.append(label_split[indx])

                        #   in order to avoid empty append to the list.
                        if len(contracted_token) > 1:
                            sent_tags.append(" ".join([tags_split[indx][1:]] * (len(contracted_token) - 1)))
                            sent_labels.append(" ".join(['I' + label_split[indx][1:]] * (len(contracted_token) - 1)))

                    #   if label is O, extend label and tags with the number of extra tokens in contraction
                    else:
                        sent_tags.append(" ".join([tags_split[indx]] * len(contracted_token)))
                        sent_labels.append(" ".join([label_split[indx]] * len(contracted_token)))

                    #   next token is also added. pass
                    is_continue = True

                else:
                    sent_tokens.append(token)
                    sent_tags.append(tags_split[indx])
                    sent_labels.append(label_split[indx])

        if is_contracted:
            out_sentences.append(" ".join(sent_tokens))
            out_tags.append(" ".join(sent_tags))
            out_labels.append(" ".join(sent_labels))
            out_indx.append(sent_indx)

    return out_sentences, out_tags, out_labels, out_indx


def augment_robustness(
        conll_path: str,
        uppercase: Optional[Dict[str, float]] = None,
        lowercase: Optional[Dict[str, float]] = None,
        title: Optional[Dict[str, float]] = None,
        add_punctuation: Optional[Dict[str, float]] = None,
        strip_punctuation: Optional[Dict[str, float]] = None,
        make_typos: Optional[Dict[str, float]] = None,
        american_to_british: Optional[Dict[str, float]] = None,
        british_to_american: Optional[Dict[str, float]] = None,
        add_context: Optional[Dict[str, float]] = None,
        contractions: Optional[Dict[str, float]] = None,
        swap_entities: Optional[Dict[str, float]] = None,
        swap_cohyponyms: Optional[Dict[str, float]] = None,
        optimized_inplace: bool = False,
        random_state: int = None,
        return_spark: bool = False,
        ending_context: List[str] = None,
        starting_context: List[str] = None,
        regex_pattern: str = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])",
        spark: SparkSession = None,
        conll_save_path: str = None,
        print_info: bool = False,
        ignore_warnings: bool = False
):
    """
    This function augments the training set by generating noisy text.
    The resulting dataset includes both the original samples and the noisy ones.
    Noise is applied at the beginning of the training, and it can also be applied after each epoch.

    :param conll_path: CoNLL file path to augment the dataset with given perturbations
    :param uppercase: A dictionary of entities and desired proportions to augment with upper case.
    All tokens in the sentence will be augmented.
    :param lowercase: A dictionary of entities and desired proportions to augment with lower case.
    All tokens in the sentence will be augmented.
    :param title: A dictionary of entities and desired proportions to augment with title case.
    All tokens in the sentence will be augmented.
    :param add_punctuation: A dictionary of entities and desired proportions
    to add or replace punctuation at the end of the sentence.
    :param strip_punctuation: A dictionary of entities and desired proportions
    to strip punctuation from the sentence.
    :param make_typos: A dictionary of entities and desired proportions to make typos in the dataset.
    :param american_to_british: A dictionary of entities and desired proportions
    to convert data to british spelling from american spelling.
    :param british_to_american: A dictionary of entities and desired proportions
    to convert data to american spelling from british spelling.
    :param add_context: A dictionary of entities and desired proportions
    to add context at the beginning or end (or both) of the sentences.
    :param contractions: A dictionary of entities and desired proportions to contract sentence.
    :param swap_entities: Named entities replaced with same entity type with same token count from terminology.
    :param swap_cohyponyms: Named entities replaced with co-hyponym from the WordNet database.
    :param optimized_inplace: Optimization algorithm to run augmentation inplace. Entities in the sentence are
    distinctly counted, which means same perturbations applied to the entities at the same time.
    :param random_state: A random state to create perturbation in the same samples of data.
    :param return_spark: Return Spark DataFrame instead of CoNLL file.
    :param regex_pattern: Regex pattern to tokenize context and contractions by splitting.
    :param spark: An active spark session to create DataFrame.
    :param conll_save_path: A path to save augmented CoNLL file.
    :param ignore_warnings: Ignore warnings about augmentation, default is False.
    :param print_info: Log informations about augmentation, default is False.
    :param starting_context: Pass custom starting context for add_context.
    :param ending_context: Pass custom ending context for add_context.
    :return: Augmented CoNLL file or Spark DataFrame.
    """
    logging.basicConfig()
    logger = logging.getLogger("Robustness Fixing")
    logger.setLevel(level=logging.WARNING)

    if print_info:
        logger.setLevel(level=logging.INFO)

    augmentation_report = dict()
    if ignore_warnings and not print_info:
        logger.setLevel(level=logging.ERROR)

    if conll_save_path is None and not return_spark:
        raise ValueError("conll_save_path or return_spark must be set to return augmented data.")

    docs = conll_reader(conll_path)

    counter = 0
    docs_indx = []

    data = []
    pos_sync_tag = []
    labels = []
    for doc in docs:
        #   keep track of the doc ending indxs
        counter += len(doc[0])
        docs_indx.append(counter)

        #   collect all doc sentences in the same list to process at the same time
        data.extend(doc[0])
        pos_sync_tag.extend(doc[1])
        labels.extend(doc[2])

    logger.info(f' {len(docs_indx)} number of documents from the CoNLL file is read.')
    logger.info(f' {len(data)} number of samples collected from the {len(docs_indx)} docs.')

    augmented_data = []
    augmented_labels = []
    augmented_tags = []

    entities = []
    for sent_labels in labels:
        for label in sent_labels.split():

            if label == 'O':
                continue

            label = label[2:]
            if label not in entities:
                entities.append(label)

    #   Save entity coverage to the augmentation_report
    augmentation_report['entity_coverage'] = dict()
    augmentation_report['augmentation_coverage'] = dict()

    num_instances = dict()
    for ent_type in entities:
        filtered_data, _, _ = filter_by_entity_type(data, pos_sync_tag, labels, ent_type=ent_type)
        num_instances[ent_type] = len(filtered_data)

        entity_coverage_info = round(len(filtered_data) / len(data), 2)
        augmentation_report['entity_coverage'][ent_type] = entity_coverage_info
        logger.info(f' Entity coverage of the data is {len(filtered_data)} '
                    f'out of {len(data)} ({entity_coverage_info}).')

    if random_state:
        random.seed(random_state)
        logger.info(f' Random state is set to {random_state}.')

    a2b_dict = _A2B_DICT
    b2a_dict = {v: k for k, v in a2b_dict.items()}

    terminology = None
    if swap_entities:
        terminology = create_terminology(data, labels)

    perturbation_dict = {
        "uppercase": uppercase,
        "lowercase": lowercase,
        "title": title,
        "add_punctuation": add_punctuation,
        "strip_punctuation": strip_punctuation,
        "make_typos": make_typos,
        "american_to_british": american_to_british,
        "british_to_american": british_to_american,
        "add_context": add_context,
        "contractions": contractions,
        "swap_entities": swap_entities,
        "swap_cohyponyms": swap_cohyponyms
    }

    perturb_func_map = {
        "uppercase": modify_capitalization_upper,
        "lowercase": modify_capitalization_lower,
        "title": modify_capitalization_title,
        "add_punctuation": add_punctuation_to_data,
        "strip_punctuation": strip_punctuation_from_data,
        "make_typos": introduce_typos,
        "american_to_british": convert_accent,
        "british_to_american": convert_accent,
        "add_context": add_context_to_data,
        "contractions": add_contractions,
        "swap_entities": swap_entities_with_terminology,
        "swap_cohyponyms": swap_with_cohyponym
    }

    perturb_args = {
        "uppercase": {},
        "lowercase": {},
        "title": {},
        "add_punctuation": {},
        "strip_punctuation": {},
        "make_typos": {},
        "american_to_british": {'lang_dict': a2b_dict},
        "british_to_american": {'lang_dict': b2a_dict},
        "add_context": {
            'ending_context': ending_context,
            'starting_context': starting_context,
            'regex_pattern': regex_pattern
        },
        "contractions": {'regex_pattern': regex_pattern},
        "swap_entities": {'terminology': terminology},
        "swap_cohyponyms": {}
    }

    if optimized_inplace:

        optimization_matrix = np.zeros((len(data), len(entities)))
        for x_indx, sent_label in enumerate(labels):
            for y_indx, ent_type in enumerate(entities):
                if ent_type in sent_label:
                    optimization_matrix[x_indx][y_indx] = 1

        for ent_type in entities:

            total = 0
            for proportions in perturbation_dict.values():

                if proportions is None:
                    continue

                if proportions.get(ent_type):
                    total += proportions[ent_type]

            if total > 1:
                raise ValueError(f" Perturbation values for {ent_type} is invalid. "
                                 f"Sum of proportion values should be smaller than 1!")

    for perturb_type, proportions in perturbation_dict.items():

        if proportions is None:
            continue

        if optimized_inplace:

            logger.info(f" Data is being augmented by {perturb_type}, optimized inplace is running.")

            max_possible_perturbation = {k: int(num_instances[k] * v) for k, v in proportions.items()}
            possible_entities = list(max_possible_perturbation.keys())
            num_possible_entity = len(possible_entities)
            sample_total = 0

            inplace_data = []
            inplace_tags = []
            inplace_labels = []

            while num_possible_entity:

                for combination in itertools.combinations(possible_entities, num_possible_entity):

                    entity_condition = np.zeros(len(entities), dtype=int)
                    for ent_type in combination:
                        entity_condition[entities.index(ent_type)] = 1

                    filter_indx, = np.where(np.all(optimization_matrix == entity_condition, axis=1))

                    filtered_data = [data[indx] for indx in filter_indx]
                    filtered_tags = [pos_sync_tag[indx] for indx in filter_indx]
                    filtered_labels = [labels[indx] for indx in filter_indx]

                    optimized_max_sample = min(max_possible_perturbation.values())
                    if optimized_max_sample > len(filter_indx):
                        optimized_max_sample = len(filter_indx)

                    if optimized_max_sample == 0:
                        continue

                    #  get sample from the filtered data
                    sample_indx, sample_data, sample_tags, sample_labels = get_sample(
                        optimized_max_sample, filter_indx, filtered_data, filtered_tags, filtered_labels)
                    sample_total += len(sample_indx)

                    #  apply transformation to the proportion
                    aug_data, aug_tags, aug_labels, aug_indx = perturb_func_map[perturb_type](
                        sample_data, sample_tags, sample_labels, **perturb_args[perturb_type])

                    inplace_data.extend(aug_data)
                    inplace_tags.extend(aug_tags)
                    inplace_labels.extend(aug_labels)

                    #   drop samples from data and optimization matrix
                    drop_indx = [sample_indx[i] for i in aug_indx]
                    remaining_indx = np.delete(np.arange(len(data)), drop_indx)
                    optimization_matrix = optimization_matrix[remaining_indx]

                    data = [data[indx] for indx in remaining_indx]
                    pos_sync_tag = [pos_sync_tag[indx] for indx in remaining_indx]
                    labels = [labels[indx] for indx in remaining_indx]

                    #   update max_possible_perturbation
                    tmp = dict()
                    for k, v in max_possible_perturbation.items():
                        if v - optimized_max_sample > 0:
                            tmp[k] = v - optimized_max_sample

                    max_possible_perturbation = tmp.copy()
                    possible_entities = list(max_possible_perturbation.keys())

                num_possible_entity -= 1

            logger.info(f" {perturb_type} augmentation is finished!")
            if sample_total == 0:
                logger.warning(
                    f" {perturb_type} could not be applied! No samples remained.")
                logger.warning(
                    f" You may also consider 'optimized_inplace=False' setting to allow multiple copies.")
                continue

            logger.info(f" Totally, {len(inplace_data)} number of samples augmented inplace.")
            if not all(max_possible_perturbation.values()):
                for k, v in max_possible_perturbation.items():
                    if v == 0:
                        continue
                    logger.warning(f" {v} number of '{k}' samples could not augmented with {perturb_type}. "
                                   f"Not enough sample!")
                    logger.warning(
                        f" You may also consider 'optimized_inplace=False' setting to allow multiple copies.")

            augmented_data.append(inplace_data)
            augmented_tags.append(inplace_tags)
            augmented_labels.append(inplace_labels)

            augmentation_coverage_info = round(len(inplace_data) / sample_total, 2)
            logger.info(
                f' Augmentation coverage of {perturb_type} is {len(inplace_data)} '
                f'out of {sample_total} ({augmentation_coverage_info}).')
            augmentation_report['augmentation_coverage'][perturb_type] = augmentation_coverage_info
            if len(inplace_data) < 50:
                logger.warning(f' There is not much samples that conversion needed! Augmentation Coverage is '
                               f'({augmentation_coverage_info})')
                logger.warning(f' You may need to increase proportion value to cover more samples.')

        else:

            for ent_type, proportion in proportions.items():

                logger.info(f" Augmenting {ent_type} with {perturb_type}.")

                #  filter the data by entity types
                filtered_data, filtered_tags, filtered_labels = filter_by_entity_type(
                    data, pos_sync_tag, labels, ent_type)

                if len(filtered_data) == 0:
                    #   0 entity sample in the data, no need to continue
                    logger.error(f' There is no such entity "{ent_type}" in the data. '
                                 f'Augmentation will continue without {ent_type}.')
                    continue

                #  get proportion from the filtered data
                sample_data, sample_tags, sample_labels = get_sample(int(proportion * len(filtered_data)),
                                                                     filtered_data, filtered_tags, filtered_labels)

                #  apply transformation to the proportion
                aug_data, aug_tags, aug_labels, _ = perturb_func_map[perturb_type](
                    sample_data, sample_tags, sample_labels, **perturb_args[perturb_type])

                augmentation_coverage_info = round(len(aug_data) / len(sample_data), 2)
                logger.info(f' {perturb_type} is applied with {len(aug_data)} number of samples ')
                logger.info(
                    f' Augmentation coverage of the data is {len(aug_data)} '
                    f'out of {len(sample_data)} ({augmentation_coverage_info}).\n')
                augmentation_report['augmentation_coverage'][perturb_type] = augmentation_coverage_info
                entity_coverage_info = augmentation_report['entity_coverage'][ent_type]

                if augmentation_coverage_info < 0.1 and proportion < 0.7:
                    logger.warning(f' There is not much samples that conversion needed! Augmentation Coverage is '
                                   f'({augmentation_coverage_info})')
                    logger.warning(f' You may need to increase proportion value to cover more samples.')

                elif entity_coverage_info * proportion < 0.05:
                    logger.warning(f' Proportion desired for {perturb_type}: ({proportion}). '
                                   f'Entity Coverage of {ent_type}: ({entity_coverage_info})')
                    logger.warning(
                        f' With desired proportion value {proportion} and '
                        f'entity coverage: ({entity_coverage_info}), '
                        f' only {len(sample_data)} number of augmented samples added to data. '
                        f'You may need to increase proportion value to add more samples!')

                augmented_data.append(aug_data)
                augmented_labels.append(aug_labels)
                augmented_tags.append(aug_tags)

    number_of_sentences = len(data)
    for aug_data, aug_tags, aug_labels in zip(augmented_data, augmented_tags, augmented_labels):
        data.extend(aug_data)
        labels.extend(aug_labels)
        pos_sync_tag.extend(aug_tags)
    num_of_augmentation_samples = len(data) - number_of_sentences

    if return_spark:
        assert spark is not None, 'It is required to pass an active SparkSession in order to return Spark DataFrame'
        return create_dataframe(spark, data, pos_sync_tag, labels)

    try:
        with open(conll_save_path, 'w') as f:
            try:
                counter = 0
                f.write("-DOCSTART- -X- -X- O\n")
                for indx, (sentence, sent_tags, sent_labels) in enumerate(zip(data, pos_sync_tag, labels)):

                    if counter < len(docs_indx) and indx == docs_indx[counter]:
                        f.write("\n-DOCSTART- -X- -X- O\n")
                        counter += 1

                    f.write("\n")
                    for token, tags, label in zip(sentence.split(), sent_tags.split(), sent_labels.split()):
                        f.write(f"{token} {' '.join(tags.split('*'))} {label}\n")

            except (IOError, OSError) as e:
                print(f"Error while writing to the {conll_save_path}.")
                print(e)

    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Error while opening the {conll_save_path}.")
        print(e)

    augmentation_report['number_of_sentences'] = number_of_sentences
    augmentation_report['number_of_augmented_samples'] = num_of_augmentation_samples

    entity_coverage_after_aug = dict()
    for ent_type in entities:
        filtered_data, _, _ = filter_by_entity_type(data, pos_sync_tag, labels, ent_type=ent_type)
        entity_coverage_after_aug[ent_type] = round(len(filtered_data) / len(data), 2)

    augmentation_report['entity_coverage_after_augmentation'] = entity_coverage_after_aug

    return augmentation_report


def test_and_augment_robustness(spark: SparkSession, pipeline_model: PipelineModel, test_file_path: str,
                                conll_path_to_augment: str, conll_save_path: str = None,
                                metric_type: Optional[str] = 'flex',
                                metrics_output_format: str = 'dictionary',
                                log_path: str = 'robustness_test_results.json',
                                noise_prob: float = 0.5,
                                test: Optional[List[str]] = None,
                                starting_context: Optional[List[str]] = None,
                                ending_context: Optional[List[str]] = None,
                                random_state=None,
                                return_spark: bool = False,
                                regex_pattern: str = "\\s+|(?=[-.:;*+,$&%\\[\\]])|(?<=[-.:;*+,$&%\\[\\]])",
                                print_info: bool = False,
                                ignore_warnings: bool = False):
    """One-liner to test and augment CoNLL for robustness.

    :param spark: An active SparkSession to create Spark DataFrame
    :param pipeline_model: PipelineModel with document assembler, sentence detector, tokenizer, word embeddings if
    applicable, NER model with 'ner' output name, NER converter with 'ner_chunk' output name
    :param test_file_path: Path to file to test robustness. Can be .txt or .conll file in CoNLL format or
    .csv file with just one column (text) with series of test samples.
    :param conll_path_to_augment: Path to the CoNLL file that will be used for augmentation purposes.
    :param conll_save_path: Path to save augmented CoNLL file.
    :param metric_type: 'strict' calculates metrics for IOB2 format, 'flex' (default) calculates for IO format
    :param metrics_output_format: 'dictionary' to get a dictionary report, 'dataframe' to get a dataframe report
    :param log_path: Path to log file, False to avoid saving test results. Default 'robustness_test_results.json'
    :param noise_prob: Proportion of value between 0 and 1 to sample from test data.
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
    :param starting_context: list of context tokens to add to beginning when running the 'add_context' test
    :param ending_context: list of context tokens to add to end when running the 'add_context' test
    :param random_state: A random state to create perturbation in the same samples of data.
    :param return_spark: Return Spark DataFrame instead of CoNLL file.
    :param regex_pattern: Regex pattern to tokenize context and contractions by splitting.
    :param print_info: Log information about augmentation, default is False.
    :param ignore_warnings: Ignore warnings about augmentation, default is False.
    """
    test_results = test_robustness(spark=spark, pipeline_model=pipeline_model, test_file_path=test_file_path,
                                   metric_type=metric_type, metrics_output_format=metrics_output_format,
                                   log_path=log_path, noise_prob=noise_prob, test=test,
                                   starting_context=starting_context, ending_context=ending_context)

    suggestions = suggest_perturbations(test_results)

    if suggestions == {}:
        print("Test metrics have over 0.9 f1-score for all perturbations. Perturbations will not be applied.")

    else:
        augment_robustness(conll_path=conll_path_to_augment,
                           uppercase=get_augmentation_proportions(suggestions, 'modify_capitalization_upper'),
                           lowercase=get_augmentation_proportions(suggestions, 'modify_capitalization_lower'),
                           title=get_augmentation_proportions(suggestions, 'modify_capitalization_title'),
                           add_punctuation=get_augmentation_proportions(suggestions, 'add_punctuation'),
                           strip_punctuation=get_augmentation_proportions(suggestions, 'strip_punctuation'),
                           make_typos=get_augmentation_proportions(suggestions, 'introduce_typos'),
                           american_to_british=get_augmentation_proportions(suggestions, 'american_to_british'),
                           british_to_american=get_augmentation_proportions(suggestions, 'british_to_american'),
                           add_context=get_augmentation_proportions(suggestions, 'add_context'),
                           contractions=get_augmentation_proportions(suggestions, 'add_contractions'),
                           swap_entities=get_augmentation_proportions(suggestions, 'swap_entities'),
                           swap_cohyponyms=get_augmentation_proportions(suggestions, 'swap_cohyponyms'),
                           random_state=random_state, return_spark=return_spark, ending_context=ending_context,
                           starting_context=starting_context, regex_pattern=regex_pattern, spark=spark,
                           conll_save_path=conll_save_path, print_info=print_info, ignore_warnings=ignore_warnings)
