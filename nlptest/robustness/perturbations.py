"""Functions to fix robustness of NER model with different kinds of perturbations in CoNLL data"""
import re
import random
import numpy as np
from copy import deepcopy
from typing import Optional, List, Dict, Tuple
import wn

from .utils import CONTRACTION_MAP, TYPO_FREQUENCY


def modify_capitalization_upper(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None,
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Convert every sentence in the data by uppercase.

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to data.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []

    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        upper_sent = sentence.upper()
        if sentence != sentence.upper():
            aug_sent.append(upper_sent)
            aug_indx.append(indx)

            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])

    return aug_indx, aug_sent, aug_tags, aug_labels


def modify_capitalization_lower(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Convert every sentence in the data by lowercase.

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to data.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []

    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        sent_lower = sentence.lower()
        if sent_lower != sentence:
            aug_sent.append(sent_lower)
            aug_indx.append(indx)

            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])

    return aug_indx, aug_sent, aug_tags, aug_labels


def modify_capitalization_title(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Convert every sentence in the data by title case.

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to data.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []

    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        title_sent = sentence.title()
        if title_sent != sentence:
            aug_sent.append(title_sent)
            aug_indx.append(indx)

            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])

    return aug_indx, aug_sent, aug_tags, aug_labels


def add_punctuation_to_data(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Adds a punctuation at the end of strings.

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    list_of_characters = ['!', '?', ',', '.', '-', ':', ';']

    aug_sent = []
    aug_labels = []
    aug_tags = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        if sentence[-1].isalnum():
            aug_sent.append(sentence + " " + random.choice(list_of_characters))
            aug_indx.append(indx)

            if tags and labels:
                aug_tags.append(tags[indx] + ' -X-*-X-')
                aug_labels.append(labels[indx] + ' O')

    return aug_indx, aug_sent, aug_tags, aug_labels


def strip_punctuation_from_data(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Strips punctuations from the sentences.

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    aug_sent = []
    aug_labels = []
    aug_tags = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        if not sentence[-1].isalnum():
            aug_sent.append(" ".join(sentence.split()[:-1]))
            aug_indx.append(indx)

            if tags and labels:
                aug_tags.append(" ".join(tags[indx].split()[:-1]))
                aug_labels.append(" ".join(labels[indx].split()[:-1]))

    return aug_indx, aug_sent, aug_tags, aug_labels


def add_typo_to_sentence(
        sentence: str,
) -> str:
    """
    Add typo to a single sentence using keyboard typo and swap typo.

    :param sentence: List of sentences to process.
    :return: Input sentence with typo added.
    """
    if len(sentence) == 1:
        return sentence

    sentence = list(sentence)
    if random.random() > 0.1:

        indx_list = list(range(len(TYPO_FREQUENCY)))
        char_list = list(TYPO_FREQUENCY.keys())

        counter = 0
        indx = -1
        while counter < 10 and indx == -1:
            indx = random.randint(0, len(sentence) - 1)
            char = sentence[indx]
            if TYPO_FREQUENCY.get(char.lower(), None):

                char_frequency = TYPO_FREQUENCY[char.lower()]

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
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Introduces typos in input sentences

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        typo_sent = add_typo_to_sentence(sentence)
        if len(typo_sent.split()) == len(sentence.split()):
            aug_sent.append(typo_sent)
            aug_indx.append(indx)

            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])

    return aug_indx, aug_sent, aug_tags, aug_labels


def create_terminology(sentences: List[str], labels: List[str]) -> Dict[str, List[str]]:
    """
    Iterate over the DataFrame to create terminology from the predictions. IOB format converted to the IO.

    :param sentences: list of sentences to process
    :param labels: corresponding labels to construct terminology.
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

                sent_tokens = sentences[sent_indx].split(' ')
                chunk = [sent_tokens[token_indx]]
                ent_type = label[2:]

            elif label.startswith('I'):

                sent_tokens = sentences[sent_indx].split(' ')
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


def swap_entities_with_terminology(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None,
        terminology: Optional[Dict[str, List[int]]] = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    This function swap named entities with the new one from the terminology extracted from passed data.

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :param terminology: Dictionary of entities and corresponding list of words.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    if terminology is None:
        terminology = create_terminology(sentences, labels)

    if labels is None:
        raise ValueError("'labels' should passed as a parameter in order to be swapped!")

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        sent_tokens = sentence.split(' ')
        sent_labels = labels[indx].split(' ')

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
        replaced_string = sentence.replace(replace_token, chosen_ent)

        new_labels = ['B-' + ent_type] + ['I-' + ent_type] * (chosen_ent_length - 1)
        new_labels = sent_labels[:replace_indx] + new_labels + sent_labels[replace_indx + len(replace_indxs):]

        aug_sent.append(replaced_string)
        aug_labels.append(" ".join(new_labels))
        aug_indx.append(indx)

        if tags:
            sent_tags = tags[indx].split(' ')
            new_tags = ["-X-*-X-"] * chosen_ent_length
            new_tags = sent_tags[:replace_indx] + new_tags + sent_tags[replace_indx + len(replace_indxs):]

            aug_tags.append(" ".join(new_tags))

    return aug_indx, aug_sent, aug_tags, aug_labels


def get_cohyponyms_wordnet(word: str) -> str:
    """
    Retrieve co-hyponym of the input string using WordNet when a hit is found.

    :param word: input string to retrieve co-hyponym
    :return: Cohyponym of the input word if exist, else original word.
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
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    This function swap named entities with a co-hyponym from the WordNet database when a hit is found.

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    if labels is None:
        raise ValueError("'labels' should be passed in order to swap named entities in the sentences!")

    #  download WordNet DB
    print('\nDownloading WordNet database to execute co-hyponym swapping.\n')
    wn.download('ewn:2020')

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        sent_tokens = sentence.split(' ')
        sent_labels = labels[indx].split(' ')

        ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sent_labels])
        ent_indx, = np.where(ent_start_pos == 1)

        #  if there is no named entity in the sentence, skip
        if len(ent_indx) == 0:
            continue

        replace_indx = np.random.choice(ent_indx)
        ent_type = sent_labels[replace_indx][2:]
        replace_indxs = [replace_indx]
        if replace_indx < len(sent_labels) - 1:
            for token_indx, label in enumerate(sent_labels[replace_indx + 1:]):
                if label == f'I-{ent_type}':
                    replace_indxs.append(token_indx + replace_indx + 1)
                else:
                    break

        replace_token = sent_tokens[replace_indx: replace_indx + len(replace_indxs)]
        replace_token = " ".join(replace_token)

        #  replace by cohyponym
        chosen_ent = get_cohyponyms_wordnet(replace_token)
        chosen_ent_length = len(chosen_ent.split(' '))
        replaced_string = sentence.replace(replace_token, chosen_ent)

        new_labels = ['B-' + ent_type] + ['I-' + ent_type] * (chosen_ent_length - 1)
        new_labels = sent_labels[:replace_indx] + new_labels + sent_labels[replace_indx + len(replace_indxs):]

        aug_sent.append(replaced_string)
        aug_labels.append(" ".join(new_labels))
        aug_indx.append(indx)

        if tags:
            sent_tags = tags[indx].split(' ')
            new_tags = ["-X-*-X-"] * chosen_ent_length
            new_tags = sent_tags[:replace_indx] + new_tags + sent_tags[replace_indx + len(replace_indxs):]
            aug_tags.append(" ".join(new_tags))

    return aug_indx, aug_sent, aug_tags, aug_labels


def convert_accent(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None,
        accent_map: Dict[str, str] = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Converts input sentences using a conversion dictionary

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :param accent_map: Dictionary with conversion terms.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    if accent_map is None:
        raise ValueError("'accent_map' should be passed in order to convert language accent in the sentences!")

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        old_sentence = sentence
        for token in sentence.split():
            if accent_map.get(token, None):
                sentence = sentence.replace(token, accent_map[token])

        if old_sentence != sentence:
            aug_sent.append(sentence)
            aug_indx.append(indx)

            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])

    return aug_indx, aug_sent, aug_tags, aug_labels


def add_context_to_data(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None,
        starting_context: Optional[List[str]] = None,
        ending_context: Optional[List[str]] = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Adds tokens at the beginning and/or at the end of strings

    :param sentences: List of sentences to process.
    :param tags: Corresponding tags to make changes according to data.
    :param labels: Corresponding labels to make changes according to sentences.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :param starting_context: list of terms (context) to input at start of sentences.
    :param ending_context: list of terms (context) to input at end of sentences.
        Ending context should not be ending with punctuation.
        It will be added if normal sentence is added with punctuation.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        method = random.choice(['start', 'end', 'combined'])

        if method == 'start':
            #   choose randomly from list of starting context
            add_tokens = random.choice(starting_context)

            #   join tokens
            add_string = " ".join(add_tokens)

            aug_sent.append(add_string + ' ' + sentence)
            aug_indx.append(indx)

            if tags and labels:
                add_tags = " ".join(['-X-*-X-'] * len(add_tokens))
                aug_tags.append(add_tags + ' ' + tags[indx])

                add_labels = " ".join(['O'] * len(add_tokens))
                aug_labels.append(add_labels + ' ' + labels[indx])

        elif method == 'end':
            #   choose randomly from list of ending context
            add_tokens = random.choice(ending_context)

            #   join tokens
            add_string = " ".join(add_tokens)

            if sentence[-1].isalnum():
                aug_sent.append(sentence + ' ' + add_string)
                aug_indx.append(indx)

                if tags and labels:
                    add_tags = " ".join(['-X-*-X-'] * len(add_tokens))
                    aug_tags.append(tags[indx] + ' ' + add_tags)

                    add_labels = " ".join(['O'] * len(add_tokens))
                    aug_labels.append(labels[indx] + ' ' + add_labels)

            else:
                aug_sent.append(sentence[:-1] + add_string + " " + sentence[-1])
                aug_indx.append(indx)

                if tags and labels:
                    tags_splitted = tags[indx].split()
                    add_tags = " ".join(['-X-*-X-'] * len(add_tokens))
                    aug_tags.append(" ".join(tags_splitted[:-1]) + ' ' + add_tags + " " + tags_splitted[-1])

                    add_labels = " ".join(['O'] * len(add_tokens))
                    aug_labels.append(labels[indx][:-1] + add_labels + " O")

        elif method == 'combined':
            #   choose randomly from list of starting and ending context
            add_ending_tokens = random.choice(ending_context)
            add_starting_tokens = random.choice(starting_context)

            #   join starting tokens
            add_starting_string = " ".join(add_starting_tokens)

            #   join ending tokens
            add_ending_string = " ".join(add_ending_tokens)

            if sentence[-1].isalnum():

                aug_sent.append(sentence + ' ' + add_ending_string)
                aug_indx.append(indx)

                if tags and labels:
                    #   align tags
                    add_starting_tags = " ".join(['-X-*-X-'] * len(add_starting_tokens))
                    add_ending_tags = " ".join(['-X-*-X-'] * len(add_ending_tokens))
                    aug_tags.append(add_starting_tags + " " + tags[indx] + " " + add_ending_tags)

                    add_ending_labels = " ".join(['O'] * len(add_ending_tokens))
                    add_starting_labels = " ".join(['O'] * len(add_starting_tokens))
                    aug_labels.append(add_starting_labels + " " + labels[indx] + " " + add_ending_labels)

            else:

                #   add context to sentence
                aug_sent.append(add_starting_string + " " + sentence[:-1] +
                                add_ending_string + " " + sentence[-1])
                aug_indx.append(indx)

                if tags and labels:
                    #   align tags
                    tags_splitted = tags[indx].split(' ')
                    add_starting_tags = " ".join(['-X-*-X-'] * len(add_starting_tokens))
                    add_ending_tags = " ".join(['-X-*-X-'] * len(add_ending_tokens))

                    aug_tags.append(add_starting_tags + " " + " ".join(tags_splitted[:-1]) +
                                    ' ' + add_ending_tags + " " + tags_splitted[-1])

                    #   align labels
                    add_ending_labels = " ".join(['O'] * len(add_ending_tokens))
                    add_starting_labels = " ".join(['O'] * len(add_starting_tokens))
                    aug_labels.append(add_starting_labels + " " + labels[indx][:-1] + add_ending_labels + " O")

    return aug_indx, aug_sent, aug_tags, aug_labels


def add_contractions(
        sentences: List[str],
        tags: List[str] = None,
        labels: List[str] = None,
        noise_prob: float = None
) -> Tuple[List[int], List[str], List[str], List[str]]:
    """
    Adds contractions in input sentences

    :param sentences: list of sentences to contract.
    :param tags: corresponding tags of sentences to align with.
    :param labels: corresponding labels of sentences to align with.
    :param noise_prob: Proportion of value between 0 and 1 to sample from the data.
    :return: List sample indexes and corresponding augmented sentences, tags and labels if provided.
    """

    def custom_replace(match):
        """
          regex replace for contraction.
        """
        token = match.group(0)
        contracted_token = CONTRACTION_MAP.get(token, CONTRACTION_MAP.get(match.lower()))

        is_upper_case = token[0]
        expanded_contraction = is_upper_case + contracted_token[1:]
        return expanded_contraction

    #  we need to iterate over them to apply all contractions
    sentences_ = deepcopy(sentences)
    tags_ = deepcopy(tags)
    labels_ = deepcopy(labels)

    aug_sent = []
    aug_tags = []
    aug_labels = []
    aug_indx = []
    for indx, sentence in enumerate(sentences):

        #   noise will not be applied
        if noise_prob and random.random() > noise_prob:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_tags.append(tags[indx])
                aug_labels.append(labels[indx])
            continue

        is_contracted = False
        for contraction in CONTRACTION_MAP:

            if re.search(f'\b({contraction})\b', sentence, flags=re.IGNORECASE | re.DOTALL):
                is_contracted = True

                sentence = re.sub(f'\b({contraction})\b', custom_replace, sentence, flags=re.IGNORECASE | re.DOTALL)

                if tags and labels:

                    match_char_start = [match.start() for match in re.finditer(contraction, sentences_[indx])]
                    match_jumps = [sentence[:indx].count(' ') - i for i, indx in enumerate(match_char_start)]

                    for jump in match_jumps:
                        # align labels
                        sent_labels = labels_[indx].split(' ')
                        sent_labels = sent_labels[:jump] + sent_labels[jump + 1:]
                        labels_[indx] = " ".join(sent_labels)

                        # align tags
                        sent_tags = tags_[indx].split(' ')
                        sent_tags = sent_tags[:jump] + sent_tags[jump + 1:]
                        tags_[indx] = " ".join(sent_tags)

        if is_contracted:
            aug_sent.append(sentence)
            aug_indx.append(indx)
            if tags and labels:
                aug_labels.append(labels_[indx])
                aug_tags.append(tags_[indx])

    return aug_indx, aug_sent, aug_tags, aug_labels


LIST_OF_PERTURBATIONS = ["capitalization_upper", "capitalization_lower", "capitalization_title", "add_punctuation",
                          "strip_punctuation", "introduce_typos", "american_to_british", "british_to_american",
                          "add_context", "add_contractions", "swap_entities", "swap_cohyponyms"]

PERTURB_FUNC_MAP = {
    "capitalization_upper": modify_capitalization_upper,
    "capitalization_lower": modify_capitalization_lower,
    "capitalization_title": modify_capitalization_title,
    "add_punctuation": add_punctuation_to_data,
    "strip_punctuation": strip_punctuation_from_data,
    "introduce_typos": introduce_typos,
    "american_to_british": convert_accent,
    "british_to_american": convert_accent,
    "add_context": add_context_to_data,
    "add_contractions": add_contractions,
    "swap_entities": swap_entities_with_terminology,
    "swap_cohyponyms": swap_with_cohyponym
}

PERTURB_DESCRIPTIONS = {
    "capitalization_upper": 'text capitalization turned into uppercase',
    "capitalization_lower": 'text capitalization turned into lowercase',
    "capitalization_title": 'text capitalization turned into title type (first letter capital)',
    "add_punctuation": 'special character at the end of the sentence is modified',
    "strip_punctuation": 'remove_punctuation_tokens',
    "introduce_typos": 'typos introduced in sentences',
    "american_to_british": 'American spelling turned into British spelling',
    "british_to_american": 'British spelling turned into American spelling',
    "add_context": 'words added at the beginning and end of sentences',
    "add_contractions": 'contractions added in sentences',
    "swap_entities": 'named entities in the sentences are replaced with same entity type',
    "swap_cohyponyms": 'named entities in the sentences are replaced with same entity type'
}
