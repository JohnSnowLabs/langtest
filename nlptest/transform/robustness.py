import random
import re
import numpy as np
from abc import ABC, abstractmethod
from functools import reduce
from typing import Dict, List, Optional


from .utils import (CONTRACTION_MAP, TYPO_FREQUENCY)
from ..utils.custom_types import Sample, Span, Transformation


class BaseRobustness(ABC):

    """
    Abstract base class for implementing robustness measures.

    Attributes:
        alias_name (str): A name or list of names that identify the robustness measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.
    """

    @staticmethod
    @abstractmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:

        """
        Abstract method that implements the robustness measure.

        Args:
            data (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented robustness measure.
        """

        return NotImplementedError()
    
    alias_name = None

class UpperCase(BaseRobustness):
    alias_name = "uppercase"
    
    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Transform a list of strings with uppercase robustness
        Args:
            sample_list: List of sentences to apply robustness.
        Returns:
            List of sentences that uppercase robustness is applied.
        """
        for sample in sample_list:
            sample.test_case = sample.original.upper()
            sample.category = "robustness"
        return sample_list


class LowerCase(BaseRobustness):
    alias_name = "lowercase"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Transform a list of strings with lowercase robustness
        Args:
            sample_list: List of sentences to apply robustness.
        Returns:
            List of sentences that lowercase robustness is applied.
        """
        for sample in sample_list:
            sample.test_case = sample.original.lower()
            sample.category = "robustness"
        return sample_list


class TitleCase(BaseRobustness):
    alias_name = 'titlecase'

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Transform a list of strings with titlecase robustness
        Args:
            sample_list: List of sentences to apply robustness.
        Returns:
            List of sentences that titlecase robustness is applied.
        """
        for sample in sample_list:
            sample.test_case = sample.original.title()
            sample.category = "robustness"
        return sample_list


class AddPunctuation(BaseRobustness):

    alias_name = 'add_punctuation'

    @staticmethod
    def transform(sample_list: List[Sample], whitelist: Optional[List[str]] = None) -> List[Sample]:
        """Add punctuation at the end of the string, if there is punctuation at the end skip it
        Args:
            sample_list: List of sentences to apply robustness.
            whitelist: Whitelist for punctuations to add to sentences.
        Returns:
            List of sentences that have punctuation at the end.
        """

        if whitelist is None:
            whitelist = ['!', '?', ',', '.', '-', ':', ';']

        for sample in sample_list:
            if sample.original[-1] not in whitelist:
                sample.test_case = sample.original[:-1] + random.choice(whitelist)
            else:
                sample.test_case = sample.original
            sample.category = "robustness"
        return sample_list


class StripPunctuation(BaseRobustness):

    alias_name = "strip_punctuation"

    @staticmethod
    def transform(sample_list: List[Sample], whitelist: Optional[List[str]] = None) -> List[Sample]:
        """Add punctuation from the string, if there isn't punctuation at the end skip it

        Args:
            sample_list: List of sentences to apply robustness.
            whitelist: Whitelist for punctuations to strip from sentences.
        Returns:
            List of sentences that punctuation is stripped.
        """

        if whitelist is None:
            whitelist = ['!', '?', ',', '.', '-', ':', ';']

        for sample in sample_list:
            if sample.original[-1] in whitelist:
                sample.test_case = sample.original[:-1]
            else:
                sample.test_case = sample.original
            sample.category = "robustness"
        return sample_list


class AddTypo(BaseRobustness):

    alias_name = 'add_typo'

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Add typo to the sentences using keyboard typo and swap typo.
        Args:
            sample_list: List of sentences to apply robustness.
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
            sample.category = "robustness"
        return sample_list


class SwapEntities(BaseRobustness):

    alias_name = 'swap_entities'

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

        for idx, sample in enumerate(sample_list):
            sent_tokens = sample.original.split(' ')
            sent_labels = labels[idx]

            ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sent_labels])
            ent_idx, = np.where(ent_start_pos == 1)

            #  no swaps since there is no entity in the sentence
            if len(ent_idx) == 0:
                sample.test_case = sample.original
                continue

            replace_idx = np.random.choice(ent_idx)
            ent_type = sent_labels[replace_idx][2:]
            replace_idxs = [replace_idx]
            if replace_idx < len(sent_labels) - 1:
                for i, label in enumerate(sent_labels[replace_idx + 1:]):
                    if label == f'I-{ent_type}':
                        replace_idxs.append(i + replace_idx + 1)
                    else:
                        break

            replace_token = sent_tokens[replace_idx: replace_idx + len(replace_idxs)]
            token_length = len(replace_token)
            replace_token = " ".join(replace_token)

            proper_entities = [ent for ent in terminology[ent_type] if len(ent.split(' ')) == token_length]
            if len(proper_entities) > 0:
                chosen_ent = random.choice(proper_entities)
            else:
                continue
            replaced_string = sample.original.replace(replace_token, chosen_ent)
            sample.test_case = replaced_string
            sample.category = "robustness"
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


class SwapCohyponyms(BaseRobustness):

    alias_name = "swap_cohyponyms"

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

        for idx, sample in enumerate(sample_list):
            sent_tokens = sample.original.split(' ')
            sent_labels = labels[idx]

            ent_start_pos = np.array([1 if label[0] == 'B' else 0 for label in sent_labels])
            ent_idx, = np.where(ent_start_pos == 1)

            #  no swaps since there is no entity in the sentence
            if len(ent_idx) == 0:
                sample.test_case = sample.original
                continue

            replace_idx = np.random.choice(ent_idx)
            ent_type = sent_labels[replace_idx][2:]
            replace_idxs = [replace_idx]
            if replace_idx < len(sent_labels) - 1:
                for i, label in enumerate(sent_labels[replace_idx + 1:]):
                    if label == f'I-{ent_type}':
                        replace_idxs.append(i + replace_idx + 1)
                    else:
                        break

            replace_token = sent_tokens[replace_idx: replace_idx + len(replace_idxs)]

            replace_token = " ".join(replace_token)
            chosen_ent = get_cohyponyms_wordnet(replace_token)
            replaced_string = sample.original.replace(replace_token, chosen_ent)
            sample.test_case = replaced_string
            sample.category = "robustness"

        return sample_list


class ConvertAccent(BaseRobustness):

    alias_name = ["american_to_british", "british_to_american"]

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
            tokens = sample.original.split(' ')
            tokens = [accent_map[t.lower()] if accent_map.get(t.lower(), None) else t for t in tokens]
            sample.test_case = ' '.join(tokens)
            sample.category = "robustness"

        return sample_list


class AddContext(BaseRobustness):

    alias_name = 'add_context'

    @staticmethod
    def transform(
            sample_list: List[Sample],
            starting_context: Optional[List[str]] = None,
            ending_context: Optional[List[str]] = None,
            strategy: List[str] = None,
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
            sample.category = "robustness"
        return sample_list


class AddContraction(BaseRobustness):

    alias_name = 'add_contraction'

    @staticmethod
    def transform(
            sample_list: List[Sample],
    ) -> List[str]:
        """Converts input sentences using a conversion dictionary
        Args:
            sample_list: List of sentences to process.
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

        for sample in sample_list:
            string = sample.original
            for contraction in CONTRACTION_MAP:
                if re.search(contraction, sample.original, flags=re.IGNORECASE | re.DOTALL):
                    string = re.sub(contraction, custom_replace, sample.original, flags=re.IGNORECASE | re.DOTALL)
            sample.test_case = string
            sample.category = "robustness"
        return sample_list
