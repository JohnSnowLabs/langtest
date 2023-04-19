import asyncio
import random
import re
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from nlptest.modelhandler.modelhandler import ModelFactory

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
    alias_name = None

    @staticmethod
    @abstractmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """
        Abstract method that implements the robustness measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented robustness measure.
        """

        return NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelFactory, **kwargs) -> List[Sample]:
        """
        Abstract method that implements the robustness measure.
        
        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the robustness measure.
        
        Returns:
            List[Sample]: The transformed data based on the implemented robustness measure.

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
        Creates a task to run the robustness measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelFactory): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the robustness measure.

        Returns:
            asyncio.Task: The task that runs the robustness measure.

        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


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
            sample.category = "robustness"
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

        assert len(sample_list) == len(labels), f"'labels' and 'sample_list' must have same lengths."

        for sample, sample_labels in zip(sample_list, labels):
            sample.category = "robustness"
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
            sample.category = "robustness"

        return sample_list


class AddContext(BaseRobustness):
    alias_name = 'add_context'

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
            sample.category = "robustness"
        return sample_list


class AddContraction(BaseRobustness):
    alias_name = 'add_contraction'

    @staticmethod
    def transform(
            sample_list: List[Sample],
    ) -> List[Sample]:
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
            replaced_string = sample.original
            transformations = []

            for contraction in CONTRACTION_MAP:
                search = re.search(contraction, sample.original, flags=re.IGNORECASE | re.DOTALL)
                if search:
                    new_string = CONTRACTION_MAP.get(search.group(), search.group())
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
            sample.category = "robustness"
        return sample_list
