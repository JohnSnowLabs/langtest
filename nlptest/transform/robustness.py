import asyncio
import random
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from nlptest.modelhandler.modelhandler import ModelFactory
from .utils import (CONTRACTION_MAP, TYPO_FREQUENCY, default_user_prompt ,ocr_typo_dict,abbreviation_dict,Slang_Nouns, Slang_Adverbs, Slang_Adjectives, dyslexia_map)
from ..utils.custom_types import Sample, Span, Transformation
from ..utils.number_to_word import ConvertNumberToWord 
from typing import List
import string
from ..utils.SoundsLikeFunctions import Search

class BaseRobustness(ABC):
    """
    Abstract base class for implementing robustness measures.

    Attributes:
        alias_name (str): A name or list of names that identify the robustness measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented robustness measure.
    """
    alias_name = None
    supported_tasks = ["ner", "text-classification",
                       "question-answering", "summarization"]

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
                if hasattr(sample, "run"):
                    sample_status = sample.run(model, **kwargs)
                    if sample_status:
                        sample.state = "done"
                else:
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
        created_task = asyncio.create_task(
            cls.run(sample_list, model, **kwargs))
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
        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = sample.upper()
            else:
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
        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = sample.lower()
            else:
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
        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = sample.title()
            else:
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

        def check_whitelist(text, whitelist):
            if text[-1] not in whitelist:
                chosen_punc = random.choice(whitelist)
                return text + chosen_punc
            else:
                return text

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = check_whitelist(sample, whitelist)
            else:
                if sample.original[-1] not in whitelist:
                    chosen_punc = random.choice(whitelist)
                    sample.test_case = sample.original + chosen_punc
                    if sample.task in ("ner", "text-classification"):
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

        def check_whitelist(text, whitelist):
            if text[-1] in whitelist:
                return text[:-1]
            else:
                return text

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = check_whitelist(sample, whitelist)
            else:
                if sample.original[-1] in whitelist:
                    sample.test_case = sample.original[:-1]
                    if sample.task in ("ner", "text-classification"):
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

        def keyboard_typo(string):

            if len(string) < 5:
                return string

            string = list(string)

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
                            chosen_char = random.choices(
                                idx_list, weights=char_frequency)
                            difference = ord(char.lower()) - \
                                ord(char_list[chosen_char[0]])
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

            return "".join(string)

        for idx, sample in enumerate(sample_list):

            if isinstance(sample, str):
                sample_list[idx] = keyboard_typo(sample)
            else:
                sample.category = "robustness"
                sample.test_case = keyboard_typo(sample.original)

        return sample_list


class SwapEntities(BaseRobustness):
    alias_name = 'swap_entities'
    supported_tasks = ["ner"]

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
            raise ValueError(
                'In order to generate test cases for swap_entities, terminology should be passed!')

        if labels is None:
            raise ValueError(
                'In order to generate test cases for swap_entities, labels should be passed!')

        assert len(sample_list) == len(
            labels), f"'labels' and 'sample_list' must have same lengths."

        for sample, sample_labels in zip(sample_list, labels):
            sample.category = "robustness"
            if all([label == "O" for label in sample_labels]):
                sample.test_case = sample.original
                continue

            sent_tokens = sample.original.split(' ')

            ent_start_pos = [1 if label[0] == 'B' else 0 for label in sample_labels]
            ent_idx = [i for i, value in enumerate(ent_start_pos) if value==1]
    
            replace_idx = random.choice(ent_idx)
            ent_type = sample_labels[replace_idx][2:]
            replace_idxs = [replace_idx]
            if replace_idx < len(sample_labels) - 1:
                for i, label in enumerate(sample_labels[replace_idx + 1:]):
                    if label == f'I-{ent_type}':
                        replace_idxs.append(i + replace_idx + 1)
                    else:
                        break

            replace_token = sent_tokens[replace_idx: replace_idx +
                                        len(replace_idxs)]
            token_length = len(replace_token)
            replace_token = " ".join(replace_token)

            chosen_ent = random.choice(terminology[ent_type])
            replace_token_pos = re.search(replace_token, sample.original)

            sample.test_case = sample.original.replace(
                replace_token, chosen_ent)
            if sample.task in ("ner", "text-classification"):
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
        def convert_accent(string: str, accent_map: Dict[str, str]) -> str:
            tokens = set(string.split(' '))
            replaced_string = string
            transformations = []

            for i, token in enumerate(tokens):
                new_token = accent_map.get(token.lower(), token)
                if new_token != token:
                    diff_len = len(new_token) - len(token)
                    nb_occurrences = len(re.findall(token, replaced_string))

                    for c in range(nb_occurrences):
                        span = re.search(token, replaced_string)
                        replaced_string = re.sub(
                            token, new_token, replaced_string, count=1)
                        
                        transformations.append(
                                Transformation(
                                    original_span=Span(
                                        start=span.start(), end=span.end(), word=token),
                                    new_span=Span(
                                        start=span.start(), end=span.end() + diff_len, word=new_token),
                                    ignore=False
                                )
                            )
            return replaced_string, transformations

        for idx, sample in enumerate(sample_list):

            if isinstance(sample, str):
                sample_list[idx], _ = convert_accent(sample, accent_map)
            else:
                if sample.task in ("ner", "text-classification"):
                    sample.test_case, sample.transformations = convert_accent(
                        sample.original, accent_map)
                else:
                    sample.test_case, _ = convert_accent(
                        sample.original, accent_map)
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

        def context(string, strategy):
            possible_methods = ['start', 'end', 'combined']
            if strategy is None:
                strategy = random.choice(possible_methods)
            elif strategy not in possible_methods:
                raise ValueError(
                    f"Add context strategy must be one of 'start', 'end', 'combined'. Cannot be {strategy}.")

            transformations = []
            
            if strategy == "start" or strategy == "combined":
                add_tokens = random.choice(starting_context)
                add_string = " ".join(add_tokens) if isinstance(
                    add_tokens, list) else add_tokens
                if string != "-":
                    string = add_string + ' ' + string
                    transformations.append(
                        Transformation(
                            original_span=Span(start=0, end=0, word=""),
                            new_span=Span(start=0, end=len(add_string) + 1, word=add_string),
                            ignore=True
                        )
                    )
            if strategy == "end" or strategy == "combined":
                add_tokens = random.choice(ending_context)
                add_string = " ".join(add_tokens) if isinstance(
                    add_tokens, list) else add_tokens
                if string != "-":
                    string = string + ' ' + add_string
                    transformations.append(
                        Transformation(
                            original_span=Span(
                            start=len(string) - 1, end=len(string), word=""),
                            new_span=Span(start=len(string), end=len(string) + len(add_string) + 1, word=add_string),
                            ignore=True
                        )
                    )
            return string, transformations

        for idx, sample in enumerate(sample_list):

            if isinstance(sample, str):
                sample_list[idx], _ = context(sample, strategy)
            else:
                if sample.task in ("ner", "text-classification"):
                    sample.test_case, sample.transformations = context(
                        sample.original, strategy)
                else:
                    sample.test_case, _ = context(
                        sample.original, strategy)
                    
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
            contracted_token = CONTRACTION_MAP.get(
                token, CONTRACTION_MAP.get(token.lower()))

            is_upper_case = token[0]
            expanded_contraction = is_upper_case + contracted_token[1:]
            return expanded_contraction

        def search_contraction(text):
            replaced_string = text
            for contraction in CONTRACTION_MAP:
                search = re.search(contraction, text,
                                   flags=re.IGNORECASE | re.DOTALL)
                if search:
                    new_string = CONTRACTION_MAP.get(
                        search.group(), search.group())
                    diff_len = len(new_string) - len(search.group())
                    replaced_string = re.sub(contraction, custom_replace, replaced_string,
                                             flags=re.IGNORECASE | re.DOTALL)

                return replaced_string

        for idx, sample in enumerate(sample_list):

            if isinstance(sample, str):
                sample_list[idx] = search_contraction(sample)
            else:
                replaced_string = sample.original
                transformations = []

                for contraction in CONTRACTION_MAP:
                    search = re.search(contraction, sample.original,
                                       flags=re.IGNORECASE | re.DOTALL)
                    if search:
                        new_string = CONTRACTION_MAP.get(
                            search.group(), search.group())

                        diff_len = len(new_string) - len(search.group())
                        replaced_string = re.sub(contraction, custom_replace, replaced_string,
                                                 flags=re.IGNORECASE | re.DOTALL)
                        if sample.task in ("ner", "text-classification"):
                            transformations.append(
                                Transformation(
                                    original_span=Span(start=search.start(
                                    ), end=search.end(), word=search.group()),
                                    new_span=Span(start=search.start(
                                    ), end=search.end() + diff_len, word=new_string),
                                    ignore=False
                                )
                            )
                sample.test_case = replaced_string
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "robustness"
        return sample_list


class DyslexiaWordSwap(BaseRobustness):
    alias_name = "dyslexia_word_swap"
    
    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Converts the string by changing some similar words from the dictonary(dyslexia_map) and outputs a string. 
           Args: sample_list: List of sentences to process.

           Returns: Returns a converted string like words like write will get converted to right. 
        """
        def dyslexia_swap(text):
            perturbed_text = text
            transformations = [] 

            for orig, perturbed in dyslexia_map.items():
                pattern = r"(?i)\b" + re.escape(orig) + r"\b"
                perturbed_text = re.sub(pattern, "__TEMP__" + perturbed, perturbed_text)
                matches = re.finditer(pattern, text)
                for match in matches:
                    start = match.start()
                    end = match.end()
                    token = text[start:end]
                    if perturbed != token:
                        transformations.append(
                            Transformation(
                                original_span=Span(start=start, end=end, word=token),
                                new_span=Span(start=start, end=start + len(perturbed), word=perturbed),
                                ignore=False
                            )
                        ) 
            perturbed_text = perturbed_text.replace( "__TEMP__", "")
            return perturbed_text, transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx],_= dyslexia_swap(sample)
            else:
                sample.test_case, transformations = dyslexia_swap(sample.original)
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "robustness"
        
        return sample_list


class NumberToWord(BaseRobustness):
    alias_name = "number_to_word"
    num = ConvertNumberToWord()

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """
        Transform a list of strings to their equivalent verbal representation
        of numbers present in the string.
        Args:
            sample_list: List of sentences to apply robustness.
        Returns:
            List of sentences that have numbers in their verbal representation.
        """

        def convert_numbers(regex, text):
            results = []
            trans = []
            transformations = []
            start_offset = 0

            for match in re.finditer(regex, text):
                token = match.group()
                words = NumberToWord.num.number_to_words(token, wantlist=True)
                new_words_len = len(' '.join(words))
                trans.append(text[start_offset:match.start()])
                trans.append(' '.join(words))
                start_offset = match.end()  
                transformations.append(
                        Transformation(
                            original_span=Span(
                                start=match.start(), end=match.end(), word=token),
                            new_span=Span(start=match.start(), end=match.start()+new_words_len, word=' '.join(words)),
                            ignore=False
                        )
                    )
                
            trans.append(text[start_offset:])
            results.append(''.join(trans))
            return ''.join(results), transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ = convert_numbers(r'(?<!\S)(\d+(\.\d+)?)(?=(\s|\n|$))', sample)
            else:
                sample.test_case, transformations = convert_numbers(r'(?<!\S)(\d+(\.\d+)?)(?=(\s|\n|$))', sample.original)
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "robustness"
        return sample_list

class AddOcrTypo(BaseRobustness):
    alias_name = "add_ocr_typo"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """
        Transforms the given sample list by introducing OCR typos.

        Args:
            sample_list (List[Sample]): The list of samples to transform.

        Returns:
            List[Sample]: The transformed list of samples.
        """

        def ocr_typo(regex, text):
            results = []
            trans = []
            transformations = []
            start_offset = 0

            for match in re.finditer(regex, text):
                token = match.group()
                corrected_token = None

                possible_corrections = [
                    key for key, value in ocr_typo_dict.items() if value == token]
                if possible_corrections:
                    corrected_token = random.choice(possible_corrections)
                else:
                    corrected_token = token

                if corrected_token != token:
                    trans.append(text[start_offset:match.start()])
                    trans.append(corrected_token)
                    start_offset = match.end()
                    transformations.append(
                        Transformation(
                            original_span=Span(
                            start=match.start(), end=match.end(), word=token),
                            new_span=Span(start=match.start(), end=match.start() + len(corrected_token), word=corrected_token),
                            ignore=False
                        )
                    )
                else:
                    trans.append(text[start_offset:match.end()])
                    start_offset = match.end()

            trans.append(text[start_offset:])
            results.append(''.join(trans))

            return ''.join(results), transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ = ocr_typo(r'[^,\s.!?]+', sample)
            else:
                sample.test_case, transformations = ocr_typo(r'[^,\s.!?]+', sample.original)
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "robustness"

        return sample_list
    
class AbbreviationInsertion(BaseRobustness):
    alias_name = "add_abbreviation"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """
        Transforms the given sample list by inserting abbreviations.

        Args:
            sample_list (List[Sample]): The list of samples to transform.

        Returns:
            List[Sample]: The transformed list of samples.
        """

        def insert_abbreviation(text):
            perturbed_text = text
            transformations = [] 

            for abbreviation, expansions in abbreviation_dict.items():
                for expansion in expansions:
                    pattern = r"(?i)\b" + re.escape(expansion) + r"\b"
                    corrected_token = abbreviation
                    perturbed_text = re.sub(pattern, corrected_token, perturbed_text)
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        start = match.start()
                        end = match.end()
                        token = text[start:end]
                        if corrected_token != token:
                    
                            transformations.append(
                                Transformation(
                                    original_span=Span(start=start, end=end, word=token),
                                    new_span=Span(start=start, end=start + len(corrected_token), word=corrected_token),
                                    ignore=False
                                )
                            ) 
            return perturbed_text, transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ = insert_abbreviation(sample)
            else:
                sample.test_case, transformations = insert_abbreviation(sample.original)
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "robustness"
        
        return sample_list
   
class AddSpeechToTextTypo(BaseRobustness):
    alias_name = "add_speech_to_text_typo"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """
        Transforms the given sample list by introducing typos simulating speech-to-text errors.
        Args:
            sample_list (List[Sample]): The list of samples to transform.
        Returns:
            List[Sample]: The transformed list of samples.
        """
        def convertToSimilarHarmony(sentence):
            words = re.findall(r"\w+(?:'\w+)*|\W", sentence)
            converted_sentence = []
            transformations = []
            index_offset = 0

            for word in words:
                if word.isspace() or all(ch in string.punctuation for ch in word):
                    converted_sentence.append(word)  # Preserve space and punctuation in the reconstructed sentence
                else:
                    try:
                        similar_words = Search.perfectHomophones(word)
                        if similar_words:
                            original_case = word[0].isupper()
                            similar_word = random.choice(similar_words)
                            if original_case and similar_word[0].islower():
                                similar_word = similar_word.capitalize()
                            elif not original_case and similar_word[0].isupper():
                                similar_word = similar_word.lower()

                            if similar_word.lower() == word.lower():
                                similar_word = word

                            if similar_word != word:
                                start_index = sentence.index(word, index_offset)
                                end_index = start_index + len(word)
                                converted_sentence.append(similar_word)
                                new_word_length = len(similar_word)
                                transformations.append(
                                    Transformation(
                                        original_span=Span(start=start_index, end=end_index, word=word),
                                        new_span=Span(start=start_index, end=start_index + new_word_length,
                                                      word=similar_word),
                                        ignore=False
                                    )
                                )
                                index_offset = end_index
                            else:
                                converted_sentence.append(word)

                        else:
                            converted_sentence.append(word)

                    except ValueError:
                        converted_sentence.append(word)

            perturbed_text = ''.join(converted_sentence)

            return perturbed_text, transformations

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ =convertToSimilarHarmony(sample)
            else:
                sample.test_case, transformations = convertToSimilarHarmony(sample.original)
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "robustness"

        return sample_list


class AddSlangifyTypo(BaseRobustness):
    alias_name = "add_slangs"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """
        Transforms the given sample list by adding slang words.
        Args:
            sample_list (List[Sample]): The list of samples to transform.
        Returns:
            List[Sample]: The transformed list of samples.
        """
        def slangify_typo(text):
            slang_words = [
                list(map(list, zip(*Slang_Nouns))),
                list(map(list, zip(*Slang_Adverbs))),
                list(map(list, zip(*Slang_Adjectives)))
            ]

            modified_toks = []
            tokens = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]|[\s]+", text)    # Include hyphenated and possessive words as single tokens
            transformations = []
            start_offset = 0

            for token in tokens:
                if token.isspace() or all(ch in string.punctuation for ch in token):
                    modified_toks.append(token)  # Preserve space and punctuation in the reconstructed text
                    continue    
                is_cap = token[0].isupper()
                replaced = False

                for slang in slang_words:
                    if token.lower() in slang[0]:
                        replacements = [i for i, x in enumerate(slang[0]) if x == token.lower()]
                        chosen_index = random.choice(replacements)
                        temp = slang[1][chosen_index]

                        if is_cap:
                            temp = temp[0].upper() + temp[1:]

                        if temp != token:
                            start_index = text.index(token, start_offset)
                            end_index = start_index + len(token)
                            modified_toks.append(temp)
                            new_word_length = len(temp)
                            transformations.append(
                                Transformation(
                                    original_span=Span(start=start_index, end=end_index, word=token),
                                    new_span=Span(start=start_index, end=start_index + new_word_length,
                                                word=temp),
                                    ignore=False
                                )
                            )
                            start_offset = end_index
                        else:
                            modified_toks.append(token)

                        replaced = True
                        break

                if not replaced:
                    modified_toks.append(token)

            modified_text = "".join(modified_toks)

            return modified_text, transformations
        

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx], _ =slangify_typo(sample)
            else:
                sample.test_case, transformations = slangify_typo(sample.original)
                if sample.task in ("ner", "text-classification"):
                    sample.transformations = transformations
                sample.category = "robustness"

        return sample_list
