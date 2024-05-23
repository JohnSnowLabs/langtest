import asyncio
from abc import ABC, abstractmethod
from ..errors import Errors
from typing import List, Optional
from langtest.modelhandler import ModelAPI
from ..utils.custom_types import Sample
import random


class BaseSensitivity(ABC):
    """Abstract base class for implementing sensitivity measures.

    Attributes:
        alias_name (str): A name or list of names that identify the sensitivity measure.

    Methods:
        transform(data: List[Sample]) -> Any: Transforms the input data into an output based on the implemented sensitivity measure.
    """

    alias_name = None
    supported_tasks = [
        "sensitivity",
        "question-answering",
    ]

    @staticmethod
    @abstractmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        """Abstract method that implements the sensitivity measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.

        Returns:
            Any: The transformed data based on the implemented sensitivity measure.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def run(sample_list: List[Sample], model: ModelAPI, **kwargs) -> List[Sample]:
        """Abstract method that implements the sensitivity measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the sensitivity measure.

        Returns:
            List[Sample]: The transformed data based on the implemented sensitivity measure.

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
    async def async_run(cls, sample_list: List[Sample], model: ModelAPI, **kwargs):
        """Creates a task to run the sensitivity measure.

        Args:
            sample_list (List[Sample]): The input data to be transformed.
            model (ModelAPI): The model to be used for evaluation.
            **kwargs: Additional arguments to be passed to the sensitivity measure.

        Returns:
            asyncio.Task: The task that runs the sensitivity measure.

        """
        created_task = asyncio.create_task(cls.run(sample_list, model, **kwargs))
        return created_task


class AddNegation(BaseSensitivity):
    """A class for negating sensitivity-related phrases in the input text.

    This class identifies common sensitivity-related phrases such as 'is', 'was', 'are', and 'were' in the input text
    and replaces them with their negations to make the text less sensitive.

    Attributes:
        alias_name (str): The alias name for this sensitivity transformation.

    Methods:
        transform(sample_list: List[Sample]) -> List[Sample]: Applies the sensitivity negation transformation to a list
            of samples.
    """

    alias_name = "add_negation"

    @staticmethod
    def transform(sample_list: List[Sample]) -> List[Sample]:
        def apply_transformation(perturbed_text):
            """
            Filters the dataset for 'is', 'was', 'are', or 'were' and negates them.

            Args:
                perturbed_text (str): The input text to be transformed.

            Returns:
                str: The transformed text with sensitivity-related phrases negated.
            """
            transformations = {
                " is ": " is not ",
                " was ": " was not ",
                " are ": " are not ",
                " were ": " were not ",
            }

            for keyword, replacement in transformations.items():
                if keyword in perturbed_text and replacement not in perturbed_text:
                    perturbed_text = perturbed_text.replace(keyword, replacement, 1)

            return perturbed_text

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = apply_transformation(sample)

        return sample_list


class AddToxicWords(BaseSensitivity):
    """A class for handling sensitivity-related phrases in the input text, specifically related to toxicity.

    Attributes:
        alias_name (str): The alias name for this sensitivity transformation.

    Methods:
        transform(
            sample_list: List[Sample],
            starting_context: Optional[List[str]] = None,
            ending_context: Optional[List[str]] = None,
            strategy: str = None,
        ) -> List[Sample]: Transform the input list of samples to add toxicity-related text.

    Raises:
        ValueError: If an invalid context strategy is provided.
    """

    alias_name = "add_toxic_words"

    @staticmethod
    def transform(
        sample_list: List[Sample],
        starting_context: Optional[List[str]] = None,
        ending_context: Optional[List[str]] = None,
        strategy: str = None,
    ) -> List[Sample]:
        """
        Transform the input list of samples to add toxicity-related text.

        Args:
            sample_list (List[Sample]): A list of samples to transform.
            starting_context (Optional[List[str]]): A list of starting context tokens.
            ending_context (Optional[List[str]]): A list of ending context tokens.
            strategy (str): The strategy for adding context. Can be 'start', 'end', or 'combined'.

        Returns:
            List[Sample]: The transformed list of samples.

        Raises:
            ValueError: If an invalid context strategy is provided.
        """

        def context(text, strategy):
            possible_methods = ["start", "end", "combined"]
            if strategy is None:
                strategy = random.choice(possible_methods)
            elif strategy not in possible_methods:
                raise ValueError(Errors.E066(strategy=strategy))

            if strategy == "start" or strategy == "combined":
                add_tokens = random.choice(starting_context)
                add_string = (
                    " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens
                )
                if text != "-":
                    text = add_string + " " + text

            if strategy == "end" or strategy == "combined":
                add_tokens = random.choice(ending_context)
                add_string = (
                    " ".join(add_tokens) if isinstance(add_tokens, list) else add_tokens
                )

                if text != "-":
                    text = text + " " + add_string

            return text

        for idx, sample in enumerate(sample_list):
            if isinstance(sample, str):
                sample_list[idx] = context(sample, strategy)

        return sample_list
