from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import psutil
from langtest.modelhandler import ModelAPI
from langtest.utils.hf_utils import HuggingFacePipeline
import re
import sys
from textwrap import dedent

# check the available of RAM or GPU memory is above 30 GB in current system


def check_memory():
    mem = psutil.virtual_memory()
    if mem.available > 30 * 1024 * 1024 * 1024:
        return True
    else:
        return False


class PrometheusEval:
    """Class for evaluating the Prometheus model."""

    def __init__(
        self,
        model_name: str = "prometheus-eval/prometheus-7b-v2.0",
        hub: str = "huggingface",
        eval_type: str = "absolute_grading",
        criteria_description: Dict[str, str] = None,
        model_kwargs: Dict[str, str] = None,
    ):
        """
        Initializes the PrometheusEval object.

        Args:
            model_name: The name of the model for evaluation.
        """
        self.model_name = model_name
        self.hub = hub
        self.input_variables = ["query", "result", "answer"]
        self.eval_type = eval_type
        self.criteria_description = criteria_description
        self.model_kwargs = model_kwargs

        try:
            # Check if memory is available
            if check_memory():
                self.pipeline = HuggingFacePipeline(
                    model_id=model_name, task="text-generation"
                )
        except MemoryError as e:
            raise MemoryError("Memory is not available to run the model", e)

    def _get_feedback(self, response_text: str) -> Optional[Tuple[str, int]]:
        """
        Get feedback from the text.

        Args:
            text: The text to extract feedback from.

        Returns:
            A tuple of feedback and result.
        """
        feedback = None
        result = None
        feedback_match = re.search(r"###Feedback:", response_text)
        result_match = re.search(r"\[RESULT\]", response_text)

        if feedback_match:
            feedback = response_text[feedback_match.end() : result_match.start()].strip()
        if result_match:
            result = response_text[result_match.end() :].strip()

        if feedback is None:
            feedback = response_text[: result_match.start()].strip()

        return feedback, result

    def evaluate(self, query: str, result: str, answer: str) -> Tuple[str, int]:
        """
        Evaluate the model.

        Args:
            query: The query for the model.
            result: The result from the model.
            answer: The expected answer.

        Returns:
            A tuple of feedback and score.
        """
        if self.eval_type == "absolute_grading":
            build_prompt = AbsoluteGrading(
                instruction=query,
                response=result,
                reference_answer=answer,
                criteria_description=self.criteria_description,
            )
            prompt = build_prompt.get_prompt()

            if self.model_kwargs:
                response = self.pipeline(prompt, **self.model_kwargs)
            else:
                response = self.pipeline(prompt, max_tokens=200, return_full_text=False)

            feedback, result = self._get_feedback(response)
            return feedback, result

    def evaluate_batch(
        self, queries: List[str], results: List[str], answers: List[str]
    ) -> List[Tuple[str, int]]:
        """
        Evaluate the model on a batch of queries.

        Args:
            queries: A list of queries for the model.
            results: A list of results from the model.
            answers: A list of expected answers.

        Returns:
            A list of tuples of feedback and score.
        """
        return [
            self.evaluate(query, result, answer)
            for query, result, answer in zip(queries, results, answers)
        ]


@dataclass
class AbsoluteGrading:
    """Class for absolute grading of the Prometheus model.

    Absolute Grading (Direct Assessment)
    Prometheus requires 4 components in the input: An instruction, a response to evaluate,
    a score rubric, and a reference answer. You could refer to the prompt format below.
    You should fill in the instruction, response, reference answer, criteria description,
    and score description for score in range of n number or True or False.
    """

    instruction: str
    response: str
    reference_answer: str
    criteria_description: Dict[str, str]

    def __post_init__(self):
        self.input_variables = ["instruction", "response", "reference_answer"]

    def get_prompt(self) -> str:
        """
        Get the prompt for the model.

        Returns:
            The prompt for the model.
        """
        s, f = self.get_score_rubric()
        prompt = dedent(
            """
        ###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets from {formatted_criteria_keys}, and a score rubric representing a evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is from {formatted_criteria_keys}. You should refer to the score rubric.
        3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (anyone of this {formatted_criteria_keys})\"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {instruction}

        ###Response to evaluate:
        {response}

        ###Reference Answer (Score in {formatted_criteria_keys}):
        {reference_answer}

        ###Score Rubrics:
        {score_rubric}

        ###Feedback:
        """
        )
        return prompt.format(
            instruction=self.instruction,
            response=self.response,
            reference_answer=self.reference_answer,
            score_rubric=s,
            formatted_criteria_keys=f,
        )

    def get_score_rubric(self) -> Dict[str, str]:
        """
        Get the score rubric for the model.

        Returns:
            The score rubric for the model.
        """
        # Format the criteria keys for the score rubric
        formatted_criteria_keys = ", ".join(
            f"'{i}'" for i in self.criteria_description.keys()
        )
        formatted_criteria_keys = f"[{formatted_criteria_keys}]"
        score_rubric = f"{formatted_criteria_keys}\n"

        # Add criteria and description to the score rubric
        for criteria, description in self.criteria_description.items():
            score_rubric += f"{criteria}: {description}\n"

        return score_rubric, formatted_criteria_keys
