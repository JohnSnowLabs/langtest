import re
import string
from textwrap import dedent
from typing import List, Mapping, Optional, Tuple
from ..utils.custom_types.helpers import HashableDict

template = """You are a teacher grading a quiz.
You are given a question, the student's answer, and the true answer, and are asked to score the student answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
STUDENT ANSWER: student's answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade the student answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between the student answer and true answer. It is OK if the student answer contains more or relevant information than the true answer, as long as it does not contain any conflicting statements. Begin!

QUESTION: {query}
STUDENT ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:"""
server_prompt = "Perform the task to the best of your ability."
input_variables = ["query", "result", "answer"]


class EvalTemplate:
    """
    The EvalTemplate class provides a method to build a prompt for evaluating student answers
    based on a given rubric. The prompt is designed for a teacher to grade a quiz by comparing
    the student's answer with the true answer and scoring it according to specified criteria.

    Methods
    -------
    build_prompt(rubic_score: Mapping[str, str] = {"CORRECT": None, "INCORRECT": None}) -> str
        Constructs and returns a grading prompt based on the provided rubric scores.

    """

    @staticmethod
    def build_prompt(
        rubic_score: Mapping[str, str] = {
            "CORRECT": None,
            "INCORRECT": None,
        }
    ):
        """ """
        grade_list = list(rubic_score.keys())
        grade_list = ", ".join(grade_list[:-1]) + f" or {grade_list[-1]}"

        eval_criteria = [
            f"{grade_name}: {criteria}\n"
            for grade_name, criteria in rubic_score.items()
            if criteria
        ]
        prompt = (
            "You are a teacher grading a quiz. You are given a question, the student's "
            "answer, and the true answer, and are asked to score the student answer as either "
            f"{grade_list}."
        )

        if eval_criteria:
            eval_criteria = "".join(eval_criteria)
            prompt += dedent(
                f"""\n\nScore the student answer based on the following criteria:\n{eval_criteria}"""
            )

        prompt += dedent(
            f"""
        Example Format:
        QUESTION: question here
        STUDENT ANSWER: student's answer here
        TRUE ANSWER: true answer here
        GRADE: {grade_list} here

        {
            ("Grade the student answers based ONLY on their factual accuracy. Ignore differences"
             " in punctuation and phrasing between the student answer and true answer. It is OK "
             "if the student answer contains more or relevant information than the true answer, as"
             " long as it does not contain any conflicting statements. Begin!")
        }

        QUESTION: {{query}}
        STUDENT ANSWER: {{result}}
        TRUE ANSWER: {{answer}}
        GRADE:"""
        )
        return prompt


class LlmEval:
    """llm_eval for evaluating question answering."""

    def __init__(self, llm, template=template, input_variables=input_variables):
        """
        Initializes the LlmEval object.

        Args:
            llm: The language model for evaluation.
            template: Template for model prompts.
            input_variables: Variables expected in the input.
            server_prompt: Server prompt for model predictions.

        Raises:
            ValueError: If input variables do not match expected values.
        """
        self.llm = llm
        self.template = template
        self.input_variables = input_variables
        self.server_prompt = server_prompt

        expected_input_vars = {"query", "answer", "result"}
        if expected_input_vars != set(self.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {self.input_variables}"
            )

    @staticmethod
    def _get_score(text: str) -> Optional[Tuple[str, int]]:
        match = re.search(r"grade:\s*(correct|incorrect)", text.strip(), re.IGNORECASE)
        if match:
            if match.group(1).upper() == "CORRECT":
                return "CORRECT", 1
            elif match.group(1).upper() == "INCORRECT":
                return "INCORRECT", 0
        try:
            first_word = (
                text.strip()
                .split()[0]
                .translate(str.maketrans("", "", string.punctuation))
            )
            if first_word.upper() == "CORRECT":
                return "CORRECT", 1
            elif first_word.upper() == "INCORRECT":
                return "INCORRECT", 0
            last_word = (
                text.strip()
                .split()[-1]
                .translate(str.maketrans("", "", string.punctuation))
            )
            if last_word.upper() == "CORRECT":
                return "CORRECT", 1
            elif last_word.upper() == "INCORRECT":
                return "INCORRECT", 0
        except IndexError:
            pass
        return None

    @staticmethod
    def _parse_string_eval_output(text: str) -> dict:
        """Parse the output text.

        Args:
            text (str): The output text to parse.

        Returns:
            Any: The parsed output.
        """
        reasoning = text.strip()
        parsed_scores = LlmEval._get_score(reasoning)
        if parsed_scores is None:
            value, score = None, None
        else:
            value, score = parsed_scores
        return {
            "reasoning": reasoning,
            "value": value,
            "score": score,
        }

    def evaluate_example(self, example: dict) -> dict:
        """
        Evaluates a single example using the language model.

        Args:
            example: Dictionary containing input details.

        Returns:
            dict: Evaluation results with reasoning, value, and score.
        """

        output = self.llm.predict(
            prompt=HashableDict(
                **{
                    "template": self.template,
                    "input_variables": self.input_variables,
                }
            ),
            text=HashableDict(**example),
            server_prompt=self.server_prompt,
        )

        parsed_result = self._parse_string_eval_output(output)

        return parsed_result

    def evaluate_batch(self, examples: List[dict]) -> List[dict]:
        """
        Evaluates a batch of examples using the language model.

        Args:
            examples: List of dictionaries containing input details.

        Returns:
            List[dict]: List of evaluation results for each example.
        """
        return [self.evaluate_example(example) for example in examples]

    def evaluate(
        self,
        inputs: List[dict],
        predictions: List[dict],
        question_key: str = "question",
        answer_key: str = "answer",
        prediction_key: str = "result",
    ) -> List[dict]:
        """Evaluate question answering examples and predictions."""
        examples = [
            {
                "query": input_example.get(question_key, ""),
                "answer": input_example.get(answer_key, ""),
                "result": prediction_example.get(prediction_key, ""),
            }
            for input_example, prediction_example in zip(inputs, predictions)
        ]

        return self.evaluate_batch(examples)
