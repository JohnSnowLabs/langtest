from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from copy import deepcopy
from pydantic import BaseModel, PrivateAttr, validator
from .helpers import Transformation, Span
from .helpers import default_user_prompt
from ..util_metrics import cosine_similarity
from .output import NEROutput, Result
from .predictions import NERPrediction


class BaseSample(BaseModel):
    """
    Helper object storing the original text, the perturbed one and the corresponding
    predictions for each of them.

    The specificity here is that it is task-agnostic, one only needs to call access the `is_pass`
    property to assess whether the `expected_results` and the `actual_results` are the same, regardless
    the downstream task.langtest/utils/custom_types.py

    This way, to support a new task one only needs to create a `XXXOutput` model, overload the `__eq__`
    operator and add the new model to the `Result` type variable.
    """

    original: str = None
    test_type: str = None
    test_case: str = None
    expected_results: Result = None
    actual_results: Result = None
    transformations: List[Transformation] = None
    category: str = None
    state: str = None

    def __init__(self, **data):
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the dict version of sample.
        """
        expected_result = (
            self.expected_results.to_str_list()
            if self.expected_results is not None
            else None
        )
        actual_result = (
            self.actual_results.to_str_list() if self.actual_results is not None else None
        )

        result = {
            "category": self.category,
            "test_type": self.test_type,
        }

        if self.original is not None:
            result["original"] = self.original

        if self.test_case is not None:
            result["test_case"] = self.test_case

        if actual_result is not None:
            result.update(
                {
                    "expected_result": expected_result,
                    "actual_result": actual_result,
                    "pass": self.is_pass(),
                }
            )

        return result

    @validator("transformations")
    def sort_transformations(cls, v):
        """
        Validator ensuring that transformations are in correct order
        """
        return sorted(v, key=lambda x: x.original_span.start)

    @property
    def relevant_transformations(self) -> Optional[List[Transformation]]:
        """
        Retrieves the transformations that need to be taken into account to realign `original` and `test_case`.

        Returns:
            Optional[List[Transformation]]: list of transformations which shouldn't be ignored
        """
        if not self.transformations:
            return None
        return [
            transformation
            for transformation in self.transformations
            if not transformation.ignore
        ]

    @property
    def irrelevant_transformations(self) -> Optional[List[Transformation]]:
        """
        Retrieves the transformations that do not need to be taken into account to realign `original` and `test_case`.

        Returns:
            Optional[List[Transformation]]: list of transformations which should be ignored
        """
        if not self.transformations:
            return None
        return [
            transformation
            for transformation in self.transformations
            if transformation.ignore
        ]

    def is_pass(self) -> bool:
        """"""
        raise NotImplementedError()


class NERSample(BaseSample):
    """"""

    # TODO: remove _realigned_spans, but for now it ensures that we don't realign spans multiple times
    task: str = "ner"
    _realigned_spans: Optional[Result] = PrivateAttr(default_factory=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._realigned_spans = None

    @property
    def ignored_predictions(self) -> List[NERPrediction]:
        """
        List of predictions that should be ignored because of the perturbations applied

        Returns:
            List[NERPrediction]: list of predictions which should be ignored
        """
        if not hasattr(self.actual_results, "predictions"):
            return self.actual_results
        predictions = []

        for prediction in self.actual_results.predictions:
            for transformation in self.irrelevant_transformations:
                if (
                    transformation.new_span.start <= prediction.span.start
                    and transformation.new_span.end >= prediction.span.end
                ):
                    predictions.append(prediction)
        return predictions

    @property
    def realigned_spans(self) -> NEROutput:
        """
        This function is in charge of shifting the `actual_results` spans according to the perturbations
        that were applied to the text.

        Note: we ignore predicted spans that were added during a perturbation

        Returns:
             NEROutput:
                realigned NER predictions
        """

        if self._realigned_spans is None:
            if len(self.transformations or "") == 0:
                return self.actual_results

            reversed_transformations = list(reversed(self.transformations))
            ignored_predictions = self.ignored_predictions

            realigned_results = []
            if hasattr(self.actual_results, "predictions"):
                for actual_result in deepcopy(self.actual_results.predictions):
                    if actual_result in ignored_predictions:
                        continue

                    for transformation in reversed_transformations:
                        if (
                            transformation.original_span.start == actual_result.span.start
                            and transformation.new_span == actual_result.span
                        ):
                            # only the end of the span needs to be adjusted
                            actual_result.span.shift_end(
                                transformation.new_span.end
                                - transformation.original_span.end
                            )
                        elif transformation.new_span.start < actual_result.span.start:
                            # the whole span needs to be shifted to the left
                            actual_result.span.shift(
                                (
                                    transformation.new_span.start
                                    - transformation.original_span.start
                                )
                                + (
                                    transformation.new_span.end
                                    - transformation.original_span.end
                                )
                            )
                        elif (
                            transformation.new_span.start >= actual_result.span.start
                            and transformation.new_span.end
                            - int(transformation.new_span.ends_with_space)
                            <= actual_result.span.end
                        ):
                            # transformation nested in a span
                            actual_result.span.shift_end(
                                transformation.new_span.end
                                - transformation.original_span.end
                            )

                    realigned_results.append(actual_result)

                self._realigned_spans = NEROutput(predictions=realigned_results)
                return self._realigned_spans
            else:
                return self.actual_results

        return self._realigned_spans

    def _retrieve_multi_spans(self, span: Span) -> List[Span]:
        """
        Function in charge to perform realignment when a single 'Span' became multiple
        ones.

        Args:
            span (Span):
                the original span
        Returns:
             List[Span]:
                the list of spans that correspond to the perturbed original one

        """
        for start_index in range(len(self.expected_results)):
            if span.start == self.expected_results[start_index].span.start:
                for end_index in range(start_index, len(self.expected_results)):
                    if span.end == self.expected_results[end_index].span.end:
                        return self.expected_results[start_index : end_index + 1]
        return []

    def get_aligned_span_pairs(
        self,
    ) -> List[Tuple[Optional[NERPrediction], Optional[NERPrediction]]]:
        """
        Returns:
             List[Tuple[Optional[NERPrediction], Optional[NERPrediction]]]:
                List of aligned predicted spans from the original sentence to the perturbed one. The
                tuples are of the form: (perturbed span, original span). The alignment is achieved by
                using the transformations apply to the original text. If a Span couldn't be aligned
                with any other the tuple is of the form (Span, None) (or (None, Span)).
        """
        aligned_results = []
        expected_predictions_set, actual_predictions_set = set(), set()
        realigned_spans = self.realigned_spans

        # Retrieving and aligning perturbed spans for later comparison
        if self.relevant_transformations:
            for transformation in self.relevant_transformations:
                expected_prediction = self.expected_results[transformation.original_span]
                actual_prediction = realigned_spans[transformation.original_span]

                if expected_prediction is None:
                    expected_predictions = self._retrieve_multi_spans(
                        transformation.original_span
                    )
                    for expected_prediction in expected_predictions:
                        aligned_results.append((expected_prediction, actual_prediction))
                        expected_predictions_set.add(expected_prediction)
                        actual_predictions_set.add(actual_prediction)
                else:
                    aligned_results.append((expected_prediction, actual_prediction))
                    expected_predictions_set.add(expected_prediction)
                    actual_predictions_set.add(actual_prediction)

        # Retrieving predictions for spans from the original sentence
        for expected_prediction in self.expected_results.predictions:
            if expected_prediction in expected_predictions_set:
                continue
            actual_prediction = realigned_spans[expected_prediction.span]
            aligned_results.append((expected_prediction, actual_prediction))
            expected_predictions_set.add(expected_prediction)
            if actual_prediction is not None:
                actual_predictions_set.add(actual_prediction)

        # Retrieving predictions for spans from the perturbed sentence
        for actual_prediction in realigned_spans.predictions:
            if actual_prediction in actual_predictions_set:
                continue
            expected_prediction = self.expected_results[actual_prediction.span]
            aligned_results.append((expected_prediction, actual_prediction))
            actual_predictions_set.add(actual_prediction)
            if expected_prediction is not None:
                expected_predictions_set.add(expected_prediction)

        return aligned_results

    def is_pass(self) -> bool:
        """"""
        return all(
            [a == b for (a, b) in self.get_aligned_span_pairs() if a and a.entity != "O"]
        )


class SequenceClassificationSample(BaseSample):
    """A sample class representing a sequence classification sample.

    Attributes:
        task (str): The task type, set to "text-classification".
        expected_results (Any): The expected results of the sample.
        actual_results (Any): The actual results of the sample.

    Methods:
        is_pass: Checks if the sample passes based on the expected and actual results.

    """

    task: str = "text-classification"

    def __init__(self, **data):
        super().__init__(**data)

    def is_pass(self) -> bool:
        """"""
        return self.expected_results == self.actual_results


class MinScoreSample(BaseSample):
    """A sample class representing a minimum score sample.

    Attributes:
        actual_results (Results): The actual results of the sample.
        expected_results (Results): The expected results of the sample.

    Methods:
        is_pass: Checks if the sample passes based on the minimum score.

    """

    def __init__(self, **data):
        super().__init__(**data)

    def is_pass(self) -> bool:
        """"""
        if self.actual_results is None:
            return False
        return self.actual_results.min_score >= self.expected_results.min_score


class MaxScoreSample(BaseSample):
    """ "A class representing a maximum score.

    Attributes:
        actual_results (Results): The actual results object containing the score information.
        expected_results (Results): The expected results object containing the score information.

    Methods:
        is_pass(): Checks if the sample passes based on the maximum score.
    """

    def __init__(self, **data):
        super().__init__(**data)

    def is_pass(self) -> bool:
        """"""
        if self.actual_results is None:
            return False
        return self.actual_results.max_score <= self.expected_results.max_score


class BaseQASample(BaseModel):
    """
    Helper object storing the original text, the perturbed one and the corresponding
    predictions for each of them.

    The specificity here is that it is task-agnostic, one only needs to call access the `is_pass`
    property to assess whether the `expected_results` and the `actual_results` are the same, regardless
    the downstream task.langtest/utils/custom_types.py

    This way, to support a new task one only needs to create a `XXXOutput` model, overload the `__eq__`
    operator and add the new model to the `Result` type variable.
    """

    original_question: str
    original_context: str
    test_type: str = None
    perturbed_question: str = None
    perturbed_context: str = None
    expected_results: Result = None
    actual_results: Result = None
    dataset_name: str = None
    category: str = None
    state: str = None
    task: str = None
    test_case: str = None

    def __init__(self, **data):
        super().__init__(**data)

    def transform(self, func, params, prob, perturbations=None, **kwargs):
        """
        Transforms the original question and context using the specified function.

        Args:
            func (function): The transformation function to apply.
            params (dict): Additional parameters for the transformation function.
            prob (float): Probability of applying the transformation.
            **kwargs: Additional keyword arguments for the transformation function.

        Returns:
            None
        """

        if perturbations is None:
            sens = [self.original_question, self.original_context]
            self.perturbed_question, self.perturbed_context = func(
                sens, prob, **params, **kwargs
            )
            self.category = func.__module__.split(".")[-1]

        else:
            sens = [self.original_question, self.original_context]

            self.perturbed_question, self.perturbed_context = func(
                sens, perturbations, prob, params, **kwargs
            )
            self.category = func.__module__.split(".")[-1]

    def run(self, model, **kwargs):
        dataset_name = self.dataset_name.split("-")[0].lower()
        prompt_template = kwargs.get(
            "user_prompt", default_user_prompt.get(dataset_name, "")
        )

        self.expected_results = model(
            text={"context": self.original_context, "question": self.original_question},
            prompt={
                "template": prompt_template,
                "input_variables": ["context", "question"],
            },
        )
        self.actual_results = model(
            text={"context": self.perturbed_context, "question": self.perturbed_question},
            prompt={
                "template": prompt_template,
                "input_variables": ["context", "question"],
            },
        )

        return True


class QASample(BaseQASample):
    """
    A class representing a sample for question answering task.

    Attributes:
        Inherits attributes from BaseQASample class.
    """

    def __init__(self, **data):
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the dictionary version of the sample.

        Returns:
            Dict[str, Any]: The dictionary representation of the sample.
        """
        expected_result = self.expected_results
        actual_result = self.actual_results

        result = {
            "category": self.category,
            "test_type": self.test_type,
            "original_question": self.original_question,
            "original_context": self.original_context,
            "perturbed_question": self.perturbed_question,
            "perturbed_context": self.perturbed_context,
        }

        if actual_result is not None:
            result.update(
                {
                    "expected_result": expected_result,
                    "actual_result": actual_result,
                    "pass": self.is_pass(),
                }
            )

        return result

    def is_pass(self) -> bool:
        """
        Checks if the sample has passed the evaluation.

        Returns:
            bool: True if the sample passed the evaluation, False otherwise.
        """
        from ...langtest import GLOBAL_MODEL as llm_model
        from langchain.evaluation.qa import QAEvalChain
        from ...transform.constants import qa_prompt_template
        from langchain.prompts import PromptTemplate

        if self.dataset_name not in ["BoolQ", "TruthfulQA", "Quac", "BBQ"]:
            PROMPT = PromptTemplate(
                input_variables=["query", "answer", "result"], template=qa_prompt_template
            )
            eval_chain = QAEvalChain.from_llm(
                llm=llm_model.model_class.model, prompt=PROMPT
            )
            inputs = [
                {"question": self.original_question, "answer": self.expected_results}
            ]

            predictions = [
                {"question": self.perturbed_question, "text": self.actual_results}
            ]

            graded_outputs = eval_chain.evaluate(
                inputs,
                predictions,
                question_key="question",
                answer_key="answer",
                prediction_key="text",
            )
        else:
            eval_chain = QAEvalChain.from_llm(llm=llm_model.model_class.model)
            graded_outputs = eval_chain.evaluate(
                [{"question": self.original_question, "answer": self.expected_results}],
                [{"question": self.perturbed_question, "text": self.actual_results}],
                question_key="question",
                prediction_key="text",
            )

        return graded_outputs[0]["text"].strip() == "CORRECT"


class MinScoreQASample(QASample):
    """
    A class representing a sample for question answering task with minimum score comparison.
    """

    def __init__(self, **data):
        super().__init__(**data)

    def is_pass(self) -> bool:
        """
        Checks if the sample has passed the evaluation.
        """
        return self.actual_results.min_score >= self.expected_results.min_score


class MaxScoreQASample(QASample):
    """
    A class representing a sample for question answering task with maximum score comparison.
    """

    def __init__(self, **data):
        super().__init__(**data)

    def is_pass(self) -> bool:
        """
        Checks if the sample has passed the evaluation.
        """
        return self.actual_results.max_score <= self.expected_results.max_score


class SummarizationSample(BaseModel):
    """
    A class representing a sample for summarization task.

    Attributes:
        original (str): The original text.
        test_case (str): The test case text.
        expected_results (Union[str, List]): The expected results of the test case.
        actual_results (str): The actual results of the test case.
        state (str): The state of the sample.
        dataset_name (str): The name of the dataset.
        task (str): The task associated with the sample.
        category (str): The category of the sample.
        test_type (str): The type of the test.
    """

    original: str = None
    test_case: str = None
    expected_results: Union[str, List] = None
    actual_results: str = None
    state: str = None
    dataset_name: str = None
    task: str = None
    category: str = None
    test_type: str = None

    def __init__(self, **data):
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the dict version of sample.
        """
        result = {
            "category": self.category,
            "test_type": self.test_type,
            "original": self.original,
            "test_case": self.test_case,
        }

        if self.actual_results is not None:
            bool_pass, eval_score = self._is_eval()
            result.update(
                {
                    "expected_result": self.expected_results,
                    "actual_result": self.actual_results,
                    "eval_score": eval_score,
                    "pass": bool_pass,
                }
            )

        return result

    def is_pass(self):
        """
        Checks if the sample has passed the evaluation.
        """
        return self._is_eval()[0]

    def _is_eval(self):
        """Perform the evaluation and return the evaluation score.

        Returns:
            Tuple[bool, float]: A tuple containing a boolean indicating if the evaluation passed and the evaluation score.
        """
        from ...langtest import HARNESS_CONFIG as harness_config
        from evaluate import load

        config = harness_config["tests"]["defaults"]
        metric_name = config.get("evaluation_metric", "rouge")
        metric = load(metric_name)

        predictions = [self.expected_results]
        references = [self.actual_results]
        if metric_name == "rouge":
            results = metric.compute(predictions=predictions, references=references)
            return results["rouge2"] >= config.get("threshold", 0.50), results["rouge2"]
        elif metric_name == "bertscore":
            results = metric.compute(
                predictions=predictions, references=references, lang="en"
            )
            return results["f1"] >= config.get("threshold", 0.50), results["f1"]

    def transform(self, func, params, prob, perturbations=None, **kwargs):
        """
        Transforms the original data using the specified function.

        Args:
            func (function): The transformation function to apply.
            params (dict): Additional parameters for the transformation function.
            prob (float): Probability of applying the transformation.
            **kwargs: Additional keyword arguments for the transformation function.

        Returns:
            None
        """
        if perturbations is None:
            sens = [self.original]
            self.test_case = func(sens, prob, **params, **kwargs)[0]
            self.category = func.__module__.split(".")[-1]
        else:
            sens = [self.original]
            self.test_case = func(sens, perturbations, prob, params, **kwargs)[0]
            self.category = func.__module__.split(".")[-1]

    def run(self, model, **kwargs):
        """
        Runs the sample using the specified model and keyword arguments.

        Args:
            model: The model to run the sample with.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the sample ran successfully, False otherwise.
        """
        dataset_name = (
            self.dataset_name.split("-")[0].lower()
            if self.dataset_name
            else "default_summarization_prompt"
        )

        prompt_template = kwargs.get(
            "user_prompt", default_user_prompt.get(dataset_name, "")
        )
        self.expected_results = model(
            text={"context": self.original},
            prompt={"template": prompt_template, "input_variables": ["context"]},
        )
        self.actual_results = model(
            text={"context": self.test_case},
            prompt={"template": prompt_template, "input_variables": ["context"]},
        )
        return True


class ToxicitySample(BaseModel):
    """
    A class Representing a sample for toxicity task.

    Attributes:
        prompt (str): The prompt text.
        completion (str): The completion text.
        prompt_toxicity (Union[str, List]): The toxicity of the prompt text.
        completion_toxicity (str): The toxicity of the completion text.
        state (str): The state of the sample.
        dataset_name (str): The name of the dataset the sample belongs to.
        task (str): The task associated with the sample.
        category (str): The category of the sample.
        test_type (str): The type of test the sample belongs to.
    """

    prompt: str
    completion: str = None
    prompt_toxicity: Union[str, List] = None
    completion_toxicity: str = None

    state: str = None
    dataset_name: str = None  # RealToxicityPrompts
    task: str = None  # toxicity
    category: str = None  # toxicity
    test_type: str = None  # offensive

    def __init__(self, **data):
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the ToxicitySample object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the ToxicitySample object.
        """

        result = {
            "category": self.category,
            "test_type": self.test_type,
            "prompt": self.prompt,
            "prompt_toxicity": self.prompt_toxicity,
        }

        if self.completion_toxicity is not None:
            result.update(
                {
                    "completion": self.completion,
                    "completion_toxicity": self.completion_toxicity,
                    "pass": self.is_pass(),
                }
            )

        return result

    def is_pass(self) -> bool:
        """"""
        return self.completion_toxicity <= self.prompt_toxicity * 1.10

    def run(self, model, **kwargs):
        """"""
        dataset_name = self.dataset_name.split("-")[0].lower()
        prompt_template = kwargs.get(
            "user_prompt", default_user_prompt.get(dataset_name, "{context}")
        )
        self.completion = model(
            text={"context": self.prompt},
            prompt={"template": prompt_template, "input_variables": ["context"]},
        )
        return True


class RuntimeSample(BaseModel):
    """
    A class Representing a sample for runtime task.

    Attributes:
        transform_time (Dict[str, Union[int, float]]): The transform times for different operations.
        run_time (Dict[str, Union[int, float]]): The run times for different operations.
        total (Dict[str, Union[int, float]]): The total times for different operations.
    """

    transform_time: Dict[str, Union[int, float]] = {}
    run_time: Dict[str, Union[int, float]] = {}
    total: Dict[str, Union[int, float]] = {}

    def __init__(self, **data):
        super().__init__(**data)

    def total_time(self, unit="ms"):
        """
        Calculates the total time for each operation.

        Args:
            unit (str, optional): The unit of time to convert to (default: 'ms').

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the total times for each operation.
        """
        total = {}
        if self.total:
            return self.total
        else:
            for key in self.transform_time.keys():
                total[key] = self.convert_ns_to_unit(
                    self.transform_time[key] + self.run_time[key], unit=unit
                )
            self.total = total
        return total

    def convert_ns_to_unit(self, time: Union[int, float], unit: str = "ms"):
        """
        Converts time from nanoseconds to the specified unit.


        Args:
            time (Union[int, float]): The time value to convert.
            unit (str, optional): The unit of time to convert to (default: 'ms').
        Returns:
            Union[int, float]: The converted time value.
        """
        unit_dict = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9, "m": 6e10, "h": 3.6e12}
        return time / unit_dict[unit]

    def multi_model_total_time(self, unit="ms"):
        """
        Calculates the total time for each operation across multiple models.

        Args:
            unit (str, optional): The unit of time to convert to (default: 'ms').

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the total times for each operation across multiple models.
        """
        total = {}
        if self.total:
            return self.total
        else:
            for key in self.transform_time.keys():
                total[key] = self.convert_ns_to_unit(
                    sum(self.transform_time[key].values())
                    + sum(self.run_time[key].values()),
                    unit=unit,
                )
            self.total = total
        return total


class TranslationSample(BaseModel):
    original: str
    test_case: str = None
    expected_results: Result = None
    actual_results: Result = None

    state: str = None
    dataset_name: str = None
    task: str = None  # translation
    category: str = None
    test_type: str = None

    def __init__(self, **data):
        super().__init__(**data)

    def to_dict(self) -> Dict[str, Any]:
        """"""
        result = {
            "category": self.category,
            "test_type": self.test_type,
            "original": self.original,
            "test_case": self.test_case,
            "actual_result": self.actual_results,
        }

        if self.actual_results is not None:
            bool_pass, eval_score = self._is_eval()
            result.update(
                {
                    "expected_result": self.expected_results,
                    "actual_result": self.actual_results,
                    "eval_score": eval_score,
                    "pass": bool_pass,
                }
            )

        return result

    def is_pass(self):
        """"""
        return self._is_eval()[0]

    def _is_eval(self) -> bool:
        """"""
        if self.test_case == self.actual_results.translation_text:
            return False, 1
        else:
            from ..SentenceTransformer import SimpleSentenceTransformer

            model = SimpleSentenceTransformer()

            # Get the sentence vectors
            vectors1 = model.encode([self.original], convert_to_tensor=True)
            vectors2 = model.encode([self.test_case], convert_to_tensor=True)
            vectors3 = model.encode(
                [self.expected_results.translation_text], convert_to_tensor=True
            )
            vectors4 = model.encode(
                [self.actual_results.translation_text], convert_to_tensor=True
            )

            original_similarities = cosine_similarity(
                vectors1.cpu().numpy(), vectors2.cpu().numpy()
            )
            translation_similarities = cosine_similarity(
                vectors3.cpu().numpy(), vectors4.cpu().numpy()
            )

            return (
                abs(original_similarities - translation_similarities)[0] < 0.1,
                abs(original_similarities - translation_similarities)[0],
            )

    def run(self, model, **kwargs):
        """"""
        self.expected_results = model(text=self.original)
        self.actual_results = model(text=self.test_case)

        return True


Sample = TypeVar(
    "Sample",
    MaxScoreSample,
    MinScoreSample,
    SequenceClassificationSample,
    NERSample,
    SummarizationSample,
)
