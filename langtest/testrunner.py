import pandas as pd
from tqdm import tqdm
from langtest.modelhandler import ModelFactory
from typing import List, Tuple
from langtest.utils.custom_types import Sample


class BaseRunner:
    """Base class for running tests on models."""

    def __init__(self, load_testcases: List[Sample], model_handler: ModelFactory) -> None:
        """Initialize the BaseRunner class.

        Args:
            load_testcases (List[Sample]): List containing the testcases to be evaluated.
            model_handler (ModelFactory): Object representing the model handler, either spaCy, SparkNLP or transformer.
        """
        self.load_testcases = load_testcases.copy()
        self._model_handler = model_handler

    # @abc.abstractmethod
    def evaluate(self) -> Tuple[List[Sample], pd.DataFrame]:
        """Abstract method to evaluate the testcases.

        Returns:
            Tuple[List[Sample], pd.DataFrame]
        """
        test_result = TestRunner(self.load_testcases, self._model_handler).evaluate()
        return test_result


class TestRunner(BaseRunner):
    """Class for running robustness tests on models."""

    def evaluate(self):
        """Evaluate the testcases and return the evaluation results.

        Returns:
            List[Sample]:
                all containing the predictions for both the original text and the perturbed one
        """
        for sample in tqdm(self.load_testcases, desc="Running test cases..."):
            if sample.state != "done":
                if sample.category not in ["Robustness", "Bias"]:
                    sample.expected_results = self._model_handler(sample.original)
                sample.actual_results = self._model_handler(sample.test_case)
                sample.state = "done"

        return self.load_testcases
