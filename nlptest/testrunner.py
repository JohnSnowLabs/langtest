import pandas as pd
from sklearn.metrics import classification_report, f1_score

from nlptest.modelhandler import ModelFactory
from typing import List, Tuple

from nlptest.utils.custom_types import Sample


class TestRunner:
    """
    Base class for running tests on models.
    """

    def __init__(
        self,
        load_testcases: List[Sample],
        model_handler: ModelFactory,
        data: List[Sample]
    ) -> None:
        """
        Initialize the TestRunner class.

        Args:
            load_testcases (List): List containing the testcases to be evaluated.
            model_handler (spark, spacy, transformer): Object representing the
            model handler, either spaCy, SparkNLP or transformer.
        """
        self.load_testcases = load_testcases.copy()
        self._model_handler = model_handler
        self._data = data

    # @abc.abstractmethod
    def evaluate(self) -> Tuple[List[Sample], pd.DataFrame]:
        """Abstract method to evaluate the testcases.

        Returns:
            Tuple[List[Sample], pd.DataFrame]
        """
        # self._model_handler.load_model()

        robustness_runner = RobustnessTestRunner(
            self.load_testcases, self._model_handler, self._data)
        robustness_result = robustness_runner.evaluate()

        # accuracy_runner = AccuracyTestRunner(self.load_testcases, self._model_handler, self._data)
        # accuracy_result = accuracy_runner.evaluate()

        return robustness_result


class RobustnessTestRunner(TestRunner):
    """Class for running robustness tests on models.
    Subclass of TestRunner.
    """

    def evaluate(self):
        """Evaluate the testcases and return the evaluation results.

        Returns:
            List[Sample]:
                all containing the predictions for both the original text and the pertubed one
        """
        for sample in self.load_testcases:
            if sample.state != "done":
                sample.expected_results = self._model_handler(sample.original)
                sample.actual_results = self._model_handler(sample.test_case)
                sample.state = "done"

        return self.load_testcases
