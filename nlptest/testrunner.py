import pandas as pd
from .modelhandler import ModelFactory


class TestRunner:
    """Base class for running tests on models.
    """

    def __init__(
            self,
            load_testcases: pd.DataFrame,
            model_handler: ModelFactory,
    ) -> None:
        """Initialize the TestRunner class.

        Args:
            load_testcases (pd.DataFrame): DataFrame containing the testcases to be evaluated.
            model_handler (spark or spacy model): Object representing the model handler, either spaCy or SparkNLP.
        """
        self.load_testcases = load_testcases.copy()
        self._model_handler = model_handler

    # @abc.abstractmethod
    def evaluate(self):
        """Abstract method to evaluate the testcases.

        Returns:
            DataFrame: DataFrame containing the evaluation results.
        """
        runner = RobustnessTestRunner(self.load_testcases, self._model_handler)
        return runner.evaluate()


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
            sample.expected_results = self._model_handler(sample.original)
            sample.actual_results = self._model_handler(sample.test_case)

        return self.load_testcases
