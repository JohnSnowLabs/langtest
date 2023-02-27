import pandas as pd
from sparknlp.base import LightPipeline
from sklearn.metrics import classification_report, f1_score
from functools import reduce

from nlptest.modelhandler import ModelFactory
from typing import List

from nlptest.utils.custom_types import Sample

class TestRunner:
    """
    Base class for running tests on models.
    """

    def __init__(
            self,
            load_testcases: pd.DataFrame,
            model_handler: ModelFactory,
            data: List[Sample]
    ) -> None:
        """
        Initialize the TestRunner class.

        Args:
            load_testcases (pd.DataFrame): DataFrame containing the testcases to be evaluated.
            model_handler (spark or spacy model): Object representing the model handler, either spaCy or SparkNLP.
        """
        self.load_testcases = load_testcases.copy()
        self._model_handler = model_handler
        self._data = data

    # @abc.abstractmethod
    def evaluate(self):
        """Abstract method to evaluate the testcases.

        Returns:
            DataFrame: DataFrame containing the evaluation results.
        """
        self._model_handler.load_model()

        robustness_runner = RobustnessTestRunner(self.load_testcases, self._model_handler, self._data)
        robustness_result = robustness_runner.evaluate()

        accuracy_runner = AccuracyTestRunner(self.load_testcases, self._model_handler, self._data)
        accuracy_result = accuracy_runner.evaluate()

        return robustness_result, accuracy_result.fillna("")


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
    
class AccuracyTestRunner(TestRunner):
    """
    Class for running accuracy related tests on models.
    Subclass of TestRunner.
    """

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluates the model's accuracy, precision, recall and f1-score per label and
        macro-f1, micro-f1 and total accuracy. 

        Returns:
            pd.Dataframe: Dataframe with the results.
        """
        y_true = pd.Series(self._data).apply(lambda x: [y.entity for y in x.expected_results.predictions])
        X_test = pd.Series(self._data).apply(lambda x: x.original)
        y_pred = X_test.apply(lambda x: self._model_handler.predict(x, predict_str=True))

        valid_indices = y_true.apply(len) == y_pred.apply(len)

        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

        y_true = y_true.explode().apply(lambda x: x.split("-")[-1])
        y_pred = y_pred.explode().apply(lambda x: x.split("-")[-1])

        y_pred = [x.split("-")[-1] for x in y_pred.tolist()]

        # if(len(y_pred) != len(y_true)):
            # raise ValueError("Please use the dataset used to train/test the model. Model and dataset has different tokenizers.")

        df_metrics = classification_report(y_true, y_pred, output_dict=True)
        df_metrics.pop("accuracy")
        df_metrics.pop("macro avg")
        df_metrics.pop("weighted avg")
        df_metrics = pd.DataFrame(df_metrics).drop("support")

        micro_f1_score = f1_score(y_true, y_pred, average="micro")
        macro_f1_score = f1_score(y_true, y_pred, average="macro")

        df_melted = pd.melt(df_metrics.reset_index(), id_vars=['index'], var_name='label', value_name='test value')
        df_melted.columns = ['Test_type', 'Test_Case', 'actual_result']

        other_metrics = {
            "test_case": ["-", "-"],
            "test_type": ["micro-f1", "macro-f1"],
            "actual_result": [micro_f1_score, macro_f1_score],
        }
        df_melted = pd.concat([pd.DataFrame(other_metrics), df_melted])

        return df_melted
