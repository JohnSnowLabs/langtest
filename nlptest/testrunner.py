import pandas as pd
from sparknlp.base import LightPipeline
from sklearn.metrics import classification_report

from nlptest.modelhandler import ModelFactory


class TestRunner:
    """
    Base class for running tests on models.
    """

    def __init__(
            self,
            load_testcases: pd.DataFrame,
            model_handler: ModelFactory,
            data: pd.DataFrame
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
        robustness_runner = RobustnessTestRunner(self.load_testcases, self._model_handler, self._data)
        accuracy_runner = AccuracyTestRunner(self.load_testcases, self._model_handler, self._data)

        robustness_result = robustness_runner.evaluate()
        accuracy_result = accuracy_runner.evaluate()

        return robustness_result.merge(accuracy_result, how="outer")


class RobustnessTestRunner(TestRunner):
    """
    Class for running robustness tests on models.
    Subclass of TestRunner.
    """

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate the testcases and return the evaluation results.

        Returns:
            pd.DataFrame: DataFrame containing the evaluation results.
        """

        self.load_testcases["expected_result"] = self.load_testcases["Original"].apply(self._model_handler.predict)
        self.load_testcases["actual_result"] = self.load_testcases["Test_Case"].apply(self._model_handler.predict)

        # Checking for any token mismatches

        final_perturbed_labels = []
        for i, r in self.load_testcases.iterrows():
            main_list = r['Test_Case'].split(' ')
            sub_list = r['Original'].split(' ')

            org_sentence_labels = list(r['expected_result'])
            perturbed_sentence_labels = list(r['actual_result'])

            if len(org_sentence_labels) == len(perturbed_sentence_labels):
                final_perturbed_labels.append(perturbed_sentence_labels)
            else:
                # HACK: it might happen that no sublist are actually found which leads to some mismatch
                # size in the dataframe. Adding an empty list to overcome this until we fix the
                # token mismatch problem
                is_added = False
                for i, _ in enumerate(main_list):
                    if main_list[i:i + len(sub_list)] == sub_list:
                        sub_list_start = i
                        sub_list_end = i + len(sub_list)
                        final_perturbed_labels.append(perturbed_sentence_labels[sub_list_start:sub_list_end])
                        is_added = True
                if not is_added:
                    final_perturbed_labels.append([])

        self._load_testcases['actual_result'] = final_perturbed_labels
        self._load_testcases = self._load_testcases.assign(
            is_pass=self._load_testcases.apply(lambda row: row['expected_result'] == row['actual_result'], axis=1))
        return self._load_testcases


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
        
        y_true = self._data["label"].explode()
        y_pred = [x.label for x in self._model_handler.predict(self._data["text"])]
        y_pred = pd.Series(y_pred).explode()

        print(classification_report(y_true, y_pred))

        res = pd.DataFrame()
        return res
