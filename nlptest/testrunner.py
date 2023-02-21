import pandas as pd
from sparknlp.base import LightPipeline
from sklearn.metrics import classification_report, f1_score
from functools import reduce

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
        robustness_result = robustness_runner.evaluate()

        accuracy_runner = AccuracyTestRunner(self.load_testcases, self._model_handler, self._data)
        accuracy_result = accuracy_runner.evaluate()

        return robustness_result, accuracy_result.fillna("")


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

        self.load_testcases['actual_result'] = final_perturbed_labels
        self.load_testcases = self.load_testcases.assign(
            is_pass=self.load_testcases.apply(lambda row: row['expected_result'] == row['actual_result'], axis=1))
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
        
        y_true = self._data["label"]
        y_pred = self._data["text"].apply(self._model_handler.predict)

        valid_indices = y_true.apply(len) == y_pred.apply(len)

        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

        y_true = y_true.explode().apply(lambda x: x.split("-")[-1])
        y_pred = y_pred.explode()

        y_pred = [x.entity.split("-")[-1] for x in y_pred.tolist()]

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
            "Test_Case": ["-", "-"],
            "Test_type": ["micro-f1", "macro-f1"],
            "actual_result": [micro_f1_score, macro_f1_score],
        }
        df_melted = pd.concat([pd.DataFrame(other_metrics), df_melted])

        return df_melted
