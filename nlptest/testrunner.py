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
