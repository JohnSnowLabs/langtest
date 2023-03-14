import os
import pickle
from collections import defaultdict
from typing import Optional, Union

import pandas as pd
import yaml

from .augmentation.fix_robustness import AugmentRobustness
from .datahandler.datasource import DataFactory
from .modelhandler import ModelFactory
from .testrunner import TestRunner
from .transform import TestFactory


class Harness:
    """ Harness is a testing class for NLP models.

    Harness class evaluates the performance of a given NLP model. Given test data is
    used to test the model. A report is generated with test results.
    """
    SUPPORTED_HUBS = ["spacy", "huggingface", "johnsnowlabs"]

    def __init__(
            self,
            task: Optional[str],
            model: Union[str],
            hub: Optional[str] = None,
            data: Optional[str] = None,
            config: Optional[Union[str, dict]] = None
    ):
        """
        Initialize the Harness object.

        Args:
            task (str, optional): Task for which the model is to be evaluated.
            model (str | ModelFactory): ModelFactory object or path to the model to be evaluated.
            hub (str, optional): model hub to load from the path. Required if path is passed as 'model'.
            data (str, optional): Path to the data to be used for evaluation.
            config (str | dict, optional): Configuration for the tests to be performed.

        Raises:
            ValueError: Invalid arguments.
        """

        super().__init__()
        self.task = task

        if isinstance(model, str):
            if hub is None:
                raise OSError(f"You need to pass the 'hub' parameter when passing a string as 'model'.")

            self.model = ModelFactory.load_model(path=model, task=task, hub=hub)
        else:
            self.model = ModelFactory(task=task, model=model)

        self.data = DataFactory(data, task=self.task).load() if data is not None else None
        self._config = self.configure(config) if config is not None else None

        self._testcases = None
        self._generated_results = None
        self.accuracy_results = None

    def configure(self, config: Union[str, dict]):
        """
        Configure the Harness with a given configuration.

        Args:
            config (str | dict): Configuration file path or dictionary
                for the tests to be performed.

        Returns:
            dict: Loaded configuration.
        """
        if type(config) == dict:
            self._config = config
        else:
            with open(config, 'r') as yml:
                self._config = yaml.safe_load(yml)
        return self._config

    def generate(self) -> "Harness":
        """
        Generates the testcases to be used when evaluating the model.

        Returns:
            None: The generated testcases are stored in `_testcases` attribute.
        """
        tests = self._config['tests']
        self._testcases = TestFactory.transform(self.data, tests)
        return self

    def run(self) -> "Harness":
        """
        Run the tests on the model using the generated testcases.

        Returns:
            None: The evaluations are stored in `generated_results` attribute.
        """
        self._generated_results, self.accuracy_results = TestRunner(self._testcases, self.model,
                                                                    self.data).evaluate()
        return self

    def report(self) -> pd.DataFrame:
        """
        Generate a report of the test results.
        Returns:
            pd.DataFrame: DataFrame containing the results of the tests.
        """
        if isinstance(self._config, dict):
            self.min_pass_dict = {j: k.get('min_pass_rate', 0.65) for i, v in \
                                  self._config['tests'].items() for j, k in v.items()}
        self.default_min_pass_dict = self._config['defaults'].get('min_pass_rate', 0.65)

        summary = defaultdict(lambda: defaultdict(int))
        for sample in self._generated_results:
            summary[sample.test_type]['category'] = sample.category
            summary[sample.test_type][str(sample.is_pass()).lower()] += 1

        report = {}
        for test_type, value in summary.items():
            pass_rate = summary[test_type]["true"] / (summary[test_type]["true"] + summary[test_type]["false"])
            min_pass_rate = self.min_pass_dict.get(test_type, self.default_min_pass_dict)
            report[test_type] = {
                "category": summary[test_type]['category'],
                "fail_count": summary[test_type]["false"],
                "pass_count": summary[test_type]["true"],
                "pass_rate": pass_rate,
                "minimum_pass_rate": min_pass_rate,
                "pass": pass_rate >= min_pass_rate
            }

        df_report = pd.DataFrame.from_dict(report, orient="index")
        df_report = df_report.reset_index(names="test_type")

        df_report['pass_rate'] = df_report['pass_rate'].apply(lambda x: "{:.0f}%".format(x * 100))
        df_report['minimum_pass_rate'] = df_report['minimum_pass_rate'].apply(lambda x: "{:.0f}%".format(x * 100))

        df_accuracy = self.accuracy_report().iloc[:2].drop("test_case", axis=1)
        df_accuracy = df_accuracy.rename({"actual_result": "pass_rate", "expected_result": "minimum_pass_rate"}, axis=1)
        df_accuracy["pass"] = df_accuracy["pass_rate"] >= df_accuracy["minimum_pass_rate"]
        df_accuracy['pass_rate'] = df_accuracy['pass_rate'].apply(lambda x: "{:.0f}%".format(x * 100))
        df_accuracy['minimum_pass_rate'] = df_accuracy['minimum_pass_rate'].apply(lambda x: "{:.0f}%".format(x * 100))
        df_accuracy['category'] = 'Accuracy'  # Temporary fix
        df_final = pd.concat([df_report, df_accuracy])

        col_to_move = 'category'
        first_column = df_final.pop('category')
        df_final.insert(0, col_to_move, first_column)
        df_final = df_final.reset_index(drop=True)

        self.df_report = df_report.fillna("-")

        return self.df_report

        # return self.report_df
    
    def generated_results_df(self) -> pd.DataFrame:
        """
        Generates an overall report with every textcase and labelwise metrics.

        Returns:
            pd.DataFrame: Generated dataframe.
        """
        generated_results_df = pd.DataFrame.from_dict([x.to_dict() for x in self._generated_results])
        accuracy_df = self.accuracy_report()
        final_df = pd.concat([generated_results_df, accuracy_df]).fillna("-")
        final_df = final_df.reset_index(drop=True)

        return final_df

    def accuracy_report(self) -> pd.DataFrame:
        """
        Generate a report of the accuracy results.

        Returns:
            pd.DataFrame: DataFrame containing the accuracy, f1, precision, recall scores.
        """

        acc_report = self.accuracy_results.copy()
        acc_report["expected_result"] = acc_report.apply(
            lambda x: self.min_pass_dict.get(x["test_case"] + x["test_type"], self.min_pass_dict.get('default', 0)),
            axis=1
        )
        acc_report["pass"] = acc_report["actual_result"] >= acc_report["expected_result"]
        return acc_report

    def augment(self, input_path, output_path, inplace=False):

        """
        Augments the data in the input file located at `input_path` and saves the result to `output_path`.

        Args:
            input_path (str): Path to the input file.
            output_path (str): Path to save the augmented data.
            inplace (bool, optional): Whether to modify the input file directly. Defaults to False.

        Returns:
            Harness: The instance of the class calling this method.

        Raises:
            ValueError: If the `pass_rate` or `minimum_pass_rate` columns have an unexpected data type.

        Note:
            This method uses an instance of `AugmentRobustness` to perform the augmentation.

        Example:
            >>> harness = Harness(...)
            >>> harness.augment("train.conll", "augmented_train.conll")
        """

        dtypes = self.df_report[['pass_rate', 'minimum_pass_rate']].dtypes.apply(
            lambda x: x.str).values.tolist()
        if dtypes != ['<i4'] * 2:
            self.df_report['pass_rate'] = self.df_report['pass_rate'].str.replace("%", "").astype(int)
            self.df_report['minimum_pass_rate'] = self.df_report['minimum_pass_rate'].str.replace("%", "").astype(int)
        _ = AugmentRobustness(
            task=self.task,
            config=self._config,
            h_report=self.df_report
        ).fix(
            input_path=input_path,
            output_path=output_path,
            inplace=inplace
        )

        return self

    def testcases(self) -> pd.DataFrame:
        """Testcases after .generate() is called"""
        final_df = pd.DataFrame([x.to_dict() for x in self._testcases]).drop(["pass", "actual_result"], errors="ignore",
                                                                             axis=1)
        final_df = final_df.reset_index(drop=True)
        return final_df

    def save(self, save_dir: str) -> None:
        """
        Save the configuration, generated testcases and the `DataFactory` to be reused later.

        Args:
            save_dir (str): path to folder to save the different files
        Returns:

        """
        if self._config is None:
            raise ValueError("The current Harness has not been configured yet. Please use the `.configure` method "
                             "before calling the `.save` method.")

        if self._testcases is None:
            raise ValueError("The test cases have not been generated yet. Please use the `.generate` method before"
                             "calling the `.save` method.")

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, "config.yaml"), 'w') as yml:
            yml.write(yaml.safe_dump(self._config))

        with open(os.path.join(save_dir, "test_cases.pkl"), "wb") as writer:
            pickle.dump(self._testcases, writer)

        with open(os.path.join(save_dir, "data.pkl"), "wb") as writer:
            pickle.dump(self.data, writer)

    @classmethod
    def load(cls, save_dir: str, task: str, model: Union[str, 'ModelFactory'], hub: str = None) -> 'Harness':
        """
        Loads a previously saved `Harness` from a given configuration and dataset

        Args:
            save_dir (str):
                path to folder containing all the needed files to load an saved `Harness`
            task (str):
                task for which the model is to be evaluated.
            model (str | ModelFactory):
                ModelFactory object or path to the model to be evaluated.
            hub (str, optional):
                model hub to load from the path. Required if path is passed as 'model'.
        Returns:
            Harness:
                `Harness` loaded from from a previous configuration along with the new model to evaluate
        """
        for filename in ["config.yaml", "test_cases.pkl", "data.pkl"]:
            if not os.path.exists(os.path.join(save_dir, filename)):
                raise OSError(f"File '{filename}' is missing to load a previously saved `Harness`.")

        harness = Harness(task=task, model=model, hub=hub)
        harness.configure(os.path.join(save_dir, "config.yaml"))

        with open(os.path.join(save_dir, "test_cases.pkl"), "rb") as reader:
            harness._testcases = pickle.load(reader)

        with open(os.path.join(save_dir, "data.pkl"), "rb") as reader:
            harness.data = pickle.load(reader)

        return harness
