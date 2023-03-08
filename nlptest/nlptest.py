from collections import defaultdict
from functools import reduce
from typing import Dict, Optional, Union

import pandas as pd
import yaml

from .datahandler.datasource import DataFactory
from .modelhandler import ModelFactory
from .testrunner import TestRunner
from .transform.perturbation import PerturbationFactory


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
            self.model = ModelFactory.load_model(path=model, task=task, hub=hub)
        else:
            self.model = ModelFactory(task=task, model=model)

        if data is not None:
            if type(data) == str:
                self.data = DataFactory(data).load()

        if config is not None:
            self._config = self.configure(config)
        else:
            self._config = None

        self.load_testcases = None
        self.generated_results = None
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
            None: The generated testcases are stored in `load_testcases` attribute.
        """
        tests = self._config['tests_types']
        self.load_testcases = PerturbationFactory(self.data, tests).transform()
        return self

    def run(self) -> "Harness":
        """
        Run the tests on the model using the generated testcases.

        Returns:
            None: The evaluations are stored in `generated_results` attribute.
        """
        self.generated_results, self.accuracy_results = TestRunner(self.load_testcases, self.model, self.data).evaluate()
        return self

    def report(self) -> pd.DataFrame:
        """
        Generate a report of the test results.

        Returns:
            pd.DataFrame: DataFrame containing the results of the tests.
        """
        if isinstance(self._config['min_pass_rate'], list):
            min_pass_dict = reduce(lambda x, y: {**x, **y}, self._config['min_pass_rate'])
        else:
            min_pass_dict = self._config['min_pass_rate']

        summary = defaultdict(lambda: defaultdict(int))
        for sample in self.generated_results:
            summary[sample.test_type][str(sample.is_pass()).lower()] += 1

        report = {}
        for test_type, value in summary.items():
            pass_rate = summary[test_type]["true"] / (summary[test_type]["true"] + summary[test_type]["false"])
            min_pass_rate = min_pass_dict.get(test_type, min_pass_dict["default"])
            report[test_type] = {
                "fail_count": summary[test_type]["false"],
                "pass_count": summary[test_type]["true"],
                "pass_rate": pass_rate,
                "minimum_pass_rate": min_pass_rate,
                "pass": pass_rate >= min_pass_rate
            }

        df_report = pd.DataFrame.from_dict(report, orient="index")
        df_report = df_report.reset_index(names="test_type")

        df_report['pass_rate'] = df_report['pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))
        df_report['minimum_pass_rate'] = df_report['minimum_pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))
        
        df_accuracy = self.accuracy_report().iloc[:2].drop("test_case", axis=1)
        df_accuracy = df_accuracy.rename({"actual_result":"pass_rate", "expected_result":"minimum_pass_rate"}, axis=1)
        df_accuracy["pass"] = df_accuracy["pass_rate"] >= df_accuracy["minimum_pass_rate"]
        df_accuracy['pass_rate'] = df_accuracy['pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))
        df_accuracy['minimum_pass_rate'] = df_accuracy['minimum_pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))

        df_final = pd.concat([df_report, df_accuracy])


        return df_final.fillna("-")
    
    def generated_results_df(self) -> pd.DataFrame:
        """
        Generates an overall report with every textcase and labelwise metrics.

        Returns:
            pd.DataFrame: Generated dataframe.
        """
        generated_results_df = pd.DataFrame.from_dict([x.to_dict() for x in self.generated_results])
        accuracy_df = self.accuracy_report()

        return pd.concat([generated_results_df,accuracy_df]).fillna("-")

    def accuracy_report(self) -> pd.DataFrame:
        """
        Generate a report of the accuracy results.

        Returns:
            pd.DataFrame: DataFrame containing the accuracy, f1, precision, recall scores.
        """

        if isinstance(self._config['min_pass_rate'], list):
            min_pass_dict = reduce(lambda x, y: {**x, **y}, self._config['min_pass_rate'])
        else:
            min_pass_dict = self._config['min_pass_rate']
        acc_report = self.accuracy_results.copy()
        acc_report["expected_result"] = acc_report.apply(
            lambda x: min_pass_dict.get(x["test_case"]+x["test_type"], min_pass_dict.get('default', 0)), axis=1
        )
        acc_report["pass"] = acc_report["actual_result"] >= acc_report["expected_result"]
        return acc_report

    def load_testcases_df(self) -> pd.DataFrame:
        """Testcases after .generate() is called"""
        return pd.DataFrame([x.to_dict() for x in self.load_testcases]).drop(["pass", "actual_result"], errors="ignore", axis=1)

    def save(self, config: str = "test_config.yml", testcases: str = "test_cases.csv",
             results: str = "test_results.csv") -> None:
        """
        Save the configuration, generated testcases, and results
        of the evaluations as yml and csv files.

        Parameters:
            config (str, optional): Path to the YAML file for the configuration.
                Default is "test_config.yml".
            testcases (str, optional): Path to the CSV file for the generated testcases.
                Default is "test_cases.csv".
            results (str, optional): Path to the CSV file for the results of the evaluations.
                Default is "test_results.csv".

        Returns:
            None
        """
        with open(config, 'w') as yml:
            yml.write(yaml.safe_dump(self._config))

        self.load_testcases.to_csv(testcases, index=None)
        self.generated_results.to_csv(results, index=None)
