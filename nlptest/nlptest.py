from collections import defaultdict
from functools import reduce
from typing import Any, Dict, Optional, Union
import pandas as pd

import pandas as pd
import yaml

from nlptest.augmentation.fix_robustness import AugmentRobustness

from .datahandler.datasource import DataFactory
from .modelhandler import ModelFactory
from .testrunner import TestRunner
from .transform.perturbation import PerturbationFactory


class Harness:
    """ Harness is a testing class for NLP models.

    Harness class evaluates the performance of a given NLP model. Given test data is
    used to test the model. A report is generated with test results.
    """

    def __init__(
            self,
            task: Optional[str],
            model: Union[str, ModelFactory, Any],
            data: Optional[str] = None,
            config: Optional[Union[str, dict]] = None
    ):
        """
        Initialize the Harness object.

        Args:
            task (str, optional): Task for which the model is to be evaluated.
            model (str | ModelFactory): ModelFactory object or path to the model to be evaluated.
            data (str, optional): Path to the data to be used for evaluation.
            config (str | dict, optional): Configuration for the tests to be performed.
        """

        super().__init__()
        self.task = task

        if isinstance(model, ModelFactory):
            assert model.task == task, \
                "The 'task' passed as argument as the 'task' with which the model has been initialized are different."
            self.model = model
        elif isinstance(model, str):
            self.model = ModelFactory(task=task, model_path=model)
        else:
            self.model = model

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

        results_df = pd.DataFrame()
        for sample in self.generated_results:
            ttype = sample.test_type
            ori = sample.original
            tes = sample.test_case
            exp = sample.expected_results
            act = sample.actual_results
            tran = sample.transformations
            isp = sample.is_pass()

            da = pd.DataFrame({
                "test_type": [ttype if ttype else None],
                "original": [ori if ori else None],
                "test_case": [tes if tes else None],
                "expected_results": [str(exp) if exp else None],
                "actual_results": [str(act) if act else None],
                "transformations": [str(tran) if tran else None],
                "is_pass":[isp if isp else None]
            })
            
            results_df = pd.concat([results_df, da], axis=0)

        df_report =  results_df.groupby('test_type')['is_pass']
        df_report = df_report.agg(
            pass_count = 'sum',
            fail_count = lambda x: x.count()-x.sum(),
            pass_rate = 'mean'
            )
        # report = {}
        # for test_type, value in summary.items():
        #     pass_rate = summary[test_type]["true"] / (summary[test_type]["true"] + summary[test_type]["false"])
        #     min_pass_rate = min_pass_dict.get(test_type, min_pass_dict["default"])
        #     report[test_type] = {
        #         "fail_count": summary[test_type]["false"],
        #         "pass_count": summary[test_type]["true"],
        #         "pass_rate": pass_rate,
        #         "minimum_pass_rate": min_pass_rate,
        #         "pass": pass_rate >= min_pass_rate
        #     }

        # df_report = pd.DataFrame.from_dict(report, orient="index")
        df_report = df_report.reset_index()
        df_report['minimum_pass_rate'] = df_report['test_type'].apply(lambda x: min_pass_dict.get(x, min_pass_dict.get('default', 0.65)))

        df_report['pass_rate'] = df_report['pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))
        df_report['minimum_pass_rate'] = df_report['minimum_pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))
        
        # df_accuracy = self.accuracy_report().iloc[:2].drop("test_case", axis=1)
        # df_accuracy = df_accuracy.rename({"actual_result":"pass_rate", "expected_result":"minimum_pass_rate", "Test_type":"test_type"}, axis=1)
        # df_accuracy["pass"] = df_accuracy["pass_rate"] >= df_accuracy["minimum_pass_rate"]
        # df_accuracy['pass_rate'] = df_accuracy['pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))
        # df_accuracy['minimum_pass_rate'] = df_accuracy['minimum_pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))


        # df_report = df_report.merge(df_accuracy, how="outer")
        self.report_df = df_report.fillna("-")

        return self.report_df
    
    def detail_report(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict([x.to_dict() for x in self.generated_results])

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

    def augment(self, data_path, save_path):
        
        aug_data = AugmentRobustness.fix(
            data_path,
            self.report_df,
            save_path,
            self._config

        )
        return aug_data

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
