from functools import reduce
from typing import Optional, Union

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
    SUPPORTED_HUBS = ["spacy", "transformers", "sparknlp", "johnsnowlabs"]
    def __init__(
            self,
            task: Optional[str],
            model: Union[str, ModelFactory],
            hub: Optional[str] = None,
            data: Optional[str] = None,
            config: Optional[Union[str, dict]] = None
    ):
        """
        Initialize the Harness object.

        Args:
            task (str, optional): Task for which the model is to be evaluated.
            model (str | ModelFactory): ModelFactory object or path to the model to be evaluated.
            backend (str, optional): model backend to load from the path. Required if path is passed as 'model'.
            data (str, optional): Path to the data to be used for evaluation.
            config (str | dict, optional): Configuration for the tests to be performed.

        Raises:
            ValueError: Invalid arguments.
        """

        super().__init__()
        self.task = task

        if isinstance(model, ModelFactory):
            assert model.task == task, \
                "The 'task' passed as argument as the 'task' with which the model has been initialized are different."
            self.model = model
        elif isinstance(model, str):
            if hub is None:
                raise ValueError("Hub should be specified when loading a model from web.")
            if hub not in self.SUPPORTED_HUBS:
                raise ValueError(f"Hub {hub} is not supported or invalid.")
                
            self.model = ModelFactory.load_model(task=task, hub=hub, path=model)

        if data is not None:
            # self.data = data
            if type(data) == str:
                self.data = DataFactory(data).load()
            # else:
            #     self.data = DataFactory.load_hf(data)
        if config is not None:
            self._config = self.configure(config)

    def configure(self, config):
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

    def generate(self) -> None:
        """
        Generates the testcases to be used when evaluating the model.

        Returns:
            None: The generated testcases are stored in `_load_testcases` attribute.
        """

        # self.data_handler =  DataFactory(data_path).load()
        # self.data_handler = self.data_handler(file_path = data_path)
        tests = self._config['tests_types']
        self._load_testcases = PerturbationFactory(self.data, tests).transform()
        return self
    # def load(self) -> pd.DataFrame:
    #     try:
    #         self._load_testcases = pd.read_csv('path/to/{self._model_path}_testcases')
    #         if self.load_testcases.empty:
    #             self.load_testcases = self.generate()
    #         # We have to make sure that loaded testcases df are editable in Qgrid
    #         return self.load_testcases
    #     except:
    #         self.generate()

    def run(self) -> "Harness":
        """
        Run the tests on the model using the generated testcases.

        Returns:
            None: The evaluations are stored in `_generated_results` attribute.
        """
        self._generated_results = TestRunner(self._load_testcases, self.model).evaluate()
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

        summary = self._generated_results.groupby('Test_type')['is_pass']
        summary = summary.agg(
            pass_count = 'sum',
            fail_count = lambda x: x.count()-x.sum(),
            pass_rate = 'mean'
            )
        summary = summary.reset_index()

        summary['minimum_pass_rate'] = summary['Test_type'].apply(
            lambda x: min_pass_dict.get(x, min_pass_dict.get('default', 0))
        )

        summary['pass'] = summary['minimum_pass_rate'] < summary['pass_rate']
        summary['pass_rate'] = summary['pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))
        summary['minimum_pass_rate'] = summary['minimum_pass_rate'].apply(lambda x: "{:.0f}%".format(x*100))

        summary.columns=["test type", "pass count", "fail count", "pass rate", "minimum pass rate", "pass"]
        return summary

    def save(
            self, config: str = "test_config.yml",
            testcases: str = "test_cases.csv",
            results: str = "test_results.csv"
    ):
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

        self._load_testcases.to_csv(testcases, index=None)
        self._generated_results.to_csv(results, index=None)
