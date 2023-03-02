from collections import defaultdict
from functools import reduce
from typing import Dict, Optional, Union

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
    SUPPORTED_HUBS = ["spacy", "transformers", "johnsnowlabs"]

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
        self.generated_results = TestRunner(self.load_testcases, self.model).evaluate()
        return self

    def report(self) -> Dict:
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
            print("=============" * 10)
            print("TEST TYPE: ", sample.test_type)
            print("ORIGINAL: ", sample.original)
            print("TEST CASE: ", sample.test_case)
            print("EXPECTED: ", sample.expected_results)
            print("ACTUAL: ", sample.realigned_spans)
            print("TRANSFORMATIONS: ", sample.transformations)
            print("IS PASS: ", sample.is_pass())
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
        return report

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
