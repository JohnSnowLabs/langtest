import abc

import pandas as pd
from typing import List, Optional
from .transform.pertubation import PerturbationFactory
from .testrunner import TestRunner
from .datahandler.datasource import DataFactory
import yaml


class Harness:

    def __init__(self, task: Optional[str], model, data: Optional[str] = None, config_path: Optional[str] = None):
        super().__init__()
        self.task = task
        self.model = model

        if data is not None:
            # self.data = data
            if type(data) == str:
                self.data = DataFactory(data).load()
            # else:
            #     self.data = DataFactory.load_hf(data)
        if config_path is not None:
            self._config = self.configure(config_path)

    def configure(self, config):
        if type(config) == dict:
            self._config = config
        else:
            with open(config, 'r') as yml:
                self._config = yaml.safe_load(yml)

        return self._config

    def generate(self) -> pd.DataFrame:
        # self.data_handler =  DataFactory(data_path).load()
        # self.data_handler = self.data_handler(file_path = data_path)
        tests = self._config['tests_types']
        self._load_testcases = PerturbationFactory(self.data, tests).transform()
        return self._load_testcases

    # def load(self) -> pd.DataFrame:
    #     try:
    #         self._load_testcases = pd.read_csv('path/to/{self._model_path}_testcases')
    #         if self.load_testcases.empty:
    #             self.load_testcases = self.generate()
    #         # We have to make sure that loaded testcases df are editable in Qgrid
    #         return self.load_testcases
    #     except: 
    #         self.generate()

    def run(self) -> None:
        self._generated_results = TestRunner(self._load_testcases, self.model).evaluate()
        return self._generated_results

    def report(self) -> pd.DataFrame:
        return self._generated_results.groupby('Test_type')['is_pass'].value_counts()

    def save(self, config: str = "test_config.yml", testcases: str = "test_cases.csv",
             results: str = "test_results.csv"):
        with open(config, 'w') as yml:
            yml.write(yaml.safe_dump(self._config))

        self._load_testcases.to_csv(testcases, index=None)
        self._generated_results.to_csv(results, index=None)
