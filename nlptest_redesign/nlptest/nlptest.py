
import pandas as pd
import numpy as np
from typing import List, Optional, Union
from .transform.pertubation import PertubationFactory
from .testrunner import TestRunner
from .datahandler.datasource import DataFactory
import yaml

class Harness:

    def __init__(self, task: Optional[str], model, data: Optional[str] = None, config : Optional[Union[str, dict]]=None) :
        super().__init__()
        self.task = task
        self.model = model
        
        if data is not None:
            # self.data = data
            if type(data) == str:
                self.data = DataFactory(data).load()
            # else:
            #     self.data = DataFactory.load_hf(data)
        if config is not None:
            self._config = self.configure(config)

    def configure(self, config):
        if type(config) == dict:
            self._config =  config
        else:
            with open(config, 'r') as yml:
                self._config = yaml.safe_load(yml)
        
        return self._config
            
       
    def generate(self) -> pd.DataFrame:
        # self.data_handler =  DataFactory(data_path).load()
        # self.data_handler = self.data_handler(file_path = data_path)
        tests = self._config['tests_types']
        if len(tests) != 0:
            self._load_testcases =  PertubationFactory(self.data, tests).transform()
        else:
            self._load_testcases = PertubationFactory(self.data).transform()
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
        self._generated_results : pd.DataFrame = TestRunner(self._load_testcases, self.model).evaluate()
        return self._generated_results

    def report(self) -> pd.DataFrame:
        # summary = pd.pivot_table()
        min_pass_dict = self._config['min_pass_rate']
        temp_df = pd.concat(
            [self._generated_results, pd.get_dummies(self._generated_results['is_pass'], prefix='bool')],
            axis=1
        )
        summary = temp_df.pivot_table(
            values=['bool_True', 'bool_False'],
            index=['Test_type'],
            aggfunc=np.sum
        ).reset_index()
        
        summary['minimum_pass_rate'] = \
            summary.apply(
                lambda x: min_pass_dict[x['Test_type']] \
                if x['Test_type'] in list(min_pass_dict.keys())\
                else min_pass_dict['default'], 
                axis=1)
        summary['pass_rate'] = summary['bool_True']/(summary['bool_True'] + summary['bool_False'])
        summary['pass'] = summary['minmium_pass_rate'] < summary['pass_rate']
        summary.columns = ['Test_type', 'fail_count', 'pass_count',	'minimum_pass_rate', 'pass_rate', 'pass']

        return summary

    def save(self, config: str = "test_config.yml", testcases: str = "test_cases.csv", results: str = "test_results.csv"): 
        with open(config, 'w') as yml:
            yml.write(yaml.safe_dump(self._config))
        
        self._load_testcases.to_csv(testcases, index=None)
        self._generated_results.to_csv(results, index=None)


        

    

    
