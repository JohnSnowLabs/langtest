import yaml
import random
import pandas as pd
from typing import List
from abc import ABC, abstractmethod

from nlptest.transform import TestFactory
from nlptest.utils.custom_types import Sample
from nlptest.datahandler.datasource import DataFactory
from nlptest.transform.utils import create_terminology


class BaseAugmentaion(ABC):

    """
    Abstract base class for data augmentation techniques.

    Attributes:
        None

    Methods:
        fix: Abstract method that should be implemented by child classes.
             This method should perform the data augmentation operation.

             Returns:
                 NotImplementedError: Raised if the method is not implemented by child classes.
    """

    @abstractmethod
    def fix(self):
        """
        Abstract method that should be implemented by child classes.
        This method should perform the data augmentation operation.

        Returns:
            NotImplementedError: Raised if the method is not implemented by child classes.
        """

        return NotImplementedError


class AugmentRobustness(BaseAugmentaion):

    """
    A class for performing a specified task with historical results.

    Attributes:

        task (str): A string indicating the task being performed.
        config (dict): A dictionary containing configuration parameters for the task.
        h_report (pandas.DataFrame): A DataFrame containing a report of historical results for the task.
        max_prop (float): The maximum proportion of improvement that can be suggested by the class methods.
                        Defaults to 0.5.

    Methods:

        __init__(self, task, h_report, config, max_prop=0.5) -> None:
            Initializes an instance of MyClass with the specified parameters.

        fix(self) -> List[Sample]:
            .

        suggestions(self, prop) -> pandas.DataFrame:
            Calculates suggestions for improving test performance based on a given report.


    """

    def __init__(self, task, h_report, config, max_prop=0.5) -> None:
        """
        Initializes an instance of MyClass with the specified parameters.

        Args:
            task (str): A string indicating the task being performed.
            h_report (pandas.DataFrame): A DataFrame containing a report of historical results for the task.
            config (dict): A dictionary containing configuration parameters for the task.
            max_prop (float): The maximum proportion of improvement that can be suggested by the class methods.
                              Defaults to 0.5.

        Returns:
            None

        """

        super().__init__()
        self.task = task
        self.config = config
        self.h_report = h_report
        self.max_prop = max_prop

        if isinstance(self.config, str):
            with open(self.config) as fread:
                self.config = yaml.safe_load(fread)

    def fix(
        self,
        input_path: str,
        output_path,
        inplace: bool = False
    ):
        """
        Applies perturbations to the input data based on the recommendations from harness reports.

        Args:
            input_path (str): The path to the input data file.
            output_path (str): The path to save the augmented data file.
            inplace (bool, optional): If True, the list of samples is modified in place.
                                      Otherwise, a new samples are add to input data. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of augmented data samples.
        """

        self.df = DataFactory(input_path, self.task)
        data = self.df.load()
        TestFactory.is_augment = True
        supported_tests = TestFactory.test_scenarios()
        suggest: pd.DataFrame = self.suggestions(self.h_report)
        sum_propotion = suggest['proportion_increase'].sum()
        if suggest.shape[0] <= 0 or suggest.empty:
            print("All tests have passed. Augmentation will not be applied in this case.")
            return None

        self.config = self._parameters_overrides(self.config, data)

        fianl_aug_data = []
        hash_map = {k: v for k, v in enumerate(data)}

        for proportion in suggest.iterrows():
            cat = proportion[-1]['category'].lower()
            if cat not in ["robustness", 'bias']:
                continue
            test = proportion[-1]['test_type'].lower()
            test_type = {
                cat: {
                    test: self.config.get('tests').get(cat).get(test)
                }
            }
            if proportion[-1]['test_type'] in supported_tests[cat]:
                sample_length = len(
                    data) * self.max_prop * (proportion[-1]['proportion_increase']/sum_propotion)
                if inplace:
                    sample_indices = random.sample(
                        range(0, len(data)), int(sample_length))
                    for each in sample_indices:
                        if test == 'swap_entities':
                            test_type['robustness']['swap_entities']['parameters']['labels'] = [
                                self.label[each]]
                        hash_map[each] = TestFactory.transform(
                            self.task,
                            [hash_map[each]], test_type)[0]

                else:
                    sample_data = random.choices(data, k=int(sample_length))
                    aug_data = TestFactory.transform(
                        self.task,
                        sample_data, test_type)
                    fianl_aug_data.extend(aug_data)
        if inplace:
            fianl_aug_data = list(hash_map.values())
            self.df.export(fianl_aug_data, output_path)
        else:
            data.extend(fianl_aug_data)
            self.df.export(data, output_path)
        TestFactory.is_augment = False
        return fianl_aug_data

    def suggestions(self, report):
        """
        Calculates suggestions for improving test performance based on a given report.

        Args:
            report (pandas.DataFrame): A DataFrame containing test results by category and test type,
                                        including pass rates and minimum pass rates.

        Returns:
            pandas.DataFrame: A DataFrame containing the following columns for each suggestion:
                                - category: the test category
                                - test_type: the type of test
                                - ratio: the pass rate divided by the minimum pass rate for the test
                                - proportion_increase: a proportion indicating how much the pass rate
                                                    should increase to reach the minimum pass rate

        """
        report['ratio'] = report['pass_rate'] / report['minimum_pass_rate']
        report['proportion_increase'] = report['ratio'].apply(
            lambda x: self._proportion_values(x)
        )
        return report[~report['proportion_increase'].isna()][['category', 'test_type', 'ratio', 'proportion_increase']]

    def _proportion_values(self, x):
        """
        Calculates a proportion indicating how much a pass rate should increase to reach a minimum pass rate.

        Args:
            x (float): The ratio of the pass rate to the minimum pass rate for a given test.

        Returns:
            float: A proportion indicating how much the pass rate should increase to reach the minimum pass rate.
                If the pass rate is greater than or equal to the minimum pass rate, returns None.
                If the pass rate is between 0.9 and 1.0 times the minimum pass rate, returns 0.05.
                If the pass rate is between 0.8 and 0.9 times the minimum pass rate, returns 0.1.
                If the pass rate is between 0.7 and 0.8 times the minimum pass rate, returns 0.2.
                If the pass rate is less than 0.7 times the minimum pass rate, returns 0.3.

        """

        if x >= 1:
            return None
        elif x > 0.9:
            return 0.05
        elif x > 0.8:
            return 0.1
        elif x > 0.7:
            return 0.2
        else:
            return 0.3

    def _parameters_overrides(self, config: dict, data_handler: List[Sample]) -> dict:
        tests = config.get('tests', {}).get('robustness', {})
        if 'swap_entities' in config.get('tests', {}).get('robustness', {}):
            df = pd.DataFrame({'text': [sample.original for sample in data_handler],
                               'label': [[i.entity for i in sample.expected_results.predictions]
                                         for sample in data_handler]})
            params = tests['swap_entities']
            params['parameters'] = {}
            params['parameters']['terminology'] = create_terminology(df)
            params['parameters']['labels'] = df.label.tolist()
            self.label = self.config.get('tests').get('robustness').get(
                'swap_entities').get('parameters').get('labels')
        return config
