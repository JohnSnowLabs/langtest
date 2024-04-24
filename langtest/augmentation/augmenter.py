import yaml

from typing import Iterable, Union
from langtest.transform import TestFactory
from langtest.tasks.task import TaskManager


class Augmenter:
    def __init__(
        self, config: Union[str, dict], task: Union[str, TaskManager], columns_info=None
    ) -> None:
        self.__config = config
        if isinstance(config, str):
            self.__config = self.load_config(config)

        self.__tests: dict = self.__config.get("tests", [])
        if isinstance(task, str):
            task = TaskManager(task)
        self.__task = task
        self.__testfactory = TestFactory
        self.features = columns_info.get("features", [])
        self.target = columns_info.get("target", None)
        self.__testfactory.is_augment = True

    def load_config(self, config: str) -> dict:
        """
        Load the configuration file.
        """
        with open(config, "r") as f:
            return yaml.safe_load(f)

    def augment(self, data: Iterable) -> str:
        """
        Augment the content.
        """
        # prepare the data for augmentation
        categories = list(self.__tests.keys())

        # iterate over the categories
        test_types = self.__tests

        testcases = []
        for category in categories:
            if category not in ["robustness", "bias"]:
                continue
            R_data = [
                self.__task.create_sample(
                    row_data=sample,
                    feature_column=self.features,
                    target_column=self.target,
                )
                for sample in data
            ]
            test_cases = self.__testfactory.transform(self.__task, R_data, test_types)
            testcases.extend(test_cases)
        return testcases

    def __or__(self, other: Iterable):
        results = self.augment(other)
        return results

    def __ror__(self, other: Iterable):
        results = self.augment(other)
        return results
