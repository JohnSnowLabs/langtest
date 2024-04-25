import yaml

from typing import Iterable, Union
from langtest.datahandler.datasource import DataFactory
from langtest.transform import TestFactory
from langtest.tasks.task import TaskManager


class Augmenter:
    def __init__(self, task: Union[str, TaskManager], config: Union[str, dict]) -> None:
        """
        Initialize the Augmenter.

        Args:
            config (Union[str, dict]): Configuration file or dictionary.
            task (Union[str, TaskManager]): Task Manager.
            columns_info ([type], optional): Columns information. Defaults to None.

        """

        self.__config = config
        if isinstance(config, str):
            self.__config = self.load_config(config)

        self.__tests: dict = self.__config.get("tests", [])
        if isinstance(task, str):
            task = TaskManager(task)
        self.__task = task

        # Test Factory and Data Factory
        self.__testfactory = TestFactory
        self.__datafactory = DataFactory

        self.__testfactory.is_augment = True

        # parameters
        self.__max_proportion = self.__tests.get("defaults", 0.6).get(
            "max_proportion", 0.6
        )

    def load_config(self, config: str) -> dict:
        """
        Load the configuration file.
        """
        with open(config, "r") as f:
            return yaml.safe_load(f)

    def augment(self, data: Union[str, Iterable]) -> str:
        """
        Augment the content.
        """
        # load the data
        if isinstance(data, dict):
            self.__datafactory = self.__datafactory(file_path=data, task=self.__task)
            data = self.__datafactory.load()
        # prepare the data for augmentation
        categories = list(self.__tests.keys())

        testcases = []
        for category in categories:
            if category not in ["robustness", "bias"]:
                continue

            test_cases = self.__testfactory.transform(self.__task, data, self.__tests)
            testcases.extend(test_cases)

        self.__augmented_data = testcases

        return self

    def prepare_hash_map(self, data: Union[str, Iterable]) -> str:
        hashmap = {index: sample for index, sample in enumerate(data)}

        return hashmap

    def save(self, file_path: str):
        """
        Save the augmented data.
        """
        self.__datafactory.export(data=self.__augmented_data, output_path=file_path)

    def __or__(self, other: Iterable):
        results = self.augment(other)
        return results

    def __ror__(self, other: Iterable):
        results = self.augment(other)
        return results
