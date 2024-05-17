import random
import yaml

from typing import Any, Dict, Iterable, Union
from langtest.datahandler.datasource import DataFactory
from langtest.transform import TestFactory
from langtest.tasks.task import TaskManager


class DataAugmenter:
    def __init__(self, task: Union[str, TaskManager], config: Union[str, dict]) -> None:
        """
        Initialize the DataAugmenter.

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
            if task in ["ner", "text-classification", "question-answering"]:
                task = TaskManager(task)
            else:
                raise ValueError(
                    "check the task name. It should be one of the following: ner, text-classification, question-answering"
                )
        self.__task = task

        # Test Factory and Data Factory
        self.__testfactory = TestFactory
        self.__datafactory = DataFactory

        self.__testfactory.is_augment = True

        # parameters
        self.__max_proportion = self.__tests.get("defaults", {}).get(
            "max_proportion", 0.6
        )
        # self.__ntests = len(v for k, v in self.__tests.items()) - 1
        self.__type = self.__config.get("parameters", {}).get("type", "proportion")
        self.__style = self.__config.get("parameters", {}).get("style", "extend")

        self.__df_config = self.__config_df()

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

        # check the style of augmentation to be applied. Default is extend
        if self.__style == "extend":
            self.extend(data)
        elif self.__style == "inplace":
            self.inplace(data)
        elif self.__style == "new":
            self.new_data(data)
        else:
            raise ValueError("Invalid style")

        return self

    def extend(self, data: Iterable) -> "DataAugmenter":
        """
        Extend the content.
        """
        # calculate the number of rows to be added
        n = len(data)

        data_cut = random.sample(data, int(n * self.__max_proportion))

        test_cases: list = self.__testfactory.transform(
            self.__task, data_cut, self.__tests
        )

        self.__augmented_data = [*data, *test_cases] if isinstance(data, list) else data

        return self

    def inplace(self, data: Iterable) -> "DataAugmenter":
        """
        Inplace augmentation.
        """
        # calculate the number of rows to be added
        size = int(len(data) * self.__max_proportion)

        # create a dictionary with index as key and data as value
        data_dict = self.prepare_hash_map(data)

        # select random rows based on the size with its index
        selected = random.sample(data_dict.keys(), int(size))

        for idx in selected:
            test_cases = self.__testfactory.transform(
                self.__task, [data_dict[idx]], self.__tests
            )
            data_dict[idx] = test_cases[0] if test_cases else data_dict[idx]

        self.__augmented_data = data_dict.values()

        return self

    def new_data(self, data: Iterable) -> "DataAugmenter":
        """
        Create new data.
        """
        # calculate the number of rows to be added
        size = int(len(data) * self.__max_proportion)

        data_cut = random.sample(data, size)

        test_cases = self.__testfactory.transform(self.__task, data_cut, self.__tests)

        self.__augmented_data = test_cases

        return self

    def size(self, category: str, test_name: str) -> int:
        return (
            self.__max_proportion
            * self.__tests.get(category, {}).get(test_name, {}).get("max_proportion", 0.6)
        ) / self.__df_config.shape[0]

    def prepare_hash_map(self, data: Union[str, Iterable]) -> Dict[str, Any]:
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

    def __config_df(self):
        """
        Configure the data frame.
        """

        import pandas as pd

        df = pd.DataFrame(columns=["category", "test_name", "proportion"])

        # read the configuration
        for category, tests in self.__tests.items():
            if category not in ["robustness", "bias"]:
                continue
            for test_name, test in tests.items():
                proportion = test.get("max_proportion", 0.6)
                temp = pd.DataFrame(
                    {
                        "category": [category],
                        "test_name": [test_name],
                        "proportion": [proportion],
                    },
                )
                df = pd.concat([df, temp], ignore_index=True)

        return df
