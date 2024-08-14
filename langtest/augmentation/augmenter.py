from collections import defaultdict
import random
import yaml
import pandas as pd

from typing import Any, Dict, Iterable, Union
from langtest.datahandler.datasource import DataFactory
from langtest.transform import TestFactory
from langtest.tasks.task import TaskManager
from langtest.utils.custom_types.sample import Sample


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
        self.__max_data_limit = self.__tests.get("parameters", {}).get("max_limit", 0.5)
        # self.__ntests = len(v for k, v in self.__tests.items()) - 1
        self.__type = self.__config.get("parameters", {}).get("type", "proportion")
        self.__style = self.__config.get("parameters", {}).get("style", "extend")

        self.__df_config = self.__initialize_config_df()

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
        if isinstance(data, dict) and not isinstance(self.__datafactory, DataFactory):
            self.__datafactory = self.__datafactory(file_path=data, task=self.__task)

            data = self.__datafactory.load()

        # generate the augmented data
        test_cases = self.__testfactory.transform(self.__task, data, self.__tests)

        # check the style of augmentation to be applied. Default is extend
        if self.__style == "extend" or self.__style == "add":
            self.extend(data, test_cases)
        elif self.__style == "inplace":
            self.inplace(data, test_cases)
        elif self.__style == "new" or self.__style == "transformed":
            self.new_data(data, test_cases)
        else:
            raise ValueError("Invalid style")

        return self

    def extend(self, data: Iterable, testcases: Iterable[Sample]) -> "DataAugmenter":
        """
        Extend the content.
        """
        # calculate the number of rows to be added
        test_cases = defaultdict(list)
        for sample in testcases:
            if sample.test_type in test_cases:
                test_cases[sample.test_type].append(sample)
            else:
                test_cases[sample.test_type] = [sample]

        final_data = []

        for _, tests in self.__tests.items():
            for test_name, _ in tests.items():
                size = self.allocated_size(test_name)

                if size == 0:
                    continue

                temp_test_cases = test_cases.get(test_name, [])
                if temp_test_cases:
                    # select random rows based on the size
                    temp_test_cases = (
                        random.choices(temp_test_cases, k=size)
                        if size < len(temp_test_cases)
                        else temp_test_cases
                    )
                    final_data.extend(temp_test_cases)

        self.__augmented_data = [*data, *final_data] if isinstance(data, list) else data

        return self

    def inplace(self, data: Iterable, testcases: Iterable) -> "DataAugmenter":
        """
        Inplace augmentation.
        """
        # calculate the number of rows to be added
        size = int(len(data) * self.allocated_size())

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

    def new_data(self, data: Iterable, testcases: Iterable) -> "DataAugmenter":
        """
        Create new data.
        """
        # calculate the number of rows to be added
        test_cases = defaultdict(list)
        for sample in testcases:
            if sample.test_type in test_cases:
                test_cases[sample.test_type].append(sample)
            else:
                test_cases[sample.test_type] = [sample]

        final_data = []
        for _, tests in self.__tests.items():
            for test_name, _ in tests.items():
                size = self.allocated_size(test_name)

                if size == 0:
                    continue

                temp_test_cases = test_cases.get(test_name, [])
                if temp_test_cases:
                    # select random rows based on the size
                    temp_test_cases = (
                        random.choices(temp_test_cases, k=size)
                        if size < len(temp_test_cases)
                        else temp_test_cases
                    )
                    final_data.extend(temp_test_cases)

        self.__augmented_data = final_data

        return self

    def allocated_size(self, test_name: str) -> int:
        """allocation size of the test to be augmented"""

        try:
            max_data_limit = (
                len(self.__datafactory)
                * self.__max_data_limit
                * self.__df_config.loc[test_name, "avg_proportion"]
            )

            return int(
                max_data_limit * self.__df_config.loc[test_name, "normalized_proportion"]
            )
        except AttributeError:
            raise ValueError(
                "Dataset is not loaded. please load the data using the `DataAugmenter.augment(data={'data_source': '..'})` method"
            )

    def prepare_hash_map(self, data: Union[Iterable[Sample], Sample]) -> Dict[str, Any]:
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

    def __initialize_config_df(self) -> pd.DataFrame:
        """
        Configure the data frame.
        """

        df = pd.DataFrame(columns=["category", "test_name", "proportion"])

        # read the configuration
        temp_data = []
        for category, tests in self.__tests.items():
            if category not in ["robustness", "bias"]:
                continue
            for test_name, test in tests.items():
                proportion = test.get("max_proportion", 0.2)
                temp_data.append(
                    {
                        "category": category,
                        "test_name": test_name,
                        "proportion": proportion,
                    }
                )
        df = pd.concat([df, pd.DataFrame(temp_data)], ignore_index=True)

        # normalize the proportion and round it to 2 decimal places
        df["normalized_proportion"] = df["proportion"] / df["proportion"].sum()
        df["normalized_proportion"] = df["normalized_proportion"].apply(
            lambda x: round(x, 2)
        )

        df["avg_proportion"] = df["proportion"].mean(numeric_only=True).round(2)

        # set the index as test_name
        df.set_index("test_name", inplace=True)

        return df
