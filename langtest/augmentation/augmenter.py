from collections import defaultdict
import random
import yaml
import pandas as pd

from typing import Any, Dict, Iterable, List, Union
from langtest.datahandler.datasource import DataFactory
from langtest.transform import TestFactory
from langtest.tasks.task import TaskManager
from langtest.utils.custom_types.sample import Sample
from langtest.logger import logger


class DataAugmenter:
    def __init__(
        self,
        task: Union[str, TaskManager],
        config: Union[str, dict],
    ) -> None:
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

        self.__tests: Dict[str, Dict[str, dict]] = self.__config.get("tests", [])
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
        elif isinstance(self.__datafactory, DataFactory):
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
        # arrange the test cases based on the test_type in a dictionary
        test_cases = defaultdict(list)
        for sample in testcases:
            if sample.test_type in test_cases:
                test_cases[sample.test_type].append(sample)
            else:
                test_cases[sample.test_type] = [sample]

        final_data = []
        # pick the test cases based on the allocated size of the test_type
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

        # append the augmented data to the original data
        self.__augmented_data = [*data, *final_data] if isinstance(data, list) else data

        return self

    def inplace(self, data: Iterable, testcases: Iterable[Sample]) -> "DataAugmenter":
        """
        Inplace augmentation.
        """
        # indices of the data and the data itself
        data_indices = self.prepare_hash_map(data, inverted=True)
        data_dict = self.prepare_hash_map(data)

        # arrange the test cases based on the test type in a dictionary
        test_cases = defaultdict(list)
        for sample in testcases:
            if sample.test_type in test_cases:
                test_cases[sample.test_type].append(sample)
            else:
                test_cases[sample.test_type] = [sample]

        # pick the test cases based on the allocated size of the test_type
        final_data: List[Sample] = []
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

        # replace the original data with the augmented data in extact position.
        for sample in final_data:
            key = (
                sample.original_question
                if hasattr(sample, "original_question")
                else sample.original
            )
            index = data_indices[key]
            data_dict[index] = sample

        self.__augmented_data = data_dict.values()

        return self

    def new_data(self, data: Iterable, testcases: Iterable[Sample]) -> "DataAugmenter":
        """
        Create new data.
        """
        # arrange the test cases based on the test type in a dictionary
        test_cases = defaultdict(list)
        for sample in testcases:
            if sample.test_type in test_cases:
                test_cases[sample.test_type].append(sample)
            else:
                test_cases[sample.test_type] = [sample]

        final_data = []

        # pick the test cases based on the allocated size of the test_type
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

        # replace the original data with the augmented data
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

    def prepare_hash_map(
        self, data: Union[Iterable[Sample], Sample], inverted=False
    ) -> Dict[str, Any]:
        if inverted:
            hashmap = {}
            for index, sample in enumerate(data):
                key = (
                    sample.original_question
                    if hasattr(sample, "original_question")
                    else sample.original
                )
                hashmap[key] = index
        else:
            hashmap = {index: sample for index, sample in enumerate(data)}

        return hashmap

    def save(self, file_path: str, for_gen_ai=False) -> None:
        """
        Save the augmented data.
        """
        try:
            # .json file allow only for_gen_ai boolean is true and task is ner
            # then file_path should be .json
            if not (for_gen_ai) and self.__task.task_name == "ner":
                if file_path.endswith(".json"):
                    raise ValueError("File path shouldn't be .json file")

            self.__datafactory.export(data=self.__augmented_data, output_path=file_path)
        except Exception as e:
            logger.error(f"Error in saving the augmented data: {e}")

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

        # Convert 'proportion' column to float
        df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")

        # normalize the proportion and round it to 2 decimal places
        df["normalized_proportion"] = df["proportion"] / df["proportion"].sum()
        df["normalized_proportion"] = df["normalized_proportion"].apply(
            lambda x: round(x, 2)
        )

        df["avg_proportion"] = df["proportion"].mean(numeric_only=True).round(2)

        # set the index as test_name
        df.set_index("test_name", inplace=True)

        return df
