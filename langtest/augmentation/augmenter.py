import yaml

from typing import Iterable, Union
from langtest.transform import TestFactory
from langtest.tasks.task import TaskManager
from langtest.utils.custom_types.sample import Sample, SequenceClassificationSample


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
        hash_map = self.prepare_hash_map(data)

        # iterate over the categories
        test_types = self.__tests

        testcases = []
        for category in categories:
            if category not in hash_map:
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

    def prepare_hash_map(self, data: Iterable[Sample]) -> dict:
        """
        Prepare the data for augmentation.
        """
        from collections import defaultdict

        hash_map = defaultdict(lambda: defaultdict(list))
        for category, test_types in self.__tests.items():
            if category == "defaults":
                continue
            hash_map[category] = {}
            for test in test_types:
                hash_map[category][test] = [
                    SequenceClassificationSample(
                        original=sample.get("text", "-"), label=sample.get("label", "-")
                    )
                    for sample in data
                ]

        return hash_map

    def __or__(self, other: Iterable):
        results = self.augment(other)
        return results

    def __ror__(self, other: Iterable):
        results = self.augment(other)
        return results


class ContentDataGenerator:
    def __init__(self) -> None:
        pass

    def generate(self, content: str) -> str:
        return content
