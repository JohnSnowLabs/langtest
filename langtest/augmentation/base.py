import os
import random
import re
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy as copy
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

from langtest.augmentation.utils import AzureOpenAIConfig, OpenAIConfig
from langtest.datahandler.datasource import DataFactory
from langtest.transform import TestFactory
from langtest.transform.utils import create_terminology
from langtest.utils.custom_types import Sample
from langtest.utils.custom_types.output import NEROutput
from langtest.utils.custom_types.predictions import NERPrediction, SequenceLabel
from langtest.utils.custom_types.sample import NERSample
from langtest.tasks import TaskManager
from ..errors import Errors


class BaseAugmentaion(ABC):
    """Abstract base class for data augmentation techniques.

    Methods:
        fix: Abstract method that should be implemented by child classes.
             This method should perform the data augmentation operation.
    """

    @abstractmethod
    def fix(self, *args, **kwargs):
        """Abstract method that should be implemented by child classes.

        This method should perform the data augmentation operation.

        Returns:
            NotImplementedError: Raised if the method is not implemented by child classes.
        """
        return NotImplementedError


class AugmentRobustness(BaseAugmentaion):
    """A class for performing a specified task with historical results.

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

    def __init__(
        self,
        task: TaskManager,
        h_report: "pd.DataFrame",
        config: Dict,
        custom_proportions: Union[Dict, List] = None,
        max_prop=0.5,
    ) -> None:
        """Initializes an instance of MyClass with the specified parameters.

        Args:
            task (str): A string indicating the task being performed.
            h_report (pandas.DataFrame): A DataFrame containing a report of historical results for the task.
            config (dict): A dictionary containing configuration parameters for the task.
            custom_proportions
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
        self.custom_proportions = custom_proportions

        if isinstance(self.config, str):
            with open(self.config) as fread:
                self.config = yaml.safe_load(fread)

    def fix(
        self,
        training_data: dict,
        output_path: str,
        export_mode: str = "add",
    ):
        """Applies perturbations to the input data based on the recommendations from harness reports.

        Args:
            training_data (dict): A dictionary containing the input data for augmentation.
            output_path (str): The path to save the augmented data file.
            export_mode (str, optional): Determines how the samples are modified or exported.
                                        - 'inplace': Modifies the list of samples in place.
                                        - 'add': Adds new samples to the input data.
                                        - 'transformed': Exports only the transformed data, excluding untransformed samples.
                                        Defaults to 'add'.

        Returns:
            List[Dict[str, Any]]: A list of augmented data samples.
        """

        # if "source" in training_data and training_data["source"] == "huggingface":
        #     self.df = HuggingFaceDataset(training_data, self.task)
        #     data = self.df.load_data(
        #         feature_column=training_data.get("feature_column", "text"),
        #         target_column=training_data.get("target_column", "label"),
        #         split=training_data.get("split", "test"),
        #         subset=training_data.get("subset", None),
        #     )
        # else:
        self.df = DataFactory(training_data, self.task)
        data = self.df.load()
        TestFactory.is_augment = True
        supported_tests = TestFactory.test_scenarios()
        suggest: pd.DataFrame = self.suggestions(self.h_report)
        sum_propotion = suggest["proportion_increase"].sum()
        if suggest.shape[0] <= 0 or suggest.empty:
            print("All tests have passed. Augmentation will not be applied in this case.")
            return None

        self.config = self._parameters_overrides(self.config, data)

        final_aug_data = []
        hash_map = {k: v for k, v in enumerate(data)}
        transformed_data = []
        for proportion in suggest.iterrows():
            cat = proportion[-1]["category"].lower()
            if cat not in ["robustness", "bias"]:
                continue
            test = proportion[-1]["test_type"].lower()
            test_type = {cat: {test: self.config.get("tests").get(cat).get(test)}}
            if proportion[-1]["test_type"] in supported_tests[cat]:
                sample_length = (
                    len(data)
                    * self.max_prop
                    * (proportion[-1]["proportion_increase"] / sum_propotion)
                )
                if export_mode in ("inplace"):
                    sample_indices = random.sample(
                        range(0, len(data)), int(sample_length)
                    )
                    for each in sample_indices:
                        if test == "swap_entities":
                            test_type["robustness"]["swap_entities"]["parameters"][
                                "labels"
                            ] = [self.label[each]]
                        res = TestFactory.transform(
                            self.task, [hash_map[each]], test_type
                        )
                        if len(res) == 0:
                            continue
                        hash_map[each] = res[0]
                else:
                    if test == "swap_entities":
                        sample_data = data[: int(sample_length)]
                        test_type["robustness"]["swap_entities"]["parameters"][
                            "labels"
                        ] = test_type["robustness"]["swap_entities"]["parameters"][
                            "labels"
                        ][
                            : int(sample_length)
                        ]
                    else:
                        sample_data = random.choices(data, k=int(sample_length))
                    aug_data = TestFactory.transform(self.task, sample_data, test_type)
                    final_aug_data.extend(aug_data)

                    if export_mode == "transformed":
                        transformed_data.extend(aug_data)
        if "." not in training_data["data_source"]:
            if export_mode == "inplace":
                final_aug_data = list(hash_map.values())
                self.df.export(final_aug_data, output_path)
            elif export_mode == "transformed":
                self.df.export(transformed_data, output_path)
            else:
                data.extend(final_aug_data)
                self.df.export(data, output_path)

            TestFactory.is_augment = False
            return final_aug_data

        else:
            if export_mode == "inplace":
                final_aug_data = list(hash_map.values())
                self.df.export(final_aug_data, output_path)
            elif export_mode == "transformed":
                self.df.export(transformed_data, output_path)
            else:
                data.extend(final_aug_data)
                self.df.export(data, output_path)

            TestFactory.is_augment = False
            return final_aug_data

    def suggestions(self, report: "pd.DataFrame") -> "pd.DataFrame":
        """Calculates suggestions for improving test performance based on a given report.

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
        report["ratio"] = report["pass_rate"] / report["minimum_pass_rate"]

        if self.custom_proportions and isinstance(self.custom_proportions, dict):
            report["proportion_increase"] = report["test_type"].map(
                self.custom_proportions
            )
        elif self.custom_proportions and isinstance(self.custom_proportions, list):
            report["proportion_increase"] = report["ratio"].apply(self._proportion_values)
            report = report[report["test_type"].isin(self.custom_proportions)]
        else:
            report["proportion_increase"] = report["ratio"].apply(self._proportion_values)

        report = report.dropna(subset=["proportion_increase"])[
            ["category", "test_type", "ratio", "proportion_increase"]
        ]
        return report

    @staticmethod
    def _proportion_values(x: float) -> Optional[float]:
        """Calculates a proportion indicating how much a pass rate should increase to reach a minimum pass rate.

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
        tests = config.get("tests", {}).get("robustness", {})
        if "swap_entities" in config.get("tests", {}).get("robustness", {}):
            df = pd.DataFrame(
                {
                    "text": [sample.original for sample in data_handler],
                    "label": [
                        [i.entity for i in sample.expected_results.predictions]
                        for sample in data_handler
                    ],
                }
            )
            params = tests["swap_entities"]
            params["parameters"] = {}
            params["parameters"]["terminology"] = create_terminology(df)
            params["parameters"]["labels"] = df.label.tolist()
            self.label = (
                self.config.get("tests")
                .get("robustness")
                .get("swap_entities")
                .get("parameters")
                .get("labels")
            )
        return config


class TemplaticAugment(BaseAugmentaion):
    """This class is used for templatic augmentation. It is a subclass of the BaseAugmentation class.

    Attributes:
        __templates:
            A string or a list of strings or samples that represents the templates for the augmentation.
        __task:
            The task for which the augmentation is being performed.
        __generate_templates:
            if set to True, generates sample templates from the given ones.
        __show_templates:
            if set to True, displays the used templates.


    Methods:
        __init__(self, templates: Union[str, List[str]], task: str):
            Initializes the TemplaticAugment class.
        fix(self, training_data: str, output_path: str, *args, **kwargs):
            Performs the templatic augmentation and exports the results to a specified path.
    """

    def __init__(
        self,
        templates: Union[str, List[str]],
        task: TaskManager,
        generate_templates=False,
        show_templates=False,
        num_extra_templates=10,
        model_config: Union[OpenAIConfig, AzureOpenAIConfig] = None,
    ) -> None:
        """This constructor for the TemplaticAugment class.

        Args:
            templates (Union[str, List[str]]): The templates to be used for the augmentation.
            task (str): The task for which the augmentation is being performed.
            generate_templates (bool, optional): if set to True, generates sample templates from the given ones.
            show_templates (bool, optional): if set to True, displays the used templates.
        """
        self.__templates: Union[str, List[str], List[Sample]] = templates
        self.__task = task

        if generate_templates:
            try:
                given_template = self.__templates[:]
                for template in given_template:
                    generated_templates: List[str] = self.__generate_templates(
                        template, num_extra_templates, model_config
                    )

                    while len(generated_templates) < num_extra_templates:
                        temp_templates = self.__generate_templates(
                            template,
                            num_extra_templates,
                            model_config,
                        )
                        generated_templates.extend(temp_templates)

                    if generated_templates:
                        # Extend the existing templates list

                        self.__templates.extend(generated_templates[:num_extra_templates])
            except ModuleNotFoundError:
                raise ImportError(Errors.E097())

            except Exception as e_msg:
                error_message = str(e_msg)
                raise Exception(Errors.E095(e=error_message))

        if show_templates:
            [print(template) for template in self.__templates]

        if isinstance(self.__templates, str) and os.path.exists(self.__templates):
            self.__templates = DataFactory(self.__templates, self.__task).load()
        elif isinstance(self.__templates, str):
            self.__templates = [self.str_to_sample(self.__templates)]
        elif isinstance(self.__templates, list) and isinstance(self.__templates[0], str):
            self.__templates = [self.str_to_sample(i) for i in self.__templates]

    def fix(
        self,
        training_data: Dict[str, Any],
        output_path: str,
        max_num: int = None,
        append_original: bool = False,
        *args,
        **kwargs,
    ) -> bool:
        """This method is used to perform the templatic augmentation.

        It takes the input data, performs the augmentation and then saves the augmented data to the output path.

        Args:
            training_data (dict): A dictionary containing the input data for augmentation.
            output_path (str): The path where the augmented data will be saved.
            max_num (int): Maximum number of new samples to generate
            append_original (bool, optional): If set to True, appends the original data to the augmented data. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            bool: Returns True upon successful completion of the method.
        """
        df = DataFactory(training_data, self.__task)
        data = df.load()
        new_data = (
            data.copy()
            if isinstance(data, (pd.DataFrame, pd.Series))
            else copy.deepcopy(data) if append_original else []
        )
        self.__search_results = self.search_sample_results(data)

        if not max_num:
            max_num = max(len(i) for i in self.__search_results.values())

        for template in self.__templates:
            for _ in range(max_num):
                new_sample = self.new_sample(template)
                if new_sample:
                    new_data.append(new_sample)

        df.export(new_data, output_path)
        return True

    @staticmethod
    def search_sample_results(
        samples: List[Sample],
    ) -> Dict[str, List[Union[NERPrediction, SequenceLabel]]]:
        """This method is used to search the results of the samples for the entities in the templates.

        Args:
            samples (List[Sample]): The samples for which the results are to be searched.

        Returns:
            Dict[str, List[Union[NERPrediction, SequenceLabel]]]: A dictionary containing the search results.
        """
        results_dict = defaultdict(list)
        for sample in samples:
            chunk = []
            ent_name = ""
            for result in sample.expected_results.predictions:
                ent = result.entity.split("-")[-1]
                if ent != "O" and ent_name == "":
                    ent_name = ent
                if result.entity.endswith(ent_name) and ent != "O":
                    result.doc_id = 0
                    result.doc_name = ""
                    chunk.append(result)
                elif len(chunk) > 0:
                    results_dict[ent_name].append(tuple(chunk))
                    ent_name = ""
                    chunk = []

            if chunk:
                results_dict[ent_name].append(tuple(chunk))
        return results_dict

    @staticmethod
    def extract_variable_names(f_string: str) -> List[str]:
        """This method is used to extract the variable names from the templates.

        Args:
            f_string (str): The template string.

        Returns:
            List[str]: A list of variable names.
        """
        pattern = r"{([^{}]*)}"
        matches = re.findall(pattern, f_string)
        variable_names = [match.strip() for match in matches]
        return variable_names

    def new_sample(self, template: Sample):
        """This method is used to generate a new sample from a template.

        Args:
            template (Sample): The template from which the new sample is to be generated.

        Returns:
            Sample: The new sample generated from the template.
        """
        template = copy(template)
        matches = re.finditer(r"{([^{}]*)}", template.original)
        cursor = 0
        other_predictions = []
        if matches:
            for match in matches:
                entity = match.group(1)
                if entity in self.__search_results:
                    prediction = random.choice(self.__search_results[entity])
                    word = " ".join(
                        i.span.word for i in prediction if isinstance(i, NERPrediction)
                    )

                    template.original = template.original.replace(
                        "{" + entity + "}", word, 1
                    )
                    for result in template.expected_results.predictions[cursor:]:
                        if prediction[0].entity.endswith(result.entity):
                            for each_prediction in prediction:
                                if isinstance(each_prediction, NERPrediction):
                                    each_prediction.chunk_tag = "-X-"
                                    each_prediction.pos_tag = "-X-"
                            other_predictions.extend(prediction)
                            cursor += 1
                            break
                        else:
                            if "{" in result.span.word and "}" in result.span.word:
                                continue
                            other_predictions.append(result)
                            cursor += 1
                else:
                    continue
            template.expected_results.predictions = (
                other_predictions + template.expected_results.predictions[cursor:]
            )
            return template
        else:
            return None

    def str_to_sample(self, template: str):
        """This method is used to convert a template string to a Sample object.

        Args:
            template (str): The template string to be converted.

        Returns:
            Sample: The Sample object generated from the template string.
        """
        if self.__task == "ner":
            template = self.add_spaces_around_punctuation(template)
            sample = NERSample()
            sample.original = template
            words = template.split()
            predictions = []
            cursor = 0
            for word in words:
                if "{" in word and "}" in word:
                    entity = word.replace("{", "").replace("}", "")
                else:
                    entity = "O"
                predictions.append(
                    NERPrediction.from_span(
                        entity,
                        word,
                        cursor,
                        cursor + len(word),
                        pos_tag="-X-",
                        chunk_tag="-X-",
                        doc_id=0,
                        doc_name="",
                    )
                )
                cursor += len(word) + 1
            sample.expected_results = NEROutput(predictions=predictions)

        elif self.__task == "text-classification":
            raise NotImplementedError

        return sample

    @property
    def templates(self):
        """Templates getter"""
        return self.__templates

    @templates.setter
    def templates(self, templates: Union[str, List[str]]):
        self.__init__(templates, self.__task)

    @property
    def task(self):
        """Task getter"""
        return self.__task

    @task.setter
    def task(self, task: str):
        self.__task = task

    @staticmethod
    def add_spaces_around_punctuation(text: str):
        """This method is used to add spaces around punctuation in a string.

        Args:
            text (str): The string to which spaces are to be added.

        Returns:
            str: The string with spaces added around punctuation.
        """
        for punct in string.punctuation:
            if punct not in ["{", "}", "_"]:
                if punct == ".":
                    # To prevent spaces being added around decimal points
                    text = re.sub(r"(\d)\.(\d)", r"\1[DOT]\2", text)

                text = text.replace(punct, f" {punct} ")

                if punct == ".":
                    # Putting back the decimal points to original state
                    text = text.replace("[DOT]", ".")

        # Removing extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def __generate_templates(
        self,
        template: str,
        num_extra_templates: int,
        model_config: Union[OpenAIConfig, AzureOpenAIConfig] = None,
    ) -> List[str]:
        """This method is used to generate extra templates from a given template."""
        from langtest.augmentation.utils import (
            generate_templates_azoi,  # azoi means Azure OpenAI
            generate_templates_openai,
            generate_templates_ollama,
        )

        params = model_config.copy() if model_config else {}

        if model_config and model_config.get("provider") == "openai":
            return generate_templates_openai(template, num_extra_templates, params)

        elif model_config and model_config.get("provider") == "azure":
            return generate_templates_azoi(template, num_extra_templates, params)

        elif model_config and model_config.get("provider") == "ollama":
            return generate_templates_ollama(template, num_extra_templates, params)

        else:
            return generate_templates_openai(template, num_extra_templates)
