import csv
import importlib
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union
from typing import Optional

import jsonlines
import pandas as pd

from langtest.utils.custom_types import sample
from .format import Formatter
from langtest.utils.custom_types import (
    NEROutput,
    NERPrediction,
    NERSample,
    QASample,
    Sample,
    SequenceClassificationOutput,
    SequenceClassificationSample,
    SequenceLabel,
    SummarizationSample,
    ToxicitySample,
    TranslationSample,
    ClinicalSample,
    SecuritySample,
    DisinformationSample,
)
from ..utils.lib_manager import try_import_lib

COLUMN_MAPPER = {
    "text-classification": {
        "text": ["text", "sentences", "sentence", "sample"],
        "label": ["label", "labels ", "class", "classes"],
    },
    "ner": {
        "text": ["text", "sentences", "sentence", "sample", "tokens"],
        "ner": [
            "label",
            "labels ",
            "class",
            "classes",
            "ner_tag",
            "ner_tags",
            "ner",
            "entity",
        ],
        "pos": ["pos_tags", "pos_tag", "pos", "part_of_speech"],
        "chunk": ["chunk_tags", "chunk_tag"],
    },
    "question-answering": {
        "text": ["question"],
        "context": ["context", "passage"],
        "answer": ["answer", "answer_and_def_correct_predictions"],
    },
    "summarization": {"text": ["text", "document"], "summary": ["summary"]},
    "toxicity": {"text": ["text"]},
    "translation": {"text": ["text", "original", "sourcestring"]},
    "security": {"text": ["text", "prompt"]},
    "clinical-tests": {
        "Patient info A": ["Patient info A"],
        "Patient info B": ["Patient info B"],
        "Diagnosis": ["Diagnosis"],
    },
    "disinformation-test": {
        "hypothesis": ["hypothesis", "thesis"],
        "statements": ["statements", "headlines"],
    },
}


class _IDataset(ABC):
    """Abstract base class for Dataset.

    Defines the load_data method that all subclasses must implement.
    """

    @abstractmethod
    def load_raw_data(self):
        """Load data from the file_path into raw format."""
        raise NotImplementedError()

    @abstractmethod
    def load_data(self):
        """Load data from the file_path into the right Sample object."""
        return NotImplementedError()

    @abstractmethod
    def export_data(self, data: List[Sample], output_path: str):
        """Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        return NotImplementedError()


class DataFactory:
    """Data factory for creating Dataset objects.

    The DataFactory class is responsible for creating instances of the
    correct Dataset type based on the file extension.
    """

    def __init__(self, file_path: dict, task: str, **kwargs) -> None:
        """Initializes DataFactory object.

        Args:
            file_path (dict): Dictionary containing 'data_source' key with the path to the dataset.
            task (str): Task to be evaluated.
        """
        if not isinstance(file_path, dict):
            raise ValueError("'file_path' must be a dictionary.")

        if "data_source" not in file_path:
            raise ValueError(
                "The 'data_source' key must be provided in the 'file_path' dictionary."
            )
        self._custom_label = file_path
        self._file_path = file_path.get("data_source")
        self._class_map = {
            cls.__name__.replace("Dataset", "").lower(): cls
            for cls in _IDataset.__subclasses__()
        }
        _, self.file_ext = os.path.splitext(self._file_path)
        if len(self.file_ext) > 0:
            self.file_ext = self.file_ext.replace(".", "")
        else:
            self._file_path = self._load_dataset(self._file_path)
            _, self.file_ext = os.path.splitext(self._file_path)
        self.task = task
        self.init_cls = None
        self.kwargs = kwargs

    def load_raw(self):
        """Loads the data into a raw format"""
        self.init_cls = self._class_map[self.file_ext.replace(".", "")](
            self._file_path, task=self.task, **self.kwargs
        )
        return self.init_cls.load_raw_data()

    def load(self) -> List[Sample]:
        """Loads the data for the correct Dataset type.

        Returns:
            list[Sample]: Loaded text data.
        """
        if len(self._custom_label) > 1 and self.file_ext == "csv":
            self.init_cls = self._class_map[self.file_ext.replace(".", "")](
                self._custom_label, task=self.task, **self.kwargs
            )
        else:
            self.init_cls = self._class_map[self.file_ext.replace(".", "")](
                self._file_path, task=self.task, **self.kwargs
            )
        return self.init_cls.load_data()

    def export(self, data: List[Sample], output_path: str) -> None:
        """Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        self.init_cls.export_data(data, output_path)

    @classmethod
    def load_curated_bias(
        cls, tests_to_filter: List[str], file_path: str
    ) -> List[Sample]:
        """Loads curated bias into a list of samples

        Args:
            tests_to_filter (List[str]): name of the tests to use
            file_path(str): path to the file to load

        Returns:
            List[Sample]: list of processed samples
        """
        data = []
        path = os.path.abspath(__file__)
        if file_path == "BoolQ":
            bias_jsonl = os.path.dirname(path)[:-7] + "/BoolQ/bias.jsonl"
            with jsonlines.open(bias_jsonl) as reader:
                for item in reader:
                    if item["test_type"] in tests_to_filter:
                        data.append(
                            QASample(
                                original_question=item["original_question"],
                                original_context=item.get("original_context", "-"),
                                perturbed_question=item["perturbed_question"],
                                perturbed_context=item.get("perturbed_context", "-"),
                                test_type=item["test_type"],
                                category=item["category"],
                                dataset_name="BoolQ",
                            )
                        )
        elif file_path == "XSum":
            bias_jsonl = os.path.dirname(path)[:-7] + "/Xsum/bias.jsonl"
            with jsonlines.open(bias_jsonl) as reader:
                for item in reader:
                    if item["test_type"] in tests_to_filter:
                        data.append(
                            SummarizationSample(
                                original=item["original"],
                                test_case=item["test_case"],
                                test_type=item["test_type"],
                                category=item["category"],
                                dataset_name="XSum",
                            )
                        )
        else:
            logging.warning(
                f"File {file_path} not supported for `load_curated_bias`, the biases will be empty."
            )
        return data

    @classmethod
    def _load_dataset(cls, dataset_name: str) -> str:
        """Loads a dataset

        Args:
            dataset_name (str): name of the dataset

        Returns:
            str: path to our data
        """
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        datasets_info = {
            "BoolQ-dev-tiny": script_dir[:-7] + "/BoolQ/dev-tiny.jsonl",
            "BoolQ-dev": script_dir[:-7] + "/BoolQ/dev.jsonl",
            "BoolQ-test-tiny": script_dir[:-7] + "/BoolQ/test-tiny.jsonl",
            "BoolQ-test": script_dir[:-7] + "/BoolQ/test.jsonl",
            "BoolQ": script_dir[:-7] + "/BoolQ/combined.jsonl",
            "NQ-open-test": script_dir[:-7] + "/NQ-open/test.jsonl",
            "NQ-open": script_dir[:-7] + "/NQ-open/combined.jsonl",
            "NQ-open-test-tiny": script_dir[:-7] + "/NQ-open/test-tiny.jsonl",
            "XSum-test-tiny": script_dir[:-7] + "/Xsum/XSum-test-tiny.jsonl",
            "XSum-test": script_dir[:-7] + "/Xsum/XSum-test.jsonl",
            "TruthfulQA-combined": script_dir[:-7]
            + "/TruthfulQA/TruthfulQA-combined.jsonl",
            "TruthfulQA-test": script_dir[:-7] + "/TruthfulQA/TruthfulQA-test.jsonl",
            "TruthfulQA-test-tiny": script_dir[:-7]
            + "/TruthfulQA/TruthfulQA-test-tiny.jsonl",
            "MMLU-test-tiny": script_dir[:-7] + "/MMLU/MMLU-test-tiny.jsonl",
            "MMLU-test": script_dir[:-7] + "/MMLU/MMLU-test.jsonl",
            "OpenBookQA-test": script_dir[:-7] + "/OpenBookQA/OpenBookQA-test.jsonl",
            "OpenBookQA-test-tiny": script_dir[:-7]
            + "/OpenBookQA/OpenBookQA-test-tiny.jsonl",
            "Quac-test": script_dir[:-7] + "/quac/Quac-test.jsonl",
            "Quac-test-tiny": script_dir[:-7] + "/quac/Quac-test-tiny.jsonl",
            "toxicity-test-tiny": script_dir[:-7] + "/toxicity/toxicity-test-tiny.jsonl",
            "NarrativeQA-test": script_dir[:-7] + "/NarrativeQA/NarrativeQA-test.jsonl",
            "NarrativeQA-test-tiny": script_dir[:-7]
            + "/NarrativeQA/NarrativeQA-test-tiny.jsonl",
            "HellaSwag-test": script_dir[:-7] + "/HellaSwag/hellaswag-test.jsonl",
            "HellaSwag-test-tiny": script_dir[:-7]
            + "/HellaSwag/hellaswag-test-tiny.jsonl",
            "Translation-test": script_dir[:-7]
            + "/Translation/translation-test-tiny.jsonl",
            "BBQ-test": script_dir[:-7] + "/BBQ/BBQ-test.jsonl",
            "BBQ-test-tiny": script_dir[:-7] + "/BBQ/BBQ-test-tiny.jsonl",
            "Prompt-Injection-Attack": script_dir[:-7]
            + "/security/Prompt-Injection-Attack.jsonl",
            "Medical-files": script_dir[:-7] + "/Clinical-Tests/Medical-files.jsonl",
            "Gastroenterology-files": script_dir[:-7]
            + "/Clinical-Tests/Gastroenterology-files.jsonl",
            "Oromaxillofacial-files": script_dir[:-7]
            + "/Clinical-Tests/Oromaxillofacial-files.jsonl",
            "ASDiv-test": script_dir[:-7] + "/asdiv/asdiv-test.jsonl",
            "ASDiv-test-tiny": script_dir[:-7] + "/asdiv/asdiv-test-tiny.jsonl",
            "Bigbench-Causal-judgment-test": script_dir[:-7]
            + "/Bigbench/CausalJudgment/causal-judgment-test.jsonl",
            "Bigbench-Causal-judgment-test-tiny": script_dir[:-7]
            + "/Bigbench/CausalJudgment/causal-judgment-test-tiny.jsonl",
            "Bigbench-DisflQA-test": script_dir[:-7]
            + "/Bigbench/DisflQA/disfl-qa-test.jsonl",
            "Bigbench-DisflQA-test-tiny": script_dir[:-7]
            + "/Bigbench/DisflQA/disfl-qa-test-tiny.jsonl",
            "Bigbench-Abstract-narrative-understanding-test-tiny": script_dir[:-7]
            + "/Bigbench/AbstractNarrativeUnderstanding/Abstract-narrative-understanding-test-tiny.jsonl",
            "Bigbench-Abstract-narrative-understanding-test": script_dir[:-7]
            + "/Bigbench/AbstractNarrativeUnderstanding/Abstract-narrative-understanding-test.jsonl",
            "Bigbench-DisambiguationQA-test": script_dir[:-7]
            + "/Bigbench/DisambiguationQA/DisambiguationQA-test.jsonl",
            "Bigbench-DisambiguationQA-test-tiny": script_dir[:-7]
            + "/Bigbench/DisambiguationQA/DisambiguationQA-test-tiny.jsonl",
            "LogiQA-test-tiny": script_dir[:-7] + "/LogiQA/LogiQA-test-tiny.jsonl",
            "LogiQA-test": script_dir[:-7] + "/LogiQA/LogiQA-test.jsonl",
            "Narrative-Wedging": script_dir[:-7]
            + "/NarrativeWedging/Narrative_Wedging.jsonl",
        }

        return datasets_info[dataset_name]


class ConllDataset(_IDataset):
    """Class to handle Conll files. Subclass of _IDataset."""

    supported_tasks = ["ner"]

    COLUMN_NAMES = {task: COLUMN_MAPPER[task] for task in supported_tasks}

    def __init__(self, file_path: str, task: str) -> None:
        """Initializes ConllDataset object.

        Args:
            file_path (str): Path to the data file.
            task (str): name of the task to perform
        """
        super().__init__()
        self._file_path = file_path

        if task != "ner":
            raise ValueError(
                f"Given task ({task}) is not matched with ner. CoNLL dataset can ne only loaded for ner!"
            )
        self.task = task

    def load_raw_data(self) -> List[Dict]:
        """Loads dataset into a list tokens and labels

        Returns:
            List[Dict]: list of dict containing tokens and labels
        """
        raw_data = []
        with open(self._file_path) as f:
            content = f.read()
            docs = [
                i.strip()
                for i in re.split(r"-DOCSTART- \S+ \S+ O", content.strip())
                if i != ""
            ]
            for d_id, doc in enumerate(docs):
                #  file content to sentence split
                sentences = re.split(r"\n\n|\n\s+\n", doc.strip())

                if sentences == [""]:
                    continue

                for sent in sentences:
                    # sentence string to token level split
                    tokens = sent.strip().split("\n")

                    # get annotations from token level split
                    token_list = [t.split() for t in tokens]

                    #  get token and labels from the split
                    raw_data.append(
                        {
                            "text": [elt[0] for elt in token_list],
                            "labels": [elt[-1] for elt in token_list],
                        }
                    )
        return raw_data

    def load_data(self) -> List[NERSample]:
        """Loads data from a CoNLL file.

        Returns:
            List[NERSample]: List of formatted sentences from the dataset.
        """
        data = []
        with open(self._file_path) as f:
            content = f.read()
            docs_strings = re.findall(r"-DOCSTART- \S+ \S+ O", content.strip())
            docs = [
                i.strip()
                for i in re.split(r"-DOCSTART- \S+ \S+ O", content.strip())
                if i != ""
            ]
            for d_id, doc in enumerate(docs):
                #  file content to sentence split
                sentences = re.split(r"\n\n|\n\s+\n", doc.strip())

                if sentences == [""]:
                    continue

                for sent in sentences:
                    # sentence string to token level split
                    tokens = sent.strip().split("\n")

                    # get annotations from token level split
                    token_list = [t.split() for t in tokens]

                    #  get token and labels from the split
                    ner_labels = []
                    cursor = 0
                    for split in token_list:
                        ner_labels.append(
                            NERPrediction.from_span(
                                entity=split[-1],
                                word=split[0],
                                start=cursor,
                                end=cursor + len(split[0]),
                                doc_id=d_id,
                                doc_name=(
                                    docs_strings[d_id] if len(docs_strings) > 0 else ""
                                ),
                                pos_tag=split[1],
                                chunk_tag=split[2],
                            )
                        )
                        # +1 to account for the white space
                        cursor += len(split[0]) + 1

                    original = " ".join([label.span.word for label in ner_labels])

                    data.append(
                        NERSample(
                            original=original,
                            expected_results=NEROutput(predictions=ner_labels),
                        )
                    )

        return data

    def export_data(self, data: List[NERSample], output_path: str):
        """Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[NERSample]):
                data to export
            output_path (str):
                path to save the data to
        """
        otext = ""
        temp_id = None
        for i in data:
            text, temp_id = Formatter.process(i, output_format="conll", temp_id=temp_id)
            otext += text + "\n"

        with open(output_path, "wb") as fwriter:
            fwriter.write(bytes(otext, encoding="utf-8"))


class JSONDataset(_IDataset):
    """Class to handle JSON dataset files. Subclass of _IDataset."""

    def __init__(self, file_path: str):
        """Initializes JSONDataset object.

        Args:
            file_path (str): Path to the data file.
        """
        super().__init__()
        self._file_path = file_path

    def load_raw_data(self):
        """Loads data into a raw list"""
        raise NotImplementedError()

    def load_data(self) -> List[Sample]:
        """Loads data into a list of Sample

        Returns:
            List[Sample]: formatted samples
        """
        raise NotImplementedError()

    def export_data(self, data: List[Sample], output_path: str):
        """Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        raise NotImplementedError()


class CSVDataset(_IDataset):
    supported_tasks = [
        "ner",
        "text-classification",
        "summarization",
        "question-answering",
    ]
    COLUMN_NAMES = {task: COLUMN_MAPPER[task] for task in supported_tasks}

    """
    A class to handle CSV files datasets. Subclass of _IDataset.

    Attributes:
        _file_path (Union[str, Dict]):
            The path to the data file or a dictionary containing "data_source" key with the path.
        task (str):
            Specifies the task of the dataset, which can be either "text-classification","ner"
            "question-answering" and "summarization".
        delimiter (str):
            The delimiter used in the CSV file to separate columns (only for file_path as str).
    """

    def __init__(self, file_path: Union[str, Dict], task: str, **kwargs) -> None:
        """
        Initializes a CustomCSVDataset object.

        Args:
            file_path (Union[str, Dict]):
                The path to the data file or a dictionary containing the following keys:
                - "data_source": The path to the data file.
                - "feature_column" (optional): Specifies the column containing input features.
                - "target_column" (optional): Specifies the column containing target labels.
            task (str):
                Specifies the task of the dataset, which can be one of the following:
                - "text-classification"
                - "ner" (Named Entity Recognition)
                - "question-answering"
                - "summarization"
            **kwargs:
                Additional keyword arguments that can be used to configure the dataset (optional).
        """
        super().__init__()
        self._file_path = file_path
        self.task = task
        if type(file_path) == dict:
            self.delimiter = self._find_delimiter(file_path["data_source"])
        else:
            if task in self.COLUMN_NAMES:
                self.COLUMN_NAMES = self.COLUMN_NAMES[self.task]
            elif "is_import" not in kwargs:
                raise ValueError(
                    f"Given task ({task}) is not matched with template. \
                    CSV dataset can ne only loaded for text-classification and ner!"
                )
            self.delimiter = self._find_delimiter(file_path)

        self.column_map = None
        self.kwargs = kwargs

    def load_raw_data(self, standardize_columns: bool = False) -> List[Dict]:
        """Loads data from a csv file into raw lists of strings

        Args:
            standardize_columns (bool): whether to standardize column names

        Returns:
            List[Dict]:
                parsed CSV file into list of dicts
        """

        if type(self._file_path) == dict:
            df = pd.read_csv(self._file_path["data_source"])

            if self.task == "text-classification":
                feature_column = self._file_path.get("feature_column", "text")
                target_column = self._file_path.get("target_column", "label")
            elif self.task == "ner":
                feature_column = self._file_path.get("feature_column", "text")
                target_column = self._file_path.get("target_column", "ner")

            if feature_column not in df.columns or target_column not in df.columns:
                raise ValueError(
                    f"Columns '{feature_column}' and '{target_column}' not found in the dataset."
                )

            if self.task == "text-classification":
                df.rename(
                    columns={feature_column: "text", target_column: "label"}, inplace=True
                )
            elif self.task == "ner":
                df.rename(
                    columns={feature_column: "text", target_column: "ner"}, inplace=True
                )
        else:
            df = pd.read_csv(self._file_path)

        raw_data = []
        if not standardize_columns:
            data = df.to_dict(orient="records")
            if self.task == "ner":
                for row in data:
                    raw_data.append(
                        {
                            key: (val if isinstance(val, list) else eval(val))
                            for key, val in row.items()
                        }
                    )
                return raw_data
            return data

        for _, row in df.iterrows():
            if not self.column_map:
                self.column_map = self._match_column_names(list(row.keys()))

            label_col = (
                self.column_map["ner"] if self.task == "ner" else self.column_map["label"]
            )

            text = row[self.column_map["text"]]
            labels = row[label_col]

            raw_data.append(
                {
                    "text": text
                    if (isinstance(text, list) or self.task != "ner")
                    else eval(text),
                    "labels": labels
                    if (isinstance(labels, list) or self.task != "ner")
                    else eval(labels),
                }
            )

        return raw_data

    def load_data(self) -> List[Sample]:
        """
        Load data from a CSV file and preprocess it based on the specified task.

        Returns:
            List[Sample]: A list of preprocessed data samples.

        Raises:
            ValueError: If the specified task is unsupported.

        Note:
            - If 'is_import' is set to True in the constructor's keyword arguments,
            the data will be imported using the specified 'file_path' and optional
            'column_map' for renaming columns.

            - If 'is_import' is set to False (default), the data will be loaded from
            a CSV file specified in 'file_path', and the 'column_map' will be
            automatically matched with the dataset columns.

            - The supported task types are: 'text-classification', 'ner',
            'summarization', and 'question-answering'. The appropriate task-specific
            loading function will be invoked to preprocess the data.
        """
        if self.kwargs.get("is_import", False):
            kwargs = self.kwargs.copy()
            kwargs.pop("is_import")
            return self._import_data(self._file_path, **kwargs)

        if type(self._file_path) == dict:
            dataset = pd.read_csv(self._file_path["data_source"])
        else:
            dataset = pd.read_csv(self._file_path)
            if not self.column_map:
                self.column_map = self._match_column_names(list(dataset.columns))

        task_functions = {
            "text-classification": self.load_data_classification,
            "ner": self.load_data_ner,
            "summarization": self.load_data_summarization,
            "question-answering": self.load_data_question_answering,
        }

        if self.task in task_functions:
            task_function = task_functions[self.task]
            return task_function(dataset)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def export_data(self, data: List[Sample], output_path: str):
        """Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        if self.task == "ner":
            final_data = defaultdict(list)
            for elt in data:
                tokens, labels, testcase_tokens, testcase_labels = Formatter.process(
                    elt, output_format="csv"
                )
                final_data["text"].append(tokens)
                final_data["labels"].append(labels)
                final_data["testcase_text"].append(testcase_tokens)
                final_data["testcase_labels"].append(testcase_labels)

            if (
                sum([len(labels) for labels in final_data["testcase_labels"]])
                * sum([len(tokens) for tokens in final_data["testcase_text"]])
                == 0
            ):
                final_data.pop("testcase_text")
                final_data.pop("testcase_labels")

            pd.DataFrame(data=final_data).to_csv(output_path, index=False)

        elif self.task == "text-classification":
            rows = []
            for s in data:
                row = Formatter.process(s, output_format="csv")
                rows.append(row)

            df = pd.DataFrame(rows, columns=list(self.COLUMN_NAMES.keys()))
            df.to_csv(output_path, index=False, encoding="utf-8")

    @staticmethod
    def _find_delimiter(file_path: str) -> property:
        """
        Helper function in charge of finding the delimiter character in a csv file.
        Args:
            file_path (str):
                location of the csv file to load
        Returns:
            property:
        """
        sniffer = csv.Sniffer()
        with open(file_path, encoding="utf-8") as fp:
            first_line = fp.readline()
            delimiter = sniffer.sniff(first_line).delimiter
        return delimiter

    def load_data_ner(
        self,
        dataset: pd.DataFrame,
    ) -> List[Sample]:
        """
        Preprocess data for Named Entity Recognition (NER) task.

        Args:
            dataset (pd.DataFrame): Input data in DataFrame format.

        Returns:
            List[Sample]: Preprocessed data samples for NER task.

        """

        if type(self._file_path) == dict:
            feature_column = self._file_path.get("feature_column", "text")
            target_column = self._file_path.get("target_column", "ner")

            if (
                feature_column not in dataset.columns
                or target_column not in dataset.columns
            ):
                raise ValueError(
                    f"Columns '{feature_column}' and '{target_column}' not found in the dataset."
                )

            dataset.rename(
                columns={feature_column: "text", target_column: "ner"},
                inplace=True,
            )

        samples = []
        for row_index, row in dataset.iterrows():
            samples.append(self._row_to_ner_sample(row.to_dict(), row_index))

        return samples

    def load_data_classification(
        self,
        dataset: pd.DataFrame,
    ) -> List[Sample]:
        """
        Load the specified split from the dataset library for classification task.

        Args:
            dataset (pd.DataFrame):
                The input dataset containing the text data and corresponding labels.
            feature_column (str, optional):
                Name of the column in the dataset containing the input text data.
                Default is "text".
            target_column (str, optional):
                Name of the column in the dataset containing the target labels for classification.
                Default is "label".

        Returns:
            List[Sample]:
                Loaded split as a list of Sample objects, where each Sample object consists
                of an input text and its corresponding label.
        """
        if type(self._file_path) == dict:
            feature_column = self._file_path.get("feature_column", "text")
            target_column = self._file_path.get("target_column", "label")

            if (
                feature_column not in dataset.columns
                or target_column not in dataset.columns
            ):
                raise ValueError(
                    f"Columns '{feature_column}' and '{target_column}' not found in the dataset."
                )

            if feature_column and target_column:
                dataset.rename(
                    columns={feature_column: "text", target_column: "label"}, inplace=True
                )

        samples = [
            self._row_to_seq_classification_sample(row) for _, row in dataset.iterrows()
        ]
        return samples

    def load_data_summarization(
        self,
        dataset: pd.DataFrame,
    ) -> List[Sample]:
        """
        Load the specified split from the dataset library for summarization task.

        Args:
            dataset (pd.DataFrame):
                The input dataset containing the document data and corresponding summaries.
            feature_column (str, optional):
                Name of the column in the dataset containing the input document data.
                Default is "document".
            target_column (str, optional):
                Name of the column in the dataset containing the target summaries for summarization.
                Default is "summary".

        Returns:
            List[Sample]:
                Loaded split as a list of Sample objects for summarization task, where each
                Sample object contains a document and its corresponding summary.
        """
        if type(self._file_path) == dict:
            feature_column = self._file_path.get("feature_column", "document")
            target_column = self._file_path.get("target_column", "summary")

            if (
                feature_column not in dataset.columns
                or target_column not in dataset.columns
            ):
                raise ValueError(
                    f"Columns '{feature_column}' and '{target_column}' not found in the dataset."
                )

            dataset.rename(
                columns={feature_column: "document", target_column: "summary"},
                inplace=True,
            )

        samples = [
            self._row_to_sample_summarization(row) for _, row in dataset.iterrows()
        ]
        return samples

    def load_data_question_answering(
        self,
        dataset: pd.DataFrame,
    ) -> List[Sample]:
        """
        Load the specified split from the dataset library for question-answering task.

        Args:
            dataset (pd.DataFrame):
                The input dataset containing the passage, question, and corresponding answers.
            feature_column (dict, optional):
                Dictionary of column names in the dataset containing the input passage and question data.
                Default is {"passage": "passage", "question": "question"}.
            target_column (str, optional):
                Name of the column in the dataset containing the target answers for question-answering.
                Default is "answer".

        Returns:
            List[QASample]:
                Loaded split as a list of QASample objects for question-answering task, where each
                QASample object contains an original question, original context (passage), and the task name.
        """
        if type(self._file_path) == dict:
            feature_column = self._file_path.get(
                "feature_column", {"passage": "passage", "question": "question"}
            )
            target_column = self._file_path.get("target_column", "answer")

            passage_column = feature_column.get("passage", None)
            question_column = feature_column.get("question")

            dataset_columns = set(dataset.columns)
            if (
                "question" not in feature_column
                or feature_column["question"] not in dataset_columns
            ):
                raise ValueError(
                    f"'feature_column' '{feature_column['question']}' not found in the dataset."
                )
            if "answer" not in target_column or target_column not in dataset_columns:
                raise ValueError(
                    f"'target_column' '{target_column}' not found in the dataset."
                )

            if passage_column in dataset.columns:
                if passage_column not in dataset_columns:
                    raise ValueError(
                        f"'feature_column' '{passage_column}' not found in the dataset."
                    )
                dataset.rename(columns={passage_column: "passage"}, inplace=True)
            else:
                dataset["passage"] = "-"

            if question_column in dataset.columns:
                dataset.rename(columns={question_column: "question"}, inplace=True)

            dataset.rename(columns={target_column: "answer"}, inplace=True)

        samples = [
            self._row_to_sample_question_answering(row) for _, row in dataset.iterrows()
        ]
        return samples

    def _row_to_ner_sample(self, row: Dict[str, List[str]], sent_index: int) -> Sample:
        """Convert a row from the dataset into a Sample for the NER task.

        Args:
            row (Dict[str, List[str]]):
                single row of the dataset
            sent_index (int): position of the sentence

        Returns:
            Sample:
                row formatted into a Sample object

        """

        if type(self._file_path) == dict:
            text_col = "text"
            ner_col = "ner"
            pos_col = "pos"
            chunk_col = "chunk"
        else:
            text_col = self.column_map["text"]
            ner_col = self.column_map["ner"]
            pos_col = self.column_map["text"]
            chunk_col = self.column_map["text"]

        for key, value in row.items():
            if isinstance(value, str):
                row[key] = eval(value)

        assert all(isinstance(value, list) for value in row.values()), ValueError(
            f"Column ({sent_index}th) values should be list that contains tokens or labels. "
            "Given CSV file has invalid values"
        )
        token_num = len(row[text_col])
        assert all(len(value) == token_num for value in row.values()), ValueError(
            f"Column ({sent_index}th) values should have same length with number of token in text, "
            f"which is {token_num}"
        )

        original = " ".join(row[text_col])
        ner_labels = list()
        cursor = 0
        for token_indx in range(len(row[text_col])):
            token = row[text_col][token_indx]
            ner_labels.append(
                NERPrediction.from_span(
                    entity=row[ner_col][token_indx],
                    word=token,
                    start=cursor,
                    end=cursor + len(token),
                    pos_tag=row[pos_col][token_indx] if row.get(pos_col, None) else None,
                    chunk_tag=row[chunk_col][token_indx]
                    if row.get(chunk_col, None)
                    else None,
                )
            )
            cursor += len(token) + 1  # +1 to account for the white space

        return NERSample(
            original=original, expected_results=NEROutput(predictions=ner_labels)
        )

    def _row_to_seq_classification_sample(self, row: pd.Series) -> Sample:
        """
        Convert a row from the dataset into a Sample for the text-classification task

        Args:
            row (pd.Series):
                Single row of the dataset as a Pandas Series

        Returns:
            Sample:
                Row formatted into a Sample object
        """
        if type(self._file_path) == dict:
            original = row.loc["text"]
            label = SequenceLabel(label=row.loc["label"], score=1)
        else:
            original = row[self.column_map["text"]]
            #   label score should be 1 since it is ground truth, required for __eq__
            label = SequenceLabel(label=row[self.column_map["label"]], score=1)

        return SequenceClassificationSample(
            original=original,
            expected_results=SequenceClassificationOutput(predictions=[label]),
        )

    def _row_to_sample_summarization(self, row: pd.Series) -> Sample:
        """
        Convert a row from the dataset into a Sample for summarization.

        Args:
            data_row (Dict[str, str]):
                Single row of the dataset.

        Returns:
            Sample:
                Row formatted into a Sample object for summarization.
        """
        if type(self._file_path) == dict:
            original = row.loc["document"]
            summary = row.loc["summary"]
        else:
            original = row[self.column_map["text"]]
            summary = row[self.column_map["summary"]]

        return SummarizationSample(
            original=original, expected_results=summary, task="summarization"
        )

    def _row_to_sample_question_answering(self, row: pd.Series) -> QASample:
        """
        Convert a row from the dataset into a QASample for question-answering.

        Args:
            row (pd.Series):
                Single row of the dataset.

        Returns:
            QASample:
                Row formatted into a QASample object for question-answering.
        """

        if type(self._file_path) == dict:
            question = row.loc["question"]
            passage = row.loc["passage"]
        else:
            question = row[self.column_map["text"]]
            passage = row[self.column_map["context"]]

        return QASample(
            original_question=question,
            original_context=passage,
            task="question-answering",
        )

    def _match_column_names(self, column_names: List[str]) -> Dict[str, str]:
        """Helper function to map original column into standardized ones.

        Args:
            column_names (List[str]):
                list of column names of the csv file

        Returns:
            Dict[str, str]:
                mapping from the original column names into 'standardized' names
        """
        column_map = {k: None for k in self.COLUMN_NAMES}
        for c in column_names:
            for key, reference_columns in self.COLUMN_NAMES.items():
                if c.lower() in reference_columns:
                    column_map[key] = c

        not_referenced_columns = {
            k: self.COLUMN_NAMES[k] for k, v in column_map.items() if v is None
        }
        if "text" in not_referenced_columns and (
            "ner" in not_referenced_columns or "label" in not_referenced_columns
        ):
            raise OSError(
                f"CSV file is invalid. CSV handler works with template column names!\n"
                f"{', '.join(not_referenced_columns.keys())} column could not be found in header.\n"
                f"You can use following namespaces:\n{not_referenced_columns}"
            )
        return column_map

    def _import_data(self, file_name, **kwargs) -> List[Sample]:
        """Helper function to import testcases from csv file after editing.

        Args:
            file_name (str):    path to the csv file
            **kwargs:           additional arguments to pass to pandas.read_csv

        Returns:
            List[Sample]:       list of samples
        """
        data = pd.read_csv(file_name, **kwargs)
        custom_names = {
            "question-answering": "qa",
            "text-classification": "sequenceclassification",
        }
        sample_models = {
            k.lower(): v for k, v in sample.__dict__.items() if k.endswith("Sample")
        }
        samples = []

        for i in data.to_dict(orient="records"):
            if self.task in custom_names:
                sample_name = custom_names[self.task] + "sample"
            else:
                sample_name = self.task.lower() + "sample"
            samples.append(sample_models[sample_name](**i))
        return samples


class JSONLDataset(_IDataset):
    """Class to handle JSONL datasets. Subclass of _IDataset."""

    supported_tasks = [
        "ner",
        "text-classification",
        "question-answering",
        "summarization",
        "toxicity",
        "translation",
        "security",
        "clinical-tests",
        "disinformation-test",
    ]
    COLUMN_NAMES = {task: COLUMN_MAPPER[task] for task in supported_tasks}

    def __init__(self, file_path: str, task: str) -> None:
        """Initializes JSONLDataset object.

        Args:
            file_path (str): Path to the data file.
            task (str): name of the task to perform
        """
        super().__init__()
        self._file_path = file_path
        self.task = task
        self.column_matcher = None

    def _match_column_names(self, column_names: List[str]) -> Dict[str, str]:
        """Helper function to map original column into standardized ones.

        Args:
            column_names (List[str]):
                list of column names of the csv file

        Returns:
            Dict[str, str]:
                mapping from the original column names into 'standardized' names
        """
        column_map = {}
        for column in column_names:
            for key, reference_columns in self.COLUMN_NAMES[self.task].items():
                if column.lower() in reference_columns:
                    column_map[key] = column

        not_referenced_columns = [
            col for col in self.COLUMN_NAMES[self.task] if col not in column_map
        ]

        if "text" in not_referenced_columns:
            raise OSError(
                f"Your dataset needs to have at least have a column with one of the following name: "
                f"{self.COLUMN_NAMES[self.task]['text']}, found: {column_names}."
            )

        for missing_col in not_referenced_columns:
            column_map[missing_col] = None
        return column_map

    def load_raw_data(self) -> List[Dict]:
        """Loads data from a JSON file into a list"""
        with jsonlines.open(self._file_path) as reader:
            data = [obj for obj in reader]
        return data

    def load_data(self) -> List[Sample]:
        """Loads data from a JSONL file and format it into a list of Sample.

        Returns:
            list[Sample]: Loaded text data.
        """
        data = []
        with jsonlines.open(self._file_path) as reader:
            for item in reader:
                if self.column_matcher is None:
                    self.column_matcher = self._match_column_names(item.keys())

                if self.task == "question-answering":
                    expected_results = item.get(self.column_matcher["answer"])
                    if isinstance(expected_results, str) or isinstance(
                        expected_results, bool
                    ):
                        expected_results = [str(expected_results)]

                    data.append(
                        QASample(
                            original_question=item[self.column_matcher["text"]],
                            original_context=item.get(
                                self.column_matcher["context"], "-"
                            ),
                            expected_results=expected_results,
                            dataset_name=self._file_path.split("/")[-2],
                        )
                    )

                elif self.task == "summarization":
                    expected_results = item.get(self.column_matcher["summary"])
                    if isinstance(expected_results, str) or isinstance(
                        expected_results, bool
                    ):
                        expected_results = [str(expected_results)]
                    data.append(
                        SummarizationSample(
                            original=item[self.column_matcher["text"]],
                            expected_results=expected_results,
                            dataset_name=self._file_path.split("/")[-2],
                        )
                    )
                elif self.task == "toxicity":
                    data.append(
                        ToxicitySample(
                            prompt=item[self.column_matcher["text"]],
                            dataset_name=self._file_path.split("/")[-2],
                        )
                    )

                elif self.task == "translation":
                    data.append(
                        TranslationSample(
                            original=item[self.column_matcher["text"]],
                            dataset_name=self._file_path.split("/")[-2],
                        )
                    )
                elif self.task == "security":
                    data.append(
                        SecuritySample(
                            prompt=item["text"],
                            task=self.task,
                            dataset_name=self._file_path.split("/")[-2],
                        )
                    )

                elif self.task == "clinical-tests":
                    data.append(
                        ClinicalSample(
                            patient_info_A=item["Patient info A"],
                            patient_info_B=item["Patient info B"],
                            diagnosis=item["Diagnosis"],
                            task=self.task,
                            dataset_name=self._file_path.split("/")[-2],
                        )
                    )
                elif self.task == "disinformation-test":
                    data.append(
                        DisinformationSample(
                            hypothesis=item["hypothesis"],
                            statements=item["statements"],
                            task=self.task,
                            dataset_name=self._file_path.split("/")[-2],
                        )
                    )
        return data

    def export_data(self, data: List[Sample], output_path: str):
        """Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[Sample]):
                data to export
            output_path (str):
                path to save the data to
        """
        raise NotImplementedError()


class HuggingFaceDataset(_IDataset):
    """Example dataset class that loads data using the Hugging Face dataset library."""

    supported_tasks = ["text-classification", "summarization", "ner"]

    LIB_NAME = "datasets"
    COLUMN_NAMES = {task: COLUMN_MAPPER[task] for task in supported_tasks}

    def __init__(self, dataset_name: str, task: str):
        """Initialize the HuggingFaceDataset class.

        Args:
            dataset_name (str):
                Name of the dataset to load.
            task (str):
                Task to be evaluated on.
        """
        self.dataset_name = dataset_name
        self.task = task
        self._check_datasets_package()

    def _check_datasets_package(self):
        """Check if the 'datasets' package is installed and import the load_dataset function.

        Raises an error if the package is not found.
        """
        if try_import_lib(self.LIB_NAME):
            dataset_module = importlib.import_module(self.LIB_NAME)
            self.load_dataset = getattr(dataset_module, "load_dataset")
        else:
            raise ModuleNotFoundError(
                f"The '{self.LIB_NAME}' package is not installed. Please install it using 'pip install {self.LIB_NAME}'."
            )

    def load_data_ner(
        self,
        feature_column: str,
        target_column: str,
        split: str,
        subset: str = None,
    ) -> List[Sample]:
        """Load the specified split from the given ner dataset."""
        feature_column = "text" if feature_column is None else feature_column
        target_column = "label" if target_column is None else target_column
        split = "test" if split is None else split

        if subset:
            dataset = self.load_dataset(self.dataset_name, name=subset, split=split)
        else:
            dataset = self.load_dataset(self.dataset_name, split=split)

        if "label" in str(type(dataset.features[target_column].feature)):
            label_names = dataset.features[target_column].feature.names
            dataset = map(
                lambda example: {
                    "tokens": example[feature_column],
                    "ner_tags": [label_names[x] for x in example[target_column]],
                },
                dataset,
            )
        else:
            dataset = map(
                lambda example: {
                    "tokens": example[feature_column],
                    "ner_tags": example[target_column],
                },
                dataset,
            )

        samples = [self._row_to_ner_sample(example) for example in dataset]
        return samples

    def load_data_classification(
        self,
        feature_column: str,
        target_column: str,
        split: str,
        subset: str = None,
    ) -> List[Sample]:
        """Load the specified split from the dataset library.

        Args:
            feature_column (str):
                Name of the feature_column column.
            target_column (str):
                Name of the target_column column.
            split (str):
                Name of the split to load (e.g., train, validation, test).
            subset (str):
                Name of the configuration.

        Returns:
            List[Sample]:
                Loaded split as a list of Sample objects.
        """
        feature_column = "text" if feature_column is None else feature_column
        target_column = "label" if target_column is None else target_column
        split = "test" if split is None else split

        if subset:
            dataset = self.load_dataset(self.dataset_name, name=subset, split=split)
        else:
            dataset = self.load_dataset(self.dataset_name, split=split)

        dataset = dataset.map(
            lambda example: {
                "text": example[feature_column],
                "label": example[target_column],
            }
        )

        samples = [self._row_to_sample_classification(example) for example in dataset]
        return samples

    def load_data_summarization(
        self,
        feature_column: str,
        target_column: str,
        split: str,
        subset: str = None,
    ) -> List[Sample]:
        """Load the specified split from the dataset for summarization task.

        Args:
            feature_column (str):
                Name of the column containing the input text or document.
            target_column (str):
                Name of the column containing the target summary.
            split (str):
                Name of the split to load (e.g., train, validation, test).
            subset (str):
                Name of the configuration or subset to load.

        Returns:
            List[Sample]:
                Loaded split as a list of Sample objects for summarization task.
        """
        feature_column = "document" if feature_column is None else feature_column
        target_column = "summary" if target_column is None else target_column
        split = "test" if split is None else split

        if subset:
            dataset = self.load_dataset(self.dataset_name, name=subset, split=split)
        else:
            dataset = self.load_dataset(self.dataset_name, split=split)

        if feature_column and target_column:
            dataset = dataset.map(
                lambda example: {
                    "document": example[feature_column],
                    "summary": example[target_column],
                }
            )

        samples = [self._row_to_sample_summarization(example) for example in dataset]
        return samples

    def load_data_qa(
        self,
        question_column: str,
        context_column: str,
        target_column: str,
        split: str,
        subset: str = None,
    ) -> List[Sample]:
        """Load the specified split from the dataset for QA task.

        Args:
            feature_column (str):
                Name of the column containing the input text or document.
            target_column (str):
                Name of the column containing the target summary.
            split (str):
                Name of the split to load (e.g., train, validation, test).
            subset (str):
                Name of the configuration or subset to load.

        Returns:
            List[Sample]:
                Loaded split as a list of Sample objects for QA task.
        """
        question_column = "question" if question_column is None else question_column
        target_column = "answer" if target_column is None else target_column
        split = "test" if split is None else split

        if subset:
            dataset = self.load_dataset(self.dataset_name, name=subset, split=split)
        else:
            dataset = self.load_dataset(self.dataset_name, split=split)

        dataset = dataset.map(
            lambda example: {
                "question": example[question_column],
                "context": example[context_column],
                "answer": example[target_column],
            }
        )

        samples = [self._row_to_sample_qa(example) for example in dataset]
        return samples

    def load_raw_data(
        self,
        split: str = "test",
        subset: str = None,
    ) -> List:
        """Loads data into a list"""
        if subset:
            dataset = self.load_dataset(self.dataset_name, name=subset, split=split)
        else:
            dataset = self.load_dataset(self.dataset_name, split=split)

        return dataset.to_list()

    def load_data(
        self,
        feature_column: Optional[str] = None,
        target_column: Optional[str] = None,
        split: Optional[str] = None,
        subset: Optional[str] = None,
    ) -> List[Sample]:
        """Load the specified data based on the task.

        Args:
            feature_column (str):
                Name of the column containing the input text or document.
            target_column (str):
                Name of the column containing the target label or summary.
            split (str):
                Name of the split to load (e.g., train, validation, test).
            subset (str):
                Name of the configuration or subset to load.

        Returns:
            List[Sample]:
                Loaded data as a list of Sample objects.

        Raises:
            ValueError:
                If an unsupported task is provided.
        """
        if self.task == "text-classification":
            return self.load_data_classification(
                feature_column, target_column, split, subset
            )
        elif self.task == "summarization":
            return self.load_data_summarization(
                feature_column, target_column, split, subset
            )
        elif self.task == "ner":
            return self.load_data_ner(feature_column, target_column, split, subset)
        else:
            raise ValueError(f"Unsupported task for HF datasets: {self.task}")

    @staticmethod
    def _row_to_sample_summarization(data_row: Dict[str, str]) -> Sample:
        """Convert a row from the dataset into a Sample for summarization.

        Args:
            data_row (Dict[str, str]):
                Single row of the dataset.

        Returns:
            Sample:
                Row formatted into a Sample object for summarization.
        """
        original = data_row.get("document", "")
        summary = data_row.get("summary", "")

        return SummarizationSample(original=original, expected_results=summary)

    @staticmethod
    def _row_to_sample_qa(data_row: Dict[str, str]) -> Sample:
        """Convert a row from the dataset into a Sample for summarization.

        Args:
            data_row (Dict[str, str]):
                Single row of the dataset.

        Returns:
            Sample:
                Row formatted into a Sample object for summarization.
        """
        context = data_row.get("context", "")
        question = data_row.get("question", "")
        answer = data_row.get("answer", "")
        if isinstance(answer, str):
            answer = [answer]

        return QASample(
            original_question=question,
            original_context=context,
            actual_results=answer,
        )

    def export_data(self, data: List[Sample], output_path: str):
        """Exports the data to the corresponding format and saves it to 'output_path'.

        Args:
            data (List[Sample]):
                Data to export.
            output_path (str):
                Path to save the data to.
        """
        rows = []
        for s in data:
            row = Formatter.process(s, output_format="csv")
            rows.append(row)

        df = pd.DataFrame(
            rows, columns=list(self.COLUMN_NAMES["text-classification"].keys())
        )
        df.to_csv(output_path, index=False, encoding="utf-8")

    def _row_to_sample_classification(self, data_row: Dict[str, str]) -> Sample:
        """Convert a row from the dataset into a Sample for text classification.

        Args:
            data_row (Dict[str, str]):
                Single row of the dataset.

        Returns:
            Sample:
                Row formatted into a Sample object.
        """
        input_column = next(
            (
                col
                for col in self.COLUMN_NAMES["text-classification"]["text"]
                if col in data_row
            ),
            None,
        )
        output_column = next(
            (
                col
                for col in self.COLUMN_NAMES["text-classification"]["label"]
                if col in data_row
            ),
            None,
        )

        original = data_row.get(input_column, "")
        label = SequenceLabel(label=data_row.get(output_column, ""), score=1)

        return SequenceClassificationSample(
            original=original,
            expected_results=SequenceClassificationOutput(predictions=[label]),
        )

    def _row_to_ner_sample(self, data_row: dict) -> Sample:
        """Convert a row from the dataset into a Sample for NER.

        Args:
            data_row (Dict[str, str]):
                Single row of the dataset.

        Returns:
            Sample:
                Row formatted into a Sample object.
        """
        input_column = next(
            (col for col in self.COLUMN_NAMES["ner"]["text"] if col in data_row),
            None,
        )
        output_column = next(
            (col for col in self.COLUMN_NAMES["ner"]["ner"] if col in data_row),
            None,
        )

        tokens = data_row.get(input_column, [])
        labels = data_row.get(output_column, [])

        #  get token and labels from the split
        ner_labels = []
        cursor = 0
        for token, label in zip(tokens, labels):
            ner_labels.append(
                NERPrediction.from_span(
                    entity=label,
                    word=token,
                    start=cursor,
                    end=cursor + len(token),
                    doc_id=0,
                    doc_name="",
                    pos_tag="XX",
                    chunk_tag="XX",
                )
            )
            # +1 to account for the white space
            cursor += len(token) + 1

        original = " ".join(tokens)
        return NERSample(
            original=original, expected_results=NEROutput(predictions=ner_labels)
        )
