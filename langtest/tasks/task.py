import re
from abc import ABC, abstractmethod
from typing import Union
from langtest.modelhandler import ModelAPI, LANGCHAIN_HUBS

# langtest exceptions
from langtest.exceptions.datasets import ColumnNameError

from langtest.utils.custom_types import (
    NEROutput,
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
    WinoBiasSample,
    LegalSample,
    FactualitySample,
    SensitivitySample,
    LLMAnswerSample,
)
from langtest.utils.custom_types.predictions import NERPrediction


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    task_registry = {}
    _name = None
    sample_class = None

    @classmethod
    @abstractmethod
    def create_sample(cls, *args, **kwargs) -> Sample:
        """Run the task."""
        pass

    @classmethod
    def load_model(cls, model_path: str, model_hub: str, *args, **kwargs):
        """Load the model."""

        models = ModelAPI.model_registry

        base_hubs = list(models.keys())
        base_hubs.remove("llm")
        supported_hubs = base_hubs + list(LANGCHAIN_HUBS.keys())

        if model_hub not in supported_hubs:
            raise AssertionError(
                f"Provided model hub is not supported. Please choose one of the supported model hubs: {supported_hubs}"
            )

        if model_hub in LANGCHAIN_HUBS:
            # LLM models
            cls.model = models["llm"][cls._name].load_model(
                hub=model_hub, path=model_path, *args, **kwargs
            )
        else:
            # JSL, Huggingface, and Spacy models
            cls.model = models[model_hub][cls._name].load_model(
                path=model_path, *args, **kwargs
            )
        return cls.model

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        task_name = re.sub(
            r"(?<=[a-z])(?=[A-Z][a-z])", "-", cls.__name__.replace("Task", "")
        ).lower()

        cls.task_registry[task_name] = cls

    def __eq__(self, __value: object) -> bool:
        """Check if the task is equal to another task."""
        if isinstance(__value, str):
            return self.__class__.__name__.replace("Task", "").lower() == __value.lower()
        return super().__eq__(__value)

    def column_mapping(self, item_keys, *args, **kwargs):
        """Return the column mapping."""

        coulumn_mapper = {}

        for key in self._default_col:
            for item in item_keys:
                if item.lower() in self._default_col[key]:
                    coulumn_mapper[key] = item
                else:
                    raise ColumnNameError(self._default_col[key], item_keys)

        return coulumn_mapper

    @property
    def get_sample_cls(self):
        """Return the sample class."""
        if self.sample_class:
            return self.sample_class
        return None


class TaskManager:
    """Task manager."""

    def __init__(self, task_name: str):
        if task_name not in BaseTask.task_registry:
            raise AssertionError(
                f"Provided task is not supported. Please choose one of the supported tasks: {list(BaseTask.task_registry.keys())}"
            )
        self.__task_name = task_name
        self.__task: BaseTask = BaseTask.task_registry[task_name]()

    def create_sample(self, *args, **kwargs):
        """Add a task to the task manager."""
        # filter out the key with contains column name
        if "feature_column" in kwargs:
            column_names = kwargs["feature_column"]
            if isinstance(column_names, dict):
                kwargs.pop("feature_column")
                kwargs.update(column_names)

        return self.__task.create_sample(*args, **kwargs)

    def model(self, *args, **kwargs) -> "ModelAPI":
        """Add a task to the task manager."""
        return self.__task.load_model(*args, **kwargs)

    def __eq__(self, __value: str) -> bool:
        """Check if the task is equal to another task."""
        return self.__task_name == __value.lower()

    def __hash__(self) -> int:
        """Return the hash of the task name."""
        return hash(self.__task_name)

    def __str__(self) -> str:
        """Return the task name."""
        return self.__task_name

    @property
    def task_name(self):
        """Return the task name."""
        return self.__task_name

    @property
    def get_sample_class(self):
        """
        Return the sample class.

        Returns:
            Sample: Sample class
        """
        return self.__task.get_sample_cls


class NERTask(BaseTask):
    """Named Entity Recognition task."""

    _name = "ner"
    _default_col = {
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
    }
    sample_class = NERSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="text",
        target_column: str = "ner",
        pos_tag: str = "pos",
        chunk_tag: str = "chunk_tag",
        *args,
        **kwargs,
    ) -> NERSample:
        """Create a sample."""
        keys = list(row_data.keys())
        if len(row_data) == 2:
            original = row_data.get(feature_column)
            expected_results = row_data.get(target_column)
        else:
            if set([feature_column, target_column, pos_tag, chunk_tag]).issubset(
                set(keys)
            ):
                # if the column names are provided, use them directly
                column_mapper = {
                    feature_column: feature_column,
                    target_column: target_column,
                    pos_tag: pos_tag,
                    chunk_tag: chunk_tag,
                }
            else:
                # auto-detect the default column names from the row_data
                column_mapper = cls.column_mapping(list(row_data.keys()))

            for key, value in row_data.items():
                if isinstance(value, str):
                    row_data[key] = value[2:-2].split(", ")
                else:
                    row_data[key] = value

            original = " ".join(row_data[column_mapper[feature_column]])
            ner_labels = list()
            cursor = 0
            for token_indx in range(len(row_data[column_mapper[feature_column]])):
                token = row_data[column_mapper[feature_column]][token_indx]
                ner_labels.append(
                    NERPrediction.from_span(
                        entity=row_data[column_mapper[target_column]][token_indx],
                        word=token,
                        start=cursor,
                        end=cursor + len(token),
                        pos_tag=row_data[column_mapper[pos_tag]][token_indx]
                        if row_data.get(column_mapper[pos_tag], None)
                        else None,
                        chunk_tag=row_data[column_mapper[chunk_tag]][token_indx]
                        if row_data.get(column_mapper[chunk_tag], None)
                        else None,
                    )
                )
                cursor += len(token) + 1  # +1 to account for the white space

            expected_results = NEROutput(predictions=ner_labels)

        return NERSample(original=original, expected_results=expected_results)


class TextClassificationTask(BaseTask):
    """Text Classification task."""

    _name = "textclassification"
    _default_col = {
        "text": ["text", "sentences", "sentence", "sample"],
        "label": ["label", "labels ", "class", "classes"],
    }
    sample_class = SequenceClassificationSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="text",
        target_column: Union[SequenceLabel, str] = "label",
    ) -> SequenceClassificationSample:
        """Create a sample."""
        keys = list(row_data.keys())

        if set([feature_column, feature_column]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {feature_column: feature_column, target_column: target_column}
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(list(row_data.keys()))

        labels = row_data.get(column_mapper[target_column])

        if isinstance(labels, SequenceLabel):
            labels = [labels]
        elif isinstance(labels, list):
            labels = [
                SequenceLabel(label=label, score=1.0) if isinstance(label, str) else label
                for label in labels
            ]
        else:
            labels = [SequenceLabel(label=labels, score=1.0)]

        return SequenceClassificationSample(
            original=row_data[column_mapper[feature_column]],
            expected_results=SequenceClassificationOutput(predictions=labels),
        )


class QuestionAnsweringTask(BaseTask):
    """Question Answering task."""

    _name = "qa"
    _default_col = {
        "text": ["question"],
        "context": ["context", "passage"],
        "answer": ["answer", "answer_and_def_correct_predictions"],
    }
    sample_class = QASample

    def create_sample(
        cls,
        row_data: dict,
        dataset_name: str = "qa",
        question: str = "text",
        context: str = "context",
        target_column: str = "answer",
    ) -> QASample:
        """Create a sample."""
        keys = list(row_data.keys())
        if set([question, context, target_column]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {
                question: question,
                context: context,
                target_column: target_column,
            }
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(keys)

        expected_results = (
            row_data.get(column_mapper[target_column], "-")
            if target_column in column_mapper
            else None
        )
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]

        return QASample(
            original_question=row_data[column_mapper[question]],
            original_context=row_data.get(column_mapper[context], "-"),
            expected_results=expected_results,
            dataset_name=dataset_name,
        )


class SummarizationTask(BaseTask):
    """Summarization task."""

    _name = "summarization"
    _default_col = {"text": ["text", "document"], "summary": ["summary"]}
    sample_class = SummarizationSample

    def create_sample(
        cls,
        row_data: dict,
        feature_column="document",
        target_column="summary",
        dataset_name: str = "default_summarization_prompt",
    ) -> SummarizationSample:
        """Create a sample."""
        keys = list(row_data.keys())

        if set([feature_column, target_column]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {feature_column: feature_column, target_column: target_column}
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(list(row_data.keys()))

        expected_results = row_data.get(column_mapper[target_column])
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]

        return SummarizationSample(
            original=row_data[column_mapper[feature_column]],
            expected_results=expected_results,
            dataset_name=dataset_name,
        )


class TranslationTask(BaseTask):
    """Translation task."""

    _name = "translation"
    _default_col = {"text": ["text", "original", "sourcestring"]}
    sample_class = TranslationSample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "translation"
    ) -> TranslationSample:
        """Create a sample."""
        keys = list(row_data.keys())

        if set([feature_column]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {feature_column: feature_column}
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(keys)

        return TranslationSample(
            original=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class ToxicityTask(BaseTask):
    """Toxicity task."""

    _name = "toxicity"
    _default_col = {"text": ["text"]}
    sample_class = ToxicitySample

    def create_sample(
        cls, row_data: dict, feature_column: str = "text", dataset_name: str = "toxicity"
    ) -> ToxicitySample:
        """Create a sample."""

        keys = list(row_data.keys())

        if set([feature_column]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {feature_column: feature_column}
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(keys)

        return ToxicitySample(
            prompt=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class SecurityTask(BaseTask):
    """Security task."""

    _name = "security"
    _default_col = {"text": ["text", "prompt"]}
    sample_class = SecuritySample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "security"
    ) -> SecuritySample:
        """Create a sample."""

        keys = list(row_data.keys())

        if set([feature_column]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {feature_column: feature_column}
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(list(row_data.keys()))

        return SecuritySample(
            prompt=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class ClinicalTestsTask(BaseTask):
    """Clinical Tests task."""

    _name = "clinicaltests"
    _default_col = {
        "Patient info A": [
            "Patient info A",
            "patient info a",
        ],
        "Patient info B": [
            "Patient info B",
            "patient info b",
        ],
        "Diagnosis": [
            "Diagnosis",
            "diagnosis",
        ],
    }
    sample_class = ClinicalSample

    def create_sample(
        cls,
        row_data: dict,
        dataset_name: str = "clinicaltests",
        patient_info_A: str = "Patient info A",
        patient_info_B: str = "Patient info B",
        diagnosis: str = "Diagnosis",
    ) -> ClinicalSample:
        """Create a sample."""

        keys = list(row_data.keys())

        if set([patient_info_A, patient_info_B, diagnosis]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {
                patient_info_A: patient_info_A,
                patient_info_B: patient_info_B,
                diagnosis: diagnosis,
            }
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(list(row_data.keys()))

        return ClinicalSample(
            patient_info_A=row_data[column_mapper[patient_info_A]],
            patient_info_B=row_data[column_mapper[patient_info_B]],
            diagnosis=row_data[column_mapper[diagnosis]],
            dataset_name=dataset_name,
        )


class DisinformationTestTask(BaseTask):
    """Disinformation Test task."""

    _name = "disinformationtest"

    _default_col = {
        "hypothesis": ["hypothesis", "thesis"],
        "statements": ["statements", "headlines"],
    }
    sample_class = DisinformationSample

    def create_sample(
        cls,
        row_data: dict,
        hypothesis: str = "hypothesis",
        statements: str = "statements",
        dataset_name: str = "disinformationtest",
    ) -> DisinformationSample:
        """Create a sample."""

        keys = list(row_data.keys())

        if set([hypothesis, statements]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {hypothesis: hypothesis, statements: statements}
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(list(row_data.keys()))

        return DisinformationSample(
            hypothesis=row_data[column_mapper["hypothesis"]],
            statements=row_data[column_mapper["statements"]],
            dataset_name=dataset_name,
        )


class PoliticalTask(BaseTask):
    """Political task."""

    _name = "political"
    _default_col = {"text": ["text", "prompt"]}
    sample_class = LLMAnswerSample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "political"
    ) -> LLMAnswerSample:
        """Create a sample."""
        keys = list(row_data.keys())

        if set([feature_column]).issubset(set(keys)):
            # if the column names are provided, use them directly
            column_mapper = {feature_column: feature_column}
        else:
            # auto-detect the default column names from the row_data
            column_mapper = cls.column_mapping(list(row_data.keys()))

        return LLMAnswerSample(
            prompt=row_data[column_mapper[feature_column]],
            dataset_name=dataset_name,
        )


class WinoBiasTask(BaseTask):
    """WinoBias task."""

    _name = "winobias"
    _default_col = {"text": ["text", "prompt"]}
    sample_class = WinoBiasSample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "winobias"
    ) -> WinoBiasSample:
        """Create a sample."""
        if (
            feature_column not in cls._default_col["text"]
            and feature_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {cls._default_col['text']} \
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )

        column_mapper = cls.column_mapping(list(row_data.keys()))

        return WinoBiasSample(
            prompt=row_data[column_mapper["text"]],
            dataset_name=dataset_name,
        )


class LegalTask(BaseTask):
    """Legal task."""

    _name = "legal"
    _default_col = {"text": ["text", "prompt"]}
    sample_class = LegalSample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "legal"
    ) -> LegalSample:
        """Create a sample."""
        if (
            feature_column not in cls._default_col["text"]
            and feature_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {cls._default_col['text']} \
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )
        column_mapper = cls.column_mapping(list(row_data.keys()))

        return LegalSample(
            prompt=row_data[column_mapper["text"]],
            dataset_name=dataset_name,
        )


class FactualityTask(BaseTask):
    """Factuality task."""

    _name = "factuality"
    _default_col = {"text": ["text", "prompt"]}
    sample_class = FactualitySample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "factuality"
    ) -> FactualitySample:
        """Create a sample."""
        if (
            feature_column not in cls._default_col["text"]
            and feature_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {cls._default_col['text']} \
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )
        column_mapper = cls.column_mapping(list(row_data.keys()))

        return FactualitySample(
            prompt=row_data[column_mapper["text"]],
            dataset_name=dataset_name,
        )


class SensitivityTask(BaseTask):
    """Sensitivity task."""

    _name = "sensitivity"
    _default_col = {"text": ["text", "prompt"]}
    sample_class = SensitivitySample

    def create_sample(
        cls, row_data: dict, feature_column="text", dataset_name: str = "sensitivity"
    ) -> SensitivitySample:
        """Create a sample."""
        if (
            feature_column not in cls._default_col["text"]
            and feature_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {cls._default_col['text']} \
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )
        column_mapper = cls.column_mapping(list(row_data.keys()))

        return SensitivitySample(
            prompt=row_data[column_mapper["text"]],
            dataset_name=dataset_name,
        )