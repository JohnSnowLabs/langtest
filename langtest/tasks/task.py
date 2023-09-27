import re
from abc import ABC, abstractmethod
from langtest.modelhandler import ModelAPI, LANGCHAIN_HUBS

# langtest exceptions
from langtest.exceptions.columns import ColumnNameError

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


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    task_registry = {}
    _name = None

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


class NERTask(BaseTask):
    """Named Entity Recognition task."""

    _name = "ner"
    _default_col = (
        {
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
    )

    def create_sample(cls, original, expected_results: NEROutput) -> NERSample:
        """Create a sample."""
        return NERSample(original=original, expected_results=expected_results)


class TextClassificationTask(BaseTask):
    """Text Classification task."""

    _name = "textclassification"
    _default_col = {
        "text": ["text", "sentences", "sentence", "sample"],
        "label": ["label", "labels ", "class", "classes"],
    }

    def create_sample(
        cls, original, labels: SequenceLabel
    ) -> SequenceClassificationSample:
        """Create a sample."""
        return SequenceClassificationSample(
            original=original,
            expected_results=SequenceClassificationOutput(predictions=[labels]),
        )


class QuestionAnsweringTask(BaseTask):
    """Question Answering task."""

    _name = "qa"
    _default_col = {
        "text": ["question"],
        "context": ["context", "passage"],
        "answer": ["answer", "answer_and_def_correct_predictions"],
    }

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
            prompt=row_data[column_mapper["text"]],
            dataset_name=dataset_name,
        )


class WinoBiasTask(BaseTask):
    """WinoBias task."""

    _name = "winobias"
    _default_col = {"text": ["text", "prompt"]}

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
