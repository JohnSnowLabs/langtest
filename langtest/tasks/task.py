import re
from abc import ABC, abstractmethod
from langtest.modelhandler import ModelAPI, LANGCHAIN_HUBS

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

    def create_sample(cls, row_data: dict, dataset_name: str = "qa") -> QASample:
        """Create a sample."""
        expected_results = row_data.get(cls._default_col["answer"])
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]
        return QASample(
            original_question=row_data[cls._default_col["text"]],
            original_context=row_data.get(cls._default_col["context"], "-"),
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

        # validate the feature_column and target_column with _default_col
        if (
            feature_column not in cls._default_col["text"]
            and feature_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {cls._default_col['text']} \
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )

        if (
            target_column not in cls._default_col["summary"]
            and target_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided target_column is not supported. \
                    \nPlease choose one of the supported target_column: {cls._default_col['summary']}\
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )

        expected_results = row_data.get(target_column)
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]

        return SummarizationSample(
            original=row_data[feature_column],
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
        if (
            feature_column not in cls._default_col["text"]
            and feature_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {cls._default_col['text']} \
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )

        return TranslationSample(
            original=row_data[feature_column],
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

        if (
            feature_column not in cls._default_col["text"]
            and feature_column not in row_data
        ):
            raise AssertionError(
                f"\nProvided feature_column is not supported.\
                    \nPlease choose one of the supported feature_column: {cls._default_col['text']} \
                    \n\nOr classifiy the features and target columns from {list(row_data.keys())}"
            )

        return ToxicitySample(
            prompt=row_data[feature_column],
            dataset_name=dataset_name,
        )


class SecurityTask(BaseTask):
    """Security task."""

    _name = "security"
    _default_col = {"text": ["text", "prompt"]}

    def create_sample(
        cls, row_data: dict, feature_column= "text", dataset_name: str = "security"
    ) -> SecuritySample:
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
        return SecuritySample(
            prompt=row_data[cls._default_col["text"]],
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
        cls, row_data: dict, dataset_name: str = "clinicaltests"
    ) -> ClinicalSample:
        """Create a sample."""

        return ClinicalSample(
            patient_info_A=row_data[cls._default_col["Patient info A"]],
            patient_info_B=row_data[cls._default_col["Patient info B"]],
            diagnosis=row_data[cls._default_col["Diagnosis"]],
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
        feature_column=["hypothesis", "statements"],
        dataset_name: str = "disinformationtest",
    ) -> DisinformationSample:
        """Create a sample."""

        return DisinformationSample(
            hypothesis=row_data[cls._default_col["hypothesis"]],
            statements=row_data[cls._default_col["statements"]],
            dataset_name=dataset_name,
        )
