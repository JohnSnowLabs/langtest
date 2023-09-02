from abc import ABC, abstractmethod
import re
from langtest.modelhandler.modelhandler import ModelFactory
from langtest.utils.custom_types.output import NEROutput

from langtest.utils.custom_types.sample import NERSample
from langtest.modelhandler import ModelAPI, LANGCHAIN_HUBS

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

        if model_hub in list(models.keys()) + list(LANGCHAIN_HUBS.keys()):
            if model_hub in LANGCHAIN_HUBS:
                # LLM models
                cls.model = models["llm"][cls._name].load_model(
                    model_hub, model_path, *args, **kwargs
                )
            else:
                # JSL, Huggingface, and Spacy models
                cls.model = models[model_hub][cls._name].load_model(
                    model_path, *args, **kwargs
                )
            return cls.model
        else:
            raise ValueError(
                f"Provided model hub is not supported. Please choose one of the supported model hubs: {list(models.keys())}"
            )

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
            raise ValueError(
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

    @property
    def task_name(self):
        """Return the task name."""
        return self.__task_name


class NERTask(BaseTask):
    """Named Entity Recognition task."""

    _name = "ner"

    def create_sample(original, ner_labels) -> NERSample:
        """Create a sample."""
        return NERSample(
            original=original, expected_results=NEROutput(predictions=ner_labels)
        )


class TextClassificationTask(BaseTask):
    """Text Classification task."""

    _name = "textclassification"

    def create_sample(original, labels: SequenceLabel) -> SequenceClassificationSample:
        """Create a sample."""
        return SequenceClassificationSample(
            original,
            expected_results=SequenceClassificationOutput(predictions=[labels]),
        )


class QuestionAnsweringTask(BaseTask):
    """Question Answering task."""

    _name = "qa"

    def create_sample(
        row_data: dict, column_matcher: dict, dataset_name: str = "qa"
    ) -> QASample:
        """Create a sample."""
        expected_results = row_data.get(column_matcher["answer"])
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]
        return QASample(
            original_question=row_data[column_matcher["text"]],
            original_context=row_data.get(column_matcher["context"], "-"),
            expected_results=expected_results,
            dataset_name=dataset_name,
        )


class SummerizationTask(BaseTask):
    """Summerization task."""

    _name = "summarization"

    def create_sample(
        row_data: dict, column_matcher: dict, dataset_name: str = "xsum"
    ) -> SummarizationSample:
        """Create a sample."""
        expected_results = row_data.get(column_matcher["summary"])
        if isinstance(expected_results, str) or isinstance(expected_results, bool):
            expected_results = [str(expected_results)]
        return SummarizationSample(
            original=row_data[column_matcher["text"]],
            expected_results=expected_results,
            dataset_name=dataset_name,
        )


class TranslationTask(BaseTask):
    """Translation task."""

    _name = "translation"

    def create_sample(
        row_data: dict, column_matcher: dict, dataset_name: str = "translation"
    ) -> TranslationSample:
        """Create a sample."""
        return TranslationSample(
            original=row_data[column_matcher["text"]],
            dataset_name=dataset_name,
        )


class ToxicityTask(BaseTask):
    """Toxicity task."""

    _name = "toxicity"

    def create_sample(
        cls, row_data: dict, column_matcher: dict, dataset_name: str = "toxicity"
    ) -> ToxicitySample:
        """Create a sample."""
        return ToxicitySample(
            prompt=row_data[column_matcher["text"]],
            dataset_name=dataset_name,
        )


class SecurityTask(BaseTask):
    """Security task."""

    _name = "security"

    def create_sample(
        cls, row_data: dict, column_matcher: dict, dataset_name: str = "security"
    ) -> SecuritySample:
        """Create a sample."""
        return SecuritySample(
            prompt=row_data["text"],
            dataset_name=dataset_name,
        )


class ClinicalTestsTask(BaseTask):
    """Clinical Tests task."""

    _name = "clinicaltests"

    def create_sample(
        row_data: dict, column_matcher: dict, dataset_name: str = "clinicaltests"
    ) -> ClinicalSample:
        """Create a sample."""
        return ClinicalSample(
            patient_info_A=row_data["Patient info A"],
            patient_info_B=row_data["Patient info B"],
            diagnosis=row_data["Diagnosis"],
            dataset_name=dataset_name,
        )
