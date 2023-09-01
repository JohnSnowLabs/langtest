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

    @staticmethod
    @abstractmethod
    def create_sample(self):
        """Run the task."""
        pass

    @staticmethod
    @abstractmethod
    def load_model(self, model_path: str, model_hub: str):
        """Load the model."""
        if model_hub in LANGCHAIN_HUBS:
            model_hub = LANGCHAIN_HUBS[model_hub]

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        task_name = re.sub(
            r'(?<!^)(?=[A-Z])', '-', 
            cls.__name__.replace("Task", "")
        ).lower()
       
        cls.task_registry[task_name] = cls
    
    def __eq__(self, __value: object) -> bool:
        """Check if the task is equal to another task."""
        if isinstance(__value, str):
            return self.__class__.__name__.replace("Task","").lower() == __value.lower()
        return super().__eq__(__value)


class TaskManager:
    """Task manager."""

    def __init__(self, task_name: str):
        if task_name not in BaseTask.task_registry:
            raise ValueError(
                f"Provided task is not supported. Please choose one of the supported tasks: {list(BaseTask.task_registry.keys())}"
            )
        self.__task_name = task_name
        self.__task: BaseTask = BaseTask.task_registry[task_name]

    def create_sample(self, *args, **kwargs):
        """Add a task to the task manager."""
        return self.__task.create_sample(*args, **kwargs)

    def model(self, *args, **kwargs) -> "ModelAPI":
        """Add a task to the task manager."""
        return self.__task.load_model(*args, **kwargs)
    
    def __eq__(self, __value: str) -> bool:
        """Check if the task is equal to another task."""
        return self.__task_name == __value.lower()
        

class NERTask(BaseTask):
    """Named Entity Recognition task."""

    def create_sample(original, ner_labels) -> NERSample:
        """Create a sample."""
        return NERSample(
            original=original, expected_results=NEROutput(predictions=ner_labels)
        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["ner"]
        return model.load_model(model_path, *args, **kwargs)

class TextClassificationTask(BaseTask):
    """Text Classification task."""

    def create_sample(original, labels: SequenceLabel) -> SequenceClassificationSample:
        """Create a sample."""
        return SequenceClassificationSample(
            original,
            expected_results=SequenceClassificationOutput(predictions=[labels]),
        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["textclassification"]
        return model.load_model(model_path, *args, **kwargs)

class QuestionAnsweringTask(BaseTask):
    """Question Answering task."""

    def create_sample(original_question,original_context, expected_results, dataset_name) -> QASample:
        """Create a sample."""
        return QASample(
                            original_question,
                            original_context,
                            expected_results,
                            dataset_name,
                        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["qa"]
        return model.load_model(model_path, *args, **kwargs)

class SummerizationTask(BaseTask):
    """ Summerization task."""

    def create_sample(original, expected_results, dataset_name) -> SummarizationSample:
        """Create a sample."""
        return SummarizationSample(
                            original,
                            expected_results,
                            dataset_name,
                        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["summarization"]
        return model.load_model(model_path, *args, **kwargs)

class TranslationTask(BaseTask):
    """ Translation task."""

    def create_sample(original, dataset_name) -> TranslationSample:
        """Create a sample."""
        return TranslationSample(
                            original,
                            dataset_name,
                        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["translation"]
        return model.load_model(model_path, *args, **kwargs)

class ToxicityTask(BaseTask):
    """ Toxicity task."""

    def create_sample(prompt, dataset_name) -> ToxicitySample:
        """Create a sample."""
        return ToxicitySample(
                            prompt,
                            dataset_name,
                        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["toxicity"]
        return model.load_model(model_path, *args, **kwargs)


class SecurityTask(BaseTask):
    """ Security task."""

    def create_sample(prompt, task, dataset_name) -> SecuritySample:
        """Create a sample."""
        return SecuritySample(
                            prompt,
                            task,
                            dataset_name,
                        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["security"]
        return model.load_model(model_path, *args, **kwargs)

class ClinicalTestsTask(BaseTask):
    """ Clinical Tests task. """

    def create_sample(patient_info_A, patient_info_B, diagnosis, dataset_name) -> ClinicalSample:
        """Create a sample."""
        return ClinicalSample(
                            patient_info_A,
                            patient_info_B,
                            diagnosis,
                            dataset_name,
                        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["clinicaltests"]
        return model.load_model(model_path, *args, **kwargs)
