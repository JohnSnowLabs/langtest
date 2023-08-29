from abc import ABC, abstractmethod
from langtest.modelhandler.modelhandler import ModelFactory
from langtest.utils.custom_types.output import NEROutput

from langtest.utils.custom_types.sample import NERSample
from langtest.modelhandler import ModelAPI, LANGCHAIN_HUBS


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
        task_name = cls.__name__.lower().replace("task", "")
        cls.task_registry[task_name] = cls


class TaskManager:
    """Task manager."""

    def __init__(self, task_name: str):
        if task_name not in BaseTask.task_registry:
            raise ValueError(
                f"Task {task_name} not supported. Please choose from {list(BaseTask.task_registry.keys())}"
            )
        self.__task: BaseTask = BaseTask.task_registry[task_name]

    def create_sample(self, *args, **kwargs):
        """Add a task to the task manager."""
        return self.__task.create_sample(*args, **kwargs)

    def model(self, *args, **kwargs) -> "ModelAPI":
        """Add a task to the task manager."""
        return self.__task.load_model(*args, **kwargs)


class NERTask(BaseTask):
    """Named Entity Recognition task."""

    def create_sample(original, expected_results) -> NERSample:
        """Create a sample."""
        return NERSample(
            original=original, expected_results=NEROutput(predicted=expected_results)
        )

    def load_model(model_path: str, model_hub: str, *args, **kwargs) -> "ModelAPI":
        """Load the model."""

        model = ModelAPI.model_registry[model_hub]["ner"]
        return model.load_model(model_path, *args, **kwargs)
        # return model
