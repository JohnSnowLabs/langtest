from abc import ABC, abstractmethod
from langtest.modelhandler.modelhandler import ModelFactory
from langtest.utils.custom_types.output import NEROutput

from langtest.utils.custom_types.sample import NERSample


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def create_sample(self):
        """Run the task."""
        pass

    @abstractmethod
    def load_model(self):
        """Load the model."""
        pass

class TaskManager:
    """Task manager."""

    def __init__(self):
        self.tasks = []

    def get_task(self, task):
        """Add a task to the task manager."""
        self.tasks.append(task)
    
    def available_tasks(self):
        """Return a list of available tasks."""
        # tasks = {key: obj for key, obj in globals().items() if isinstance(obj, type) and issubclass(obj, Task)}
        pass 

class NERTask(BaseTask):
    """Named Entity Recognition task."""

    def __init__(self, name):
        super().__init__(name)

    def create_sample(self, original, expected_results) -> NERSample:
        """Create a sample."""
        return NERSample(
            original=original,
            expected_results=NEROutput(predicted=expected_results)
        )

    def load_model(self, model_path: str, model_hub: str) -> ModelFactory:
        """Load the model."""
        pass