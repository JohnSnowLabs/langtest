from abc import ABC, abstractmethod
from typing import List

from transformers import pipeline

from ..utils.custom_types import NEROutput


class _ModelHandler(ABC):

    @abstractmethod
    def load_model(self):
        """"""
        return NotImplementedError()

    @abstractmethod
    def predict(self, text: str, *args, **kwargs):
        """"""
        return NotImplementedError()


class ModelFactory:
    """
    Args:
        model_path (str):
            path to model to use
        task (str):
            task to perform
    """
    SUPPORTED_TASKS = ["ner"]

    def __init__(
            self,
            model_path: str,
            task: str
    ):
        assert task in self.SUPPORTED_TASKS, \
            ValueError(f"Task '{task}' not supported. Please choose one of {', '.join(self.SUPPORTED_TASKS)}")

        self.model_path = model_path
        self.task = task

        class_map = {
            cls.__name__.replace("PretrainedModel", "").lower(): cls for cls in _ModelHandler.__subclasses__()
        }
        self.model_class = class_map[self.task](self.model_path)

    def load_model(self) -> None:
        """"""
        self.model_class.load_model()

    def predict(self, text: str, **kwargs) -> List[NEROutput]:
        """"""
        return self.model_class(text=text, **kwargs)

    def __call__(self, text: str, *args, **kwargs) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.model_class(text=text, **kwargs)


class NERPretrainedModel(_ModelHandler):
    """
    Args:
        model_path (str):
            path to model to use
    """

    def __init__(
            self,
            model_path: str
    ):
        self.model_path = model_path
        self.model = None

    def load_model(self) -> None:
        """"""
        self.model = pipeline(model=self.model_path, task="ner", ignore_labels=[])

    def predict(self, text: str, **kwargs) -> List[NEROutput]:
        """"""
        if self.model is None:
            raise OSError(f"The model '{self.model_path}' has not been loaded yet. Please call "
                          f"the '.load_model' method before running predictions.")
        prediction = self.model(text)

        if kwargs.get("group_entities"):
            prediction = [group for group in self.model.group_entities(prediction) if group["entity_group"] != "O"]

        return [NEROutput(**pred) for pred in prediction]

    def __call__(self, text: str, *args, **kwargs) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)
