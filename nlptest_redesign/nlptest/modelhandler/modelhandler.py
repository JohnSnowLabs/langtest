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
    """"""

    def __init__(
            self,
            model_path: str,
            task: str
    ) -> None:
        self.model_path = model_path
        self.task = task

        class_map = {
            cls.__name__.replace("PretrainedModel", "").lower(): cls for cls in _ModelHandler.__subclasses__()
        }
        self.model_class = class_map[self.task](self.model_path)
        self.model_class.load_model()

    def predict(self, text: str, **kwargs) -> List[NEROutput]:
        """"""
        return self.model_class(text=text, **kwargs)

    def __call__(self, text: str, *args, **kwargs) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.model_class(text=text, **kwargs)


class NERPretrainedModel(_ModelHandler):
    """"""

    def __init__(
            self,
            model_path: str
    ):
        self.model_path = model_path

    def load_model(self):
        """"""
        self.model = pipeline(model=self.model_path, task="ner", ignore_labels=[])

    def predict(self, text: str, **kwargs) -> List[NEROutput]:
        """"""
        prediction = self.model(text)

        if kwargs.get("group_entities"):
            prediction = [group for group in self.model.group_entities(prediction) if group["entity_group"] != "O"]

        return [NEROutput(**pred) for pred in prediction]

    def __call__(self, text: str, *args, **kwargs) -> List[NEROutput]:
        """Alias of the 'predict' method"""
        return self.predict(text=text, **kwargs)
