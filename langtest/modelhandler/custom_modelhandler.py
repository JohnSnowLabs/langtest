from typing import Any

from langtest.utils.custom_types.predictions import SequenceLabel
from .modelhandler import _ModelHandler
from abc import ABC, abstractmethod


class PretrainedCustomModel(ABC):
    def __init__(self, model: Any) -> None:
        self.model = model
        if not hasattr(self.model, "predict"):
            raise ValueError("Model must have a predict method")

    @classmethod
    def load_model(cls, model: Any) -> "Any":
        return cls(model)

    @abstractmethod
    def predict(self, text: str, *args, **kwargs):
        try:
            return self.model.predict(text, *args, **kwargs)
        except Exception as e:
            raise e

    def __call__(self, text: str) -> None:
        return self.predict(text=text)


class PretrainedModelForNER(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForTextClassification(PretrainedCustomModel, _ModelHandler):
    def predict(self, text: str, *args, **kwargs) -> SequenceLabel:
        try:
            out = self.model.predict(text, *args, **kwargs)
            return SequenceLabel(label=out)
        except Exception as e:
            raise e


class PretrainedModelForQA(PretrainedCustomModel, _ModelHandler):
    def predict(self, text: str, *args, **kwargs):
        try:
            out = self.model.predict(text, *args, **kwargs)
            return out
        except Exception as e:
            raise e
