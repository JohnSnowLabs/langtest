from typing import Any
from langtest.utils.custom_types.output import SequenceClassificationOutput

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


class PretrainedModelForTextClassification(PretrainedCustomModel, _ModelHandler):
    def predict(self, text: str, *args, **kwargs) -> SequenceClassificationOutput:
        try:
            out = self.model.predict(text, *args, **kwargs)
            if not isinstance(out, list):
                out = [out]
            return SequenceClassificationOutput(
                predictions=[SequenceLabel(label=elt, score=1) for elt in out]
            )
        except Exception as e:
            raise e


class PretrainedModelForQA(PretrainedCustomModel, _ModelHandler):
    def predict(self, text: str, *args, **kwargs):
        try:
            out = self.model.predict(text, *args, **kwargs)
            return out
        except Exception as e:
            raise e


class PretrainedModelForCrowsPairs(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForWinoBias(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForTranslation(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForSummarization(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForToxicity(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForSecurity(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForPolitical(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForDisinformationTest(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForFactualityTest(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForSensitivityTest(PretrainedCustomModel, _ModelHandler):
    pass


class PretrainedModelForSycophancyTest(PretrainedCustomModel, _ModelHandler):
    pass
